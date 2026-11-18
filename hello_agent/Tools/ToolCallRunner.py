from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
import time
from typing import Optional

from .ArgumentValidator import validate_arguments
from .ToolError import (
    RetryPolicy,
    ToolArgumentError,
    ToolCallResult,
    ToolNotFoundError,
    classify_exception,
)

logger = logging.getLogger(__name__)


class ToolCallRunner:
    def __init__(self, registry, config=None):
        self.registry = registry
        self.config = config

    def _policy_for(self, tool) -> RetryPolicy:
        # 工具显式配置优先，否则使用 AgentConfig 全局默认值
        if getattr(tool, "retry_policy", None) is not None:
            return tool.retry_policy

        if self.config is None:
            return RetryPolicy()

        return RetryPolicy(
            max_attempts=getattr(self.config, "tool_max_attempts", 4),
            timeout_seconds=getattr(self.config, "tool_timeout_seconds", 30.0),
            base_delay=getattr(self.config, "tool_retry_base_delay", 0.5),
            max_delay=getattr(self.config, "tool_retry_max_delay", 4.0),
            jitter=getattr(self.config, "tool_retry_jitter", 0.3),
            idempotent=getattr(self.config, "tool_retry_idempotent", True),
        )

    @staticmethod
    def _delay(attempt: int, policy: RetryPolicy, error) -> float:
        if getattr(error, "retry_after", None):
            return max(0.0, float(error.retry_after))
        delay = min(policy.base_delay * (2 ** (attempt - 1)), policy.max_delay)
        return delay + random.uniform(0, policy.jitter)

    async def run(self, tool_name: str, kwargs: dict) -> ToolCallResult:
        started = time.perf_counter()

        tool = self.registry.get_tool(tool_name)
        if tool is None:
            error = ToolNotFoundError(
                f"找不到名为 [{tool_name}] 的工具",
                details={"available_tools": list(self.registry._tools.keys())},
            )
            return self._fail(tool_name, error, 0, started)

        # 参数错误不重试，直接返回给模型修正
        try:
            validate_arguments(tool, kwargs)
        except ToolArgumentError as error:
            return self._fail(tool_name, error, 0, started)

        policy = self._policy_for(tool)
        max_attempts = max(1, policy.max_attempts)
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw = tool.execute(**kwargs)
                if inspect.isawaitable(raw):
                    raw = await asyncio.wait_for(
                        asyncio.ensure_future(raw),
                        timeout=policy.timeout_seconds,
                    )

                content = (
                    raw if isinstance(raw, str)
                    else json.dumps(raw, ensure_ascii=False, default=str)
                )
                return ToolCallResult(
                    tool_name=tool_name,
                    ok=True,
                    content=content,
                    attempts=attempt,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                last_error = classify_exception(exc)

                if (
                    not last_error.retryable
                    or not policy.idempotent
                    or attempt >= max_attempts
                ):
                    break

                delay = self._delay(attempt, policy, last_error)
                logger.warning(
                    "tool=%s attempt=%d/%d error=%s retry_in=%.2fs",
                    tool_name, attempt, max_attempts,
                    last_error.error_type, delay,
                )
                await asyncio.sleep(delay)

        return self._fail(tool_name, last_error, attempt, started)

    @staticmethod
    def _fail(tool_name, error, attempts, started) -> ToolCallResult:
        if error is None:
            error = ToolArgumentError("unknown tool error")
        return ToolCallResult(
            tool_name=tool_name,
            ok=False,
            error=error,
            attempts=attempts,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )