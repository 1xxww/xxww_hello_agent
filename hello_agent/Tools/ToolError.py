from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Optional


class ToolError(Exception):
    """所有工具错误的基类。retryable 决定执行器是否重试。"""
    error_type = "tool_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.retry_after = retry_after

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
        }


class ToolNotFoundError(ToolError):
    error_type = "tool_not_found"
    retryable = False


class ToolArgumentError(ToolError):
    error_type = "invalid_arguments"
    retryable = False


class ToolAuthError(ToolError):
    error_type = "auth_error"
    retryable = False


class ToolBusinessError(ToolError):
    error_type = "business_error"
    retryable = False


class ToolTransportError(ToolError):
    error_type = "transport_error"
    retryable = True


class ToolTimeoutError(ToolTransportError):
    error_type = "timeout"


class ToolRateLimitError(ToolTransportError):
    error_type = "rate_limited"


class ToolServerError(ToolTransportError):
    error_type = "server_error"


@dataclass
class RetryPolicy:
    max_attempts: int = 4          # 首次 + 3 次重试
    timeout_seconds: float = 30.0
    base_delay: float = 0.5
    max_delay: float = 4.0
    jitter: float = 0.3
    idempotent: bool = True        # 写操作必须显式设为 False


@dataclass
class ToolCallResult:
    tool_name: str
    ok: bool
    content: str = ""
    error: Optional[ToolError] = None
    attempts: int = 1
    latency_ms: int = 0

def classify_exception(exc: BaseException) -> ToolError:
    """把第三方库异常映射成可判断是否重试的 ToolError。"""
    if isinstance(exc, ToolError):
        return exc

    # CancelledError 是 BaseException，必须原样抛出
    if isinstance(exc, asyncio.CancelledError):
        raise exc

    if isinstance(exc, TimeoutError):
        return ToolTimeoutError(str(exc))

    if isinstance(exc, (ConnectionError, BrokenPipeError)):
        return ToolTransportError(str(exc))

    if isinstance(exc, json.JSONDecodeError):
        return ToolArgumentError(str(exc))

    # pydantic 参数校验失败
    try:
        from pydantic import ValidationError
        if isinstance(exc, ValidationError):
            return ToolArgumentError(str(exc))
    except ImportError:
        pass

    # httpx
    try:
        import httpx
        if isinstance(exc, httpx.TimeoutException):
            return ToolTimeoutError(str(exc))
        if isinstance(exc, httpx.ConnectError):
            return ToolTransportError(str(exc))
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 429:
                retry_after = exc.response.headers.get("retry-after")
                return ToolRateLimitError(
                    str(exc),
                    retry_after=float(retry_after) if retry_after else None,
                )
            if 500 <= status < 600:
                return ToolServerError(str(exc))
            if status in (401, 403):
                return ToolAuthError(str(exc))
            if 400 <= status < 500:
                return ToolArgumentError(str(exc))
    except ImportError:
        pass

    # requests
    try:
        import requests
        if isinstance(exc, requests.Timeout):
            return ToolTimeoutError(str(exc))
        if isinstance(exc, requests.ConnectionError):
            return ToolTransportError(str(exc))
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            status = exc.response.status_code
            if status == 429:
                return ToolRateLimitError(str(exc))
            if status >= 500:
                return ToolServerError(str(exc))
            if status in (401, 403):
                return ToolAuthError(str(exc))
            return ToolArgumentError(str(exc))
    except ImportError:
        pass

    # anyio / MCP 连接断开
    try:
        from anyio import BrokenResourceError, ClosedResourceError
        if isinstance(exc, (ClosedResourceError, BrokenResourceError)):
            return ToolTransportError(str(exc))
    except ImportError:
        pass

    # 未知异常默认不重试，避免重复执行有副作用的工具
    return ToolError(str(exc))
