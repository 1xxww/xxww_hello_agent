"""最小验证脚本：检查 HelloAgent 工具重试机制。

直接运行：
    python tests/verify_tool_retry.py
"""

import asyncio
from types import SimpleNamespace

from hello_agent.Agent.SimpleAgent import SimpleAgent
from hello_agent.Tools.MCPTool import MCPTool
from hello_agent.Tools.Tool import Tool
from hello_agent.Tools.ToolCallRunner import ToolCallRunner
from hello_agent.Tools.ToolError import (
    RetryPolicy,
    ToolBusinessError,
    ToolCallResult,
)
from hello_agent.Tools.ToolRegistry import ToolRegistry


class FlakyTool(Tool):
    def __init__(self, name="flaky", failures=2, exc=None, policy=None):
        super().__init__(name, "test tool", policy)
        self.failures = failures
        self.calls = 0
        self.exc = exc or ConnectionError("network down")

    async def execute(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "ok"

    def get_parameters(self):
        return []


class SchemaTool(Tool):
    def __init__(self, name="weather", policy=None):
        super().__init__(name, "weather tool", policy)

    async def execute(self, city: str = ""):
        return f"weather of {city}"

    def get_parameters(self):
        return []

    def to_openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }


class SlowTool(Tool):
    def __init__(self, name="slow", policy=None):
        super().__init__(name, "slow tool", policy)
        self.calls = 0

    async def execute(self, **kwargs):
        self.calls += 1
        await asyncio.sleep(0.2)
        return "late"

    def get_parameters(self):
        return []


class CancelTool(Tool):
    def __init__(self, policy=None):
        super().__init__("cancel", "cancel tool", policy)

    async def execute(self, **kwargs):
        raise asyncio.CancelledError()

    def get_parameters(self):
        return []


class DummyLLM:
    provider = "dummy"


class FakeSession:
    async def call_tool(self, name, arguments=None):
        return SimpleNamespace(
            content=[SimpleNamespace(text="MCP business error")],
            isError=True,
        )


def registry_with(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register_tool(tool)
    return registry


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(name)


async def main():
    zero = RetryPolicy(max_attempts=4, base_delay=0, jitter=0)

    # 1. 前两次失败，第三次成功
    tool = FlakyTool(failures=2, policy=zero)
    runner = ToolCallRunner(registry_with(tool))
    result = await runner.run("flaky", {})
    check(
        "瞬时失败自动重试",
        result.ok and result.attempts == 3 and tool.calls == 3,
        f"ok={result.ok} attempts={result.attempts} calls={tool.calls}",
    )

    # 2. 一直失败，走满 4 次
    tool = FlakyTool(failures=10, policy=zero)
    runner = ToolCallRunner(registry_with(tool))
    result = await runner.run("flaky", {})
    check(
        "重试上限",
        (not result.ok) and result.attempts == 4 and tool.calls == 4
        and result.error.error_type == "transport_error",
        f"attempts={result.attempts} calls={tool.calls} error={result.error.error_type}",
    )

    # 3. 参数错误不重试
    tool = SchemaTool(policy=zero)
    runner = ToolCallRunner(registry_with(tool))
    result = await runner.run("weather", {"keyword": "西湖"})
    check(
        "参数错误不重试",
        (not result.ok) and result.attempts == 0
        and result.error.error_type == "invalid_arguments"
        and "city" in result.error.message,
        f"attempts={result.attempts} error={result.error.error_type}",
    )

    # 4. 非幂等工具不重试
    tool = FlakyTool(
        failures=10,
        policy=RetryPolicy(max_attempts=4, base_delay=0, jitter=0, idempotent=False),
    )
    runner = ToolCallRunner(registry_with(tool))
    result = await runner.run("flaky", {})
    check(
        "非幂等工具只执行一次",
        (not result.ok) and result.attempts == 1 and tool.calls == 1,
        f"attempts={result.attempts} calls={tool.calls}",
    )

    # 5. 超时会重试
    tool = SlowTool(policy=RetryPolicy(max_attempts=3, timeout_seconds=0.05, base_delay=0, jitter=0))
    runner = ToolCallRunner(registry_with(tool))
    result = await runner.run("slow", {})
    check(
        "超时重试",
        (not result.ok) and result.attempts == 3 and tool.calls == 3
        and result.error.error_type == "timeout",
        f"attempts={result.attempts} calls={tool.calls} error={result.error.error_type}",
    )

    # 6. 不存在的工具
    runner = ToolCallRunner(registry_with())
    result = await runner.run("missing", {})
    check(
        "工具不存在",
        (not result.ok) and result.attempts == 0
        and result.error.error_type == "tool_not_found",
        f"error={result.error.error_type}",
    )

    # 7. CancelledError 不能被吞
    runner = ToolCallRunner(registry_with(CancelTool(policy=zero)))
    cancelled = False
    try:
        await runner.run("cancel", {})
    except asyncio.CancelledError:
        cancelled = True
    check("取消异常原样抛出", cancelled)

    # 8. MCP isError 转成业务错误
    mcp_tool = MCPTool(
        name="mcp_test",
        original_name="test",
        description="test",
        input_schema={},
        session=FakeSession(),
        retry_policy=zero,
    )
    business_error = False
    try:
        await mcp_tool.execute()
    except ToolBusinessError:
        business_error = True
    check("MCP isError 转业务错误", business_error)

    # 9. 成功结果保留 attempts，供 Agent 观察
    result = ToolCallResult(tool_name="x", ok=True, content="ok", attempts=3)
    check("成功结果携带 attempts", result.ok and result.attempts == 3)

    # 10. SimpleAgent 真正接入 ToolCallRunner
    flaky = FlakyTool(failures=2, policy=zero)
    agent = SimpleAgent(
        name="retry-test-agent",
        llm=DummyLLM(),
        tool_registry=registry_with(flaky),
    )
    result = await agent._aexecute_tool_call("flaky", "{}")
    observation = agent._format_tool_result(result)
    check(
        "SimpleAgent 接入重试",
        result.ok and result.attempts == 3 and "执行结果" in observation,
        f"ok={result.ok} attempts={result.attempts}",
    )

    weather = SchemaTool(policy=zero)
    agent = SimpleAgent(
        name="retry-test-agent-2",
        llm=DummyLLM(),
        tool_registry=registry_with(weather),
    )
    result = await agent._aexecute_tool_call("weather", '{"keyword": "西湖"}')
    observation = agent._format_tool_result(result)
    check(
        "SimpleAgent 参数错误回传 observation",
        (not result.ok) and "invalid_arguments" in observation and "city" in observation,
        observation[:120].replace("\n", " "),
    )

    print("\n全部重试机制验证通过。")


if __name__ == "__main__":
    asyncio.run(main())
