import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .Tool import Tool
from .ToolError import (
    ToolBusinessError,
    ToolTransportError,
    ToolTimeoutError,
    classify_exception,
)

logger = logging.getLogger(__name__)


# 1. 这个类是“真正呈现给 Agent”的工具,它不再负责管理子进程，只负责“发消息”
class MCPTool(Tool):
    """
    具体的 MCP 工具。Agent 看到的是这个对象。
    例如：它代表 `create_issue` 这个动作，而不是代表整个 Github MCP 服务。
    """

    def __init__(
        self,
        name: str,
        original_name: str,
        description: str,
        input_schema: dict,
        session: ClientSession,
        retry_policy=None,
    ):
        super().__init__(name, description, retry_policy=retry_policy)
        self._original_name = original_name
        self._input_schema = input_schema
        self._session = session  # 持有 Server 的连接句柄

    def get_parameters(self) -> Dict[str, Any]:
        """完美适配大模型的 Schema 获取"""
        return self._input_schema

    def get_openai_schema(self) -> Dict[str, Any]:
        """生成 OpenAI function calling 标准 Schema（供 Registry/Agent 注入提示词）"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._input_schema or {"type": "object", "properties": {}},
            },
        }

    async def execute(self, **kwargs) -> str:
        if not self._session:
            raise ToolTransportError("MCP session is disconnected")

        timeout = self.retry_policy.timeout_seconds if self.retry_policy else 30.0
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise classify_exception(e) from e

        content_parts = [
            item.text for item in result.content if hasattr(item, "text")
        ]
        text = "\n".join(content_parts)

        # MCP 协议里的业务错误标记
        if getattr(result, "isError", False):
            raise ToolBusinessError(text or f"MCP tool {self.name} returned an error")

        return text


# 2. 这个类是“幕后管理者”,负责拉起进程，并“孵化”出上面的 MCPTool 列表
class MCPServerManager:
    """
    管理 MCP 子进程的生命周期，并作为工厂生成 MCPTool 实例。
    """

    def __init__(
            self,
            name: str,
            command: str,
            args: List[str],
            env: Optional[Dict[str, str]] = None,
            retry_policy=None
    ):
        self.server_name = name
        self._server_params = StdioServerParameters(command=command, args=args, env=env)
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()  # 安全管理异步上下文的利器
        self._retry_policy = retry_policy

    async def start_and_get_tools(self) -> List[MCPTool]:
        """
        核心方法：启动服务，并直接返回 Agent 可以直接使用的标准 Tool 列表
        """
        logger.info(f"[{self.server_name}] Starting MCP server...")

        # 安全地进入 stdio 上下文
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(self._server_params)
        )
        read_stream, write_stream = stdio_transport

        # 安全地进入 session 上下文
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        logger.info(f"[{self.server_name}] MCP session initialized")

        # 拉取底层工具清单
        resp = await self._session.list_tools()

        # 【关键步骤】将底层数据转换为 Agent 标准 Tool 实例
        agent_tools = []
        for tool_info in resp.tools:
            # tool_info.inputSchema 已经是标准的 JSON Schema 格式
            tool_instance = MCPTool(
                name=f"{self.server_name}_{tool_info.name}",  # 加上前缀防止不同Server工具重名冲突
                original_name=tool_info.name,
                description=tool_info.description or "",
                input_schema=tool_info.input_schema,
                session=self._session  # 把活着的连接传给工具
            )
            agent_tools.append(tool_instance)

        logger.info(f"[{self.server_name}] Generated {len(agent_tools)} BaseTools for Agent.")
        return agent_tools

    async def stop(self):
        """安全关闭所有资源，杀死子进程"""
        logger.info(f"[{self.server_name}] Stopping MCP server...")
        await self._exit_stack.aclose()
        self._session = None