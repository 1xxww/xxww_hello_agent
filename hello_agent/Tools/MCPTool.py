
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .Tool import Tool

logger = logging.getLogger(__name__)


# 1. 这个类是“真正呈现给 Agent”的工具,它不再负责管理子进程，只负责“发消息”
class MCPTool(Tool):
    """
    具体的 MCP 工具。Agent 看到的是这个对象。
    例如：它代表 `create_issue` 这个动作，而不是代表整个 Github MCP 服务。
    """

    def __init__(self, name: str, original_name: str, description: str, input_schema: dict, session: ClientSession):
        self.name = name
        self._original_name = original_name  # 给底层 Server 看的原名：create_issue
        self.description = description
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
        """
        Agent 直接调用这个执行逻辑。
        大模型不需要传 tool_name，因为这个实例本身就是这个 tool。
        """
        if not self._session:
            return "error: MCP session is disconnected."

        try:
            # 直接使用自身的 name 去调用底层的 session
            result = await self._session.call_tool(self._original_name, arguments=kwargs)

            content_parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    content_parts.append(item.text)
            return "\n".join(content_parts)
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"


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
    ):
        self.server_name = name
        self._server_params = StdioServerParameters(command=command, args=args, env=env)
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()  # 安全管理异步上下文的利器

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