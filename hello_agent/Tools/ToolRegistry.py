import json
import inspect
from typing import List, Dict, Any
from .Tool import Tool
from .ToolError import ToolError, ToolNotFoundError


class ToolRegistry:
    """HelloAgents工具注册表 (只认 Tool 对象，大道至简)"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """统一注册入口"""
        if tool.name in self._tools:
            print(f"⚠️ 警告:工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取所有工具的 OpenAI 标准 Schema 列表"""
        schemas = []
        for tool in self._tools.values():
            # MCP 工具自带 get_openai_schema；普通工具用基类的 to_openai_schema
            try:
                if hasattr(tool, "get_openai_schema"):
                    schemas.append(tool.get_openai_schema())
                elif hasattr(tool, "to_openai_schema"):
                    schemas.append(tool.to_openai_schema())
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 生成工具 '{getattr(tool, 'name', '?')}' 的 schema 失败: {e}")
        return schemas

    def get_tool(self, tool_name: str):
        return self._tools.get(tool_name)

    async def execute_tool_raw(self, tool_name: str, **kwargs) -> str:
        """执行工具，失败时抛 ToolError。ToolCallRunner 用这个入口。"""
        tool = self.get_tool(tool_name)
        if tool is None:
            raise ToolNotFoundError(
                f"找不到名为 [{tool_name}] 的工具",
                details={"available_tools": list(self._tools.keys())},
            )

        raw_result = tool.execute(**kwargs)
        if inspect.isawaitable(raw_result):
            result = await raw_result
        else:
            result = raw_result

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """向后兼容接口：把异常转成字符串返回。"""
        try:
            return await self.execute_tool_raw(tool_name, **kwargs)
        except Exception as e:
            return f"❌ 工具 [{tool_name}] 执行异常: {str(e)}"