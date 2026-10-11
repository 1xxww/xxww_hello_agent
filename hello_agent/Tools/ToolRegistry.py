import json
import inspect
from typing import List, Dict, Any
from .Tool import Tool


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

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """统一执行工具入口"""
        if tool_name not in self._tools:
            return f"❌ 找不到名为 [{tool_name}] 的工具。"

        tool = self._tools[tool_name]
        try:
            # 统一执行
            raw_result = tool.execute(**kwargs)

            # 魔法兼容：如果是异步工具就 await，如果是同步工具直接拿结果
            if inspect.isawaitable(raw_result):
                result = await raw_result
            else:
                result = raw_result

            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)

        except Exception as e:
            return f"❌ 工具 [{tool_name}] 执行异常: {str(e)}"