from typing import Optional, Callable, Dict, Any
from ..LLM.Myllm import MyLLM
from .SimpleAgent import SimpleAgent
from .AgentConfig import AgentConfig
from ..Tools import ToolRegistry

# 定义回调函数
def my_console_listener(event: dict):
    print(f"📢 [事件捕获] Agent [{event['agent_name']}] 调用了工具 <{event['tool_name']}>")
    print(f"   📥 入参: {event['parsed_parameters']}")
    print(f"   📤 结果: {event['result'][:50]}...") # 截断显示
# 定义监听器接收的字典数据格式（类型别名）
ToolCallPayload = Dict[str, Any]
# 声明 Callable 类型：接收一个 dict，无返回值 (None)
ToolCallListener = Callable[[ToolCallPayload], None]

class ToolAwareSimpleAgent(SimpleAgent):
    def __init__(
        self,
        name: str,
        llm: MyLLM,
        system_prompt: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True,
        tool_call_listener: Optional[ToolCallListener] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
        )
        # 挂载监听器函数
        self._tool_call_listener = tool_call_listener

    def _notify_listener(self, tool_name: str, raw_parameters: Any, result: str):
        """统一的通知触发器（安全解析参数并回调）"""
        if not self._tool_call_listener:
            return

        # 尝试使用基类的解析方法获得字典格式参数，解析失败时保留原字符串
        try:
            parsed_params = self._parse_tool_parameters(raw_parameters)
        except Exception:
            parsed_params = raw_parameters

        # 构建数据载荷 (Payload)
        payload: ToolCallPayload = {
            "agent_name": self.name,
            "tool_name": tool_name,
            "raw_parameters": raw_parameters,
            "parsed_parameters": parsed_params,
            "result": result,
        }

        # 安全触发回调，捕获监听器内部可能抛出的异常，防止影响主流程
        try:
            self._tool_call_listener(payload)
        except Exception as e:
            print(f"⚠️ [{self.name}] 工具调用监听器执行异常: {e}")

    # 1. 拦截同步工具执行
    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        # 调用基类原生逻辑执行工具
        result = super()._execute_tool_call(tool_name, parameters)
        # 触发通知
        self._notify_listener(tool_name, parameters, result)
        return result

    # 2. 拦截异步工具执行（保持 async 流程）
    async def _aexecute_tool_call(self, tool_name: str, parameters: str) -> str:
        # 调用基类原生逻辑异步执行工具
        result = await super()._aexecute_tool_call(tool_name, parameters)
        # 触发通知
        self._notify_listener(tool_name, parameters, result)
        return result