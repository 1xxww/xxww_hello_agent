"""Agent基类"""
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict
# 这是你的其他模块
from .Message import Message
from ..LLM import MyLLM
from .AgentConfig import AgentConfig
from ..Tools import Tool
from ..Log import LoggingService
import logging


class Agent(ABC):
    """Agent基类（全面拥抱异步）"""

    def __init__(
            self,
            name: str,
            llm: MyLLM,
            system_prompt: Optional[str] = None,
            config: Optional[AgentConfig] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or AgentConfig()
        self._history: List[Message] = []
        self.tools = List[Tool]
        LoggingService.setup(log_level=self.config.log_level if hasattr(self.config, 'log_level') else "INFO")
        self.logger = logging.getLogger(self.name)

    def register_tools(self, tools: List[Tool]):
        """将你的 MCPTool 列表注册到 Agent 中"""
        self.tools.extend(tools)

    @abstractmethod
    async def run(self, input_text: str, **kwargs) -> str:
        """
        运行Agent (改为 async def)
        子类在这里实现具体的循环逻辑 (Thought -> Action -> Observation)
        """
        pass

    def _build_full_messages(self, current_input: str) -> List[Dict[str, Any]]:
        """内部方法：组装发给大模型的完整消息体"""
        messages = []
        # 1. 注入 System Prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 2. 注入历史记录 (假设 Message 对象有 to_dict 方法)
        for msg in self._history:
            messages.append(msg.to_dict())

        # 3. 注入当前输入
        messages.append({"role": "user", "content": current_input})
        return messages

    # ==========================================
    # 下面这两个方法保持不变，写得很棒
    # ==========================================
    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)
        if len(self._history)>=self.config.max_history_length:
            self._history.pop(0)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> List[Message]:
        """获取历史记录"""
        return self._history.copy()

    def log(self, message: str):
        """打印日志"""
        if self.config.verbose:
            self.logger.info(message)

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"