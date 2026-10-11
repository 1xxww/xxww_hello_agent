from ..LLM.Myllm import MyLLM
from .SimpleAgent import SimpleAgent
from .AgentConfig import AgentConfig
from ..Tools import ToolRegistry
from ..Memory import MemoryManager,MemoryConfig
from typing import Optional

class MemoryAgent(SimpleAgent):

    def __init__(
            self,
            name: str,
            llm: MyLLM,
            system_prompt: Optional[str] = None,
            config: Optional[AgentConfig] = None,
            tool_registry: Optional['ToolRegistry'] = None,
            enable_tool_calling: bool = True,
            memory_manager: Optional[MemoryManager] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
        )
        self.memory_manager=memory_manager or MemoryManager(
            config=MemoryConfig(),
            user_id=self.name,
            enable_working=True,
            enable_episodic=True,
        )




