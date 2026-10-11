from .Agent import Agent, SimpleAgent, ToolAwareSimpleAgent
from .Agent.AgentConfig import AgentConfig
from .Agent.Message import Message
from .LLM import MyLLM
from .Log.LoggingService import LoggingService
from .Tools.ToolParameter import ToolParameter
from .Tools.ToolRegistry import ToolRegistry
from .Tools.Tool import Tool

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "SimpleAgent",
    "ToolAwareSimpleAgent",
    "AgentConfig",
    "Message",
    "ContextBuilder",
    "MyLLM",
    "MemoryManager",
    "MemoryConfig",
    "MemoryItem",
    "LoggingService",
    "MCPTool",
    "MemoryTool",
    "NoteTool",
    "RAGTool",
    "SearchTool",
    "ToolParameter",
    "ToolRegistry",
    "TerminalTool",
    "Tool",
]


def __getattr__(name: str):
    """按需懒加载重型子模块（避免轻量 Agent 啟動时被动加载 PyTorch / SentenceTransformers 等 ~700MB 内存）"""
    if name == "ContextBuilder":
        from .Context import ContextBuilder
        return ContextBuilder
    if name in ("MemoryManager", "MemoryConfig", "MemoryItem"):
        from . import Memory
        return getattr(Memory, name)
    if name in ("MCPTool", "MemoryTool", "NoteTool", "RAGTool", "SearchTool", "TerminalTool"):
        from . import Tools
        return getattr(Tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
