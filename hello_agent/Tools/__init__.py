from .ToolParameter import ToolParameter
from .ToolRegistry import ToolRegistry
from .Tool import Tool

__all__ = [
    "MCPTool",
    "MemoryTool",
    "NoteTool",
    "RAGTool",
    "SearchTool",
    "TerminalTool",
    "ToolParameter",
    "ToolRegistry",
    "Tool",
]


def __getattr__(name: str):
    """按需懒加载重型工具（避免仅导入 ToolRegistry 时连带加载 PyTorch / Qdrant 等重型库）"""
    if name == "MCPTool":
        from .MCPTool import MCPTool
        return MCPTool
    if name == "MemoryTool":
        from .MemoryTool import MemoryTool
        return MemoryTool
    if name == "NoteTool":
        from .NoteTool import NoteTool
        return NoteTool
    if name == "RAGTool":
        from .RAGTool import RAGTool
        return RAGTool
    if name == "SearchTool":
        from .SearchTool import SearchTool
        return SearchTool
    if name == "TerminalTool":
        from .TerminaTool import TerminalTool
        return TerminalTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
