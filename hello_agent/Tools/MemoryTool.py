from .Tool import Tool
from ..Memory import MemoryConfig,MemoryManager,MemoryItem
from datetime import datetime
from typing import List, Dict, Any
from .ToolParameter import ToolParameter

class MemoryTool(Tool):
    """记忆工具 - 为Agent提供记忆功能"""

    def __init__(
            self,
            user_id: str = "default_user",
            memory_config: MemoryConfig = None,
            memory_types: List[str] = None
    ):
        super().__init__(
            name="memory",
            description="记忆工具 - 可以存储和检索对话历史、知识和经验"
        )

        # 初始化记忆管理器
        self.memory_config = memory_config or MemoryConfig()
        self.memory_types = memory_types or ["working", "episodic"]
        self.current_session_id = None

        self.memory_manager = MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
            enable_perceptual="perceptual" in self.memory_types
        )

    def execute(self, **kwargs) -> Any:
        """执行记忆操作

        支持的操作：
        - add: 添加记忆（支持4种类型: working/episodic/semantic/perceptual）
        - search: 搜索记忆
        - summary: 获取记忆摘要
        - stats: 获取统计信息
        - update: 更新记忆
        - remove: 删除记忆
        - forget: 遗忘记忆（多种策略）
        - consolidate: 整合记忆（短期→长期）
        - clear_all: 清空所有记忆
        """

        match kwargs.pop("action", None):
            case "add":
                return self._add_memory(**kwargs)
            case "search":
                return self._search_memory(**kwargs)
            case "search_raw":
                return self._search_memory_raw(**kwargs)
            case "forget":
                return self._forget(**kwargs)
            case "consolidate":
                return self._consolidate(**kwargs)
            case unknown:
                return f"❌ 未知操作: '{unknown}'，仅支持: add, search, search_raw, forget, consolidate"



    def get_parameters(self) -> List[ToolParameter]:
        return [
            # ── 唯一必填：动作判别器 ──
            ToolParameter(
                name="action",
                type="string",
                description="记忆操作: add=添加记忆 / search=语义检索(返回文本) / search_raw=检索原始数据 / forget=按策略遗忘 / consolidate=提升为长期记忆",
                required=True,
            ),
            # ── add 专用（对齐 _add_memory）──
            ToolParameter(name="content", type="string",
                          description="记忆内容(仅 action=add)", required=False),
            ToolParameter(name="memory_type", type="string",
                          description="记忆类型: working/episodic/semantic/perceptual(仅 action=add)",
                          required=False, default="working"),
            ToolParameter(name="importance", type="number",
                          description="重要性 0~1(仅 action=add)", required=False, default=0.5),
            ToolParameter(name="file_path", type="string",
                          description="文件路径，memory_type=perceptual 时使用(仅 action=add)", required=False),
            ToolParameter(name="modality", type="string",
                          description="感知模态，如 text/image/audio(仅 action=add 的 perceptual)", required=False),
            # ── search 专用（对齐 _search_memory）──
            ToolParameter(name="query", type="string",
                          description="搜索关键词(仅 action=search)", required=False),
            ToolParameter(name="limit", type="integer",
                          description="返回条数(仅 action=search)", required=False, default=5),
            ToolParameter(name="memory_types", type="array",
                          description="限定类型列表，如 ['working','episodic'](仅 action=search)", required=False),
            ToolParameter(name="min_importance", type="number",
                          description="最小重要性阈值(仅 action=search)", required=False, default=0.1),
            # ── forget 专用（对齐 _forget）──
            ToolParameter(name="strategy", type="string",
                          description="遗忘策略: importance_based/time_based(仅 action=forget)",
                          required=False, default="importance_based"),
            ToolParameter(name="threshold", type="number",
                          description="重要性阈值(仅 action=forget)", required=False, default=0.1),
            ToolParameter(name="max_age_days", type="integer",
                          description="最大保留天数(仅 action=forget)", required=False, default=30),
            # ── consolidate 专用（对齐 _consolidate）──
            ToolParameter(name="from_type", type="string",
                          description="源记忆类型(仅 action=consolidate)", required=False, default="working"),
            ToolParameter(name="to_type", type="string",
                          description="目标记忆类型(仅 action=consolidate)", required=False, default="episodic"),
            ToolParameter(name="importance_threshold", type="number",
                          description="提升阈值(仅 action=consolidate)", required=False, default=0.7),
        ]
    def _add_memory(
            self,
            content: str = "",
            memory_type: str = None,
            importance: float = 0.5,
            file_path: str = None,
            modality: str = None,
            **metadata
    ) -> str:
        """添加记忆"""
        try:
            # 确保会话ID存在
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 感知记忆文件支持
            if memory_type == "perceptual" and file_path:
                inferred = modality or self._infer_modality(file_path)
                metadata.setdefault("modality", inferred)
                metadata.setdefault("raw_data", file_path)

            # 添加会话信息到元数据
            metadata.update({
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat()
            })

            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata,
                auto_classify=False
            )

            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"

        except Exception as e:
            return f"❌ 添加记忆失败: {str(e)}"

    # 搜索记忆str
    def _search_memory(
            self,
            query: str,
            limit: int = 5,
            memory_types: List[str] = None,
            memory_type: str = None,
            min_importance: float = 0.1
    ) -> str:
        """搜索记忆"""
        try:
            # 参数标准化处理
            if memory_type and not memory_types:
                memory_types = [memory_type]

            results = self.memory_manager.retrieve_memories(
                query=query,
                limit=limit,
                memory_types=memory_types,
                min_importance=min_importance
            )

            if not results:
                return f"🔍 未找到与 '{query}' 相关的记忆"

            # 格式化结果
            formatted_results = []
            formatted_results.append(f"🔍 找到 {len(results)} 条相关记忆:")

            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆"
                }.get(memory.memory_type, memory.memory_type)

                content_preview = memory.content[:80] + "..." if len(memory.content) > 80 else memory.content
                importance = memory.metadata.get("importance", 0.5)
                formatted_results.append(
                    f"{i}. [{memory_type_label}] {content_preview} (重要性: {importance:.2f})"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            return f"❌ 搜索记忆失败: {str(e)}"

    # 搜索记忆返回dict
    def _search_memory_raw(
            self,
            query: str,
            limit: int = 5,
            memory_types: List[str] = None,
            memory_type: str = None,
            min_importance: float = 0.1
    ) -> List[dict]:
        """获取记忆原始结构化数据，不做字符串格式化，供ContextBuilder使用"""
        if memory_type and not memory_types:
            memory_types = [memory_type]
        results = self.memory_manager.retrieve_memories(
            query=query,
            limit=limit,
            memory_types=memory_types,
            min_importance=min_importance
        )
        # MemoryItem对象转字典，方便ContextBuilder解析
        raw_list = []
        for mem in results:
            raw_list.append({
                "item_id": mem.item_id,
                "content": mem.content,
                "timestamp": mem.created_at,
                "memory_type": mem.memory_type,
                "metadata": mem.metadata
            })
        return raw_list

    # 遗忘记忆
    def _forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> str:
        """遗忘记忆（支持多种策略）"""
        try:
            count = self.memory_manager.forget_memories(
                strategy=strategy,
                threshold=threshold,
                max_age_days=max_age_days
            )
            return f"🧹 已遗忘 {count} 条记忆（策略: {strategy}）"
        except Exception as e:
            return f"❌ 遗忘记忆失败: {str(e)}"


    # 整合记忆
    def _consolidate(self, from_type: str = "working", to_type: str = "episodic",
                     importance_threshold: float = 0.7) -> str:
        """整合记忆（将重要的短期记忆提升为长期记忆）"""
        try:
            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold,
            )
            return f"🔄 已整合 {count} 条记忆为长期记忆（{from_type} → {to_type}，阈值={importance_threshold}）"
        except Exception as e:
            return f"❌ 整合记忆失败: {str(e)}"
