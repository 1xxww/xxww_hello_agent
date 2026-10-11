import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from .MemoryBase import MemoryConfig, MemoryItem
from ..Memory.types import *


class MemoryManager:
    """记忆管理器 - 统一的记忆操作接口（门面模式）
    4 个核心方法 = 记忆的完整生命周期：
    - add_memory          写入（把信息变成 MemoryItem 存进对应类型）
    - retrieve_memories   读取（跨类型检索、过滤、合并排序）
    - forget_memories     遗忘（容量控制，防止无限膨胀）
    - consolidate_memories 整合（短期 → 长期，模仿睡眠巩固）
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = False,
        enable_episodic: bool = False,
        enable_semantic: bool = False,
        enable_perceptual: bool = False
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id

        # 初始化各类型记忆（未完成的类型跳过，不影响整体）
        self.memory_types = {}

        if enable_working:
            try:
                self.memory_types['working'] = WorkingMemory(self.config)
            except Exception as e:
                print(f"[WARN] 工作记忆初始化失败，已跳过: {e}")

        if enable_episodic:
            try:
                self.memory_types['episodic'] = EpisodicMemory(self.config)
            except Exception as e:
                print(f"[WARN] 情景记忆初始化失败，已跳过: {e}")

        if enable_semantic:
            try:
                self.memory_types['semantic'] = SemanticMemory(self.config)
            except Exception as e:
                print(f"[WARN] 语义记忆初始化失败，已跳过: {e}")

        if enable_perceptual:
            try:
                self.memory_types['perceptual'] = PerceptualMemory(self.config)
            except Exception as e:
                print(f"[WARN] 感知记忆初始化失败，已跳过: {e}")

    # ==================== ① 写入 ====================

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        auto_classify: bool = False
    ) -> str:
        """写入一条记忆，返回 memory_id。

        思路：类型路由（门面职责）→ 组装统一载体 MemoryItem → 交给具体类型存储。
        """
        if auto_classify:
            memory_type = self._auto_classify(content)

        store = self.memory_types.get(memory_type)
        if store is None:
            raise ValueError(
                f"记忆类型 '{memory_type}' 未启用或未实现，当前可用: {list(self.memory_types.keys())}"
            )
        if not hasattr(store, "add"):
            raise ValueError(f"记忆类型 '{memory_type}' 未实现 add 接口")

        # 业务字段（重要性/用户ID）统一收进 metadata，保持 MemoryItem 模型稳定
        meta = dict(metadata or {})
        meta.setdefault("importance", importance)
        meta.setdefault("user_id", self.user_id)

        item = MemoryItem(
            item_id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            metadata=meta,
        )
        return store.add(item)

    @staticmethod
    def _auto_classify(content: str) -> str:
        """简易自动分类（学习用启发式，生产可换成 LLM 分类）"""
        if any(k in content for k in ("学习", "知识", "概念", "定义", "是")):
            return "semantic"
        if any(k in content for k in ("今天", "刚才", "昨天", "遇到", "发生")):
            return "episodic"
        return "working"

    # ==================== ② 读取 ====================

    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.1,
        **kwargs
    ) -> List[MemoryItem]:
        """跨类型检索：各类型打分 → 重要性过滤 → 合并排序。

        思路：门面不关心各类型内部怎么打分（策略模式），只负责
        收集、按统一规则过滤（min_importance）和合并排序。
        """
        types = memory_types or list(self.memory_types.keys())
        results: List[MemoryItem] = []

        for t in types:
            store = self.memory_types.get(t)
            if store is None or not hasattr(store, "retrieve"):
                continue
            try:
                items = store.retrieve(query, limit=limit, user_id=self.user_id, **kwargs)
            except Exception as e:
                print(f"[WARN] 记忆类型 '{t}' 检索失败: {e}")
                continue
            for item in items:
                importance = float(item.metadata.get("importance", 0.5)) if item.metadata else 0.5
                if importance >= min_importance:
                    results.append(item)

        # 合并排序：按重要性降序（不同类型间的统一可比较维度）
        results.sort(
            key=lambda it: float(it.metadata.get("importance", 0.5)),
            reverse=True
        )
        return results[:limit]

    # ==================== ③ 遗忘 ====================

    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30
    ) -> int:
        """按策略清理记忆，返回清理条数。

        思路：把"怎么清理"下放给各类型自己（有的按重要性、
        有的按时间），门面只负责遍历并汇总数量。
        """
        total = 0
        for store in self.memory_types.values():
            if not hasattr(store, "forget"):
                continue
            try:
                total += store.forget(
                    strategy=strategy,
                    threshold=threshold,
                    max_age_days=max_age_days,
                    user_id = self.user_id,
                )
            except Exception as e:
                print(f"[WARN] 记忆类型遗忘失败: {e}")
        return total

    # ==================== ④ 整合（短期 → 长期） ====================

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7
    ) -> int:
        """把高重要性的短期记忆提升为长期记忆，返回提升条数。

        思路（模仿睡眠巩固）：
        1. 从源类型取出重要性 ≥ 阈值的记忆
        2. 复制成目标类型的新 MemoryItem（保留原文 + 记录来源）
        3. 写入目标类型，并从源类型移除
        """
        from_store = self.memory_types.get(from_type)
        to_store = self.memory_types.get(to_type)
        if from_store is None or to_store is None:
            raise ValueError(
                f"整合需要 '{from_type}' 和 '{to_type}' 都启用，当前可用: {list(self.memory_types.keys())}"
            )
        if not hasattr(from_store, "all_items") or not hasattr(from_store, "remove_by_id"):
            raise ValueError(f"记忆类型 '{from_type}' 不支持整合所需接口")

        candidates = [
            m for m in from_store.all_items()
            if (not self.user_id or m.metadata.get("user_id") == self.user_id)
               and float(m.metadata.get("importance", 0.5)) >= importance_threshold
        ]

        count = 0
        for m in candidates:
            promoted = MemoryItem(
                item_id=str(uuid.uuid4()),
                content=m.content,
                memory_type=to_type,
                metadata={
                    **m.metadata,
                    "origin_id": m.item_id,
                    "promoted_at": datetime.now().isoformat(),
                },
            )
            to_store.add(promoted)
            from_store.remove_by_id(m.item_id)
            count += 1
        return count
