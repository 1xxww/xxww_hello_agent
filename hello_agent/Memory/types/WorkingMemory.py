import math
import re
from datetime import datetime
from typing import List,Optional
from ..MemoryBase import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    """工作记忆实现
    特点：
    - 容量有限（默认50条）+ TTL自动清理
    - 纯内存存储，访问速度极快
    - 组合词法检索：TF-IDF 词项权重相似度 + 关键词匹配
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.max_capacity = config.working_memory_capacity or 50
        self.max_age_time = config.working_memory_ttl or 60
        self.memories: List[MemoryItem] = []

    # ==================== 生命周期管理 ====================

    def _expire_old_memories(self):
        """TTL清理：超过 max_age_minutes 的记忆移除"""
        now = datetime.now()
        self.memories = [
            m for m in self.memories
            if (now - m.created_at).total_seconds() / 60 < self.max_age_time
        ]

    def _remove_lowest_priority_memory(self):
        """容量管理：移除重要性最低（同分取最旧）的一条"""
        if not self.memories:
            return
        self.memories.sort(key=lambda m: (self._importance(m), m.created_at))
        self.memories.pop(0)

    @staticmethod
    def _importance(m: MemoryItem) -> float:
        """重要性统一从 metadata 读取（MemoryItem 无 importance 字段）"""
        return float(m.metadata.get("importance", 0.5))

    # ==================== 核心 CRUD ====================

    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆，返回 memory_id"""
        self._expire_old_memories()
        if len(self.memories) >= self.max_capacity:
            self._remove_lowest_priority_memory()
        self.memories.append(memory_item)
        return memory_item.item_id

    def retrieve(self, query: str, limit: int = 5, user_id: Optional[str] = None,**kwargs) -> List[MemoryItem]:
        """组合词法检索：TF 词项相似度 + 关键词匹配 + 时间衰减 + 重要性权重"""
        self._expire_old_memories()
        # ✅ 1. 先做多用户数据过滤（只挑选当前用户的记忆，或者没有标记 user_id 的公共记忆）
        candidate_memories = self.memories
        if user_id:
            candidate_memories = [
                m for m in self.memories
                if m.metadata.get("user_id") == user_id or not m.metadata.get("user_id")
            ]

        if not query:
            return candidate_memories[:limit]

        tf_scores = self._try_tfidf_search(query)
        scored = []
        # ✅ 2. 遍历 candidate_memories（而不是原来的 self.memories）
        for memory in candidate_memories:
            vec = tf_scores.get(memory.item_id, 0.0)
            kw = self._calculate_keyword_score(query, memory.content)
            base = vec * 0.7 + kw * 0.3 if vec > 0 else kw
            decay = self._calculate_time_decay(memory.created_at)
            final = base * decay * (0.8 + self._importance(memory) * 0.4)
            if final > 0:
                scored.append((final, memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored[:limit]]

    def all_items(self) -> List[MemoryItem]:
        """供 consolidate 读取全部记忆"""
        self._expire_old_memories()
        return list(self.memories)

    def remove_by_id(self, item_id: str) -> bool:
        """供 consolidate 移除已提升的记忆"""
        before = len(self.memories)
        self.memories = [m for m in self.memories if m.item_id != item_id]
        return len(self.memories) < before

    def forget(self, strategy: str = "importance_based",
               threshold: float = 0.1, max_age_days: int = 30,user_id: Optional[str] = None) -> int:
        """遗忘策略：
        - importance_based: 移除重要性 < threshold 的记忆
        - time_based: 移除超过 max_age_days 的记忆
        - 其他: 清空全部
        """
        now = datetime.now()
        kept = []
        removed_count = 0

        for m in self.memories:
            # 如果不是当前用户的记忆，绝对不能动，直接保留！
            if user_id and m.metadata.get("user_id") != user_id:
                kept.append(m)
                continue

            # 是当前用户的记忆，才按策略判断是否需要删除
            should_delete = False
            if strategy == "importance_based":
                should_delete = (self._importance(m) < threshold)
            elif strategy == "time_based":
                should_delete = ((now - m.created_at).days >= max_age_days)
            else:
                should_delete = True  # 清空策略

            if should_delete:
                removed_count += 1
            else:
                kept.append(m)

        self.memories = kept
        return removed_count

    # ==================== 评分算法 ====================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简易分词：中文按单字、英文/数字按整词（学习用简化版）"""
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())

    def _try_tfidf_search(self, query: str) -> dict:
        """简化 TF 评分：查询词项在内容中出现的频率占比"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}
        scores = {}
        for memory in self.memories:
            content_tokens = self._tokenize(memory.content)
            if not content_tokens:
                continue
            hits = sum(content_tokens.count(t) for t in set(query_tokens))
            scores[memory.item_id] = hits / len(content_tokens)
        return scores

    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """关键词匹配得分：查询词项命中比例"""
        query_tokens = set(self._tokenize(query))
        content_tokens = set(self._tokenize(content))
        if not query_tokens:
            return 0.0
        return len(query_tokens & content_tokens) / len(query_tokens)

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """时间衰减：越新分越高（指数衰减，半衰期 = max_age_minutes）"""
        age_minutes = max(0.0, (datetime.now() - timestamp).total_seconds() / 60)
        return math.exp(-age_minutes / self.max_age_time)
