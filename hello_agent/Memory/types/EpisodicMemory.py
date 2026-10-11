from typing import Any, Dict, List,Optional

from ..MemoryBase import BaseMemory, MemoryConfig, MemoryItem,run_async
from hello_agent.Storage.document_store import SQLiteDocumentStore
from hello_agent.Storage.qdrant_store import QdrantVectorStore
from hello_agent.Storage.neo4j_store import create_embedding_model_with_fallback

COLLECTION = "episodic"  # 情景记忆专用 Qdrant 集合


class EpisodicMemory(BaseMemory):
    """情景记忆实现

    特点：
    - SQLite + Qdrant 混合存储：SQLite 存正文，Qdrant 存向量
    - 支持时间序列和会话级检索
    - 结构化过滤 + 语义向量检索
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.doc_store = SQLiteDocumentStore(config.database_path)
        self.vector_store = QdrantVectorStore(config.qdrant_url, config.qdrant_api_key)
        self.embedder = create_embedding_model_with_fallback()
        self.sessions: Dict[str, List[str]] = {}  # session_id -> item_ids
        self._ready = False

    # ==================== 写入 ====================

    def add(self, memory_item: MemoryItem) -> str:
        """同步契约（BaseMemory），内部转异步执行"""
        run_async(self._async_add(memory_item))
        return memory_item.item_id

    async def _async_add(self, memory_item: MemoryItem):
        await self._ensure_ready()

        # 1. SQLite 存正文（元数据）
        await self.doc_store.put(memory_item)

        # 2. 向量化
        vecs = await self.embedder.encode([memory_item.content])

        # 3. Qdrant 存向量（payload 带过滤字段）
        await self.vector_store.upsert(
            collection_name=COLLECTION,
            item_ids=[memory_item.item_id],
            vectors=vecs,
            payloads=[{**memory_item.metadata, "memory_type": "episodic"}],
        )

        # 4. 会话索引（支持按会话检索）
        sid = memory_item.metadata.get("session_id", "default")
        self.sessions.setdefault(sid, []).append(memory_item.item_id)

    # ==================== 检索 ====================

    def retrieve(self, query: str, limit: int = 5,user_id: Optional[str] = None, **kwargs) -> List[MemoryItem]:
        """向量检索：Qdrant 找相似 → SQLite 取正文 → 相似度×重要性评分"""
        return run_async(self._async_retrieve(query, limit,user_id=user_id))

    async def _async_retrieve(self, query: str, limit: int,user_id: Optional[str] = None,) -> List[MemoryItem]:
        await self._ensure_ready()

        qv = (await self.embedder.encode([query]))[0]
        qdrant_filter = {"memory_type": "episodic"}
        if user_id:
            qdrant_filter["user_id"] = user_id
        hits = await self.vector_store.search(
            collection_name=COLLECTION,
            query_vector=qv,
            top_k=max(limit * 5, 20),
            filter=qdrant_filter,
        )

        scored = []
        for h in hits:
            item = await self.doc_store.get(str(h.id))
            if item is None:
                continue
            importance = float(item.metadata.get("importance", 0.5))
            # 评分：向量相似度 × 重要性权重
            score = float(getattr(h, "score", 0.0)) * (0.8 + importance * 0.4)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:limit]]

    # ==================== 懒初始化 ====================

    async def _ensure_ready(self):
        """首次使用时连接 SQLite、探测维度、建 Qdrant 集合（含 payload 索引）"""
        if self._ready:
            return
        await self.doc_store.connect()
        self._dim = len((await self.embedder.encode(["probe"]))[0])
        await self.vector_store.ensure_collection(COLLECTION, self._dim)
        self._ready = True
