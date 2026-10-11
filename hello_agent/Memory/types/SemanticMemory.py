from typing import Any, Dict, List,Optional

from ..MemoryBase import BaseMemory, MemoryConfig, MemoryItem,run_async
from hello_agent.Storage.qdrant_store import QdrantVectorStore
from hello_agent.Storage.neo4j_store import create_embedding_model_with_fallback

COLLECTION = "semantic"  # 语义记忆专用 Qdrant 集合


class SemanticMemory(BaseMemory):
    """语义记忆实现（向量版）

    特点：
    - 使用远程 Embedding（阿里 text-embedding-v4）进行文本向量化
    - 向量检索快速相似度匹配
    - TODO: 接入 Neo4j 知识图谱，实现"向量 + 图"混合检索与实体关系推理
      （实体/关系缓存字段已预留：self.entities / self.relations）
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config)
        self.storage_backend = storage_backend
        self.embedder = create_embedding_model_with_fallback()
        self.vector_store = QdrantVectorStore(config.qdrant_url, config.qdrant_api_key)

        # TODO: 接入 Neo4jGraphStore 后保存实体与关系
        self.entities: Dict[str, Any] = {}
        self.relations: List[Any] = []

        self._ready = False

    # ==================== 写入 ====================

    def add(self, memory_item: MemoryItem) -> str:
        """同步契约（BaseMemory），内部转异步执行"""
        run_async(self._async_add(memory_item))
        return memory_item.item_id

    async def _async_add(self, memory_item: MemoryItem):
        await self._ensure_ready()

        vecs = await self.embedder.encode([memory_item.content])
        # payload 携带 content，便于检索时还原正文（向量版不依赖 SQLite）
        await self.vector_store.upsert(
            collection_name=COLLECTION,
            item_ids=[memory_item.item_id],
            vectors=vecs,
            payloads=[{
                **memory_item.metadata,
                "memory_type": "semantic",
                "content": memory_item.content,
            }],
        )
        # TODO: 实体/关系提取并写入 Neo4j 图数据库

    # ==================== 检索 ====================

    def retrieve(self, query: str, limit: int = 5, user_id: Optional[str] = None,**kwargs) -> List[MemoryItem]:
        """向量检索（TODO: 合并图检索结果，复用原 _combine_and_rank_results 框架）"""
        return run_async(self._async_retrieve(query, limit))

    async def _async_retrieve(self, query: str, limit: int,user_id: Optional[str] = None) -> List[MemoryItem]:
        await self._ensure_ready()

        qv = (await self.embedder.encode([query]))[0]
        qdrant_filter = {"memory_type": "semantic"}
        if user_id:
            qdrant_filter["user_id"] = user_id

        hits = await self.vector_store.search(
            collection_name=COLLECTION,
            query_vector=qv,
            top_k=max(limit * 5, 20),
            filter=qdrant_filter,
        )

        # TODO: 合并图检索结果
        scored = [(float(getattr(h, "score", 0.0)), h) for h in hits]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            MemoryItem(
                item_id=str(h.id),
                content=(h.payload or {}).get("content", ""),
                memory_type="semantic",
                metadata=(h.payload or {}),
            )
            for _, h in scored[:limit]
        ]

    # ==================== 懒初始化 ====================

    async def _ensure_ready(self):
        """首次使用时探测维度、建 Qdrant 集合（含 payload 索引）"""
        if self._ready:
            return
        self._dim = len((await self.embedder.encode(["probe"]))[0])
        await self.vector_store.ensure_collection(COLLECTION, self._dim)
        self._ready = True
