from typing import List, Optional, Dict, Any
from ..Memory import MemoryItem
from ..LLM import MyLLM
from ..RAG.Document import DocumentProcessor, DocumentChunk
from ..Storage.document_store import SQLiteDocumentStore
from ..Storage.neo4j_store import create_embedding_model_with_fallback
from ..Storage.qdrant_store import QdrantVectorStore


class RAGPipeline:
    """
    端到端RAG流水线
    一条Pipeline绑定：集合名称 + rag_namespace
    支持：文档入库、语义检索
    """
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: Optional[str],
        collection_name: str,
        rag_namespace: str,
        dashscope_key: Optional[str] = None,
        sqlite_db_path: str = None
    ):
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace

        # 存储层
        self.vector_store = QdrantVectorStore(qdrant_url, qdrant_api_key)
        if not sqlite_db_path:
            sqlite_db_path = f"./rag_{collection_name}.db"
        self.doc_store = SQLiteDocumentStore(sqlite_db_path)

        # 嵌入器 + 文档处理器
        self.embedder = create_embedding_model_with_fallback(dashscope_key=dashscope_key)
        self.doc_processor = DocumentProcessor()

        # 向量维度（自动由embedder编码一次探测，也可配置传入）
        self._vector_size: Optional[int] = None

    async def _init_collection(self):
        """探测向量维度，自动创建Qdrant集合"""
        if self._vector_size is None:
            test_vec = await self.embedder.encode(["test"])
            self._vector_size = len(test_vec[0])
        await self.vector_store.ensure_collection(self.collection_name, self._vector_size)
        await self.doc_store.connect()

    async def add_document(self, file_path: str):
        """
        入口：上传文档
        流程：解析文件 → 切块 → 转为MemoryItem → 向量化 → 双库写入
        """
        await self._init_collection()
        chunks: List[DocumentChunk] = await self.doc_processor.load_and_split(file_path)

        # 构造批量数据
        item_ids = []
        contents = []
        payloads = []
        memory_items: List[MemoryItem] = []

        for chunk in chunks:
            item_id = chunk.chunk_id
            meta = {
                "rag_namespace": self.rag_namespace,
                "memory_type": "rag_chunk",
                "source_path": chunk.source_path,
                "page_index": chunk.page_index,
                **chunk.metadata
            }
            mem_item = MemoryItem(
                item_id=item_id,
                content=chunk.content,
                memory_type="rag_chunk",
                metadata=meta
            )
            memory_items.append(mem_item)
            item_ids.append(item_id)
            contents.append(chunk.content)
            payloads.append(meta)

        # 批量向量化
        vectors = await self.embedder.encode(contents)

        # 写入SQLite元数据
        for item in memory_items:
            await self.doc_store.put(item)

        # 写入Qdrant向量
        await self.vector_store.upsert(
            collection_name=self.collection_name,
            item_ids=item_ids,
            vectors=vectors,
            payloads=payloads
        )

    def _prompt_mqe(self,query: str, n: int) -> List[str]:
        """使用LLM生成多样化的查询扩展"""
        try:
            llm = MyLLM()
            prompt = [
                {"role": "system",
                 "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
                {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"}
            ]
            text = llm.chat(prompt)
            lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
            outs = [ln for ln in lines if ln]
            return outs[:n] or [query]
        except Exception:
            return [query]

    def _prompt_hyde(self,query: str) -> Optional[str]:
        """生成假设性文档用于改善检索"""
        try:
            llm = MyLLM()
            prompt = [
                {"role": "system",
                 "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
                {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"}
            ]
            return llm.chat(prompt)
        except Exception:
            return None

    async def search_vectors_expanded(
            self,
            store=None,
            query: str = "",
            top_k: int = 8,
            rag_namespace: Optional[str] = None,
            only_rag_data: bool = True,
            score_threshold: Optional[float] = None,
            enable_mqe: bool = False,
            mqe_expansions: int = 2,
            enable_hyde: bool = False,
            candidate_pool_multiplier: int = 4,
    ) -> List[Dict]:
        """
        Search with query expansion using unified embedding and Qdrant.
        """
        if not query:
            return []


        # 查询扩展
        expansions: List[str] = [query]

        if enable_mqe and mqe_expansions > 0:
            expansions.extend(self._prompt_mqe(query, mqe_expansions))
        if enable_hyde:
            hyde_text = self._prompt_hyde(query)
            if hyde_text:
                expansions.append(hyde_text)

        # 去重和修剪
        uniq: List[str] = []
        for e in expansions:
            if e and e not in uniq:
                uniq.append(e)
        expansions = uniq[: max(1, len(uniq))]

        # 分配候选池
        pool = max(top_k * candidate_pool_multiplier, 20)
        per = max(1, pool // max(1, len(expansions)))

        # 构建RAG数据过滤器
        where = {"memory_type": "rag_chunk"}
        if only_rag_data:
            where["is_rag_data"] = True
            where["data_source"] = "rag_pipeline"
        if rag_namespace:
            where["rag_namespace"] = rag_namespace

        # 收集所有扩展查询的结果
        agg: Dict[str, Dict] = {}
        for q in expansions:
            vec_result = await self.embedder.encode([q])
            qv = vec_result[0]
            hits = store.search_similar(
                query_vector=qv,
                limit=per,
                score_threshold=score_threshold,
                where=where
            )
            for h in hits:
                mid = h.get("metadata", {}).get("memory_id", h.get("id"))
                s = float(h.get("score", 0.0))
                if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                    agg[mid] = h

        # 按分数排序返回
        merged = list(agg.values())
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[:top_k]


def create_rag_pipeline(
    qdrant_url: str,
    qdrant_api_key: Optional[str],
    collection_name: str,
    rag_namespace: str,
    dashscope_key: Optional[str] = None,
    sqlite_db_path: Optional[str] = None
) -> RAGPipeline:
    """
    工厂函数，上层RAGTool直接调用
    解决你之前：未解析引用 create_rag_pipeline
    """
    pipeline = RAGPipeline(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        rag_namespace=rag_namespace,
        dashscope_key=dashscope_key,
        sqlite_db_path=sqlite_db_path
    )
    return pipeline