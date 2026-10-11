from typing import List, Dict, Optional, Any, Callable
import asyncio
import os
from qdrant_client import AsyncQdrantClient, models
from dotenv import load_dotenv

load_dotenv()


async def _with_retry(operation: Callable, retries: int = 3, delay: float = 1.0):
    """网络操作重试：应对瞬时连接抖动（Qdrant Cloud 偶发 TLS/超时）"""
    last_exc = None
    for attempt in range(retries):
        try:
            return await operation()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
    raise last_exc


class QdrantVectorStore:
    def __init__(self, url: str, api_key: Optional[str] = None):
        self.url = url or os.environ.get("QDRANT_URL")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            check_compatibility=False,  # 跳过版本探测，避免启动时无谓警告
            timeout=60,  # 云端实例（如巴西区）从国内访问延迟高，放宽超时
        )

    async def ensure_collection(self, collection_name: str, vector_size: int):
        async def _ensure():
            exists = await self.client.collection_exists(collection_name)
            if not exists:
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
            # 为过滤字段建 payload 索引（Qdrant Cloud 强制要求，否则 filter 查询报 400）
            for field, schema in [
                ("memory_type", models.PayloadSchemaType.KEYWORD),
                ("user_id", models.PayloadSchemaType.KEYWORD),
                ("rag_namespace", models.PayloadSchemaType.KEYWORD),
                ("data_source", models.PayloadSchemaType.KEYWORD),
                ("is_rag_data", models.PayloadSchemaType.BOOL),
            ]:
                try:
                    await self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=schema,
                    )
                except Exception:
                    pass  # 索引已存在等情形忽略
        await _with_retry(_ensure)

    async def upsert(self, collection_name: str,
                     item_ids: List[str],
                     vectors: List[List[float]],
                     payloads: List[Dict[str, Any]]):
        points = [
            models.PointStruct(
                id=idx,
                vector=vec,
                payload=payload
            )
            for idx, vec, payload in zip(item_ids, vectors, payloads)
        ]

        async def _upsert():
            await self.client.upsert(collection_name=collection_name, points=points)
        await _with_retry(_upsert)

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[models.ScoredPoint]:
        q_filter = None
        if filter:
            conditions = []
            for k, v in filter.items():
                conditions.append(models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v)
                ))
            q_filter = models.Filter(must=conditions)

        async def _search():
            res = await self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=q_filter
            )
            return res.points
        return await _with_retry(_search)

    async def delete(self, collection_name: str, item_ids: List[str]):
        async def _delete():
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=item_ids)
            )
        await _with_retry(_delete)
