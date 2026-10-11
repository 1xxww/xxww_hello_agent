from typing import List, Optional
import dashscope
import os
from dotenv import load_dotenv

load_dotenv()

class BaseEmbedder:
    async def encode(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class DashScopeEmbedder(BaseEmbedder):
    # 阿里 DashScope 文本向量模型单次调用上限（text-embedding-v4）
    BATCH_SIZE = 10

    def __init__(self, api_key: str, model_name: str = None, base_url: str = None):
        self.api_key = api_key or os.environ.get("EMBED_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        self.model_name = (
            model_name
            or os.environ.get("EMBED_MODEL_NAME")
            or os.environ.get("DASHSCOPE_MODEL_NAME")
            or "text-embedding-v4"
        )
        # 专属网关地址（阿里云 MaaS），默认用官方 DashScope 地址
        base_url = base_url or os.environ.get("EMBED_BASE_URL")
        if base_url:
            dashscope.base_http_api_url = base_url

    async def encode(self, texts: List[str]) -> List[List[float]]:
        dashscope.api_key = self.api_key
        all_vecs: List[List[float]] = []

        # 分批调用：单次最多 10 条，超出会报 400（batch size is invalid）
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            resp = dashscope.TextEmbedding.call(
                model=self.model_name,
                input=batch
            )
            # DashScope 成功响应没有 code 字段（为空），失败时 code 是错误码字符串，
            # 统一以 HTTP status_code 判断成败
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding api error:{resp.message or resp.code}")
            # 按 text_index 排序，保证返回顺序与输入一致
            items = sorted(
                resp.output["embeddings"],
                key=lambda x: x.get("text_index", 0),
            )
            all_vecs.extend(item["embedding"] for item in items)

        return all_vecs

class LocalSTEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    async def encode(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts)
        return [v.tolist() for v in vecs]

def create_embedding_model_with_fallback(
    dashscope_key: Optional[str] = None,
    local_model: str = "all-MiniLM-L6-v2"
) -> BaseEmbedder:
    """
    工厂函数：带降级策略
    1. 有 EMBED_API_KEY / DASHSCOPE_API_KEY → 使用远程Embedding（阿里 MaaS）
    2. 无key / 远程调用异常 → 切换本地sentence-transformers
    """
    # 优先显式参数，其次 EMBED_*（你的配置），兼容旧 DASHSCOPE_* 键名
    api_key = (
        dashscope_key
        or os.environ.get("EMBED_API_KEY")
    )
    if api_key and api_key.strip():
        try:
            return DashScopeEmbedder(api_key)
        except Exception:
            pass
    return LocalSTEmbedder(local_model)