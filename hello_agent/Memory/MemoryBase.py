import asyncio
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, ConfigDict, Field
import os
from dotenv import load_dotenv


load_dotenv()


class _LoopRunner:
    """持久事件循环运行器

    为什么需要：AsyncQdrantClient / aiosqlite 等异步资源绑定在创建时的
    事件循环上。若每次用 asyncio.run() 新建 loop，第二次调用就会报
    "Event loop is closed"。用一个常驻后台线程的固定 loop 解决。
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


_runner = _LoopRunner()


def run_async(coro):
    """在同步上下文中执行协程（统一走持久事件循环，连接可复用）"""
    return _runner.run(coro)

class MemoryItem(BaseModel):
    """
    统一记忆条目模型
    工作记忆/情景记忆/语义记忆/感知记忆/RAG块 通用载体
    """
    item_id: str
    content: str
    memory_type: str       # working / episodic / semantic / perceptual / rag_chunk
    created_at: datetime = Field(default_factory=datetime.now)  # 本地时间，与项目其他 datetime.now() 保持一致
    expire_at: Optional[datetime] = None   # TTL过期时间，工作记忆使用
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None  # 可选向量

    model_config = ConfigDict(arbitrary_types_allowed=True)

class MemoryConfig(BaseModel):
    """记忆模块全局配置"""
    namespace: str = "default"
    top_k: int = 5
    enable_rerank: bool = False
    ttl_seconds: Optional[int] = None  # 用于工作记忆TTL
    embedding_model: Optional[str] = Field(default_factory=lambda: os.getenv("EMBED_MODEL_NAME", "text-embedding-v4"))

    # ---- 工作记忆（纯内存） ----
    working_memory_capacity: int = 50   # 容量上限（条）
    working_memory_ttl: int = 60        # 过期时间（分钟）

    # ---- 持久化存储（情景/语义记忆用） ----
    database_path: str = "./memory.db"
    qdrant_url: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_URL"))
    qdrant_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))

class BaseMemory(ABC):
    """
    所有记忆类型的抽象父类
    强制统一CRUD接口，MemoryManager依靠这套接口调度
    """
    def __init__(self, config: MemoryConfig):
        self.config: MemoryConfig = config

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """写入一条记忆，返回 memory_id（MemoryManager.add_memory 调用）"""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索记忆，返回按相关度排序的 MemoryItem 列表"""
        raise NotImplementedError