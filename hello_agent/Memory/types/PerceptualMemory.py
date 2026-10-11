from typing import List
from ..MemoryBase import BaseMemory, MemoryConfig, MemoryItem


class PerceptualMemory(BaseMemory):
    """感知记忆实现（未实现，占位 stub）

    设计目标：
    - 支持多模态数据（文本、图像、音频等）
    - 跨模态相似性搜索
    - 感知数据的语义理解

    接入多模态编码器后的实现步骤：
    1. text  → 文本编码器（复用 create_embedding_model_with_fallback）
    2. image → CLIP 图像编码
    3. audio → CLAP 音频编码
    4. 按模态分库存储（perceptual_text / perceptual_image / perceptual_audio）
    5. payload 记录 modality、raw_data 路径，检索时支持按模态过滤
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config)
        self.storage_backend = storage_backend

    def add(self, memory_item: MemoryItem) -> str:
        """写入感知记忆（未实现：需先接入多模态编码器 CLIP/CLAP）"""
        raise NotImplementedError(
            "感知记忆 add 尚未实现：需先接入多模态编码器（CLIP/CLAP）"
        )

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索感知记忆（未实现：需先接入多模态编码器 CLIP/CLAP）"""
        raise NotImplementedError(
            "感知记忆 retrieve 尚未实现：需先接入多模态编码器（CLIP/CLAP）"
        )
