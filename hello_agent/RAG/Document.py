import os
import uuid
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element


class DocumentChunk:
    """文档切块实体，临时载体，流转到RAGPipeline存入记忆库"""
    def __init__(
        self,
        chunk_id: str,
        content: str,
        source_path: str,
        page_index: int = 0,
        metadata: dict = None
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.source_path = source_path
        self.page_index = page_index
        self.metadata = metadata or {}


class DocumentProcessor:
    """
    增强版文档处理器：
    1. 自动解析多格式文档 → 自行组装标准Markdown（标题、列表、段落）
    2. 先根据Markdown标题层级进行初次切分
    3. 再使用递归分割器对超长段落二次切分
    """
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 80
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 标题分割器：识别各级标题，保留标题上下文
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False
        )

        # 递归文本分割器：对标题块内超长文本二次切割
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
            keep_separator=True
        )

    @staticmethod
    def _elements_to_markdown(elements: List[Element]) -> str:
        """手动实现：将unstructured解析后的元素列表转换为Markdown字符串"""
        md_lines = []
        for elem in elements:
            cat = elem.category
            txt = elem.text.strip()
            if not txt:
                continue

            if cat == "Title":
                md_lines.append(f"# {txt}")
            elif cat == "Header":
                md_lines.append(f"## {txt}")
            elif cat == "SubHeader":
                md_lines.append(f"### {txt}")
            elif cat == "ListItem":
                md_lines.append(f"- {txt}")
            elif cat == "Table":
                md_lines.append(f"\n> TABLE:\n{txt}\n")
            else:
                # Paragraph, NarrativeText等普通正文
                md_lines.append(txt)
        return "\n\n".join(md_lines)

    async def load_file_to_markdown(self, file_path: str) -> str:
        """加载任意文档，解析并转换为Markdown文本"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        elements = partition(filename=str(path))
        md_content = self._elements_to_markdown(elements)
        return md_content


    async def load_and_split(self, file_path: str) -> List[DocumentChunk]:
        """
        完整链路：文件 → Markdown → 按标题层级切分 → 超长块二次分割 → 输出DocumentChunk
        """
        md_text = await self.load_file_to_markdown(file_path)
        source_abs_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)

        # 第一步：按标题分割，每一块自带标题元数据
        header_sections = self.header_splitter.split_text(md_text)

        chunk_list: List[DocumentChunk] = []
        page_counter = 0

        for section in header_sections:
            # section.metadata 自动携带各级标题信息 {h1:xxx, h2:xxx...}
            section_meta = section.metadata.copy()
            section_text = section.page_content.strip()
            if not section_text:
                continue

            # 第二步：单个标题段落过长，继续二次切分
            sub_chunks = self.text_splitter.split_text(section_text)

            for sub_text in sub_chunks:
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=sub_text,
                    source_path=source_abs_path,
                    page_index=page_counter,
                    metadata={
                        "filename": filename,
                        **section_meta
                    }
                )
                chunk_list.append(chunk)
                page_counter += 1

        return chunk_list
