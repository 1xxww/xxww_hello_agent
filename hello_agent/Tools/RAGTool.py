import asyncio
import os
from typing import Dict, Any, List, Optional

from .Tool import Tool
from .ToolParameter import ToolParameter
from ..LLM.Myllm import MyLLM
from ..RAG.Pipeline import create_rag_pipeline


class RAGTool(Tool):
    """RAG工具

    提供完整的 RAG 能力：
    - 添加多格式文档（PDF、Office、图片、音频等）
    - 智能检索与召回
    - LLM 增强问答
    - 知识库管理
    """

    def __init__(
            self,
            knowledge_base_path: str = "./knowledge_base",
            qdrant_url: str = None,
            qdrant_api_key: str = None,
            collection_name: str = "rag_knowledge_base",
            rag_namespace: str = "default",
            dashscope_key: Optional[str] = None,
            sqlite_db_path: Optional[str] = None
    ):
        # 初始化RAG管道
        self._pipelines: Dict[str, Any] = {}
        self.llm = MyLLM()
        super().__init__(
            name="RAG",
            description="检索增强工具，可读取已上传的文档知识库。当问题需要查阅文档资料、查找文档内相关信息时调用此工具获取参考内容。"
        )

        # 创建默认管道（未显式传入时回退到环境变量）
        default_pipeline = create_rag_pipeline(
            qdrant_url=qdrant_url or os.environ.get("QDRANT_URL"),
            qdrant_api_key=qdrant_api_key or os.environ.get("QDRANT_API_KEY"),
            collection_name=collection_name,
            rag_namespace=rag_namespace,
            dashscope_key=dashscope_key or os.environ.get("DASHSCOPE_API_KEY"),
            sqlite_db_path=sqlite_db_path
        )
        self._pipelines[rag_namespace] = default_pipeline
        self._default_namespace = rag_namespace

    # ==================== 内部工具方法 ====================

    def _get_pipeline(self, rag_namespace: Optional[str] = None) -> Any:
        ns = rag_namespace or self._default_namespace
        return self._pipelines.get(ns)

    @staticmethod
    def _run_async(coro):
        """在同步上下文中执行协程（兼容调用方已有事件循环的场景）"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    async def _retrieve(self, pipeline, query: str, limit: int, rag_namespace: str) -> List[Dict[str, Any]]:
        """向量检索：编码 → Qdrant搜索 → SQLite取正文"""
        qv = (await pipeline.embedder.encode([query]))[0]
        hits = await pipeline.vector_store.search(
            collection_name=pipeline.collection_name,
            query_vector=qv,
            top_k=limit,
            filter={"memory_type": "rag_chunk", "rag_namespace": rag_namespace}
        )

        results = []
        for h in hits:
            item = None
            try:
                item = await pipeline.doc_store.get(str(h.id))
            except Exception:
                item = None
            results.append({
                "id": str(h.id),
                "score": float(getattr(h, "score", 0.0)),
                "content": item.content if item else (h.payload or {}).get("content", ""),
                "source_path": (h.payload or {}).get("source_path", ""),
                "page_index": (h.payload or {}).get("page_index", 0),
            })
        return results

    # ==================== execute：LeaningAssistant 调用入口 ====================

    def execute(self, parameters: Dict[str, Any]) -> Any:
        if parameters["action"] == "add_document":
            return self._execute_add_document(**parameters)
        if parameters["action"] == "ask":
            return self._execute_ask(**parameters)
        if parameters["action"] == "search":
            return self._execute_search(**parameters)
        if parameters["action"] == "search_raw":
            return self._execute_search_raw(**parameters)
        if parameters["action"] == "stats":
            return self._execute_stats()
        return {"success": False, "error": f"未知操作: {parameters['action']}，支持 add_document / ask / search / stats"}

    def _execute_add_document(self, file_path: str = None, rag_namespace: str = None, **kwargs) -> Dict[str, Any]:
        if not file_path:
            return {"success": False, "error": "缺少 file_path 参数"}
        pipeline = self._get_pipeline(rag_namespace)
        if pipeline is None:
            return {"success": False, "error": f"未找到命名空间: {rag_namespace or self._default_namespace}"}
        try:
            self._run_async(pipeline.add_document(file_path))
            return {"success": True, "message": f"文档已入库: {file_path}"}
        except Exception as e:
            import traceback
            traceback.print_exc()  # 控制台保留完整堆栈，便于排查
            return {"success": False, "error": f"文档入库失败: {type(e).__name__}: {e}"}

    def _execute_ask(self, question: str = "", limit: int = 5, rag_namespace: str = None, **kwargs) -> str:
        if not question:
            return "❌ 缺少问题"
        pipeline = self._get_pipeline(rag_namespace)
        if pipeline is None:
            return "❌ 未找到对应知识库"
        try:
            results = self._run_async(
                self._retrieve(pipeline, question, int(limit), rag_namespace or self._default_namespace)
            )
        except Exception as e:
            return f"❌ 检索失败: {type(e).__name__}: {e}"
        if not results:
            return "未找到相关内容。"

        context = "\n\n".join(f"[{r['source_path']}] {r['content']}" for r in results)
        try:
            answer = self.llm.chat([
                {"role": "system", "content": "你是学习助手，根据提供的资料回答问题，简洁准确。"},
                {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"}
            ])
            return answer or "（无回答）"
        except Exception as e:
            return f"❌ 生成回答失败: {e}"

    # RAG搜索返回str
    def _execute_search(self, query: str = "", limit: int = 5, rag_namespace: str = None, **kwargs) -> str:
        if not query:
            return "❌ 缺少 query 参数"
        pipeline = self._get_pipeline(rag_namespace)
        if pipeline is None:
            return "❌ 未找到对应知识库"
        try:
            results = self._run_async(
                self._retrieve(pipeline, query, int(limit), rag_namespace or self._default_namespace)
            )
        except Exception as e:
            return f"❌ 检索失败: {type(e).__name__}: {e}"
        if not results:
            return "🔍 未找到相关结果"
        lines = [f"🔍 找到 {len(results)} 条结果:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. [score={r['score']:.3f}] {r['content'][:80]}")
        return "\n".join(lines)
    #RAG搜索返回dict
    def _execute_search_raw(self, query: str = "", limit: int = 5, rag_namespace: str = None, **kwargs) -> List[dict]:
        """RAG检索，返回原始字典列表，不拼接字符串，供ContextBuilder使用"""
        pipeline = self._get_pipeline(rag_namespace)
        if pipeline is None:
            return []
        results = self._run_async(
            self._retrieve(pipeline, query, int(limit), rag_namespace or self._default_namespace)
        )
        if not results:
            return []
        return results

    def _execute_stats(self) -> Dict[str, Any]:
        return {
            "pipelines": {
                ns: {"collection": p.collection_name, "namespace": p.rag_namespace}
                for ns, p in self._pipelines.items()
            },
            "default_namespace": self._default_namespace
        }

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型: add_document=文档入库 / ask=问答 / search=检索 / stats=统计",
                required=True
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="待入库文档路径(仅 add_document 需要)",
                required=False
            ),
            ToolParameter(
                name="question",
                type="string",
                description="问题(仅 ask 需要)",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="检索词(仅 search 需要)",
                required=False
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回条数，默认5",
                required=False
            ),
            ToolParameter(
                name="rag_namespace",
                type="string",
                description="知识库命名空间，默认使用初始化时的命名空间",
                required=False
            ),
        ]
