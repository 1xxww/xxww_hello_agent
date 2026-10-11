import time
from typing import Any, Dict,Optional
from datetime import datetime
import os
import json
from hello_agent.Tools.RAGTool import RAGTool
from hello_agent.Tools.MemoryTool import MemoryTool

class LeaningAssistant:
    """"学习助手"""
    def __init__(self,user_id:str= "default_user_name"):
        """"user_id 用来隔绝不同用户之间的数据"""
        self.user_id = user_id
        """"根据系统时间生成session_id"""
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


        """初始化memorytool和RAGtool"""
        self.memory_tool = MemoryTool(user_id=user_id)
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")

        # 学习统计
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0
        }

        # 当前加载的文档
        self.current_document = None

    def load_document(self,path:str) -> Dict[str, Any]:
        """加载PDF文档到知识库

        Args:
            path: 文件路径

        Returns:
            Dict: 包含success和message的结果
    """
        #确认文件存在
        if not os.path.exists(path):
            return {"success": False, "message": f"文件不存在: {path}"}

        start_time = time.time()

        #处理文件: MarkItDown转换 → 智能分块 → 向量化
        result = self.rag_tool.execute(
            "add_document",
            file_path=path,
            chunk_size=1000,
            chunk_overlap=200
        )

        process_time = time.time() - start_time

        if result.get("success", False):
            self.current_document = os.path.basename(path)
            self.stats["documents_loaded"] += 1

            # 【MemoryTool】记录到学习记忆
            self.memory_tool.execute(
                "add",
                content=f"加载了文档《{self.current_document}》",
                memory_type="episodic",
                importance=0.9,
                event_type="document_loaded",
                session_id=self.session_id
            )

            return {
                "success": True,
                "message": f"加载成功！(耗时: {process_time:.1f}秒)",
                "document": self.current_document
            }
        else:
            return {
                "success": False,
                "message": f"加载失败: {result.get('error', '未知错误')}"
            }

    def ask(self,question:str,use_advanced_search: bool = True) -> str:
        """向文档提问

        Args:
            question: 用户问题
            use_advanced_search: 是否使用高级检索（MQE + HyDE）

        Returns:
            str: 答案
        """

        if not self.current_document:
            return "请先加载文档"

        # 【MemoryTool】记录问题到工作记忆
        self.memory_tool.execute(
            "add",
            content=f"提问: {question}",
            memory_type="working",
            importance=0.6,
            session_id=self.session_id
        )

        # 【RAGTool】使用高级检索获取答案
        answer = self.rag_tool.execute(
            "ask",
            question=question,
            limit=5,
            enable_advanced_search=use_advanced_search,
            enable_mqe=use_advanced_search,
            enable_hyde=use_advanced_search
        )

        # 【MemoryTool】记录到情景记忆
        self.memory_tool.execute(
            "add",
            content=f"关于'{question}'的学习",
            memory_type="episodic",
            importance=0.7,
            event_type="qa_interaction",
            session_id=self.session_id
        )

        self.stats["questions_asked"] += 1

        return answer

    def add_note(self, content: str, concept: Optional[str] = None):
        """添加学习笔记"""
        self.memory_tool.execute(
            "add",
            content=content,
            memory_type="semantic",
            importance=0.8,
            concept=concept or "general",
            session_id=self.session_id
        )
        self.stats["concepts_learned"] += 1

    def recall(self, query: str, limit: int = 5) -> str:
        """回顾学习历程"""
        result = self.memory_tool.execute(
            "search",
            query=query,
            limit=limit
        )
        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        return {
            "会话时长": f"{duration:.0f}秒",
            "加载文档": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "学习笔记": self.stats["concepts_learned"],
            "当前文档": self.current_document or "未加载"
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """生成学习报告"""
        memory_summary = self.memory_tool.execute("summary", limit=10)
        rag_stats = self.rag_tool.execute("stats")

        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"]
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats
        }

        if save_to_file:
            report_file = f"learning_report_{self.session_id}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            report["report_file"] = report_file

        return report
