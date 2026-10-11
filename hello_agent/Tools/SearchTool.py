from typing import  List
from .Tool import Tool
from .ToolParameter import ToolParameter
import os
from dotenv import load_dotenv
from typing import Optional
import serpapi
from tavily import TavilyClient
load_dotenv()
# 实现Tool抽象类：网页搜索工具

class SearchTool(Tool):
    """
    智能混合搜索工具

    支持多种搜索引擎后端，智能选择最佳搜索源:
    1. 混合模式 (hybrid) - 智能选择TAVILY或SERPAPI
    2. Tavily API (tavily) - 专业AI搜索
    3. SerpApi (serpapi) - 传统Google搜索
    """

    def __init__(
        self,
        backend: str = "hybrid",
        tavily_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        serpapi_api_key: Optional[str] = None,
    ):
        super().__init__(
            name="search",
            description="一个智能网页搜索引擎。支持混合搜索模式，自动选择最佳搜索源。"
        )
        self.backend = backend
        # 显式入参优先，环境变量兜底
        self.tavily_key = tavily_api_key or tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_api_key or serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.search_sources = []
        self._setup_search_sources()

    def _search_with_tavily(self, query: str) -> str:
        """使用Tavily搜索"""
        tavily_client = TavilyClient(api_key=self.tavily_key)
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=3
        )

        result = f"🎯 Tavily AI搜索结果:{response.get('answer', '未找到直接答案')}\n\n"

        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:200]}...\n"
            result += f"    来源: {item.get('url', '')}\n\n"

        return result

    def _search_with_serpapi(self, query: str) -> str:
        """使用SerpApi搜索"""

        search = serpapi.GoogleSearch({
            "q": query,
            "api_key": self.serpapi_key,
            "num": 3
        })

        results = search.get_dict()

        result = "🔗 Google搜索结果:\n"
        if "organic_results" in results:
            for i, res in enumerate(results["organic_results"][:3], 1):
                result += f"[{i}] {res.get('title', '')}\n"
                result += f"    {res.get('snippet', '')}\n\n"

        return result

    def _setup_search_sources(self):
        """设置可用的搜索源"""
        # 检查Tavily可用性
        if self.tavily_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.search_sources.append("tavily")
                print("✅ Tavily搜索源已启用")
            except ImportError:
                print("⚠️ Tavily库未安装")

        # 检查SerpApi可用性
        if self.serpapi_key:
            try:
                import serpapi
                self.search_sources.append("serpapi")
                print("✅ SerpApi搜索源已启用")
            except ImportError:
                print("⚠️ SerpApi库未安装")

        if self.search_sources:
            print(f"🔧 可用搜索源: {', '.join(self.search_sources)}")
        else:
            print("⚠️ 没有可用的搜索源，请配置API密钥")

    def execute(self, **kwargs) -> str:
        """执行智能搜索"""
        query=kwargs["query"]
        if not query.strip():
            return "❌ 错误:搜索查询不能为空"

        # 检查是否有可用的搜索源
        if not self.search_sources:
            return """❌ 没有可用的搜索源，请配置以下API密钥之一:

            1. Tavily API: 设置环境变量 TAVILY_API_KEY
               获取地址: https://tavily.com/
            
            2. SerpAPI: 设置环境变量 SERPAPI_API_KEY
               获取地址: https://serpapi.com/
            
            配置后重新运行程序。"""

        print(f"🔍 开始智能搜索: {query}")

        # 尝试多个搜索源，返回最佳结果
        for source in self.search_sources:
            try:
                if source == "tavily":
                    result = self._search_with_tavily(query)
                    if result and "未找到" not in result:
                        return f"📊 Tavily AI搜索结果:\n\n{result}"

                elif source == "serpapi":
                    result = self._search_with_serpapi(query)
                    if result and "未找到" not in result:
                        return f"🌐 SerpApi Google搜索结果:\n\n{result}"

            except Exception as e:
                print(f"⚠️ {source} 搜索失败: {e}")
                continue

        return "❌ 所有搜索源都失败了，请检查网络连接和API密钥配置"

    def get_parameters(self) ->List[ToolParameter]:
        param_query = ToolParameter(
            name="query",
            type="string",
            description="需要搜索的关键词或问题",
            required=True
        )
        return [
            param_query,
        ]
