import os
from typing import Optional, Any, List, Union, Dict, Iterator, AsyncIterator
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================================================================
# 1. 声明式服务商配置注册表 (Registry Table)
# 增删或扩展厂商只需在此添加配置项，核心代码逻辑零修改
# =========================================================================
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "url_hint": "api.deepseek.com",
        "key_prefix": "ds-",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "url_hint": "api.openai.com",
        "key_prefix": "sk-",
    },
    "dashscope": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "url_hint": "dashscope.aliyuncs.com",
        "key_prefix": "sk-dash",
    },
    "zhipu": {
        "env_key": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "default_model": "glm-4-flash",
        "url_hint": "open.bigmodel.cn",
    },
    "siliconflow": {
        "env_key": "SILICONFLOW_API_KEY",
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "url_hint": "api.siliconflow.cn",
        "key_prefix": "sf-",
    },
    "modelscope": {
        "env_key": "MODELSCOPE_API_KEY",
        "base_url": "https://api-inference.modelscope.cn/v1/",
        "default_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "url_hint": "api-inference.modelscope.cn",
        "key_prefix": "ms-",
    },
    "baichuan": {
        "env_key": "BAICHUAN_API_KEY",
        "base_url": "https://api.baichuan-ai.com/v1",
        "default_model": "Baichuan4",
        "url_hint": "api.baichuan-ai.com",
    },
    "minimax": {
        "env_key": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "abab6.5s-chat",
        "url_hint": "api.minimax.chat",
    },
    "step": {
        "env_key": "STEP_API_KEY",
        "base_url": "https://api.stepfun.com/v1",
        "default_model": "step-1-8k",
        "url_hint": "api.stepfun.com",
    },
    "ark": {
        "env_key": "ARK_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "doubao-pro-32k",
        "url_hint": "ark.cn-beijing.volces.com",
    },
    "ollama": {
        "env_key": "OLLAMA_API_KEY",
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "qwen2.5:7b",
        "url_hint": ":11434",
    },
    "vllm": {
        "env_key": "VLLM_API_KEY",
        "base_url": "http://127.0.0.1:8000/v1",
        "default_model": "default",
        "url_hint": ":8000",
    },
    "llama_cpp": {
        "env_key": "LLAMA_CPP_API_KEY",
        "base_url": "http://127.0.0.1:8080/v1",
        "default_model": "default",
        "url_hint": ":8080",
    },
    "modelscope_local": {
        "env_key": "MODELSCOPE_API_KEY",
        "base_url": "http://127.0.0.1:8001/v1",
        "default_model": "default",
        "url_hint": ":8001",
    },
    "local": {
        "env_key": "LOCAL_API_KEY",
        "base_url": None,
        "default_model": "default",
    },
    "auto": {
        "env_key": "LLM_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "default",
    },
}

LOCAL_PROVIDERS = {"ollama", "vllm", "llama_cpp", "modelscope_local", "local"}


# =========================================================================
# 2. 独立解析辅助函数 (向后兼容历史调用)
# =========================================================================
def auto_detect_provider(api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """自动检测服务商（严格优先级：显式入参特征识别 > 专属环境变量 > 兜底 auto）"""
    return MyLLM._detect_provider(api_key, base_url)


def resolve_credentials(provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """解析服务商的 API 密钥与请求地址"""
    p_cfg = PROVIDERS.get(provider, {})
    env_key = p_cfg.get("env_key", f"{provider.upper()}_API_KEY")
    ak = api_key or os.getenv(env_key) or os.getenv("LLM_API_KEY")
    bu = base_url or os.getenv("LLM_BASE_URL") or p_cfg.get("base_url")
    return ak, bu


# =========================================================================
# 3. 核心 MyLLM 统一客户端
# =========================================================================
class MyLLM:
    """
    现代化大语言模型网关。
    特性：
    - 统一兼容 OpenAI 协议的所有云端与本地大模型；
    - 纯配置表驱动，零冗余分支；
    - 支持 'provider/model' 现代语法糖（如 MyLLM('deepseek/deepseek-chat')）；
    - 支持纯字符串输入自动归一化为标准消息结构；
    - 统一请求载荷构建，DRY 原则消除重复代码。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
        **extra_kwargs
    ):
        # 1. 语法糖解析：支持 "deepseek/deepseek-chat" 格式自动拆分
        if model and "/" in model and not provider:
            parts = model.split("/", 1)
            candidate_p = parts[0].lower()
            if candidate_p in PROVIDERS:
                provider, model = candidate_p, parts[1]

        # 2. 自动推导服务商
        self.provider = provider or self._detect_provider(api_key, base_url)
        p_cfg = PROVIDERS.get(self.provider, {})

        # 3. 解析凭证与接口地址（显式入参 > 环境变量 > 注册表预设）
        env_key = p_cfg.get("env_key", f"{self.provider.upper()}_API_KEY")
        self.api_key = api_key or os.getenv(env_key) or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or p_cfg.get("base_url")

        # 本地推理服务无需真实密钥，填入占位符防空校验报错
        if self.provider in LOCAL_PROVIDERS and not self.api_key:
            self.api_key = "dummy_local_key"

        # 4. 友好、保姆级的缺失报错引导
        if not self.api_key and self.provider not in LOCAL_PROVIDERS:
            raise ValueError(
                f"\n❌ [{self.provider}] 未找到有效的 API Key！\n"
                f"为保护隐私与额度安全，框架不内置默认私钥。\n"
                f"请通过以下任一方式配置：\n"
                f"  1. 代码显式传入：MyLLM(api_key='your_key_here', provider='{self.provider}')\n"
                f"  2. 在本地项目根目录 .env 文件中配置：{env_key}=your_key_here\n"
                f"可参考根目录的 .env.example 模板文件进行快速配置。"
            )

        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"

        # 5. 模型名称兜底保障（保证 self.model 永不为 None）
        self.model = (
            model
            or os.getenv("LLM_MODEL_ID")
            or p_cfg.get("default_model")
            or "gpt-4o-mini"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_kwargs = extra_kwargs

        # 6. 初始化官方双客户端（同步与异步）
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def __repr__(self) -> str:
        return f"MyLLM(provider='{self.provider}', model='{self.model}', base_url='{self.base_url}')"

    @staticmethod
    def _detect_provider(api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
        """纯数据驱动的服务商自动推导"""
        # 优先级 1：显式传入的 BaseURL
        if base_url:
            bu_lower = base_url.lower()
            for p, cfg in PROVIDERS.items():
                if cfg.get("url_hint") and cfg["url_hint"] in bu_lower:
                    return p
            if any(h in bu_lower for h in ("localhost", "127.0.0.1", "0.0.0.0")):
                return "local"

        # 优先级 2：显式传入的 API Key 前缀
        if api_key:
            for p, cfg in PROVIDERS.items():
                if cfg.get("key_prefix") and api_key.startswith(cfg["key_prefix"]):
                    return p

        # 优先级 3：专属厂商环境变量
        for p, cfg in PROVIDERS.items():
            if cfg.get("env_key") and os.getenv(cfg["env_key"]):
                return p

        # 优先级 4：通用环境变量
        actual_bu = os.getenv("LLM_BASE_URL")
        if actual_bu:
            for p, cfg in PROVIDERS.items():
                if cfg.get("url_hint") and cfg["url_hint"] in actual_bu.lower():
                    return p

        actual_key = os.getenv("LLM_API_KEY")
        if actual_key:
            for p, cfg in PROVIDERS.items():
                if cfg.get("key_prefix") and actual_key.startswith(cfg["key_prefix"]):
                    return p

        # 优先级 5：兜底
        return "deepseek" if not os.getenv("LLM_BASE_URL") else "auto"

    def _normalize_messages(self, messages: Union[str, List[Any]]) -> List[Dict[str, str]]:
        """输入归一化：支持纯字符串、Message 对象列表或原始字典列表"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        norm = []
        for m in messages:
            if isinstance(m, dict):
                norm.append(m)
            elif hasattr(m, "to_dict"):
                norm.append(m.to_dict())
            elif hasattr(m, "role") and hasattr(m, "content"):
                norm.append({"role": m.role, "content": m.content})
            else:
                norm.append({"role": "user", "content": str(m)})
        return norm

    def _build_payload(self, messages: Any, stream: bool = False, **kwargs) -> Dict[str, Any]:
        """统一构建 OpenAI 兼容请求载荷"""
        payload = {
            "model": kwargs.pop("model", self.model),
            "messages": self._normalize_messages(messages),
            "temperature": kwargs.pop("temperature", self.temperature),
            "stream": stream,
            **self.extra_kwargs,
            **kwargs,
        }
        if self.max_tokens and "max_tokens" not in payload:
            payload["max_tokens"] = self.max_tokens
        return payload

    # ==================== 对外标准调用接口 ====================

    def chat(self, messages: Any, stream: bool = False, **override_kwargs) -> Union[str, Iterator[str]]:
        """【同步】对话调用。当 stream=True 时自动返回生成器"""
        if stream:
            return self.chat_stream(messages, **override_kwargs)
        payload = self._build_payload(messages, stream=False, **override_kwargs)
        resp = self._client.chat.completions.create(**payload)
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: Any, **override_kwargs) -> Iterator[str]:
        """【同步】流式打字机生成器"""
        payload = self._build_payload(messages, stream=True, **override_kwargs)
        response_stream = self._client.chat.completions.create(**payload)
        for chunk in response_stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    async def achat(self, messages: Any, stream: bool = False, **override_kwargs) -> Union[str, AsyncIterator[str]]:
        """【异步】非阻塞对话调用。当 stream=True 时自动返回异步生成器"""
        if stream:
            return self.achat_stream(messages, **override_kwargs)
        payload = self._build_payload(messages, stream=False, **override_kwargs)
        resp = await self._async_client.chat.completions.create(**payload)
        return resp.choices[0].message.content or ""

    async def achat_stream(self, messages: Any, **override_kwargs) -> AsyncIterator[str]:
        """【异步】流式生成器"""
        payload = self._build_payload(messages, stream=True, **override_kwargs)
        response_stream = await self._async_client.chat.completions.create(**payload)
        async for chunk in response_stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    async def achat_with_tools(self, messages: Any, tools_schema: list, tool_choice: Any = "auto", **kwargs) -> Any:
        """【异步】带 Function Calling 工具调用的原生响应对象返回"""
        payload = self._build_payload(
            messages,
            stream=False,
            tools=tools_schema,
            tool_choice=tool_choice,
            **kwargs
        )
        return await self._async_client.chat.completions.create(**payload)
