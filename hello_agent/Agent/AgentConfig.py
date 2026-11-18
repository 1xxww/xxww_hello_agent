"""配置管理"""
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


class AgentConfig(BaseModel):
    max_history_length: int = 10
    max_tool_iterations: int = 5
    verbose: bool = True
    temperature: Optional[float] = None

    # 工具重试
    tool_max_attempts: int = 4
    tool_timeout_seconds: float = 30.0
    tool_retry_base_delay: float = 0.5
    tool_retry_max_delay: float = 4.0
    tool_retry_jitter: float = 0.3
    tool_retry_idempotent: bool = True

    # 同一个工具连续参数错误的次数上限
    max_argument_errors_per_tool: int = 2

