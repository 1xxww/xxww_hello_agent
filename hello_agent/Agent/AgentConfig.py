"""配置管理"""
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


class AgentConfig(BaseModel):
    """专门控制 Agent 运行行为的配置"""
    # 1. 记忆行为
    max_history_length: int = 10      # 历史上下文保留的最大轮数（防止爆 Token）

    # 2. 执行与工具策略
    max_tool_iterations: int = 5      # 单次任务允许工具反思的最大循环次数（防止死循环）

    # 3. 调试与可观测性
    verbose: bool = True              # 是否打印每一步的思考与执行过程（调试开关）

    # 4. 模型推理覆盖参数（可选）
    temperature: Optional[float] = None  # 如果指定，则覆盖 LLM 的默认温度

