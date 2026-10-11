from hello_agent import Agent
import re
from typing import Optional, List, Tuple
from hello_agent import ToolRegistry
from hello_agent import MyLLM
from hello_agent.Agent.AgentConfig import AgentConfig
from hello_agent.Agent import Message


class ReActAgent(Agent):
    MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

    ## 可用工具
    {tools}

    ## 工作流程
    请严格按照以下格式进行回应，每次只能执行一个步骤:

    Thought: 分析当前问题，思考需要什么信息或采取什么行动。
    Action: 选择一个行动，格式必须是以下之一:
    - `{{tool_name}}[{{tool_input}}]` - 调用指定工具
    - `Finish[最终答案]` - 当你有足够信息给出最终答案时

    ## 重要提醒
    1. 每次回应必须包含Thought和Action两部分
    2. 工具调用的格式必须严格遵循:工具名[参数]
    3. 只有当你确信有足够信息回答问题时，才使用Finish
    4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

    ## 当前任务
    **Question:** {question}

    ## 执行历史
    {history}

    现在开始你的推理和行动:
    """
    def __init__(
        self,
        name: str,
        llm:MyLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else self.MY_REACT_PROMPT
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行ReAct Agent"""
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1. 构建提示词
            tools_desc = self.tool_registry.get_tool_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            # 2. 调用LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.chat(messages=messages, **kwargs)

            # 3. 解析输出
            thought, action = self._parse_output(response_text)

            # 4. 检查完成条件
            if action and action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer

            # 5. 执行工具调用
            if action:
                tool_name, tool_input = self._parse_action(action)
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")

        # 达到最大步数
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    def _parse_output(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析LLM输出，提取 Thought 和 Action
        返回 (thought, action)
        示例文本：
        Thought: 需要搜索最新行业数据
        Action: search[2026人工智能市场规模]
        """
        # 匹配 Thought: 直到遇到 Action: 或者文本结束
        pat_thought = r"Thought:\s*(.*?)(?=\nAction:|$)"
        # 匹配 Action: 后面整行内容
        pat_action = r"Action:\s*(.+)"

        m_thought = re.search(pat_thought, response_text, re.DOTALL)
        m_action = re.search(pat_action, response_text)

        thought = m_thought.group(1).strip() if m_thought else None
        action = m_action.group(1).strip() if m_action else None

        return thought, action

    def _parse_action(self, action: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析 Action 格式：name[content]
        输入：search[查询内容] / Finish[最终答案]
        返回 (名称, 参数内容)
        格式错误返回 ("", "")
        """
        # 前后允许少量空格容错
        pat = r"^\s*([^\[]+)\[\s*(.*?)\s*\]\s*$"
        match = re.match(pat, action)
        if not match:
            return "", ""
        act_name = match.group(1).strip()
        act_input = match.group(2).strip()
        return act_name, act_input