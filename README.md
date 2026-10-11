# ⚡ HelloAgent

<div align="center">

**面向生产级应用的高性能、模块化 Python AI Agent 核心开发框架**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Supported-purple.svg)](https://modelcontextprotocol.io/)
[![DeepSeek & OpenAI](https://img.shields.io/badge/LLM-DeepSeek%20%7C%20OpenAI%20%7C%20Qwen-emerald.svg)](https://platform.deepseek.com/)

[核心特性](#-核心特性) • [架构全景](#-架构全景) • [快速开始](#-快速开始) • [模块设计](#-模块设计) • [环境配置](#-环境配置)

</div>

---

## 🌟 核心特性

- 🤖 **统一大模型网关 (`MyLLM`)**
  - 原生适配 DeepSeek、OpenAI、阿里通义千问 (DashScope)、ModelScope、SiliconFlow 以及本地模型 (Ollama / vLLM / One-API)。
  - 支持多渠道服务商自适应探测、流式 Token 生成与统一参数配置。

- 🧠 **四层类脑认知记忆体系 (`Memory`)**
  - **工作记忆 (Working Memory)**：基于滑动窗口的即时会话上下文管理。
  - **情境记忆 (Episodic Memory)**：带有精确时间戳与因果关系的事件历程归档。
  - **语义记忆 (Semantic Memory)**：依托向量数据库持久化的抽象概念与长期知识沉淀。
  - **感知记忆 (Perceptual Memory)**：多模态感知与环境观测缓冲区。

- 🛠️ **工业级工具引擎与 MCP 原生集成 (`Tools & MCP`)**
  - 规范的 Tool 抽象基类与基于 Pydantic 的严密参数签名自动解析。
  - 原生打通 Anthropic 提出的 **Model Context Protocol (MCP)** 标准协议，支持以 Stdio 隔离进程快速挂载高德地图等三方外部工具链。
  - 内置计算器、文件便签 (`NoteTool`)、终端执行器 (`TerminaTool`)、知识库检索 (`RAGTool`) 与智能搜索 (`SearchTool`)。

- 📚 **混合知识库与动态检索 (`RAG & Storage`)**
  - 支持 Qdrant 高性能向量检索 + SQLite / 文档实体结构化双引擎存储。
  - 动态文本分块、嵌入降级回退与语义相关度重排。

- 🎯 **自适应上下文预算管理器 (`ContextBuilder`)**
  - 依据 LLM 最大 Token 窗口进行优先级切片打包 (`ContextPacket`)，兼顾系统指令、记忆召回、检索文档与历史对话的动态配额。

- ⚡ **PEP 562 按需懒加载优化**
  - 核心入口实现按需惰性导入，将基础微服务冷启动耗时大幅压缩，常驻内存占用降低超 70%。

---

## 🏗️ 架构全景

```text
HelloAgent/
├── hello_agent/               # 核心框架包
│   ├── Agent/                 # 智能体基类与核心形态实现
│   │   ├── Agent.py           # 抽象基类：生命周期、工具挂载与系统提示词
│   │   ├── SimpleAgent.py     # 基于 Function Calling 的多轮工具调用智能体
│   │   ├── ToolAwareSimpleAgent.py # 具备工具执行生命周期可观测性的智能体
│   │   └── AgentConfig.py     # Pydantic 声明式配置
│   ├── LLM/                   # 大模型统一接入网关 (MyLLM)
│   ├── Tools/                 # 工具注册中心、抽象基类与内置工具集
│   ├── Mcp/                   # Model Context Protocol (MCP) 标准进程连接桥
│   ├── Memory/                # 认知记忆体系 (Working / Episodic / Semantic / Perceptual)
│   ├── Context/               # Token 预算控制与上下文数据包构建 (ContextBuilder)
│   ├── RAG/                   # 文档解析、切分与 RAG 流水线
│   └── Storage/               # Qdrant 向量存储、SQLite 关系存储与 Neo4j 适配器
├── CodebaseMaintainer/        # 样例应用：代码库深度维护与分析智能体
├── LeaningAssistant/          # 样例应用：PDF 知识库与伴学问答智能体
├── ReActAgent/                # 样例应用：经典的 Thought-Action-Observation 推理循环
├── pyproject.toml             # 工程打包与依赖声明
└── .env.example               # 环境变量配置模板
```

---

## 🚀 快速开始

### 1. 安装环境

```bash
# 克隆仓库
git clone git@github.com:1xxww/xxww_hello_agent.git
cd xxww_hello_agent

# 建议在 Python 3.10+ 虚拟环境下安装
pip install -e .
```

### 2. 配置环境变量

复制配置模板并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：
```ini
# DeepSeek 推荐配置
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL_ID=deepseek-chat
```

### 3. 最简代码示例

#### 编写一个工具调用智能体：

```python
import asyncio
from hello_agent.LLM.Myllm import MyLLM
from hello_agent.Agent.SimpleAgent import SimpleAgent
from hello_agent.Tools.Tool import Tool
from hello_agent.Tools.ToolParameter import ToolParameter

# 1. 定义一个自定义工具
class AddTool(Tool):
    def __init__(self):
        super().__init__(
            name="add_numbers",
            description="计算两个整数之和",
            parameters=[
                ToolParameter(name="a", type="integer", description="第一个数字", required=True),
                ToolParameter(name="b", type="integer", description="第二个数字", required=True),
            ]
        )

    def run(self, a: int, b: int) -> str:
        return str(a + b)

# 2. 初始化大模型与智能体
llm = MyLLM()
agent = SimpleAgent(
    name="MathAssistant",
    llm=llm,
    system_prompt="你是一个善于使用工具辅助计算的数学助手。"
)
agent.register_tool(AddTool())

# 3. 运行对话与自动工具决策
async def main():
    response = await agent.run("请帮我计算 1024 加 2048 等于多少？")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ⚙️ 环境配置说明

项目通过 `python-dotenv` 自动加载 `.env` 配置，完整字段可参考 [`.env.example`](.env.example)：

| 环境变量 | 必选/可选 | 描述说明 | 示例取值 |
| :--- | :---: | :--- | :--- |
| `DEEPSEEK_API_KEY` | 必选 (默认) | DeepSeek 开放平台 API 密钥 | `sk-...` |
| `LLM_MODEL_ID` | 可选 | 默认调用的模型 ID | `deepseek-chat` / `gpt-4o-mini` |
| `AMAP_API_KEY` | 可选 | 高德地图 Web 服务 Key (供 MCP 服务使用) | 高德开放平台申请 |
| `TAVILY_API_KEY` | 可选 | Tavily AI 搜索服务密钥 (联网检索) | `tvly-...` |
| `QDRANT_URL` | 可选 | Qdrant 向量数据库服务地址 | `http://localhost:6333` |

---

## 📄 开源许可证

本项目采用 [MIT License](LICENSE) 授权许可。
