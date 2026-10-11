# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**HelloAgent** is a modular Python-based AI agent framework supporting multiple LLM providers, tool use, memory systems, RAG pipelines, and specialized agent implementations (learning assistant, travel planner, codebase maintainer).

## Development Environment & Commands

The project uses Python 3.11 with a local virtual environment (`.venv`).

### Virtual Environment
- Windows activation: `.venv\Scripts\activate`
- Interpreter path: `.venv\Scripts\python.exe`

### Package Installation
- **Editable Development Install:**
  ```bash
  pip install -e .
  # With optional features (MCP, RAG, Search):
  pip install -e ".[all]"
  ```

### Running Applications & Agents
- **Gradio Learning Assistant UI:**
  ```bash
  python gradio_app.py
  # Accessible at http://127.0.0.1:7860
  ```
- **Travel Assistant Backend API (FastAPI):**
  ```bash
  cd travelAssistant/backend
  uvicorn server:app --reload --port 8000
  # API Health Check: http://localhost:8000/api/health
  # OpenAPI Docs: http://localhost:8000/docs
  ```
- **Travel Assistant Frontend Web App (React + Vite + Tailwind):**
  ```bash
  cd travelAssistant/frontend
  npm run dev
  # Accessible at http://localhost:5173
  ```
- **AMAP MCP Server standalone test:**
  ```bash
  python travelAssistant/backend/mcp_test.py
  ```
- **Codebase Maintainer Agent:**
  ```bash
  python -c "from CodebaseMaintainer import CodebaseMaintainer; from Myllm import MyLLM; m = CodebaseMaintainer('test', '.', 'maintainer', MyLLM()); print(m.run('检查项目结构'))"
  ```

### Dependencies & Requirements
- Packages are installed in `.venv` (see `pip list`). Key dependencies include `openai`, `mcp`, `pydantic`, `qdrant-client`, `gradio`, `sentence-transformers`, `torch`, `aiosqlite`, `python-dotenv`.
- Configuration is loaded from `.env` via `python-dotenv`.

## Architecture & Core Modules

All framework core code is packaged under `hello_agent/`:

### 1. LLM Integration (`hello_agent/LLM/Myllm.py`, `hello_agent/Agent/AgentConfig.py`)
- `MyLLM`: Core wrapper around OpenAI-compatible APIs (OpenAI, DeepSeek, ModelScope, DashScope, SiliconFlow, vLLM/Ollama).
- `auto_detect_provider()`: Auto-detects provider by environment variable presence, base URL patterns, and API key prefixes.
- `AgentConfig.py`: Pydantic settings class controlling model ID, temperature, log levels, and max tokens.

### 2. Agent Hierarchy (`hello_agent/Agent/`)
- `Agent.py` (`Agent`): Base abstract class defining async lifecycle (`run`), tool mounting (`register_tools`), message history (`_history`), and system prompt handling.
- `SimpleAgent.py` (`SimpleAgent`): Direct conversation and iterative tool calling (`_run_with_tools`) using standard OpenAI function calling schemas.
- `ReActAgent/ReActAgent.py` (`ReActAgent`): Classic ReAct loop (`Thought` -> `Action: tool[input]` -> `Observation`).
- `ToolAwareSimpleAgent.py`: Agent with tool lifecycle event interception and logging.
- `CodebaseMaintainer/CodebaseMaintainer.py`: Long-horizon maintenance agent integrating `ContextBuilder`, `NoteTool`, `TerminalTool`, and `MemoryTool`.
- `LeaningAssistant/LeaningAssistant.py`: Learning companion combining PDF document ingestion, RAG retrieval, and memory logging.
- `travelAssistant/backend/TripPlannerAgent.py`: Multi-agent travel planner coordinating attraction, weather, and hotel sub-agents with an MCP-backed tool suite.

### 3. Tool System (`hello_agent/Tools/`)
- `Tool.py` & `ToolParameter.py`: Base abstract class and parameter schema definitions.
- `ToolRegistry.py`: Central registry for registering, discovering, and generating function schemas for LLMs.
- `MCPTool.py` (`MCPServerManager`, `MCPTool`): Bridge connecting Model Context Protocol (MCP) servers (stdio transport) to Agent tools.
- Specialized built-in tools:
  - `MemoryTool.py`: Interacts with hierarchical memory.
  - `RAGTool.py`: Document indexing and semantic retrieval.
  - `TerminaTool.py`: Workspace command execution.
  - `NoteTool.py`: File-based note taking.
  - `travelAssistant/backend/mcp_services/amap_mcp.py`: Travel-assistant dedicated MCP service providing Gaode (Amap) weather and POI search.
  - `travelAssistant/backend/ticket_service.py`: Real-time 12306 train schedule & ticket price retrieval service with Gaode distance fallback.

### 4. Memory Architecture (`hello_agent/Memory/`)
- `MemoryManager.py`: Facade orchestrating four cognitive memory types:
  - `WorkingMemory.py`: Short-term sliding-window working memory.
  - `EpisodicMemory.py`: Time-stamped event history.
  - `SemanticMemory.py`: Vector-backed conceptual knowledge.
  - `PerceptualMemory.py`: Sensory/multimodal memory buffer.
- `MemoryBase.py`: Defines `MemoryItem` and `MemoryConfig`.

### 5. RAG & Storage (`hello_agent/RAG/`, `hello_agent/Storage/`)
- `RAG/pipeline.py` (`RAGPipeline`): End-to-end ingest and retrieval binding a Qdrant collection and SQLite document store.
- `RAG/document.py` (`DocumentProcessor`): Text splitting and chunking.
- `storage/qdrant_store.py`: `QdrantVectorStore` client with retry logic for vector embeddings.
- `storage/document_store.py`: `SQLiteDocumentStore` for storing full text metadata (`memory_items`).
- `storage/neo4j_store.py`: Embedding model initialization (`create_embedding_model_with_fallback`) supporting DashScope and HuggingFace/SentenceTransformers.

### 6. Context Management (`hello_agent/Context/`)
- `ContextBuilder.py`: Token-budget-aware context assembly merging system instructions, memory recalls, RAG results, and conversation history into priority-ordered packets (`ContextPacket`).

## Environment Variables (`.env`)

Critical environment variables used across the project:
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `LLM_MODEL_ID`: Default LLM configuration.
- `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION`: Qdrant vector database connection.
- `AMAP_API_KEY`: Gaode Maps API key for weather and POI queries in `Mcp/amap_mcp.py`.
- `DASHSCOPE_API_KEY`: Alibaba DashScope embeddings.
- `TAVILY_API_KEY` / `SERPAPI_API_KEY`: Web search APIs used by `Tools/SearchTool.py`.
