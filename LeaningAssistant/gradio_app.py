# -*- coding: utf-8 -*-
"""学习助手 - Gradio Web 应用

启动方式（在项目根目录）：
    python gradio_app.py
浏览器访问：http://127.0.0.1:7860

前置条件：
    1. pip install gradio
    2. Qdrant 服务已启动（RAG 检索依赖，docker run -p 6333:6333 qdrant/qdrant）
    3. .env 里有 DEEPSEEK_API_KEY（LLM 问答）与 QDRANT_URL/QDRANT_API_KEY
"""
import json
import os
import sys

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文/emoji 打印报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 嵌入模型缓存到项目内（首次运行会自动下载 all-MiniLM-L6-v2）
# 国内网络慢可取消下行注释，走 HuggingFace 镜像
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.hf_cache"))
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr

from LeaningAssistant import LeaningAssistant

# 全局单例助手（演示用；多用户场景可按 user_id 建立字典缓存）
assistant = LeaningAssistant(user_id="gradio_demo")

# ==================== 事件处理函数 ====================

def ask_question(message: str, history):
    """对话问答：走 RAG 检索 + LLM 回答"""
    try:
        answer = assistant.ask(message)
    except Exception as e:
        answer = f"❌ 出错了: {e}"
    history = history or []
    history.append((message, answer))
    return history, ""

def upload_document(file) -> str:
    """上传并入库文档"""
    if file is None:
        return "❌ 请先选择文件"
    try:
        result = assistant.load_document(file.name)
    except Exception as e:
        return f"❌ 加载失败: {e}"
    return result.get("message", str(result))

def add_note(content: str) -> str:
    """添加一条学习笔记到记忆"""
    if not content or not content.strip():
        return "❌ 笔记内容不能为空"
    try:
        return str(assistant.add_note(content))
    except Exception as e:
        return f"❌ 出错了: {e}"

def recall_memory(query: str) -> str:
    """从记忆里检索相关内容"""
    if not query or not query.strip():
        return "❌ 请输入检索关键词"
    try:
        return str(assistant.recall(query))
    except Exception as e:
        return f"❌ 出错了: {e}"

def show_stats() -> str:
    """当前会话统计"""
    stats = assistant.get_stats()
    return "\n".join(f"{k}: {v}" for k, v in stats.items())

def show_report() -> str:
    """生成学习报告"""
    try:
        report = assistant.generate_report()
    except Exception as e:
        return (f"❌ 生成报告失败: {e}\n"
                "💡 提示：generate_report 依赖 memory_tool.execute('summary')，"
                "而 MemoryTool 尚未实现 _summary_memory，可作为练习补上。")
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)

# ==================== 界面布局 ====================

with gr.Blocks(title="📚 学习助手") as demo:
    gr.Markdown("""
    # 📚 学习助手
    基于 **Memory（记忆） + RAG（检索增强）** 的学习工具：上传文档入库 → 基于文档提问 → 记录学习笔记 → 回忆检索 → 生成学习报告。
    """)

    with gr.Row():
        stats_box = gr.Textbox(label="📊 会话统计", value=show_stats(), lines=4, interactive=False)
        gr.Button("🔄 刷新统计").click(show_stats, outputs=stats_box)

    with gr.Tabs():
        # ---------- Tab 1: 对话问答 ----------
        with gr.Tab("💬 对话问答"):
            chatbot = gr.Chatbot(label="问答记录", height=400)
            with gr.Row():
                question_box = gr.Textbox(label="你的问题", placeholder="例如：这份文档主要讲了什么？", scale=5)
                send_btn = gr.Button("发送", variant="primary", scale=1)
            send_btn.click(ask_question, inputs=[question_box, chatbot], outputs=[chatbot, question_box])

        # ---------- Tab 2: 上传文档 ----------
        with gr.Tab("📄 上传文档"):
            gr.Markdown("支持 PDF / Word / 文本 / Markdown 等格式，上传后自动切块并向量化入库。")
            file_input = gr.File(label="选择文档", file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".pptx", ".csv"])
            upload_btn = gr.Button("🚀 开始入库", variant="primary")
            upload_result = gr.Textbox(label="入库结果", interactive=False)
            upload_btn.click(upload_document, inputs=file_input, outputs=upload_result)

        # ---------- Tab 3: 学习笔记 ----------
        with gr.Tab("📝 学习笔记"):
            note_input = gr.Textbox(label="笔记内容", placeholder="记录一条知识点或心得…", lines=4)
            note_btn = gr.Button("💾 保存笔记", variant="primary")
            note_result = gr.Textbox(label="保存结果", interactive=False)
            note_btn.click(add_note, inputs=note_input, outputs=note_result)

        # ---------- Tab 4: 回忆检索 ----------
        with gr.Tab("🔍 回忆"):
            recall_input = gr.Textbox(label="回忆关键词", placeholder="例如：昨天学过的概念")
            recall_btn = gr.Button("🔎 回忆", variant="primary")
            recall_result = gr.Textbox(label="回忆结果", lines=8, interactive=False)
            recall_btn.click(recall_memory, inputs=recall_input, outputs=recall_result)

        # ---------- Tab 5: 学习报告 ----------
        with gr.Tab("📋 学习报告"):
            report_btn = gr.Button("📑 生成报告", variant="primary")
            report_box = gr.Textbox(label="报告内容", lines=12, interactive=False)
            report_btn.click(show_report, outputs=report_box)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, theme=gr.themes.Soft())