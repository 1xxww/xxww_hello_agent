from .Agent import Agent
from ..LLM.Myllm import MyLLM
from typing import Optional, Any,Iterator
from .AgentConfig import AgentConfig
from .Message import Message
from ..Tools import ToolRegistry
import re
import asyncio


class SimpleAgent(Agent):
    def __init__(
            self,
            name: str,
            llm: MyLLM,
            system_prompt: Optional[str] = None,
            config: Optional[AgentConfig] = None,
            tool_registry: Optional['ToolRegistry'] = None,
            enable_tool_calling: bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")


    def run (self,input_text:str,max_tool_iterations: int = 10, **kwargs) -> str:
        """
        重写的运行方法 - 实现简单对话逻辑，支持可选工具调用
        """
        self.log(f"🤖⚡ [Async] {self.name} 正在处理")

        #初始化message
        messages=[]
        final_system_prompt = self._get_enhanced_system_prompt()
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，使用简单对话逻辑
        if not self.enable_tool_calling:
            response = self.llm.chat(messages=messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            self.log(f"✅ {self.name} 响应完成")

            return response
        # 支持多轮工具调用的逻辑
        return self._run_with_tools(messages, input_text, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述（返回的是 OpenAI function schema 列表）
        schemas = self.tool_registry.get_tools_schema()
        if not schemas:
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题（**只能调用下列名字的工具**）:\n"
        for s in schemas:
            fn = s.get("function", s) if isinstance(s, dict) else {}
            name = fn.get("name", "?")
            desc = fn.get("description", "")
            params = fn.get("parameters", {}) or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required = params.get("required", []) if isinstance(params, dict) else []

            tools_section += f"- **{name}**: {desc}\n"
            for pname, pinfo in props.items():
                pinfo = pinfo if isinstance(pinfo, dict) else {}
                need = "必填" if pname in required else "可选"
                tools_section += (
                    f"    - {pname} ({pinfo.get('type', 'string')}, {need}): "
                    f"{pinfo.get('description', '')}\n"
                )

        tools_section += "\n## 工具调用格式（唯一允许的格式）\n"
        tools_section += "当需要使用工具时，必须严格使用以下格式，且一次只输出一条调用:\n"
        tools_section += "`[TOOL_CALL:工具名:{\"参数名\": \"参数值\"}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:{\"query\": \"Python编程\"}]` 或 `[TOOL_CALL:memory:{\"action\": \"recall\"}]`\n\n"
        tools_section += "严禁使用 `<tool_calls>`、`<invoke>`、`<parameter>`、`function_call` 等 XML/标签格式，"
        tools_section += "这些格式系统无法识别；参数部分必须是合法的 JSON 字符串。\n"

        return base_prompt + tools_section

    def _run_with_tools(self, messages: list, input_text: str,  **kwargs) -> str:
        """支持工具调用的运行逻辑"""
        current_iteration = 0
        final_response = ""

        while current_iteration < self.config.max_tool_iterations:
            # 调用LLM
            response = self.llm.chat(messages=messages, **kwargs)

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                self.log(f"🔧 检测到 {len(tool_calls)} 个工具调用")
                # 执行所有工具调用并收集结果
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # 从响应中移除工具调用标记
                    clean_response = clean_response.replace(call['original'], "")

                # 构建包含工具结果的消息
                messages.append({"role": "assistant", "content": clean_response})

                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                messages.append(
                    {"role": "user", "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue

            # 没有工具调用，这是最终回答
            final_response = response
            break

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= self.config.max_tool_iterations and not final_response:
            final_response = self.llm.chat(messages=messages, **kwargs)

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        self.log(f"✅ {self.name} 响应完成")

        return final_response

    async def arun(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        异步运行方法 - 完美支持并发，不阻塞主线程
        """
        self.log(f"🤖⚡ [Async] {self.name} 正在进行异步处理")
        # 初始化 message
        messages = []
        final_system_prompt = self._get_enhanced_system_prompt()
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})
        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})
        # 如果没有启用工具调用，直接异步调用LLM
        if not self.enable_tool_calling:
            response = await self.llm.achat( messages=messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            self.log(f"✅⚡ [Async] {self.name} 响应完成")
            return response
        # 支持多轮工具调用的异步逻辑
        return await self._arun_with_tools(messages, input_text,  **kwargs)

    async def _arun_with_tools(self, messages: list, input_text: str, **kwargs) -> str:
        """异步执行多工具调用的逻辑"""
        current_iteration = 0
        final_response = ""
        while current_iteration < self.config.max_tool_iterations:
            # 异步调用LLM
            response = await self.llm.achat( messages=messages, **kwargs)
            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)
            if tool_calls:
                self.log(f"🔧⚡ 检测到 {len(tool_calls)} 个工具调用，准备并发执行...")

                # 使用 asyncio.gather 并发执行所有的工具调用
                tasks = [self._aexecute_tool_call(call['tool_name'], call['parameters']) for call in tool_calls]
                tool_results = await asyncio.gather(*tasks)
                clean_response = response
                for i, call in enumerate(tool_calls):
                    # 从响应中移除工具调用标记
                    clean_response = clean_response.replace(call['original'], "")
                # 构建包含工具结果的消息
                messages.append({"role": "assistant", "content": clean_response})
                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                messages.append(
                    {"role": "user", "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"}
                )
                current_iteration += 1
                continue
            # 没有工具调用，这是最终回答
            final_response = response
            break
        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= self.config.max_tool_iterations and not final_response:
            final_response = await self.llm.achat( messages=messages, **kwargs)
        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        self.log(f"✅⚡ [Async] {self.name} 响应完成")
        return final_response

    async def _aexecute_tool_call(self, tool_name: str, parameters: str) -> str:
        """异步执行单个工具调用"""
        if not self.tool_registry:
            return f"❌ 错误:未配置工具注册表"
        try:
            param_dict = self._parse_tool_parameters(parameters)

            # 统一走注册表入口：它同时支持同步/异步工具，且在当前事件循环内执行
            # （MCP 工具的 session 绑定在启动它的事件循环上，不能用 to_thread 换线程）
            result = await self.tool_registry.execute_tool(tool_name, **param_dict)
            return f"🔧 工具 {tool_name} 执行结果:\n{result}"
        except Exception as e:
            return f"❌ 工具调用失败:{str(e)}"

    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用（兼容两种大模型输出方言）"""
        import json

        tool_calls = []

        # [TOOL_CALL:工具名:参数]
        # 放宽后行断言，支持前后有文本或换行
        pattern = r'\[TOOL_CALL:([^:]+):(\{.*?\})\]'
        for match in re.finditer(pattern, text, re.DOTALL):
            tool_name = match.group(1).strip()
            parameters = match.group(2).strip()
            tool_calls.append({
                'tool_name': tool_name,
                'parameters': parameters,
                'original': match.group(0)
            })

        if tool_calls:
            return tool_calls

        # XML 风格，部分模型会输出
        # <invoke name="工具名"><parameter name="k">v</parameter></invoke>
        for invoke in re.finditer(
            r'<invoke\s+name=["\']?([^"\'>\s]+)["\']?\s*>(.*?)</invoke>', text, re.DOTALL
        ):
            name = invoke.group(1).strip()
            body = invoke.group(2)
            params = {}
            for pname, pval in re.findall(
                r'<parameter\s+name=["\']?([^"\'>\s]+)["\']?\s*>(.*?)</parameter>', body, re.DOTALL
            ):
                params[pname.strip()] = pval.strip()
            tool_calls.append({
                'tool_name': name,
                'parameters': json.dumps(params, ensure_ascii=False),
                'original': invoke.group(0),
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """同步执行工具调用（MCP 工具建议优先走 arun 异步路径）"""
        if not self.tool_registry:
            return f"❌ 错误:未配置工具注册表"

        try:
            param_dict = self._parse_tool_parameters(parameters)
            coro = self.tool_registry.execute_tool(tool_name, **param_dict)
            try:
                asyncio.get_running_loop()
                # 已在事件循环中：同步方法无法直接 await，交给新线程跑
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    result = ex.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)

            return f"🔧 工具 {tool_name} 执行结果:\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败:{str(e)}"

    def _parse_tool_parameters(self, parameters: Any) -> dict:
        """参数解析：兼容 JSON、单引号 JSON、key=value 逗号分隔、空参数四种情况"""
        import json

        if isinstance(parameters, dict):
            return parameters

        raw = (parameters or "").strip()
        if not raw:
            return {}

        # 去掉可能的代码块包裹
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        # 1) 标准 JSON
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # 2) 单引号 JSON（大模型常见输出）
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # 3) key=value,key=value 格式（提示词里的格式）
        result = {}
        for pair in re.split(r"[,，;；]", raw):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            result[key.strip().strip("\"'")] = value.strip().strip("\"'")

        if result:
            return result

        # 都不匹配 → 抛出带原文的错误，便于排查
        raise ValueError(f"无法解析工具参数: {raw!r}")

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        自定义的流式运行方法
        """
        self.log(f"🌊 {self.name} 开始流式输出")

        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # 流式调用LLM
        full_response = ""
        print("📝 实时响应: ", end="")
        for chunk in self.llm.chat_stream(messages=messages, **kwargs):
            full_response += chunk
            print(chunk, end="", flush=True)
            yield chunk
        print()  # 换行

        # 保存完整对话到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        self.log(f"✅ {self.name} 流式响应完成")



