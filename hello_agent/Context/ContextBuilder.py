from typing import List, Optional
from datetime import datetime
from .ContextPacket import ContextConfig,ContextPacket
from ..Agent import Message
from ..Tools import RAGTool,MemoryTool


class ContextBuilder:
    def __init__(self,
                 config: ContextConfig,
                 memory_tool: Optional[MemoryTool] = None,
                 rag_tool: Optional[RAGTool] = None,):

        self.config: ContextConfig = config
        self.rag_tool: RAGTool = rag_tool
        self.memory_tool: MemoryTool = memory_tool

    def build(self,
              user_query: str,
              conversation_history: Optional[List[Message]] = None,
              system_instructions: Optional[str] = None,
              custom_packets: Optional[List[ContextPacket]] = None
              ) -> List[ContextPacket]:
        """
        对外入口：构建上下文信息包列表
        :param user_query: 用户问题
        :param conversation_history: 对话历史消息列表
        :param system_instructions: 系统提示词
        :param custom_packets: 自定义上下文包
        :return: 经过token预算筛选后的 ContextPacket 列表
        """
        # 1.汇集全部候选数据包（系统指令+记忆检索+RAG检索+对话历史+自定义包）
        candidate_packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history,
            system_instructions=system_instructions,
            custom_packets=custom_packets
        )

        # 2.从config读取可用token预算
        available_tokens = self.config.max_tokens

        # 3.执行筛选：按综合分数(相关性+近因性)贪心选出不超token预算的包
        selected_packets = self._select(
            packets=candidate_packets,
            user_query=user_query,
            available_tokens=available_tokens
        )

        return selected_packets

    def _gather(
            self,
            user_query: str,
            conversation_history: Optional[List[Message]] = None,
            system_instructions: Optional[str] = None,
            custom_packets: Optional[List[ContextPacket]] = None
    ) -> List[ContextPacket]:
        """汇集所有候选信息

        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            system_instructions: 系统指令
            custom_packets: 自定义信息包

        Returns:
            List[ContextPacket]: 候选信息列表
        """
        packets = []

        # 1. 添加系统指令(最高优先级,不参与评分)
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                timestamp=datetime.now(),
                token_count=self._count_tokens(system_instructions),
                relevance_score=1.0,  # 系统指令始终保留
                metadata={"type": "system_instruction", "priority": "high"}
            ))

        # 2. 从记忆系统检索相关记忆
        if self.memory_tool:
            try:
                memory_results = self.memory_tool.execute({
                    "action": "search_raw",
                    "query": user_query,
                    "limit": 10,
                    "min_importance": 0.3
                })
                # 解析记忆结果并转换为 ContextPacket
                memory_packets = self._parse_memory_results(memory_results, user_query)
                packets.extend(memory_packets)
            except Exception as e:
                print(f"[WARNING] 记忆检索失败: {e}")

        # 3. 从 RAG 系统检索相关知识
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.execute({
                    "action": "search_raw",
                    "query": user_query,
                    "limit": 5,
                    "min_score": 0.3
                })
                # 解析 RAG 结果并转换为 ContextPacket
                rag_packets = self._parse_rag_results(rag_results, user_query)
                packets.extend(rag_packets)
            except Exception as e:
                print(f"[WARNING] RAG 检索失败: {e}")

        # 4. 添加对话历史(仅保留最近的 N 条)
        if conversation_history:
            recent_history = conversation_history[-5:]  # 默认保留最近 5 条
            for msg in recent_history:
                packets.append(ContextPacket(
                    content=f"{msg.role}: {msg.content}",
                    timestamp=msg.timestamp if hasattr(msg, 'timestamp') else datetime.now(),
                    token_count=self._count_tokens(msg.content),
                    relevance_score=0.6,  # 历史消息的基础相关性
                    metadata={"type": "conversation_history", "role": msg.role}
                ))

        # 5. 添加自定义信息包
        if custom_packets:
            packets.extend(custom_packets)

        print(f"[ContextBuilder] 汇集了 {len(packets)} 个候选信息包")
        return packets

    def _parse_memory_results(self, memory_results) -> List[ContextPacket]:
        """
        将MemoryTool记忆检索输出转为ContextPacket
        memory_results: MemoryTool.execute({"action":"search"})返回的结果列表
        """
        packet_list: List[ContextPacket] = []

        # 遍历每一条记忆记录
        for mem_item in memory_results:
            # 提取字段，做容错保护，防止key不存在报错
            content = mem_item.get("content", "")
            score = mem_item.get("score", 0.5)  # 记忆召回相似度分数
            ts = mem_item.get("timestamp", datetime.now())
            meta = mem_item.get("metadata", {})
            item_id = mem_item.get("item_id", "")

            packet = ContextPacket(
                content=content,
                timestamp=ts,
                token_count=self._count_tokens(content),
                relevance_score=score,
                metadata={
                    "type": "memory_result",
                    "item_id": item_id,
                    **meta
                }
            )
            packet_list.append(packet)

        return packet_list

    def _parse_rag_results(self, rag_results) -> List[ContextPacket]:
        """
        将RAGTool检索输出转为ContextPacket
        rag_results：RAGTool.execute({"action":"search"})返回的chunk列表
        """
        packet_list: List[ContextPacket] = []

        for chunk in rag_results:
            content = chunk.get("content", "")
            score = chunk.get("score", 0.5)
            source_path = chunk.get("source_path", "")
            chunk_id = chunk.get("chunk_id", "")
            meta = chunk.get("metadata", {})

            packet = ContextPacket(
                content=content,
                timestamp=datetime.now(),  # RAG文档块没有真实访问时间，使用当前时间
                token_count=self._count_tokens(content),
                relevance_score=score,
                metadata={
                    "type": "rag_result",
                    "chunk_id": chunk_id,
                    "source_path": source_path,
                    **meta
                }
            )
            packet_list.append(packet)

        return packet_list


    def _select(
            self,
            packets: List[ContextPacket],
            user_query: str,
            available_tokens: int
    ) -> List[ContextPacket]:
        """选择最相关的信息包

        Args:
            packets: 候选信息包列表
            user_query: 用户查询(用于计算相关性)
            available_tokens: 可用的 token 数量

        Returns:
            List[ContextPacket]: 选中的信息包列表
        """
        # 1. 分离系统指令和其他信息
        system_packets = [p for p in packets if p.metadata.get("type") == "system_instruction"]
        other_packets = [p for p in packets if p.metadata.get("type") != "system_instruction"]

        # 2. 计算系统指令占用的 token
        system_tokens = sum(p.token_count for p in system_packets)
        remaining_tokens = available_tokens - system_tokens

        if remaining_tokens <= 0:
            print("[WARNING] 系统指令已占满所有 token 预算")
            return system_packets

        # 3. 为其他信息计算综合分数
        scored_packets = []
        for packet in other_packets:
            # 计算相关性分数(如果尚未计算)
            if packet.relevance_score == 0.5:  # 默认值,需要重新计算
                relevance = self._calculate_relevance(packet.content, user_query)
                packet.relevance_score = relevance

            # 计算新近性分数
            recency = self._calculate_recency(packet.timestamp)

            # 综合分数 = 相关性权重 × 相关性 + 新近性权重 × 新近性
            combined_score = (
                    self.config.relevance_weight * packet.relevance_score +
                    self.config.recency_weight * recency
            )

            # 过滤低于最小相关性阈值的信息
            if packet.relevance_score >= self.config.min_relevance:
                scored_packets.append((combined_score, packet))

        # 4. 按分数降序排序
        scored_packets.sort(key=lambda x: x[0], reverse=True)

        # 5. 贪心选择:按分数从高到低填充,直到达到 token 上限
        selected = system_packets.copy()
        current_tokens = system_tokens

        for score, packet in scored_packets:
            if current_tokens + packet.token_count <= available_tokens:
                selected.append(packet)
                current_tokens += packet.token_count
            else:
                # Token 预算已满,停止选择
                break

        print(f"[ContextBuilder] 选择了 {len(selected)} 个信息包,共 {current_tokens} tokens")
        return selected

    def _calculate_relevance(self, content: str, query: str) -> float:
        """计算内容与查询的相关性

        使用简单的关键词重叠算法。在生产环境中,可以替换为向量相似度计算。

        Args:
            content: 内容文本
            query: 查询文本

        Returns:
            float: 相关性分数(0.0-1.0)
        """
        # 分词(简单实现,可以使用更复杂的分词器)
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.0

        # Jaccard 相似度
        intersection = content_words & query_words
        union = content_words | query_words

        return len(intersection) / len(union) if union else 0.0

    def _calculate_recency(self, timestamp: datetime) -> float:
        """计算时间近因性分数

        使用指数衰减模型,24小时内保持高分,之后逐渐衰减。

        Args:
            timestamp: 信息的时间戳

        Returns:
            float: 新近性分数(0.0-1.0)
        """
        import math

        age_hours = (datetime.now() - timestamp).total_seconds() / 3600

        # 指数衰减:24小时内保持高分,之后逐渐衰减
        decay_factor = 0.1  # 衰减系数
        recency_score = math.exp(-decay_factor * age_hours / 24)

        return max(0.1, min(1.0, recency_score))  # 限制在 [0.1, 1.0] 范围内

    def _structure(self, selected_packets: List[ContextPacket], user_query: str) -> str:
        """将选中的信息包组织成结构化的上下文模板

        Args:
            selected_packets: 选中的信息包列表
            user_query: 用户查询

        Returns:
            str: 结构化的上下文字符串
        """
        # 按类型分组
        system_instructions = []
        evidence = []
        context = []

        for packet in selected_packets:
            packet_type = packet.metadata.get("type", "general")

            if packet_type == "system_instruction":
                system_instructions.append(packet.content)
            elif packet_type in ["rag_result", "knowledge"]:
                evidence.append(packet.content)
            else:
                context.append(packet.content)

        # 构建结构化模板
        sections = []

        # [Role & Policies]
        if system_instructions:
            sections.append("[Role & Policies]\n" + "\n".join(system_instructions))

        # [Task]
        sections.append(f"[Task]\n{user_query}")

        # [Evidence]
        if evidence:
            sections.append("[Evidence]\n" + "\n---\n".join(evidence))

        # [Context]
        if context:
            sections.append("[Context]\n" + "\n".join(context))

        # [Output]
        sections.append("[Output]\n请基于以上信息,提供准确、有据的回答。")

        return "\n\n".join(sections)

    def _compress(self, context: str, max_tokens: int) -> str:
        """压缩超限的上下文

        Args:
            context: 原始上下文
            max_tokens: 最大 token 限制

        Returns:
            str: 压缩后的上下文
        """
        current_tokens = self._count_tokens(context)

        if current_tokens <= max_tokens:
            return context  # 无需压缩

        print(f"[ContextBuilder] 上下文超限({current_tokens} > {max_tokens}),执行压缩")

        # 分区压缩:保持结构完整性
        sections = context.split("\n\n")
        compressed_sections = []
        current_total = 0

        for section in sections:
            section_tokens = self._count_tokens(section)

            if current_total + section_tokens <= max_tokens:
                # 完整保留
                compressed_sections.append(section)
                current_total += section_tokens
            else:
                # 部分保留
                remaining_tokens = max_tokens - current_total
                if remaining_tokens > 50:  # 至少保留 50 tokens
                    # 简单截断(生产环境中可以使用 LLM 摘要)
                    truncated = self._truncate_text(section, remaining_tokens)
                    compressed_sections.append(truncated + "\n[... 内容已压缩 ...]")
                break

        compressed_context = "\n\n".join(compressed_sections)
        final_tokens = self._count_tokens(compressed_context)
        print(f"[ContextBuilder] 压缩完成: {current_tokens} -> {final_tokens} tokens")

        return compressed_context

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数量

        Args:
            text: 原始文本
            max_tokens: 最大 token 数量

        Returns:
            str: 截断后的文本
        """
        # 简单实现:按字符比例估算
        # 生产环境中应该使用精确的 tokenizer
        char_per_token = len(text) / self._count_tokens(text) if self._count_tokens(text) > 0 else 4
        max_chars = int(max_tokens * char_per_token)

        return text[:max_chars]

    def _count_tokens(self, text: str) -> int:
        """估算文本的 token 数量

        Args:
            text: 文本内容

        Returns:
            int: token 数量
        """
        # 简单估算:中文 1 字符 ≈ 1 token,英文 1 单词 ≈ 1.3 tokens
        # 生产环境中应该使用实际的 tokenizer
        chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        english_words = len([w for w in text.split() if w])

        return int(chinese_chars + english_words * 1.3)
