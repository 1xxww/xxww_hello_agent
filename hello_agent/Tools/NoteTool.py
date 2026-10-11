from .Tool import Tool
from typing import Any, Dict, List, Optional,Tuple
import os
import json
from .ToolParameter import ToolParameter

class NoteTool(Tool):
    def __init__(self, workspace: str):
        super().__init__(
            name="NoteTool",
            description="这是一个工作记忆与笔记工具，用于记录复杂任务拆解、中间结论、数据锁定及遗留阻塞点。")
        self.workspace = workspace
        self.index_file = os.path.join(self.workspace, "note_index.json")
        self.index: Dict[str, Dict] = {}

        # 初始化目录
        os.makedirs(self.workspace, exist_ok=True)
        # 加载本地索引
        self._load_index()

    def _load_index(self):
        """从磁盘加载笔记索引"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf‑8") as f:
                    self.index = json.load(f)
            except Exception:
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        """保存索引到磁盘json"""
        with open(self.index_file, "w", encoding="utf‑8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _create_note(
            self,
            title: str,
            content: str,
            note_type: str = "general",
            tags: Optional[List[str]] = None
    ) -> str:
        """创建笔记

        Args:
            title: 笔记标题
            content: 笔记内容(Markdown格式)
            note_type: 笔记类型(task_state/conclusion/blocker/action/reference/general)
            tags: 标签列表

        Returns:
            str: 笔记ID
        """
        from datetime import datetime

        # 1. 生成唯一ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        note_id = f"note_{timestamp}_{len(self.index)}"

        # 2. 构建元数据
        metadata = {
            "id": note_id,
            "title": title,
            "type": note_type,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # 3. 构建完整的 Markdown 文件内容
        md_content = self._build_markdown(metadata, content)

        # 4. 保存到文件
        file_path = os.path.join(self.workspace, f"{note_id}.md")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        # 5. 更新索引
        metadata["file_path"] = file_path
        self.index[note_id] = metadata
        self._save_index()

        return note_id

    def _build_markdown(self, metadata: Dict, content: str) -> str:
        """构建 Markdown 文件内容(YAML + 正文)"""
        import yaml

        # YAML 前置元数据
        yaml_header = yaml.dump(metadata, allow_unicode=True, sort_keys=False)

        # 组合格式
        return f"---\n{yaml_header}---\n\n{content}"

    def _read_note(self, note_id: str) -> Dict:
        """读取笔记内容

        Args:
            note_id: 笔记ID

        Returns:
            Dict: 包含元数据和内容的字典
        """
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")

        file_path = self.index[note_id]["file_path"]

        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # 解析 YAML 元数据和 Markdown 正文
        metadata, content = self._parse_markdown(raw_content)

        return {
            "metadata": metadata,
            "content": content
        }

    def _parse_markdown(self, raw_content: str) -> Tuple[Dict, str]:
        """解析 Markdown 文件(分离 YAML 和正文)"""
        import yaml

        # 查找 YAML 分隔符
        parts = raw_content.split('---\n', 2)

        if len(parts) >= 3:
            # 有 YAML 前置元数据
            yaml_str = parts[1]
            content = parts[2].strip()
            metadata = yaml.safe_load(yaml_str)
        else:
            # 无元数据,全部作为正文
            metadata = {}
            content = raw_content.strip()

        return metadata, content

    def _update_note(
            self,
            note_id: str,
            title: Optional[str] = None,
            content: Optional[str] = None,
            note_type: Optional[str] = None,
            tags: Optional[List[str]] = None
    ) -> str:
        """更新笔记

        Args:
            note_id: 笔记ID
            title: 新标题(可选)
            content: 新内容(可选)
            note_type: 新类型(可选)
            tags: 新标签(可选)

        Returns:
            str: 操作结果消息
        """
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")

        # 1. 读取现有笔记
        note = self._read_note(note_id)
        metadata = note["metadata"]
        old_content = note["content"]

        # 2. 更新字段
        if title:
            metadata["title"] = title
        if note_type:
            metadata["type"] = note_type
        if tags is not None:
            metadata["tags"] = tags
        if content is not None:
            old_content = content

        # 更新时间戳
        from datetime import datetime
        metadata["updated_at"] = datetime.now().isoformat()

        # 3. 重新构建并保存
        md_content = self._build_markdown(metadata, old_content)
        file_path = metadata["file_path"]

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        # 4. 更新索引
        self.index[note_id] = metadata
        self._save_index()

        return f"✅ 笔记已更新: {metadata['title']}"

    def _search_notes(
            self,
            query: str,
            limit: int = 10,
            note_type: Optional[str] = None,
            tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """搜索笔记

        Args:
            query: 搜索关键词
            limit: 返回数量限制
            note_type: 按类型过滤(可选)
            tags: 按标签过滤(可选)

        Returns:
            List[Dict]: 匹配的笔记列表
        """
        results = []
        query_lower = query.lower()

        for note_id, metadata in self.index.items():
            # 类型过滤
            if note_type and metadata.get("type") != note_type:
                continue

            # 标签过滤
            if tags:
                note_tags = set(metadata.get("tags", []))
                if not note_tags.intersection(tags):
                    continue

            # 读取笔记内容
            try:
                note = self._read_note(note_id)
                content = note["content"]
                title = metadata.get("title", "")

                # 在标题和内容中搜索
                if query_lower in title.lower() or query_lower in content.lower():
                    results.append({
                        "note_id": note_id,
                        "title": title,
                        "type": metadata.get("type"),
                        "tags": metadata.get("tags", []),
                        "content": content,
                        "updated_at": metadata.get("updated_at")
                    })
            except Exception as e:
                print(f"[WARNING] 读取笔记 {note_id} 失败: {e}")
                continue

        # 按更新时间排序
        results.sort(key=lambda x: x["updated_at"], reverse=True)

        return results[:limit]

    def _list_notes(
            self,
            note_type: Optional[str] = None,
            tags: Optional[List[str]] = None,
            limit: int = 20
    ) -> List[Dict]:
        """列出笔记(按更新时间倒序)

        Args:
            note_type: 按类型过滤(可选)
            tags: 按标签过滤(可选)
            limit: 返回数量限制

        Returns:
            List[Dict]: 笔记元数据列表
        """
        results = []

        for note_id, metadata in self.index.items():
            # 类型过滤
            if note_type and metadata.get("type") != note_type:
                continue

            # 标签过滤
            if tags:
                note_tags = set(metadata.get("tags", []))
                if not note_tags.intersection(tags):
                    continue

            results.append(metadata)

        # 按更新时间排序
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return results[:limit]

    def _summary(self) -> Dict[str, Any]:
        """生成笔记摘要统计

        Returns:
            Dict: 统计信息
        """
        total_count = len(self.index)

        # 按类型统计
        type_counts = {}
        for metadata in self.index.values():
            note_type = metadata.get("type", "general")
            type_counts[note_type] = type_counts.get(note_type, 0) + 1

        # 最近更新的笔记
        recent_notes = sorted(
            self.index.values(),
            key=lambda x: x.get("updated_at", ""),
            reverse=True
        )[:5]

        return {
            "total_notes": total_count,
            "type_distribution": type_counts,
            "recent_notes": [
                {
                    "id": note["id"],
                    "title": note.get("title", ""),
                    "type": note.get("type"),
                    "updated_at": note.get("updated_at")
                }
                for note in recent_notes
            ]
        }

    def _delete_note(self, note_id: str) -> str:
        """删除笔记

        Args:
            note_id: 笔记ID

        Returns:
            str: 操作结果消息
        """
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")

        # 1. 删除文件
        file_path = self.index[note_id]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)

        # 2. 从索引中移除
        title = self.index[note_id].get("title", note_id)
        del self.index[note_id]
        self._save_index()

        return f"✅ 笔记已删除: {title}"

    def execute(self, **kwargs) -> Any:
        """
        function‑call入口
        parameters: {
            "action": "create_note",
            "title":"xxx",
            "content":"xxx",
            ...
        }
        """
        action = kwargs.get("action", "")
        try:
            if action == "create_note":
                note_id = self._create_note(
                    title=kwargs["title"],
                    content=kwargs["content"],
                    note_type=kwargs.get("note_type", "general"),
                    tags=kwargs.get("tags")
                )
                return f"✅ 创建笔记成功，note_id:{note_id}"
            elif action == "read_note":
                note = self._read_note(kwargs["note_id"])
                return (f"📖 笔记 [{kwargs['note_id']}] 内容如下:\n"
                        f"标题: {note['metadata'].get('title')}\n"
                        f"类型: {note['metadata'].get('type')}\n"
                        f"正文:\n{note['content']}")

            elif action == "update_note":
                return self._update_note(
                    note_id=kwargs["note_id"],
                    title=kwargs.get("title"),
                    content=kwargs.get("content"),
                    note_type=kwargs.get("note_type"),
                    tags=kwargs.get("tags")
                )

            elif action == "delete_note":
                return self._delete_note(kwargs["note_id"])

            elif action == "search_notes":
                res = self._search_notes(
                    query=kwargs["query"],
                    limit=kwargs.get("limit", 10),
                    note_type=kwargs.get("note_type"),
                    tags=kwargs.get("tags")
                )
                if not res:
                    return "🔍没有找到匹配的笔记"
                lines = ["🔍搜索结果:"]
                for idx, item in enumerate(res, 1):
                    lines.append(f"{idx}. [{item['note_id']}] {item['title']} 类型:{item['type']}")
                return "\n".join(lines)

            elif action == "list_notes":
                res = self._list_notes(
                    note_type=kwargs.get("note_type"),
                    tags=kwargs.get("tags"),
                    limit=kwargs.get("limit", 20)
                )
                if not res:
                    return "📄暂无笔记"
                lines = ["📄笔记列表:"]
                for idx, item in enumerate(res, 1):
                    lines.append(f"{idx}. {item['id']} | {item['title']} | type:{item['type']}")
                return "\n".join(lines)

            elif action == "summary":
                s = self._summary()
                text = f"""📊笔记统计
                        总笔记数:{s['total_notes']}
                        类型分布:{s['type_distribution']}
                        最近5条:"""
                for r in s["recent_notes"]:
                    text += f"\n‑ {r['id']} {r['title']}"
                return text
            else:
                return f"❌未知action:{action},支持动作:create_note/read_note/update_note/delete_note/search_notes/list_notes/summary"
        except Exception as e:
            return f"❌NoteTool执行异常:{str(e)}"

    def get_parameters(self) -> List[ToolParameter]:
        """定义function call参数"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型，可选：create_note / read_note / update_note / delete_note / search_notes / list_notes / summary",
                required=True
            ),
            ToolParameter(
                name="note_id",
                type="string",
                description="笔记ID，read_note、update、delete时需要",
                required=False
            ),
            ToolParameter(
                name="title",
                type="string",
                description="笔记标题，创建/更新笔记使用",
                required=False
            ),
            ToolParameter(
                name="content",
                type="string",
                description="笔记Markdown内容，创建/更新笔记使用",
                required=False
            ),
            ToolParameter(
                name="note_type",
                type="string",
                description="笔记类型:task_state/conclusion/blocker/action/reference/general",
                required=False
            ),
            ToolParameter(
                name="tags",
                type="array",
                description="标签列表，字符串数组",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索关键词，search_notes使用",
                required=False
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回结果数量上限",
                required=False
            )
        ]