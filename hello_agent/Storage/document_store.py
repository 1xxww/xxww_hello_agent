import aiosqlite
import json
from typing import Optional
from ..Memory import MemoryItem

class SQLiteDocumentStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self._conn = await aiosqlite.connect(self.db_path)
        await self._init_table()

    async def _init_table(self):
        create_sql = """
        CREATE TABLE IF NOT EXISTS memory_items (
            item_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expire_at TEXT,
            metadata TEXT,
            embedding BLOB NULL
        )
        """
        await self._conn.execute(create_sql)
        await self._conn.commit()

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def put(self, item: MemoryItem):
        meta_json = json.dumps(item.metadata, ensure_ascii=False)
        expire_str = item.expire_at.isoformat() if item.expire_at else None
        created_str = item.created_at.isoformat()
        # embedding向量这里可选不入库，向量存在Qdrant
        sql = """
        INSERT OR REPLACE INTO memory_items
        (item_id, content, memory_type, created_at, expire_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        await self._conn.execute(sql, (
            item.item_id,
            item.content,
            item.memory_type,
            created_str,
            expire_str,
            meta_json
        ))
        await self._conn.commit()

    async def get(self, item_id: str) -> Optional[MemoryItem]:
        sql = "SELECT * FROM memory_items WHERE item_id = ?"
        cursor = await self._conn.execute(sql, (item_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    async def delete(self, item_id: str):
        sql = "DELETE FROM memory_items WHERE item_id = ?"
        await self._conn.execute(sql, (item_id,))
        await self._conn.commit()

    async def clear_by_type(self, memory_type: str):
        sql = "DELETE FROM memory_items WHERE memory_type = ?"
        await self._conn.execute(sql, (memory_type,))
        await self._conn.commit()

    def _row_to_item(self, row) -> MemoryItem:
        from datetime import datetime
        # 表有 7 列（含 embedding BLOB），embedding 向量存 Qdrant 不入库
        item_id, content, mtype, created_str, expire_str, meta_str, _ = row
        meta = json.loads(meta_str)
        created_at = datetime.fromisoformat(created_str)
        expire_at = datetime.fromisoformat(expire_str) if expire_str else None
        return MemoryItem(
            item_id=item_id,
            content=content,
            memory_type=mtype,
            created_at=created_at,
            expire_at=expire_at,
            metadata=meta
        )