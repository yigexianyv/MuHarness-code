"""运行原文记录：运行中途和中断后都能查到模型实际看过的原始消息。

继承的会话历史只登记引用（会话 ID、条数、哈希），不复制；本次运行新增的消息
按产生顺序逐条追加。完整序号 = 继承条数 + 本次序号，与摘要覆盖水位使用同一套编号。
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH
from app.models.types import Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_history_refs (
    run_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    inherited_count INTEGER NOT NULL CHECK (inherited_count >= 0),
    inherited_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_messages (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL CHECK (idx >= 0),
    message_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, idx)
);
"""


def history_sha256(messages: Sequence[Message]) -> str:
    """继承历史的指纹；读取时重新计算，对不上说明会话历史已被改写。"""
    digest = hashlib.sha256()
    for message in messages:
        digest.update(message.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RunHistoryRef:
    run_id: str
    conversation_id: str | None
    inherited_count: int
    inherited_sha256: str


class SQLiteRunMessageStore:

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    async def begin(
        self,
        run_id: str,
        *,
        conversation_id: str | None,
        history: Sequence[Message],
    ) -> None:
        async with self._connect() as database:
            await database.execute(
                """
                INSERT OR IGNORE INTO run_history_refs (
                    run_id, conversation_id, inherited_count, inherited_sha256,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    conversation_id,
                    len(history),
                    history_sha256(history),
                    _now_iso(),
                ),
            )
            await database.commit()

    async def append(
        self,
        run_id: str,
        messages: Sequence[Message],
        *,
        start_index: int,
    ) -> None:
        if not messages:
            return
        now = _now_iso()
        async with self._connect() as database:
            await database.executemany(
                """
                INSERT OR IGNORE INTO run_messages (
                    run_id, idx, message_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (run_id, index, message.model_dump_json(), now)
                    for index, message in enumerate(messages, start=start_index)
                ],
            )
            await database.commit()

    async def history_ref(self, run_id: str) -> RunHistoryRef | None:
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT run_id, conversation_id, inherited_count, inherited_sha256
                FROM run_history_refs WHERE run_id = ?
                """,
                (run_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return RunHistoryRef(
            run_id=row["run_id"],
            conversation_id=row["conversation_id"],
            inherited_count=row["inherited_count"],
            inherited_sha256=row["inherited_sha256"],
        )

    async def count(self, run_id: str) -> int:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT COUNT(*) AS total FROM run_messages WHERE run_id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()
        return int(row["total"]) if row is not None else 0

    async def load(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[tuple[int, Message], ...]:
        """按本次运行内的序号读取新增消息。"""
        query = (
            "SELECT idx, message_json FROM run_messages WHERE run_id = ? AND idx >= ?"
            " ORDER BY idx ASC"
        )
        params: tuple[object, ...] = (run_id, offset)
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        async with self._connect() as database:
            cursor = await database.execute(query, params)
            rows = await cursor.fetchall()
        return tuple(
            (row["idx"], Message.model_validate_json(row["message_json"]))
            for row in rows
        )

    async def start_recording(
        self,
        run_id: str,
        *,
        conversation_id: str | None,
        history: Sequence[Message],
    ) -> RunMessageRecorder:
        await self.begin(run_id, conversation_id=conversation_id, history=history)
        return RunMessageRecorder(self, run_id, inherited_count=len(history))

    async def delete_runs(self, run_ids: Sequence[str]) -> None:
        if not run_ids:
            return
        async with self._connect() as database:
            for table in ("run_messages", "run_history_refs"):
                await database.executemany(
                    f"DELETE FROM {table} WHERE run_id = ?",
                    [(run_id,) for run_id in run_ids],
                )
            await database.commit()

    async def delete_for_conversation(
        self,
        conversation_id: str,
        *,
        run_ids: Sequence[str] = (),
    ) -> None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT run_id FROM run_history_refs WHERE conversation_id = ?",
                (conversation_id,),
            )
            linked = {row["run_id"] for row in await cursor.fetchall()}
        await self.delete_runs(sorted(linked | set(run_ids)))

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        try:
            yield database
        finally:
            await database.close()


class RunMessageRecorder:
    """一次运行的追加记录器：只写尚未记录的稳定消息，记录失败不影响运行。"""

    def __init__(
        self,
        store: SQLiteRunMessageStore,
        run_id: str,
        *,
        inherited_count: int,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._inherited_count = inherited_count
        self._recorded = 0

    async def sync(self, messages: Sequence[Message]) -> None:
        """messages 是包含继承历史在内的完整消息序列。"""
        own = messages[self._inherited_count :]
        pending = own[self._recorded :]
        if not pending:
            return
        try:
            await self._store.append(
                self._run_id,
                pending,
                start_index=self._recorded,
            )
        except Exception:
            return
        self._recorded += len(pending)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "RunHistoryRef",
    "RunMessageRecorder",
    "SQLiteRunMessageStore",
    "history_sha256",
]
