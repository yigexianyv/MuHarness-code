
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH

from .summary import ConversationSummaryState, RollingConversationSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id TEXT PRIMARY KEY,
    summary_json TEXT NOT NULL,
    covered_message_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""


class SQLiteConversationSummaryStore:

    # 函数说明：SQLiteConversationSummaryStore.__init__
    # 用途：初始化 SQLiteConversationSummaryStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   database_path：SQLite 数据库路径，类型 `str | Path`；默认
    # `DEFAULT_DATABASE_PATH`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(database_path).expanduser().resolve` → `Path(database_path).expanduser` →
    # `Path`。
    # 副作用与资源：
    #   更新对象字段：`self.database_path`。
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    # 函数说明：SQLiteConversationSummaryStore.initialize
    # 用途：初始化SQLiteConversationSummaryStore，供模型上下文与输入预算使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.database_path.parent.mkdir`
    # → `self._connect` → `database.executescript` → `database.commit`。
    # 副作用与资源：
    #   文件或资源访问：`self.database_path.parent.mkdir`。
    async def initialize(self) -> None:

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    # 函数说明：SQLiteConversationSummaryStore.load
    # 用途：加载SQLiteConversationSummaryStore，供模型上下文与输入预算使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `ConversationSummaryState | None`；按分支返回 `None`；
    # `ConversationSummaryState(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchone` → `ConversationSummaryState` →
    # `RollingConversationSummary.model_validate_json`。
    # 分支与异常：
    #   当 `row is None` 时，返回 `None`。
    # 副作用与资源：
    #   数据库操作：SELECT conversation_summaries；连接与事务边界以 with/提交语句为准。
    async def load(
        self,
        conversation_id: str,
    ) -> ConversationSummaryState | None:

        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT summary_json, covered_message_count
                FROM conversation_summaries
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return ConversationSummaryState(
            summary=RollingConversationSummary.model_validate_json(row[0]),
            covered_message_count=row[1],
        )

    # 函数说明：SQLiteConversationSummaryStore.save
    # 用途：保存SQLiteConversationSummaryStore，供模型上下文与输入预算使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   state：当前状态快照，类型 `ConversationSummaryState`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `state.summary.model_dump_json` →
    # `datetime.now(UTC).isoformat` → `datetime.now` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：INSERT conversation_summaries/SET；连接与事务边界以 with/提交语句为
    # 准。
    async def save(
        self,
        conversation_id: str,
        state: ConversationSummaryState,
    ) -> None:

        async with self._connect() as database:
            await database.execute(
                """
                INSERT INTO conversation_summaries (
                    conversation_id,
                    summary_json,
                    covered_message_count,
                    updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    summary_json = excluded.summary_json,
                    covered_message_count = excluded.covered_message_count,
                    updated_at = excluded.updated_at
                """,
                (
                    conversation_id,
                    state.summary.model_dump_json(),
                    state.covered_message_count,
                    datetime.now(UTC).isoformat(),
                ),
            )
            await database.commit()

    # 函数说明：SQLiteConversationSummaryStore.delete
    # 用途：删除SQLiteConversationSummaryStore，供模型上下文与输入预算使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `bool`；返回 `cursor.rowcount > 0`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：DELETE conversation_summaries；连接与事务边界以 with/提交语句为准。
    async def delete(self, conversation_id: str) -> bool:

        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            )
            await database.commit()
        return cursor.rowcount > 0

    # 函数说明：SQLiteConversationSummaryStore._connect
    # 用途：连接SQLiteConversationSummaryStore，供模型上下文与输入预算使用。
    # 返回：异步生成器，逐项产出 `database`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`aiosqlite.connect` →
    # `database.execute` → `database.close`。
    # 副作用与资源：
    #   数据库操作：PRAGMA；连接与事务边界以 with/提交语句为准。
    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        await database.execute("PRAGMA foreign_keys = ON")
        try:
            yield database
        finally:
            await database.close()


__all__ = ["SQLiteConversationSummaryStore"]
