

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import aiosqlite

from app.models.types import Message, ToolCall
from app.paths import default_database_path

from .models import Conversation, ConversationConstraints, ConversationMessageRecord

if TYPE_CHECKING:
    from app.runtime.context.summary import ConversationSummaryState
    from app.runtime.context.tool_views import ToolResultView

DEFAULT_DATABASE_PATH = default_database_path()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    name TEXT,
    tool_call_id TEXT,
    tool_calls_json TEXT NOT NULL DEFAULT '[]',
    reasoning TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(conversation_id, sequence),
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence
ON messages(conversation_id, sequence);

CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
ON conversations(updated_at DESC);

CREATE TABLE IF NOT EXISTS message_appends (
    idempotency_key TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id TEXT PRIMARY KEY,
    summary_json TEXT NOT NULL,
    covered_message_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tool_result_views (
    conversation_id TEXT NOT NULL,
    source_sequence INTEGER NOT NULL CHECK (source_sequence >= 0),
    raw_message_sha256 TEXT NOT NULL,
    representation TEXT NOT NULL CHECK (representation IN ('full', 'excerpt')),
    content TEXT,
    PRIMARY KEY(conversation_id, source_sequence),
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conversation_constraints (
    conversation_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    updated_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""

MAX_CONSTRAINTS_CHARS = 4000


class ConstraintsRevisionConflict(ValueError):
    """保存时的基准版本已过期：别处已经改过"必须记住的事项"。"""

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"必须记住的事项已被更新（当前第 {actual} 版），请刷新后再保存"
        )


class SQLiteConversationStore:

    # 函数说明：SQLiteConversationStore.__init__
    # 用途：初始化 SQLiteConversationStore；参数及实际保存的实例字段见下方说明。
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

    # 函数说明：SQLiteConversationStore.initialize
    # 用途：初始化SQLiteConversationStore，供会话生命周期与历史持久化使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.database_path.parent.mkdir`
    # → `self._connect` → `database.executescript` → `_ensure_column` →
    # `database.commit`。
    # 副作用与资源：
    #   文件或资源访问：`self.database_path.parent.mkdir`。
    async def initialize(self) -> None:

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await _ensure_column(database, "messages", "reasoning", "TEXT")
            await database.commit()

    # 函数说明：SQLiteConversationStore.create
    # 用途：创建SQLiteConversationStore，供会话生命周期与历史持久化使用。
    # 参数：
    #   title：面向用户的标题，类型 `str`；默认 `'新会话'`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`；默认 `()`。
    # 返回：类型 `Conversation`；返回 `conversation`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_title` → `uuid4` →
    # `_now_iso` → `self._connect` → `database.execute` → `self._insert_messages`；另有
    # 1 个调用点。
    # 分支与异常：
    #   当 `conversation is None` 时，抛出 `RuntimeError('创建会话后无法重新读取会话')`
    # 。
    # 副作用与资源：
    #   数据库操作：INSERT conversations；连接与事务边界以 with/提交语句为准。
    async def create(
        self,
        *,
        title: str = "新会话",
        messages: Sequence[Message] = (),
    ) -> Conversation:

        normalized_title = _normalize_title(title)
        conversation_id = uuid4().hex
        now = _now_iso()
        async with self._connect() as database:
            await database.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, normalized_title, now, now),
            )
            await self._insert_messages(database, conversation_id, messages, now)
            await database.commit()

        conversation = await self.get(conversation_id)
        if conversation is None:
            raise RuntimeError("创建会话后无法重新读取会话")
        return conversation

    # 函数说明：SQLiteConversationStore.get
    # 用途：获取SQLiteConversationStore，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `Conversation | None`；返回
    # `_conversation_from_row(row) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchone` → `_conversation_from_row`。
    async def get(self, conversation_id: str) -> Conversation | None:

        async with self._connect() as database:
            cursor = await database.execute(
                _CONVERSATION_SELECT + " WHERE c.id = ? GROUP BY c.id",
                (conversation_id,),
            )
            row = await cursor.fetchone()
        return _conversation_from_row(row) if row is not None else None

    # 函数说明：SQLiteConversationStore.resolve
    # 用途：解析或定位SQLiteConversationStore，供会话生命周期与历史持久化使用。
    # 参数：
    #   identifier：待规范化的标识，类型 `str`。
    # 返回：类型 `Conversation | None`；按分支返回 `None`；`exact`；
    # `_conversation_from_row(rows[0]) if rows else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `_conversation_from_row`。
    # 分支与异常：
    #   当 `not normalized` 时，返回 `None`。
    #   当 `exact is not None` 时，返回 `exact`。
    #   当 `len(rows) > 1` 时，抛出 `ValueError(f'会话 ID 前缀不唯一：{identifier}')`。
    async def resolve(self, identifier: str) -> Conversation | None:

        normalized = identifier.strip()
        if not normalized:
            return None

        exact = await self.get(normalized)
        if exact is not None:
            return exact

        async with self._connect() as database:
            cursor = await database.execute(
                _CONVERSATION_SELECT
                + " WHERE c.id LIKE ? GROUP BY c.id ORDER BY c.updated_at DESC LIMIT 2",
                (f"{normalized}%",),
            )
            rows = await cursor.fetchall()
        if len(rows) > 1:
            raise ValueError(f"会话 ID 前缀不唯一：{identifier}")
        return _conversation_from_row(rows[0]) if rows else None

    # 函数说明：SQLiteConversationStore.latest
    # 用途：获取最新的SQLiteConversationStore，供会话生命周期与历史持久化使用。
    # 返回：类型 `Conversation | None`；返回
    # `conversations[0] if conversations else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.list`。
    async def latest(self) -> Conversation | None:

        conversations = await self.list(limit=1)
        return conversations[0] if conversations else None

    # 函数说明：SQLiteConversationStore.list
    # 用途：列出SQLiteConversationStore，供会话生命周期与历史持久化使用。
    # 参数：
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `20`。
    # 返回：类型 `tuple[Conversation, ...]`；返回
    # `tuple((_conversation_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `_conversation_from_row`。
    # 分支与异常：
    #   当 `limit < 1` 时，抛出 `ValueError('limit must be at least 1')`。
    async def list(self, *, limit: int = 20) -> tuple[Conversation, ...]:

        if limit < 1:
            raise ValueError("limit must be at least 1")
        async with self._connect() as database:
            cursor = await database.execute(
                _CONVERSATION_SELECT
                + " GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
        return tuple(_conversation_from_row(row) for row in rows)

    # 函数说明：SQLiteConversationStore.load_messages
    # 用途：加载消息序列，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `tuple[Message, ...]`；返回
    # `tuple((_message_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `_message_from_row`。
    # 分支与异常：
    #   当 `await self.get(conversation_id) is None` 时，抛出
    # `KeyError(f'会话不存在：{conversation_id}')`。
    # 副作用与资源：
    #   数据库操作：SELECT messages；连接与事务边界以 with/提交语句为准。
    async def load_messages(self, conversation_id: str) -> tuple[Message, ...]:

        if await self.get(conversation_id) is None:
            raise KeyError(f"会话不存在：{conversation_id}")
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT role, content, name, tool_call_id, tool_calls_json, reasoning
                FROM messages
                WHERE conversation_id = ?
                ORDER BY sequence ASC
                """,
                (conversation_id,),
            )
            rows = await cursor.fetchall()
        return tuple(_message_from_row(row) for row in rows)

    # 函数说明：SQLiteConversationStore.load_summary_state
    # 用途：加载摘要状态，供会话生命周期与历史持久化使用。
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
    async def load_summary_state(
        self, conversation_id: str,
    ) -> ConversationSummaryState | None:
        from app.runtime.context.summary import (
            ConversationSummaryState,
            RollingConversationSummary,
        )

        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT summary_json, covered_message_count "
                "FROM conversation_summaries "
                "WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return ConversationSummaryState(
            summary=RollingConversationSummary.model_validate_json(row["summary_json"]),
            covered_message_count=row["covered_message_count"],
        )

    # 函数说明：SQLiteConversationStore.load_tool_result_views
    # 用途：加载工具结果，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `tuple[ToolResultView, ...]`；返回
    # `tuple((ToolResultView.model_validate(dict(row)) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `ToolResultView.model_validate`。
    # 副作用与资源：
    #   数据库操作：SELECT tool_result_views；连接与事务边界以 with/提交语句为准。
    async def load_tool_result_views(
        self, conversation_id: str,
    ) -> tuple[ToolResultView, ...]:
        from app.runtime.context.tool_views import ToolResultView

        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT source_sequence, raw_message_sha256, representation, content "
                "FROM tool_result_views WHERE conversation_id = ? "
                "ORDER BY source_sequence ASC",
                (conversation_id,),
            )
            rows = await cursor.fetchall()
        return tuple(ToolResultView.model_validate(dict(row)) for row in rows)

    # 函数说明：SQLiteConversationStore.search_messages
    # 用途：检索消息序列，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   query：检索查询文本，类型 `str`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `10`。
    # 返回：类型 `tuple[ConversationMessageRecord, ...]`；返回
    # `tuple((_message_record_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `_message_record_from_row`。
    # 分支与异常：
    #   当 `not normalized` 时，抛出 `ValueError('query must be a non-empty string')`。
    #   当 `limit < 1 or limit > 20` 时，抛出
    # `ValueError('limit must be between 1 and 20')`。
    #   当 `await self.get(conversation_id) is None` 时，抛出
    # `KeyError(f'会话不存在：{conversation_id}')`。
    # 副作用与资源：
    #   数据库操作：SELECT messages；连接与事务边界以 with/提交语句为准。
    async def search_messages(
        self,
        conversation_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[ConversationMessageRecord, ...]:

        normalized = query.strip()
        if not normalized:
            raise ValueError("query must be a non-empty string")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        if await self.get(conversation_id) is None:
            raise KeyError(f"会话不存在：{conversation_id}")
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT sequence, role, content, name, tool_call_id,
                       tool_calls_json, reasoning, created_at
                FROM messages
                WHERE conversation_id = ?
                  AND instr(lower(coalesce(content, '')), lower(?)) > 0
                ORDER BY sequence DESC LIMIT ?
                """,
                (conversation_id, normalized, limit),
            )
            rows = await cursor.fetchall()
        return tuple(_message_record_from_row(row) for row in rows)

    # 函数说明：SQLiteConversationStore.load_message_window
    # 用途：加载消息，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   sequence：事件或记录顺序号，类型 `int`。
    #   before：`before`输入或配置值，类型 `int`；默认 `2`。
    #   after：`after`输入或配置值，类型 `int`；默认 `2`。
    # 返回：类型 `tuple[ConversationMessageRecord, ...]`；返回
    # `tuple((_message_record_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `_message_record_from_row`。
    # 分支与异常：
    #   当 `sequence < 0` 时，抛出 `ValueError('sequence cannot be negative')`。
    #   当 `before < 0 or after < 0 or before > 10 or (after > 10)` 时，抛出
    # `ValueError('before and after must be between 0 and 10')`。
    #   当 `await self.get(conversation_id) is None` 时，抛出
    # `KeyError(f'会话不存在：{conversation_id}')`。
    # 副作用与资源：
    #   数据库操作：SELECT messages；连接与事务边界以 with/提交语句为准。
    async def load_message_window(
        self,
        conversation_id: str,
        sequence: int,
        *,
        before: int = 2,
        after: int = 2,
    ) -> tuple[ConversationMessageRecord, ...]:

        if sequence < 0:
            raise ValueError("sequence cannot be negative")
        if before < 0 or after < 0 or before > 10 or after > 10:
            raise ValueError("before and after must be between 0 and 10")
        if await self.get(conversation_id) is None:
            raise KeyError(f"会话不存在：{conversation_id}")
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT sequence, role, content, name, tool_call_id,
                       tool_calls_json, reasoning, created_at
                FROM messages
                WHERE conversation_id = ? AND sequence BETWEEN ? AND ?
                ORDER BY sequence ASC
                """,
                (conversation_id, max(0, sequence - before), sequence + after),
            )
            rows = await cursor.fetchall()
        return tuple(_message_record_from_row(row) for row in rows)

    # 函数说明：SQLiteConversationStore.replace_messages
    # 用途：替换消息序列，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    # 返回：类型 `Conversation`；返回 `await self.save_history_state(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.load_tool_result_views` →
    # `raw_message_sha256` → `self.save_history_state`。
    async def replace_messages(
        self,
        conversation_id: str,
        messages: Sequence[Message],
    ) -> Conversation:
        # Manual history edits invalidate the summary, but unchanged raw tool
        # messages keep their frozen views even when SQLite row IDs change.
        from app.runtime.context.tool_views import raw_message_sha256

        views = await self.load_tool_result_views(conversation_id)
        matching = tuple(
            view
            for view in views
            if view.source_sequence < len(messages)
            and messages[view.source_sequence].role.value == "tool"
            and raw_message_sha256(messages[view.source_sequence])
            == view.raw_message_sha256
        )
        return await self.save_history_state(
            conversation_id,
            messages,
            summary_state=None,
            tool_result_views=matching,
        )

    # 函数说明：SQLiteConversationStore.save_history_state
    # 用途：保存历史状态，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`。
    #   tool_result_views：工具结果的固定模型视图，类型 `Sequence[ToolResultView]`；默认
    #  `()`。
    # 返回：类型 `Conversation`；返回 `conversation`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`seen_sequences.add` →
    # `raw_message_sha256` → `_now_iso` → `self._connect` → `database.execute` →
    # `cursor.fetchone`；另有 4 个调用点。
    # 分支与异常：
    #   当 `summary_state is not None and…` 时，抛出
    # `ValueError('summary coverage exceeds the raw history')`。
    #   当 `sequence in seen_sequences` 时，抛出
    # `ValueError('duplicate tool result view sequence')`。
    #   当 `sequence >= len(messages) or messages[sequence].role.value…` 时，抛出
    # `ValueError(…)`。
    #   当 `await cursor.fetchone() is None` 时，抛出
    # `KeyError(f'会话不存在：{conversation_id}')`。
    # 副作用与资源：
    #   数据库操作：SELECT conversations、DELETE messages、DELETE tool_result_views、
    # INSERT tool_result_views、DELETE conversation_summaries、INSERT
    # conversation_summaries/SET、UPDATE conversations；连接与事务边界以 with/提交语句为
    # 准。
    async def save_history_state(
        self,
        conversation_id: str,
        messages: Sequence[Message],
        *,
        summary_state: ConversationSummaryState | None,
        tool_result_views: Sequence[ToolResultView] = (),
    ) -> Conversation:
        """Commit raw history, its summary and frozen tool views as one state."""
        from app.runtime.context.tool_views import raw_message_sha256

        messages = tuple(messages)
        if (
            summary_state is not None
            and summary_state.covered_message_count > len(messages)
        ):
            raise ValueError("summary coverage exceeds the raw history")
        seen_sequences: set[int] = set()
        for view in tool_result_views:
            sequence = view.source_sequence
            if sequence in seen_sequences:
                raise ValueError("duplicate tool result view sequence")
            seen_sequences.add(sequence)
            if (
                sequence >= len(messages)
                or messages[sequence].role.value != "tool"
                or raw_message_sha256(messages[sequence]) != view.raw_message_sha256
            ):
                raise ValueError("tool result view does not match the raw history")

        now = _now_iso()
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            cursor = await database.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            if await cursor.fetchone() is None:
                raise KeyError(f"会话不存在：{conversation_id}")
            await database.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            await self._insert_messages(database, conversation_id, messages, now)
            await database.execute(
                "DELETE FROM tool_result_views WHERE conversation_id = ?",
                (conversation_id,),
            )
            if tool_result_views:
                await database.executemany(
                    "INSERT INTO tool_result_views (conversation_id, source_sequence, "
                    "raw_message_sha256, representation, content) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            conversation_id, view.source_sequence,
                            view.raw_message_sha256, view.representation, view.content,
                        )
                        for view in tool_result_views
                    ],
                )
            if summary_state is None:
                await database.execute(
                    "DELETE FROM conversation_summaries WHERE conversation_id = ?",
                    (conversation_id,),
                )
            else:
                await database.execute(
                    "INSERT INTO conversation_summaries "
                    "(conversation_id, summary_json, covered_message_count, "
                    "updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(conversation_id) DO UPDATE SET "
                    "summary_json = excluded.summary_json, "
                    "covered_message_count = excluded.covered_message_count, "
                    "updated_at = excluded.updated_at",
                    (
                        conversation_id, summary_state.summary.model_dump_json(),
                        summary_state.covered_message_count, now,
                    ),
                )
            await database.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            await database.commit()

        conversation = await self.get(conversation_id)
        if conversation is None:
            raise RuntimeError("更新会话后无法重新读取会话")
        return conversation

    # 函数说明：SQLiteConversationStore.append_messages
    # 用途：在会话末尾追加消息；同一个 idempotency_key 只追加一次。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   idempotency_key：键输入或配置值，类型 `str`。
    # 返回：类型 `bool`；按分支返回 `False`；`True`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_now_iso` → `self._connect` →
    # `database.execute` → `cursor.fetchone` → `database.rollback` →
    # `self._insert_messages`；另有 1 个调用点。
    # 分支与异常：
    #   当 `not idempotency_key` 时，抛出
    # `ValueError('idempotency_key cannot be empty')`。
    #   `await cursor.fetchone() is None` 分支在完成前置处理后抛出
    # `KeyError(f'会话不存在：{conversation_id}')`。
    #   `await cursor.fetchone() is not None` 分支在完成前置处理后返回 `False`。
    # 副作用与资源：
    #   数据库操作：SELECT conversations、SELECT message_appends、SELECT messages、
    # INSERT message_appends、UPDATE conversations；连接与事务边界以 with/提交语句为准。
    async def append_messages(
        self,
        conversation_id: str,
        messages: Sequence[Message],
        *,
        idempotency_key: str,
    ) -> bool:

        """在会话末尾追加消息；同一个 idempotency_key 只追加一次。返回是否真的写入。

        用于会话之外产生的消息（例如长任务的最终回复）。键和消息在同一个事务里写入，
        进程在任何时刻退出后重试都不会出现重复消息。
        """
        if not idempotency_key:
            raise ValueError("idempotency_key cannot be empty")
        now = _now_iso()
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            cursor = await database.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            if await cursor.fetchone() is None:
                await database.rollback()
                raise KeyError(f"会话不存在：{conversation_id}")
            cursor = await database.execute(
                "SELECT 1 FROM message_appends WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            if await cursor.fetchone() is not None:
                await database.rollback()
                return False
            cursor = await database.execute(
                "SELECT COALESCE(MAX(sequence) + 1, 0) FROM messages "
                "WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            await self._insert_messages(
                database,
                conversation_id,
                messages,
                now,
                start_sequence=int(row[0]),
            )
            await database.execute(
                "INSERT INTO message_appends "
                "(idempotency_key, conversation_id, created_at) "
                "VALUES (?, ?, ?)",
                (idempotency_key, conversation_id, now),
            )
            await database.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            await database.commit()
        return True

    # 函数说明：SQLiteConversationStore.rename
    # 用途：重命名SQLiteConversationStore，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   title：面向用户的标题，类型 `str`。
    # 返回：类型 `Conversation`；返回 `conversation`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_now_iso` → `self._connect` →
    # `database.execute` → `_normalize_title` → `database.commit`。
    # 分支与异常：
    #   当 `cursor.rowcount == 0` 时，抛出 `KeyError(f'会话不存在：{conversation_id}')`
    # 。
    #   当 `conversation is None` 时，抛出
    # `RuntimeError('重命名会话后无法重新读取会话')`。
    # 副作用与资源：
    #   数据库操作：UPDATE conversations；连接与事务边界以 with/提交语句为准。
    async def get_constraints(self, conversation_id: str) -> ConversationConstraints:
        """读取"必须记住的事项"；从未设置时返回空文本、第 0 版。"""
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT text, revision, updated_at
                FROM conversation_constraints
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            if await self.get(conversation_id) is None:
                raise KeyError(f"会话不存在：{conversation_id}")
            return ConversationConstraints(conversation_id=conversation_id)
        return ConversationConstraints(
            conversation_id=conversation_id,
            text=row["text"],
            revision=row["revision"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def constraints_for_request(self, conversation_id: str) -> tuple[str, int]:
        """供 AgentLoop 每次组装请求时读取最新版本。"""
        constraints = await self.get_constraints(conversation_id)
        return constraints.text, constraints.revision

    async def set_constraints(
        self,
        conversation_id: str,
        text: str,
        *,
        expected_revision: int | None = None,
    ) -> ConversationConstraints:
        """保存新版本；运行中保存也会在下一次模型请求生效。"""
        normalized = text.replace("\r\n", "\n").strip()
        if len(normalized) > MAX_CONSTRAINTS_CHARS:
            raise ValueError(f"必须记住的事项不能超过 {MAX_CONSTRAINTS_CHARS} 个字符")
        now = _now_iso()
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            cursor = await database.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            if await cursor.fetchone() is None:
                await database.rollback()
                raise KeyError(f"会话不存在：{conversation_id}")
            cursor = await database.execute(
                """
                SELECT text, revision FROM conversation_constraints
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            row = await cursor.fetchone()
            current_text = row["text"] if row is not None else ""
            current = row["revision"] if row is not None else 0
            if expected_revision is not None and expected_revision != current:
                await database.rollback()
                raise ConstraintsRevisionConflict(expected_revision, current)
            if normalized == current_text:
                await database.rollback()
                return await self.get_constraints(conversation_id)
            await database.execute(
                """
                INSERT INTO conversation_constraints (
                    conversation_id, text, revision, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    text = excluded.text,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, normalized, current + 1, now),
            )
            await database.commit()
        return await self.get_constraints(conversation_id)

    async def rename(self, conversation_id: str, title: str) -> Conversation:

        now = _now_iso()
        async with self._connect() as database:
            cursor = await database.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (_normalize_title(title), now, conversation_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"会话不存在：{conversation_id}")
            await database.commit()

        conversation = await self.get(conversation_id)
        if conversation is None:
            raise RuntimeError("重命名会话后无法重新读取会话")
        return conversation

    # 函数说明：SQLiteConversationStore.delete
    # 用途：删除SQLiteConversationStore，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `bool`；返回 `cursor.rowcount > 0`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：DELETE conversations；连接与事务边界以 with/提交语句为准。
    async def delete(self, conversation_id: str) -> bool:

        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            await database.commit()
        return cursor.rowcount > 0

    # 函数说明：SQLiteConversationStore._connect
    # 用途：连接SQLiteConversationStore，供会话生命周期与历史持久化使用。
    # 返回：异步生成器，逐项产出 `database`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`aiosqlite.connect` →
    # `database.execute` → `database.close`。
    # 副作用与资源：
    #   更新对象字段：`database.row_factory`。
    #   数据库操作：PRAGMA；连接与事务边界以 with/提交语句为准。
    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        await database.execute("PRAGMA foreign_keys = ON")
        try:
            yield database
        finally:
            await database.close()

    # 函数说明：SQLiteConversationStore._insert_messages
    # 用途：插入消息序列，供会话生命周期与历史持久化使用。
    # 参数：
    #   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
    #   conversation_id：目标会话标识，类型 `str`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   created_at：`created_at`输入或配置值，类型 `str`。
    #   start_sequence：`start_sequence`输入或配置值，类型 `int`；默认 `0`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps` →
    # `database.executemany`。
    # 副作用与资源：
    #   数据库操作：INSERT messages；连接与事务边界以 with/提交语句为准。
    @staticmethod
    async def _insert_messages(
        database: aiosqlite.Connection,
        conversation_id: str,
        messages: Sequence[Message],
        created_at: str,
        *,
        start_sequence: int = 0,
    ) -> None:
        rows = [
            (
                conversation_id,
                sequence,
                message.role.value,
                message.content,
                message.name,
                message.tool_call_id,
                json.dumps(
                    [
                        tool_call.model_dump(mode="json")
                        for tool_call in message.tool_calls
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                message.reasoning,
                created_at,
            )
            for sequence, message in enumerate(messages, start=start_sequence)
        ]
        if rows:
            await database.executemany(
                """
                INSERT INTO messages (
                    conversation_id,
                    sequence,
                    role,
                    content,
                    name,
                    tool_call_id,
                    tool_calls_json,
                    reasoning,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


_CONVERSATION_SELECT = """
SELECT
    c.id,
    c.title,
    c.created_at,
    c.updated_at,
    COUNT(m.id) AS message_count
FROM conversations AS c
LEFT JOIN messages AS m ON m.conversation_id = c.id
"""


# 函数说明：_conversation_from_row
# 用途：将数据库行解析为会话记录。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `id`、`title`、
# `created_at`、`updated_at`、`message_count`。
# 返回：类型 `Conversation`；返回 `Conversation(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Conversation` →
# `datetime.fromisoformat`。
def _conversation_from_row(row: aiosqlite.Row) -> Conversation:
    return Conversation(
        id=row["id"],
        title=row["title"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        message_count=row["message_count"],
    )


# 函数说明：_ensure_column
# 用途：确保`column`，供会话生命周期与历史持久化使用。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
#   table：`table`输入或配置值，类型 `str`。
#   column：`column`输入或配置值，类型 `str`。
#   definition：工具定义输入或配置值，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute` → `cursor.fetchall`
# 。
async def _ensure_column(
    database: aiosqlite.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:

    cursor = await database.execute(f"PRAGMA table_info({table})")
    columns = {row["name"] for row in await cursor.fetchall()}
    if column not in columns:
        await database.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
        )


# 函数说明：_message_from_row
# 用途：将数据库行解析为消息记录，恢复持久化的消息字段。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `tool_calls_json`、
# `role`、`content`、`name`、`tool_call_id`、`reasoning`。
# 返回：类型 `Message`；返回 `Message(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` → `Message` →
# `ToolCall.model_validate`。
def _message_from_row(row: aiosqlite.Row) -> Message:
    raw_tool_calls = json.loads(row["tool_calls_json"])
    return Message(
        role=row["role"],
        content=row["content"],
        name=row["name"],
        tool_call_id=row["tool_call_id"],
        tool_calls=tuple(ToolCall.model_validate(item) for item in raw_tool_calls),
        reasoning=row["reasoning"],
    )


# 函数说明：_message_record_from_row
# 用途：记录数据库行，供会话生命周期与历史持久化使用。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `sequence`、
# `created_at`。
# 返回：类型 `ConversationMessageRecord`；返回 `ConversationMessageRecord(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationMessageRecord` →
# `_message_from_row` → `datetime.fromisoformat`。
def _message_record_from_row(row: aiosqlite.Row) -> ConversationMessageRecord:
    return ConversationMessageRecord(
        sequence=row["sequence"],
        message=_message_from_row(row),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# 函数说明：_normalize_title
# 用途：规范化`title`，供会话生命周期与历史持久化使用。
# 参数：
#   title：面向用户的标题，类型 `str`。
# 返回：类型 `str`；返回 `normalized[:80] or '新会话'`。
def _normalize_title(title: str) -> str:
    normalized = " ".join(title.split()).strip()
    return normalized[:80] or "新会话"


# 函数说明：_now_iso
# 用途：获取当前时间并转换为 ISO 格式文本。
# 返回：类型 `str`；返回 `datetime.now(UTC).isoformat()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now(UTC).isoformat` →
# `datetime.now`。
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
