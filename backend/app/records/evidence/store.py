
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

from app.domain.conversation.store import DEFAULT_DATABASE_PATH

from .models import EvidenceDocument, EvidenceRecord, EvidenceSearchHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    content TEXT NOT NULL,
    content_chars INTEGER NOT NULL,
    content_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    task_id TEXT,
    task_step_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, tool_call_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_conversation_created
ON evidence(conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_evidence_conversation_task
ON evidence(conversation_id, task_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_evidence_run
ON evidence(run_id, created_at ASC);
"""
_EVIDENCE_ID_RE = re.compile(r"^[0-9a-f]{4,32}$")
DEFAULT_MAX_EVIDENCE_ITEM_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_EVIDENCE_TOTAL_BYTES = 512 * 1024 * 1024


class EvidenceCapacityError(ValueError):
    pass


class SQLiteEvidenceStore:

    # 函数说明：SQLiteEvidenceStore.__init__
    # 用途：初始化 SQLiteEvidenceStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   database_path：SQLite 数据库路径，类型 `str | Path`；默认
    # `DEFAULT_DATABASE_PATH`。
    #   max_item_bytes：`max_item_bytes`输入或配置值，类型 `int`；默认
    # `DEFAULT_MAX_EVIDENCE_ITEM_BYTES`。
    #   max_total_bytes：`max_total_bytes`输入或配置值，类型 `int`；默认
    # `DEFAULT_MAX_EVIDENCE_TOTAL_BYTES`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(database_path).expanduser().resolve` → `Path(database_path).expanduser` →
    # `Path`。
    # 分支与异常：
    #   当 `max_item_bytes < 1 or max_total_bytes < 1` 时，抛出
    # `ValueError('evidence capacity limits must be positive')`。
    #   当 `max_item_bytes > max_total_bytes` 时，抛出
    # `ValueError('max_item_bytes cannot exceed max_total_bytes')`。
    # 副作用与资源：
    #   更新对象字段：`self.database_path`、`self.max_item_bytes`、
    # `self.max_total_bytes`。
    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        max_item_bytes: int = DEFAULT_MAX_EVIDENCE_ITEM_BYTES,
        max_total_bytes: int = DEFAULT_MAX_EVIDENCE_TOTAL_BYTES,
    ) -> None:
        if max_item_bytes < 1 or max_total_bytes < 1:
            raise ValueError("evidence capacity limits must be positive")
        if max_item_bytes > max_total_bytes:
            raise ValueError("max_item_bytes cannot exceed max_total_bytes")
        self.database_path = Path(database_path).expanduser().resolve()
        self.max_item_bytes = max_item_bytes
        self.max_total_bytes = max_total_bytes

    # 函数说明：SQLiteEvidenceStore.initialize
    # 用途：初始化SQLiteEvidenceStore，供原始工具证据持久化使用。
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

    # 函数说明：SQLiteEvidenceStore.create
    # 用途：创建SQLiteEvidenceStore，供原始工具证据持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    #   run_id：目标运行标识，类型 `str`；读取键 `run_id`。
    #   tool_call_id：工具调用标识，类型 `str`；读取键 `tool_call_id`。
    #   tool_name：工具名称，类型 `str`；读取键 `tool_name`。
    #   content：内容正文，类型 `str`。
    #   sha256：`sha256`输入或配置值，类型 `str`。
    #   task_id：目标任务标识，类型 `str | None`；默认 `None`。
    #   task_step_id：任务步骤标识，类型 `str | None`；默认 `None`。
    # 返回：类型 `EvidenceRecord`；按分支返回 `_record_from_row(existing)`；
    # `EvidenceRecord(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required` → `uuid4` →
    # `datetime.now` → `content.encode` → `self._connect` → `database.execute`；另有 6
    # 个调用点。
    # 分支与异常：
    #   `existing is not None` 分支在完成前置处理后返回 `_record_from_row(existing)`。
    #   当 `existing['sha256'] != sha256 or existing['conversation_id']…` 时，抛出
    # `ValueError(…)`。
    #   当 `content_bytes > self.max_item_bytes` 时，抛出 `EvidenceCapacityError(…)`。
    #   当 `total_bytes + content_bytes > self.max_total_bytes` 时，抛出
    # `EvidenceCapacityError(…)`。
    #   捕获 `Exception` 后，重新抛出原异常。
    # 副作用与资源：
    #   数据库操作：SELECT evidence、INSERT evidence；连接与事务边界以 with/提交语句为准
    # 。
    async def create(
        self,
        *,
        conversation_id: str,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        content: str,
        sha256: str,
        task_id: str | None = None,
        task_step_id: str | None = None,
    ) -> EvidenceRecord:

        conversation_id = _required(conversation_id, "conversation_id")
        run_id = _required(run_id, "run_id")
        tool_call_id = _required(tool_call_id, "tool_call_id")
        tool_name = _required(tool_name, "tool_name")
        evidence_id = uuid4().hex
        created_at = datetime.now(UTC)
        content_bytes = len(content.encode("utf-8"))
        async with self._connect() as database:
            try:
                await database.execute("BEGIN IMMEDIATE")
                cursor = await database.execute(
                    "SELECT * FROM evidence WHERE run_id = ? AND tool_call_id = ?",
                    (run_id, tool_call_id),
                )
                existing = await cursor.fetchone()
                if existing is not None:
                    if (
                        existing["sha256"] != sha256
                        or existing["conversation_id"] != conversation_id
                        or existing["tool_name"] != tool_name
                    ):
                        raise ValueError(
                            "evidence conflict: tool call already has different facts"
                        )
                    await database.commit()
                    return _record_from_row(existing)
                if content_bytes > self.max_item_bytes:
                    raise EvidenceCapacityError(
                        "evidence item exceeds max_item_bytes: "
                        f"{content_bytes} > {self.max_item_bytes}"
                    )
                cursor = await database.execute(
                    "SELECT COALESCE(SUM(content_bytes), 0) AS total FROM evidence"
                )
                total_bytes = int((await cursor.fetchone())["total"])
                if total_bytes + content_bytes > self.max_total_bytes:
                    raise EvidenceCapacityError(
                        "evidence store exceeds max_total_bytes: "
                        f"{total_bytes + content_bytes} > {self.max_total_bytes}"
                    )
                await database.execute(
                    """
                    INSERT INTO evidence (
                        id, conversation_id, run_id, tool_call_id, tool_name,
                        content_type, content, content_chars, content_bytes, sha256,
                        task_id, task_step_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence_id,
                        conversation_id,
                        run_id,
                        tool_call_id,
                        tool_name,
                        "text/plain; charset=utf-8",
                        content,
                        len(content),
                        content_bytes,
                        sha256,
                        task_id,
                        task_step_id,
                        created_at.isoformat(),
                    ),
                )
                await database.commit()
            except Exception:
                await database.rollback()
                raise
        return EvidenceRecord(
            id=evidence_id,
            conversation_id=conversation_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content_chars=len(content),
            content_bytes=content_bytes,
            sha256=sha256,
            task_id=task_id,
            task_step_id=task_step_id,
            created_at=created_at,
        )

    # 函数说明：SQLiteEvidenceStore.resolve
    # 用途：解析或定位SQLiteEvidenceStore，供原始工具证据持久化使用。
    # 参数：
    #   identifier：待规范化的标识，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    # 返回：类型 `EvidenceDocument | None`；按分支返回 `None`；
    # `EvidenceDocument(record=_record_from_row(rows[0]), content=rows[0]['content'])`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`identifier.strip().lower` →
    # `_EVIDENCE_ID_RE.fullmatch` → `self._connect` → `database.execute` → `_required` →
    #  `cursor.fetchall`；另有 2 个调用点。
    # 分支与异常：
    #   当 `not _EVIDENCE_ID_RE.fullmatch(normalized)` 时，抛出 `ValueError(…)`。
    #   当 `len(rows) > 1` 时，抛出
    # `ValueError(f'Evidence ID 前缀不唯一：{identifier}')`。
    #   当 `not rows` 时，返回 `None`。
    # 副作用与资源：
    #   数据库操作：SELECT evidence；连接与事务边界以 with/提交语句为准。
    async def resolve(
        self,
        identifier: str,
        *,
        conversation_id: str,
    ) -> EvidenceDocument | None:

        normalized = identifier.strip().lower()
        if not _EVIDENCE_ID_RE.fullmatch(normalized):
            raise ValueError(
                "evidence id must be a 4-32 character hexadecimal prefix"
            )
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM evidence
                WHERE conversation_id = ? AND id LIKE ?
                ORDER BY created_at DESC LIMIT 2
                """,
                (_required(conversation_id, "conversation_id"), f"{normalized}%"),
            )
            rows = await cursor.fetchall()
        if len(rows) > 1:
            raise ValueError(f"Evidence ID 前缀不唯一：{identifier}")
        if not rows:
            return None
        return EvidenceDocument(
            record=_record_from_row(rows[0]),
            content=rows[0]["content"],
        )

    # 函数说明：SQLiteEvidenceStore.search
    # 用途：检索SQLiteEvidenceStore，供原始工具证据持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    #   query：检索查询文本，类型 `str`。
    #   tool_name：工具名称，类型 `str | None`；默认 `None`。
    #   task_id：目标任务标识，类型 `str | None`；默认 `None`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `10`。
    # 返回：类型 `tuple[EvidenceSearchHit, ...]`；返回 `tuple(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required` → `self._connect` →
    # `database.execute` → `cursor.fetchall` → `EvidenceSearchHit` → `_record_from_row`
    # ；另有 1 个调用点。
    # 分支与异常：
    #   当 `limit < 1 or limit > 20` 时，抛出
    # `ValueError('limit must be between 1 and 20')`。
    #   当 `not normalized_query` 时，抛出
    # `ValueError('query must be a non-empty string')`。
    async def search(
        self,
        *,
        conversation_id: str,
        query: str,
        tool_name: str | None = None,
        task_id: str | None = None,
        limit: int = 10,
    ) -> tuple[EvidenceSearchHit, ...]:

        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be a non-empty string")
        filters = ["conversation_id = ?", "instr(lower(content), lower(?)) > 0"]
        parameters: list[object] = [
            _required(conversation_id, "conversation_id"),
            normalized_query,
        ]
        if tool_name:
            filters.append("tool_name = ?")
            parameters.append(tool_name.strip())
        if task_id:
            filters.append("task_id = ?")
            parameters.append(task_id.strip())
        parameters.append(limit)
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM evidence WHERE "
                + " AND ".join(filters)
                + " ORDER BY created_at DESC LIMIT ?",
                parameters,
            )
            rows = await cursor.fetchall()
        return tuple(
            EvidenceSearchHit(
                record=_record_from_row(row),
                snippet=_snippet(row["content"], normalized_query),
            )
            for row in rows
        )

    # 函数说明：SQLiteEvidenceStore.list_recent
    # 用途：列出近期项，供原始工具证据持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `12`。
    # 返回：类型 `tuple[EvidenceRecord, ...]`；返回
    # `tuple((_record_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_required` → `cursor.fetchall` → `_record_from_row`。
    # 分支与异常：
    #   当 `limit < 1` 时，抛出 `ValueError('limit must be at least 1')`。
    # 副作用与资源：
    #   数据库操作：SELECT evidence；连接与事务边界以 with/提交语句为准。
    async def list_recent(
        self,
        *,
        conversation_id: str,
        limit: int = 12,
    ) -> tuple[EvidenceRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM evidence WHERE conversation_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (_required(conversation_id, "conversation_id"), limit),
            )
            rows = await cursor.fetchall()
        return tuple(_record_from_row(row) for row in rows)

    # 函数说明：SQLiteEvidenceStore.list_for_task
    # 用途：列出任务，供原始工具证据持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    #   task_id：目标任务标识，类型 `str`；读取键 `task_id`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `20`。
    # 返回：类型 `tuple[EvidenceRecord, ...]`；返回
    # `tuple((_record_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_required` → `cursor.fetchall` → `_record_from_row`。
    # 副作用与资源：
    #   数据库操作：SELECT evidence；连接与事务边界以 with/提交语句为准。
    async def list_for_task(
        self,
        *,
        conversation_id: str,
        task_id: str,
        limit: int = 20,
    ) -> tuple[EvidenceRecord, ...]:
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM evidence
                WHERE conversation_id = ? AND task_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (
                    _required(conversation_id, "conversation_id"),
                    _required(task_id, "task_id"),
                    limit,
                ),
            )
            rows = await cursor.fetchall()
        return tuple(_record_from_row(row) for row in rows)

    # 函数说明：SQLiteEvidenceStore.delete_for_conversation
    # 用途：删除会话，供原始工具证据持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    # 返回：类型 `int`；返回 `max(cursor.rowcount, 0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required` → `self._connect` →
    # `database.execute` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：DELETE evidence；连接与事务边界以 with/提交语句为准。
    async def delete_for_conversation(self, conversation_id: str) -> int:

        normalized = _required(conversation_id, "conversation_id")
        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM evidence WHERE conversation_id = ?",
                (normalized,),
            )
            await database.commit()
        return max(cursor.rowcount, 0)

    # 函数说明：SQLiteEvidenceStore._connect
    # 用途：连接SQLiteEvidenceStore，供原始工具证据持久化使用。
    # 返回：异步生成器，逐项产出 `database`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`aiosqlite.connect` →
    # `database.close`。
    # 副作用与资源：
    #   更新对象字段：`database.row_factory`。
    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        try:
            yield database
        finally:
            await database.close()


# 函数说明：_record_from_row
# 用途：记录数据库行，供原始工具证据持久化使用。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `id`、`conversation_id`
# 、`run_id`、`tool_call_id`、`tool_name`、`content_type`、`content_chars`、
# `content_bytes`、`sha256`、`task_id`、`task_step_id`、`created_at`。
# 返回：类型 `EvidenceRecord`；返回 `EvidenceRecord(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`EvidenceRecord` →
# `datetime.fromisoformat`。
def _record_from_row(row: aiosqlite.Row) -> EvidenceRecord:
    return EvidenceRecord(
        id=row["id"],
        conversation_id=row["conversation_id"],
        run_id=row["run_id"],
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
        content_type=row["content_type"],
        content_chars=row["content_chars"],
        content_bytes=row["content_bytes"],
        sha256=row["sha256"],
        task_id=row["task_id"],
        task_step_id=row["task_step_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# 函数说明：_snippet
# 用途：在原始工具证据持久化中处理 `_snippet`，通过 `content.casefold().find` 完成首个内
# 部处理步骤。
# 参数：
#   content：内容正文，类型 `str`。
#   query：检索查询文本，类型 `str`。
#   radius：`radius`输入或配置值，类型 `int`；默认 `180`。
# 返回：类型 `str`；按分支返回 `content[:radius * 2]`；
# `f'{prefix}{content[start:end]}{suffix}'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`content.casefold().find` →
# `content.casefold` → `query.casefold`。
# 分支与异常：
#   当 `index < 0` 时，返回 `content[:radius * 2]`。
def _snippet(content: str, query: str, *, radius: int = 180) -> str:
    index = content.casefold().find(query.casefold())
    if index < 0:
        return content[: radius * 2]
    start = max(0, index - radius)
    end = min(len(content), index + len(query) + radius)
    prefix = "…" if start else ""
    suffix = "…" if end < len(content) else ""
    return f"{prefix}{content[start:end]}{suffix}"


# 函数说明：_required
# 用途：在原始工具证据持久化中处理 `_required`，通过 `value.strip` 完成首个内部处理步骤
# 。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
#   field_name：待校验的字段名称，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 分支与异常：
#   当 `not normalized` 时，抛出
# `ValueError(f'{field_name} must be a non-empty string')`。
def _required(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


__all__ = [
    "DEFAULT_MAX_EVIDENCE_ITEM_BYTES",
    "DEFAULT_MAX_EVIDENCE_TOTAL_BYTES",
    "EvidenceCapacityError",
    "SQLiteEvidenceStore",
]
