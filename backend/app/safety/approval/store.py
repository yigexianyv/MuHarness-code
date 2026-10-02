
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH

from .models import ApprovalRequest, ApprovalRequestStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    conversation_id TEXT,
    tool_name TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    arguments_json TEXT NOT NULL DEFAULT '{}',
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_approvals_status_created
ON approvals(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_approvals_run_id
ON approvals(run_id);
"""


class SQLiteApprovalStore:

    # 函数说明：SQLiteApprovalStore.__init__
    # 用途：初始化 SQLiteApprovalStore；参数及实际保存的实例字段见下方说明。
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

    # 函数说明：SQLiteApprovalStore.initialize
    # 用途：初始化SQLiteApprovalStore，供人工审批请求与等待使用。
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

    # 函数说明：SQLiteApprovalStore.create
    # 用途：创建SQLiteApprovalStore，供人工审批请求与等待使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str | None`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   tool_name：工具名称，类型 `str`；读取键 `tool_name`。
    #   tool_call_id：工具调用标识，类型 `str`；读取键 `tool_call_id`。
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any] | None`；默认
    # `None`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`；默认 `''`。
    # 返回：类型 `ApprovalRequest`；返回 `await self.require(approval_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`uuid4` → `_now` → `self._connect`
    #  → `database.execute` → `_optional_identifier` → `_required`；另有 3 个调用点。
    # 副作用与资源：
    #   数据库操作：INSERT approvals；连接与事务边界以 with/提交语句为准。
    async def create(
        self,
        *,
        run_id: str | None,
        conversation_id: str | None,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any] | None = None,
        reason: str = "",
    ) -> ApprovalRequest:

        approval_id = uuid4().hex
        now = _now()
        arguments = arguments or {}
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            await database.execute(
                """
                INSERT INTO approvals (
                    id, run_id, conversation_id, tool_name, tool_call_id,
                    arguments_json, reason, status, created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    _optional_identifier(run_id),
                    _optional_identifier(conversation_id),
                    _required(tool_name, "tool_name"),
                    _required(tool_call_id, "tool_call_id"),
                    json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    reason or "",
                    ApprovalRequestStatus.PENDING.value,
                    now,
                    None,
                ),
            )
            await database.commit()
        return await self.require(approval_id)

    # 函数说明：SQLiteApprovalStore.get
    # 用途：获取SQLiteApprovalStore，供人工审批请求与等待使用。
    # 参数：
    #   approval_id：审批请求标识，类型 `str`；读取键 `approval_id`。
    # 返回：类型 `ApprovalRequest | None`；返回
    # `_approval_from_row(row) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_required` → `cursor.fetchone` → `_approval_from_row`。
    # 副作用与资源：
    #   数据库操作：SELECT approvals；连接与事务边界以 with/提交语句为准。
    async def get(self, approval_id: str) -> ApprovalRequest | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM approvals WHERE id = ?",
                (_required(approval_id, "approval_id"),),
            )
            row = await cursor.fetchone()
        return _approval_from_row(row) if row is not None else None

    # 函数说明：SQLiteApprovalStore.require
    # 用途：获取并校验必需的SQLiteApprovalStore，供人工审批请求与等待使用。
    # 参数：
    #   approval_id：审批请求标识，类型 `str`。
    # 返回：类型 `ApprovalRequest`；返回 `approval`。
    # 分支与异常：
    #   当 `approval is None` 时，抛出
    # `KeyError(f'ApprovalRequest 不存在：{approval_id}')`。
    async def require(self, approval_id: str) -> ApprovalRequest:
        approval = await self.get(approval_id)
        if approval is None:
            raise KeyError(f"ApprovalRequest 不存在：{approval_id}")
        return approval

    # 函数说明：SQLiteApprovalStore.list
    # 用途：列出SQLiteApprovalStore，供人工审批请求与等待使用。
    # 参数：
    #   status：目标状态，类型 `ApprovalRequestStatus | str | None`；默认 `None`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`；读取键 `run_id`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`；读取键
    # `conversation_id`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `50`。
    # 返回：类型 `tuple[ApprovalRequest, ...]`；返回
    # `tuple((_approval_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalRequestStatus` →
    # `_required` → `self._connect` → `database.execute` → `cursor.fetchall` →
    # `_approval_from_row`。
    # 分支与异常：
    #   当 `limit < 1` 时，抛出 `ValueError('limit must be at least 1')`。
    async def list(
        self,
        *,
        status: ApprovalRequestStatus | str | None = None,
        run_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
    ) -> tuple[ApprovalRequest, ...]:

        if limit < 1:
            raise ValueError("limit must be at least 1")
        clauses: list[str] = []
        parameters: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            parameters.append(ApprovalRequestStatus(status).value)
        if run_id is not None:
            clauses.append("run_id = ?")
            parameters.append(_required(run_id, "run_id"))
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            parameters.append(_required(conversation_id, "conversation_id"))
        query = "SELECT * FROM approvals"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)

        async with self._connect() as database:
            cursor = await database.execute(query, tuple(parameters))
            rows = await cursor.fetchall()
        return tuple(_approval_from_row(row) for row in rows)

    # 函数说明：SQLiteApprovalStore.resolve
    # 用途：解析或定位SQLiteApprovalStore，供人工审批请求与等待使用。
    # 参数：
    #   approval_id：审批请求标识，类型 `str`。
    #   status：目标状态，类型 `ApprovalRequestStatus | str`。
    # 返回：类型 `ApprovalRequest`；返回 `await self.require(approval_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalRequestStatus` →
    # `self._connect` → `database.execute` → `_require_row` → `_now` → `database.commit`
    # ；另有 1 个调用点。
    # 分支与异常：
    #   当 `resolved is ApprovalRequestStatus.PENDING` 时，抛出
    # `ValueError('cannot resolve approval back to PENDING')`。
    #   当 `current.status is not ApprovalRequestStatus.PENDING` 时，抛出
    # `ValueError(…)`。
    # 副作用与资源：
    #   数据库操作：UPDATE approvals；连接与事务边界以 with/提交语句为准。
    async def resolve(
        self,
        approval_id: str,
        status: ApprovalRequestStatus | str,
    ) -> ApprovalRequest:

        resolved = ApprovalRequestStatus(status)
        if resolved is ApprovalRequestStatus.PENDING:
            raise ValueError("cannot resolve approval back to PENDING")
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            current = await _require_row(database, approval_id)
            if current.status is not ApprovalRequestStatus.PENDING:
                raise ValueError(
                    f"approval already resolved: {approval_id} "
                    f"({current.status.value})"
                )
            now = _now()
            await database.execute(
                """
                UPDATE approvals
                SET status = ?, resolved_at = ?
                WHERE id = ?
                """,
                (resolved.value, now, approval_id),
            )
            await database.commit()
        return await self.require(approval_id)

    # 函数说明：SQLiteApprovalStore.cancel_pending_for_run
    # 用途：取消待处理项运行，供人工审批请求与等待使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`；读取键 `run_id`。
    # 返回：类型 `int`；返回 `cursor.rowcount`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required` → `self._connect` →
    # `database.execute` → `_now` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：UPDATE approvals；连接与事务边界以 with/提交语句为准。
    async def cancel_pending_for_run(self, run_id: str) -> int:

        normalized = _required(run_id, "run_id")
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            cursor = await database.execute(
                """
                UPDATE approvals
                SET status = ?, resolved_at = ?
                WHERE run_id = ? AND status = ?
                """,
                (
                    ApprovalRequestStatus.CANCELLED.value,
                    _now(),
                    normalized,
                    ApprovalRequestStatus.PENDING.value,
                ),
            )
            await database.commit()
        return cursor.rowcount

    # 函数说明：SQLiteApprovalStore.cancel_pending_for_conversation
    # 用途：取消待处理项会话，供人工审批请求与等待使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    # 返回：类型 `int`；返回 `cursor.rowcount`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required` → `self._connect` →
    # `database.execute` → `_now` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：UPDATE approvals；连接与事务边界以 with/提交语句为准。
    async def cancel_pending_for_conversation(self, conversation_id: str) -> int:

        normalized = _required(conversation_id, "conversation_id")
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            cursor = await database.execute(
                """
                UPDATE approvals
                SET status = ?, resolved_at = ?
                WHERE conversation_id = ? AND status = ?
                """,
                (
                    ApprovalRequestStatus.CANCELLED.value,
                    _now(),
                    normalized,
                    ApprovalRequestStatus.PENDING.value,
                ),
            )
            await database.commit()
        return cursor.rowcount

    # 函数说明：SQLiteApprovalStore.delete_for_conversation
    # 用途：删除会话，供人工审批请求与等待使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`；默认 `()`。
    # 返回：类型 `int`；返回 `max(cursor.rowcount, 0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required` → `self._connect` →
    # `database.execute` → `database.commit`。
    async def delete_for_conversation(
        self,
        conversation_id: str,
        *,
        run_ids: tuple[str, ...] = (),
    ) -> int:

        normalized = _required(conversation_id, "conversation_id")
        query = "DELETE FROM approvals WHERE conversation_id = ?"
        parameters: list[object] = [normalized]
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            query += f" OR run_id IN ({placeholders})"
            parameters.extend(run_ids)
        async with self._connect() as database:
            cursor = await database.execute(query, tuple(parameters))
            await database.commit()
        return max(cursor.rowcount, 0)

    # 函数说明：SQLiteApprovalStore.reconcile_orphans
    # 用途：核对并协调`orphans`，供人工审批请求与等待使用。
    # 参数：
    #   active_run_ids：活跃项运行输入或配置值，类型 `set[str]`。
    # 返回：类型 `int`；返回 `cursor.rowcount`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_now` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：UPDATE approvals；连接与事务边界以 with/提交语句为准。
    async def reconcile_orphans(self, active_run_ids: set[str]) -> int:

        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            if active_run_ids:
                placeholders = ",".join("?" for _ in active_run_ids)
                cursor = await database.execute(
                    f"""
                    UPDATE approvals
                    SET status = ?, resolved_at = ?
                    WHERE status = ?
                      AND (run_id IS NULL OR run_id NOT IN ({placeholders}))
                    """,
                    (
                        ApprovalRequestStatus.CANCELLED.value,
                        _now(),
                        ApprovalRequestStatus.PENDING.value,
                        *active_run_ids,
                    ),
                )
            else:
                cursor = await database.execute(
                    """
                    UPDATE approvals
                    SET status = ?, resolved_at = ?
                    WHERE status = ?
                    """,
                    (
                        ApprovalRequestStatus.CANCELLED.value,
                        _now(),
                        ApprovalRequestStatus.PENDING.value,
                    ),
                )
            await database.commit()
        return cursor.rowcount

    # 函数说明：SQLiteApprovalStore._connect
    # 用途：连接SQLiteApprovalStore，供人工审批请求与等待使用。
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


# 函数说明：_now
# 用途：获取当前时间，供记录时间字段或调度判断使用。
# 返回：类型 `str`；返回 `datetime.now(UTC).isoformat()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now(UTC).isoformat` →
# `datetime.now`。
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 函数说明：_required
# 用途：在人工审批请求与等待中处理 `_required`，通过 `value.strip` 完成首个内部处理步骤
# 。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
#   field：待校验的字段名，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 分支与异常：
#   当 `not normalized` 时，抛出 `ValueError(f'{field} cannot be empty')`。
def _required(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


# 函数说明：_optional_identifier
# 用途：规范化可选标识，允许调用方省略该字段。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str | None`；读取键 `identifier`。
# 返回：类型 `str | None`；按分支返回 `None`；`_required(value, 'identifier')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_required`。
# 分支与异常：
#   当 `value is None` 时，返回 `None`。
def _optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    return _required(value, "identifier")


# 函数说明：_require_row
# 用途：获取并校验必需的数据库行，供人工审批请求与等待使用。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
#   approval_id：审批请求标识，类型 `str`；读取键 `approval_id`。
# 返回：类型 `ApprovalRequest`；返回 `_approval_from_row(row)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute` → `_required` →
# `cursor.fetchone` → `_approval_from_row`。
# 分支与异常：
#   当 `row is None` 时，抛出 `KeyError(f'ApprovalRequest 不存在：{approval_id}')`。
# 副作用与资源：
#   数据库操作：SELECT approvals；连接与事务边界以 with/提交语句为准。
async def _require_row(
    database: aiosqlite.Connection,
    approval_id: str,
) -> ApprovalRequest:
    cursor = await database.execute(
        "SELECT * FROM approvals WHERE id = ?",
        (_required(approval_id, "approval_id"),),
    )
    row = await cursor.fetchone()
    if row is None:
        raise KeyError(f"ApprovalRequest 不存在：{approval_id}")
    return _approval_from_row(row)


# 函数说明：_approval_from_row
# 用途：返回 `ApprovalRequest(…)`，提供 人工审批请求与等待 的派生值。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `id`、`run_id`、
# `conversation_id`、`tool_name`、`tool_call_id`、`arguments_json`、`reason`、`status`、
# `created_at`、`resolved_at`。
# 返回：类型 `ApprovalRequest`；返回 `ApprovalRequest(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalRequest` → `json.loads` →
# `ApprovalRequestStatus` → `datetime.fromisoformat`。
def _approval_from_row(row: aiosqlite.Row) -> ApprovalRequest:
    return ApprovalRequest(
        id=row["id"],
        run_id=row["run_id"],
        conversation_id=row["conversation_id"],
        tool_name=row["tool_name"],
        tool_call_id=row["tool_call_id"],
        arguments=json.loads(row["arguments_json"] or "{}"),
        reason=row["reason"] or "",
        status=ApprovalRequestStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        resolved_at=(
            datetime.fromisoformat(row["resolved_at"])
            if row["resolved_at"] is not None
            else None
        ),
    )


__all__ = ["SQLiteApprovalStore"]
