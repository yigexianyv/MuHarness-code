
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH
from app.models.types import AgentMode

from .models import (
    _ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    Run,
    RunStatus,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    status TEXT NOT NULL,
    user_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT,
    stop_reason TEXT,
    recovered_from_run_id TEXT,
    source TEXT,
    source_id TEXT,
    scheduled_for TEXT,
    triggered_at TEXT,
    mode TEXT NOT NULL DEFAULT 'normal'
);

CREATE INDEX IF NOT EXISTS idx_runs_conversation_updated
ON runs(conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_runs_status_updated
ON runs(status, created_at DESC);
"""

_PROVENANCE_COLUMNS = (
    "source TEXT",
    "source_id TEXT",
    "scheduled_for TEXT",
    "triggered_at TEXT",
    "mode TEXT NOT NULL DEFAULT 'normal'",
)


class RunAlreadyExists(ValueError):
    """同一个 Run ID 已经创建过；调用方绝不能再用它启动一次。"""

    # 函数说明：RunAlreadyExists.__init__
    # 用途：初始化 RunAlreadyExists；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.run_id`。
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"run already exists: {run_id}")


class SQLiteRunStore:

    # 函数说明：SQLiteRunStore.__init__
    # 用途：初始化 SQLiteRunStore；参数及实际保存的实例字段见下方说明。
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

    # 函数说明：SQLiteRunStore.initialize
    # 用途：初始化SQLiteRunStore，供运行调度与状态持久化使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.database_path.parent.mkdir`
    # → `self._connect` → `database.executescript` → `database.execute` →
    # `database.commit`。
    # 分支与异常：
    #   捕获 `aiosqlite.OperationalError` 后，忽略该异常并继续当前流程。
    # 副作用与资源：
    #   文件或资源访问：`self.database_path.parent.mkdir`。
    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            for column in _PROVENANCE_COLUMNS:
                try:
                    await database.execute(
                        f"ALTER TABLE runs ADD COLUMN {column}"
                    )
                except aiosqlite.OperationalError:
                    pass
            await database.commit()

    # 函数说明：SQLiteRunStore.create
    # 用途：创建SQLiteRunStore，供运行调度与状态持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   user_message：当前用户消息，类型 `str`；默认 `''`。
    #   recovered_from_run_id：运行标识，类型 `str | None`；默认 `None`。
    #   source：输入来源或原始数据，类型 `str | None`；默认 `None`。
    #   source_id：来源记录标识，类型 `str | None`；默认 `None`。
    #   scheduled_for：计划触发时间，类型 `datetime | None`；默认 `None`。
    #   triggered_at：实际触发时间，类型 `datetime | None`；默认 `None`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`；默认 `AgentMode.NORMAL`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`；读取键 `run_id`。
    # 返回：类型 `Run`；返回 `run`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier` → `uuid4` →
    #  `_now` → `self._connect` → `database.execute` → `_optional_identifier`；另有 9 个
    # 调用点。
    # 分支与异常：
    #   捕获 `aiosqlite.IntegrityError` 后，转换或抛出 `RunAlreadyExists(run_id)`。
    # 副作用与资源：
    #   数据库操作：INSERT runs；连接与事务边界以 with/提交语句为准。
    async def create(
        self,
        *,
        conversation_id: str | None = None,
        user_message: str = "",
        recovered_from_run_id: str | None = None,
        source: str | None = None,
        source_id: str | None = None,
        scheduled_for: datetime | None = None,
        triggered_at: datetime | None = None,
        mode: AgentMode = AgentMode.NORMAL,
        run_id: str | None = None,
    ) -> Run:

        # run_id 可由调用方预先分配并落盘（长任务先记下 ID 再启动），主键保证只创建一次
        run_id = (
            _required_identifier(run_id, "run_id") if run_id is not None else uuid4().hex
        )
        now = _now()
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            try:
                await database.execute(
                    """
                    INSERT INTO runs (
                        run_id, conversation_id, status, user_message,
                        created_at, updated_at, recovered_from_run_id,
                        source, source_id, scheduled_for, triggered_at, mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        _optional_identifier(conversation_id),
                        RunStatus.PENDING.value,
                        user_message,
                        now,
                        now,
                        _optional_identifier(recovered_from_run_id),
                        source,
                        _optional_identifier(source_id),
                        (
                            scheduled_for.astimezone(UTC).isoformat()
                            if scheduled_for is not None
                            else None
                        ),
                        (
                            triggered_at.astimezone(UTC).isoformat()
                            if triggered_at is not None
                            else None
                        ),
                        AgentMode(mode).value,
                    ),
                )
            except aiosqlite.IntegrityError as exc:
                await database.rollback()
                raise RunAlreadyExists(run_id) from exc
            await database.commit()
        run = await self.require(run_id)
        return run

    # 函数说明：SQLiteRunStore.get
    # 用途：获取SQLiteRunStore，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`；读取键 `run_id`。
    # 返回：类型 `Run | None`；返回 `_run_from_row(row) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_required_identifier` → `cursor.fetchone` → `_run_from_row`
    # 。
    # 副作用与资源：
    #   数据库操作：SELECT runs；连接与事务边界以 with/提交语句为准。
    async def get(self, run_id: str) -> Run | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (_required_identifier(run_id, "run_id"),),
            )
            row = await cursor.fetchone()
        return _run_from_row(row) if row is not None else None

    # 函数说明：SQLiteRunStore.list_runs
    # 用途：列出运行集合，供运行调度与状态持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`；读取键
    # `conversation_id`。
    #   status：目标状态，类型 `RunStatus | str | None`；默认 `None`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `20`。
    # 返回：类型 `tuple[Run, ...]`；返回 `tuple((_run_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier` →
    # `RunStatus` → `self._connect` → `database.execute` → `cursor.fetchall` →
    # `_run_from_row`。
    # 分支与异常：
    #   当 `limit < 1` 时，抛出 `ValueError('limit must be at least 1')`。
    async def list_runs(
        self,
        *,
        conversation_id: str | None = None,
        status: RunStatus | str | None = None,
        limit: int = 20,
    ) -> tuple[Run, ...]:

        if limit < 1:
            raise ValueError("limit must be at least 1")
        clauses: list[str] = []
        parameters: list[object] = []
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            parameters.append(_required_identifier(conversation_id, "conversation_id"))
        if status is not None:
            clauses.append("status = ?")
            parameters.append(RunStatus(status).value)
        query = "SELECT * FROM runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)

        async with self._connect() as database:
            cursor = await database.execute(query, tuple(parameters))
            rows = await cursor.fetchall()
        return tuple(_run_from_row(row) for row in rows)

    # 函数说明：SQLiteRunStore.list_for_conversation
    # 用途：列出会话，供运行调度与状态持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    #   active_only：活跃项输入或配置值，类型 `bool`；默认 `False`。
    # 返回：类型 `tuple[Run, ...]`；返回 `tuple((_run_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier` →
    # `self._connect` → `database.execute` → `cursor.fetchall` → `_run_from_row`。
    async def list_for_conversation(
        self,
        conversation_id: str,
        *,
        active_only: bool = False,
    ) -> tuple[Run, ...]:

        normalized = _required_identifier(conversation_id, "conversation_id")
        query = "SELECT * FROM runs WHERE conversation_id = ?"
        parameters: list[object] = [normalized]
        if active_only:
            query += " AND status IN (?, ?)"
            parameters.extend(
                (RunStatus.PENDING.value, RunStatus.RUNNING.value)
            )
        query += " ORDER BY created_at DESC"
        async with self._connect() as database:
            cursor = await database.execute(query, tuple(parameters))
            rows = await cursor.fetchall()
        return tuple(_run_from_row(row) for row in rows)

    # 函数说明：SQLiteRunStore.delete_for_conversation
    # 用途：删除会话，供运行调度与状态持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    # 返回：类型 `int`；返回 `max(cursor.rowcount, 0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier` →
    # `self._connect` → `database.execute` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：DELETE runs；连接与事务边界以 with/提交语句为准。
    async def delete_for_conversation(self, conversation_id: str) -> int:

        normalized = _required_identifier(conversation_id, "conversation_id")
        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM runs WHERE conversation_id = ?",
                (normalized,),
            )
            await database.commit()
        return max(cursor.rowcount, 0)

    # 函数说明：SQLiteRunStore.update_status
    # 用途：更新状态，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   status：目标状态，类型 `RunStatus`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    #   stop_reason：运行终止原因，类型 `str | None`；默认 `None`。
    # 返回：类型 `Run`；返回 `await self.require(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`RunStatus` → `self._connect` →
    # `database.execute` → `_require_row` → `_now` → `database.commit`；另有 1 个调用点
    # 。
    # 分支与异常：
    #   当 `status not in allowed` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   数据库操作：UPDATE runs；连接与事务边界以 with/提交语句为准。
    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        stop_reason: str | None = None,
    ) -> Run:

        if not isinstance(status, RunStatus):
            status = RunStatus(status)
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            current = await _require_row(database, run_id)
            allowed = _ALLOWED_TRANSITIONS[current.status]
            if status not in allowed:
                raise ValueError(
                    f"invalid run transition: {current.status.value} -> "
                    f"{status.value}"
                )
            now = _now()
            started_at = (
                current.started_at
                if current.started_at is not None
                else (now if status is RunStatus.RUNNING else None)
            )
            completed_at = (
                now
                if status in TERMINAL_STATUSES
                else None
            )
            await database.execute(
                """
                UPDATE runs
                SET status = ?, started_at = COALESCE(?, started_at),
                    updated_at = ?, completed_at = ?, error = ?, stop_reason = ?
                WHERE run_id = ?
                """,
                (
                    status.value,
                    started_at,
                    now,
                    completed_at,
                    error,
                    stop_reason,
                    run_id,
                ),
            )
            await database.commit()
        return await self.require(run_id)

    # 函数说明：SQLiteRunStore.mark_started
    # 用途：标记`started`，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Run`；返回 `await self.update_status(run_id, RunStatus.RUNNING)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.update_status`。
    async def mark_started(self, run_id: str) -> Run:

        return await self.update_status(run_id, RunStatus.RUNNING)

    # 函数说明：SQLiteRunStore.mark_completed
    # 用途：标记已完成项，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   stop_reason：运行终止原因，类型 `str | None`；默认 `None`。
    # 返回：类型 `Run`；返回
    # `await self.update_status(run_id, RunStatus.COMPLETED, stop_reason=stop_reason)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.update_status`。
    async def mark_completed(
        self,
        run_id: str,
        *,
        stop_reason: str | None = None,
    ) -> Run:
        return await self.update_status(
            run_id,
            RunStatus.COMPLETED,
            stop_reason=stop_reason,
        )

    # 函数说明：SQLiteRunStore.mark_failed
    # 用途：标记`failed`，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    #   stop_reason：运行终止原因，类型 `str | None`；默认 `None`。
    # 返回：类型 `Run`；返回 `await self.update_status(run_id, RunStatus.FAILED, error=
    # error, stop_reason=stop_reason)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.update_status`。
    async def mark_failed(
        self,
        run_id: str,
        *,
        error: str | None = None,
        stop_reason: str | None = None,
    ) -> Run:
        return await self.update_status(
            run_id,
            RunStatus.FAILED,
            error=error,
            stop_reason=stop_reason,
        )

    # 函数说明：SQLiteRunStore.mark_cancelled
    # 用途：标记`cancelled`，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    # 返回：类型 `Run`；返回
    # `await self.update_status(run_id, RunStatus.CANCELLED, error=error)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.update_status`。
    async def mark_cancelled(
        self,
        run_id: str,
        *,
        error: str | None = None,
    ) -> Run:
        return await self.update_status(
            run_id,
            RunStatus.CANCELLED,
            error=error,
        )

    # 函数说明：SQLiteRunStore.mark_interrupted
    # 用途：标记`interrupted`，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    # 返回：类型 `Run`；返回
    # `await self.update_status(run_id, RunStatus.INTERRUPTED, error=error)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.update_status`。
    async def mark_interrupted(
        self,
        run_id: str,
        *,
        error: str | None = None,
    ) -> Run:
        return await self.update_status(
            run_id,
            RunStatus.INTERRUPTED,
            error=error,
        )

    # 函数说明：SQLiteRunStore.require
    # 用途：获取并校验必需的SQLiteRunStore，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Run`；返回 `run`。
    # 分支与异常：
    #   当 `run is None` 时，抛出 `KeyError(f'Run 不存在：{run_id}')`。
    async def require(self, run_id: str) -> Run:
        run = await self.get(run_id)
        if run is None:
            raise KeyError(f"Run 不存在：{run_id}")
        return run

    # 函数说明：SQLiteRunStore._connect
    # 用途：连接SQLiteRunStore，供运行调度与状态持久化使用。
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


# 函数说明：_required_identifier
# 用途：校验必需的记录标识，拒绝缺失或不合要求的输入。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
#   field：待校验的字段名，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 分支与异常：
#   当 `not normalized` 时，抛出 `ValueError(f'{field} cannot be empty')`。
def _required_identifier(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


# 函数说明：_optional_identifier
# 用途：规范化可选标识，允许调用方省略该字段。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str | None`；读取键 `identifier`。
# 返回：类型 `str | None`；按分支返回 `None`；
# `_required_identifier(value, 'identifier')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier`。
# 分支与异常：
#   当 `value is None` 时，返回 `None`。
def _optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    return _required_identifier(value, "identifier")


# 函数说明：_require_row
# 用途：获取并校验必需的数据库行，供运行调度与状态持久化使用。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
#   run_id：目标运行标识，类型 `str`；读取键 `run_id`。
# 返回：类型 `Run`；返回 `_run_from_row(row)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute` →
# `_required_identifier` → `cursor.fetchone` → `_run_from_row`。
# 分支与异常：
#   当 `row is None` 时，抛出 `KeyError(f'Run 不存在：{run_id}')`。
# 副作用与资源：
#   数据库操作：SELECT runs；连接与事务边界以 with/提交语句为准。
async def _require_row(database: aiosqlite.Connection, run_id: str) -> Run:
    cursor = await database.execute(
        "SELECT * FROM runs WHERE run_id = ?",
        (_required_identifier(run_id, "run_id"),),
    )
    row = await cursor.fetchone()
    if row is None:
        raise KeyError(f"Run 不存在：{run_id}")
    return _run_from_row(row)


# 函数说明：_run_from_row
# 用途：运行数据库行，供运行调度与状态持久化使用。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `run_id`、
# `conversation_id`、`status`、`user_message`、`created_at`、`started_at`、`updated_at`
# 、`completed_at`、`error`、`stop_reason`、`recovered_from_run_id`、`source`、
# `source_id`、`scheduled_for`、`triggered_at`、`mode`。
# 返回：类型 `Run`；返回 `Run(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Run` → `RunStatus` →
# `_parse_datetime` → `AgentMode`。
def _run_from_row(row: aiosqlite.Row) -> Run:
    return Run(
        id=row["run_id"],
        conversation_id=row["conversation_id"],
        status=RunStatus(row["status"]),
        user_message=row["user_message"] or "",
        created_at=_parse_datetime(row["created_at"]),
        started_at=(
            _parse_datetime(row["started_at"])
            if row["started_at"] is not None
            else None
        ),
        updated_at=_parse_datetime(row["updated_at"]),
        completed_at=(
            _parse_datetime(row["completed_at"])
            if row["completed_at"] is not None
            else None
        ),
        error=row["error"],
        stop_reason=row["stop_reason"],
        recovered_from_run_id=row["recovered_from_run_id"],
        source=row["source"] if "source" in row.keys() else None,
        source_id=(
            row["source_id"] if "source_id" in row.keys() else None
        ),
        scheduled_for=(
            _parse_datetime(row["scheduled_for"])
            if "scheduled_for" in row.keys()
            and row["scheduled_for"] is not None
            else None
        ),
        triggered_at=(
            _parse_datetime(row["triggered_at"])
            if "triggered_at" in row.keys()
            and row["triggered_at"] is not None
            else None
        ),
        mode=(
            AgentMode(row["mode"])
            if "mode" in row.keys() and row["mode"] is not None
            else AgentMode.NORMAL
        ),
    )


# 函数说明：_parse_datetime
# 用途：解析日期时间，供运行调度与状态持久化使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `datetime`；返回 `datetime.fromisoformat(value)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.fromisoformat`。
def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


__all__ = ["RunAlreadyExists", "SQLiteRunStore"]
