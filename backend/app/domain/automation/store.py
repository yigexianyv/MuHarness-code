
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH

from .models import Automation, AutomationStatus, Schedule

_AUTOMATION_TRANSITIONS: dict[AutomationStatus, frozenset[AutomationStatus]] = {
    AutomationStatus.ACTIVE: frozenset(
        {
            AutomationStatus.PAUSED,
            AutomationStatus.COMPLETED,
            AutomationStatus.CANCELLED,
        }
    ),
    AutomationStatus.PAUSED: frozenset(
        {
            AutomationStatus.ACTIVE,
            AutomationStatus.CANCELLED,
        }
    ),
    AutomationStatus.COMPLETED: frozenset(),
    AutomationStatus.CANCELLED: frozenset(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS automations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL,
    conversation_id TEXT,
    status TEXT NOT NULL,
    schedule_json TEXT NOT NULL,
    next_run_at TEXT,
    last_run_at TEXT,
    last_run_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_automations_status_updated
ON automations(status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_automations_conversation
ON automations(conversation_id, updated_at DESC);
"""


class SQLiteAutomationStore:

    # 函数说明：SQLiteAutomationStore.__init__
    # 用途：初始化 SQLiteAutomationStore；参数及实际保存的实例字段见下方说明。
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

    # 函数说明：SQLiteAutomationStore.initialize
    # 用途：初始化SQLiteAutomationStore，供定时任务调度使用。
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

    # 函数说明：SQLiteAutomationStore.create
    # 用途：创建SQLiteAutomationStore，供定时任务调度使用。
    # 参数：
    #   title：面向用户的标题，类型 `str`。
    #   prompt：本次调用使用的提示文本，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   schedule：调度输入或配置值，类型 `Schedule`。
    #   next_run_at：下一次运行时间，类型 `datetime`。
    # 返回：类型 `Automation`；返回 `automation`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`uuid4` → `_now` → `self._connect`
    #  → `database.execute` → `_optional_identifier` → `schedule.model_dump_json`；另有
    # 4 个调用点。
    # 副作用与资源：
    #   数据库操作：INSERT automations；连接与事务边界以 with/提交语句为准。
    async def create(
        self,
        *,
        title: str,
        prompt: str,
        conversation_id: str | None,
        schedule: Schedule,
        next_run_at: datetime,
    ) -> Automation:

        automation_id = uuid4().hex
        now = _now()
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            await database.execute(
                """
                INSERT INTO automations (
                    id, title, prompt, conversation_id, status, schedule_json,
                    next_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    automation_id,
                    title,
                    prompt,
                    _optional_identifier(conversation_id),
                    AutomationStatus.ACTIVE.value,
                    schedule.model_dump_json(),
                    next_run_at.astimezone(UTC).isoformat(),
                    now,
                    now,
                ),
            )
            await database.commit()
        automation = await self.require(automation_id)
        return automation

    # 函数说明：SQLiteAutomationStore.get
    # 用途：获取SQLiteAutomationStore，供定时任务调度使用。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`；读取键 `automation_id`。
    # 返回：类型 `Automation | None`；返回
    # `_automation_from_row(row) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_required_identifier` → `cursor.fetchone` →
    # `_automation_from_row`。
    # 副作用与资源：
    #   数据库操作：SELECT automations；连接与事务边界以 with/提交语句为准。
    async def get(self, automation_id: str) -> Automation | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM automations WHERE id = ?",
                (_required_identifier(automation_id, "automation_id"),),
            )
            row = await cursor.fetchone()
        return _automation_from_row(row) if row is not None else None

    # 函数说明：SQLiteAutomationStore.resolve
    # 用途：解析或定位SQLiteAutomationStore，供定时任务调度使用。
    # 参数：
    #   identifier：待规范化的标识，类型 `str`。
    # 返回：类型 `Automation | None`；按分支返回 `None`；`exact`；
    # `_automation_from_row(rows[0]) if rows else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `_automation_from_row`。
    # 分支与异常：
    #   当 `not normalized` 时，返回 `None`。
    #   当 `exact is not None` 时，返回 `exact`。
    #   当 `len(rows) > 1` 时，抛出
    # `ValueError(f'Automation ID 前缀不唯一：{identifier}')`。
    # 副作用与资源：
    #   数据库操作：SELECT automations；连接与事务边界以 with/提交语句为准。
    async def resolve(self, identifier: str) -> Automation | None:

        normalized = identifier.strip()
        if not normalized:
            return None
        exact = await self.get(normalized)
        if exact is not None:
            return exact
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM automations
                WHERE id LIKE ?
                ORDER BY updated_at DESC LIMIT 2
                """,
                (f"{normalized}%",),
            )
            rows = await cursor.fetchall()
        if len(rows) > 1:
            raise ValueError(f"Automation ID 前缀不唯一：{identifier}")
        return _automation_from_row(rows[0]) if rows else None

    # 函数说明：SQLiteAutomationStore.list
    # 用途：列出SQLiteAutomationStore，供定时任务调度使用。
    # 参数：
    #   status：目标状态，类型 `AutomationStatus | str | None`；默认 `None`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`；读取键
    # `conversation_id`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `50`。
    # 返回：类型 `tuple[Automation, ...]`；返回
    # `tuple((_automation_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AutomationStatus` →
    # `_required_identifier` → `self._connect` → `database.execute` → `cursor.fetchall`
    # → `_automation_from_row`。
    # 分支与异常：
    #   当 `limit < 1` 时，抛出 `ValueError('limit must be at least 1')`。
    async def list(
        self,
        *,
        status: AutomationStatus | str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
    ) -> tuple[Automation, ...]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        clauses: list[str] = []
        parameters: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            parameters.append(AutomationStatus(status).value)
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            parameters.append(_required_identifier(conversation_id, "conversation_id"))
        query = "SELECT * FROM automations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC LIMIT ?"
        parameters.append(limit)

        async with self._connect() as database:
            cursor = await database.execute(query, tuple(parameters))
            rows = await cursor.fetchall()
        return tuple(_automation_from_row(row) for row in rows)

    # 函数说明：SQLiteAutomationStore.list_for_conversation
    # 用途：列出会话，供定时任务调度使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    # 返回：类型 `tuple[Automation, ...]`；返回
    # `tuple((_automation_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier` →
    # `self._connect` → `database.execute` → `cursor.fetchall` → `_automation_from_row`
    # 。
    # 副作用与资源：
    #   数据库操作：SELECT automations；连接与事务边界以 with/提交语句为准。
    async def list_for_conversation(
        self,
        conversation_id: str,
    ) -> tuple[Automation, ...]:

        normalized = _required_identifier(conversation_id, "conversation_id")
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM automations
                WHERE conversation_id = ?
                ORDER BY updated_at DESC
                """,
                (normalized,),
            )
            rows = await cursor.fetchall()
        return tuple(_automation_from_row(row) for row in rows)

    # 函数说明：SQLiteAutomationStore.delete_for_conversation
    # 用途：删除会话，供定时任务调度使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    # 返回：类型 `int`；返回 `max(cursor.rowcount, 0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier` →
    # `self._connect` → `database.execute` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：DELETE automations；连接与事务边界以 with/提交语句为准。
    async def delete_for_conversation(self, conversation_id: str) -> int:

        normalized = _required_identifier(conversation_id, "conversation_id")
        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM automations WHERE conversation_id = ?",
                (normalized,),
            )
            await database.commit()
        return max(cursor.rowcount, 0)

    # 函数说明：SQLiteAutomationStore.update_status
    # 用途：更新状态，供定时任务调度使用。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    #   status：目标状态，类型 `AutomationStatus`。
    #   next_run_at：下一次运行时间，类型 `datetime | None`；默认 `None`。
    # 返回：类型 `Automation`；返回 `await self.require(automation_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_require_row` → `next_run_at.astimezone(UTC).isoformat` →
    # `next_run_at.astimezone` → `_now`；另有 2 个调用点。
    # 分支与异常：
    #   当 `status not in allowed` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   数据库操作：UPDATE automations；连接与事务边界以 with/提交语句为准。
    async def update_status(
        self,
        automation_id: str,
        status: AutomationStatus,
        *,
        next_run_at: datetime | None = None,
    ) -> Automation:

        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            current = await _require_row(database, automation_id)
            allowed = _AUTOMATION_TRANSITIONS[current.status]
            if status not in allowed:
                raise ValueError(
                    f"invalid automation transition: {current.status.value} -> "
                    f"{status.value}"
                )
            await database.execute(
                """
                UPDATE automations
                SET status = ?, next_run_at = COALESCE(?, next_run_at),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    (
                        next_run_at.astimezone(UTC).isoformat()
                        if next_run_at is not None
                        else None
                    ),
                    _now(),
                    automation_id,
                ),
            )
            await database.commit()
        return await self.require(automation_id)

    # 函数说明：SQLiteAutomationStore.set_next_run_at
    # 用途：设置运行，供定时任务调度使用。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    #   next_run_at：下一次运行时间，类型 `datetime`。
    # 返回：类型 `Automation`；返回 `await self.require(automation_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_require_row` → `next_run_at.astimezone(UTC).isoformat` →
    # `next_run_at.astimezone` → `_now`；另有 2 个调用点。
    # 副作用与资源：
    #   数据库操作：UPDATE automations；连接与事务边界以 with/提交语句为准。
    async def set_next_run_at(
        self,
        automation_id: str,
        next_run_at: datetime,
    ) -> Automation:

        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            await _require_row(database, automation_id)
            await database.execute(
                """
                UPDATE automations
                SET next_run_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (next_run_at.astimezone(UTC).isoformat(), _now(), automation_id),
            )
            await database.commit()
        return await self.require(automation_id)

    # 函数说明：SQLiteAutomationStore.mark_triggered
    # 用途：标记`triggered`，供定时任务调度使用。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    #   last_run_id：运行标识，类型 `str`；读取键 `last_run_id`。
    #   last_run_at：运行输入或配置值，类型 `datetime`。
    #   next_run_at：下一次运行时间，类型 `datetime | None`；默认 `None`。
    # 返回：类型 `Automation`；返回 `await self.require(automation_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_require_row` → `_required_identifier` →
    # `last_run_at.astimezone(UTC).isoformat` → `last_run_at.astimezone`；另有 5 个调用
    # 点。
    # 副作用与资源：
    #   数据库操作：UPDATE automations；连接与事务边界以 with/提交语句为准。
    async def mark_triggered(
        self,
        automation_id: str,
        *,
        last_run_id: str,
        last_run_at: datetime,
        next_run_at: datetime | None = None,
    ) -> Automation:

        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            await _require_row(database, automation_id)
            await database.execute(
                """
                UPDATE automations
                SET last_run_id = ?, last_run_at = ?,
                    next_run_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    _required_identifier(last_run_id, "last_run_id"),
                    last_run_at.astimezone(UTC).isoformat(),
                    (
                        next_run_at.astimezone(UTC).isoformat()
                        if next_run_at is not None
                        else None
                    ),
                    _now(),
                    automation_id,
                ),
            )
            await database.commit()
        return await self.require(automation_id)

    # 函数说明：SQLiteAutomationStore.require
    # 用途：获取并校验必需的SQLiteAutomationStore，供定时任务调度使用。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    # 返回：类型 `Automation`；返回 `automation`。
    # 分支与异常：
    #   当 `automation is None` 时，抛出
    # `KeyError(f'Automation 不存在：{automation_id}')`。
    async def require(self, automation_id: str) -> Automation:
        automation = await self.get(automation_id)
        if automation is None:
            raise KeyError(f"Automation 不存在：{automation_id}")
        return automation

    # 函数说明：SQLiteAutomationStore._connect
    # 用途：连接SQLiteAutomationStore，供定时任务调度使用。
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
# 用途：获取并校验必需的数据库行，供定时任务调度使用。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
#   automation_id：自动化任务标识，类型 `str`；读取键 `automation_id`。
# 返回：类型 `Automation`；返回 `_automation_from_row(row)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute` →
# `_required_identifier` → `cursor.fetchone` → `_automation_from_row`。
# 分支与异常：
#   当 `row is None` 时，抛出 `KeyError(f'Automation 不存在：{automation_id}')`。
# 副作用与资源：
#   数据库操作：SELECT automations；连接与事务边界以 with/提交语句为准。
async def _require_row(
    database: aiosqlite.Connection,
    automation_id: str,
) -> Automation:
    cursor = await database.execute(
        "SELECT * FROM automations WHERE id = ?",
        (_required_identifier(automation_id, "automation_id"),),
    )
    row = await cursor.fetchone()
    if row is None:
        raise KeyError(f"Automation 不存在：{automation_id}")
    return _automation_from_row(row)


# 函数说明：_automation_from_row
# 用途：将数据库行解析为自动化任务及其调度字段。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `id`、`title`、`prompt`
# 、`conversation_id`、`status`、`schedule_json`、`next_run_at`、`last_run_at`、
# `last_run_id`、`created_at`、`updated_at`。
# 返回：类型 `Automation`；返回 `Automation(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Automation` → `AutomationStatus` →
# `Schedule.model_validate_json` → `_parse_datetime`。
def _automation_from_row(row: aiosqlite.Row) -> Automation:
    return Automation(
        id=row["id"],
        title=row["title"] or "",
        prompt=row["prompt"],
        conversation_id=row["conversation_id"],
        status=AutomationStatus(row["status"]),
        schedule=Schedule.model_validate_json(row["schedule_json"]),
        next_run_at=(
            _parse_datetime(row["next_run_at"])
            if row["next_run_at"] is not None
            else None
        ),
        last_run_at=(
            _parse_datetime(row["last_run_at"])
            if row["last_run_at"] is not None
            else None
        ),
        last_run_id=row["last_run_id"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


# 函数说明：_parse_datetime
# 用途：解析日期时间，供定时任务调度使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `datetime`；返回 `datetime.fromisoformat(value)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.fromisoformat`。
def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


__all__ = ["SQLiteAutomationStore"]
