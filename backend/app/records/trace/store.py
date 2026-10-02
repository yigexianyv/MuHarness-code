
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime as datetime
from pathlib import Path

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH
from app.runtime.agent.events import AgentEvent, AgentEventHandler, AgentEventType
from app.runtime.agent.result import AgentStopReason as AgentStopReason

from ._mapping import trace_from_row as _map_trace_row
from ._projection import project_event as _project_event
from ._queries import _SCHEMA, _TraceQueries
from .models import AgentRunTrace
from .models import RunStatus as RunStatus


class SQLiteTraceStore:
    """Coordinate persistence while retaining the existing transaction boundaries."""

    # 函数说明：SQLiteTraceStore.__init__
    # 用途：初始化 SQLiteTraceStore；参数及实际保存的实例字段见下方说明。
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

    # 函数说明：SQLiteTraceStore.initialize
    # 用途：初始化SQLiteTraceStore，供执行轨迹与用量查询使用。
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

    # 函数说明：SQLiteTraceStore.record_event
    # 用途：记录事件，供执行轨迹与用量查询使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`event.event_time.isoformat` →
    # `self._connect` → `_TraceQueries` → `queries.ensure_run` → `queries.append_event`
    # → `event.model_dump_json`；另有 2 个调用点。
    async def record_event(self, event: AgentEvent) -> None:

        event_time = event.event_time.isoformat()
        async with self._connect() as database:
            queries = _TraceQueries(database)
            await queries.ensure_run(
                run_id=event.run_id,
                conversation_id=event.conversation_id,
                event_time=event_time,
                provider=event.provider,
                model=event.model,
            )
            await queries.append_event(
                event_id=event.event_id,
                run_id=event.run_id,
                sequence=event.sequence,
                event_type=event.type.value,
                event_time=event_time,
                payload_json=event.model_dump_json(exclude_none=True),
            )
            await self._update_run(database, event)
            await database.commit()

    # 函数说明：SQLiteTraceStore.get
    # 用途：获取SQLiteTraceStore，供执行轨迹与用量查询使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `AgentRunTrace | None`；返回
    # `_trace_from_row(row) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `_TraceQueries(database).find_run` → `_TraceQueries` → `_trace_from_row`。
    async def get(self, run_id: str) -> AgentRunTrace | None:

        async with self._connect() as database:
            row = await _TraceQueries(database).find_run(run_id)
        return _trace_from_row(row) if row is not None else None

    # 函数说明：SQLiteTraceStore.resolve
    # 用途：解析或定位SQLiteTraceStore，供执行轨迹与用量查询使用。
    # 参数：
    #   identifier：待规范化的标识，类型 `str`。
    # 返回：类型 `AgentRunTrace | None`；按分支返回 `None`；`exact`；
    # `_trace_from_row(rows[0]) if rows else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `_TraceQueries(database).prefix_runs` → `_TraceQueries` → `_trace_from_row`。
    # 分支与异常：
    #   当 `not normalized` 时，返回 `None`。
    #   当 `exact is not None` 时，返回 `exact`。
    #   当 `len(rows) > 1` 时，抛出 `ValueError(f'Run ID 前缀不唯一：{identifier}')`。
    async def resolve(self, identifier: str) -> AgentRunTrace | None:

        normalized = identifier.strip()
        if not normalized:
            return None
        exact = await self.get(normalized)
        if exact is not None:
            return exact

        async with self._connect() as database:
            rows = await _TraceQueries(database).prefix_runs(normalized)
        if len(rows) > 1:
            raise ValueError(f"Run ID 前缀不唯一：{identifier}")
        return _trace_from_row(rows[0]) if rows else None

    # 函数说明：SQLiteTraceStore.list_runs
    # 用途：列出运行集合，供执行轨迹与用量查询使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `20`。
    # 返回：类型 `tuple[AgentRunTrace, ...]`；返回
    # `tuple((_trace_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_TraceQueries.prepare_list_runs`
    # → `self._connect` → `_TraceQueries(database).list_runs` → `_TraceQueries` →
    # `_trace_from_row`。
    # 分支与异常：
    #   当 `limit < 1` 时，抛出 `ValueError('limit must be at least 1')`。
    async def list_runs(
        self,
        *,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> tuple[AgentRunTrace, ...]:

        if limit < 1:
            raise ValueError("limit must be at least 1")
        query = _TraceQueries.prepare_list_runs(
            conversation_id=conversation_id,
            limit=limit,
        )
        async with self._connect() as database:
            rows = await _TraceQueries(database).list_runs(query)
        return tuple(_trace_from_row(row) for row in rows)

    # 函数说明：SQLiteTraceStore.load_events
    # 用途：加载事件序列，供执行轨迹与用量查询使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `tuple[AgentEvent, ...]`；返回
    # `tuple((AgentEvent.model_validate_json(row['payload_json']) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `_TraceQueries(database).events` → `_TraceQueries` →
    # `AgentEvent.model_validate_json`。
    # 分支与异常：
    #   当 `await self.get(run_id) is None` 时，抛出 `KeyError(f'Run 不存在：{run_id}')`
    # 。
    async def load_events(self, run_id: str) -> tuple[AgentEvent, ...]:

        if await self.get(run_id) is None:
            raise KeyError(f"Run 不存在：{run_id}")
        async with self._connect() as database:
            rows = await _TraceQueries(database).events(run_id)
        return tuple(
            AgentEvent.model_validate_json(row["payload_json"])
            for row in rows
        )

    # 函数说明：SQLiteTraceStore.load_event_payloads_for_conversation
    # 用途：加载事件会话，供执行轨迹与用量查询使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`；默认 `()`。
    # 返回：类型 `tuple[str, ...]`；返回 `tuple((row['payload_json'] for row in rows))`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `_TraceQueries.prepare_conversation_event_payloads` → `self._connect` →
    # `_TraceQueries(database).conversation_event_payloads` → `_TraceQueries`。
    # 分支与异常：
    #   当 `not normalized` 时，抛出 `ValueError('conversation_id cannot be empty')`。
    async def load_event_payloads_for_conversation(
        self,
        conversation_id: str,
        *,
        run_ids: tuple[str, ...] = (),
    ) -> tuple[str, ...]:

        normalized = conversation_id.strip()
        if not normalized:
            raise ValueError("conversation_id cannot be empty")
        query = _TraceQueries.prepare_conversation_event_payloads(normalized, run_ids)
        async with self._connect() as database:
            rows = await _TraceQueries(database).conversation_event_payloads(query)
        return tuple(row["payload_json"] for row in rows)

    # 函数说明：SQLiteTraceStore.delete_for_conversation
    # 用途：删除会话，供执行轨迹与用量查询使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`；默认 `()`。
    # 返回：类型 `int`；返回 `max(cursor.rowcount, 0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `_TraceQueries.prepare_delete_conversation` → `self._connect` →
    # `_TraceQueries(database).delete_conversation` → `_TraceQueries` →
    # `database.commit`。
    # 分支与异常：
    #   当 `not normalized` 时，抛出 `ValueError('conversation_id cannot be empty')`。
    async def delete_for_conversation(
        self,
        conversation_id: str,
        *,
        run_ids: tuple[str, ...] = (),
    ) -> int:

        normalized = conversation_id.strip()
        if not normalized:
            raise ValueError("conversation_id cannot be empty")
        query = _TraceQueries.prepare_delete_conversation(normalized, run_ids)
        async with self._connect() as database:
            cursor = await _TraceQueries(database).delete_conversation(query)
            await database.commit()
        return max(cursor.rowcount, 0)

    # 函数说明：SQLiteTraceStore.delete
    # 用途：删除SQLiteTraceStore，供执行轨迹与用量查询使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `bool`；返回 `cursor.rowcount > 0`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `_TraceQueries(database).delete_run` → `_TraceQueries` → `database.commit`。
    async def delete(self, run_id: str) -> bool:

        async with self._connect() as database:
            cursor = await _TraceQueries(database).delete_run(run_id)
            await database.commit()
        return cursor.rowcount > 0

    # 函数说明：SQLiteTraceStore.next_sequence
    # 用途：在执行轨迹与用量查询中处理 `next_sequence`，通过 `self._connect` 完成首个内
    # 部处理步骤。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `int`；返回 `int(row[0])`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `_TraceQueries(database).next_sequence_row` → `_TraceQueries`。
    async def next_sequence(self, run_id: str) -> int:

        async with self._connect() as database:
            row = await _TraceQueries(database).next_sequence_row(run_id)
        return int(row[0])

    # 函数说明：SQLiteTraceStore._update_run
    # 用途：更新运行，供执行轨迹与用量查询使用。
    # 参数：
    #   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `_TraceQueries(database).update_run` → `_TraceQueries` → `_project_event`。
    @staticmethod
    async def _update_run(
        database: aiosqlite.Connection,
        event: AgentEvent,
    ) -> None:
        await _TraceQueries(database).update_run(_project_event(event))

    # 函数说明：SQLiteTraceStore._connect
    # 用途：连接SQLiteTraceStore，供执行轨迹与用量查询使用。
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


class SQLiteTraceEventHandler(AgentEventHandler):

    # 函数说明：SQLiteTraceEventHandler.__init__
    # 用途：初始化 SQLiteTraceEventHandler；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteTraceStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.store`。
    def __init__(self, store: SQLiteTraceStore) -> None:
        self.store = store

    # 函数说明：SQLiteTraceEventHandler.emit
    # 用途：发出SQLiteTraceEventHandler，供执行轨迹与用量查询使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.store.record_event`。
    # 分支与异常：
    #   当 `event.type in (AgentEventType.MODEL_OUTPUT_DELTA,…` 时，返回 `None`。
    async def emit(self, event: AgentEvent) -> None:
        if event.type in (
            AgentEventType.MODEL_OUTPUT_DELTA,
            AgentEventType.MODEL_REASONING_DELTA,
        ):
            return
        await self.store.record_event(event)


# 函数说明：_trace_from_row
# 用途：将数据库行解析为执行轨迹记录。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`。
# 返回：类型 `AgentRunTrace`；返回 `_map_trace_row(row)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_map_trace_row`。
def _trace_from_row(row: aiosqlite.Row) -> AgentRunTrace:
    return _map_trace_row(row)
