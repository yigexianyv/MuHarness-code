"""Trace SQL operations bound to a connection owned by the store.

These helpers never open connections or commit transactions. Run updates keep
SQLite's COALESCE/MAX semantics, including updates after an ignored event insert.
"""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

from ._projection import _RunUpdate
from .models import RunStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    provider TEXT,
    model TEXT,
    steps INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    event_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agent_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    type TEXT NOT NULL,
    event_time TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(run_id, sequence),
    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_started_at
ON agent_runs(started_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation
ON agent_runs(conversation_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_events_run_sequence
ON agent_events(run_id, sequence);
"""


@dataclass(frozen=True)
class _TraceQuery:
    statement: str
    parameters: tuple[object, ...]


class _TraceQueries:
    # 函数说明：_TraceQueries.__init__
    # 用途：初始化 _TraceQueries；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._database`。
    def __init__(self, database: aiosqlite.Connection) -> None:
        self._database = database

    # 函数说明：_TraceQueries.ensure_run
    # 用途：确保运行，供执行轨迹与用量查询使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   event_time：事件时间输入或配置值，类型 `str`。
    #   provider：模型或搜索服务商，类型 `str | None`。
    #   model：模型名称，类型 `str | None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute`。
    # 副作用与资源：
    #   数据库操作：INSERT agent_runs；连接与事务边界以 with/提交语句为准。
    async def ensure_run(
        self,
        *,
        run_id: str,
        conversation_id: str | None,
        event_time: str,
        provider: str | None,
        model: str | None,
    ) -> None:
        await self._database.execute(
            """
                INSERT OR IGNORE INTO agent_runs (
                    run_id,
                    conversation_id,
                    status,
                    started_at,
                    provider,
                    model
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
            (
                run_id,
                conversation_id,
                RunStatus.RUNNING.value,
                event_time,
                provider,
                model,
            ),
        )

    # 函数说明：_TraceQueries.append_event
    # 用途：追加事件，供执行轨迹与用量查询使用。
    # 参数：
    #   event_id：事件标识，类型 `str`。
    #   run_id：目标运行标识，类型 `str`。
    #   sequence：事件或记录顺序号，类型 `int`。
    #   event_type：事件输入或配置值，类型 `str`。
    #   event_time：事件时间输入或配置值，类型 `str`。
    #   payload_json：载荷JSON 数据输入或配置值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute`。
    # 副作用与资源：
    #   数据库操作：INSERT agent_events；连接与事务边界以 with/提交语句为准。
    async def append_event(
        self,
        *,
        event_id: str,
        run_id: str,
        sequence: int,
        event_type: str,
        event_time: str,
        payload_json: str,
    ) -> None:
        await self._database.execute(
            """
                INSERT OR IGNORE INTO agent_events (
                    event_id,
                    run_id,
                    sequence,
                    type,
                    event_time,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
            (
                event_id,
                run_id,
                sequence,
                event_type,
                event_time,
                payload_json,
            ),
        )

    # 函数说明：_TraceQueries.update_run
    # 用途：更新运行，供执行轨迹与用量查询使用。
    # 参数：
    #   update：`update`输入或配置值，类型 `_RunUpdate`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute`。
    # 副作用与资源：
    #   数据库操作：UPDATE agent_runs/agent_events；连接与事务边界以 with/提交语句为准。
    async def update_run(self, update: _RunUpdate) -> None:
        await self._database.execute(
            """
            UPDATE agent_runs
            SET
                conversation_id = COALESCE(?, conversation_id),
                status = COALESCE(?, status),
                completed_at = COALESCE(?, completed_at),
                provider = COALESCE(?, provider),
                model = COALESCE(?, model),
                steps = MAX(steps, ?),
                stop_reason = COALESCE(?, stop_reason),
                input_tokens = MAX(input_tokens, ?),
                output_tokens = MAX(output_tokens, ?),
                total_tokens = MAX(total_tokens, ?),
                event_count = (
                    SELECT COUNT(*) FROM agent_events WHERE run_id = ?
                )
            WHERE run_id = ?
            """,
            (
                update.conversation_id,
                update.status.value if update.status else None,
                update.completed_at,
                update.provider,
                update.model,
                update.steps,
                update.stop_reason.value if update.stop_reason else None,
                update.input_tokens,
                update.output_tokens,
                update.total_tokens,
                update.run_id,
                update.run_id,
            ),
        )

    # 函数说明：_TraceQueries.find_run
    # 用途：运行`find`，供执行轨迹与用量查询使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `aiosqlite.Row | None`；返回 `await cursor.fetchone()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute` →
    # `cursor.fetchone`。
    # 副作用与资源：
    #   数据库操作：SELECT agent_runs；连接与事务边界以 with/提交语句为准。
    async def find_run(self, run_id: str) -> aiosqlite.Row | None:
        cursor = await self._database.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?",
            (run_id,),
        )
        return await cursor.fetchone()

    # 函数说明：_TraceQueries.prefix_runs
    # 用途：在执行轨迹与用量查询中处理 `prefix_runs`，通过 `self._database.execute` 完成
    # 首个内部处理步骤。
    # 参数：
    #   prefix：`prefix`输入或配置值，类型 `str`。
    # 返回：类型 `list[aiosqlite.Row]`；返回 `await cursor.fetchall()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute` →
    # `cursor.fetchall`。
    # 副作用与资源：
    #   数据库操作：SELECT agent_runs；连接与事务边界以 with/提交语句为准。
    async def prefix_runs(self, prefix: str) -> list[aiosqlite.Row]:
        cursor = await self._database.execute(
            """
                SELECT * FROM agent_runs
                WHERE run_id LIKE ?
                ORDER BY started_at DESC
                LIMIT 2
                """,
            (f"{prefix}%",),
        )
        return await cursor.fetchall()

    # 函数说明：_TraceQueries.prepare_list_runs
    # 用途：准备列表运行集合，供执行轨迹与用量查询使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   limit：本次返回或处理的数量上限，类型 `int`。
    # 返回：类型 `_TraceQuery`；返回 `_TraceQuery(query, parameters)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_TraceQuery`。
    @staticmethod
    def prepare_list_runs(
        *,
        conversation_id: str | None,
        limit: int,
    ) -> _TraceQuery:
        query = "SELECT * FROM agent_runs"
        parameters: tuple[object, ...]
        if conversation_id is None:
            parameters = (limit,)
        else:
            query += " WHERE conversation_id = ?"
            parameters = (conversation_id, limit)
        query += " ORDER BY started_at DESC LIMIT ?"
        return _TraceQuery(query, parameters)

    # 函数说明：_TraceQueries.list_runs
    # 用途：列出运行集合，供执行轨迹与用量查询使用。
    # 参数：
    #   query：检索查询文本，类型 `_TraceQuery`。
    # 返回：类型 `list[aiosqlite.Row]`；返回 `await cursor.fetchall()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute` →
    # `cursor.fetchall`。
    async def list_runs(self, query: _TraceQuery) -> list[aiosqlite.Row]:
        cursor = await self._database.execute(query.statement, query.parameters)
        return await cursor.fetchall()

    # 函数说明：_TraceQueries.events
    # 用途：在执行轨迹与用量查询中处理 `events`，通过 `self._database.execute` 完成首个
    # 内部处理步骤。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `list[aiosqlite.Row]`；返回 `await cursor.fetchall()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute` →
    # `cursor.fetchall`。
    # 副作用与资源：
    #   数据库操作：SELECT agent_events；连接与事务边界以 with/提交语句为准。
    async def events(self, run_id: str) -> list[aiosqlite.Row]:
        cursor = await self._database.execute(
            """
                SELECT payload_json FROM agent_events
                WHERE run_id = ?
                ORDER BY sequence ASC
                """,
            (run_id,),
        )
        return await cursor.fetchall()

    # 函数说明：_TraceQueries.prepare_conversation_event_payloads
    # 用途：准备会话事件，供执行轨迹与用量查询使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`。
    # 返回：类型 `_TraceQuery`；返回 `_TraceQuery(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_TraceQuery`。
    @staticmethod
    def prepare_conversation_event_payloads(
        conversation_id: str,
        run_ids: tuple[str, ...],
    ) -> _TraceQuery:
        where = "r.conversation_id = ?"
        parameters: list[object] = [conversation_id]
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            where += f" OR r.run_id IN ({placeholders})"
            parameters.extend(run_ids)
        return _TraceQuery(
            f"""
                SELECT e.payload_json
                FROM agent_events AS e
                JOIN agent_runs AS r ON r.run_id = e.run_id
                WHERE {where}
                ORDER BY r.started_at ASC, e.sequence ASC
                """,
            tuple(parameters),
        )

    # 函数说明：_TraceQueries.conversation_event_payloads
    # 用途：在执行轨迹与用量查询中处理 `conversation_event_payloads`，通过
    # `self._database.execute` 完成首个内部处理步骤。
    # 参数：
    #   query：检索查询文本，类型 `_TraceQuery`。
    # 返回：类型 `list[aiosqlite.Row]`；返回 `await cursor.fetchall()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute` →
    # `cursor.fetchall`。
    async def conversation_event_payloads(
        self,
        query: _TraceQuery,
    ) -> list[aiosqlite.Row]:
        cursor = await self._database.execute(query.statement, query.parameters)
        return await cursor.fetchall()

    # 函数说明：_TraceQueries.prepare_delete_conversation
    # 用途：准备会话，供执行轨迹与用量查询使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`。
    # 返回：类型 `_TraceQuery`；返回 `_TraceQuery(query, tuple(parameters))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_TraceQuery`。
    @staticmethod
    def prepare_delete_conversation(
        conversation_id: str,
        run_ids: tuple[str, ...],
    ) -> _TraceQuery:
        query = "DELETE FROM agent_runs WHERE conversation_id = ?"
        parameters: list[object] = [conversation_id]
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            query += f" OR run_id IN ({placeholders})"
            parameters.extend(run_ids)
        return _TraceQuery(query, tuple(parameters))

    # 函数说明：_TraceQueries.delete_conversation
    # 用途：删除会话，供执行轨迹与用量查询使用。
    # 参数：
    #   query：检索查询文本，类型 `_TraceQuery`。
    # 返回：类型 `aiosqlite.Cursor`；返回
    # `await self._database.execute(query.statement, query.parameters)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute`。
    # 副作用与资源：
    #   数据库操作：PRAGMA；连接与事务边界以 with/提交语句为准。
    async def delete_conversation(self, query: _TraceQuery) -> aiosqlite.Cursor:
        await self._database.execute("PRAGMA foreign_keys = ON")
        return await self._database.execute(query.statement, query.parameters)

    # 函数说明：_TraceQueries.delete_run
    # 用途：删除运行，供执行轨迹与用量查询使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `aiosqlite.Cursor`；返回 `await self._database.execute('DELETE FROM
    # agent_runs WHERE run_id = ?', (run_id,))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute`。
    # 副作用与资源：
    #   数据库操作：DELETE agent_runs；连接与事务边界以 with/提交语句为准。
    async def delete_run(self, run_id: str) -> aiosqlite.Cursor:
        return await self._database.execute(
            "DELETE FROM agent_runs WHERE run_id = ?",
            (run_id,),
        )

    # 函数说明：_TraceQueries.next_sequence_row
    # 用途：在执行轨迹与用量查询中处理 `next_sequence_row`，通过
    # `self._database.execute` 完成首个内部处理步骤。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `aiosqlite.Row`；返回 `await cursor.fetchone()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._database.execute` →
    # `cursor.fetchone`。
    # 副作用与资源：
    #   数据库操作：SELECT agent_events；连接与事务边界以 with/提交语句为准。
    async def next_sequence_row(self, run_id: str) -> aiosqlite.Row:
        cursor = await self._database.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 FROM agent_events "
            "WHERE run_id = ?",
            (run_id,),
        )
        return await cursor.fetchone()
