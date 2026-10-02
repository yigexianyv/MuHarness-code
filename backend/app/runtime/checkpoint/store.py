
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH
from app.models.types import Message, MessageRole, ToolCall, ToolResult
from app.runtime.agent.result import AgentStopReason

from .models import CheckpointPhase, CheckpointStatus, RunCheckpoint

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_checkpoints (
    run_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    user_message_json TEXT NOT NULL,
    status TEXT NOT NULL,
    phase TEXT NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    pending_tool_calls_json TEXT NOT NULL DEFAULT '[]',
    completed_tool_results_json TEXT NOT NULL DEFAULT '[]',
    stop_reason TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    recovered_by_run_id TEXT,
    revision INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_run_checkpoints_conversation_updated
ON run_checkpoints(conversation_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_run_checkpoints_status
ON run_checkpoints(status, updated_at DESC);
"""


class SQLiteCheckpointStore:

    # 函数说明：SQLiteCheckpointStore.__init__
    # 用途：初始化 SQLiteCheckpointStore；参数及实际保存的实例字段见下方说明。
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

    # 函数说明：SQLiteCheckpointStore.initialize
    # 用途：初始化SQLiteCheckpointStore，供运行检查点持久化使用。
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

    # 函数说明：SQLiteCheckpointStore.start
    # 用途：创建运行检查点，记录恢复所需的初始状态。
    # 参数：
    #   run_id：目标运行标识，类型 `str`；读取键 `run_id`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   user_message：当前用户消息，类型 `Message`。
    # 返回：类型 `RunCheckpoint`；返回 `await self._require(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now(UTC).isoformat` →
    # `datetime.now` → `self._connect` → `database.execute` → `_required_identifier` →
    # `_optional_identifier`；另有 3 个调用点。
    # 分支与异常：
    #   当 `user_message.role is not MessageRole.USER` 时，抛出
    # `ValueError('checkpoint user_message must have user role')`。
    #   捕获 `aiosqlite.IntegrityError` 后，转换或抛出
    # `ValueError(f'Checkpoint 已存在：{run_id}')`。
    # 副作用与资源：
    #   数据库操作：INSERT run_checkpoints；连接与事务边界以 with/提交语句为准。
    async def start(
        self,
        run_id: str,
        *,
        conversation_id: str | None,
        user_message: Message,
    ) -> RunCheckpoint:

        """创建运行检查点，记录恢复所需的初始状态。"""
        if user_message.role is not MessageRole.USER:
            raise ValueError("checkpoint user_message must have user role")
        now = datetime.now(UTC).isoformat()
        async with self._connect() as database:
            try:
                await database.execute(
                    """
                    INSERT INTO run_checkpoints (
                        run_id, conversation_id, user_message_json, status, phase,
                        started_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _required_identifier(run_id, "run_id"),
                        _optional_identifier(conversation_id),
                        user_message.model_dump_json(),
                        CheckpointStatus.RUNNING.value,
                        CheckpointPhase.STARTING.value,
                        now,
                        now,
                    ),
                )
            except aiosqlite.IntegrityError as exc:
                raise ValueError(f"Checkpoint 已存在：{run_id}") from exc
            await database.commit()
        return await self._require(run_id)

    # 函数说明：SQLiteCheckpointStore.before_model
    # 用途：在模型调用前保存消息与摘要状态。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   step：当前任务步骤，类型 `int`。
    # 返回：类型 `RunCheckpoint`；返回 `await self._update_running(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update_running`。
    async def before_model(self, run_id: str, *, step: int) -> RunCheckpoint:

        """在模型调用前保存消息与摘要状态。"""
        return await self._update_running(
            run_id,
            phase=CheckpointPhase.MODEL_REQUEST,
            step=step,
            pending_tool_calls=(),
        )

    # 函数说明：SQLiteCheckpointStore.before_tools
    # 用途：在工具轮次前保存待执行调用，供中断后恢复。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   step：当前任务步骤，类型 `int`。
    #   tool_calls：待执行的结构化工具调用，类型 `Sequence[ToolCall]`。
    # 返回：类型 `RunCheckpoint`；返回 `await self._update_running(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update_running`。
    # 分支与异常：
    #   当 `not tool_calls` 时，抛出
    # `ValueError('before_tools requires at least one tool call')`。
    async def before_tools(
        self,
        run_id: str,
        *,
        step: int,
        tool_calls: Sequence[ToolCall],
    ) -> RunCheckpoint:

        """在工具轮次前保存待执行调用，供中断后恢复。"""
        if not tool_calls:
            raise ValueError("before_tools requires at least one tool call")
        return await self._update_running(
            run_id,
            phase=CheckpointPhase.TOOL_EXECUTION,
            step=step,
            pending_tool_calls=tuple(tool_calls),
        )

    # 函数说明：SQLiteCheckpointStore.complete_tool
    # 用途：记录单个工具调用完成情况，避免恢复时重复执行。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    # 返回：类型 `RunCheckpoint`；返回 `await self._require(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_require_row` → `_require_running` → `_write_progress` →
    # `database.commit`；另有 1 个调用点。
    # 分支与异常：
    #   当 `len(matching) != 1` 时，抛出 `ValueError(…)`。
    async def complete_tool(
        self,
        run_id: str,
        result: ToolResult,
    ) -> RunCheckpoint:

        """记录单个工具调用完成情况，避免恢复时重复执行。"""
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            checkpoint = await _require_row(database, run_id)
            _require_running(checkpoint)
            pending = list(checkpoint.pending_tool_calls)
            matching = [
                call for call in pending if call.id == result.tool_call_id
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"工具调用不在 Checkpoint 待执行集合中：{result.tool_call_id}"
                )
            pending = [call for call in pending if call.id != result.tool_call_id]
            completed = (*checkpoint.completed_tool_results, result)
            phase = (
                CheckpointPhase.TOOL_EXECUTION
                if pending
                else CheckpointPhase.TOOL_RESULTS_READY
            )
            await _write_progress(
                database,
                checkpoint,
                phase=phase,
                step=checkpoint.step,
                pending_tool_calls=pending,
                completed_tool_results=completed,
            )
            await database.commit()
        return await self._require(run_id)

    # 函数说明：SQLiteCheckpointStore.complete
    # 用途：完成SQLiteCheckpointStore，供运行检查点持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   stop_reason：运行终止原因，类型 `AgentStopReason`。
    # 返回：类型 `RunCheckpoint`；返回 `await self._finish(run_id, status=
    # CheckpointStatus.COMPLETED, stop_reason=stop_reason)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._finish`。
    async def complete(
        self,
        run_id: str,
        *,
        stop_reason: AgentStopReason,
    ) -> RunCheckpoint:
        return await self._finish(
            run_id,
            status=CheckpointStatus.COMPLETED,
            stop_reason=stop_reason,
        )

    # 函数说明：SQLiteCheckpointStore.fail
    # 用途：记录失败SQLiteCheckpointStore，供运行检查点持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   stop_reason：运行终止原因，类型 `AgentStopReason`。
    #   error：异常或错误信息，类型 `str | None`。
    # 返回：类型 `RunCheckpoint`；返回 `await self._finish(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._finish`。
    async def fail(
        self,
        run_id: str,
        *,
        stop_reason: AgentStopReason,
        error: str | None,
    ) -> RunCheckpoint:
        return await self._finish(
            run_id,
            status=CheckpointStatus.FAILED,
            stop_reason=stop_reason,
            error=error,
        )

    # 函数说明：SQLiteCheckpointStore.interrupt
    # 用途：中断SQLiteCheckpointStore，供运行检查点持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    # 返回：类型 `RunCheckpoint`；返回 `await self._finish(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._finish`。
    async def interrupt(
        self,
        run_id: str,
        *,
        error: str | None = None,
    ) -> RunCheckpoint:

        return await self._finish(
            run_id,
            status=CheckpointStatus.INTERRUPTED,
            error=error,
            preserve_phase=True,
        )

    # 函数说明：SQLiteCheckpointStore.recover_running
    # 用途：将进程退出时仍标为运行中的检查点转为可恢复状态。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`；读取键
    # `conversation_id`。
    # 返回：类型 `tuple[RunCheckpoint, ...]`；返回 `tuple(recovered)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier` →
    # `self._connect` → `database.execute` → `cursor.fetchall` → `self.interrupt`。
    async def recover_running(
        self,
        *,
        conversation_id: str | None = None,
    ) -> tuple[RunCheckpoint, ...]:

        """将进程退出时仍标为运行中的检查点转为可恢复状态。"""
        query = "SELECT run_id FROM run_checkpoints WHERE status = ?"
        parameters: list[object] = [CheckpointStatus.RUNNING.value]
        if conversation_id is not None:
            query += " AND conversation_id = ?"
            parameters.append(_required_identifier(conversation_id, "conversation_id"))
        async with self._connect() as database:
            cursor = await database.execute(query, tuple(parameters))
            run_ids = [row["run_id"] for row in await cursor.fetchall()]
        recovered: list[RunCheckpoint] = []
        for run_id in run_ids:
            recovered.append(
                await self.interrupt(
                    run_id,
                    error="process ended before Run reached a terminal state",
                )
            )
        return tuple(recovered)

    # 函数说明：SQLiteCheckpointStore.latest_unrecovered
    # 用途：获取最新的`unrecovered`，供运行检查点持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    # 返回：类型 `RunCheckpoint | None`；返回
    # `_checkpoint_from_row(row) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_required_identifier` → `cursor.fetchone` →
    # `_checkpoint_from_row`。
    # 副作用与资源：
    #   数据库操作：SELECT run_checkpoints；连接与事务边界以 with/提交语句为准。
    async def latest_unrecovered(
        self,
        conversation_id: str,
    ) -> RunCheckpoint | None:

        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM run_checkpoints
                WHERE conversation_id = ?
                  AND status = ?
                  AND recovered_by_run_id IS NULL
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (
                    _required_identifier(conversation_id, "conversation_id"),
                    CheckpointStatus.INTERRUPTED.value,
                ),
            )
            row = await cursor.fetchone()
        return _checkpoint_from_row(row) if row is not None else None

    # 函数说明：SQLiteCheckpointStore.get_unrecovered
    # 用途：获取`unrecovered`，供运行检查点持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`；读取键 `run_id`。
    # 返回：类型 `RunCheckpoint | None`；返回
    # `_checkpoint_from_row(row) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_required_identifier` → `cursor.fetchone` →
    # `_checkpoint_from_row`。
    # 副作用与资源：
    #   数据库操作：SELECT run_checkpoints；连接与事务边界以 with/提交语句为准。
    async def get_unrecovered(self, run_id: str) -> RunCheckpoint | None:

        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM run_checkpoints
                WHERE run_id = ?
                  AND status = ?
                  AND recovered_by_run_id IS NULL
                LIMIT 1
                """,
                (
                    _required_identifier(run_id, "run_id"),
                    CheckpointStatus.INTERRUPTED.value,
                ),
            )
            row = await cursor.fetchone()
        return _checkpoint_from_row(row) if row is not None else None

    # 函数说明：SQLiteCheckpointStore.mark_recovered
    # 用途：标记`recovered`，供运行检查点持久化使用。
    # 参数：
    #   interrupted_run_id：运行标识，类型 `str`。
    #   recovered_by_run_id：运行标识，类型 `str`；读取键 `recovered_by_run_id`。
    # 返回：类型 `RunCheckpoint`；返回 `await self._require(interrupted_run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` → `_require_row` →
    #  `database.execute` → `_required_identifier` → `datetime.now(UTC).isoformat` →
    # `datetime.now`；另有 2 个调用点。
    # 分支与异常：
    #   当 `checkpoint.status is not CheckpointStatus.INTERRUPTED` 时，抛出
    # `ValueError('only interrupted checkpoint can be recovered')`。
    # 副作用与资源：
    #   数据库操作：UPDATE run_checkpoints；连接与事务边界以 with/提交语句为准。
    async def mark_recovered(
        self,
        interrupted_run_id: str,
        *,
        recovered_by_run_id: str,
    ) -> RunCheckpoint:

        async with self._connect() as database:
            checkpoint = await _require_row(database, interrupted_run_id)
            if checkpoint.status is not CheckpointStatus.INTERRUPTED:
                raise ValueError("only interrupted checkpoint can be recovered")
            await database.execute(
                """
                UPDATE run_checkpoints
                SET recovered_by_run_id = ?, updated_at = ?, revision = revision + 1
                WHERE run_id = ?
                """,
                (
                    _required_identifier(recovered_by_run_id, "recovered_by_run_id"),
                    datetime.now(UTC).isoformat(),
                    interrupted_run_id,
                ),
            )
            await database.commit()
        return await self._require(interrupted_run_id)

    # 函数说明：SQLiteCheckpointStore.get
    # 用途：获取SQLiteCheckpointStore，供运行检查点持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`；读取键 `run_id`。
    # 返回：类型 `RunCheckpoint | None`；返回
    # `_checkpoint_from_row(row) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_required_identifier` → `cursor.fetchone` →
    # `_checkpoint_from_row`。
    # 副作用与资源：
    #   数据库操作：SELECT run_checkpoints；连接与事务边界以 with/提交语句为准。
    async def get(self, run_id: str) -> RunCheckpoint | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM run_checkpoints WHERE run_id = ?",
                (_required_identifier(run_id, "run_id"),),
            )
            row = await cursor.fetchone()
        return _checkpoint_from_row(row) if row is not None else None

    # 函数说明：SQLiteCheckpointStore.list
    # 用途：列出SQLiteCheckpointStore，供运行检查点持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`；读取键
    # `conversation_id`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `20`。
    # 返回：类型 `tuple[RunCheckpoint, ...]`；返回
    # `tuple((_checkpoint_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier` →
    # `self._connect` → `database.execute` → `cursor.fetchall` → `_checkpoint_from_row`
    # 。
    # 分支与异常：
    #   当 `limit < 1` 时，抛出 `ValueError('limit must be at least 1')`。
    async def list(
        self,
        *,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> tuple[RunCheckpoint, ...]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        query = "SELECT * FROM run_checkpoints"
        parameters: list[object] = []
        if conversation_id is not None:
            query += " WHERE conversation_id = ?"
            parameters.append(_required_identifier(conversation_id, "conversation_id"))
        query += " ORDER BY updated_at DESC LIMIT ?"
        parameters.append(limit)
        async with self._connect() as database:
            cursor = await database.execute(query, tuple(parameters))
            rows = await cursor.fetchall()
        return tuple(_checkpoint_from_row(row) for row in rows)

    # 函数说明：SQLiteCheckpointStore.delete_for_conversation
    # 用途：删除会话，供运行检查点持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`；读取键 `conversation_id`。
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`；默认 `()`。
    # 返回：类型 `int`；返回 `max(cursor.rowcount, 0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier` →
    # `self._connect` → `database.execute` → `database.commit`。
    async def delete_for_conversation(
        self,
        conversation_id: str,
        *,
        run_ids: tuple[str, ...] = (),
    ) -> int:

        normalized = _required_identifier(conversation_id, "conversation_id")
        query = "DELETE FROM run_checkpoints WHERE conversation_id = ?"
        parameters: list[object] = [normalized]
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            query += f" OR run_id IN ({placeholders})"
            parameters.extend(run_ids)
        async with self._connect() as database:
            cursor = await database.execute(query, tuple(parameters))
            await database.commit()
        return max(cursor.rowcount, 0)

    # 函数说明：SQLiteCheckpointStore._update_running
    # 用途：更新`running`，供运行检查点持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   phase：`phase`输入或配置值，类型 `CheckpointPhase`。
    #   step：当前任务步骤，类型 `int`。
    #   pending_tool_calls：待处理项工具调用集合输入或配置值，类型 `Sequence[ToolCall]`
    # 。
    # 返回：类型 `RunCheckpoint`；返回 `await self._require(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_require_row` → `_require_running` → `_write_progress` →
    # `database.commit`；另有 1 个调用点。
    # 分支与异常：
    #   当 `step < 1` 时，抛出 `ValueError('checkpoint step must be at least 1')`。
    #   当 `checkpoint.pending_tool_calls` 时，抛出 `ValueError(…)`。
    async def _update_running(
        self,
        run_id: str,
        *,
        phase: CheckpointPhase,
        step: int,
        pending_tool_calls: Sequence[ToolCall],
    ) -> RunCheckpoint:
        if step < 1:
            raise ValueError("checkpoint step must be at least 1")
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            checkpoint = await _require_row(database, run_id)
            _require_running(checkpoint)
            if checkpoint.pending_tool_calls:
                raise ValueError(
                    "cannot advance checkpoint while tool calls are pending"
                )
            await _write_progress(
                database,
                checkpoint,
                phase=phase,
                step=step,
                pending_tool_calls=pending_tool_calls,
                completed_tool_results=checkpoint.completed_tool_results,
            )
            await database.commit()
        return await self._require(run_id)

    # 函数说明：SQLiteCheckpointStore._finish
    # 用途：结束SQLiteCheckpointStore，供运行检查点持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   status：目标状态，类型 `CheckpointStatus`。
    #   stop_reason：运行终止原因，类型 `AgentStopReason | None`；默认 `None`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    #   preserve_phase：`preserve_phase`输入或配置值，类型 `bool`；默认 `False`。
    # 返回：类型 `RunCheckpoint`；返回 `await self._require(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_require_row` → `_require_running` →
    # `datetime.now(UTC).isoformat` → `datetime.now`；另有 2 个调用点。
    # 分支与异常：
    #   当 `status is CheckpointStatus.COMPLETED and…` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   数据库操作：UPDATE run_checkpoints；连接与事务边界以 with/提交语句为准。
    async def _finish(
        self,
        run_id: str,
        *,
        status: CheckpointStatus,
        stop_reason: AgentStopReason | None = None,
        error: str | None = None,
        preserve_phase: bool = False,
    ) -> RunCheckpoint:
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            checkpoint = await _require_row(database, run_id)
            _require_running(checkpoint)
            if (
                status is CheckpointStatus.COMPLETED
                and checkpoint.pending_tool_calls
            ):
                raise ValueError(
                    "completed checkpoint cannot contain pending tool calls"
                )
            now = datetime.now(UTC).isoformat()
            await database.execute(
                """
                UPDATE run_checkpoints
                SET status = ?, phase = ?, stop_reason = ?, error = ?,
                    updated_at = ?, completed_at = ?, revision = revision + 1
                WHERE run_id = ?
                """,
                (
                    status.value,
                    (
                        checkpoint.phase.value
                        if preserve_phase
                        else CheckpointPhase.FINISHED.value
                    ),
                    stop_reason.value if stop_reason else None,
                    error,
                    now,
                    now,
                    run_id,
                ),
            )
            await database.commit()
        return await self._require(run_id)

    # 函数说明：SQLiteCheckpointStore._require
    # 用途：获取并校验必需的SQLiteCheckpointStore，供运行检查点持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `RunCheckpoint`；返回 `checkpoint`。
    # 分支与异常：
    #   当 `checkpoint is None` 时，抛出 `KeyError(f'Checkpoint 不存在：{run_id}')`。
    async def _require(self, run_id: str) -> RunCheckpoint:
        checkpoint = await self.get(run_id)
        if checkpoint is None:
            raise KeyError(f"Checkpoint 不存在：{run_id}")
        return checkpoint

    # 函数说明：SQLiteCheckpointStore._connect
    # 用途：连接SQLiteCheckpointStore，供运行检查点持久化使用。
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


# 函数说明：_require_row
# 用途：获取并校验必需的数据库行，供运行检查点持久化使用。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
#   run_id：目标运行标识，类型 `str`；读取键 `run_id`。
# 返回：类型 `RunCheckpoint`；返回 `_checkpoint_from_row(row)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute` →
# `_required_identifier` → `cursor.fetchone` → `_checkpoint_from_row`。
# 分支与异常：
#   当 `row is None` 时，抛出 `KeyError(f'Checkpoint 不存在：{run_id}')`。
# 副作用与资源：
#   数据库操作：SELECT run_checkpoints；连接与事务边界以 with/提交语句为准。
async def _require_row(
    database: aiosqlite.Connection,
    run_id: str,
) -> RunCheckpoint:
    cursor = await database.execute(
        "SELECT * FROM run_checkpoints WHERE run_id = ?",
        (_required_identifier(run_id, "run_id"),),
    )
    row = await cursor.fetchone()
    if row is None:
        raise KeyError(f"Checkpoint 不存在：{run_id}")
    return _checkpoint_from_row(row)


# 函数说明：_write_progress
# 用途：写入`progress`，供运行检查点持久化使用。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
#   checkpoint：检查点输入或配置值，类型 `RunCheckpoint`。
#   phase：`phase`输入或配置值，类型 `CheckpointPhase`。
#   step：当前任务步骤，类型 `int`。
#   pending_tool_calls：传给 `_dump_models` 的输入，类型 `Sequence[ToolCall]`。
#   completed_tool_results：传给 `_dump_models` 的输入，类型 `Sequence[ToolResult]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute` → `_dump_models` →
# `datetime.now(UTC).isoformat` → `datetime.now`。
# 副作用与资源：
#   数据库操作：UPDATE run_checkpoints；连接与事务边界以 with/提交语句为准。
async def _write_progress(
    database: aiosqlite.Connection,
    checkpoint: RunCheckpoint,
    *,
    phase: CheckpointPhase,
    step: int,
    pending_tool_calls: Sequence[ToolCall],
    completed_tool_results: Sequence[ToolResult],
) -> None:
    await database.execute(
        """
        UPDATE run_checkpoints
        SET phase = ?, step = ?, pending_tool_calls_json = ?,
            completed_tool_results_json = ?, updated_at = ?,
            revision = revision + 1
        WHERE run_id = ?
        """,
        (
            phase.value,
            step,
            _dump_models(pending_tool_calls),
            _dump_models(completed_tool_results),
            datetime.now(UTC).isoformat(),
            checkpoint.run_id,
        ),
    )


# 函数说明：_require_running
# 用途：获取并校验必需的`running`，供运行检查点持久化使用。
# 参数：
#   checkpoint：检查点输入或配置值，类型 `RunCheckpoint`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   当 `checkpoint.status is not CheckpointStatus.RUNNING` 时，抛出
# `ValueError(f'Checkpoint 已结束，不能继续更新：{checkpoint.run_id}')`。
def _require_running(checkpoint: RunCheckpoint) -> None:
    if checkpoint.status is not CheckpointStatus.RUNNING:
        raise ValueError(
            f"Checkpoint 已结束，不能继续更新：{checkpoint.run_id}"
        )


# 函数说明：_checkpoint_from_row
# 用途：将数据库行解析为运行检查点及保存的恢复数据。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `run_id`、
# `conversation_id`、`user_message_json`、`status`、`phase`、`step`、
# `pending_tool_calls_json`、`completed_tool_results_json`、`stop_reason`、`error`、
# `started_at`、`updated_at`、`completed_at`、`recovered_by_run_id`、`revision`。
# 返回：类型 `RunCheckpoint`；返回 `RunCheckpoint(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RunCheckpoint` →
# `Message.model_validate_json` → `ToolCall.model_validate` → `json.loads` →
# `ToolResult.model_validate` → `datetime.fromisoformat`。
def _checkpoint_from_row(row: aiosqlite.Row) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=row["run_id"],
        conversation_id=row["conversation_id"],
        user_message=Message.model_validate_json(row["user_message_json"]),
        status=row["status"],
        phase=row["phase"],
        step=row["step"],
        pending_tool_calls=tuple(
            ToolCall.model_validate(item)
            for item in json.loads(row["pending_tool_calls_json"])
        ),
        completed_tool_results=tuple(
            ToolResult.model_validate(item)
            for item in json.loads(row["completed_tool_results_json"])
        ),
        stop_reason=row["stop_reason"],
        error=row["error"],
        started_at=datetime.fromisoformat(row["started_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        completed_at=(
            datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None
        ),
        recovered_by_run_id=row["recovered_by_run_id"],
        revision=row["revision"],
    )


# 函数说明：_dump_models
# 用途：返回 `json.dumps(…)`，提供 运行检查点持久化 的派生值。
# 参数：
#   models：模型输入或配置值，类型 `Sequence[ToolCall] | Sequence[ToolResult]`。
# 返回：类型 `str`；返回 `json.dumps(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
def _dump_models(models: Sequence[ToolCall] | Sequence[ToolResult]) -> str:
    return json.dumps(
        [model.model_dump(mode="json") for model in models],
        ensure_ascii=False,
        separators=(",", ":"),
    )


# 函数说明：_required_identifier
# 用途：校验必需的记录标识，拒绝缺失或不合要求的输入。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
#   field_name：待校验的字段名称，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 分支与异常：
#   当 `not isinstance(value, str)` 时，抛出
# `TypeError(f'{field_name} must be a string')`。
#   当 `not normalized` 时，抛出 `ValueError(f'{field_name} cannot be empty')`。
def _required_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


# 函数说明：_optional_identifier
# 用途：规范化可选标识，允许调用方省略该字段。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str | None`；读取键 `conversation_id`。
# 返回：类型 `str | None`；按分支返回 `None`；
# `_required_identifier(value, 'conversation_id')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_identifier`。
# 分支与异常：
#   当 `value is None` 时，返回 `None`。
def _optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    return _required_identifier(value, "conversation_id")


__all__ = ["SQLiteCheckpointStore"]
