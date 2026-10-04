"""步骤检查点：每次请求模型之前，记下"回到这一步之前"所需的全部状态。

写入时机是上下文准备（含压缩）完成之后、调用模型之前，所以这一行描述的是
"上一步已经做完、这一步还没决定"的时刻：
- message_count：到这一刻为止的原始消息数（继承历史 + 本次运行），用于截取会话前缀；
- summary / tool views：这一刻生效的摘要和工具结果视图，分支会话直接沿用；
- snapshot_id：这一刻的工作区文件快照；拍不了时 snapshot_error 说明原因。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from .snapshots import WorkspaceSnapshotStore

if TYPE_CHECKING:
    from app.runtime.context.summary import ConversationSummaryState
    from app.runtime.context.tool_views import ToolResultView

MAX_SNAPSHOT_STEPS = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_steps (
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL CHECK (step >= 1),
    message_count INTEGER NOT NULL CHECK (message_count >= 0),
    summary_json TEXT,
    tool_views_json TEXT NOT NULL DEFAULT '[]',
    snapshot_id TEXT,
    snapshot_error TEXT,
    constraints_revision INTEGER,
    task_context_text TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, step)
);

-- 发给模型的工具消息原文，按内容哈希去重（同一段摘录在多步里只存一份）
CREATE TABLE IF NOT EXISTS run_request_tool_contents (
    sha256 TEXT PRIMARY KEY,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_workspace_end (
    run_id TEXT PRIMARY KEY,
    snapshot_id TEXT,
    snapshot_error TEXT,
    created_at TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class RunStep:
    run_id: str
    step: int
    message_count: int
    summary_json: str | None
    tool_views_json: str
    snapshot_id: str | None
    snapshot_error: str | None
    constraints_revision: int | None
    task_context_text: str | None
    created_at: datetime


class SQLiteRunStepStore:

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            cursor = await database.execute("PRAGMA table_info(run_steps)")
            columns = {row["name"] for row in await cursor.fetchall()}
            if "request_tools_json" not in columns:
                await database.execute(
                    "ALTER TABLE run_steps ADD COLUMN request_tools_json TEXT"
                )
            await database.commit()

    async def record_request_tools(
        self,
        run_id: str,
        step: int,
        views: Sequence[dict[str, Any]],
    ) -> None:
        """保存这一步请求里每条工具消息的长度信息和发给模型的原文。"""
        index: list[dict[str, Any]] = []
        contents: list[tuple[str, str]] = []
        for view in views:
            entry = {key: value for key, value in view.items() if key != "content"}
            content = view.get("content")
            if isinstance(content, str):
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                entry["content_sha256"] = digest
                contents.append((digest, content))
            index.append(entry)
        async with self._connect() as database:
            await database.executemany(
                "INSERT OR IGNORE INTO run_request_tool_contents (sha256, content) "
                "VALUES (?, ?)",
                contents,
            )
            await database.execute(
                "UPDATE run_steps SET request_tools_json = ? "
                "WHERE run_id = ? AND step = ?",
                (json.dumps(index, ensure_ascii=False), run_id, step),
            )
            await database.commit()

    async def request_tool_view(
        self,
        run_id: str,
        step: int,
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        """第 step 步请求里这次工具调用的样子；content 是模型实际收到的原文。"""
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT request_tools_json FROM run_steps "
                "WHERE run_id = ? AND step = ?",
                (run_id, step),
            )
            row = await cursor.fetchone()
            if row is None or not row["request_tools_json"]:
                return None
            entry = next(
                (
                    item
                    for item in json.loads(row["request_tools_json"])
                    if item.get("tool_call_id") == tool_call_id
                ),
                None,
            )
            if entry is None:
                return None
            digest = entry.get("content_sha256")
            if digest:
                cursor = await database.execute(
                    "SELECT content FROM run_request_tool_contents WHERE sha256 = ?",
                    (digest,),
                )
                content_row = await cursor.fetchone()
                entry["content"] = content_row["content"] if content_row else None
            return entry

    async def record(
        self,
        *,
        run_id: str,
        step: int,
        message_count: int,
        summary_json: str | None,
        tool_views_json: str,
        snapshot_id: str | None,
        snapshot_error: str | None,
        constraints_revision: int | None,
        task_context_text: str | None,
    ) -> None:
        async with self._connect() as database:
            # 同一步重试（空回复、伪工具调用修正）时以最后一次请求为准
            await database.execute(
                """
                INSERT OR REPLACE INTO run_steps (
                    run_id, step, message_count, summary_json, tool_views_json,
                    snapshot_id, snapshot_error, constraints_revision,
                    task_context_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step,
                    message_count,
                    summary_json,
                    tool_views_json,
                    snapshot_id,
                    snapshot_error,
                    constraints_revision,
                    task_context_text,
                    _now_iso(),
                ),
            )
            await database.commit()

    async def record_end(
        self,
        run_id: str,
        *,
        snapshot_id: str | None,
        snapshot_error: str | None,
    ) -> None:
        async with self._connect() as database:
            await database.execute(
                """
                INSERT OR REPLACE INTO run_workspace_end (
                    run_id, snapshot_id, snapshot_error, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (run_id, snapshot_id, snapshot_error, _now_iso()),
            )
            await database.commit()

    async def list(self, run_id: str) -> tuple[RunStep, ...]:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM run_steps WHERE run_id = ? ORDER BY step ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return tuple(_step_from_row(row) for row in rows)

    async def get(self, run_id: str, step: int) -> RunStep | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM run_steps WHERE run_id = ? AND step = ?",
                (run_id, step),
            )
            row = await cursor.fetchone()
        return _step_from_row(row) if row is not None else None

    async def end_snapshot(self, run_id: str) -> str | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT snapshot_id FROM run_workspace_end WHERE run_id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()
        return row["snapshot_id"] if row is not None else None

    async def delete_runs(self, run_ids: Sequence[str]) -> None:
        if not run_ids:
            return
        async with self._connect() as database:
            for table in ("run_steps", "run_workspace_end"):
                await database.executemany(
                    f"DELETE FROM {table} WHERE run_id = ?",
                    [(run_id,) for run_id in run_ids],
                )
            await database.commit()

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        try:
            yield database
        finally:
            await database.close()


class RunStepRecorder:
    """一次运行的步骤记录器。记录失败只影响能否回退，不影响运行本身。"""

    def __init__(
        self,
        store: SQLiteRunStepStore,
        snapshots: WorkspaceSnapshotStore | None,
        run_id: str,
    ) -> None:
        self._store = store
        self._snapshots = snapshots
        self._run_id = run_id
        self._snapshot_steps = 0

    async def record(
        self,
        *,
        step: int,
        message_count: int,
        summary_state: ConversationSummaryState | None,
        tool_result_views: Sequence[ToolResultView],
        constraints_revision: int | None,
        task_context_text: str | None,
        request_tool_views: Sequence[dict[str, Any]] = (),
    ) -> None:
        snapshot_id, snapshot_error = await self._capture()
        try:
            await self._store.record(
                run_id=self._run_id,
                step=step,
                message_count=message_count,
                summary_json=(
                    summary_state.model_dump_json()
                    if summary_state is not None
                    else None
                ),
                tool_views_json=json.dumps(
                    [view.model_dump(mode="json") for view in tool_result_views],
                    ensure_ascii=False,
                ),
                snapshot_id=snapshot_id,
                snapshot_error=snapshot_error,
                constraints_revision=constraints_revision,
                task_context_text=task_context_text,
            )
            await self._store.record_request_tools(
                self._run_id, step, request_tool_views
            )
        except Exception:
            return

    async def finish(self) -> None:
        if self._snapshots is None:
            return
        try:
            result = await self._snapshots.capture()
            await self._store.record_end(
                self._run_id,
                snapshot_id=result.snapshot_id,
                snapshot_error=result.error,
            )
        except Exception:
            return

    async def _capture(self) -> tuple[str | None, str | None]:
        if self._snapshots is None:
            return None, "未配置工作区快照"
        if self._snapshot_steps >= MAX_SNAPSHOT_STEPS:
            return None, f"超过 {MAX_SNAPSHOT_STEPS} 步，此后不再保存文件快照"
        try:
            result = await self._snapshots.capture()
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        if result.snapshot_id is not None:
            self._snapshot_steps += 1
        return result.snapshot_id, result.error


def _step_from_row(row: aiosqlite.Row) -> RunStep:
    return RunStep(
        run_id=row["run_id"],
        step=row["step"],
        message_count=row["message_count"],
        summary_json=row["summary_json"],
        tool_views_json=row["tool_views_json"],
        snapshot_id=row["snapshot_id"],
        snapshot_error=row["snapshot_error"],
        constraints_revision=row["constraints_revision"],
        task_context_text=row["task_context_text"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "MAX_SNAPSHOT_STEPS",
    "RunStep",
    "RunStepRecorder",
    "SQLiteRunStepStore",
]
