"""回到第 N 步之前重新执行。

流程：预览（只读）→ 确认 → 先备份当前工作区 → 恢复文件 → 新建分支会话。
分支会话包含第 N 步之前的全部消息、当时的摘要和工具结果视图，以及当前的
"必须记住的事项"；纠正内容作为草稿交给用户发送，第 1~N-1 步不会重跑。
旧会话和旧运行记录保持不变。

不回退的东西：网络请求、长期记忆、任务状态、数据库、已发布的产物，以及
工作区之外的文件。预览里会把这些操作列出来，草稿里也会告诉模型。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import aiosqlite
from pydantic import BaseModel, ConfigDict

from app.models.types import AgentMode, Message, MessageRole
from app.runtime.context.summary import ConversationSummaryState
from app.runtime.context.tool_views import ToolResultView
from app.runtime.run.messages_store import SQLiteRunMessageStore, history_sha256
from app.runtime.run.models import TERMINAL_STATUSES, Run

from .snapshots import Manifest, WorkspaceSnapshotStore, manifest_digest
from .steps import RunStep, SQLiteRunStepStore

if TYPE_CHECKING:
    from app.domain.conversation import SQLiteConversationStore

# 只读或纯查询：不需要回退，也不会列进"不会回退的操作"
READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "get_current_time",
        "history_read",
        "history_search",
        "memory_read",
        "memory_search",
        "memory_list",
        "task_get",
        "task_list",
        "web_search",
        "skill_read",
        "skill_resource_read",
        "artifact_list",
        "automation_get",
        "automation_list",
        "tool_search",
    }
)
# 只改工作区文件：文件恢复已经覆盖
FILE_TOOLS = frozenset({"write_file"})
_SHELL_TOOLS = frozenset({"run_shell_command"})

_ARGUMENT_PREVIEW_CHARS = 160

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_rewinds (
    rewind_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    status TEXT NOT NULL,
    backup_snapshot_id TEXT NOT NULL,
    conversation_id TEXT,
    draft TEXT,
    files_restored INTEGER NOT NULL DEFAULT 0,
    files_deleted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_forks (
    conversation_id TEXT PRIMARY KEY,
    source_conversation_id TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    source_step INTEGER NOT NULL,
    rewind_key TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

ActiveWorkCheck = Callable[[], Awaitable[str | None]]


class RewindError(ValueError):
    """不能回退：原因直接展示给用户。"""


class RewindConflict(RewindError):
    """预览之后工作区又变了，需要重新预览。"""


class RewindStepInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    message_count: int
    has_snapshot: bool
    snapshot_error: str | None = None
    rewindable: bool
    reason: str | None = None


class RewindFileChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    action: Literal["restore", "delete"]
    # 运行结束后又被别的操作改过（其他会话、用户手动修改）
    changed_after_run: bool = False


class IrreversibleOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int | None
    tool: str
    summary: str
    note: str


class RewindPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    preview_id: str
    run_id: str
    step: int
    files: tuple[RewindFileChange, ...]
    skipped_files: tuple[str, ...]
    irreversible: tuple[IrreversibleOperation, ...]
    blocked_reason: str | None = None


class RewindResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rewind_key: str
    run_id: str
    step: int
    conversation_id: str
    draft: str
    files_restored: int
    files_deleted: int
    status: str


class ConversationFork(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    source_conversation_id: str
    source_run_id: str
    source_step: int
    rewind_key: str
    undone: bool = False


class RewindService:

    def __init__(
        self,
        database_path: str | Path,
        *,
        conversation_store: SQLiteConversationStore,
        run_lookup: Callable[[str], Awaitable[Run | None]],
        run_message_store: SQLiteRunMessageStore,
        step_store: SQLiteRunStepStore,
        snapshots: WorkspaceSnapshotStore,
        active_work: ActiveWorkCheck,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self._conversations = conversation_store
        self._run_lookup = run_lookup
        self._messages = run_message_store
        self._steps = step_store
        self._snapshots = snapshots
        self._active_work = active_work
        # 回退会改写工作区：同一进程内一次只做一个
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    async def list_steps(self, run_id: str) -> tuple[RewindStepInfo, ...]:
        run = await self._run_lookup(run_id)
        run_reason = _run_ineligibility(run)
        steps = await self._steps.list(run_id)
        has_messages = await self._messages.history_ref(run_id) is not None
        infos: list[RewindStepInfo] = []
        for step in steps:
            reason = run_reason
            if reason is None and not has_messages:
                reason = "该执行没有原文记录，无法重建会话"
            if reason is None and step.snapshot_id is None:
                reason = f"这一步没有文件快照（{step.snapshot_error or '未知原因'}）"
            infos.append(
                RewindStepInfo(
                    step=step.step,
                    message_count=step.message_count,
                    has_snapshot=step.snapshot_id is not None,
                    snapshot_error=step.snapshot_error,
                    rewindable=reason is None,
                    reason=reason,
                )
            )
        return tuple(infos)

    async def preview(self, run_id: str, step: int) -> RewindPreview:
        run, step_row, target = await self._target(run_id, step)
        current = await self._snapshots.current_manifest()
        plan = self._snapshots.plan(target, current)
        end_manifest = await self._end_manifest(run_id)
        files = [
            RewindFileChange(
                path=path,
                action="restore",
                changed_after_run=_changed_after_run(path, current, end_manifest),
            )
            for path in plan.restore
        ] + [
            RewindFileChange(
                path=path,
                action="delete",
                changed_after_run=_changed_after_run(path, current, end_manifest),
            )
            for path in plan.delete
        ]
        return RewindPreview(
            preview_id=_preview_id(run_id, step, step_row, current),
            run_id=run_id,
            step=step,
            files=tuple(files),
            skipped_files=plan.skipped,
            irreversible=await self._irreversible(run, step_row),
            blocked_reason=await self._active_work(),
        )

    async def apply(
        self,
        *,
        run_id: str,
        step: int,
        preview_id: str,
        correction: str,
        rewind_key: str,
    ) -> RewindResult:
        correction = correction.strip()
        if not correction:
            raise RewindError("请写下纠正内容，告诉模型这一步应该怎么做")
        async with self._lock:
            existing = await self._get_rewind(rewind_key)
            if existing is not None:
                if existing["run_id"] != run_id or existing["step"] != step:
                    raise RewindError("同一个回退请求编号不能用于不同的步骤")
                if existing["status"] in {"completed", "undone"}:
                    return _result_from_row(existing)
            run, step_row, target = await self._target(run_id, step)
            busy = await self._active_work()
            if busy is not None:
                raise RewindError(busy)
            if existing is None:
                current = await self._snapshots.current_manifest()
                if _preview_id(run_id, step, step_row, current) != preview_id:
                    raise RewindConflict("工作区在预览之后发生了变化，请重新预览确认")
                backup = await self._snapshots.capture()
                if backup.snapshot_id is None:
                    raise RewindError(f"无法备份当前工作区，已取消回退：{backup.error}")
                await self._insert_rewind(
                    rewind_key, run_id, step, backup.snapshot_id
                )
            # 中途崩溃后用同一个编号重试，会从上次停下的地方继续
            if existing is None or existing["status"] == "started":
                plan = await self._snapshots.restore(target)
                await self._update_rewind(
                    rewind_key,
                    status="files_restored",
                    files_restored=len(plan.restore),
                    files_deleted=len(plan.delete),
                )
            prefix = await self._prefix_messages(run, step_row)
            source = await self._conversations.get(run.conversation_id or "")
            conversation = await self._conversations.create(
                title=f"重做：{source.title if source else '会话'}（第 {step} 步）"
            )
            await self._conversations.save_history_state(
                conversation.id,
                prefix,
                summary_state=_summary_state(step_row),
                tool_result_views=_tool_views(step_row),
            )
            if run.conversation_id is not None:
                constraints = await self._conversations.get_constraints(
                    run.conversation_id
                )
                if constraints.text:
                    await self._conversations.set_constraints(
                        conversation.id, constraints.text
                    )
            irreversible = await self._irreversible(run, step_row)
            draft = build_rewind_draft(
                step=step,
                correction=correction,
                irreversible=irreversible,
                task_context_text=step_row.task_context_text,
            )
            await self._insert_fork(
                conversation.id,
                source_conversation_id=run.conversation_id or "",
                source_run_id=run_id,
                source_step=step,
                rewind_key=rewind_key,
            )
            await self._update_rewind(
                rewind_key,
                status="completed",
                conversation_id=conversation.id,
                draft=draft,
            )
            row = await self._get_rewind(rewind_key)
            assert row is not None
            return _result_from_row(row)

    async def undo(self, rewind_key: str) -> RewindResult:
        """把工作区文件恢复到回退之前（分支会话保留）。"""
        async with self._lock:
            row = await self._get_rewind(rewind_key)
            if row is None:
                raise RewindError("找不到这次回退记录")
            if row["status"] == "undone":
                return _result_from_row(row)
            if row["status"] != "completed":
                raise RewindError("这次回退还没有完成，不能撤销")
            busy = await self._active_work()
            if busy is not None:
                raise RewindError(busy)
            backup = await self._snapshots.load(row["backup_snapshot_id"])
            if backup is None:
                raise RewindError("回退前的备份已丢失，无法撤销")
            await self._snapshots.restore(backup)
            await self._update_rewind(rewind_key, status="undone")
            row = await self._get_rewind(rewind_key)
            assert row is not None
            return _result_from_row(row)

    async def fork_for(self, conversation_id: str) -> ConversationFork | None:
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT f.*, r.status AS rewind_status
                FROM conversation_forks AS f
                LEFT JOIN run_rewinds AS r ON r.rewind_key = f.rewind_key
                WHERE f.conversation_id = ?
                """,
                (conversation_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return ConversationFork(
            conversation_id=row["conversation_id"],
            source_conversation_id=row["source_conversation_id"],
            source_run_id=row["source_run_id"],
            source_step=row["source_step"],
            rewind_key=row["rewind_key"],
            undone=row["rewind_status"] == "undone",
        )

    async def delete_for_conversation(
        self,
        conversation_id: str,
        *,
        run_ids: Sequence[str] = (),
    ) -> None:
        await self._steps.delete_runs(run_ids)
        async with self._connect() as database:
            await database.execute(
                "DELETE FROM conversation_forks WHERE conversation_id = ?",
                (conversation_id,),
            )
            await database.commit()

    async def _target(
        self,
        run_id: str,
        step: int,
    ) -> tuple[Run, RunStep, Manifest]:
        run = await self._run_lookup(run_id)
        reason = _run_ineligibility(run)
        if reason is not None:
            raise RewindError(reason)
        assert run is not None
        step_row = await self._steps.get(run_id, step)
        if step_row is None:
            raise RewindError(f"第 {step} 步没有检查点记录")
        if step_row.snapshot_id is None:
            raise RewindError(
                f"第 {step} 步没有文件快照（{step_row.snapshot_error or '未知原因'}）"
            )
        target = await self._snapshots.load(step_row.snapshot_id)
        if target is None:
            raise RewindError(f"第 {step} 步的文件快照已丢失")
        return run, step_row, target

    async def _end_manifest(self, run_id: str) -> Manifest | None:
        end_id = await self._steps.end_snapshot(run_id)
        return await self._snapshots.load(end_id) if end_id else None

    async def _prefix_messages(
        self,
        run: Run,
        step_row: RunStep,
    ) -> tuple[Message, ...]:
        ref = await self._messages.history_ref(run.id)
        if ref is None:
            raise RewindError("该执行没有原文记录，无法重建会话")
        inherited: tuple[Message, ...] = ()
        if ref.inherited_count:
            if ref.conversation_id is None or (
                await self._conversations.get(ref.conversation_id) is None
            ):
                raise RewindError("原会话已不存在，无法还原之前的历史")
            history = await self._conversations.load_messages(ref.conversation_id)
            inherited = tuple(history[: ref.inherited_count])
            if (
                len(inherited) != ref.inherited_count
                or history_sha256(inherited) != ref.inherited_sha256
            ):
                raise RewindError("原会话的历史在该执行之后被改写，无法还原")
        own_needed = step_row.message_count - ref.inherited_count
        own = (
            tuple(
                message
                for _, message in await self._messages.load(
                    run.id, offset=0, limit=own_needed
                )
            )
            if own_needed > 0
            else ()
        )
        prefix = (*inherited, *own)
        if len(prefix) != step_row.message_count:
            raise RewindError(
                f"执行的原文记录不完整，无法重建第 {step_row.step} 步之前的会话"
            )
        return prefix

    async def _irreversible(
        self,
        run: Run,
        step_row: RunStep,
    ) -> tuple[IrreversibleOperation, ...]:
        ref = await self._messages.history_ref(run.id)
        if ref is None:
            return ()
        steps = await self._steps.list(run.id)
        start = max(0, step_row.message_count - ref.inherited_count)
        operations: list[IrreversibleOperation] = []
        for index, message in await self._messages.load(run.id, offset=start):
            if message.role is not MessageRole.ASSISTANT:
                continue
            absolute = ref.inherited_count + index
            for call in message.tool_calls:
                if call.name in READ_ONLY_TOOLS or call.name in FILE_TOOLS:
                    continue
                note = (
                    "命令对工作区文件的修改会被恢复，但对数据库、网络、"
                    "工作区之外文件的影响不会回退"
                    if call.name in _SHELL_TOOLS
                    else "这类操作的影响不会回退"
                )
                operations.append(
                    IrreversibleOperation(
                        step=_step_of(absolute, steps),
                        tool=call.name,
                        summary=_arguments_preview(call.arguments),
                        note=note,
                    )
                )
        return tuple(operations)

    async def _get_rewind(self, rewind_key: str) -> aiosqlite.Row | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM run_rewinds WHERE rewind_key = ?",
                (rewind_key,),
            )
            return await cursor.fetchone()

    async def _insert_rewind(
        self,
        rewind_key: str,
        run_id: str,
        step: int,
        backup_snapshot_id: str,
    ) -> None:
        now = _now_iso()
        async with self._connect() as database:
            await database.execute(
                """
                INSERT INTO run_rewinds (
                    rewind_key, run_id, step, status, backup_snapshot_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'started', ?, ?, ?)
                """,
                (rewind_key, run_id, step, backup_snapshot_id, now, now),
            )
            await database.commit()

    async def _update_rewind(self, rewind_key: str, **fields: object) -> None:
        assignments = ", ".join(f"{name} = ?" for name in fields)
        async with self._connect() as database:
            await database.execute(
                f"UPDATE run_rewinds SET {assignments}, updated_at = ? "
                "WHERE rewind_key = ?",
                (*fields.values(), _now_iso(), rewind_key),
            )
            await database.commit()

    async def _insert_fork(
        self,
        conversation_id: str,
        *,
        source_conversation_id: str,
        source_run_id: str,
        source_step: int,
        rewind_key: str,
    ) -> None:
        async with self._connect() as database:
            await database.execute(
                """
                INSERT OR REPLACE INTO conversation_forks (
                    conversation_id, source_conversation_id, source_run_id,
                    source_step, rewind_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    source_conversation_id,
                    source_run_id,
                    source_step,
                    rewind_key,
                    _now_iso(),
                ),
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


def build_rewind_draft(
    *,
    step: int,
    correction: str,
    irreversible: Sequence[IrreversibleOperation],
    task_context_text: str | None,
) -> str:
    """分支会话的第一条消息：说明发生了什么、哪些没撤销，再给出纠正。"""
    lines = [
        f"【回到第 {step} 步之前重新执行】",
        f"原执行第 {step} 步及之后的操作已撤销，"
        f"工作区文件已恢复到第 {step} 步之前的状态。",
    ]
    if irreversible:
        lines.append("以下操作的影响没有撤销，请按已发生处理：")
        lines.extend(
            f"- 第 {operation.step or '?'} 步 {operation.tool}：{operation.summary}"
            for operation in irreversible[:10]
        )
        if len(irreversible) > 10:
            lines.append(f"- 另有 {len(irreversible) - 10} 项")
    if task_context_text:
        lines.append("当时的任务状态：")
        lines.append(task_context_text.strip())
    lines.append(f"我的纠正：{correction.strip()}")
    lines.append("请根据纠正重新决定接下来怎么做。")
    return "\n".join(lines)


def _run_ineligibility(run: Run | None) -> str | None:
    if run is None:
        return "执行不存在"
    if run.status not in TERMINAL_STATUSES:
        return "执行尚未结束，请先停止或等待完成"
    if run.mode is not AgentMode.NORMAL:
        return "只支持普通模式的执行（PLAN 和长任务角色不支持重做）"
    if run.source not in (None, "manual"):
        return "自动化任务和长任务的执行不支持重做"
    if run.conversation_id is None:
        return "该执行没有关联会话"
    return None


def _preview_id(run_id: str, step: int, step_row: RunStep, current: Manifest) -> str:
    return hashlib.sha256(
        f"{run_id}:{step}:{step_row.snapshot_id}:{manifest_digest(current)}".encode()
    ).hexdigest()


def _changed_after_run(
    path: str,
    current: Manifest,
    end_manifest: Manifest | None,
) -> bool:
    if end_manifest is None:
        return False
    now = current.get(path)
    at_end = end_manifest.get(path)
    return (now.sha256 if now else None) != (at_end.sha256 if at_end else None)


def _step_of(message_index: int, steps: Sequence[RunStep]) -> int | None:
    found: int | None = None
    for step in steps:
        if step.message_count <= message_index:
            found = step.step
    return found


def _arguments_preview(arguments: object) -> str:
    text = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False)
    )
    if len(text) > _ARGUMENT_PREVIEW_CHARS:
        return text[: _ARGUMENT_PREVIEW_CHARS - 1] + "…"
    return text


def _summary_state(step_row: RunStep) -> ConversationSummaryState | None:
    if not step_row.summary_json:
        return None
    return ConversationSummaryState.model_validate_json(step_row.summary_json)


def _tool_views(step_row: RunStep) -> tuple[ToolResultView, ...]:
    return tuple(
        ToolResultView.model_validate(item)
        for item in json.loads(step_row.tool_views_json or "[]")
    )


def _result_from_row(row: aiosqlite.Row) -> RewindResult:
    return RewindResult(
        rewind_key=row["rewind_key"],
        run_id=row["run_id"],
        step=row["step"],
        conversation_id=row["conversation_id"] or "",
        draft=row["draft"] or "",
        files_restored=row["files_restored"],
        files_deleted=row["files_deleted"],
        status=row["status"],
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "ConversationFork",
    "IrreversibleOperation",
    "RewindConflict",
    "RewindError",
    "RewindFileChange",
    "RewindPreview",
    "RewindResult",
    "RewindService",
    "RewindStepInfo",
    "build_rewind_draft",
]
