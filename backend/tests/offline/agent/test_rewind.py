"""执行回溯（第二阶段）：步骤检查点、工作区快照、回到第 N 步之前重新执行。"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from app.domain.conversation import SQLiteConversationStore
from app.models.types import Message, MessageRole, ToolCall, ToolDefinition
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.context import ContextManager
from app.runtime.rewind import (
    SQLiteRunStepStore,
    WorkspaceSnapshotStore,
)
from app.runtime.rewind import snapshots as snapshots_module
from app.runtime.rewind.service import (
    RewindConflict,
    RewindError,
    RewindService,
)
from app.runtime.run import SQLiteRunMessageStore
from app.runtime.run.models import Run, RunStatus
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from tests.offline.agent.test_agent_runtime import (
    DeterministicTokenEstimator,
    fake_registry,
    model_response,
)

# 换行、BOM、GBK、二进制：恢复后必须逐字节一致
_ORIGINAL_BYTES = "﻿第一行\r\n".encode() + "中文".encode("gbk") + b"\x00\xff\r\n"


class WriteFileFake(BaseTool):
    """模拟 write_file：直接写工作区里的文件。"""

    definition = ToolDefinition(
        name="write_file",
        description="write",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"],
        },
    )

    def __init__(self, root) -> None:
        self.root = root

    async def execute(self, arguments: dict[str, object]) -> str:
        target = self.root / str(arguments["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(str(arguments["text"]).encode("utf-8"))
        return "ok"


class HttpRequestFake(BaseTool):
    definition = ToolDefinition(
        name="http_request",
        description="post",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )

    async def execute(self, arguments: dict[str, object]) -> str:
        return "posted"


def _call(call_id: str, name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


async def _setup(tmp_path):
    database = tmp_path / "data" / "muharness.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    conversations = SQLiteConversationStore(database)
    await conversations.initialize()
    run_messages = SQLiteRunMessageStore(database)
    await run_messages.initialize()
    steps = SQLiteRunStepStore(database)
    await steps.initialize()
    snapshots = WorkspaceSnapshotStore(database, workspace)
    await snapshots.initialize()
    return database, workspace, conversations, run_messages, steps, snapshots


class _World:
    """一次完整的被回退执行：3 步，第 1、2 步写文件，第 2 步还发了网络请求。"""

    def __init__(self, tmp_path) -> None:
        self.tmp_path = tmp_path
        self.runs: dict[str, Run] = {}
        self.busy: str | None = None

    async def build(self) -> None:
        (
            self.database,
            self.workspace,
            self.conversations,
            self.run_messages,
            self.steps,
            self.snapshots,
        ) = await _setup(self.tmp_path)
        (self.workspace / "keep.bin").write_bytes(_ORIGINAL_BYTES)
        history = (
            Message(role=MessageRole.USER, content="旧问题"),
            Message(role=MessageRole.ASSISTANT, content="旧回答"),
        )
        self.conversation = await self.conversations.create(
            title="修复分页", messages=history
        )
        await self.conversations.set_constraints(self.conversation.id, "只修改 backend")
        registry, _ = fake_registry(
            [
                model_response(
                    tool_calls=(_call("c1", "write_file", path="a.txt", text="v1"),)
                ),
                model_response(
                    tool_calls=(
                        _call("c2", "write_file", path="a.txt", text="v2"),
                        _call("c3", "write_file", path="src/b.txt", text="new"),
                        _call("c4", "http_request", url="https://example.test/hook"),
                    )
                ),
                model_response(content="完成"),
            ]
        )
        tools = ToolRegistry()
        tools.register(WriteFileFake(self.workspace))
        tools.register(HttpRequestFake())
        runtime = AgentRuntime(
            registry,
            tools,
            provider="fake",
            context_manager=ContextManager(estimator=DeterministicTokenEstimator()),
            run_message_store=self.run_messages,
            run_step_store=self.steps,
            workspace_snapshots=self.snapshots,
        )
        result = await runtime.run(
            "修复分页",
            history=history,
            conversation_id=self.conversation.id,
            run_id="run-1",
        )
        assert result.ok
        self.result = result
        await self.conversations.save_history_state(
            self.conversation.id, result.messages, summary_state=None
        )
        now = datetime.now(UTC)
        self.runs["run-1"] = Run(
            id="run-1",
            conversation_id=self.conversation.id,
            status=RunStatus.COMPLETED,
            user_message="修复分页",
            created_at=now,
            updated_at=now,
            source="manual",
        )

        async def lookup(run_id: str) -> Run | None:
            return self.runs.get(run_id)

        async def active_work() -> str | None:
            return self.busy

        self.service = RewindService(
            self.database,
            conversation_store=self.conversations,
            run_lookup=lookup,
            run_message_store=self.run_messages,
            step_store=self.steps,
            snapshots=self.snapshots,
            active_work=active_work,
        )
        await self.service.initialize()


@pytest.fixture
async def world(tmp_path) -> _World:
    built = _World(tmp_path)
    await built.build()
    return built


@pytest.mark.asyncio
async def test_every_step_records_message_count_and_snapshot(world: _World) -> None:
    steps = await world.steps.list("run-1")
    assert [step.step for step in steps] == [1, 2, 3]
    # 继承 2 条 + 用户消息；之后每步多一条模型回复和对应的工具结果
    assert [step.message_count for step in steps] == [3, 5, 9]
    assert all(step.snapshot_id for step in steps)
    assert len({step.snapshot_id for step in steps}) == 3

    first = await world.snapshots.load(steps[0].snapshot_id)
    assert set(first) == {"keep.bin"}
    second = await world.snapshots.load(steps[1].snapshot_id)
    assert set(second) == {"keep.bin", "a.txt"}
    assert await world.steps.end_snapshot("run-1") is not None


@pytest.mark.asyncio
async def test_preview_lists_file_changes_and_irreversible_operations(
    world: _World,
) -> None:
    infos = await world.service.list_steps("run-1")
    assert [info.rewindable for info in infos] == [True, True, True]

    preview = await world.service.preview("run-1", 2)
    actions = {change.path: change.action for change in preview.files}
    assert actions == {"a.txt": "restore", "src/b.txt": "delete"}
    assert not any(change.changed_after_run for change in preview.files)
    assert [(op.step, op.tool) for op in preview.irreversible] == [
        (2, "http_request")
    ]
    assert preview.blocked_reason is None
    # 预览是只读的
    assert (world.workspace / "a.txt").read_text() == "v2"


@pytest.mark.asyncio
async def test_apply_restores_bytes_and_forks_conversation(world: _World) -> None:
    (world.workspace / "keep.bin").write_bytes(b"changed later")
    preview = await world.service.preview("run-1", 2)
    keep = next(change for change in preview.files if change.path == "keep.bin")
    assert keep.changed_after_run is True

    result = await world.service.apply(
        run_id="run-1",
        step=2,
        preview_id=preview.preview_id,
        correction="不要新建 src/b.txt，只改 a.txt",
        rewind_key="key-1",
    )

    assert (world.workspace / "a.txt").read_text() == "v1"
    assert not (world.workspace / "src" / "b.txt").exists()
    assert (world.workspace / "keep.bin").read_bytes() == _ORIGINAL_BYTES
    assert result.status == "completed"
    assert result.files_deleted == 1

    forked = await world.conversations.load_messages(result.conversation_id)
    assert forked == world.result.messages[:5]
    assert forked[-1].role is MessageRole.TOOL
    assert forked[-1].tool_call_id == "c1"
    assert (
        await world.conversations.get_constraints(result.conversation_id)
    ).text == "只修改 backend"
    fork = await world.service.fork_for(result.conversation_id)
    assert fork is not None
    assert (fork.source_conversation_id, fork.source_run_id, fork.source_step) == (
        world.conversation.id,
        "run-1",
        2,
    )
    assert "不要新建 src/b.txt" in result.draft
    assert "http_request" in result.draft
    # 原会话不变
    assert await world.conversations.load_messages(world.conversation.id) == (
        world.result.messages
    )

    again = await world.service.apply(
        run_id="run-1",
        step=2,
        preview_id="ignored",
        correction="不要新建 src/b.txt，只改 a.txt",
        rewind_key="key-1",
    )
    assert again.conversation_id == result.conversation_id

    undone = await world.service.undo("key-1")
    assert undone.status == "undone"
    assert (world.workspace / "a.txt").read_text() == "v2"
    assert (world.workspace / "src" / "b.txt").read_text() == "new"
    assert (world.workspace / "keep.bin").read_bytes() == b"changed later"
    assert (await world.service.fork_for(result.conversation_id)).undone is True


@pytest.mark.asyncio
async def test_forked_conversation_can_continue_with_draft(world: _World) -> None:
    preview = await world.service.preview("run-1", 1)
    result = await world.service.apply(
        run_id="run-1",
        step=1,
        preview_id=preview.preview_id,
        correction="先读文件再写",
        rewind_key="key-2",
    )
    forked = await world.conversations.load_messages(result.conversation_id)
    assert forked[-1].content == "修复分页"
    assert not (world.workspace / "a.txt").exists()

    registry, adapter = fake_registry([model_response(content="好的")])
    continued = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        context_manager=ContextManager(estimator=DeterministicTokenEstimator()),
    ).run(result.draft, history=forked, conversation_id=result.conversation_id)
    assert continued.ok
    sent = adapter.requests[0].messages
    assert sent[-1].content == result.draft
    assert any(message.content == "修复分页" for message in sent)


@pytest.mark.asyncio
async def test_apply_refuses_when_workspace_changed_or_work_active(
    world: _World,
) -> None:
    preview = await world.service.preview("run-1", 2)
    (world.workspace / "a.txt").write_text("edited after preview")
    with pytest.raises(RewindConflict):
        await world.service.apply(
            run_id="run-1",
            step=2,
            preview_id=preview.preview_id,
            correction="x",
            rewind_key="key-3",
        )
    assert (world.workspace / "a.txt").read_text() == "edited after preview"

    world.busy = "有执行正在进行"
    fresh = await world.service.preview("run-1", 2)
    assert fresh.blocked_reason == "有执行正在进行"
    with pytest.raises(RewindError, match="有执行正在进行"):
        await world.service.apply(
            run_id="run-1",
            step=2,
            preview_id=fresh.preview_id,
            correction="x",
            rewind_key="key-4",
        )
    with pytest.raises(RewindError):
        await world.service.apply(
            run_id="run-1",
            step=2,
            preview_id=fresh.preview_id,
            correction="   ",
            rewind_key="key-5",
        )


@pytest.mark.asyncio
async def test_ineligible_runs_cannot_rewind(world: _World) -> None:
    world.runs["run-1"] = world.runs["run-1"].model_copy(
        update={"source": "automation"}
    )
    infos = await world.service.list_steps("run-1")
    assert not any(info.rewindable for info in infos)
    assert "自动化" in (infos[0].reason or "")
    with pytest.raises(RewindError):
        await world.service.preview("run-1", 1)

    world.runs["run-1"] = world.runs["run-1"].model_copy(
        update={"source": "manual", "status": RunStatus.RUNNING}
    )
    with pytest.raises(RewindError, match="尚未结束"):
        await world.service.preview("run-1", 1)


@pytest.mark.asyncio
async def test_snapshot_skips_large_files_symlinks_and_dependency_dirs(
    tmp_path,
    monkeypatch,
) -> None:
    _, workspace, _, _, _, snapshots = await _setup(tmp_path)
    monkeypatch.setattr(snapshots_module, "MAX_FILE_BYTES", 10)
    (workspace / "small.txt").write_bytes(b"tiny")
    (workspace / "big.log").write_bytes(b"x" * 50)
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "dep.js").write_text("ignored")
    try:
        os.symlink(workspace / "small.txt", workspace / "link.txt")
        has_symlink = True
    except OSError:
        has_symlink = False

    captured = await snapshots.capture()
    assert captured.snapshot_id is not None
    manifest = await snapshots.load(captured.snapshot_id)
    assert manifest["small.txt"].sha256 is not None
    assert manifest["big.log"].skipped == "too_large"
    assert "node_modules/dep.js" not in manifest
    if has_symlink:
        assert manifest["link.txt"].skipped == "symlink"

    # 过大文件在目标时刻没有内容：回退时既不恢复也不删除
    (workspace / "big.log").write_bytes(b"y" * 60)
    (workspace / "huge-new.bin").write_bytes(b"z" * 60)
    (workspace / "small.txt").write_bytes(b"changed")
    plan = await snapshots.restore(manifest)
    assert plan.restore == ("small.txt",)
    assert plan.delete == ()
    assert (workspace / "small.txt").read_bytes() == b"tiny"
    assert (workspace / "big.log").read_bytes() == b"y" * 60
    assert (workspace / "huge-new.bin").exists()


@pytest.mark.asyncio
async def test_snapshot_limits_report_error_instead_of_partial_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    _, workspace, _, _, _, snapshots = await _setup(tmp_path)
    monkeypatch.setattr(snapshots_module, "MAX_SNAPSHOT_FILES", 2)
    for index in range(3):
        (workspace / f"f{index}.txt").write_text(str(index))
    captured = await snapshots.capture()
    assert captured.snapshot_id is None
    assert "2 个" in (captured.error or "")


@pytest.mark.asyncio
async def test_steps_without_snapshot_are_not_rewindable(tmp_path) -> None:
    database, workspace, conversations, run_messages, steps, _ = await _setup(
        tmp_path
    )
    conversation = await conversations.create(title="t")
    registry, _ = fake_registry([model_response(content="完成")])
    await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        context_manager=ContextManager(estimator=DeterministicTokenEstimator()),
        run_message_store=run_messages,
        run_step_store=steps,
        workspace_snapshots=None,
    ).run("问题", conversation_id=conversation.id, run_id="run-x")
    now = datetime.now(UTC)
    run = Run(
        id="run-x",
        conversation_id=conversation.id,
        status=RunStatus.COMPLETED,
        created_at=now,
        updated_at=now,
    )

    async def lookup(run_id: str) -> Run | None:
        return run

    async def idle() -> str | None:
        return None

    service = RewindService(
        database,
        conversation_store=conversations,
        run_lookup=lookup,
        run_message_store=run_messages,
        step_store=steps,
        snapshots=WorkspaceSnapshotStore(database, workspace),
        active_work=idle,
    )
    await service.initialize()
    [info] = await service.list_steps("run-x")
    assert info.rewindable is False
    assert "没有文件快照" in (info.reason or "")


@pytest.mark.asyncio
async def test_rewind_rpc_methods(world: _World) -> None:
    from types import SimpleNamespace

    from app.server.rpc.methods.runs import (
        run_rewind_apply,
        run_rewind_preview,
        run_rewind_undo,
        run_steps_list,
    )
    from app.server.rpc.protocol import JsonRpcError

    ctx = SimpleNamespace(application=SimpleNamespace(rewind_service=world.service))
    listed = await run_steps_list({"run_id": "run-1"}, ctx)
    assert [info.step for info in listed["steps"]] == [1, 2, 3]

    preview = (await run_rewind_preview({"run_id": "run-1", "step": 2}, ctx))[
        "preview"
    ]
    with pytest.raises(JsonRpcError) as outdated:
        await run_rewind_apply(
            {
                "run_id": "run-1",
                "step": 2,
                "preview_id": "stale",
                "correction": "x",
                "rewind_key": "k",
            },
            ctx,
        )
    assert outdated.value.data == {"reason": "preview_outdated"}

    applied = await run_rewind_apply(
        {
            "run_id": "run-1",
            "step": 2,
            "preview_id": preview.preview_id,
            "correction": "只改 a.txt",
            "rewind_key": "k2",
        },
        ctx,
    )
    assert applied["rewind"].status == "completed"
    undone = await run_rewind_undo({"rewind_key": "k2"}, ctx)
    assert undone["rewind"].status == "undone"
    with pytest.raises(JsonRpcError):
        await run_rewind_preview({"run_id": "run-1", "step": 0}, ctx)
    with pytest.raises(JsonRpcError):
        await run_rewind_preview({"run_id": "run-1", "step": 9}, ctx)
