
from __future__ import annotations

import pytest

from app.models.types import Message, MessageRole, ToolCall, ToolResult
from app.runtime.agent.result import AgentStopReason
from app.runtime.checkpoint import (
    CheckpointPhase,
    CheckpointStatus,
    SQLiteCheckpointStore,
)

_USER_MESSAGE = Message(role=MessageRole.USER, content="继续任务")


# 函数说明：store
# 用途：保存回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `SQLiteCheckpointStore`；返回 `instance`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteCheckpointStore` →
# `instance.initialize`。
@pytest.fixture
async def store(tmp_path) -> SQLiteCheckpointStore:
    instance = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await instance.initialize()
    return instance


# 函数说明：test_checkpoint_completed_lifecycle
# 用途：回归验证回归测试与测试辅助中的 `checkpoint_completed_lifecycle` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `SQLiteCheckpointStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.start` → `store.before_model` →
#  `store.complete`。
# 分支与异常：
#   验证条件：`started.status is CheckpointStatus.RUNNING`。
#   验证条件：`started.phase is CheckpointPhase.STARTING`。
#   验证条件：`requesting.phase is CheckpointPhase.MODEL_REQUEST`。
#   验证条件：`completed.status is CheckpointStatus.COMPLETED`。
async def test_checkpoint_completed_lifecycle(
    store: SQLiteCheckpointStore,
) -> None:
    started = await store.start(
        "run-1",
        conversation_id="conv-1",
        user_message=_USER_MESSAGE,
    )
    assert started.status is CheckpointStatus.RUNNING
    assert started.phase is CheckpointPhase.STARTING

    requesting = await store.before_model("run-1", step=1)
    assert requesting.phase is CheckpointPhase.MODEL_REQUEST
    completed = await store.complete(
        "run-1",
        stop_reason=AgentStopReason.FINAL_ANSWER,
    )

    assert completed.status is CheckpointStatus.COMPLETED
    assert completed.phase is CheckpointPhase.FINISHED
    assert completed.completed_at is not None
    assert completed.revision == 3


# 函数说明：test_checkpoint_preserves_pending_and_completed_tools_after_interrupt
# 用途：回归验证回归测试与测试辅助中的
# `checkpoint_preserves_pending_and_completed_tools_after_interrupt` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `SQLiteCheckpointStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `store.start` →
# `store.before_model` → `store.before_tools` → `ToolResult` → `store.complete_tool`；另
# 有 3 个调用点。
# 分支与异常：
#   验证条件：`loaded is not None`。
#   验证条件：`loaded.status is CheckpointStatus.INTERRUPTED`。
#   验证条件：`loaded.phase is CheckpointPhase.TOOL_EXECUTION`。
#   验证条件：`loaded.pending_tool_calls == (second,)`。
async def test_checkpoint_preserves_pending_and_completed_tools_after_interrupt(
    store: SQLiteCheckpointStore,
) -> None:
    first = ToolCall(id="call-1", name="write_file", arguments={"path": "a"})
    second = ToolCall(id="call-2", name="send_email", arguments={"to": "a"})
    await store.start(
        "run-tools",
        conversation_id="conv-1",
        user_message=_USER_MESSAGE,
    )
    await store.before_model("run-tools", step=1)
    await store.before_tools("run-tools", step=1, tool_calls=(first, second))
    first_result = ToolResult(
        tool_call_id=first.id,
        tool_name=first.name,
        success=True,
        output="written",
        duration_ms=2,
    )
    await store.complete_tool("run-tools", first_result)
    interrupted = await store.interrupt(
        "run-tools",
        error="CancelledError",
    )

    reopened = SQLiteCheckpointStore(store.database_path)
    await reopened.initialize()
    loaded = await reopened.get(interrupted.run_id)

    assert loaded is not None
    assert loaded.status is CheckpointStatus.INTERRUPTED
    assert loaded.phase is CheckpointPhase.TOOL_EXECUTION
    assert loaded.pending_tool_calls == (second,)
    assert loaded.completed_tool_results == (first_result,)


# 函数说明：test_startup_recovers_stale_running_checkpoint
# 用途：回归验证回归测试与测试辅助中的 `startup_recovers_stale_running_checkpoint` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `SQLiteCheckpointStore`；读取键 `other-run`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `store.start` →
# `store.before_model` → `store.before_tools` → `store.recover_running`。
# 分支与异常：
#   验证条件：`[item.run_id for item in recovered] == ['stale-run']`。
#   验证条件：`recovered[0].status is CheckpointStatus.INTERRUPTED`。
#   验证条件：`recovered[0].pending_tool_calls == (call,)`。
#   验证条件：`other is not None and other.status is CheckpointStatus.RUNNING`。
async def test_startup_recovers_stale_running_checkpoint(
    store: SQLiteCheckpointStore,
) -> None:
    call = ToolCall(id="uncertain", name="write_file", arguments={})
    await store.start(
        "stale-run",
        conversation_id="conv-1",
        user_message=_USER_MESSAGE,
    )
    await store.before_model("stale-run", step=2)
    await store.before_tools("stale-run", step=2, tool_calls=(call,))
    await store.start(
        "other-run",
        conversation_id="conv-2",
        user_message=_USER_MESSAGE,
    )

    recovered = await store.recover_running(conversation_id="conv-1")

    assert [item.run_id for item in recovered] == ["stale-run"]
    assert recovered[0].status is CheckpointStatus.INTERRUPTED
    assert recovered[0].pending_tool_calls == (call,)
    other = await store.get("other-run")
    assert other is not None and other.status is CheckpointStatus.RUNNING


# 函数说明：test_interrupted_checkpoint_is_recovered_only_after_explicit_mark
# 用途：回归验证回归测试与测试辅助中的
# `interrupted_checkpoint_is_recovered_only_after_explicit_mark` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `SQLiteCheckpointStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.start` → `store.interrupt` →
# `store.latest_unrecovered` → `store.mark_recovered`。
# 分支与异常：
#   验证条件：`latest is not None and latest.run_id == 'old-run'`。
#   验证条件：`marked.recovered_by_run_id == 'new-run'`。
#   验证条件：`await store.latest_unrecovered('conv-1') is None`。
async def test_interrupted_checkpoint_is_recovered_only_after_explicit_mark(
    store: SQLiteCheckpointStore,
) -> None:
    await store.start(
        "old-run",
        conversation_id="conv-1",
        user_message=_USER_MESSAGE,
    )
    await store.interrupt("old-run", error="process stopped")

    latest = await store.latest_unrecovered("conv-1")
    assert latest is not None and latest.run_id == "old-run"

    marked = await store.mark_recovered(
        "old-run",
        recovered_by_run_id="new-run",
    )
    assert marked.recovered_by_run_id == "new-run"
    assert await store.latest_unrecovered("conv-1") is None


# 函数说明：test_failed_checkpoint_keeps_stop_reason
# 用途：回归验证回归测试与测试辅助中的 `failed_checkpoint_keeps_stop_reason` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `SQLiteCheckpointStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.start` → `store.before_model` →
#  `store.fail`。
# 分支与异常：
#   验证条件：`failed.status is CheckpointStatus.FAILED`。
#   验证条件：`failed.stop_reason is AgentStopReason.MODEL_ERROR`。
#   验证条件：`failed.error == 'connection failed'`。
async def test_failed_checkpoint_keeps_stop_reason(
    store: SQLiteCheckpointStore,
) -> None:
    await store.start(
        "failed-run",
        conversation_id="conv-1",
        user_message=_USER_MESSAGE,
    )
    await store.before_model("failed-run", step=1)
    failed = await store.fail(
        "failed-run",
        stop_reason=AgentStopReason.MODEL_ERROR,
        error="connection failed",
    )

    assert failed.status is CheckpointStatus.FAILED
    assert failed.stop_reason is AgentStopReason.MODEL_ERROR
    assert failed.error == "connection failed"


# 函数说明：test_checkpoint_cannot_skip_unresolved_tool_calls
# 用途：回归验证回归测试与测试辅助中的 `checkpoint_cannot_skip_unresolved_tool_calls` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `SQLiteCheckpointStore`；读取键 `run-pending`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `store.start` →
# `store.before_tools` → `pytest.raises` → `store.before_model` → `store.complete`。
# 分支与异常：
#   验证条件：`unchanged is not None`。
#   验证条件：`unchanged.pending_tool_calls == (call,)`。
#   预期异常：`pytest.raises(ValueError, match='tool calls are pending')`。
#   预期异常：`pytest.raises(ValueError, match='pending tool calls')`。
async def test_checkpoint_cannot_skip_unresolved_tool_calls(
    store: SQLiteCheckpointStore,
) -> None:
    call = ToolCall(id="pending", name="write_file", arguments={})
    await store.start(
        "run-pending",
        conversation_id="conv-1",
        user_message=_USER_MESSAGE,
    )
    await store.before_tools("run-pending", step=1, tool_calls=(call,))

    with pytest.raises(ValueError, match="tool calls are pending"):
        await store.before_model("run-pending", step=2)
    with pytest.raises(ValueError, match="pending tool calls"):
        await store.complete(
            "run-pending",
            stop_reason=AgentStopReason.FINAL_ANSWER,
        )

    unchanged = await store.get("run-pending")
    assert unchanged is not None
    assert unchanged.pending_tool_calls == (call,)
