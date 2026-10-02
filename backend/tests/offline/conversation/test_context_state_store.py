from __future__ import annotations

import json

import aiosqlite
import pytest

from app.domain.conversation import SQLiteConversationStore
from app.models.types import Message, MessageRole, ToolCall
from app.runtime.context.summary import (
    ConversationSummaryState,
    RollingConversationSummary,
)
from app.runtime.context.tool_views import ToolResultViewState


# 函数说明：_messages
# 用途：返回
# `(Message(role=MessageRole.USER, content='run'), Message(role=MessageRole.ASSISTANT,…`
# ，提供 回归测试与测试辅助 的派生值。
# 参数：
#   output：工具、模型或转换步骤的输出，类型 `str`；默认 `'x' * 1000`。
# 返回：类型 `tuple[Message, ...]`；返回
# `(Message(role=MessageRole.USER, content='run'), Message(role=MessageRole.ASSISTANT,…`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolCall` → `json.dumps`
# 。
def _messages(output: str = "x" * 1000) -> tuple[Message, ...]:
    return (
        Message(role=MessageRole.USER, content="run"),
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=(ToolCall(id="reused-call", name="execute", arguments={}),),
        ),
        Message(
            role=MessageRole.TOOL, tool_call_id="reused-call", name="execute",
            content=json.dumps(
                {"output": output, "success": True, "evidence_id": "raw"}
            ),
        ),
        Message(role=MessageRole.ASSISTANT, content="done"),
    )


# 函数说明：_summary
# 用途：返回 `ConversationSummaryState(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   objective：`objective`输入或配置值，类型 `str`；默认 `'continue'`。
# 返回：类型 `ConversationSummaryState`；返回 `ConversationSummaryState(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationSummaryState` →
# `RollingConversationSummary`。
def _summary(objective: str = "continue") -> ConversationSummaryState:
    return ConversationSummaryState(
        summary=RollingConversationSummary(current_objective=objective),
        covered_message_count=3,
    )


# 函数说明：_views
# 用途：在回归测试与测试辅助中处理 `_views`，通过 `state.project` 完成首个内部处理步骤。
# 参数：
#   messages：本次处理的消息序列。
# 返回：返回 `state.snapshot(messages)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolResultViewState` →
# `state.project` → `state.snapshot`。
def _views(messages):
    state = ToolResultViewState((), max_output_chars=100, head_chars=10, tail_chars=10)
    state.project(messages)
    return state.snapshot(messages)


# 函数说明：test_terminal_state_survives_restart_without_overwriting_raw
# 用途：回归验证回归测试与测试辅助中的
# `terminal_state_survives_restart_without_overwriting_raw` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `store.create` → `_messages` → `_views` →
# `store.save_history_state`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`restored == raw`。
#   验证条件：`saved_views == views`。
#   验证条件：`await reopened.load_summary_state(conversation.id) == _summary()`。
#   验证条件：`projected[2].content == views[0].content`。
async def test_terminal_state_survives_restart_without_overwriting_raw(
    tmp_path,
) -> None:
    store = SQLiteConversationStore(tmp_path / "state.db")
    await store.initialize()
    conversation = await store.create()
    raw = _messages()
    views = _views(raw)
    await store.save_history_state(
        conversation.id, raw, summary_state=_summary(), tool_result_views=views,
    )

    reopened = SQLiteConversationStore(store.database_path)
    await reopened.initialize()
    restored = await reopened.load_messages(conversation.id)
    saved_views = await reopened.load_tool_result_views(conversation.id)
    assert restored == raw
    assert saved_views == views
    assert await reopened.load_summary_state(conversation.id) == _summary()
    projected = ToolResultViewState(
        restored, saved_views, max_output_chars=1, head_chars=0, tail_chars=0,
    ).project(restored)
    assert projected[2].content == views[0].content
    assert len(json.loads(restored[2].content)["output"]) == 1000


# 函数说明：test_failure_rolls_back_history_summary_and_views_together
# 用途：回归验证回归测试与测试辅助中的
# `failure_rolls_back_history_summary_and_views_together` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `store.create` → `_messages` → `_views` → `_summary`；另有 8 个调
# 用点。
# 资源/并发边界：
# `pytest.raises(aiosqlite.IntegrityError, match='forced transaction failure')`，上下文
# 退出时执行相应清理。
# 分支与异常：
#   验证条件：`await store.load_messages(conversation.id) == old`。
#   验证条件：`await store.load_summary_state(conversation.id) == old_summary`。
#   验证条件：`await store.load_tool_result_views(conversation.id) == old_views`。
#   预期异常：
# `pytest.raises(aiosqlite.IntegrityError, match='forced transaction failure')`。
# 副作用与资源：
#   数据库操作：CREATE ON；连接与事务边界以 with/提交语句为准。
async def test_failure_rolls_back_history_summary_and_views_together(tmp_path) -> None:
    store = SQLiteConversationStore(tmp_path / "state.db")
    await store.initialize()
    conversation = await store.create()
    old = _messages("old" * 1000)
    old_views = _views(old)
    old_summary = _summary("old")
    await store.save_history_state(
        conversation.id, old, summary_state=old_summary, tool_result_views=old_views,
    )
    async with aiosqlite.connect(store.database_path) as database:
        await database.execute(
            "CREATE TRIGGER reject_summary_update BEFORE UPDATE "
            "ON conversation_summaries "
            "BEGIN SELECT RAISE(ABORT, 'forced transaction failure'); END"
        )
        await database.commit()

    new = _messages("new" * 1000)
    with pytest.raises(aiosqlite.IntegrityError, match="forced transaction failure"):
        await store.save_history_state(
            conversation.id, new, summary_state=_summary("new"),
            tool_result_views=_views(new),
        )
    assert await store.load_messages(conversation.id) == old
    assert await store.load_summary_state(conversation.id) == old_summary
    assert await store.load_tool_result_views(conversation.id) == old_views


# 函数说明：test_replacing_sqlite_rows_preserves_matching_frozen_views
# 用途：回归验证回归测试与测试辅助中的
# `replacing_sqlite_rows_preserves_matching_frozen_views` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `store.create` → `_messages` → `_views` →
# `store.save_history_state`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`await store.load_tool_result_views(conversation.id) == views`。
#   验证条件：`await store.load_summary_state(conversation.id) is None`。
#   验证条件：`await store.load_tool_result_views(conversation.id) == ()`。
#   验证条件：`ToolResultViewState(changed).project(changed) == changed`。
async def test_replacing_sqlite_rows_preserves_matching_frozen_views(tmp_path) -> None:
    store = SQLiteConversationStore(tmp_path / "state.db")
    await store.initialize()
    conversation = await store.create()
    raw = _messages()
    views = _views(raw)
    await store.save_history_state(
        conversation.id, raw, summary_state=_summary(), tool_result_views=views,
    )
    await store.replace_messages(conversation.id, raw)
    assert await store.load_tool_result_views(conversation.id) == views
    assert await store.load_summary_state(conversation.id) is None

    changed = _messages("changed" * 1000)
    await store.replace_messages(conversation.id, changed)
    assert await store.load_tool_result_views(conversation.id) == ()
    assert ToolResultViewState(changed).project(changed) == changed


# 函数说明：test_same_call_id_in_new_run_does_not_reuse_old_view
# 用途：回归验证回归测试与测试辅助中的 `same_call_id_in_new_run_does_not_reuse_old_view`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `store.create` → `_messages` → `_views` → `ToolResultViewState`；
# 另有 4 个调用点。
# 分支与异常：
#   验证条件：`[view.source_sequence for view in saved] == [2, 6]`。
#   验证条件：`[view.representation for view in saved] == ['excerpt', 'full']`。
#   验证条件：`projected[6] == combined[6]`。
async def test_same_call_id_in_new_run_does_not_reuse_old_view(tmp_path) -> None:
    store = SQLiteConversationStore(tmp_path / "state.db")
    await store.initialize()
    conversation = await store.create()
    first = _messages()
    first_views = _views(first)
    combined = (*first, *_messages("short"))
    state = ToolResultViewState(first, first_views, max_output_chars=100)
    projected = state.project(combined)
    await store.save_history_state(
        conversation.id, combined, summary_state=None,
        tool_result_views=state.snapshot(combined),
    )
    saved = await store.load_tool_result_views(conversation.id)
    assert [view.source_sequence for view in saved] == [2, 6]
    assert [view.representation for view in saved] == ["excerpt", "full"]
    assert projected[6] == combined[6]


# 函数说明：test_invalid_context_state_does_not_change_raw_history
# 用途：回归验证回归测试与测试辅助中的
# `invalid_context_state_does_not_change_raw_history` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `_messages` → `store.create` → `_views` → `pytest.raises`；另有 4
#  个调用点。
# 分支与异常：
#   验证条件：`await store.load_messages(conversation.id) == raw`。
#   验证条件：`await store.load_tool_result_views(conversation.id) == ()`。
#   预期异常：`pytest.raises(ValueError, match='does not match')`。
#   预期异常：`pytest.raises(ValueError, match='coverage exceeds')`。
async def test_invalid_context_state_does_not_change_raw_history(tmp_path) -> None:
    store = SQLiteConversationStore(tmp_path / "state.db")
    await store.initialize()
    raw = _messages()
    conversation = await store.create(messages=raw)
    view = _views(raw)[0].model_copy(update={"source_sequence": 0})
    with pytest.raises(ValueError, match="does not match"):
        await store.save_history_state(
            conversation.id, raw, summary_state=None, tool_result_views=(view,),
        )
    invalid_summary = _summary().model_copy(update={"covered_message_count": 99})
    with pytest.raises(ValueError, match="coverage exceeds"):
        await store.save_history_state(
            conversation.id, raw, summary_state=invalid_summary,
        )
    assert await store.load_messages(conversation.id) == raw
    assert await store.load_tool_result_views(conversation.id) == ()


# 函数说明：test_clear_and_delete_remove_context_sidecars
# 用途：回归验证回归测试与测试辅助中的 `clear_and_delete_remove_context_sidecars` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `store.create` → `_messages` → `store.save_history_state` →
# `_summary`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`await store.load_summary_state(conversation.id) is None`。
#   验证条件：`await store.load_tool_result_views(conversation.id) == ()`。
#   验证条件：`await store.delete(conversation.id)`。
async def test_clear_and_delete_remove_context_sidecars(tmp_path) -> None:
    store = SQLiteConversationStore(tmp_path / "state.db")
    await store.initialize()
    conversation = await store.create()
    raw = _messages()
    await store.save_history_state(
        conversation.id, raw, summary_state=_summary(), tool_result_views=_views(raw),
    )
    await store.replace_messages(conversation.id, ())
    assert await store.load_summary_state(conversation.id) is None
    assert await store.load_tool_result_views(conversation.id) == ()
    await store.save_history_state(
        conversation.id, raw, summary_state=_summary(), tool_result_views=_views(raw),
    )
    assert await store.delete(conversation.id)
    assert await store.load_summary_state(conversation.id) is None
    assert await store.load_tool_result_views(conversation.id) == ()


# 函数说明：test_legacy_conversation_loads_without_view_migration_rewrites
# 用途：回归验证回归测试与测试辅助中的
# `legacy_conversation_loads_without_view_migration_rewrites` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `_messages` → `store.create` → `store.load_tool_result_views` →
# `store.load_messages`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`await store.load_tool_result_views(conversation.id) == ()`。
#   验证条件：`ToolResultViewState(restored).project(restored) == raw`。
async def test_legacy_conversation_loads_without_view_migration_rewrites(
    tmp_path,
) -> None:
    store = SQLiteConversationStore(tmp_path / "legacy.db")
    await store.initialize()
    raw = _messages()
    conversation = await store.create(messages=raw)
    await store.initialize()
    assert await store.load_tool_result_views(conversation.id) == ()
    restored = await store.load_messages(conversation.id)
    assert ToolResultViewState(restored).project(restored) == raw
