import pytest

from app.models.types import Message, MessageRole, ToolCall
from app.runtime.context.inventory import ContextInventory
from app.runtime.context.projection import apply_summary
from app.runtime.context.summary import (
    ConversationSummaryState,
    RollingConversationSummary,
)


# 函数说明：_state
# 用途：返回 `ConversationSummaryState(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   covered：`covered`输入或配置值。
# 返回：返回 `ConversationSummaryState(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationSummaryState` →
# `RollingConversationSummary`。
def _state(covered):
    return ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="continue"),
        covered_message_count=covered,
    )


# 函数说明：_round
# 用途：返回 `(Message(role=MessageRole.ASSISTANT, tool_calls=(ToolCall(id=call_id, name
# ='read'),)),…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   call_id：工具调用标识。
# 返回：返回 `(Message(role=MessageRole.ASSISTANT, tool_calls=(ToolCall(id=call_id, name
# ='read'),)),…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolCall`。
def _round(call_id):
    return (
        Message(
            role=MessageRole.ASSISTANT, tool_calls=(ToolCall(id=call_id, name="read"),)
        ),
        Message(role=MessageRole.TOOL, tool_call_id=call_id, content="output"),
    )


# 函数说明：test_saved_summary_preserves_pinned_raw_user_without_shifting_source
# 用途：回归验证回归测试与测试辅助中的
# `saved_summary_preserves_pinned_raw_user_without_shifting_source` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` →
# `ContextInventory.build` → `_state`。
# 分支与异常：
#   验证条件：`inventory.source_messages == source`。
#   验证条件：`[entry.source_index for entry in inventory.entries] == [None, 0, 3, 4]`。
#   验证条件：`source[0] in inventory.messages`。
def test_saved_summary_preserves_pinned_raw_user_without_shifting_source() -> None:
    source = (
        Message(role=MessageRole.USER, content="exact constraints"),
        *_round("a"),
        *_round("b"),
    )
    inventory = ContextInventory.build(
        source,
        history_count=len(source),
        summary_state=_state(3),
        protected_source_indices=(0,),
    )
    assert inventory.source_messages == source
    assert [entry.source_index for entry in inventory.entries] == [None, 0, 3, 4]
    assert source[0] in inventory.messages


# 函数说明：test_invalid_saved_watermark_keeps_original_protocol
# 用途：回归验证回归测试与测试辅助中的 `invalid_saved_watermark_keeps_original_protocol`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   covered：传给 `_state` 的输入。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` →
# `ContextInventory.build` → `_state`。
# 分支与异常：
#   验证条件：`inventory.messages == source`。
#   验证条件：`inventory.summary_state is None`。
@pytest.mark.parametrize("covered", [2, 6])
def test_invalid_saved_watermark_keeps_original_protocol(covered) -> None:
    source = (Message(role=MessageRole.USER, content="go"), *_round("a"))
    inventory = ContextInventory.build(
        source, history_count=len(source), summary_state=_state(covered)
    )
    assert inventory.messages == source
    assert inventory.summary_state is None


# 函数说明：test_fixed_tool_projection_preserves_raw_coordinates_and_metadata
# 用途：回归验证回归测试与测试辅助中的
# `fixed_tool_projection_preserves_raw_coordinates_and_metadata` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_round` → `Message` →
# `ContextInventory.from_projection`。
# 分支与异常：
#   验证条件：
# `[entry.source_index for entry in inventory.entries] == [None, 0, 1, 2, 3]`。
#   验证条件：`inventory.source_messages[1].content == 'output'`。
#   验证条件：`inventory.entries[2].message.tool_call_id == 'a'`。
def test_fixed_tool_projection_preserves_raw_coordinates_and_metadata() -> None:
    source = (*_round("a"), *_round("b"))
    view = source[1].model_copy(update={"content": "fixed evidence reference"})
    runtime = Message(role=MessageRole.SYSTEM, content="runtime instructions")
    inventory = ContextInventory.from_projection(
        source, (), (runtime, source[0], view, *source[2:]), None
    )
    assert [entry.source_index for entry in inventory.entries] == [None, 0, 1, 2, 3]
    assert inventory.source_messages[1].content == "output"
    assert inventory.entries[2].message.tool_call_id == "a"


# 函数说明：test_repeated_user_text_is_aligned_by_complete_order
# 用途：回归验证回归测试与测试辅助中的 `repeated_user_text_is_aligned_by_complete_order`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` →
# `ContextInventory.from_projection`。
# 分支与异常：
#   验证条件：`[entry.source_index for entry in inventory.entries] == [0, 1, 2]`。
def test_repeated_user_text_is_aligned_by_complete_order() -> None:
    repeated = Message(role=MessageRole.USER, content="same")
    source = (repeated, Message(role=MessageRole.ASSISTANT, content="first"), repeated)
    inventory = ContextInventory.from_projection(
        source, (), tuple(m.model_copy() for m in source), None
    )
    assert [entry.source_index for entry in inventory.entries] == [0, 1, 2]


# 函数说明：test_reused_tool_ids_do_not_advance_watermark_by_id
# 用途：回归验证回归测试与测试辅助中的 `reused_tool_ids_do_not_advance_watermark_by_id`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_round` → `Message` →
# `ContextInventory.build` → `apply_summary` → `_state`。
# 分支与异常：
#   验证条件：`projected.messages[-2:] == source[-2:]`。
#   验证条件：`[entry.source_index for entry in projected.entries][-2:] == [3, 4]`。
def test_reused_tool_ids_do_not_advance_watermark_by_id() -> None:
    source = (
        *_round("same"),
        Message(role=MessageRole.USER, content="next"),
        *_round("same"),
    )
    inventory = ContextInventory.build(
        source, history_count=len(source), keep_recent_tool_rounds=0
    )
    projected = apply_summary(inventory, _state(2))
    assert projected.messages[-2:] == source[-2:]
    assert [entry.source_index for entry in projected.entries][-2:] == [3, 4]


# 函数说明：test_summary_watermark_cannot_split_a_complete_tool_round
# 用途：回归验证回归测试与测试辅助中的
# `summary_watermark_cannot_split_a_complete_tool_round` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_round` → `ContextInventory.build` →
# `pytest.raises` → `apply_summary` → `_state`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='splits')`。
def test_summary_watermark_cannot_split_a_complete_tool_round() -> None:
    source = (*_round("a"), *_round("b"))
    inventory = ContextInventory.build(
        source, history_count=len(source), keep_recent_tool_rounds=0
    )
    with pytest.raises(ValueError, match="splits"):
        apply_summary(inventory, _state(1))


# 函数说明：test_pinned_indices_must_reference_canonical_source
# 用途：回归验证回归测试与测试辅助中的 `pinned_indices_must_reference_canonical_source`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   index：当前位置或索引。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `ContextInventory.build` → `Message`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='protected_source_indices')`。
@pytest.mark.parametrize("index", [-1, 1])
def test_pinned_indices_must_reference_canonical_source(index) -> None:
    with pytest.raises(ValueError, match="protected_source_indices"):
        ContextInventory.build(
            (Message(role=MessageRole.USER, content="go"),),
            protected_source_indices=(index,),
        )


# 函数说明：test_unknown_runtime_messages_survive_compaction_without_becoming_sources
# 用途：回归验证回归测试与测试辅助中的
# `unknown_runtime_messages_survive_compaction_without_becoming_sources` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_round` → `Message` →
# `ContextInventory.from_projection` → `apply_summary` → `_state` → `next`。
# 分支与异常：
#   验证条件：`runtime in projected.messages`。
#   验证条件：`next((entry for entry in projected.entries if entry.message == runtime)).
# source_index is None`。
def test_unknown_runtime_messages_survive_compaction_without_becoming_sources() -> None:
    source = (*_round("a"), *_round("b"))
    runtime = Message(
        role=MessageRole.USER, name="runtime-data", content="not authorization"
    )
    inventory = ContextInventory.from_projection(
        source, (), (runtime, *source), None, keep_recent_tool_rounds=0
    )
    projected = apply_summary(inventory, _state(2))
    assert runtime in projected.messages
    assert (
        next(
            entry for entry in projected.entries if entry.message == runtime
        ).source_index
        is None
    )


# 函数说明：test_task_insertion_preserves_repeated_round_coordinates
# 用途：回归验证回归测试与测试辅助中的
# `task_insertion_preserves_repeated_round_coordinates` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` →
# `ContextInventory.from_projection`。
# 分支与异常：
#   验证条件：
# `[entry.source_index for entry in inventory.entries] == [None, *range(len(source))]`。
def test_task_insertion_preserves_repeated_round_coordinates() -> None:
    user = Message(role=MessageRole.USER, content="current goal")
    source = (user, *sum((_round("call_0") for _ in range(6)), ()))
    task = Message(
        role=MessageRole.USER, name="muharness_active_task", content="progress"
    )
    prepared = (task, *source)
    inventory = ContextInventory.from_projection(
        source, (), prepared, None, protected_source_indices=(0,)
    )
    assert [entry.source_index for entry in inventory.entries] == [
        None,
        *range(len(source)),
    ]


# 函数说明：test_named_canonical_system_is_preserved_covered_or_uncovered
# 用途：回归验证回归测试与测试辅助中的
# `named_canonical_system_is_preserved_covered_or_uncovered` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   covered：传给 `_state` 的输入。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_state` →
# `ContextInventory.build` → `apply_summary` → `next`。
# 分支与异常：
#   验证条件：`rule in saved.messages`。
#   验证条件：`rule in projected.messages`。
#   验证条件：`next((entry for entry in projected.entries if entry.message == rule)).
# source_index == 1`。
@pytest.mark.parametrize("covered", [1, 2])
def test_named_canonical_system_is_preserved_covered_or_uncovered(covered) -> None:
    rule = Message(
        role=MessageRole.SYSTEM,
        name="muharness_rolling_summary",
        content="Permanent rule: never disclose credentials.",
    )
    source = (
        Message(role=MessageRole.USER, content="old"),
        rule,
        Message(role=MessageRole.USER, content="current"),
    )
    state = _state(covered)
    saved = ContextInventory.build(
        source,
        history_count=len(source),
        summary_state=state,
        protected_source_indices=(2,),
    )
    assert rule in saved.messages
    raw = ContextInventory.build(
        source, history_count=len(source), protected_source_indices=(2,)
    )
    projected = apply_summary(raw, state)
    assert rule in projected.messages
    assert (
        next(entry for entry in projected.entries if entry.message == rule).source_index
        == 1
    )
