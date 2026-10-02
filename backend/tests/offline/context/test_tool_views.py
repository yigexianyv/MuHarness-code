from __future__ import annotations

import json

import pytest

from app.models.types import Message, MessageRole
from app.runtime.context.tool_views import (
    ToolResultView,
    ToolResultViewState,
    raw_message_sha256,
)


# 函数说明：_tool
# 用途：返回 `Message(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   output：工具、模型或转换步骤的输出，类型 `str`。
#   **metadata：额外关键字参数，按实现处理或转交。
# 返回：类型 `Message`；返回 `Message(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `json.dumps`。
def _tool(output: str, **metadata) -> Message:
    return Message(
        role=MessageRole.TOOL,
        tool_call_id="reused-call",
        name="execute",
        content=json.dumps({"output": output, "success": True, **metadata}),
    )


# 函数说明：test_first_projection_freezes_excerpt_and_preserves_raw_envelope
# 用途：回归验证回归测试与测试辅助中的
# `first_projection_freezes_excerpt_and_preserves_raw_envelope` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `ToolResultViewState` →
# `state.project` → `json.loads` → `json.dumps` → `envelope['output'].startswith`；另有
# 1 个调用点。
# 分支与异常：
#   验证条件：`projected is not raw`。
#   验证条件：`raw.content == json.dumps(original)`。
#   验证条件：`envelope['output'].startswith(output[:20])`。
#   验证条件：`envelope['output'].endswith(output[-10:])`。
def test_first_projection_freezes_excerpt_and_preserves_raw_envelope() -> None:
    output = "HEAD" + "中文正文" * 300 + "TAIL"
    raw = _tool(
        output, success=False, error="failed", evidence_id="evidence-1",
        output_chars=len(output), output_sha256="original-digest",
        output_truncated=False, evidence_error=None,
    )
    state = ToolResultViewState((), max_output_chars=100, head_chars=20, tail_chars=10)

    projected = state.project((raw,))[0]
    envelope = json.loads(projected.content)
    original = json.loads(raw.content)
    assert projected is not raw
    assert raw.content == json.dumps(original)
    assert envelope["output"].startswith(output[:20])
    assert envelope["output"].endswith(output[-10:])
    assert "characters omitted from the middle" in envelope["output"]
    assert len(envelope["output"]) < len(output)
    assert {key: value for key, value in envelope.items() if key != "output"} == {
        key: value for key, value in original.items() if key != "output"
    }
    assert projected.tool_call_id == raw.tool_call_id
    assert projected.name == raw.name
    assert state.project((raw,))[0].content == projected.content


# 函数说明：test_restored_excerpt_is_byte_identical_after_limit_changes
# 用途：回归验证回归测试与测试辅助中的
# `restored_excerpt_is_byte_identical_after_limit_changes` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `ToolResultViewState` →
# `state.project` → `state.snapshot` → `restored.project` → `restored.snapshot`。
# 分支与异常：
#   验证条件：`restored.project((raw,))[0].content == content`。
#   验证条件：`restored.snapshot((raw,)) == state.snapshot((raw,))`。
def test_restored_excerpt_is_byte_identical_after_limit_changes() -> None:
    raw = _tool("x" * 1000)
    state = ToolResultViewState((), max_output_chars=100, head_chars=20, tail_chars=10)
    content = state.project((raw,))[0].content
    restored = ToolResultViewState(
        (raw,), state.snapshot((raw,)), max_output_chars=10, head_chars=1, tail_chars=1,
    )
    assert restored.project((raw,))[0].content == content
    assert restored.snapshot((raw,)) == state.snapshot((raw,))


# 函数说明：test_first_full_decision_is_not_retroactively_shortened
# 用途：回归验证回归测试与测试辅助中的
# `first_full_decision_is_not_retroactively_shortened` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `ToolResultViewState` →
# `state.project` → `state.snapshot` → `restored.project`。
# 分支与异常：
#   验证条件：`state.project((raw,)) == (raw,)`。
#   验证条件：`views[0].representation == 'full'`。
#   验证条件：`views[0].content is None`。
#   验证条件：`restored.project((raw,)) == (raw,)`。
def test_first_full_decision_is_not_retroactively_shortened() -> None:
    raw = _tool("x" * 500)
    state = ToolResultViewState((), max_output_chars=1000)
    assert state.project((raw,)) == (raw,)
    views = state.snapshot((raw,))
    assert views[0].representation == "full"
    assert views[0].content is None
    restored = ToolResultViewState((raw,), views, max_output_chars=1, head_chars=0)
    assert restored.project((raw,)) == (raw,)


# 函数说明：test_legacy_history_without_views_is_conservatively_full
# 用途：回归验证回归测试与测试辅助中的
# `legacy_history_without_views_is_conservatively_full` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `ToolResultViewState` →
# `state.project` → `state.snapshot`。
# 分支与异常：
#   验证条件：`state.project((raw,)) == (raw,)`。
#   验证条件：`state.snapshot((raw,))[0].representation == 'full'`。
def test_legacy_history_without_views_is_conservatively_full() -> None:
    raw = _tool("x" * 1000)
    state = ToolResultViewState((raw,), max_output_chars=100, head_chars=10)
    assert state.project((raw,)) == (raw,)
    assert state.snapshot((raw,))[0].representation == "full"


# 函数说明：test_same_call_id_at_different_sequences_has_independent_decisions
# 用途：回归验证回归测试与测试辅助中的
# `same_call_id_at_different_sequences_has_independent_decisions` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `Message` →
# `ToolResultViewState` → `state.project` → `state.snapshot`。
# 分支与异常：
#   验证条件：`projected[0] != first`。
#   验证条件：`projected[2] == second`。
#   验证条件：`[view.source_sequence for view in state.snapshot(raw)] == [0, 2]`。
#   验证条件：
# `[view.representation for view in state.snapshot(raw)] == ['excerpt', 'full']`。
def test_same_call_id_at_different_sequences_has_independent_decisions() -> None:
    first = _tool("a" * 1000)
    second = _tool("short")
    user = Message(role=MessageRole.USER, content="again")
    raw = (first, user, second)
    state = ToolResultViewState((), max_output_chars=100, head_chars=10, tail_chars=10)
    projected = state.project(raw)
    assert projected[0] != first
    assert projected[2] == second
    assert [view.source_sequence for view in state.snapshot(raw)] == [0, 2]
    assert [view.representation for view in state.snapshot(raw)] == ["excerpt", "full"]


# 函数说明：test_changed_raw_hash_does_not_reuse_old_excerpt
# 用途：回归验证回归测试与测试辅助中的 `changed_raw_hash_does_not_reuse_old_excerpt` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `ToolResultViewState` →
# `state.project` → `state.snapshot` → `restored.project` → `restored.snapshot`；另有 1
# 个调用点。
# 分支与异常：
#   验证条件：`restored.project((new,)) == (new,)`。
#   验证条件：
# `restored.snapshot((new,))[0].raw_message_sha256 == raw_message_sha256(new)`。
#   验证条件：`restored.snapshot((new,))[0].representation == 'full'`。
def test_changed_raw_hash_does_not_reuse_old_excerpt() -> None:
    old = _tool("a" * 1000)
    state = ToolResultViewState((), max_output_chars=100, head_chars=10, tail_chars=10)
    state.project((old,))
    new = _tool("b" * 1000)
    restored = ToolResultViewState((new,), state.snapshot((old,)), max_output_chars=1)
    assert restored.project((new,)) == (new,)
    assert restored.snapshot((new,))[0].raw_message_sha256 == raw_message_sha256(new)
    assert restored.snapshot((new,))[0].representation == "full"


# 函数说明：test_non_envelope_tool_results_remain_full
# 用途：回归验证回归测试与测试辅助中的 `non_envelope_tool_results_remain_full` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   content：内容正文。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolResultViewState` →
# `state.project` → `state.snapshot`。
# 分支与异常：
#   验证条件：`state.project((raw,)) == (raw,)`。
#   验证条件：`state.snapshot((raw,))[0].representation == 'full'`。
@pytest.mark.parametrize(
    "content", [None, "x" * 1000, "{broken", "[]", '{"output":42}'],
)
def test_non_envelope_tool_results_remain_full(content) -> None:
    raw = Message(role=MessageRole.TOOL, content=content, tool_call_id="call")
    state = ToolResultViewState((), max_output_chars=1, head_chars=0, tail_chars=0)
    assert state.project((raw,)) == (raw,)
    assert state.snapshot((raw,))[0].representation == "full"


# 函数说明：test_snapshot_filters_views_for_replaced_or_removed_messages
# 用途：回归验证回归测试与测试辅助中的
# `snapshot_filters_views_for_replaced_or_removed_messages` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `ToolResultViewState` →
# `state.project` → `state.snapshot`。
# 分支与异常：
#   验证条件：`len(state.snapshot((first,))) == 1`。
#   验证条件：`state.snapshot((second,)) == ()`。
def test_snapshot_filters_views_for_replaced_or_removed_messages() -> None:
    first = _tool("a" * 1000)
    second = _tool("b" * 1000)
    state = ToolResultViewState((), max_output_chars=100, head_chars=10, tail_chars=10)
    state.project((first, second))
    assert len(state.snapshot((first,))) == 1
    assert state.snapshot((second,)) == ()


# 函数说明：test_restored_stale_future_view_cannot_control_new_results
# 用途：回归验证回归测试与测试辅助中的
# `restored_stale_future_view_cannot_control_new_results` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `ToolResultViewState` →
# `prior.project` → `prior.snapshot` → `restored.project` → `restored.snapshot`。
# 分支与异常：
#   验证条件：`restored.project((raw,)) == (raw,)`。
#   验证条件：`restored.snapshot((raw,))[0].representation == 'full'`。
def test_restored_stale_future_view_cannot_control_new_results() -> None:
    raw = _tool("x" * 1000)
    prior = ToolResultViewState((), max_output_chars=100, head_chars=10, tail_chars=10)
    prior.project((raw,))
    stale = prior.snapshot((raw,))[0]
    restored = ToolResultViewState((), (stale,), max_output_chars=2000)
    assert restored.project((raw,)) == (raw,)
    assert restored.snapshot((raw,))[0].representation == "full"


# 函数说明：test_view_identity_covers_raw_protocol_metadata_and_reasoning
# 用途：回归验证回归测试与测试辅助中的
# `view_identity_covers_raw_protocol_metadata_and_reasoning` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `raw_message_sha256`。
# 分支与异常：
#   验证条件：`raw_message_sha256(raw) == raw_message_sha256(raw.model_copy())`。
#   验证条件：`raw_message_sha256(raw) != raw_message_sha256(raw.model_copy(update={'
# reasoning': 'new'}))`。
#   验证条件：`raw_message_sha256(raw) != raw_message_sha256(raw.model_copy(update={'
# tool_call_id': 'other'}))`。
def test_view_identity_covers_raw_protocol_metadata_and_reasoning() -> None:
    raw = _tool("body")
    assert raw_message_sha256(raw) == raw_message_sha256(raw.model_copy())
    assert raw_message_sha256(raw) != raw_message_sha256(
        raw.model_copy(update={"reasoning": "new"})
    )
    assert raw_message_sha256(raw) != raw_message_sha256(
        raw.model_copy(update={"tool_call_id": "other"})
    )


# 函数说明：test_conflicting_or_invalid_views_are_rejected
# 用途：回归验证回归测试与测试辅助中的 `conflicting_or_invalid_views_are_rejected` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `ToolResultView` →
# `raw_message_sha256` → `pytest.raises` → `ToolResultViewState`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='conflicting')`。
#   预期异常：`pytest.raises(ValueError, match='excerpt must')`。
#   预期异常：`pytest.raises(ValueError, match='cannot be negative')`。
def test_conflicting_or_invalid_views_are_rejected() -> None:
    raw = _tool("body")
    full = ToolResultView(
        source_sequence=0, raw_message_sha256=raw_message_sha256(raw),
        representation="full",
    )
    excerpt = full.model_copy(update={"representation": "excerpt", "content": "{}"})
    with pytest.raises(ValueError, match="conflicting"):
        ToolResultViewState((raw,), (full, excerpt))
    with pytest.raises(ValueError, match="excerpt must"):
        ToolResultView(
            source_sequence=0, raw_message_sha256=raw_message_sha256(raw),
            representation="excerpt",
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        ToolResultViewState((), head_chars=-1)
