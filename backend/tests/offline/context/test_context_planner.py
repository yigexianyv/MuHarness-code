from app.models.types import Message, MessageRole, ToolCall
from app.runtime.context.inventory import ContextInventory
from app.runtime.context.planner import summary_frontier


# 函数说明：_round
# 用途：返回 `(Message(role=MessageRole.ASSISTANT, tool_calls=(ToolCall(id=str(index),
# name='read'),)),…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   index：当前位置或索引。
# 返回：返回 `(Message(role=MessageRole.ASSISTANT, tool_calls=(ToolCall(id=str(index),
# name='read'),)),…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolCall`。
def _round(index):
    return (
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=(ToolCall(id=str(index), name="read"),),
        ),
        Message(role=MessageRole.TOOL, tool_call_id=str(index), content="result"),
    )


# 函数说明：test_current_run_frontier_can_cross_pinned_user_and_completed_tools
# 用途：回归验证回归测试与测试辅助中的
# `current_run_frontier_can_cross_pinned_user_and_completed_tools` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` →
# `ContextInventory.build` → `summary_frontier`。
# 分支与异常：
#   验证条件：`window.cutoff == 5`。
#   验证条件：
# `tuple((m for block in window.conversation_blocks for m in block)) == source[:5]`。
def test_current_run_frontier_can_cross_pinned_user_and_completed_tools() -> None:
    source = (
        Message(role=MessageRole.USER, content="user constraints"),
        *sum((_round(i) for i in range(4)), ()),
    )
    inventory = ContextInventory.build(
        source, history_count=len(source), protected_source_indices=(0,)
    )
    window = summary_frontier(
        inventory,
        covered_message_count=0,
        keep_recent_conversation_blocks=4,
        keep_recent_tool_rounds=2,
    )
    assert window.cutoff == 5
    assert tuple(m for block in window.conversation_blocks for m in block) == source[:5]


# 函数说明：test_malformed_pending_round_is_a_hard_frontier
# 用途：回归验证回归测试与测试辅助中的 `malformed_pending_round_is_a_hard_frontier` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_round` → `Message` → `ToolCall` →
# `summary_frontier` → `ContextInventory.build`。
# 分支与异常：
#   验证条件：`window.cutoff == 2`。
def test_malformed_pending_round_is_a_hard_frontier() -> None:
    source = (
        *_round(0),
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=(ToolCall(id="pending", name="write"),),
        ),
        Message(role=MessageRole.USER, content="later"),
    )
    window = summary_frontier(
        ContextInventory.build(source, history_count=len(source)),
        covered_message_count=0,
        keep_recent_conversation_blocks=0,
        keep_recent_tool_rounds=0,
    )
    assert window.cutoff == 2


# 函数说明：test_plain_chat_keeps_recent_conversation_blocks
# 用途：回归验证回归测试与测试辅助中的 `plain_chat_keeps_recent_conversation_blocks` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `summary_frontier` →
# `ContextInventory.build`。
# 分支与异常：
#   验证条件：`window.cutoff == 4`。
def test_plain_chat_keeps_recent_conversation_blocks() -> None:
    source = tuple(Message(role=MessageRole.USER, content=str(i)) for i in range(6))
    window = summary_frontier(
        ContextInventory.build(source, history_count=len(source)),
        covered_message_count=0,
        keep_recent_conversation_blocks=2,
        keep_recent_tool_rounds=2,
    )
    assert window.cutoff == 4


# 函数说明：test_frontier_is_based_on_source_not_runtime_injection_count
# 用途：回归验证回归测试与测试辅助中的
# `frontier_is_based_on_source_not_runtime_injection_count` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` →
# `ContextInventory.from_projection` → `summary_frontier`。
# 分支与异常：
#   验证条件：`inventory.entries[0].source_index is None`。
#   验证条件：`window.cutoff == 5`。
def test_frontier_is_based_on_source_not_runtime_injection_count() -> None:
    source = (
        Message(role=MessageRole.USER, content="go"),
        *sum((_round(i) for i in range(4)), ()),
    )
    injected = Message(
        role=MessageRole.USER, name="muharness_active_task", content="data"
    )
    inventory = ContextInventory.from_projection(
        source, (), (injected, *source), None, protected_source_indices=(0,)
    )
    assert inventory.entries[0].source_index is None
    window = summary_frontier(
        inventory,
        covered_message_count=0,
        keep_recent_conversation_blocks=4,
        keep_recent_tool_rounds=2,
    )
    assert window.cutoff == 5
