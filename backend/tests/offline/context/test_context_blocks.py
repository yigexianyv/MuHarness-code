
from __future__ import annotations

from app.models.types import Message, MessageRole, ToolCall
from app.runtime.context import (
    BlockType,
    ConversationBlock,
    MalformedToolBlock,
    MessageBlock,
    SystemBlock,
    ToolRoundBlock,
    partition_messages,
)


# 函数说明：_system
# 用途：返回 `Message(role=MessageRole.SYSTEM, content=content)`，提供 回归测试与测试辅
# 助 的派生值。
# 参数：
#   content：内容正文，类型 `str`；默认 `'sys'`。
# 返回：类型 `Message`；返回 `Message(role=MessageRole.SYSTEM, content=content)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
def _system(content: str = "sys") -> Message:
    return Message(role=MessageRole.SYSTEM, content=content)


# 函数说明：_user
# 用途：返回 `Message(role=MessageRole.USER, content=content)`，提供 回归测试与测试辅助
# 的派生值。
# 参数：
#   content：内容正文，类型 `str`；默认 `'hi'`。
# 返回：类型 `Message`；返回 `Message(role=MessageRole.USER, content=content)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
def _user(content: str = "hi") -> Message:
    return Message(role=MessageRole.USER, content=content)


# 函数说明：_assistant
# 用途：返回
# `Message(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)`，提供 回
# 归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str | None`；默认 `'yo'`。
#   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`；默认 `()`。
# 返回：类型 `Message`；返回
# `Message(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
def _assistant(
    content: str | None = "yo",
    tool_calls: tuple[ToolCall, ...] = (),
) -> Message:
    return Message(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)


# 函数说明：_tool
# 用途：返回 `Message(role=MessageRole.TOOL, tool_call_id=call_id, name='web_search',
# content=content)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   call_id：工具调用标识，类型 `str`。
#   content：内容正文，类型 `str`；默认 `'result'`。
# 返回：类型 `Message`；返回 `Message(role=MessageRole.TOOL, tool_call_id=call_id, name=
# 'web_search', content=content)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
def _tool(call_id: str, content: str = "result") -> Message:
    return Message(
        role=MessageRole.TOOL,
        tool_call_id=call_id,
        name="web_search",
        content=content,
    )


# 函数说明：_types
# 用途：返回 `[block.block_type for block in blocks]`，提供 回归测试与测试辅助 的派生值
# 。
# 参数：
#   blocks：`blocks`输入或配置值，类型 `tuple[MessageBlock, ...]`。
# 返回：类型 `list[BlockType]`；返回 `[block.block_type for block in blocks]`。
def _types(blocks: tuple[MessageBlock, ...]) -> list[BlockType]:
    return [block.block_type for block in blocks]


# 函数说明：test_partition_system_and_conversation
# 用途：回归验证回归测试与测试辅助中的 `partition_system_and_conversation` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_system` → `_user` → `_assistant` →
# `partition_messages` → `_types`。
# 分支与异常：
#   验证条件：`_types(blocks) == [BlockType.SYSTEM, BlockType.CONVERSATION]`。
#   验证条件：`isinstance(blocks[0], SystemBlock)`。
#   验证条件：`isinstance(blocks[1], ConversationBlock)`。
#   验证条件：`blocks[0].messages == (_system('你是助手'),)`。
def test_partition_system_and_conversation() -> None:
    messages = (_system("你是助手"), _user("你好"), _assistant("你好！"))

    blocks = partition_messages(messages)

    assert _types(blocks) == [BlockType.SYSTEM, BlockType.CONVERSATION]
    assert isinstance(blocks[0], SystemBlock)
    assert isinstance(blocks[1], ConversationBlock)
    assert blocks[0].messages == (_system("你是助手"),)
    assert blocks[1].messages == (_user("你好"), _assistant("你好！"))


# 函数说明：test_partition_tool_round
# 用途：回归验证回归测试与测试辅助中的 `partition_tool_round` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `_system` → `_user` →
# `_assistant` → `_tool` → `partition_messages`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`_types(blocks) == [BlockType.SYSTEM, BlockType.CONVERSATION, BlockType.
# TOOL_ROUND, BlockType.CONVERSATION]`。
#   验证条件：`isinstance(blocks[2], ToolRoundBlock)`。
#   验证条件：`blocks[2].messages == (_assistant(tool_calls=(call,)), _tool('c1'))`。
#   验证条件：`blocks[3].messages == (_assistant('结果如下'),)`。
def test_partition_tool_round() -> None:
    call = ToolCall(id="c1", name="web_search", arguments={"query": "AI"})
    messages = (
        _system("sys"),
        _user("搜索"),
        _assistant(tool_calls=(call,)),
        _tool("c1"),
        _assistant("结果如下"),
    )

    blocks = partition_messages(messages)

    assert _types(blocks) == [
        BlockType.SYSTEM,
        BlockType.CONVERSATION,
        BlockType.TOOL_ROUND,
        BlockType.CONVERSATION,
    ]
    assert isinstance(blocks[2], ToolRoundBlock)
    assert blocks[2].messages == (_assistant(tool_calls=(call,)), _tool("c1"))
    assert blocks[3].messages == (_assistant("结果如下"),)


# 函数说明：test_consecutive_system_messages_merge
# 用途：回归验证回归测试与测试辅助中的 `consecutive_system_messages_merge` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`partition_messages` → `_system` →
# `_user` → `_types`。
# 分支与异常：
#   验证条件：`isinstance(blocks[0], SystemBlock)`。
#   验证条件：`len(blocks[0]) == 2`。
#   验证条件：`_types(blocks) == [BlockType.SYSTEM, BlockType.CONVERSATION]`。
def test_consecutive_system_messages_merge() -> None:
    blocks = partition_messages((_system("a"), _system("b"), _user("x")))

    assert isinstance(blocks[0], SystemBlock)
    assert len(blocks[0]) == 2
    assert _types(blocks) == [BlockType.SYSTEM, BlockType.CONVERSATION]


# 函数说明：test_multiple_tool_rounds_are_separate_blocks
# 用途：回归验证回归测试与测试辅助中的 `multiple_tool_rounds_are_separate_blocks` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `_assistant` → `_tool` →
# `partition_messages` → `_types`。
# 分支与异常：
#   验证条件：`_types(blocks) == [BlockType.TOOL_ROUND, BlockType.TOOL_ROUND]`。
#   验证条件：`len(blocks[0]) == 2`。
#   验证条件：`len(blocks[1]) == 2`。
def test_multiple_tool_rounds_are_separate_blocks() -> None:
    call1 = ToolCall(id="c1", name="t", arguments={})
    call2 = ToolCall(id="c2", name="t", arguments={})
    messages = (
        _assistant(tool_calls=(call1,)),
        _tool("c1"),
        _assistant(tool_calls=(call2,)),
        _tool("c2"),
    )

    blocks = partition_messages(messages)

    assert _types(blocks) == [BlockType.TOOL_ROUND, BlockType.TOOL_ROUND]
    assert len(blocks[0]) == 2
    assert len(blocks[1]) == 2


# 函数说明：test_multi_tool_round_requires_all_matching_results
# 用途：回归验证回归测试与测试辅助中的 `multi_tool_round_requires_all_matching_results`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `_assistant` → `_tool` →
# `partition_messages`。
# 分支与异常：
#   验证条件：`len(blocks) == 1`。
#   验证条件：`isinstance(blocks[0], ToolRoundBlock)`。
#   验证条件：`blocks[0].messages == messages`。
def test_multi_tool_round_requires_all_matching_results() -> None:
    call1 = ToolCall(id="c1", name="first", arguments={})
    call2 = ToolCall(id="c2", name="second", arguments={})
    messages = (
        _assistant(tool_calls=(call1, call2)),
        _tool("c2"),
        _tool("c1"),
    )

    blocks = partition_messages(messages)

    assert len(blocks) == 1
    assert isinstance(blocks[0], ToolRoundBlock)
    assert blocks[0].messages == messages


# 函数说明：test_mismatched_tool_result_becomes_malformed_block
# 用途：回归验证回归测试与测试辅助中的 `mismatched_tool_result_becomes_malformed_block`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `_assistant` → `_tool` →
# `partition_messages`。
# 分支与异常：
#   验证条件：`len(blocks) == 1`。
#   验证条件：`isinstance(blocks[0], MalformedToolBlock)`。
#   验证条件：`blocks[0].messages == messages`。
#   验证条件：`'exactly match' in blocks[0].reason`。
def test_mismatched_tool_result_becomes_malformed_block() -> None:
    call = ToolCall(id="expected", name="search", arguments={})
    messages = (_assistant(tool_calls=(call,)), _tool("unexpected"))

    blocks = partition_messages(messages)

    assert len(blocks) == 1
    assert isinstance(blocks[0], MalformedToolBlock)
    assert blocks[0].messages == messages
    assert "exactly match" in blocks[0].reason


# 函数说明：test_incomplete_tool_round_becomes_malformed_block
# 用途：回归验证回归测试与测试辅助中的 `incomplete_tool_round_becomes_malformed_block`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `_assistant` →
# `partition_messages`。
# 分支与异常：
#   验证条件：`isinstance(blocks[0], MalformedToolBlock)`。
#   验证条件：`blocks[0].messages == (message,)`。
def test_incomplete_tool_round_becomes_malformed_block() -> None:
    call = ToolCall(id="missing-result", name="search", arguments={})
    message = _assistant(tool_calls=(call,))

    blocks = partition_messages((message,))

    assert isinstance(blocks[0], MalformedToolBlock)
    assert blocks[0].messages == (message,)


# 函数说明：test_orphan_tool_result_becomes_malformed_block
# 用途：回归验证回归测试与测试辅助中的 `orphan_tool_result_becomes_malformed_block` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool` → `partition_messages` →
# `_user` → `_assistant` → `_types`。
# 分支与异常：
#   验证条件：`_types(blocks) == [BlockType.CONVERSATION, BlockType.MALFORMED_TOOL,
# BlockType.CONVERSATION]`。
#   验证条件：`isinstance(blocks[1], MalformedToolBlock)`。
#   验证条件：`blocks[1].messages == (message,)`。
def test_orphan_tool_result_becomes_malformed_block() -> None:
    message = _tool("orphan")

    blocks = partition_messages((_user(), message, _assistant()))

    assert _types(blocks) == [
        BlockType.CONVERSATION,
        BlockType.MALFORMED_TOOL,
        BlockType.CONVERSATION,
    ]
    assert isinstance(blocks[1], MalformedToolBlock)
    assert blocks[1].messages == (message,)


# 函数说明：test_conversation_turns_split_on_new_user
# 用途：回归验证回归测试与测试辅助中的 `conversation_turns_split_on_new_user` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_user` → `_assistant` →
# `partition_messages` → `_types`。
# 分支与异常：
#   验证条件：`_types(blocks) == [BlockType.CONVERSATION, BlockType.CONVERSATION]`。
#   验证条件：`blocks[0].messages == (_user('u1'), _assistant('a1'))`。
#   验证条件：`blocks[1].messages == (_user('u2'), _assistant('a2'))`。
def test_conversation_turns_split_on_new_user() -> None:
    messages = (_user("u1"), _assistant("a1"), _user("u2"), _assistant("a2"))

    blocks = partition_messages(messages)

    assert _types(blocks) == [BlockType.CONVERSATION, BlockType.CONVERSATION]
    assert blocks[0].messages == (_user("u1"), _assistant("a1"))
    assert blocks[1].messages == (_user("u2"), _assistant("a2"))


# 函数说明：test_empty_messages_partition_to_empty
# 用途：回归验证回归测试与测试辅助中的 `empty_messages_partition_to_empty` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`partition_messages`。
# 分支与异常：
#   验证条件：`partition_messages([]) == ()`。
def test_empty_messages_partition_to_empty() -> None:
    assert partition_messages([]) == ()


# 函数说明：test_partition_preserves_message_order
# 用途：回归验证回归测试与测试辅助中的 `partition_preserves_message_order` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_system` → `_user` → `_assistant` →
# `ToolCall` → `_tool` → `partition_messages`。
# 分支与异常：
#   验证条件：`reconstructed == messages`。
def test_partition_preserves_message_order() -> None:
    messages = (
        _system("s"),
        _user("u1"),
        _assistant("a1"),
        _assistant(tool_calls=(ToolCall(id="c1", name="t", arguments={}),)),
        _tool("c1"),
        _user("u2"),
        _assistant("a2"),
    )

    blocks = partition_messages(messages)
    reconstructed = tuple(message for block in blocks for message in block.messages)

    assert reconstructed == messages
