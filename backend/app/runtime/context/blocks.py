
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.models.types import Message, MessageRole


class BlockType(StrEnum):
    SYSTEM = "system"
    CONVERSATION = "conversation"
    TOOL_ROUND = "tool_round"
    MALFORMED_TOOL = "malformed_tool"


@dataclass(frozen=True)
class MessageBlock:

    messages: tuple[Message, ...]

    # 函数说明：MessageBlock.block_type
    # 用途：处理模型上下文与输入预算中的 `block_type` 数据；结果及边界条件见下方说明。
    # 返回：类型 `BlockType`；不返回结果值（隐式 None）。
    @property
    def block_type(self) -> BlockType:
        raise NotImplementedError

    # 函数说明：MessageBlock.__len__
    # 用途：返回 `len(self.messages)`，提供 MessageBlock 的派生值。
    # 返回：类型 `int`；返回 `len(self.messages)`。
    def __len__(self) -> int:
        return len(self.messages)


@dataclass(frozen=True)
class SystemBlock(MessageBlock):

    # 函数说明：SystemBlock.block_type
    # 用途：返回 `BlockType.SYSTEM`，提供 SystemBlock 的派生值。
    # 返回：类型 `BlockType`；返回 `BlockType.SYSTEM`。
    @property
    def block_type(self) -> BlockType:
        return BlockType.SYSTEM


@dataclass(frozen=True)
class ConversationBlock(MessageBlock):

    # 函数说明：ConversationBlock.block_type
    # 用途：返回 `BlockType.CONVERSATION`，提供 ConversationBlock 的派生值。
    # 返回：类型 `BlockType`；返回 `BlockType.CONVERSATION`。
    @property
    def block_type(self) -> BlockType:
        return BlockType.CONVERSATION


@dataclass(frozen=True)
class ToolRoundBlock(MessageBlock):

    # 函数说明：ToolRoundBlock.__post_init__
    # 用途：处理模型上下文与输入预算中的 `__post_init__` 数据；结果及边界条件见下方说明
    # 。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Counter`。
    # 分支与异常：
    #   当 `not self.messages` 时，抛出
    # `ValueError('ToolRoundBlock messages cannot be empty')`。
    #   当 `assistant.role is not MessageRole.ASSISTANT or not…` 时，抛出
    # `ValueError(…)`。
    #   当 `any((not call_id for call_id in expected_ids))` 时，抛出
    # `ValueError('ToolCall id cannot be empty')`。
    #   当 `len(set(expected_ids)) != len(expected_ids)` 时，抛出 `ValueError(…)`。
    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("ToolRoundBlock messages cannot be empty")
        assistant = self.messages[0]
        if assistant.role is not MessageRole.ASSISTANT or not assistant.tool_calls:
            raise ValueError(
                "ToolRoundBlock must start with an assistant tool call message"
            )

        expected_ids = [call.id for call in assistant.tool_calls]
        if any(not call_id for call_id in expected_ids):
            raise ValueError("ToolCall id cannot be empty")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("ToolCall ids must be unique within one tool round")

        result_messages = self.messages[1:]
        if any(message.role is not MessageRole.TOOL for message in result_messages):
            raise ValueError("ToolRoundBlock may only contain trailing tool results")
        result_ids = [message.tool_call_id for message in result_messages]
        if any(not tool_call_id for tool_call_id in result_ids):
            raise ValueError("ToolResult message requires tool_call_id")
        if Counter(result_ids) != Counter(expected_ids):
            raise ValueError(
                "ToolResult tool_call_id values must exactly match ToolCall ids"
            )

    # 函数说明：ToolRoundBlock.block_type
    # 用途：返回 `BlockType.TOOL_ROUND`，提供 ToolRoundBlock 的派生值。
    # 返回：类型 `BlockType`；返回 `BlockType.TOOL_ROUND`。
    @property
    def block_type(self) -> BlockType:
        return BlockType.TOOL_ROUND


@dataclass(frozen=True)
class MalformedToolBlock(MessageBlock):

    reason: str = "malformed tool protocol"

    # 函数说明：MalformedToolBlock.block_type
    # 用途：返回 `BlockType.MALFORMED_TOOL`，提供 MalformedToolBlock 的派生值。
    # 返回：类型 `BlockType`；返回 `BlockType.MALFORMED_TOOL`。
    @property
    def block_type(self) -> BlockType:
        return BlockType.MALFORMED_TOOL


# 函数说明：partition_messages
# 用途：按系统消息、对话与工具轮次切分历史，保留压缩所需的结构边界。
# 参数：
#   messages：本次处理的消息序列，类型 `Sequence[Message]`。
# 返回：类型 `tuple[MessageBlock, ...]`；返回 `tuple(blocks)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`flush_conversation` → `SystemBlock` →
#  `ToolRoundBlock` → `MalformedToolBlock`。
# 分支与异常：
#   `message.role is MessageRole.SYSTEM` 分支在完成前置处理后跳过当前循环项。
#   `message.role is MessageRole.USER` 分支在完成前置处理后跳过当前循环项。
#   `message.role is MessageRole.ASSISTANT and message.tool_calls` 分支在完成前置处理后
# 跳过当前循环项。
#   捕获 `ValueError` 后，执行异常处理调用 `blocks.append`、`MalformedToolBlock`、
# `tuple`。
def partition_messages(
    messages: Sequence[Message],
) -> tuple[MessageBlock, ...]:

    """按系统消息、对话与工具轮次切分历史，保留压缩所需的结构边界。"""
    blocks: list[MessageBlock] = []
    conversation: list[Message] = []

    # 函数说明：partition_messages.flush_conversation
    # 用途：在模型上下文与输入预算中处理 `flush_conversation`，通过 `blocks.append` 完成
    # 首个内部处理步骤。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationBlock`。
    # 闭包依赖：从外层读取 `blocks`。
    def flush_conversation() -> None:
        nonlocal conversation
        if conversation:
            blocks.append(ConversationBlock(tuple(conversation)))
            conversation = []

    index = 0
    count = len(messages)
    while index < count:
        message = messages[index]
        if message.role is MessageRole.SYSTEM:
            flush_conversation()
            system = [message]
            index += 1
            while index < count and messages[index].role is MessageRole.SYSTEM:
                system.append(messages[index])
                index += 1
            blocks.append(SystemBlock(tuple(system)))
            continue
        if message.role is MessageRole.USER:
            flush_conversation()
            conversation.append(message)
            index += 1
            continue
        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            flush_conversation()
            tool_round = [message]
            index += 1
            while index < count and messages[index].role is MessageRole.TOOL:
                tool_round.append(messages[index])
                index += 1
            try:
                blocks.append(ToolRoundBlock(tuple(tool_round)))
            except ValueError as exc:
                blocks.append(
                    MalformedToolBlock(tuple(tool_round), reason=str(exc))
                )
            continue
        if message.role is MessageRole.TOOL:
            flush_conversation()
            orphan_results = [message]
            index += 1
            while index < count and messages[index].role is MessageRole.TOOL:
                orphan_results.append(messages[index])
                index += 1
            blocks.append(
                MalformedToolBlock(
                    tuple(orphan_results),
                    reason="orphan tool result without assistant tool call",
                )
            )
            continue
        conversation.append(message)
        index += 1

    flush_conversation()
    return tuple(blocks)


__all__ = [
    "BlockType",
    "ConversationBlock",
    "MalformedToolBlock",
    "MessageBlock",
    "SystemBlock",
    "ToolRoundBlock",
    "partition_messages",
]
