"""Choose one legal raw-history prefix for a pressure-triggered compaction."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.types import Message

from .blocks import BlockType, partition_messages
from .inventory import ContextInventory


@dataclass(frozen=True)
class SummaryWindow:
    cutoff: int
    # Complete source blocks, including tools. The historical attribute name is
    # retained for callers; tool evidence is no longer excluded from summaries.
    conversation_blocks: tuple[tuple[Message, ...], ...]


# 函数说明：summary_frontier
# 用途：在模型上下文与输入预算中处理 `summary_frontier`，通过 `starts.append` 完成首个内
# 部处理步骤。
# 参数：
#   inventory：上下文清单输入或配置值，类型 `ContextInventory`。
#   covered_message_count：传给 `SummaryWindow` 的输入，类型 `int`。
#   keep_recent_conversation_blocks：近期项会话输入或配置值，类型 `int`。
#   keep_recent_tool_rounds：近期项工具输入或配置值，类型 `int`。
# 返回：类型 `SummaryWindow`；按分支返回 `SummaryWindow(covered_message_count, ())`；
# `SummaryWindow(cutoff, source)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SummaryWindow` → `partition_messages`
#  → `barriers.update`。
# 分支与异常：
#   当 `covered_message_count < 0` 时，抛出
# `ValueError('covered_message_count cannot be negative')`。
#   当 `keep_recent_conversation_blocks < 0 or…` 时，抛出
# `ValueError('recent block counts cannot be negative')`。
#   当 `covered_message_count > len(history)` 时，返回
# `SummaryWindow(covered_message_count, ())`。
def summary_frontier(
    inventory: ContextInventory,
    *,
    covered_message_count: int,
    keep_recent_conversation_blocks: int,
    keep_recent_tool_rounds: int,
) -> SummaryWindow:
    if covered_message_count < 0:
        raise ValueError("covered_message_count cannot be negative")
    if keep_recent_conversation_blocks < 0 or keep_recent_tool_rounds < 0:
        raise ValueError("recent block counts cannot be negative")
    history = inventory.source_messages[: inventory.history_count]
    if covered_message_count > len(history):
        return SummaryWindow(covered_message_count, ())
    blocks = partition_messages(history[covered_message_count:])
    starts: list[int] = []
    position = covered_message_count
    for block in blocks:
        starts.append(position)
        position += len(block.messages)
    tools = [
        i for i, block in enumerate(blocks) if block.block_type is BlockType.TOOL_ROUND
    ]
    conversations = [
        i
        for i, block in enumerate(blocks)
        if block.block_type is BlockType.CONVERSATION
    ]
    latest_pin = max(inventory.protected_source_indices, default=-1)
    # Keep the user's request verbatim without blocking all later tool work.
    active_tool_work = latest_pin >= 0 and any(starts[i] > latest_pin for i in tools)
    if active_tool_work:
        conversations = [i for i in conversations if starts[i] > latest_pin]
    barriers = {
        i
        for i, block in enumerate(blocks)
        if block.block_type is BlockType.MALFORMED_TOOL
    }
    if keep_recent_conversation_blocks:
        barriers.update(conversations[-keep_recent_conversation_blocks:])
    if keep_recent_tool_rounds:
        eligible_tools = tools
        if (
            keep_recent_conversation_blocks
            and len(conversations) > keep_recent_conversation_blocks
        ):
            last_foldable = conversations[-keep_recent_conversation_blocks - 1]
            eligible_tools = [i for i in tools if i > last_foldable]
        barriers.update(eligible_tools[-keep_recent_tool_rounds:])
    cutoff = min((starts[i] for i in barriers), default=len(history))
    source = tuple(
        block.messages
        for i, block in enumerate(blocks)
        if block.block_type in {BlockType.CONVERSATION, BlockType.TOOL_ROUND}
        and starts[i] + len(block.messages) <= cutoff
    )
    return SummaryWindow(cutoff, source)
