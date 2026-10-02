"""Render verified context plans while preserving provenance and protocol."""

from __future__ import annotations

from dataclasses import replace

from app.models.types import MessageRole

from .blocks import BlockType
from .inventory import ContextEntry, ContextInventory
from .summary import ConversationSummaryState


# 函数说明：apply_summary
# 用途：应用摘要，供模型上下文与输入预算使用。
# 参数：
#   inventory：上下文清单输入或配置值，类型 `ContextInventory`。
#   state：当前状态快照，类型 `ConversationSummaryState`。
# 返回：类型 `ContextInventory`；按分支返回 `inventory`；
# `replace(inventory.with_entries(kept), summary_state=state)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_summary_protection` →
# `inventory.summary_state.summary.to_message` → `kept.insert` → `ContextEntry` →
# `state.summary.to_message` → `replace`；另有 1 个调用点。
# 分支与异常：
#   当 `covered > inventory.history_count` 时，抛出 `ValueError(…)`。
#   当 `not covered` 时，返回 `inventory`。
#   `unit.block_type is BlockType.TOOL_ROUND and covered_entries` 分支在完成前置处理后跳
# 过当前循环项。
#   当 `uncovered_history` 时，抛出
# `ValueError('summary watermark splits a complete tool round')`。
def apply_summary(
    inventory: ContextInventory,
    state: ConversationSummaryState,
) -> ContextInventory:
    """Replace a covered raw-history prefix without deleting current content.

    Unknown provenance and protected units remain intact.  A watermark through
    the middle of a complete protocol round is rejected instead of splitting it.
    """
    covered = state.covered_message_count
    if covered > inventory.history_count:
        raise ValueError("summary covered_message_count exceeds history length")
    if not covered:
        return inventory
    kept: list[ContextEntry] = []
    for unit in inventory.units:
        covered_entries = tuple(
            entry
            for entry in unit.entries
            if entry.source_index is not None
            and entry.source_index < covered
            and entry.message.role is not MessageRole.SYSTEM
        )
        if unit.block_type is BlockType.TOOL_ROUND and covered_entries:
            uncovered_history = any(
                entry.source_index is not None
                and covered <= entry.source_index < inventory.history_count
                for entry in unit.entries
            )
            if uncovered_history:
                raise ValueError("summary watermark splits a complete tool round")
            removable = not _summary_protection(unit.protected_reasons) and len(
                covered_entries
            ) == len(unit.entries)
            if removable:
                continue
            kept.extend(unit.entries)
            continue
        for entry in unit.entries:
            if (
                entry.source_index is None
                and inventory.summary_state is not None
                and entry.message == inventory.summary_state.summary.to_message()
            ):
                continue
            if (
                not unit.protected_reasons
                and entry.source_index is not None
                and entry.source_index < covered
                and entry.message.role is not MessageRole.SYSTEM
            ):
                continue
            kept.append(entry)

    insertion = 0
    while insertion < len(kept) and kept[insertion].message.role is MessageRole.SYSTEM:
        insertion += 1
    kept.insert(insertion, ContextEntry(state.summary.to_message(), None))
    # Runtime snapshots are backend-generated data. Never deduplicate named
    # canonical user/tool messages: their raw coordinates remain authoritative.
    task_entries = [
        entry
        for entry in kept
        if entry.source_index is None and entry.message.name == "muharness_active_task"
    ]
    if task_entries:
        latest_task = task_entries[-1]
        kept = [
            entry
            for entry in kept
            if entry.source_index is not None
            or entry.message.name != "muharness_active_task"
            or entry is latest_task
        ]
    return replace(inventory.with_entries(kept), summary_state=state)


# 函数说明：_summary_protection
# 用途：返回 `tuple((reason for reason in reasons if reason != 'recent_tool_round'))`，
# 提供 模型上下文与输入预算 的派生值。
# 参数：
#   reasons：`reasons`输入或配置值，类型 `tuple[str, ...]`。
# 返回：类型 `tuple[str, ...]`；返回
# `tuple((reason for reason in reasons if reason != 'recent_tool_round'))`。
def _summary_protection(reasons: tuple[str, ...]) -> tuple[str, ...]:
    # Recent-round protection belongs to tool selection.  The summary frontier
    # separately protects rounds relevant to the surviving recent conversation;
    # stale rounds before that frontier may be covered by the existing policy.
    return tuple(reason for reason in reasons if reason != "recent_tool_round")


__all__ = ["apply_summary"]
