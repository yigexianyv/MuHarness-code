"""Source-addressed context units used by planning and request projection.

Coordinates refer to the unmodified input, never to a shortened request.  A
generated summary or a compatibility message with uncertain provenance has no
source coordinate and must not be mistaken for removable history.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass

from app.models.types import Message, MessageRole

from .blocks import BlockType, partition_messages
from .summary import ConversationSummaryState


@dataclass(frozen=True)
class ContextEntry:
    message: Message
    source_index: int | None


@dataclass(frozen=True)
class ContextUnit:
    index: int
    entries: tuple[ContextEntry, ...]
    block_type: BlockType
    protected_reasons: tuple[str, ...] = ()

    # 函数说明：ContextUnit.messages
    # 用途：返回 `tuple((entry.message for entry in self.entries))`，提供 ContextUnit 的
    # 派生值。
    # 返回：类型 `tuple[Message, ...]`；返回
    # `tuple((entry.message for entry in self.entries))`。
    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(entry.message for entry in self.entries)


@dataclass(frozen=True)
class ContextInventory:
    source_messages: tuple[Message, ...]
    history_count: int
    entries: tuple[ContextEntry, ...]
    units: tuple[ContextUnit, ...]
    keep_recent_tool_rounds: int = 2
    protect_current: bool = False
    summary_state: ConversationSummaryState | None = None
    protected_source_indices: tuple[int, ...] = ()

    # 函数说明：ContextInventory.messages
    # 用途：返回 `tuple((entry.message for entry in self.entries))`，提供
    # ContextInventory 的派生值。
    # 返回：类型 `tuple[Message, ...]`；返回
    # `tuple((entry.message for entry in self.entries))`。
    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(entry.message for entry in self.entries)

    # 函数说明：ContextInventory.summary_applied
    # 用途：在模型上下文与输入预算中处理 `summary_applied`，通过
    # `self.summary_state.summary.to_message` 完成首个内部处理步骤。
    # 返回：类型 `bool`；按分支返回 `False`；`any(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self.summary_state.summary.to_message`。
    # 分支与异常：
    #   当 `self.summary_state is None` 时，返回 `False`。
    @property
    def summary_applied(self) -> bool:
        if self.summary_state is None:
            return False
        summary_message = self.summary_state.summary.to_message()
        return any(
            entry.source_index is None and entry.message == summary_message
            for entry in self.entries
        )

    # 函数说明：ContextInventory.raw_units
    # 用途：处理模型上下文与输入预算中的 `raw_units` 数据；结果及边界条件见下方说明。
    # 返回：类型 `tuple[ContextUnit, ...]`；返回 `_make_units(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_units` → `ContextEntry`。
    @property
    def raw_units(self) -> tuple[ContextUnit, ...]:
        """The original protocol boundaries survive request-side omissions."""
        return _make_units(
            tuple(
                ContextEntry(message, index)
                for index, message in enumerate(self.source_messages)
            ),
            history_count=self.history_count,
            keep_recent_tool_rounds=self.keep_recent_tool_rounds,
            protect_current=self.protect_current,
            protected_source_indices=self.protected_source_indices,
        )

    # 函数说明：ContextInventory.build
    # 用途：构建ContextInventory，供模型上下文与输入预算使用。
    # 参数：
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   history_count：原始历史消息数量，类型 `int`；默认 `0`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   keep_recent_tool_rounds：近期项工具输入或配置值，类型 `int`；默认 `2`。
    #   protect_current：`protect_current`输入或配置值，类型 `bool`；默认 `False`。
    #   protected_source_indices：必须保留的原始消息索引，类型 `Sequence[int]`；默认
    # `()`。
    # 返回：类型 `ContextInventory`；返回 `cls._with_entries(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextEntry` →
    # `_valid_watermark` → `summary_state.summary.to_message` → `cls._with_entries`。
    # 分支与异常：
    #   当 `not 0 <= history_count <= len(source)` 时，抛出 `ValueError(…)`。
    #   当 `keep_recent_tool_rounds < 0` 时，抛出
    # `ValueError('keep_recent_tool_rounds cannot be negative')`。
    #   当 `any(…)` 时，抛出 `ValueError(…)`。
    @classmethod
    def build(
        cls,
        messages: Sequence[Message],
        *,
        history_count: int = 0,
        summary_state: ConversationSummaryState | None = None,
        keep_recent_tool_rounds: int = 2,
        protect_current: bool = False,
        protected_source_indices: Sequence[int] = (),
    ) -> ContextInventory:
        source = tuple(messages)
        if not 0 <= history_count <= len(source):
            raise ValueError("history_count must be within the source messages")
        if keep_recent_tool_rounds < 0:
            raise ValueError("keep_recent_tool_rounds cannot be negative")
        protected = tuple(sorted(set(protected_source_indices)))
        if any(index < 0 or index >= len(source) for index in protected):
            raise ValueError(
                "protected_source_indices must address raw source messages"
            )
        entries = tuple(
            ContextEntry(message, index) for index, message in enumerate(source)
        )
        applied_state = None
        if (
            summary_state is not None
            and 0 < summary_state.covered_message_count <= history_count
            and _valid_watermark(
                source[:history_count], summary_state.covered_message_count
            )
        ):
            applied_state = summary_state
            covered = summary_state.covered_message_count
            entries = (
                *(
                    entry
                    for entry in entries[:covered]
                    if entry.message.role is MessageRole.SYSTEM
                ),
                ContextEntry(summary_state.summary.to_message(), None),
                *(
                    entry
                    for entry in entries[:covered]
                    if entry.source_index in protected
                    and entry.message.role is not MessageRole.SYSTEM
                ),
                *entries[covered:],
            )
        return cls._with_entries(
            source,
            history_count,
            entries,
            keep_recent_tool_rounds=keep_recent_tool_rounds,
            protect_current=protect_current,
            summary_state=applied_state,
            protected_source_indices=protected,
        )

    # 函数说明：ContextInventory.from_projection
    # 用途：在模型上下文与输入预算中处理 `from_projection`，通过 `cls.build` 完成首个内
    # 部处理步骤。
    # 参数：
    #   raw_history：传给 `tuple` 的输入，类型 `Sequence[Message]`。
    #   current_messages：传给 `tuple` 的输入，类型 `Sequence[Message]`。
    #   prepared_messages：传给 `tuple` 的输入，类型 `Sequence[Message]`。
    #   previous_state：状态输入或配置值，类型 `ConversationSummaryState | None`。
    #   keep_recent_tool_rounds：近期项工具输入或配置值，类型 `int`；默认 `2`。
    #   protected_source_indices：必须保留的原始消息索引，类型 `Sequence[int]`；默认
    # `()`。
    # 返回：类型 `ContextInventory`；返回 `cls._with_entries(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`cls.build` → `_compatible` →
    # `_map_ordered_projection` → `cls._with_entries` → `ContextEntry`。
    # 分支与异常：
    #   `ordered is not None` 分支在完成前置处理后返回 `cls._with_entries(…)`。
    @classmethod
    def from_projection(
        cls,
        raw_history: Sequence[Message],
        current_messages: Sequence[Message],
        prepared_messages: Sequence[Message],
        previous_state: ConversationSummaryState | None,
        *,
        keep_recent_tool_rounds: int = 2,
        protected_source_indices: Sequence[int] = (),
    ) -> ContextInventory:
        """Adapt legacy reducer inputs without relying on object identity.

        Current messages are anchored from the right before matching history.
        The remaining mapping is monotonic and only accepts unambiguous matches;
        uncertain provenance is retained instead of guessed.  Tool results may
        have changed content, but their protocol metadata must still match.
        """
        history = tuple(raw_history)
        current = tuple(current_messages)
        prepared = tuple(prepared_messages)
        base = cls.build(
            (*history, *current),
            history_count=len(history),
            summary_state=previous_state,
            keep_recent_tool_rounds=keep_recent_tool_rounds,
            protected_source_indices=protected_source_indices,
        )
        mapped: list[int | None] = [None] * len(prepared)
        prepared_end = len(prepared)
        current_end = len(current)
        while (
            prepared_end
            and current_end
            and _compatible(prepared[prepared_end - 1], current[current_end - 1])
        ):
            prepared_end -= 1
            current_end -= 1
            mapped[prepared_end] = len(history) + current_end

        # A surviving current suffix provides an upper source bound.  Anything
        # before it can only map earlier in the original, monotonically.
        source_limit = len(history) + current_end
        source_prefix = tuple(
            entry
            for entry in base.entries
            if entry.source_index is None or entry.source_index < source_limit
        )
        ordered = _map_ordered_projection(source_prefix, prepared[:prepared_end])
        if ordered is not None:
            mapped[:prepared_end] = ordered
            return cls._with_entries(
                base.source_messages,
                base.history_count,
                tuple(
                    ContextEntry(message, source_index)
                    for message, source_index in zip(prepared, mapped, strict=True)
                ),
                keep_recent_tool_rounds=keep_recent_tool_rounds,
                protect_current=False,
                summary_state=base.summary_state,
                protected_source_indices=base.protected_source_indices,
            )

        candidates = tuple(
            entry for entry in source_prefix if entry.source_index is not None
        )
        cursor = -1
        for index, message in enumerate(prepared[:prepared_end]):
            matches = [
                entry.source_index
                for entry in candidates
                if entry.source_index is not None
                and entry.source_index > cursor
                and _compatible(message, entry.message)
            ]
            if len(matches) == 1:
                mapped[index] = matches[0]
                cursor = matches[0]

        return cls._with_entries(
            base.source_messages,
            base.history_count,
            tuple(
                ContextEntry(message, source_index)
                for message, source_index in zip(prepared, mapped, strict=True)
            ),
            keep_recent_tool_rounds=keep_recent_tool_rounds,
            protect_current=False,
            summary_state=base.summary_state,
            protected_source_indices=base.protected_source_indices,
        )

    # 函数说明：ContextInventory.with_entries
    # 用途：返回 `self._with_entries(…)`，提供 ContextInventory 的派生值。
    # 参数：
    #   entries：传给 `tuple` 的输入，类型 `Sequence[ContextEntry]`。
    # 返回：类型 `ContextInventory`；返回 `self._with_entries(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._with_entries`。
    def with_entries(self, entries: Sequence[ContextEntry]) -> ContextInventory:
        return self._with_entries(
            self.source_messages,
            self.history_count,
            tuple(entries),
            keep_recent_tool_rounds=self.keep_recent_tool_rounds,
            protect_current=self.protect_current,
            summary_state=self.summary_state,
            protected_source_indices=self.protected_source_indices,
        )

    # 函数说明：ContextInventory._with_entries
    # 用途：返回 `cls(…)`，提供 ContextInventory 的派生值。
    # 参数：
    #   source_messages：消息序列输入或配置值，类型 `tuple[Message, ...]`。
    #   history_count：原始历史消息数量，类型 `int`。
    #   entries：传给 `_make_units` 的输入，类型 `tuple[ContextEntry, ...]`。
    #   keep_recent_tool_rounds：近期项工具输入或配置值，类型 `int`。
    #   protect_current：`protect_current`输入或配置值，类型 `bool`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   protected_source_indices：必须保留的原始消息索引，类型 `tuple[int, ...]`；默认
    # `()`。
    # 返回：类型 `ContextInventory`；返回 `cls(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`cls` → `_make_units`。
    @classmethod
    def _with_entries(
        cls,
        source_messages: tuple[Message, ...],
        history_count: int,
        entries: tuple[ContextEntry, ...],
        *,
        keep_recent_tool_rounds: int,
        protect_current: bool,
        summary_state: ConversationSummaryState | None = None,
        protected_source_indices: tuple[int, ...] = (),
    ) -> ContextInventory:
        return cls(
            source_messages=source_messages,
            history_count=history_count,
            entries=entries,
            units=_make_units(
                entries,
                history_count=history_count,
                keep_recent_tool_rounds=keep_recent_tool_rounds,
                protect_current=protect_current,
                protected_source_indices=protected_source_indices,
            ),
            keep_recent_tool_rounds=keep_recent_tool_rounds,
            protect_current=protect_current,
            summary_state=summary_state,
            protected_source_indices=protected_source_indices,
        )


# 函数说明：_compatible
# 用途：在模型上下文与输入预算中处理 `_compatible`，通过 `projected.model_dump` 完成首个
# 内部处理步骤。
# 参数：
#   projected：`projected`输入或配置值，类型 `Message`。
#   source：输入来源或原始数据，类型 `Message`。
# 返回：类型 `bool`；按分支返回
# `projected.model_dump(exclude={'content'}) == source.model_dump(exclude={'content'})`
# ；`projected == source`。
# 分支与异常：
#   当 `projected.role is MessageRole.TOOL and source.role is…` 时，返回
# `projected.model_dump(exclude={'content'}) ==…`。
def _compatible(projected: Message, source: Message) -> bool:
    if projected.role is MessageRole.TOOL and source.role is MessageRole.TOOL:
        return projected.model_dump(exclude={"content"}) == source.model_dump(
            exclude={"content"}
        )
    return projected == source


# 函数说明：_valid_watermark
# 用途：在模型上下文与输入预算中处理 `_valid_watermark`，通过 `boundaries.add` 完成首个
# 内部处理步骤。
# 参数：
#   history：原始会话历史，类型 `tuple[Message, ...]`。
#   covered：`covered`输入或配置值，类型 `int`。
# 返回：类型 `bool`；按分支返回 `False`；`covered in boundaries`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`partition_messages` →
# `boundaries.add`。
# 分支与异常：
#   当 `block.block_type is BlockType.MALFORMED_TOOL and offset <…` 时，返回 `False`。
def _valid_watermark(history: tuple[Message, ...], covered: int) -> bool:
    offset = 0
    boundaries = {0}
    for block in partition_messages(history):
        if block.block_type is BlockType.MALFORMED_TOOL and offset < covered:
            return False
        offset += len(block.messages)
        boundaries.add(offset)
    return covered in boundaries


# 函数说明：_map_ordered_projection
# 用途：在模型上下文与输入预算中处理 `_map_ordered_projection`，通过 `pending.popleft`
# 完成首个内部处理步骤。
# 参数：
#   source：输入来源或原始数据，类型 `tuple[ContextEntry, ...]`。
#   prepared：传给 `len` 的输入，类型 `tuple[Message, ...]`。
# 返回：类型 `list[int | None] | None`；按分支返回
# `[entry.source_index for entry in source]`；`None`；
# `[next(iter(options)) if len(options) == 1 else None for options in possibilities]`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_compatible` → `partition_messages` →
#  `deque` → `defaultdict` → `pending.popleft` → `visited.add`；另有 4 个调用点。
# 分支与异常：
#   当 `len(source) == len(prepared) and all((_compatible(message,…` 时，返回
# `[entry.source_index for entry in source]`。
#   当 `terminal not in visited` 时，返回 `None`。
def _map_ordered_projection(
    source: tuple[ContextEntry, ...], prepared: tuple[Message, ...]
) -> list[int | None] | None:
    """Align ordered content with optional *whole* protocol-round omissions.

    Repeated ordinary text is disambiguated by the complete sequence.  If two
    indistinguishable tool rounds could explain a result, only coordinates
    shared by every valid alignment are accepted. An inserted runtime message
    can only be skipped when it cannot match any source entry; potentially
    ambiguous messages continue to use the conservative fallback.
    """
    if len(source) == len(prepared) and all(
        _compatible(message, entry.message)
        for entry, message in zip(source, prepared, strict=True)
    ):
        return [entry.source_index for entry in source]

    skip_rounds: dict[int, int] = {}
    offset = 0
    for block in partition_messages(tuple(entry.message for entry in source)):
        end = offset + len(block.messages)
        if block.block_type is BlockType.TOOL_ROUND:
            skip_rounds[offset] = end
        offset = end
    start = (0, 0)
    terminal = (len(source), len(prepared))
    pending = deque([start])
    visited = {start}
    reverse: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    matches: list[tuple[tuple[int, int], tuple[int, int], int | None]] = []
    runtime_insertions = {
        index
        for index, message in enumerate(prepared)
        if not any(_compatible(message, entry.message) for entry in source)
    }
    while pending:
        position, projected_position = pending.popleft()
        origin = (position, projected_position)
        successors: list[tuple[int, int]] = []
        if (
            position < len(source)
            and projected_position < len(prepared)
            and _compatible(prepared[projected_position], source[position].message)
        ):
            destination = (position + 1, projected_position + 1)
            successors.append(destination)
            matches.append((origin, destination, source[position].source_index))
        if position in skip_rounds:
            successors.append((skip_rounds[position], projected_position))
        if projected_position in runtime_insertions:
            destination = (position, projected_position + 1)
            successors.append(destination)
            matches.append((origin, destination, None))
        for destination in successors:
            reverse[destination].append(origin)
            if destination not in visited:
                visited.add(destination)
                pending.append(destination)
    if terminal not in visited:
        return None

    accepted = {terminal}
    pending = deque([terminal])
    while pending:
        for previous in reverse[pending.popleft()]:
            if previous not in accepted:
                accepted.add(previous)
                pending.append(previous)
    possibilities: list[set[int | None]] = [set() for _ in prepared]
    for origin, destination, source_index in matches:
        if origin in accepted and destination in accepted:
            possibilities[origin[1]].add(source_index)
    return [
        next(iter(options)) if len(options) == 1 else None for options in possibilities
    ]


# 函数说明：_make_units
# 用途：构造`units`，供模型上下文与输入预算使用。
# 参数：
#   entries：条目输入或配置值，类型 `tuple[ContextEntry, ...]`。
#   history_count：原始历史消息数量，类型 `int`。
#   keep_recent_tool_rounds：近期项工具输入或配置值，类型 `int`。
#   protect_current：`protect_current`输入或配置值，类型 `bool`。
#   protected_source_indices：必须保留的原始消息索引，类型 `tuple[int, ...]`；默认 `()`
# 。
# 返回：类型 `tuple[ContextUnit, ...]`；返回 `tuple(result)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`partition_messages` → `ContextUnit`。
def _make_units(
    entries: tuple[ContextEntry, ...],
    *,
    history_count: int,
    keep_recent_tool_rounds: int,
    protect_current: bool,
    protected_source_indices: tuple[int, ...] = (),
) -> tuple[ContextUnit, ...]:
    blocks = partition_messages(tuple(entry.message for entry in entries))
    tool_indices = [
        index
        for index, block in enumerate(blocks)
        if block.block_type is BlockType.TOOL_ROUND
    ]
    recent = (
        set(tool_indices[-keep_recent_tool_rounds:])
        if keep_recent_tool_rounds
        else set()
    )
    result: list[ContextUnit] = []
    offset = 0
    for index, block in enumerate(blocks):
        unit_entries = entries[offset : offset + len(block.messages)]
        offset += len(block.messages)
        reasons: list[str] = []
        if block.block_type is BlockType.SYSTEM:
            reasons.append("system")
        if block.block_type is BlockType.MALFORMED_TOOL:
            reasons.append("malformed_tool_protocol")
        if block.block_type is not BlockType.SYSTEM and any(
            entry.source_index is None for entry in unit_entries
        ):
            reasons.append("unknown_source")
        if index in recent:
            reasons.append("recent_tool_round")
        known_sources = tuple(
            entry.source_index
            for entry in unit_entries
            if entry.source_index is not None
        )
        has_history = any(source < history_count for source in known_sources)
        has_current = any(source >= history_count for source in known_sources)
        if block.block_type is BlockType.TOOL_ROUND and has_history and has_current:
            reasons.append("cross_history_protocol")
        if protect_current and has_current:
            reasons.append("current_messages")
        if any(source in protected_source_indices for source in known_sources):
            reasons.append("pinned_source")
        result.append(
            ContextUnit(index, unit_entries, block.block_type, tuple(reasons))
        )
    return tuple(result)


__all__ = ["ContextEntry", "ContextInventory", "ContextUnit"]
