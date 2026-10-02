from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.models.types import Message, MessageRole, ModelUsage, add_model_usage

from ..blocks import ConversationBlock, partition_messages
from ..inventory import ContextInventory
from ..planner import summary_frontier
from ..projection import apply_summary
from ..summarizer import ContextSummarizer, SummaryGenerationError
from ..summary import (
    ConversationSummaryState,
)

TokenCounter = Callable[[tuple[Message, ...]], int]


@dataclass(frozen=True)
class ConversationReductionResult:
    messages: tuple[Message, ...]
    estimated_input_tokens: int
    summary_state: ConversationSummaryState | None = None
    summarized_conversation_blocks: int = 0
    summary_usage: ModelUsage = field(default_factory=ModelUsage)
    summary_provider: str | None = None
    summary_model: str | None = None
    summary_duration_ms: float | None = None
    reached_target: bool = False
    error: str | None = None


class ConversationReducer:
    # 函数说明：ConversationReducer.__init__
    # 用途：初始化 ConversationReducer；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   summarizer：传给 `getattr` 的输入，类型 `ContextSummarizer`。
    #   keep_recent_conversation_blocks：近期项会话输入或配置值，类型 `int`；默认 `4`。
    #   keep_recent_tool_rounds：近期项工具输入或配置值，类型 `int`；默认 `2`。
    #   large_fold_span_tokens：Token 数量或 Token 预算，类型 `int`；默认 `50000`。
    #   large_fold_max_output_tokens：Token 数量或 Token 预算，类型 `int | None`；默认
    # `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_optional_text`。
    # 分支与异常：
    #   当 `keep_recent_conversation_blocks < 0` 时，抛出 `ValueError(…)`。
    #   当 `keep_recent_tool_rounds < 0` 时，抛出
    # `ValueError('keep_recent_tool_rounds cannot be negative')`。
    #   当 `large_fold_span_tokens < 0` 时，抛出
    # `ValueError('large_fold_span_tokens cannot be negative')`。
    #   当 `large_fold_max_output_tokens is not None and…` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   更新对象字段：`self._summarizer`、`self.summary_provider`、`self.summary_model`
    # 、`self.keep_recent_conversation_blocks`、`self.keep_recent_tool_rounds`、
    # `self._large_fold_span_tokens`、`self._large_fold_max_output_tokens`。
    def __init__(
        self,
        summarizer: ContextSummarizer,
        *,
        keep_recent_conversation_blocks: int = 4,
        keep_recent_tool_rounds: int = 2,
        large_fold_span_tokens: int = 50_000,
        large_fold_max_output_tokens: int | None = None,
    ) -> None:
        if keep_recent_conversation_blocks < 0:
            raise ValueError("keep_recent_conversation_blocks cannot be negative")
        if keep_recent_tool_rounds < 0:
            raise ValueError("keep_recent_tool_rounds cannot be negative")
        if large_fold_span_tokens < 0:
            raise ValueError("large_fold_span_tokens cannot be negative")
        if large_fold_max_output_tokens is not None and (
            large_fold_max_output_tokens <= 0
        ):
            raise ValueError("large_fold_max_output_tokens must be greater than zero")
        self._summarizer = summarizer
        self.summary_provider = _optional_text(
            getattr(summarizer, "provider_hint", None)
        )
        self.summary_model = _optional_text(getattr(summarizer, "model_hint", None))
        self.keep_recent_conversation_blocks = keep_recent_conversation_blocks
        self.keep_recent_tool_rounds = keep_recent_tool_rounds
        self._large_fold_span_tokens = large_fold_span_tokens
        self._large_fold_max_output_tokens = large_fold_max_output_tokens

    # 函数说明：ConversationReducer.reduce
    # 用途：兼容消息列表调用，将来源对齐交给上下文清单。
    # 参数：
    #   raw_history：传给 `ContextInventory.from_projection` 的输入，类型
    # `Sequence[Message]`。
    #   prepared_messages：传给 `ContextInventory.from_projection` 的输入，类型
    # `Sequence[Message]`。
    #   current_messages：传给 `ContextInventory.from_projection` 的输入，类型
    # `Sequence[Message]`。
    #   previous_state：传给 `ContextInventory.from_projection` 的输入，类型
    # `ConversationSummaryState | None`。
    #   initial_estimated_input_tokens：Token 数量或 Token 预算，类型 `int`。
    #   target_tokens：Token 数量或 Token 预算，类型 `int`。
    #   estimate：`estimate`输入或配置值，类型 `TokenCounter`。
    # 返回：类型 `ConversationReductionResult`；返回 `result`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextInventory.from_projection`
    #  → `self.reduce_inventory`。
    async def reduce(
        self,
        *,
        raw_history: Sequence[Message],
        prepared_messages: Sequence[Message],
        current_messages: Sequence[Message],
        previous_state: ConversationSummaryState | None,
        initial_estimated_input_tokens: int,
        target_tokens: int,
        estimate: TokenCounter,
    ) -> ConversationReductionResult:
        """兼容消息列表调用，将来源对齐交给上下文清单。"""
        inventory = ContextInventory.from_projection(
            raw_history,
            current_messages,
            prepared_messages,
            previous_state,
            keep_recent_tool_rounds=self.keep_recent_tool_rounds,
        )
        result, _ = await self.reduce_inventory(
            inventory,
            previous_state=previous_state,
            initial_estimated_input_tokens=initial_estimated_input_tokens,
            target_tokens=target_tokens,
            estimate=estimate,
        )
        return result

    # 函数说明：ConversationReducer.reduce_inventory
    # 用途：摘要原历史的合法连续前缀，再按来源坐标投影请求。
    # 参数：
    #   inventory：传给 `summary_frontier` 的输入，类型 `ContextInventory`。
    #   previous_state：传给 `self._unchanged` 的输入，类型
    # `ConversationSummaryState | None`。
    #   initial_estimated_input_tokens：Token 数量或 Token 预算，类型 `int`。
    #   target_tokens：Token 数量或 Token 预算，类型 `int`。
    #   estimate：`estimate`输入或配置值，类型 `TokenCounter`。
    # 返回：类型 `tuple[ConversationReductionResult, ContextInventory]`；按分支返回 `(
    # self._unchanged(prepared, previous_state, initial_estimated_input_tokens,
    # target_tokens,…`；`(ConversationReductionResult(messages=reduced_messages,
    # estimated_input_tokens=estimated,…`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._unchanged` →
    # `ContextInventory.build` → `summary_frontier` → `ModelUsage` → `time.perf_counter`
    #  → `self._summarizer.summarize`；另有 8 个调用点。
    # 分支与异常：
    #   当 `covered > len(raw)` 时，返回 `(self._unchanged(prepared, previous_state,…`。
    #   当 `validated.summary_state is None` 时，返回
    # `(self._unchanged(prepared, previous_state,…`。
    #   当 `window.cutoff <= covered or not window.conversation_blocks` 时，返回
    # `(self._unchanged(prepared, previous_state,…`。
    #   捕获 `Exception` 后，跳过当前循环项，继续处理后续项。
    async def reduce_inventory(
        self,
        inventory: ContextInventory,
        *,
        previous_state: ConversationSummaryState | None,
        initial_estimated_input_tokens: int,
        target_tokens: int,
        estimate: TokenCounter,
    ) -> tuple[ConversationReductionResult, ContextInventory]:
        """摘要原历史的合法连续前缀，再按来源坐标投影请求。"""
        raw = inventory.source_messages[: inventory.history_count]
        prepared = inventory.messages
        covered = previous_state.covered_message_count if previous_state else 0
        if covered > len(raw):
            return self._unchanged(
                prepared,
                previous_state,
                initial_estimated_input_tokens,
                target_tokens,
                "summary covered_message_count exceeds current history length",
            ), inventory
        if covered:
            # A caller may provide an unprojected inventory or a saved state
            # from another history. Validate the raw prefix independently of
            # whether the summary happens to appear in this request projection.
            validated = ContextInventory.build(
                raw,
                history_count=len(raw),
                summary_state=previous_state,
                keep_recent_tool_rounds=self.keep_recent_tool_rounds,
                protected_source_indices=inventory.protected_source_indices,
            )
            if validated.summary_state is None:
                return self._unchanged(
                    prepared,
                    previous_state,
                    initial_estimated_input_tokens,
                    target_tokens,
                    "summary covered_message_count is not a valid raw-history boundary",
                ), inventory

        # The frontier reads raw source coordinates, not projected units.
        # Removing a tool round therefore never advances persisted coverage.
        window = summary_frontier(
            inventory,
            covered_message_count=covered,
            keep_recent_conversation_blocks=self.keep_recent_conversation_blocks,
            keep_recent_tool_rounds=self.keep_recent_tool_rounds,
        )
        if window.cutoff <= covered or not window.conversation_blocks:
            return self._unchanged(
                prepared,
                previous_state,
                initial_estimated_input_tokens,
                target_tokens,
                "no complete foldable prefix remains",
            ), inventory

        visible_by_source = {
            entry.source_index: entry.message
            for entry in inventory.entries
            if entry.source_index is not None
        }
        source_messages = tuple(
            visible_by_source.get(index, message)
            for index, message in enumerate(raw[covered : window.cutoff], start=covered)
            if message.role is not MessageRole.SYSTEM
        )
        previous_summary = previous_state.summary if previous_state else None
        fold_span_tokens = max(0, initial_estimated_input_tokens - target_tokens)
        summary_output_limit = (
            self._large_fold_max_output_tokens
            if (
                self._large_fold_max_output_tokens is not None
                and fold_span_tokens >= self._large_fold_span_tokens
            )
            else None
        )
        total_usage = ModelUsage()
        last_error = "summary generation failed"
        started = time.perf_counter()
        for attempt in range(2):
            try:
                generated = (
                    await self._summarizer.summarize(
                        previous_summary,
                        source_messages,
                        max_output_tokens=summary_output_limit,
                    )
                    if attempt == 0
                    else await self._summarizer.retry_compact(
                        previous_summary,
                        source_messages,
                        reason=last_error,
                        max_output_tokens=summary_output_limit,
                    )
                )
            except Exception as exc:
                total_usage = _add_usage(total_usage, _error_usage(exc))
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            total_usage = _add_usage(total_usage, generated.usage)
            if not any(
                (
                    generated.summary.current_objective,
                    generated.summary.user_constraints,
                    generated.summary.key_decisions,
                    generated.summary.completed_work,
                    generated.summary.current_state,
                    generated.summary.pending_work,
                    generated.summary.important_facts,
                )
            ):
                last_error = "summary model returned an empty summary"
                continue
            new_state = ConversationSummaryState(
                summary=generated.summary,
                covered_message_count=window.cutoff,
            )
            reduced_inventory = apply_summary(inventory, new_state)
            reduced_messages = reduced_inventory.messages
            estimated = estimate(reduced_messages)
            if estimated < initial_estimated_input_tokens:
                return ConversationReductionResult(
                    messages=reduced_messages,
                    estimated_input_tokens=estimated,
                    summary_state=new_state,
                    summarized_conversation_blocks=sum(
                        isinstance(block, ConversationBlock)
                        for block in partition_messages(source_messages)
                    ),
                    summary_usage=total_usage,
                    summary_provider=self.summary_provider,
                    summary_model=self.summary_model,
                    summary_duration_ms=(time.perf_counter() - started) * 1000,
                    reached_target=estimated <= target_tokens,
                ), reduced_inventory
            last_error = "generated summary did not reduce the request context"

        return self._unchanged(
            prepared,
            previous_state,
            initial_estimated_input_tokens,
            target_tokens,
            last_error,
            summary_usage=total_usage,
            summary_provider=self.summary_provider,
            summary_model=self.summary_model,
            summary_duration_ms=(time.perf_counter() - started) * 1000,
        ), inventory

    # 函数说明：ConversationReducer._unchanged
    # 用途：返回 `ConversationReductionResult(…)`，提供 ConversationReducer 的派生值。
    # 参数：
    #   messages：本次处理的消息序列，类型 `tuple[Message, ...]`。
    #   state：当前状态快照，类型 `ConversationSummaryState | None`。
    #   estimated：`estimated`输入或配置值，类型 `int`。
    #   target_tokens：Token 数量或 Token 预算，类型 `int`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    #   summary_usage：摘要用量输入或配置值，类型 `ModelUsage | None`；默认 `None`。
    #   summary_provider：摘要服务商输入或配置值，类型 `str | None`；默认 `None`。
    #   summary_model：摘要模型输入或配置值，类型 `str | None`；默认 `None`。
    #   summary_duration_ms：摘要输入或配置值，类型 `float | None`；默认 `None`。
    # 返回：类型 `ConversationReductionResult`；返回 `ConversationReductionResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationReductionResult` →
    # `ModelUsage`。
    @staticmethod
    def _unchanged(
        messages: tuple[Message, ...],
        state: ConversationSummaryState | None,
        estimated: int,
        target_tokens: int,
        error: str | None = None,
        *,
        summary_usage: ModelUsage | None = None,
        summary_provider: str | None = None,
        summary_model: str | None = None,
        summary_duration_ms: float | None = None,
    ) -> ConversationReductionResult:
        return ConversationReductionResult(
            messages=messages,
            estimated_input_tokens=estimated,
            summary_state=state,
            summary_usage=summary_usage or ModelUsage(),
            summary_provider=summary_provider,
            summary_model=summary_model,
            summary_duration_ms=summary_duration_ms,
            reached_target=estimated <= target_tokens,
            error=error,
        )


# 函数说明：_error_usage
# 用途：处理模型上下文与输入预算中的 `_error_usage` 数据；结果及边界条件见下方说明。
# 参数：
#   error：异常或错误信息，类型 `Exception`。
# 返回：类型 `ModelUsage`；按分支返回 `error.usage`；`ModelUsage()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelUsage`。
# 分支与异常：
#   当 `isinstance(error, SummaryGenerationError)` 时，返回 `error.usage`。
def _error_usage(error: Exception) -> ModelUsage:
    if isinstance(error, SummaryGenerationError):
        return error.usage
    return ModelUsage()


# 函数说明：_optional_text
# 用途：在模型上下文与输入预算中处理 `_optional_text`，通过 `str(value).strip` 完成首个
# 内部处理步骤。
# 参数：
#   value：待校验、规范化或转换的值，类型 `object`。
# 返回：类型 `str | None`；按分支返回 `None`；`normalized or None`。
# 分支与异常：
#   当 `value is None` 时，返回 `None`。
def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


# 函数说明：_add_usage
# 用途：添加用量，供模型上下文与输入预算使用。
# 参数：
#   left：传给 `add_model_usage` 的输入，类型 `ModelUsage`。
#   right：传给 `add_model_usage` 的输入，类型 `ModelUsage`。
# 返回：类型 `ModelUsage`；返回 `add_model_usage(left, right)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`add_model_usage`。
def _add_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
    return add_model_usage(left, right)


# 函数说明：build_summary_candidate
# 用途：构建摘要候选，供模型上下文与输入预算使用。
# 参数：
#   raw_history：传给 `tuple` 的输入，类型 `Sequence[Message]`。
#   current_messages：传给 `tuple` 的输入，类型 `Sequence[Message]`。
#   state：当前状态快照，类型 `ConversationSummaryState | None`。
# 返回：类型 `tuple[tuple[Message, ...], int]`；返回
# `(messages, len(messages) - len(current))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextInventory.build`。
def build_summary_candidate(
    raw_history: Sequence[Message],
    current_messages: Sequence[Message],
    state: ConversationSummaryState | None,
) -> tuple[tuple[Message, ...], int]:

    history = tuple(raw_history)
    current = tuple(current_messages)
    inventory = ContextInventory.build(
        (*history, *current),
        history_count=len(history),
        summary_state=state,
    )
    messages = inventory.messages
    return messages, len(messages) - len(current)


__all__ = [
    "ConversationReducer",
    "ConversationReductionResult",
    "build_summary_candidate",
]
