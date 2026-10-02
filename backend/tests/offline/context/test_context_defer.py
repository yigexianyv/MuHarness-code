from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.models.types import Message, MessageRole, ModelUsage
from app.runtime.context import (
    ContextBudgetPolicy,
    ContextManager,
    ContextSettings,
    ContextSummarizer,
    ConversationReducer,
    ModelCapabilities,
    RollingConversationSummary,
    SummaryGenerationResult,
)
from app.runtime.context.capabilities import CapabilitySource


# 函数说明：_caps
# 用途：返回 `ModelCapabilities(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   context_window：上下文输入或配置值，类型 `int`；默认 `131072`。
#   max_output：输出输入或配置值，类型 `int`；默认 `8192`。
# 返回：类型 `ModelCapabilities`；返回 `ModelCapabilities(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelCapabilities`。
def _caps(
    context_window: int = 131_072,
    max_output: int = 8_192,
) -> ModelCapabilities:
    return ModelCapabilities(
        provider="qwen",
        model="qwen3.7-plus",
        context_window=context_window,
        max_output_tokens=max_output,
        source=CapabilitySource.BUILTIN,
    )


# 函数说明：test_budget_compact_lines_default_to_twice_soft_line
# 用途：回归验证回归测试与测试辅助中的 `budget_compact_lines_default_to_twice_soft_line`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudgetPolicy` →
# `policy.compute` → `_caps`。
# 分支与异常：
#   验证条件：`budget.trigger_tokens == 51200`。
#   验证条件：`budget.target_tokens == 28800`。
#   验证条件：`budget.compact_ceiling_tokens == 128000`。
#   验证条件：`budget.forced_target_tokens == 64000`。
def test_budget_compact_lines_default_to_twice_soft_line() -> None:

    policy = ContextBudgetPolicy(safety_margin_tokens=4_096)
    budget = policy.compute(
        _caps(context_window=1_048_576, max_output=4_096),
        max_output_tokens=4_096,
    )

    assert budget.trigger_tokens == 51_200
    assert budget.target_tokens == 28_800
    assert budget.compact_ceiling_tokens == 128_000
    assert budget.forced_target_tokens == 64_000


# 函数说明：test_budget_compact_ceiling_never_exceeds_hard_trigger
# 用途：回归验证回归测试与测试辅助中的
# `budget_compact_ceiling_never_exceeds_hard_trigger` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudgetPolicy` →
# `policy.compute` → `_caps`。
# 分支与异常：
#   验证条件：`budget.compact_ceiling_tokens == budget.hard_trigger_tokens`。
#   验证条件：`budget.forced_target_tokens == budget.target_tokens`。
def test_budget_compact_ceiling_never_exceeds_hard_trigger() -> None:

    policy = ContextBudgetPolicy(safety_margin_tokens=100)
    budget = policy.compute(
        _caps(context_window=2_140, max_output=512),
        max_output_tokens=512,
    )

    assert budget.compact_ceiling_tokens == budget.hard_trigger_tokens
    assert budget.forced_target_tokens == budget.target_tokens


# 函数说明：test_budget_compact_input_tokens_override
# 用途：回归验证回归测试与测试辅助中的 `budget_compact_input_tokens_override` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudgetPolicy` →
# `policy.compute` → `_caps`。
# 分支与异常：
#   验证条件：`budget.compact_ceiling_tokens == 96000`。
#   验证条件：`budget.forced_target_tokens == 48000`。
def test_budget_compact_input_tokens_override() -> None:
    policy = ContextBudgetPolicy(
        safety_margin_tokens=100,
        compact_input_tokens=96_000,
    )
    budget = policy.compute(
        _caps(context_window=1_048_576, max_output=4_096),
        max_output_tokens=4_096,
    )

    assert budget.compact_ceiling_tokens == 96_000
    assert budget.forced_target_tokens == 48_000


# 函数说明：test_budget_rejects_non_positive_compact_input_tokens
# 用途：回归验证回归测试与测试辅助中的
# `budget_rejects_non_positive_compact_input_tokens` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `ContextBudgetPolicy`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='compact_input_tokens')`。
def test_budget_rejects_non_positive_compact_input_tokens() -> None:
    with pytest.raises(ValueError, match="compact_input_tokens"):
        ContextBudgetPolicy(compact_input_tokens=0)


# 函数说明：test_settings_rejects_lower_large_fold_summary_cap
# 用途：回归验证回归测试与测试辅助中的 `settings_rejects_lower_large_fold_summary_cap`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `ContextSettings`。
# 分支与异常：
#   预期异常：
# `pytest.raises(ValueError, match='context_summary_max_output_tokens_large_fold')`。
def test_settings_rejects_lower_large_fold_summary_cap() -> None:
    with pytest.raises(
        ValueError,
        match="context_summary_max_output_tokens_large_fold",
    ):
        ContextSettings(
            _env_file=None,
            context_summary_max_output_tokens_large_fold=512,
        )


class _TargetCapturingReducer(ConversationReducer):
    # 函数说明：_TargetCapturingReducer.__init__
    # 用途：初始化 _TargetCapturingReducer；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super` →
    # `_FixedSummarizer`。
    # 副作用与资源：
    #   更新对象字段：`self.targets`。
    def __init__(self) -> None:
        super().__init__(_FixedSummarizer())
        self.targets: list[int] = []

    # 函数说明：_TargetCapturingReducer.reduce_inventory
    # 用途：在回归测试与测试辅助中处理 `reduce_inventory`，通过 `self.targets.append` 完
    # 成首个内部处理步骤。
    # 参数：
    #   inventory：传给 `super().reduce_inventory` 的输入。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：返回 `await super().reduce_inventory(inventory, **kwargs)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().reduce_inventory` →
    # `super`。
    async def reduce_inventory(self, inventory, **kwargs):
        self.targets.append(kwargs["target_tokens"])
        return await super().reduce_inventory(inventory, **kwargs)


class _FixedSummarizer(ContextSummarizer):
    # 函数说明：_FixedSummarizer.summarize
    # 用途：生成摘要_FixedSummarizer，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要，类型 `RollingConversationSummary | None`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `SummaryGenerationResult`；返回 `SummaryGenerationResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SummaryGenerationResult` →
    # `RollingConversationSummary` → `ModelUsage`。
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        return SummaryGenerationResult(
            summary=RollingConversationSummary(current_objective="目标"),
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


# 函数说明：_big_history
# 用途：在回归测试与测试辅助中处理 `_big_history`，通过 `messages.append` 完成首个内部处
# 理步骤。
# 参数：
#   blocks：传给 `range` 的输入，类型 `int`；默认 `6`。
# 返回：类型 `tuple[Message, ...]`；返回 `tuple(messages)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
def _big_history(blocks: int = 6) -> tuple[Message, ...]:
    messages: list[Message] = []
    for index in range(blocks):
        messages.append(
            Message(role=MessageRole.USER, content=f"问题 {index} " + "细" * 200)
        )
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=f"回答 {index} " + "节" * 200,
            )
        )
    return tuple(messages)


# 函数说明：test_prepare_forwards_compaction_target_override
# 用途：回归验证回归测试与测试辅助中的 `prepare_forwards_compaction_target_override` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_TargetCapturingReducer` →
# `ContextManager` → `ContextBudgetPolicy` → `_big_history` → `Message` →
# `manager.prepare`。
# 分支与异常：
#   验证条件：`reducer.targets == [64000]`。
@pytest.mark.asyncio
async def test_prepare_forwards_compaction_target_override() -> None:
    reducer = _TargetCapturingReducer()
    manager = ContextManager(
        budget_policy=ContextBudgetPolicy(
            safety_margin_tokens=100,
            preferred_input_tokens=1_000,
        ),
        conversation_reducer=reducer,
    )
    history = _big_history()
    current = (Message(role=MessageRole.USER, content="当前问题"),)

    await manager.prepare(
        (*history, *current),
        model="qwen3.7-plus",
        provider="qwen",
        history_count=len(history),
        compaction_target_tokens=64_000,
    )

    assert reducer.targets == [64_000]


# 函数说明：test_prepare_uses_forced_target_without_override
# 用途：回归验证回归测试与测试辅助中的 `prepare_uses_forced_target_without_override` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_TargetCapturingReducer` →
# `ContextManager` → `ContextBudgetPolicy` → `_big_history` → `Message` →
# `manager.prepare`。
# 分支与异常：
#   验证条件：`reducer.targets == [decision.target_tokens]`。
@pytest.mark.asyncio
async def test_prepare_uses_forced_target_without_override() -> None:
    reducer = _TargetCapturingReducer()
    manager = ContextManager(
        budget_policy=ContextBudgetPolicy(
            safety_margin_tokens=100,
            preferred_input_tokens=1_000,
            compact_input_tokens=400,
        ),
        conversation_reducer=reducer,
    )
    history = _big_history()
    current = (Message(role=MessageRole.USER, content="当前问题"),)

    decision = await manager.prepare(
        (*history, *current),
        model="qwen3.7-plus",
        provider="qwen",
        history_count=len(history),
    )

    assert reducer.targets == [decision.target_tokens]


# 函数说明：test_prepare_rejects_non_positive_compaction_target
# 用途：回归验证回归测试与测试辅助中的 `prepare_rejects_non_positive_compaction_target`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextManager` → `pytest.raises` →
# `manager.prepare` → `Message`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='compaction_target_tokens must be greater
#  than zero')`。
@pytest.mark.asyncio
async def test_prepare_rejects_non_positive_compaction_target() -> None:
    manager = ContextManager()
    with pytest.raises(
        ValueError,
        match="compaction_target_tokens must be greater than zero",
    ):
        await manager.prepare(
            (Message(role=MessageRole.USER, content="hi"),),
            compaction_target_tokens=0,
        )


# 函数说明：test_conversation_count_is_diagnostic_not_a_second_gate
# 用途：回归验证回归测试与测试辅助中的
# `conversation_count_is_diagnostic_not_a_second_gate` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextManager` → `ContextSettings` →
#  `_big_history` → `manager.prepare`。
# 分支与异常：
#   验证条件：`decision.unsummarized_conversation_blocks == 2`。
#   验证条件：`decision.requires_compaction is False`。
#   验证条件：`decision.messages == history`。
@pytest.mark.asyncio
async def test_conversation_count_is_diagnostic_not_a_second_gate() -> None:
    manager = ContextManager(
        context_settings=ContextSettings(
            _env_file=None,
        ),
    )
    history = _big_history(blocks=2)
    decision = await manager.prepare(history)
    assert decision.unsummarized_conversation_blocks == 2
    assert decision.requires_compaction is False
    assert decision.messages == history


class _OutputLimitCapturingSummarizer(ContextSummarizer):
    # 函数说明：_OutputLimitCapturingSummarizer.__init__
    # 用途：初始化 _OutputLimitCapturingSummarizer；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.limits`。
    def __init__(self) -> None:
        self.limits: list[int | None] = []

    # 函数说明：_OutputLimitCapturingSummarizer.summarize
    # 用途：生成摘要_OutputLimitCapturingSummarizer，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要，类型 `RollingConversationSummary | None`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `SummaryGenerationResult`；返回 `SummaryGenerationResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SummaryGenerationResult` →
    # `RollingConversationSummary` → `ModelUsage`。
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        self.limits.append(max_output_tokens)
        return SummaryGenerationResult(
            summary=RollingConversationSummary(current_objective="目标"),
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


# 函数说明：_flat_estimate
# 用途：估算`flat`，供回归测试与测试辅助使用。
# 参数：
#   messages：本次处理的消息序列，类型 `Sequence[Message]`。
# 返回：类型 `int`；返回 `sum((len(message.content or '') + 4 for message in messages))`
# 。
def _flat_estimate(messages: Sequence[Message]) -> int:
    return sum(len(message.content or "") + 4 for message in messages)


# 函数说明：test_reducer_raises_summary_limit_for_large_fold
# 用途：回归验证回归测试与测试辅助中的 `reducer_raises_summary_limit_for_large_fold` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_OutputLimitCapturingSummarizer` →
# `ConversationReducer` → `_big_history` → `reducer.reduce`。
# 分支与异常：
#   验证条件：`result.error is None`。
#   验证条件：`summarizer.limits == [2048]`。
@pytest.mark.asyncio
async def test_reducer_raises_summary_limit_for_large_fold() -> None:
    summarizer = _OutputLimitCapturingSummarizer()
    reducer = ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=1,
        keep_recent_tool_rounds=0,
        large_fold_span_tokens=1_000,
        large_fold_max_output_tokens=2_048,
    )
    history = _big_history(blocks=4)

    result = await reducer.reduce(
        raw_history=history,
        prepared_messages=history,
        current_messages=(),
        previous_state=None,
        initial_estimated_input_tokens=5_000,
        target_tokens=1_000,
        estimate=_flat_estimate,
    )

    assert result.error is None
    assert summarizer.limits == [2_048]


# 函数说明：test_reducer_keeps_default_limit_for_small_fold
# 用途：回归验证回归测试与测试辅助中的 `reducer_keeps_default_limit_for_small_fold` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_OutputLimitCapturingSummarizer` →
# `ConversationReducer` → `_big_history` → `reducer.reduce`。
# 分支与异常：
#   验证条件：`summarizer.limits == [None]`。
@pytest.mark.asyncio
async def test_reducer_keeps_default_limit_for_small_fold() -> None:
    summarizer = _OutputLimitCapturingSummarizer()
    reducer = ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=1,
        keep_recent_tool_rounds=0,
        large_fold_span_tokens=1_000,
        large_fold_max_output_tokens=2_048,
    )
    history = _big_history(blocks=4)

    await reducer.reduce(
        raw_history=history,
        prepared_messages=history,
        current_messages=(),
        previous_state=None,
        initial_estimated_input_tokens=1_400,
        target_tokens=1_000,
        estimate=_flat_estimate,
    )

    assert summarizer.limits == [None]


# 函数说明：test_reducer_without_large_fold_config_never_overrides
# 用途：回归验证回归测试与测试辅助中的
# `reducer_without_large_fold_config_never_overrides` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_OutputLimitCapturingSummarizer` →
# `ConversationReducer` → `_big_history` → `reducer.reduce`。
# 分支与异常：
#   验证条件：`summarizer.limits == [None]`。
@pytest.mark.asyncio
async def test_reducer_without_large_fold_config_never_overrides() -> None:
    summarizer = _OutputLimitCapturingSummarizer()
    reducer = ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=1,
        keep_recent_tool_rounds=0,
    )
    history = _big_history(blocks=4)

    await reducer.reduce(
        raw_history=history,
        prepared_messages=history,
        current_messages=(),
        previous_state=None,
        initial_estimated_input_tokens=100_000,
        target_tokens=1_000,
        estimate=_flat_estimate,
    )

    assert summarizer.limits == [None]
