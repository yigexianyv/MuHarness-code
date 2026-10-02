from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

from app.models.types import Message, ModelUsage, ToolDefinition

from .blocks import ConversationBlock, ToolRoundBlock, partition_messages
from .budget import ContextBudgetPolicy, build_budget_policy
from .capabilities import ModelCapabilityRegistry, build_model_capability_registry
from .config import ContextSettings
from .inventory import ContextInventory
from .projection import apply_summary
from .reducers import ConversationReducer
from .summary import SUMMARY_MESSAGE_NAMES, ConversationSummaryState
from .tokens import TokenEstimator, default_token_estimator


class ContextCompactionStage(StrEnum):
    NONE = "none"
    # Values retained solely to read historical traces, not execution branches.
    TOOL_RESULTS = "tool_results"
    TOOL_ROUNDS = "tool_rounds"
    TOOL_RESULTS_AND_ROUNDS = "tool_results_and_rounds"
    ROLLING_SUMMARY = "rolling_summary"
    TOOL_AND_ROLLING_SUMMARY = "tool_and_rolling_summary"


@dataclass(frozen=True)
class ContextDecision:
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    provider: str | None = None
    model: str | None = None
    original_estimated_input_tokens: int | None = None
    prepared_input_tokens: int | None = None
    estimated_input_tokens: int | None = None
    context_window: int | None = None
    reserved_output_tokens: int | None = None
    safety_margin_tokens: int | None = None
    input_budget: int | None = None
    working_input_budget: int | None = None
    hard_trigger_tokens: int | None = None
    hard_target_tokens: int | None = None
    trigger_tokens: int | None = None
    target_tokens: int | None = None
    tool_result_budget_tokens: int | None = None
    compact_ceiling_tokens: int | None = None
    forced_target_tokens: int | None = None
    tool_result_tokens_before: int = 0
    tool_result_tokens_after: int = 0
    tool_schema_tokens: int = 0
    message_tokens_before: int = 0
    message_tokens_after: int = 0
    unsummarized_conversation_blocks: int = 0
    conversation_block_limit: int | None = None
    conversation_block_triggered: bool = False
    original_usage_ratio: float | None = None
    prepared_usage_ratio: float | None = None
    usage_ratio: float | None = None
    requires_compaction: bool = False
    exceeds_input_budget: bool = False
    capability_source: str | None = None
    trimmed: bool = False
    compaction_stage: ContextCompactionStage = ContextCompactionStage.NONE
    reached_target: bool = True
    needs_next_compaction_stage: bool = False
    compacted_tool_results: int = 0
    removed_tool_rounds: int = 0
    summary_state: ConversationSummaryState | None = None
    summary_updated: bool = False
    summarized_conversation_blocks: int = 0
    summary_usage: ModelUsage = field(default_factory=ModelUsage)
    summary_provider: str | None = None
    summary_model: str | None = None
    summary_duration_ms: float | None = None
    summary_error: str | None = None
    reason: str | None = None


class ContextManager:
    """Append unchanged requests until one pressure gate starts a new epoch."""

    # 函数说明：ContextManager.__init__
    # 用途：初始化 ContextManager；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   estimator：`estimator`输入或配置值，类型 `TokenEstimator | None`；默认 `None`。
    #   registry：工具、模型或能力注册表，类型 `ModelCapabilityRegistry | None`；默认
    # `None`。
    #   budget_policy：预算输入或配置值，类型 `ContextBudgetPolicy | None`；默认 `None`
    # 。
    #   context_settings：上下文设置输入或配置值，类型 `ContextSettings | None`；默认
    # `None`。
    #   keep_recent_tool_rounds：近期项工具输入或配置值，类型 `int | None`；默认 `None`
    # 。
    #   conversation_reducer：会话输入或配置值，类型 `ConversationReducer | None`；默认
    # `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextSettings` →
    # `default_token_estimator` → `build_model_capability_registry` →
    # `build_budget_policy`。
    # 分支与异常：
    #   当 `self._keep_recent_tool_rounds < 0` 时，抛出
    # `ValueError('keep_recent_tool_rounds cannot be negative')`。
    # 副作用与资源：
    #   更新对象字段：`self._keep_recent_tool_rounds`、`self._estimator`、
    # `self._registry`、`self._budget_policy`、`self._conversation_reducer`、
    # `self._tool_view_limits`。
    def __init__(
        self,
        estimator: TokenEstimator | None = None,
        *,
        registry: ModelCapabilityRegistry | None = None,
        budget_policy: ContextBudgetPolicy | None = None,
        context_settings: ContextSettings | None = None,
        keep_recent_tool_rounds: int | None = None,
        conversation_reducer: ConversationReducer | None = None,
    ) -> None:
        settings = context_settings or ContextSettings()
        self._keep_recent_tool_rounds = (
            settings.context_keep_recent_tool_rounds
            if keep_recent_tool_rounds is None
            else keep_recent_tool_rounds
        )
        if self._keep_recent_tool_rounds < 0:
            raise ValueError("keep_recent_tool_rounds cannot be negative")
        self._estimator = estimator or default_token_estimator()
        self._registry = registry or build_model_capability_registry(
            context_settings=settings
        )
        self._budget_policy = budget_policy or build_budget_policy(settings)
        self._conversation_reducer = conversation_reducer
        self._tool_view_limits = {
            "max_output_chars": settings.context_max_tool_result_chars,
            "head_chars": settings.context_tool_result_head_chars,
            "tail_chars": settings.context_tool_result_tail_chars,
        }

    # 函数说明：ContextManager.estimator
    # 用途：返回 `self._estimator`，提供 ContextManager 的派生值。
    # 返回：类型 `TokenEstimator`；返回 `self._estimator`。
    @property
    def estimator(self) -> TokenEstimator:
        return self._estimator

    # 函数说明：ContextManager.registry
    # 用途：返回 `self._registry`，提供 ContextManager 的派生值。
    # 返回：类型 `ModelCapabilityRegistry`；返回 `self._registry`。
    @property
    def registry(self) -> ModelCapabilityRegistry:
        return self._registry

    # 函数说明：ContextManager.keep_recent_tool_rounds
    # 用途：返回 `self._keep_recent_tool_rounds`，提供 ContextManager 的派生值。
    # 返回：类型 `int`；返回 `self._keep_recent_tool_rounds`。
    @property
    def keep_recent_tool_rounds(self) -> int:
        return self._keep_recent_tool_rounds

    # 函数说明：ContextManager.tool_view_limits
    # 用途：返回 `dict(self._tool_view_limits)`，提供 ContextManager 的派生值。
    # 返回：类型 `dict[str, int]`；返回 `dict(self._tool_view_limits)`。
    @property
    def tool_view_limits(self) -> dict[str, int]:
        return dict(self._tool_view_limits)

    # 函数说明：ContextManager.prepare
    # 用途：构造模型输入并测量预算，仅在集中阈值触发时尝试摘要；保留原始来源和摘要覆盖水
    # 位，返回完整上下文决策。
    # 参数：
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   source_messages：传给 `tuple` 的输入，类型 `Sequence[Message] | None`；默认
    # `None`。
    #   protected_source_indices：必须保留的原始消息索引，类型 `Sequence[int]`；默认
    # `()`。
    #   tools：可用工具定义或工具实例集合，类型 `Sequence[ToolDefinition]`；默认 `()`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    #   history_count：原始历史消息数量，类型 `int | None`；默认 `None`。
    #   keep_recent_tool_rounds：近期项工具输入或配置值，类型 `int | None`；默认 `None`
    # 。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   compaction_target_tokens：Token 数量或 Token 预算，类型 `int | None`；默认
    # `None`。
    #   allow_compaction：`allow_compaction`输入或配置值，类型 `bool`；默认 `True`。
    # 返回：类型 `ContextDecision`；返回 `ContextDecision(…)`。
    # 设计约束：摘要必须与原始消息覆盖水位匹配；预算决策不能把任务状态提升为系统授权。
    # 设计约束：摘要失败或无法达到目标仍返回测量结果及错误信息，交由上层决定是否继续或终
    # 止。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextInventory.build` →
    # `valid_state.summary.to_message` → `ContextInventory.from_projection` →
    # `apply_summary` → `self.registry.lookup` → `self._budget_policy.compute`；另有 6
    # 个调用点。
    # 分支与异常：
    #   当 `not 0 <= count <= len(source)` 时，抛出 `ValueError(…)`。
    #   当 `keep < 0` 时，抛出
    # `ValueError('keep_recent_tool_rounds cannot be negative')`。
    #   当 `compaction_target_tokens is not None and…` 时，抛出 `ValueError(…)`。
    async def prepare(
        self,
        messages: Sequence[Message],
        *,
        source_messages: Sequence[Message] | None = None,
        protected_source_indices: Sequence[int] = (),
        tools: Sequence[ToolDefinition] = (),
        model: str | None = None,
        provider: str | None = None,
        max_output_tokens: int | None = None,
        history_count: int | None = None,
        keep_recent_tool_rounds: int | None = None,
        summary_state: ConversationSummaryState | None = None,
        compaction_target_tokens: int | None = None,
        allow_compaction: bool = True,
    ) -> ContextDecision:
        candidate = tuple(messages)
        source = tuple(source_messages) if source_messages is not None else candidate
        # Explicit canonical sources include this run's completed work. The
        # optional count remains a source-boundary API, never a defer strategy.
        count = (
            len(source)
            if source_messages is not None or history_count is None
            else history_count
        )
        if not 0 <= count <= len(source):
            raise ValueError("history_count must be within the messages range")
        keep = (
            self.keep_recent_tool_rounds
            if keep_recent_tool_rounds is None
            else keep_recent_tool_rounds
        )
        if keep < 0:
            raise ValueError("keep_recent_tool_rounds cannot be negative")
        if compaction_target_tokens is not None and compaction_target_tokens <= 0:
            raise ValueError("compaction_target_tokens must be greater than zero")
        checked = ContextInventory.build(
            source,
            history_count=count,
            summary_state=summary_state,
            keep_recent_tool_rounds=keep,
            protected_source_indices=protected_source_indices,
        )
        valid_state = checked.summary_state
        already_applied = valid_state is not None and any(
            message.name in SUMMARY_MESSAGE_NAMES
            and message == valid_state.summary.to_message()
            for message in candidate
        )
        inventory = ContextInventory.from_projection(
            source[:count],
            source[count:],
            candidate,
            valid_state if already_applied else None,
            keep_recent_tool_rounds=keep,
            protected_source_indices=protected_source_indices,
        )
        if valid_state is not None and not already_applied:
            inventory = apply_summary(inventory, valid_state)
        capabilities = self.registry.lookup(provider, model)
        budget = self._budget_policy.compute(
            capabilities, max_output_tokens=max_output_tokens
        )
        meter = _RequestMeter(self, tuple(tools), model=model, provider=provider)
        before = meter.measure(inventory.messages)
        target = (
            budget.forced_target_tokens
            if compaction_target_tokens is None
            else compaction_target_tokens
        )
        pressure = (
            before.request >= budget.compact_ceiling_tokens
            or compaction_target_tokens is not None
        )
        folded = None
        if pressure and allow_compaction and self._conversation_reducer is not None:
            folded, inventory = await self._conversation_reducer.reduce_inventory(
                inventory,
                previous_state=valid_state,
                initial_estimated_input_tokens=before.request,
                target_tokens=target,
                estimate=meter.request,
            )
        after = meter.measure(inventory.messages)
        resulting_state = folded.summary_state if folded is not None else valid_state
        summary_updated = (
            resulting_state is not None
            and resulting_state.covered_message_count
            > (valid_state.covered_message_count if valid_state is not None else 0)
        )
        reached = not pressure or after.request <= target
        covered = (
            resulting_state.covered_message_count if resulting_state is not None else 0
        )
        pending_blocks = sum(
            isinstance(block, ConversationBlock)
            for block in partition_messages(source[covered:count])
        )
        error = folded.error if folded is not None else None
        if pressure and not allow_compaction:
            error = "compaction retry suppressed for unchanged failed source"
        elif pressure and self._conversation_reducer is None:
            error = "compaction is required but no summarizer is configured"
        reason = (
            f"strategy=append_then_compact;pressure={pressure};"
            f"trigger={budget.compact_ceiling_tokens};target={target};"
            f"original_estimated={before.request};"
            f"prepared_input_tokens={after.request};summary_updated={summary_updated}"
        )
        return ContextDecision(
            messages=inventory.messages,
            tools=tuple(tools),
            provider=capabilities.provider,
            model=capabilities.model,
            original_estimated_input_tokens=before.request,
            prepared_input_tokens=after.request,
            estimated_input_tokens=after.request,
            context_window=budget.context_window,
            reserved_output_tokens=budget.reserved_output_tokens,
            safety_margin_tokens=budget.safety_margin_tokens,
            input_budget=budget.input_budget,
            working_input_budget=budget.working_input_budget,
            hard_trigger_tokens=budget.hard_trigger_tokens,
            hard_target_tokens=budget.hard_target_tokens,
            trigger_tokens=budget.compact_ceiling_tokens,
            target_tokens=target,
            tool_result_budget_tokens=budget.tool_result_budget_tokens,
            compact_ceiling_tokens=budget.compact_ceiling_tokens,
            forced_target_tokens=budget.forced_target_tokens,
            tool_result_tokens_before=before.tool_results,
            tool_result_tokens_after=after.tool_results,
            tool_schema_tokens=meter.schema_tokens,
            message_tokens_before=before.messages,
            message_tokens_after=after.messages,
            unsummarized_conversation_blocks=pending_blocks,
            original_usage_ratio=before.request / budget.input_budget,
            prepared_usage_ratio=after.request / budget.input_budget,
            usage_ratio=after.request / budget.input_budget,
            requires_compaction=pressure,
            exceeds_input_budget=after.request > budget.input_budget,
            capability_source=capabilities.source.value,
            trimmed=inventory.messages != candidate,
            compaction_stage=(
                ContextCompactionStage.ROLLING_SUMMARY
                if summary_updated
                else ContextCompactionStage.NONE
            ),
            reached_target=reached,
            needs_next_compaction_stage=pressure and not reached,
            summary_state=resulting_state,
            summary_updated=summary_updated,
            summarized_conversation_blocks=(
                folded.summarized_conversation_blocks if folded else 0
            ),
            summary_usage=folded.summary_usage if folded else ModelUsage(),
            summary_provider=folded.summary_provider if folded else None,
            summary_model=folded.summary_model if folded else None,
            summary_duration_ms=folded.summary_duration_ms if folded else None,
            summary_error=error,
            reason=reason,
        )

    # 函数说明：ContextManager.remeasure
    # 用途：重新测量最终请求的消息、工具定义与预算占比，更新原决策而不再次触发上下文压缩
    # 。
    # 参数：
    #   decision：权限、上下文或审计决策，类型 `ContextDecision`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   tools：可用工具定义或工具实例集合，类型 `Sequence[ToolDefinition]`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `ContextDecision`；返回 `replace(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.registry.lookup` →
    # `self._budget_policy.compute` → `_RequestMeter` → `meter.measure` → `replace`。
    def remeasure(
        self,
        decision: ContextDecision,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
        model: str | None = None,
        provider: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ContextDecision:
        """Price final payload changes without triggering another compaction."""
        capabilities = self.registry.lookup(
            provider or decision.provider, model or decision.model
        )
        budget = self._budget_policy.compute(
            capabilities, max_output_tokens=max_output_tokens
        )
        meter = _RequestMeter(
            self,
            tuple(tools),
            model=model or decision.model,
            provider=provider or decision.provider,
        )
        costs = meter.measure(tuple(messages))
        target = decision.target_tokens or budget.forced_target_tokens
        return replace(
            decision,
            messages=tuple(messages),
            tools=tuple(tools),
            provider=capabilities.provider,
            model=capabilities.model,
            prepared_input_tokens=costs.request,
            estimated_input_tokens=costs.request,
            context_window=budget.context_window,
            reserved_output_tokens=budget.reserved_output_tokens,
            safety_margin_tokens=budget.safety_margin_tokens,
            input_budget=budget.input_budget,
            working_input_budget=budget.working_input_budget,
            hard_trigger_tokens=budget.hard_trigger_tokens,
            hard_target_tokens=budget.hard_target_tokens,
            compact_ceiling_tokens=budget.compact_ceiling_tokens,
            forced_target_tokens=budget.forced_target_tokens,
            trigger_tokens=budget.compact_ceiling_tokens,
            tool_schema_tokens=meter.schema_tokens,
            message_tokens_after=costs.messages,
            tool_result_tokens_after=costs.tool_results,
            original_usage_ratio=(decision.original_estimated_input_tokens or 0)
            / budget.input_budget,
            prepared_usage_ratio=costs.request / budget.input_budget,
            usage_ratio=costs.request / budget.input_budget,
            exceeds_input_budget=costs.request > budget.input_budget,
            reached_target=not decision.requires_compaction or costs.request <= target,
            needs_next_compaction_stage=decision.requires_compaction
            and costs.request > target,
        )

    # 函数说明：ContextManager._estimate_messages
    # 用途：估算消息序列，供模型上下文与输入预算使用。
    # 参数：
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   model：模型名称，类型 `str | None`。
    #   provider：模型或搜索服务商，类型 `str | None`。
    # 返回：类型 `int`；按分支返回 `method(messages, model=model, provider=provider)`；`
    # self.estimator.estimate_request(messages, tools=(), model=model, provider=provider
    # )`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`callable` → `method` →
    # `self.estimator.estimate_request`。
    # 分支与异常：
    #   当 `callable(method)` 时，返回
    # `method(messages, model=model, provider=provider)`。
    def _estimate_messages(
        self, messages: Sequence[Message], *, model: str | None, provider: str | None
    ) -> int:
        method = getattr(self.estimator, "estimate_messages", None)
        if callable(method):
            return method(messages, model=model, provider=provider)
        return self.estimator.estimate_request(
            messages, tools=(), model=model, provider=provider
        )

    # 函数说明：ContextManager._estimate_tools
    # 用途：估算工具集合，供模型上下文与输入预算使用。
    # 参数：
    #   tools：可用工具定义或工具实例集合，类型 `Sequence[ToolDefinition]`。
    #   model：模型名称，类型 `str | None`。
    #   provider：模型或搜索服务商，类型 `str | None`。
    # 返回：类型 `int`；按分支返回 `method(tools, model=model, provider=provider)`；
    # `self.estimator.estimate_request((), tools=tools, model=model, provider=provider)`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`callable` → `method` →
    # `self.estimator.estimate_request`。
    # 分支与异常：
    #   当 `callable(method)` 时，返回 `method(tools, model=model, provider=provider)`。
    def _estimate_tools(
        self,
        tools: Sequence[ToolDefinition],
        *,
        model: str | None,
        provider: str | None,
    ) -> int:
        method = getattr(self.estimator, "estimate_tools", None)
        if callable(method):
            return method(tools, model=model, provider=provider)
        return self.estimator.estimate_request(
            (), tools=tools, model=model, provider=provider
        )


@dataclass(frozen=True)
class _RequestCosts:
    request: int
    messages: int
    tool_results: int


class _RequestMeter:
    """Measure complete candidates; only proven additive estimators use caching."""

    # 函数说明：_RequestMeter.__init__
    # 用途：初始化 _RequestMeter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `ContextManager`。
    #   tools：可用工具定义或工具实例集合，类型 `tuple[ToolDefinition, ...]`。
    #   model：模型名称，类型 `str | None`。
    #   provider：模型或搜索服务商，类型 `str | None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`vars` → `manager._estimate_tools`
    # 。
    # 副作用与资源：
    #   更新对象字段：`self._manager`、`self._tools`、`self._model`、`self._provider`、
    # `self._cache_additive_costs`、`self._message_costs`、`self.schema_tokens`。
    def __init__(
        self,
        manager: ContextManager,
        tools: tuple[ToolDefinition, ...],
        *,
        model: str | None,
        provider: str | None,
    ) -> None:
        self._manager, self._tools = manager, tools
        self._model, self._provider = model, provider
        self._cache_additive_costs = (
            type(manager) is ContextManager
            and type(manager.estimator) is TokenEstimator
            and not any(
                name in vars(manager.estimator)
                for name in (
                    "estimate_request",
                    "estimate_messages",
                    "estimate_tools",
                    "estimate_text",
                )
            )
        )
        self._message_costs: dict[tuple[object, ...], int] = {}
        self.schema_tokens = manager._estimate_tools(
            tools, model=model, provider=provider
        )

    # 函数说明：_RequestMeter.request
    # 用途：只有确认估算器具有可加性时才复用消息与工具成本；其他情况按完整请求重新计费。
    # 参数：
    #   messages：本次处理的消息序列，类型 `tuple[Message, ...]`。
    # 返回：类型 `int`；按分支返回 `self.messages(messages) + self.schema_tokens`；
    # `self._manager.estimator.estimate_request(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.messages` →
    # `self._manager.estimator.estimate_request`。
    # 分支与异常：
    #   当 `self._cache_additive_costs` 时，返回
    # `self.messages(messages) + self.schema_tokens`。
    def request(self, messages: tuple[Message, ...]) -> int:
        if self._cache_additive_costs:
            return self.messages(messages) + self.schema_tokens
        return self._manager.estimator.estimate_request(
            messages, tools=self._tools, model=self._model, provider=self._provider
        )

    # 函数说明：_RequestMeter.messages
    # 用途：在模型上下文与输入预算中处理 `messages`，通过 `self._message_cost` 完成首个
    # 内部处理步骤。
    # 参数：
    #   messages：本次处理的消息序列，类型 `tuple[Message, ...]`。
    # 返回：类型 `int`；按分支返回
    # `sum((self._message_cost(message) for message in messages))`；`self._manager.
    # _estimate_messages(messages, model=self._model, provider=self._provider)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._message_cost` →
    # `self._manager._estimate_messages`。
    # 分支与异常：
    #   当 `self._cache_additive_costs` 时，返回
    # `sum((self._message_cost(message) for message in messages))`。
    def messages(self, messages: tuple[Message, ...]) -> int:
        if self._cache_additive_costs:
            return sum(self._message_cost(message) for message in messages)
        return self._manager._estimate_messages(
            messages, model=self._model, provider=self._provider
        )

    # 函数说明：_RequestMeter._message_cost
    # 用途：以消息角色、正文、调用标识和工具参数组成缓存键，复用同一消息的 Token 估算结
    # 果。
    # 参数：
    #   message：单条消息或通知，类型 `Message`。
    # 返回：类型 `int`；返回 `self._message_costs[key]`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps` →
    # `self._manager._estimate_messages`。
    def _message_cost(self, message: Message) -> int:
        key = (
            message.role,
            message.content,
            message.name,
            message.tool_call_id,
            tuple(
                (
                    call.name,
                    call.arguments
                    if isinstance(call.arguments, str)
                    else json.dumps(
                        call.arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                )
                for call in message.tool_calls
            ),
        )
        if key not in self._message_costs:
            self._message_costs[key] = self._manager._estimate_messages(
                (message,), model=self._model, provider=self._provider
            )
        return self._message_costs[key]

    # 函数说明：_RequestMeter.measure
    # 用途：测量_RequestMeter，供模型上下文与输入预算使用。
    # 参数：
    #   messages：本次处理的消息序列，类型 `tuple[Message, ...]`。
    # 返回：类型 `_RequestCosts`；返回 `_RequestCosts(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_RequestCosts` → `self.request` →
    #  `self.messages` → `_estimate_tool_result_tokens`。
    def measure(self, messages: tuple[Message, ...]) -> _RequestCosts:
        return _RequestCosts(
            request=self.request(messages),
            messages=self.messages(messages),
            tool_results=_estimate_tool_result_tokens(messages, self.messages),
        )


# 函数说明：_estimate_tool_result_tokens
# 用途：估算工具结果Token 用量，供模型上下文与输入预算使用。
# 参数：
#   messages：本次处理的消息序列，类型 `Sequence[Message]`。
#   estimate：`estimate`输入或配置值，类型 `Callable[[tuple[Message, ...]], int]`。
# 返回：类型 `int`；返回 `estimate(results)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`partition_messages` → `estimate`。
def _estimate_tool_result_tokens(
    messages: Sequence[Message], estimate: Callable[[tuple[Message, ...]], int]
) -> int:
    results = tuple(
        message
        for block in partition_messages(messages)
        if isinstance(block, ToolRoundBlock)
        for message in block.messages[1:]
    )
    return estimate(results)


__all__ = ["ContextCompactionStage", "ContextDecision", "ContextManager"]
