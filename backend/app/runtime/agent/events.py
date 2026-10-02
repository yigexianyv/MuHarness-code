from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.types import Message, ModelUsage, ToolCall, ToolResult
from app.tools.approval import ApprovalDecision

from .result import AgentError, AgentResult, AgentStopReason


class AgentEventType(StrEnum):
    AGENT_STARTED = "agent_started"
    MODEL_STARTED = "model_started"
    MODEL_OUTPUT_DELTA = "model_output_delta"
    MODEL_REASONING_DELTA = "model_reasoning_delta"
    MODEL_COMPLETED = "model_completed"
    RUN_BUDGET_WARNING = "run_budget_warning"
    RUN_BUDGET_FINALIZING = "run_budget_finalizing"
    RUN_BUDGET_EXCEEDED = "run_budget_exceeded"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
    TOOL_APPROVAL_COMPLETED = "tool_approval_completed"
    MEMORY_REFLECTION_STARTED = "memory_reflection_started"
    MEMORY_REFLECTION_COMPLETED = "memory_reflection_completed"
    MEMORY_REFLECTION_FAILED = "memory_reflection_failed"
    MEMORY_REFLECTION_SKIPPED = "memory_reflection_skipped"
    MEMORY_MAINTENANCE_STARTED = "memory_maintenance_started"
    MEMORY_MAINTENANCE_COMPLETED = "memory_maintenance_completed"
    MEMORY_MAINTENANCE_FAILED = "memory_maintenance_failed"
    MEMORY_MAINTENANCE_SKIPPED = "memory_maintenance_skipped"
    SKILL_ACTIVATED = "skill_activated"
    SKILL_ACTIVATION_FAILED = "skill_activation_failed"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_CANCELLED = "agent_cancelled"


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    conversation_id: str | None = None
    sequence: int = Field(default=0, ge=0)
    type: AgentEventType
    event_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step: int | None = Field(default=None, ge=1)
    provider: str | None = None
    model: str | None = None
    delta: str | None = None
    reasoning_delta: str | None = None
    message: Message | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    usage: ModelUsage | None = None
    model_finish_reason: str | None = None
    model_duration_ms: float | None = Field(default=None, ge=0)
    requested_max_output_tokens: int | None = Field(default=None, gt=0)
    reasoning_chars: int | None = Field(default=None, ge=0)
    stop_reason: AgentStopReason | None = None
    error: AgentError | None = None
    result: AgentResult | None = None
    approval_decision: ApprovalDecision | None = None
    rule_id: str | None = None
    rule_description: str | None = None
    original_estimated_input_tokens: int | None = Field(default=None, ge=0)
    prepared_input_tokens: int | None = Field(default=None, ge=0)
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    context_trimmed: bool | None = None
    context_window: int | None = Field(default=None, ge=0)
    input_budget: int | None = Field(default=None, ge=0)
    working_input_budget: int | None = Field(default=None, ge=0)
    hard_trigger_tokens: int | None = Field(default=None, ge=0)
    hard_target_tokens: int | None = Field(default=None, ge=0)
    usage_ratio: float | None = Field(default=None, ge=0.0)
    trigger_tokens: int | None = Field(default=None, ge=0)
    target_tokens: int | None = Field(default=None, ge=0)
    tool_result_budget_tokens: int | None = Field(default=None, ge=0)
    tool_result_tokens_before: int | None = Field(default=None, ge=0)
    tool_result_tokens_after: int | None = Field(default=None, ge=0)
    tool_schema_tokens: int | None = Field(default=None, ge=0)
    message_tokens_before: int | None = Field(default=None, ge=0)
    message_tokens_after: int | None = Field(default=None, ge=0)
    unsummarized_conversation_blocks: int | None = Field(default=None, ge=0)
    conversation_block_limit: int | None = Field(default=None, gt=0)
    conversation_block_triggered: bool | None = None
    requires_compaction: bool | None = None
    exceeds_input_budget: bool | None = None
    capability_source: str | None = None
    original_usage_ratio: float | None = Field(default=None, ge=0.0)
    prepared_usage_ratio: float | None = Field(default=None, ge=0.0)
    compaction_stage: str | None = None
    compacted_tool_results: int | None = Field(default=None, ge=0)
    removed_tool_rounds: int | None = Field(default=None, ge=0)
    reached_target: bool | None = None
    needs_next_compaction_stage: bool | None = None
    summary_updated: bool | None = None
    summarized_conversation_blocks: int | None = Field(default=None, ge=0)
    summary_usage: ModelUsage | None = None
    summary_provider: str | None = None
    summary_model: str | None = None
    summary_duration_ms: float | None = Field(default=None, ge=0.0)
    summary_error: str | None = None
    cache_prefix_reused: bool | None = None
    cache_prefix_message_count: int | None = Field(default=None, ge=0)
    reflection_triggered: bool | None = None
    reflection_action: str | None = None
    reflection_duration_ms: float | None = Field(default=None, ge=0.0)
    reflection_attempts: int | None = Field(default=None, ge=0)
    reflection_finish_reason: str | None = None
    reflection_error: str | None = None
    reflection_skip_reason: str | None = None
    reflection_memory_id: str | None = None
    reflection_mutation_applied: bool | None = None
    reflection_maintenance_required: bool | None = None
    reflection_retention_candidate_ids: tuple[str, ...] = ()
    reflection_input_json: str | None = None
    reflection_raw_output: str | None = None
    maintenance_triggered: bool | None = None
    maintenance_action: str | None = None
    maintenance_duration_ms: float | None = Field(default=None, ge=0.0)
    maintenance_error: str | None = None
    maintenance_skip_reason: str | None = None
    maintenance_memory_id: str | None = None
    maintenance_reason: str | None = None
    maintenance_active_count: int | None = Field(default=None, ge=0)
    maintenance_max_active: int | None = Field(default=None, gt=0)
    maintenance_candidate_ids: tuple[str, ...] = ()
    maintenance_remaining_overflow: int | None = Field(default=None, ge=0)
    skill_name: str | None = None
    skill_scope: str | None = None
    skill_error: str | None = None
    available_skill_count: int | None = Field(default=None, ge=0)
    skill_catalog_tokens: int | None = Field(default=None, ge=0)
    active_skill_names: tuple[str, ...] = ()
    active_skill_tokens: int | None = Field(default=None, ge=0)
    active_skill_message_names: tuple[str, ...] = ()
    recall_candidate_ids: tuple[str, ...] = ()
    recall_mode: str | None = None
    prefix_decision: str | None = None
    prefix_rebuild_reason: str | None = None
    compact_ceiling_tokens: int | None = Field(default=None, ge=0)
    forced_target_tokens: int | None = Field(default=None, ge=0)
    run_budget_status: str | None = None
    run_budget_reason: str | None = None
    run_budget_chargeable_tokens: int | None = Field(default=None, ge=0)
    run_budget_model_calls: int | None = Field(default=None, ge=0)
    run_budget_warning_tokens: int | None = Field(default=None, ge=1)
    run_budget_finalization_tokens: int | None = Field(default=None, ge=1)
    run_budget_hard_tokens: int | None = Field(default=None, ge=1)
    run_budget_warning_model_calls: int | None = Field(default=None, ge=1)
    run_budget_finalization_model_calls: int | None = Field(default=None, ge=1)
    run_budget_hard_model_calls: int | None = Field(default=None, ge=1)

    # 函数说明：AgentEvent.validate_run_id
    # 用途：校验并规范化模型字段 'run_id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `normalized`。
    # 分支与异常：
    #   当 `not normalized` 时，抛出 `ValueError('run_id cannot be empty')`。
    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:

        normalized = value.strip()
        if not normalized:
            raise ValueError("run_id cannot be empty")
        return normalized

    # 函数说明：AgentEvent.normalize_event_time
    # 用途：校验并规范化模型字段 'event_time'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime`。
    # 返回：类型 `datetime`；返回 `value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出
    # `ValueError('event_time must include timezone information')`。
    @field_validator("event_time")
    @classmethod
    def normalize_event_time(cls, value: datetime) -> datetime:

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must include timezone information")
        return value.astimezone(UTC)


class AgentEventHandler(ABC):
    # 函数说明：AgentEventHandler.emit
    # 用途：发出AgentEventHandler，供模型与工具执行循环使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    @abstractmethod
    async def emit(self, event: AgentEvent) -> None:
        pass


class NullEventHandler(AgentEventHandler):
    # 函数说明：NullEventHandler.emit
    # 用途：发出NullEventHandler，供模型与工具执行循环使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def emit(self, event: AgentEvent) -> None:
        pass


class InMemoryEventHandler(AgentEventHandler):
    # 函数说明：InMemoryEventHandler.__init__
    # 用途：初始化 InMemoryEventHandler；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._events`。
    def __init__(self) -> None:
        self._events: list[AgentEvent] = []

    # 函数说明：InMemoryEventHandler.emit
    # 用途：发出InMemoryEventHandler，供模型与工具执行循环使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def emit(self, event: AgentEvent) -> None:
        self._events.append(event)

    # 函数说明：InMemoryEventHandler.events
    # 用途：返回 `tuple(self._events)`，提供 InMemoryEventHandler 的派生值。
    # 返回：类型 `tuple[AgentEvent, ...]`；返回 `tuple(self._events)`。
    @property
    def events(self) -> tuple[AgentEvent, ...]:
        return tuple(self._events)

    # 函数说明：InMemoryEventHandler.clear
    # 用途：清理InMemoryEventHandler，供模型与工具执行循环使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._events.clear`。
    def clear(self) -> None:
        self._events.clear()


class CompositeEventHandler(AgentEventHandler):
    # 函数说明：CompositeEventHandler.__init__
    # 用途：初始化 CompositeEventHandler；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   *handlers：额外位置参数，按实现向内部调用传递。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._handlers`。
    def __init__(self, *handlers: AgentEventHandler) -> None:
        self._handlers = handlers

    # 函数说明：CompositeEventHandler.emit
    # 用途：发出CompositeEventHandler，供模型与工具执行循环使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`handler.emit`。
    # 分支与异常：
    #   捕获 `Exception` 后，跳过当前循环项，继续处理后续项。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def emit(self, event: AgentEvent) -> None:
        for handler in self._handlers:
            try:
                await handler.emit(event)
            except Exception:
                continue


__all__ = [
    "AgentEvent",
    "AgentEventHandler",
    "AgentEventType",
    "CompositeEventHandler",
    "InMemoryEventHandler",
    "NullEventHandler",
]
