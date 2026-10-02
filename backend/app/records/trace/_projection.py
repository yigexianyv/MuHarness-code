"""Derive Run update fields from an event without reading or mutating storage."""

from __future__ import annotations

from dataclasses import dataclass

from app.runtime.agent.events import AgentEvent, AgentEventType
from app.runtime.agent.result import AgentStopReason

from .models import RunStatus


@dataclass(frozen=True)
class _RunUpdate:
    """An event's proposed fields; SQLite retains the existing merge rules."""

    run_id: str
    conversation_id: str | None
    status: RunStatus | None
    completed_at: str | None
    provider: str | None
    model: str | None
    steps: int
    stop_reason: AgentStopReason | None
    input_tokens: int
    output_tokens: int
    total_tokens: int


# 函数说明：project_event
# 用途：在执行轨迹与用量查询中处理 `project_event`，通过 `event.event_time.isoformat` 完
# 成首个内部处理步骤。
# 参数：
#   event：待记录或转发的事件，类型 `AgentEvent`。
# 返回：类型 `_RunUpdate`；返回 `_RunUpdate(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`event.event_time.isoformat` →
# `_RunUpdate`。
def project_event(event: AgentEvent) -> _RunUpdate:
    status: RunStatus | None = None
    completed_at: str | None = None
    if event.type is AgentEventType.AGENT_COMPLETED:
        status = RunStatus.COMPLETED
        completed_at = event.event_time.isoformat()
    elif event.type is AgentEventType.AGENT_CANCELLED:
        status = RunStatus.CANCELLED
        completed_at = event.event_time.isoformat()
    elif event.type is AgentEventType.AGENT_FAILED:
        status = (
            RunStatus.INTERRUPTED
            if event.stop_reason is AgentStopReason.INTERRUPTED
            else RunStatus.FAILED
        )
        completed_at = event.event_time.isoformat()

    updates_main_model = event.type in {
        AgentEventType.AGENT_STARTED,
        AgentEventType.MODEL_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.AGENT_COMPLETED,
        AgentEventType.AGENT_FAILED,
    }
    provider = event.provider if updates_main_model else None
    model = event.model if updates_main_model else None
    usage = event.usage if updates_main_model else None
    return _RunUpdate(
        run_id=event.run_id,
        conversation_id=event.conversation_id,
        status=status,
        completed_at=completed_at,
        provider=provider,
        model=model,
        steps=event.step or 0,
        stop_reason=event.stop_reason,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
    )
