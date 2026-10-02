
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.types import ModelUsage
from app.runtime.agent.result import AgentStopReason
from app.runtime.run.models import RunStatus

__all__ = ["AgentRunTrace", "RunStatus", "RunUsageSummary"]


class AgentRunTrace(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    conversation_id: str | None = None
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    provider: str | None = None
    model: str | None = None
    steps: int = Field(default=0, ge=0)
    stop_reason: AgentStopReason | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    event_count: int = Field(default=0, ge=0)


class RunUsageSummary(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    main_agent: ModelUsage = Field(default_factory=ModelUsage)
    context_summary: ModelUsage = Field(default_factory=ModelUsage)
    memory_reflection: ModelUsage = Field(default_factory=ModelUsage)
    memory_maintenance: ModelUsage = Field(default_factory=ModelUsage)
    provider_total: ModelUsage = Field(default_factory=ModelUsage)
    tool_schema_tokens_estimated: int = Field(default=0, ge=0)
    memory_reflection_status: str = "not_run"
    memory_reflection_skip_reason: str | None = None
    context_summary_status: str = "not_run"
    context_summary_provider: str | None = None
    context_summary_model: str | None = None
    context_summary_duration_ms: float = Field(default=0.0, ge=0.0)
    main_agent_chargeable_tokens: int = Field(default=0, ge=0)
    run_budget_status: str = "not_configured"
    run_budget_reason: str | None = None
    run_budget_warning_tokens: int | None = Field(default=None, ge=1)
    run_budget_finalization_tokens: int | None = Field(default=None, ge=1)
    run_budget_hard_tokens: int | None = Field(default=None, ge=1)
    run_budget_warning_model_calls: int | None = Field(default=None, ge=1)
    run_budget_finalization_model_calls: int | None = Field(default=None, ge=1)
    run_budget_hard_model_calls: int | None = Field(default=None, ge=1)
