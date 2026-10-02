
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.types import AgentMode


class RunStatus(StrEnum):

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.INTERRUPTED,
        }
    ),
    RunStatus.INTERRUPTED: frozenset(),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

TERMINAL_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.INTERRUPTED,
    }
)


class Run(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    conversation_id: str | None = None
    status: RunStatus
    user_message: str = ""
    created_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    stop_reason: str | None = None
    recovered_from_run_id: str | None = None
    source: str | None = None
    source_id: str | None = None
    scheduled_for: datetime | None = None
    triggered_at: datetime | None = None
    mode: AgentMode = AgentMode.NORMAL

    # 函数说明：Run.normalize_identifier
    # 用途：校验并规范化模型字段 'id'、'conversation_id'、'recovered_from_run_id'、'
    # source_id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str | None`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not normalized` 时，抛出 `ValueError('run identifiers cannot be empty')`。
    @field_validator("id", "conversation_id", "recovered_from_run_id", "source_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("run identifiers cannot be empty")
        return normalized

    # 函数说明：Run.normalize_datetime
    # 用途：校验并规范化模型字段 'created_at'、'started_at'、'updated_at'、'completed_at
    # '、'scheduled_for'、'triggered_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime | None`。
    # 返回：类型 `datetime | None`；按分支返回 `None`；`value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator(
        "created_at",
        "started_at",
        "updated_at",
        "completed_at",
        "scheduled_for",
        "triggered_at",
    )
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run datetimes must include timezone information")
        return value.astimezone(UTC)


__all__ = [
    "TERMINAL_STATUSES",
    "Run",
    "RunStatus",
    "_ALLOWED_TRANSITIONS",
]
