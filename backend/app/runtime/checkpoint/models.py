
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.types import Message, MessageRole, ToolCall, ToolResult
from app.runtime.agent.result import AgentStopReason


class CheckpointStatus(StrEnum):

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CheckpointPhase(StrEnum):

    STARTING = "starting"
    MODEL_REQUEST = "model_request"
    TOOL_EXECUTION = "tool_execution"
    TOOL_RESULTS_READY = "tool_results_ready"
    FINISHED = "finished"


class RunCheckpoint(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    conversation_id: str | None = None
    user_message: Message
    status: CheckpointStatus
    phase: CheckpointPhase
    step: int = Field(default=0, ge=0)
    pending_tool_calls: tuple[ToolCall, ...] = ()
    completed_tool_results: tuple[ToolResult, ...] = ()
    stop_reason: AgentStopReason | None = None
    error: str | None = None
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    recovered_by_run_id: str | None = None
    revision: int = Field(default=1, ge=1)

    # 函数说明：RunCheckpoint.validate_user_message
    # 用途：校验并规范化模型字段 'user_message'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `Message`。
    # 返回：类型 `Message`；返回 `value`。
    # 分支与异常：
    #   当 `value.role is not MessageRole.USER` 时，抛出
    # `ValueError('checkpoint user_message must have user role')`。
    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, value: Message) -> Message:
        if value.role is not MessageRole.USER:
            raise ValueError("checkpoint user_message must have user role")
        return value

    # 函数说明：RunCheckpoint.normalize_identifier
    # 用途：校验并规范化模型字段 'run_id'、'conversation_id'、'recovered_by_run_id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str | None`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not normalized` 时，抛出
    # `ValueError('checkpoint identifiers cannot be empty')`。
    @field_validator("run_id", "conversation_id", "recovered_by_run_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("checkpoint identifiers cannot be empty")
        return normalized

    # 函数说明：RunCheckpoint.normalize_datetime
    # 用途：校验并规范化模型字段 'started_at'、'updated_at'、'completed_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime | None`。
    # 返回：类型 `datetime | None`；按分支返回 `None`；`value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator("started_at", "updated_at", "completed_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checkpoint datetimes must include timezone information")
        return value.astimezone(UTC)


__all__ = [
    "CheckpointPhase",
    "CheckpointStatus",
    "RunCheckpoint",
]
