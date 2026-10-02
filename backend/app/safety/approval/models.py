
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApprovalRequestStatus(StrEnum):

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    CANCELLED = "cancelled"


class ApprovalRequest(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    run_id: str | None = None
    conversation_id: str | None = None
    tool_name: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    status: ApprovalRequestStatus = ApprovalRequestStatus.PENDING
    created_at: datetime
    resolved_at: datetime | None = None

    # 函数说明：ApprovalRequest.normalize_identifier
    # 用途：校验并规范化模型字段 'run_id'、'conversation_id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str | None`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not normalized` 时，抛出
    # `ValueError('approval identifiers cannot be empty')`。
    @field_validator("run_id", "conversation_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("approval identifiers cannot be empty")
        return normalized

    # 函数说明：ApprovalRequest.normalize_datetime
    # 用途：校验并规范化模型字段 'created_at'、'resolved_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime | None`。
    # 返回：类型 `datetime | None`；按分支返回 `None`；`value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator("created_at", "resolved_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval datetimes must include timezone information")
        return value.astimezone(UTC)


__all__ = ["ApprovalRequest", "ApprovalRequestStatus"]
