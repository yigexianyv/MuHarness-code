
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationSource(StrEnum):
    MANUAL = "manual"
    AUTOMATION = "automation"


class TriggerContext(BaseModel):

    model_config = ConfigDict(extra="forbid")

    source: ConversationSource
    automation_id: str | None = None
    scheduled_for: datetime | None = None
    triggered_at: datetime | None = None

    # 函数说明：TriggerContext.normalize_datetime
    # 用途：校验并规范化模型字段 'scheduled_for'、'triggered_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime | None`。
    # 返回：类型 `datetime | None`；按分支返回 `None`；`value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator("scheduled_for", "triggered_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trigger datetimes must include timezone information")
        return value.astimezone(UTC)


class ConversationInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = None
    content: str
    trigger: TriggerContext = Field(
        default_factory=lambda: TriggerContext(source=ConversationSource.MANUAL)
    )


__all__ = [
    "ConversationInput",
    "ConversationSource",
    "TriggerContext",
]
