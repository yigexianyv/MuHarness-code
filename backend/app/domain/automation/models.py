
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

_DEFAULT_TIMEZONE = "UTC"


class AutomationStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ScheduleKind(StrEnum):
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"


class Schedule(BaseModel):

    model_config = ConfigDict(extra="forbid")

    kind: ScheduleKind
    run_at: datetime | None = None
    interval_seconds: float | None = None
    cron_expr: str | None = None
    timezone: str = _DEFAULT_TIMEZONE

    # 函数说明：Schedule.validate_timezone
    # 用途：校验并规范化模型字段 'timezone'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `value`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ZoneInfo`。
    # 分支与异常：
    #   捕获 `Exception` 后，转换或抛出 `ValueError(f'invalid timezone: {value}')`。
    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError(f"invalid timezone: {value}") from exc
        return value

    # 函数说明：Schedule.validate_run_at
    # 用途：校验并规范化模型字段 'run_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime | None`。
    # 返回：类型 `datetime | None`；按分支返回 `None`；`value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator("run_at")
    @classmethod
    def validate_run_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "run_at must include an explicit timezone offset (e.g. "
                "2026-08-20T09:00:00+08:00)"
            )
        return value.astimezone(UTC)

    # 函数说明：Schedule.validate_interval
    # 用途：校验并规范化模型字段 'interval_seconds'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `float | None`。
    # 返回：类型 `float | None`；按分支返回 `None`；`value`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value <= 0` 时，抛出 `ValueError('interval_seconds must be > 0')`。
    @field_validator("interval_seconds")
    @classmethod
    def validate_interval(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("interval_seconds must be > 0")
        return value


class Automation(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = ""
    prompt: str
    conversation_id: str | None = None
    status: AutomationStatus
    schedule: Schedule
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_run_id: str | None = None
    created_at: datetime
    updated_at: datetime

    # 函数说明：Automation.normalize_identifier
    # 用途：校验并规范化模型字段 'id'、'conversation_id'、'last_run_id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str | None`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not normalized` 时，抛出
    # `ValueError('automation identifiers cannot be empty')`。
    @field_validator("id", "conversation_id", "last_run_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("automation identifiers cannot be empty")
        return normalized

    # 函数说明：Automation.normalize_datetime
    # 用途：校验并规范化模型字段 'next_run_at'、'last_run_at'、'created_at'、'updated_at
    # '。
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
        "next_run_at",
        "last_run_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("automation datetimes must include timezone information")
        return value.astimezone(UTC)


__all__ = [
    "Automation",
    "AutomationStatus",
    "Schedule",
    "ScheduleKind",
]
