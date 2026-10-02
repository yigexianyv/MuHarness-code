
from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.types import ModelUsage

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
_MEMORY_ID_RE = re.compile(r"^M[0-9]{3,}$")


class MaintenanceAction(StrEnum):

    ARCHIVE = "archive"
    DEFER = "defer"


class MemoryMaintenanceConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="MEMORY_MAINTENANCE_",
        extra="ignore",
    )

    enabled: bool = True
    provider: str | None = None
    model: str | None = None
    max_output_tokens: int = Field(default=800, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    candidate_limit: int = Field(default=5, ge=1, le=10)
    max_actions: int = Field(default=3, ge=1, le=10)

    # 函数说明：MemoryMaintenanceConfig.normalize_optional_text
    # 用途：校验并规范化模型字段 'provider'、'model'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`value.strip() or None`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('maintenance provider and model must be strings')`。
    @field_validator("provider", "model", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("maintenance provider and model must be strings")
        return value.strip() or None


class MemoryMaintenanceCandidate(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    summary: str
    content: str
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime
    access_count: int = Field(ge=0)

    # 函数说明：MemoryMaintenanceCandidate.normalize_id
    # 用途：校验并规范化模型字段 'id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().upper` →
    # `_MEMORY_ID_RE.fullmatch`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('maintenance candidate ID must be a string')`。
    #   当 `not _MEMORY_ID_RE.fullmatch(normalized)` 时，抛出 `ValueError(…)`。
    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("maintenance candidate ID must be a string")
        normalized = value.strip().upper()
        if not _MEMORY_ID_RE.fullmatch(normalized):
            raise ValueError("maintenance candidate ID must use the Mxxx format")
        return normalized


class MemoryMaintenanceInput(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    active_count: int = Field(ge=0)
    max_active: int = Field(gt=0)
    required_slots: int = Field(default=0, ge=0)
    candidates: tuple[MemoryMaintenanceCandidate, ...]


class MemoryMaintenanceDecision(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: MaintenanceAction
    memory_id: str | None = None
    reason: str

    # 函数说明：MemoryMaintenanceDecision.normalize_optional_id
    # 用途：校验并规范化模型字段 'memory_id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().upper` →
    # `_MEMORY_ID_RE.fullmatch`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('maintenance memory_id must be a string')`。
    #   当 `not _MEMORY_ID_RE.fullmatch(normalized)` 时，抛出
    # `ValueError('maintenance memory_id must use the Mxxx format')`。
    @field_validator("memory_id", mode="before")
    @classmethod
    def normalize_optional_id(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("maintenance memory_id must be a string")
        normalized = value.strip().upper()
        if not _MEMORY_ID_RE.fullmatch(normalized):
            raise ValueError("maintenance memory_id must use the Mxxx format")
        return normalized

    # 函数说明：MemoryMaintenanceDecision.normalize_reason
    # 用途：校验并规范化模型字段 'reason'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('maintenance reason must be a string')`。
    #   当 `not normalized` 时，抛出 `ValueError('maintenance reason cannot be empty')`
    # 。
    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("maintenance reason must be a string")
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("maintenance reason cannot be empty")
        return normalized

    # 函数说明：MemoryMaintenanceDecision.validate_action_fields
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `MemoryMaintenanceDecision`；返回 `self`。
    # 分支与异常：
    #   当 `self.action is MaintenanceAction.ARCHIVE and self.memory_id…` 时，抛出
    # `ValueError('archive decision requires memory_id')`。
    #   当 `self.action is MaintenanceAction.DEFER and self.memory_id…` 时，抛出
    # `ValueError('defer decision cannot contain memory_id')`。
    @model_validator(mode="after")
    def validate_action_fields(self) -> MemoryMaintenanceDecision:
        if self.action is MaintenanceAction.ARCHIVE and self.memory_id is None:
            raise ValueError("archive decision requires memory_id")
        if self.action is MaintenanceAction.DEFER and self.memory_id is not None:
            raise ValueError("defer decision cannot contain memory_id")
        return self


class MemoryMaintenanceProposal(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: MemoryMaintenanceDecision | None = None
    provider: str | None = None
    model: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    error: str | None = None


__all__ = [
    "MaintenanceAction",
    "MemoryMaintenanceCandidate",
    "MemoryMaintenanceConfig",
    "MemoryMaintenanceDecision",
    "MemoryMaintenanceInput",
    "MemoryMaintenanceProposal",
]
