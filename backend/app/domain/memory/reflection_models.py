
from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.types import ModelUsage

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
_MEMORY_ID_RE = re.compile(r"^M[0-9]{3,}$")


class ReflectionAction(StrEnum):

    NONE = "none"
    CREATE = "create"
    UPDATE = "update"


class MemoryReflectionConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="MEMORY_REFLECTION_",
        extra="ignore",
    )

    enabled: bool = True
    provider: str | None = None
    model: str | None = None
    max_output_tokens: int = Field(default=2_000, gt=0)
    max_attempts: int = Field(default=2, ge=1, le=2)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_tool_context_chars: int = Field(default=8_000, ge=0, le=20_000)
    capture_raw_io: bool = False
    gate_enabled: bool = True

    # 函数说明：MemoryReflectionConfig.normalize_optional_text
    # 用途：校验并规范化模型字段 'provider'、'model'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`value.strip() or None`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('reflection provider and model must be strings')`。
    @field_validator("provider", "model", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("reflection provider and model must be strings")
        return value.strip() or None


class MemoryReflectionInput(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    conversation_id: str | None = None
    user_input: str
    final_answer: str
    tool_context: tuple[str, ...] = ()
    recalled_memory_ids: tuple[str, ...] = ()
    core_memory: str = ""
    memory_index: str = ""
    task_context: str = ""

    # 函数说明：MemoryReflectionInput.normalize_recalled_ids
    # 用途：校验并规范化模型字段 'recalled_memory_ids'。
    # 参数：
    #   values：待处理的值集合，类型 `tuple[str, ...]`。
    # 返回：类型 `tuple[str, ...]`；返回 `tuple(normalized)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().upper` →
    # `_MEMORY_ID_RE.fullmatch`。
    # 分支与异常：
    #   当 `not _MEMORY_ID_RE.fullmatch(memory_id)` 时，抛出
    # `ValueError('recalled memory IDs must use the Mxxx format')`。
    @field_validator("recalled_memory_ids")
    @classmethod
    def normalize_recalled_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            memory_id = value.strip().upper()
            if not _MEMORY_ID_RE.fullmatch(memory_id):
                raise ValueError("recalled memory IDs must use the Mxxx format")
            if memory_id not in normalized:
                normalized.append(memory_id)
        return tuple(normalized)


class ReflectionDecision(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ReflectionAction
    memory_id: str | None = None
    title: str | None = None
    summary: str | None = None
    content: str | None = None
    reason: str

    # 函数说明：ReflectionDecision.normalize_optional_text
    # 用途：校验并规范化模型字段 'memory_id'、'title'、'summary'、'content'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`value.strip() or None`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('reflection decision text fields must be strings')`。
    @field_validator(
        "memory_id",
        "title",
        "summary",
        "content",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("reflection decision text fields must be strings")
        return value.strip() or None

    # 函数说明：ReflectionDecision.normalize_reason
    # 用途：校验并规范化模型字段 'reason'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('reflection reason must be a string')`。
    #   当 `not normalized` 时，抛出 `ValueError('reflection reason cannot be empty')`。
    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("reflection reason must be a string")
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("reflection reason cannot be empty")
        return normalized

    # 函数说明：ReflectionDecision.validate_action_fields
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `ReflectionDecision`；按分支返回 `self`；
    # `self.model_copy(update={'memory_id': self.memory_id.upper()})`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_MEMORY_ID_RE.fullmatch` →
    # `self.memory_id.upper`。
    # 分支与异常：
    #   `self.action is ReflectionAction.NONE` 分支在完成前置处理后返回 `self`。
    #   当 `any(…)` 时，抛出
    # `ValueError('none decision cannot contain mutation fields')`。
    #   `self.action is ReflectionAction.CREATE` 分支在完成前置处理后返回 `self`。
    #   当 `self.memory_id is not None` 时，抛出
    # `ValueError('create decision cannot contain memory_id')`。
    # 副作用与资源：
    #   数据库操作：CREATE、UPDATE decision；连接与事务边界以 with/提交语句为准。
    @model_validator(mode="after")
    def validate_action_fields(self) -> ReflectionDecision:
        if self.action is ReflectionAction.NONE:
            if any(
                value is not None
                for value in (
                    self.memory_id,
                    self.title,
                    self.summary,
                    self.content,
                )
            ):
                raise ValueError("none decision cannot contain mutation fields")
            return self
        if self.action is ReflectionAction.CREATE:
            if self.memory_id is not None:
                raise ValueError("create decision cannot contain memory_id")
            if not self.title or not self.summary or not self.content:
                raise ValueError("create decision requires title, summary, and content")
            return self
        if not self.memory_id or not _MEMORY_ID_RE.fullmatch(self.memory_id.upper()):
            raise ValueError("update decision requires a valid memory_id")
        if not self.title or not self.summary or not self.content:
            raise ValueError(
                "update decision requires title, summary, and complete content"
            )
        return self.model_copy(update={"memory_id": self.memory_id.upper()})


class MemoryReflectionProposal(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ReflectionDecision | None = None
    provider: str | None = None
    model: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    attempts: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    error: str | None = None
    input_json: str | None = None
    raw_output: str | None = None


__all__ = [
    "MemoryReflectionConfig",
    "MemoryReflectionInput",
    "MemoryReflectionProposal",
    "ReflectionAction",
    "ReflectionDecision",
]
