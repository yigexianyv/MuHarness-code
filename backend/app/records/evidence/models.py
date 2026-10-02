
from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceRecord(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    conversation_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    content_type: str = "text/plain; charset=utf-8"
    content_chars: int = Field(ge=0)
    content_bytes: int = Field(ge=0)
    sha256: str
    task_id: str | None = None
    task_step_id: str | None = None
    created_at: datetime

    # 函数说明：EvidenceRecord.normalize_created_at
    # 用途：校验并规范化模型字段 'created_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime`。
    # 返回：类型 `datetime`；返回 `value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出
    # `ValueError('evidence created_at must include timezone')`。
    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evidence created_at must include timezone")
        return value.astimezone(UTC)


class EvidenceDocument(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: EvidenceRecord
    content: str


class EvidenceSearchHit(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: EvidenceRecord
    snippet: str


__all__ = ["EvidenceDocument", "EvidenceRecord", "EvidenceSearchHit"]
