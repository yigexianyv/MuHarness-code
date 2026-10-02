
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArtifactKind(StrEnum):
    FILE = "file"
    URL = "url"


class Artifact(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: ArtifactKind
    title: str = ""
    description: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    size_bytes: int = 0
    sha256: str | None = None
    run_id: str | None = None
    conversation_id: str | None = None
    task_id: str | None = None
    source_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 函数说明：Artifact.id_required
    # 用途：校验并规范化模型字段 'id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `normalized`。
    # 分支与异常：
    #   当 `not normalized` 时，抛出 `ValueError('artifact id cannot be empty')`。
    @field_validator("id")
    @classmethod
    def id_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("artifact id cannot be empty")
        return normalized

    # 函数说明：Artifact.created_at_valid
    # 用途：校验并规范化模型字段 'created_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime`。
    # 返回：类型 `datetime`；返回 `value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出
    # `ValueError('created_at must include timezone information')`。
    @field_validator("created_at")
    @classmethod
    def created_at_valid(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        return value.astimezone(UTC)

    # 函数说明：Artifact.public_dict
    # 用途：将当前记录转为字典载荷，具体公开字段及转换规则由返回表达式确定。
    # 返回：类型 `dict[str, object]`；字典，包含字段 `id`、`kind`、`title`、
    # `description`、`filename`、`mime_type`、`size_bytes`、`sha256`、`run_id`、
    # `conversation_id`、`task_id`、`source_url`、`created_at`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.created_at.isoformat`。
    def public_dict(self) -> dict[str, object]:

        return {
            "id": self.id,
            "kind": self.kind.value,
            "title": self.title,
            "description": self.description,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "task_id": self.task_id,
            "source_url": self.source_url,
            "created_at": (
                self.created_at.isoformat() if self.created_at is not None else None
            ),
        }


__all__ = ["Artifact", "ArtifactKind"]
