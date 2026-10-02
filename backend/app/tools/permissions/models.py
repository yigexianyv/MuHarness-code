
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..approval import ApprovalDecision, ApprovalResponse, ApprovalScope


class PermissionEffect(StrEnum):

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionRule(BaseModel):

    model_config = ConfigDict(extra="forbid")

    id: str
    tool_name: str
    scope: ApprovalScope
    scope_id: str
    effect: PermissionEffect = PermissionEffect.ALLOW
    matcher_type: str
    matcher: dict[str, Any]
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 函数说明：PermissionRule.validate_persisted_scope
    # 用途：校验并规范化模型字段 'scope'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `ApprovalScope`。
    # 返回：类型 `ApprovalScope`；返回 `value`。
    # 分支与异常：
    #   当 `value is ApprovalScope.ONCE` 时，抛出 `ValueError(…)`。
    @field_validator("scope")
    @classmethod
    def validate_persisted_scope(cls, value: ApprovalScope) -> ApprovalScope:
        if value is ApprovalScope.ONCE:
            raise ValueError("ONCE approval cannot be stored as a permission rule")
        return value

    # 函数说明：PermissionRule.normalize_created_at
    # 用途：校验并规范化模型字段 'created_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime`。
    # 返回：类型 `datetime`；按分支返回 `value.replace(tzinfo=UTC)`；
    # `value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.replace` → `value.astimezone`。
    # 分支与异常：
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，返回
    # `value.replace(tzinfo=UTC)`。
    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class PermissionVerdict(BaseModel):

    model_config = ConfigDict(extra="forbid")

    effect: PermissionEffect
    rule_id: str | None = None
    rule: PermissionRule | None = None


__all__ = [
    "ApprovalDecision",
    "ApprovalResponse",
    "ApprovalScope",
    "PermissionEffect",
    "PermissionRule",
    "PermissionVerdict",
]
