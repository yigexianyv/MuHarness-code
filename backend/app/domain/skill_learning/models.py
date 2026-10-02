
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

_SKILL_NAME_MAX_LENGTH = 64


# 函数说明：_normalize_text
# 用途：规范化文本，供技能候选提炼与审核使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `str`；返回 `' '.join(value.split()).strip()`。
def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()




class TaskCard(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    title: str
    description: str | None = None
    goal: str | None = None
    constraints: tuple[str, ...] = ()
    key_facts: tuple[str, ...] = ()
    final_steps: tuple[str, ...] = ()
    created_at: datetime
    completed_at: datetime | None = None
    run_count: int = 0

    # 函数说明：TaskCard.normalize_datetime
    # 用途：校验并规范化模型字段 'created_at'、'completed_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime | None`。
    # 返回：类型 `datetime | None`；按分支返回 `None`；`value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator("created_at", "completed_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("task card datetimes must include timezone information")
        return value.astimezone(UTC)

    # 函数说明：TaskCard.render
    # 用途：生成展示文本TaskCard，供技能候选提炼与审核使用。
    # 返回：类型 `str`；返回 `self.model_dump_json()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.model_dump_json`。
    def render(self) -> str:

        return self.model_dump_json()




class TaskPatternCluster(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    task_ids: tuple[str, ...]
    pattern_name: str
    description: str
    similarity_reason: str
    reusable_value: str

    # 函数说明：TaskPatternCluster.normalize_required_text
    # 用途：校验并规范化模型字段 'id'、'pattern_name'、'description'、'similarity_reason
    # '、'reusable_value'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_text`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('cluster text fields must be strings')`。
    #   当 `not normalized` 时，抛出 `ValueError('cluster text fields cannot be empty')`
    # 。
    @field_validator(
        "id",
        "pattern_name",
        "description",
        "similarity_reason",
        "reusable_value",
    )
    @classmethod
    def normalize_required_text(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("cluster text fields must be strings")
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError("cluster text fields cannot be empty")
        return normalized

    # 函数说明：TaskPatternCluster.normalize_task_ids
    # 用途：校验并规范化模型字段 'task_ids'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `tuple[str, ...]`；返回 `tuple(seen)`。
    # 分支与异常：
    #   当 `not isinstance(value, (list, tuple))` 时，抛出
    # `TypeError('task_ids must be a list')`。
    #   当 `not isinstance(item, str)` 时，抛出
    # `TypeError('task_ids must contain only strings')`。
    #   当 `not seen` 时，抛出 `ValueError('task_ids cannot be empty')`。
    @field_validator("task_ids")
    @classmethod
    def normalize_task_ids(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("task_ids must be a list")
        seen: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("task_ids must contain only strings")
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.append(normalized)
        if not seen:
            raise ValueError("task_ids cannot be empty")
        return tuple(seen)


class PatternMiningResult(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    clusters: tuple[TaskPatternCluster, ...] = ()




class SkillCandidateStatus(StrEnum):

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SkillCandidateAction(StrEnum):

    CREATE = "create"
    UPDATE = "update"


class SkillCandidateOrigin(StrEnum):

    PATTERN_MINING = "pattern_mining"
    AGENT_PROPOSAL = "agent_proposal"


class SkillCandidate(BaseModel):

    model_config = ConfigDict(extra="forbid")

    id: str
    origin: SkillCandidateOrigin = SkillCandidateOrigin.PATTERN_MINING
    action: SkillCandidateAction
    proposed_name: str
    description: str
    reason: str
    procedure: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()
    source_task_ids: tuple[str, ...] = ()
    source_run_ids: tuple[str, ...] = ()
    source_conversation_id: str | None = None
    source_tool_call_id: str | None = None
    existing_skill_name: str | None = None
    status: SkillCandidateStatus = SkillCandidateStatus.PENDING
    created_at: datetime
    reviewed_at: datetime | None = None
    evidence_summary: str = ""

    # 函数说明：SkillCandidate.normalize_required_text
    # 用途：校验并规范化模型字段 'id'、'proposed_name'、'description'、'reason'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_text`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('candidate text fields must be strings')`。
    #   当 `not normalized` 时，抛出
    # `ValueError('candidate text fields cannot be empty')`。
    @field_validator("id", "proposed_name", "description", "reason", mode="before")
    @classmethod
    def normalize_required_text(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("candidate text fields must be strings")
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError("candidate text fields cannot be empty")
        return normalized

    # 函数说明：SkillCandidate.validate_proposed_name
    # 用途：校验并规范化模型字段 'proposed_name'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `validate_skill_name(value)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_skill_name`。
    # 分支与异常：
    #   捕获 `ValueError` 后，转换或抛出
    # `ValueError(f'proposed skill name is invalid: {exc}')`。
    @field_validator("proposed_name")
    @classmethod
    def validate_proposed_name(cls, value: str) -> str:
        from app.domain.skills import validate_skill_name

        try:
            return validate_skill_name(value)
        except ValueError as exc:
            raise ValueError(f"proposed skill name is invalid: {exc}") from exc

    # 函数说明：SkillCandidate.normalize_optional_skill_name
    # 用途：校验并规范化模型字段 'existing_skill_name'、'source_conversation_id'、'
    # source_tool_call_id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized or None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_text`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('existing_skill_name must be a string or None')`。
    @field_validator(
        "existing_skill_name",
        "source_conversation_id",
        "source_tool_call_id",
        mode="before",
    )
    @classmethod
    def normalize_optional_skill_name(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("existing_skill_name must be a string or None")
        normalized = _normalize_text(value)
        return normalized or None

    # 函数说明：SkillCandidate.normalize_datetime
    # 用途：校验并规范化模型字段 'created_at'、'reviewed_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime | None`。
    # 返回：类型 `datetime | None`；按分支返回 `None`；`value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator("created_at", "reviewed_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("candidate datetimes must include timezone information")
        return value.astimezone(UTC)

    # 函数说明：SkillCandidate.normalize_entries
    # 用途：校验并规范化模型字段 'procedure'、'pitfalls'、'verification'、'
    # source_task_ids'、'source_run_ids'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `tuple[str, ...]`；按分支返回 `()`；`tuple(seen)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_text`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `()`。
    #   当 `not isinstance(value, (list, tuple))` 时，抛出
    # `TypeError('candidate entries must be lists')`。
    #   当 `not isinstance(item, str)` 时，抛出
    # `TypeError('candidate entries must contain only strings')`。
    @field_validator(
        "procedure",
        "pitfalls",
        "verification",
        "source_task_ids",
        "source_run_ids",
        mode="before",
    )
    @classmethod
    def normalize_entries(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise TypeError("candidate entries must be lists")
        seen: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("candidate entries must contain only strings")
            normalized = _normalize_text(item)
            if normalized and normalized not in seen:
                seen.append(normalized)
        return tuple(seen)

    # 函数说明：SkillCandidate.validate_action
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `SkillCandidate`；返回 `self`。
    # 分支与异常：
    #   当 `self.action is SkillCandidateAction.UPDATE and (not…` 时，抛出
    # `ValueError('update candidate requires existing_skill_name')`。
    #   当 `self.action is SkillCandidateAction.CREATE and…` 时，抛出 `ValueError(…)`。
    #   当 `not self.source_task_ids` 时，抛出 `ValueError(…)`。
    #   当 `not (self.source_run_ids and self.source_conversation_id…` 时，抛出
    # `ValueError(…)`。
    # 副作用与资源：
    #   数据库操作：UPDATE candidate、CREATE；连接与事务边界以 with/提交语句为准。
    @model_validator(mode="after")
    def validate_action(self) -> SkillCandidate:
        if self.action is SkillCandidateAction.UPDATE and not self.existing_skill_name:
            raise ValueError("update candidate requires existing_skill_name")
        if (
            self.action is SkillCandidateAction.CREATE
            and self.existing_skill_name is not None
        ):
            raise ValueError("create candidate cannot have existing_skill_name")
        if self.origin is SkillCandidateOrigin.PATTERN_MINING:
            if not self.source_task_ids:
                raise ValueError(
                    "pattern mining candidate must reference at least one source task"
                )
        elif not (
            self.source_run_ids
            and self.source_conversation_id
            and self.source_tool_call_id
        ):
            raise ValueError(
                "agent proposal candidate requires run, conversation, "
                "and tool call provenance"
            )
        if not self.procedure:
            raise ValueError("candidate must contain at least one procedure step")
        return self


__all__ = [
    "PatternMiningResult",
    "SkillCandidate",
    "SkillCandidateAction",
    "SkillCandidateOrigin",
    "SkillCandidateStatus",
    "TaskCard",
    "TaskPatternCluster",
]
