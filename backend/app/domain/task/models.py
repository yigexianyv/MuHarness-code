
from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

TASK_ID_LENGTH = 32
_TASK_ID_RE = re.compile(rf"^[0-9a-f]{{{TASK_ID_LENGTH}}}$")
_MAX_TITLE_CHARS = 500
_MAX_TEXT_CHARS = 4_000
_MAX_ENTRY_CHARS = 2_000
_MAX_ENTRIES = 100
_MAX_STEPS = 100
_MAX_CONTRACT_CHARS = 12_000
MAX_APPLIED_OPS = 200


class TaskStatus(StrEnum):

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(StrEnum):

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TaskStepStatus(StrEnum):

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"
    # 被用户修订取代；只能由长任务（MEA）写入，必须引用修订 id
    SUPERSEDED = "superseded"


# 已经结束、不再需要执行的步骤状态；任务完成要求所有步骤都在其中
CLOSED_STEP_STATUSES = frozenset({TaskStepStatus.DONE, TaskStepStatus.SUPERSEDED})


class TaskStep(BaseModel):

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    status: TaskStepStatus = TaskStepStatus.TODO
    note: str | None = None
    acceptance: str | None = None  # 步骤验收标准，MEA 步骤审计的依据
    origin_requirements_revision: int = Field(default=1, ge=1)
    superseded_by: str | None = None  # 取代依据的用户修订 id，例如 A2
    superseded_reason: str | None = None

    # 函数说明：TaskStep.normalize_required_text
    # 用途：校验并规范化模型字段 'id'、'title'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    #   info：`info`输入或配置值，类型 `ValidationInfo`。
    # 返回：类型 `str`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_text`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('task step id and title must be strings')`。
    #   当 `not normalized` 时，抛出
    # `ValueError('task step id and title cannot be empty')`。
    #   当 `len(normalized) > maximum` 时，抛出
    # `ValueError('task step id or title is too long')`。
    @field_validator("id", "title", mode="before")
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> str:

        if not isinstance(value, str):
            raise TypeError("task step id and title must be strings")
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError("task step id and title cannot be empty")
        maximum = 128 if info.field_name == "id" else _MAX_TITLE_CHARS
        if len(normalized) > maximum:
            raise ValueError("task step id or title is too long")
        return normalized

    # 函数说明：TaskStep.normalize_note
    # 用途：校验并规范化模型字段 'note'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_optional_text`。
    # 分支与异常：
    #   当 `normalized is not None and len(normalized) > _MAX_TEXT_CHARS` 时，抛出
    # `ValueError('task step note is too long')`。
    @field_validator("note", mode="before")
    @classmethod
    def normalize_note(cls, value: object) -> str | None:

        normalized = _normalize_optional_text(value)
        if normalized is not None and len(normalized) > _MAX_TEXT_CHARS:
            raise ValueError("task step note is too long")
        return normalized

    # 函数说明：TaskStep.normalize_step_text
    # 用途：校验并规范化模型字段 'acceptance'、'superseded_reason'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_optional_text`。
    # 分支与异常：
    #   当 `normalized is not None and len(normalized) >…` 时，抛出 `ValueError(…)`。
    @field_validator("acceptance", "superseded_reason", mode="before")
    @classmethod
    def normalize_step_text(cls, value: object) -> str | None:

        normalized = _normalize_optional_text(value)
        if normalized is not None and len(normalized) > _MAX_ENTRY_CHARS:
            raise ValueError("task step acceptance or superseded_reason is too long")
        return normalized

    # 函数说明：TaskStep.normalize_superseded_by
    # 用途：校验并规范化模型字段 'superseded_by'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；返回
    # `normalized.upper() if normalized is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_optional_text` →
    # `normalized.upper`。
    @field_validator("superseded_by", mode="before")
    @classmethod
    def normalize_superseded_by(cls, value: object) -> str | None:

        normalized = _normalize_optional_text(value)
        return normalized.upper() if normalized is not None else None

    # 函数说明：TaskStep.validate_status_note
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `TaskStep`；返回 `self`。
    # 分支与异常：
    #   当 `not self.note` 时，抛出 `ValueError(…)`。
    #   当 `(self.status is TaskStepStatus.SUPERSEDED) != (…` 时，抛出 `ValueError(…)`。
    @model_validator(mode="after")
    def validate_status_note(self) -> TaskStep:

        if self.status in {TaskStepStatus.DONE, TaskStepStatus.BLOCKED}:
            if not self.note:
                raise ValueError(
                    f"task step with status {self.status.value} requires a note"
                )
        if (self.status is TaskStepStatus.SUPERSEDED) != (self.superseded_by is not None):
            raise ValueError(
                "superseded step requires superseded_by, and only superseded steps "
                "may carry it"
            )
        return self


class Task(BaseModel):

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str | None = None
    goal: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    constraints: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    key_facts: tuple[str, ...] = ()
    steps: tuple[TaskStep, ...] = ()
    contract: str | None = None  # MEA Manager 维护的任务契约全文，保留换行
    # MEA 已应用的操作：操作 ID → 内容摘要；和内容在同一次原子写入里保存
    applied_ops: dict[str, str] = Field(default_factory=dict)
    owner_conversation_id: str = Field(min_length=1, frozen=True)
    run_ids: tuple[str, ...] = ()
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    revision: int = Field(default=1, ge=1)

    # 函数说明：Task.normalize_optional_text
    # 用途：校验并规范化模型字段 'title'、'description'、'goal'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    #   info：`info`输入或配置值，类型 `ValidationInfo`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized or None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_text`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('text fields must be strings or None')`。
    #   当 `len(normalized) > maximum` 时，抛出
    # `ValueError(f'task {info.field_name} is too long')`。
    @field_validator("title", "description", "goal", mode="before")
    @classmethod
    def normalize_optional_text(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> str | None:

        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("text fields must be strings or None")
        normalized = _normalize_text(value)
        maximum = _MAX_TITLE_CHARS if info.field_name == "title" else _MAX_TEXT_CHARS
        if len(normalized) > maximum:
            raise ValueError(f"task {info.field_name} is too long")
        return normalized or None

    # 函数说明：Task.normalize_entries
    # 用途：校验并规范化模型字段 'constraints'、'state'、'key_facts'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `tuple[str, ...]`；返回 `_normalize_entries(value)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_entries`。
    @field_validator(
        "constraints",
        "state",
        "key_facts",
        mode="before",
    )
    @classmethod
    def normalize_entries(cls, value: object) -> tuple[str, ...]:

        return _normalize_entries(value)

    # 函数说明：Task.normalize_contract
    # 用途：校验并规范化模型字段 'contract'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized or None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.replace`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('task contract must be a string or None')`。
    #   当 `len(normalized) > _MAX_CONTRACT_CHARS` 时，抛出
    # `ValueError('task contract is too long')`。
    @field_validator("contract", mode="before")
    @classmethod
    def normalize_contract(cls, value: object) -> str | None:

        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("task contract must be a string or None")
        normalized = value.replace("\r\n", "\n").strip()
        if len(normalized) > _MAX_CONTRACT_CHARS:
            raise ValueError("task contract is too long")
        return normalized or None

    # 函数说明：Task.title_required
    # 用途：校验并规范化模型字段 'title'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str | None`。
    # 返回：类型 `str`；返回 `value`。
    # 分支与异常：
    #   当 `not value` 时，抛出 `ValueError('task title cannot be empty')`。
    @field_validator("title")
    @classmethod
    def title_required(cls, value: str | None) -> str:
        if not value:
            raise ValueError("task title cannot be empty")
        return value

    # 函数说明：Task.id_required
    # 用途：校验并规范化模型字段 'id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().lower` →
    # `_TASK_ID_RE.fullmatch`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出 `TypeError('task id must be a string')`
    # 。
    #   当 `not _TASK_ID_RE.fullmatch(normalized)` 时，抛出 `ValueError(…)`。
    @field_validator("id", mode="before")
    @classmethod
    def id_required(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("task id must be a string")
        normalized = value.strip().lower()
        if not _TASK_ID_RE.fullmatch(normalized):
            raise ValueError("task id must be a 32-character hexadecimal string")
        return normalized

    # 函数说明：Task.normalize_owner_conversation_id
    # 用途：校验并规范化模型字段 'owner_conversation_id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_optional_text`。
    # 分支与异常：
    #   当 `normalized is None` 时，抛出
    # `ValueError('task owner_conversation_id cannot be empty')`。
    @field_validator("owner_conversation_id", mode="before")
    @classmethod
    def normalize_owner_conversation_id(cls, value: object) -> str:

        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("task owner_conversation_id cannot be empty")
        return normalized

    # 函数说明：Task.normalize_datetime
    # 用途：校验并规范化模型字段 'created_at'、'updated_at'、'completed_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime | None`。
    # 返回：类型 `datetime | None`；按分支返回 `None`；`value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator("created_at", "updated_at", "completed_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:

        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("task datetimes must include timezone information")
        return value.astimezone(UTC)

    # 函数说明：Task.validate_invariants
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `Task`；返回 `self`。
    # 分支与异常：
    #   当 `len(step_ids) != len(set(step_ids))` 时，抛出
    # `ValueError('task step ids must be unique')`。
    #   当 `len(self.steps) > _MAX_STEPS` 时，抛出 `ValueError(…)`。
    #   当 `in_progress_count > 1` 时，抛出
    # `ValueError('task can contain at most one in_progress step')`。
    #   当 `self.status is TaskStatus.PAUSED and in_progress_count` 时，抛出
    # `ValueError('paused task cannot contain an in_progress step')`。
    @model_validator(mode="after")
    def validate_invariants(self) -> Task:

        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("task step ids must be unique")
        if len(self.steps) > _MAX_STEPS:
            raise ValueError(f"task cannot contain more than {_MAX_STEPS} steps")
        in_progress_count = sum(
            step.status is TaskStepStatus.IN_PROGRESS for step in self.steps
        )
        if in_progress_count > 1:
            raise ValueError("task can contain at most one in_progress step")
        if self.status is TaskStatus.PAUSED and in_progress_count:
            raise ValueError("paused task cannot contain an in_progress step")
        if self.status is TaskStatus.COMPLETED and self.steps:
            if any(step.status not in CLOSED_STEP_STATUSES for step in self.steps):
                raise ValueError(
                    "completed task requires all steps to be done or superseded"
                )
        if len(self.applied_ops) > MAX_APPLIED_OPS:
            raise ValueError(
                f"task applied_ops cannot contain more than {MAX_APPLIED_OPS} entries"
            )
        for field_name in ("constraints", "state", "key_facts"):
            if len(getattr(self, field_name)) > _MAX_ENTRIES:
                raise ValueError(
                    f"task {field_name} cannot contain more than {_MAX_ENTRIES} entries"
                )
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        if self.completed_at is not None:
            if self.completed_at < self.created_at:
                raise ValueError("completed_at cannot be earlier than created_at")
        return self

    # 函数说明：Task.progress_summary
    # 用途：处理任务状态与步骤管理中的 `progress_summary` 数据；结果及边界条件见下方说明
    # 。
    # 返回：类型 `str`；按分支返回
    # `f'[{status_text}] {self.title} ({done}/{total} 步骤完成)'`；
    # `f'[{status_text}] {self.title}'`。
    # 分支与异常：
    #   当 `total` 时，返回 `f'[{status_text}] {self.title} ({done}/{total} 步骤完成)'`
    # 。
    @property
    def progress_summary(self) -> str:

        total = sum(
            1 for step in self.steps if step.status is not TaskStepStatus.SUPERSEDED
        )
        done = sum(1 for step in self.steps if step.status is TaskStepStatus.DONE)
        status_text = self.status.value
        if total:
            return f"[{status_text}] {self.title} ({done}/{total} 步骤完成)"
        return f"[{status_text}] {self.title}"


class TaskPatch(BaseModel):

    model_config = ConfigDict(extra="forbid")

    goal: str | None = None
    status: TaskStatus | None = None
    state: tuple[str, ...] | None = None
    add_constraints: tuple[str, ...] = ()
    add_key_facts: tuple[str, ...] = ()
    replace_steps: tuple[TaskStep, ...] | None = None
    step_id: str | None = None
    step_status: TaskStepStatus | None = None
    step_note: str | None = None
    run_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=1)

    # 函数说明：TaskPatch.normalize_optional_text
    # 用途：校验并规范化模型字段 'goal'、'step_note'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；返回 `_normalize_optional_text(value)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_optional_text`。
    @field_validator("goal", "step_note", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        return _normalize_optional_text(value)

    # 函数说明：TaskPatch.normalize_optional_identifier
    # 用途：校验并规范化模型字段 'step_id'、'run_id'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_optional_text`。
    @field_validator("step_id", "run_id", mode="before")
    @classmethod
    def normalize_optional_identifier(cls, value: object) -> str | None:
        normalized = _normalize_optional_text(value)
        return normalized

    # 函数说明：TaskPatch.normalize_patch_entries
    # 用途：校验并规范化模型字段 'state'、'add_constraints'、'add_key_facts'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `tuple[str, ...] | None`；按分支返回 `None`；
    # `_normalize_entries(value)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_entries`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    @field_validator(
        "state",
        "add_constraints",
        "add_key_facts",
        mode="before",
    )
    @classmethod
    def normalize_patch_entries(cls, value: object) -> tuple[str, ...] | None:
        if value is None:
            return None
        return _normalize_entries(value)

    # 函数说明：TaskPatch.validate_step_update
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `TaskPatch`；返回 `self`。
    # 分支与异常：
    #   当 `(self.step_id is None) != (self.step_status is None)` 时，抛出
    # `ValueError(…)`。
    #   当 `self.step_note is not None and self.step_id is None` 时，抛出
    # `ValueError('step_note requires step_id and step_status')`。
    #   当 `'replace_steps' in self.model_fields_set and self.step_id…` 时，抛出
    # `ValueError(…)`。
    @model_validator(mode="after")
    def validate_step_update(self) -> TaskPatch:
        if (self.step_id is None) != (self.step_status is None):
            raise ValueError("step_id and step_status must be provided together")
        if self.step_note is not None and self.step_id is None:
            raise ValueError("step_note requires step_id and step_status")
        if "replace_steps" in self.model_fields_set and self.step_id is not None:
            raise ValueError(
                "replace_steps cannot be combined with a single step update"
            )
        return self

    # 函数说明：TaskPatch.has_changes
    # 用途：判断`changes`是否满足当前实现的条件。
    # 返回：类型 `bool`；返回 `explicit_nullable or any((self.status is not None, bool(
    # self.add_constraints), bool(…`。
    @property
    def has_changes(self) -> bool:

        explicit_nullable = bool(
            {"goal", "state", "replace_steps"} & self.model_fields_set
        )
        return explicit_nullable or any(
            (
                self.status is not None,
                bool(self.add_constraints),
                bool(self.add_key_facts),
                self.step_id is not None,
                self.run_id is not None,
            )
        )


# 函数说明：_normalize_text
# 用途：规范化文本，供任务状态与步骤管理使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `str`；返回 `' '.join(value.split()).strip()`。
def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


# 函数说明：_normalize_optional_text
# 用途：规范化文本，供任务状态与步骤管理使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `object`。
# 返回：类型 `str | None`；按分支返回 `None`；`normalized or None`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_text`。
# 分支与异常：
#   当 `value is None` 时，返回 `None`。
#   当 `not isinstance(value, str)` 时，抛出
# `TypeError('optional text must be a string or None')`。
def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text must be a string or None")
    normalized = _normalize_text(value)
    return normalized or None


# 函数说明：_normalize_entries
# 用途：规范化条目，供任务状态与步骤管理使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `object`。
# 返回：类型 `tuple[str, ...]`；按分支返回 `()`；`tuple(normalized)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_text` → `seen.add`。
# 分支与异常：
#   当 `value is None` 时，返回 `()`。
#   当 `not isinstance(entry, str)` 时，抛出 `TypeError('task entries must be strings')`
# 。
#   当 `len(text) > _MAX_ENTRY_CHARS` 时，抛出 `ValueError('task entry is too long')`。
def _normalize_entries(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Iterable[object] = (value,) if isinstance(value, str) else value
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in values:
        if not isinstance(entry, str):
            raise TypeError("task entries must be strings")
        text = _normalize_text(entry)
        if len(text) > _MAX_ENTRY_CHARS:
            raise ValueError("task entry is too long")
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return tuple(normalized)


__all__ = [
    "CLOSED_STEP_STATUSES",
    "MAX_APPLIED_OPS",
    "Task",
    "TASK_ID_LENGTH",
    "TaskPriority",
    "TaskPatch",
    "TaskStatus",
    "TaskStep",
    "TaskStepStatus",
]
