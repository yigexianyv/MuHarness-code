
"""MEA 对 Task 的幂等写入操作。

每个操作有固定的操作 ID（例如 ``{mea}/r003/plan``）和确定的内容。``FileTaskStore.apply_op``
在任务锁内按顺序判断：

1. 操作 ID 已记录且内容摘要相同 → ``already_applied``，不重复写；
2. 操作 ID 已记录但摘要不同 → 抛 ``OpIdReusedError``（程序错误，不能当成已完成）；
3. 前置条件不成立 → ``conflict``，不强写；
4. 否则写入内容，并把“操作 ID → 摘要”和内容放在同一次原子文件写入里。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import TaskStep, TaskStepStatus

if TYPE_CHECKING:
    from .models import Task

_MAX_OP_ID_CHARS = 200


class OpResult(StrEnum):

    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    CONFLICT = "conflict"


class OpIdReusedError(RuntimeError):

    # 函数说明：OpIdReusedError.__init__
    # 用途：初始化 OpIdReusedError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   op_id：幂等操作标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.op_id`。
    def __init__(self, op_id: str) -> None:
        self.op_id = op_id
        super().__init__(
            f"task op id {op_id!r} was already applied with different content"
        )


class OpPrecondition(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int | None = Field(default=None, ge=1)
    step_id: str | None = None
    step_in: tuple[TaskStepStatus, ...] = ()
    all_steps_closed: bool = False

    # 函数说明：OpPrecondition.validate_step_check
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `OpPrecondition`；返回 `self`。
    # 分支与异常：
    #   当 `bool(self.step_id) != bool(self.step_in)` 时，抛出
    # `ValueError('step_id and step_in must be provided together')`。
    @model_validator(mode="after")
    def validate_step_check(self) -> OpPrecondition:
        if bool(self.step_id) != bool(self.step_in):
            raise ValueError("step_id and step_in must be provided together")
        return self


class StepSupersede(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(min_length=1)
    amendment_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class StepStatusChange(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(min_length=1)
    status: TaskStepStatus
    note: str | None = None

    # 函数说明：StepStatusChange.only_execution_statuses
    # 用途：校验并规范化模型字段 'status'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `TaskStepStatus`。
    # 返回：类型 `TaskStepStatus`；返回 `value`。
    # 分支与异常：
    #   当 `value not in allowed` 时，抛出 `ValueError(…)`。
    @field_validator("status")
    @classmethod
    def only_execution_statuses(cls, value: TaskStepStatus) -> TaskStepStatus:
        allowed = {TaskStepStatus.TODO, TaskStepStatus.IN_PROGRESS, TaskStepStatus.DONE}
        if value not in allowed:
            raise ValueError("step status change must be todo, in_progress or done")
        return value


class TaskOp(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    op_id: str = Field(min_length=1, max_length=_MAX_OP_ID_CHARS)
    precondition: OpPrecondition = Field(default_factory=OpPrecondition)
    state: tuple[str, ...] | None = None  # None = 不改；() = 清空
    set_contract: bool = False
    contract: str | None = None
    add_steps: tuple[TaskStep, ...] = ()
    supersede: tuple[StepSupersede, ...] = ()
    step_status: StepStatusChange | None = None
    complete_task: bool = False

    # 函数说明：TaskOp.validate_content
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `TaskOp`；返回 `self`。
    # 分支与异常：
    #   当 `not self.set_contract and self.contract is not None` 时，抛出
    # `ValueError('contract requires set_contract=True')`。
    #   当 `step.status is not TaskStepStatus.TODO` 时，抛出
    # `ValueError(f'new step {step.id} must start as todo')`。
    #   当 `not step.acceptance` 时，抛出
    # `ValueError(f'new step {step.id} requires acceptance')`。
    #   当 `not (self.state is not None or self.set_contract or…` 时，抛出
    # `ValueError('task op must contain at least one change')`。
    @model_validator(mode="after")
    def validate_content(self) -> TaskOp:
        if not self.set_contract and self.contract is not None:
            raise ValueError("contract requires set_contract=True")
        for step in self.add_steps:
            if step.status is not TaskStepStatus.TODO:
                raise ValueError(f"new step {step.id} must start as todo")
            if not step.acceptance:
                raise ValueError(f"new step {step.id} requires acceptance")
        if not (
            self.state is not None
            or self.set_contract
            or self.add_steps
            or self.supersede
            or self.step_status is not None
            or self.complete_task
        ):
            raise ValueError("task op must contain at least one change")
        return self

    # 函数说明：TaskOp.digest
    # 用途：内容摘要（不含操作 ID）。
    # 返回：类型 `str`；返回 `hashlib.sha256(payload.encode('utf-8')).hexdigest()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps` →
    # `hashlib.sha256(payload.encode('utf-8')).hexdigest` → `hashlib.sha256` →
    # `payload.encode`。
    def digest(self) -> str:
        """内容摘要（不含操作 ID）。同一个操作 ID 只能对应一份内容。"""

        payload = json.dumps(
            self.model_dump(mode="json", exclude={"op_id"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class OpOutcome:

    result: OpResult
    task: Task
    reason: str | None = None

    # 函数说明：OpOutcome.applied
    # 用途：返回 `self.result in (OpResult.APPLIED, OpResult.ALREADY_APPLIED)`，提供
    # OpOutcome 的派生值。
    # 返回：类型 `bool`；返回
    # `self.result in (OpResult.APPLIED, OpResult.ALREADY_APPLIED)`。
    @property
    def applied(self) -> bool:
        return self.result in (OpResult.APPLIED, OpResult.ALREADY_APPLIED)


__all__ = [
    "OpIdReusedError",
    "OpOutcome",
    "OpPrecondition",
    "OpResult",
    "StepStatusChange",
    "StepSupersede",
    "TaskOp",
]
