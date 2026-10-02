
"""MEA 的持久化模型：一次长任务（MeaRun）和它的每一轮（MeaRound）。"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.task import TaskOp


class MeaStatus(StrEnum):

    RUNNING = "running"
    WAITING_USER = "waiting_user"
    PAUSED = "paused"
    FINALIZING = "finalizing"  # 终结决定已落盘：不再跑轮次、不再接受修订
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_MEA_STATUSES = frozenset(
    {MeaStatus.COMPLETED, MeaStatus.BLOCKED, MeaStatus.FAILED, MeaStatus.CANCELLED}
)
# 这些状态下拒绝新的修订；之后到达的补充按新请求处理
CLOSED_MEA_STATUSES = TERMINAL_MEA_STATUSES | {MeaStatus.FINALIZING}


class RoundKind(StrEnum):

    NORMAL = "normal"
    AUDIT_ONLY = "audit_only"
    FINAL_AUDIT = "final_audit"
    RECOVERY_AUDIT = "recovery_audit"


STEP_AUDIT_KINDS = frozenset({RoundKind.AUDIT_ONLY, RoundKind.RECOVERY_AUDIT})


class RoundPhase(StrEnum):

    MANAGING = "managing"
    PLANNED = "planned"
    EXECUTING = "executing"
    EXECUTED = "executed"
    AUDITING = "auditing"
    AUDITED = "audited"
    APPLIED = "applied"
    INTERRUPTED = "interrupted"
    ABANDONED = "abandoned"


# 终态轮次：不会再推进
CLOSED_ROUND_PHASES = frozenset({RoundPhase.APPLIED, RoundPhase.ABANDONED})


class OnceNote(BaseModel):

    model_config = ConfigDict(extra="forbid")

    text: str
    consumed_in_round: int | None = None


class CompletionDecision(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Literal["completed", "blocked"]
    round_index: int
    auditor_run_id: str | None = None
    requirements_revision: int
    reason: str | None = None
    decided_at: datetime


class MeaRun(BaseModel):

    model_config = ConfigDict(extra="forbid")

    id: str
    task_id: str
    conversation_id: str
    status: MeaStatus = MeaStatus.RUNNING
    round_budget: int = Field(default=25, ge=1)
    requirements_revision: int = Field(default=1, ge=1)
    completion_decision: CompletionDecision | None = None
    extra_tools: tuple[str, ...] = ()
    # Executor 在沙箱内的 shell 调用是否自动批准（Auditor 的只读 shell 总是自动批准）
    auto_approve_sandbox: bool = True
    once_notes: tuple[OnceNote, ...] = ()
    pending_question: str | None = None
    pending_choices: tuple[str, ...] = ()
    pause_requested: bool = False
    # 停滞检测只看这一轮之后的轮次；你点继续、回答或补充要求时更新，免得旧记录立刻再次触发
    stall_guard_after: int = Field(default=0, ge=0)
    abort_reason: str | None = None
    final_response: str | None = None
    created_at: datetime
    updated_at: datetime


class StepAuditPatch(BaseModel):
    """步骤审计的结论。先落盘再应用到 Task，恢复时原样重放、不重新计算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: TaskOp
    stale: bool = False


class MeaRound(BaseModel):

    model_config = ConfigDict(extra="forbid")

    mea_run_id: str
    index: int = Field(ge=1)
    kind: RoundKind = RoundKind.NORMAL
    phase: RoundPhase = RoundPhase.MANAGING
    abandon_reason: str | None = None
    interrupt_reason: str | None = None
    route: str | None = None
    step_id: str | None = None
    plan_text: str | None = None
    subtask: str | None = None
    focus: str | None = None
    related_refs: tuple[str, ...] = ()
    plan_proposal: TaskOp | None = None
    plan_base_task_revision: int | None = None
    plan_requirements_revision: int | None = None
    exec_requirements_revision: int | None = None
    audit_requirements_revision: int | None = None
    manager_run_id: str | None = None
    executor_run_id: str | None = None
    auditor_run_id: str | None = None
    executor_output: str | None = None
    executor_rejections: dict[str, int] = Field(default_factory=dict)
    snapshot_before: str | None = None
    auditor_report: str | None = None
    audit_output_truncated: bool = False
    audit_status: str | None = None
    integrity_status: str | None = None
    contract_audit_status: str | None = None
    step_acceptance: str | None = None
    verdict_patch: StepAuditPatch | None = None
    stale_requirements: bool = False
    harness_feedback: str | None = None
    created_at: datetime
    updated_at: datetime

    # 函数说明：MeaRound.ref
    # 用途：返回 `f'round_{self.index:03d}'`，提供 MeaRound 的派生值。
    # 返回：类型 `str`；返回 `f'round_{self.index:03d}'`。
    @property
    def ref(self) -> str:
        return f"round_{self.index:03d}"

    # 函数说明：MeaRound.op_id
    # 用途：返回 `f'{self.mea_run_id}/r{self.index:03d}/{name}'`，提供 MeaRound 的派生值
    # 。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `str`；返回 `f'{self.mea_run_id}/r{self.index:03d}/{name}'`。
    def op_id(self, name: str) -> str:
        return f"{self.mea_run_id}/r{self.index:03d}/{name}"

    # 函数说明：MeaRound.append_feedback
    # 用途：追加`feedback`，供规划、执行、审计协作使用。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `MeaRound`；返回
    # `self.model_copy(update={'harness_feedback': combined})`。
    def append_feedback(self, text: str) -> MeaRound:
        combined = "\n\n".join(part for part in (self.harness_feedback, text) if part)
        return self.model_copy(update={"harness_feedback": combined})


# 函数说明：now_utc
# 用途：获取 UTC 时区的当前时间。
# 返回：类型 `datetime`；返回 `datetime.now(UTC)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now`。
def now_utc() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "CLOSED_MEA_STATUSES",
    "CLOSED_ROUND_PHASES",
    "STEP_AUDIT_KINDS",
    "TERMINAL_MEA_STATUSES",
    "CompletionDecision",
    "MeaRound",
    "MeaRun",
    "MeaStatus",
    "OnceNote",
    "RoundKind",
    "RoundPhase",
    "StepAuditPatch",
    "now_utc",
]
