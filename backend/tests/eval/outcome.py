"""一次用例执行的全部可观察结果。

``Outcome`` 是纯数据：驱动器负责从运行时收集，检查和报告只读它。
这样检查逻辑不依赖真实模型，离线自测可以直接构造 ``Outcome``。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OUTPUT_KEEP_CHARS = 4_000


class ToolUse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    call_id: str | None = None
    round_index: int | None = None
    arguments: dict[str, Any] | str = Field(default_factory=dict)
    success: bool
    error: str | None = None
    output: str | None = None  # 只保留前 OUTPUT_KEEP_CHARS 个字符
    exit_code: int | None = None


class ApprovalUse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    decision: str


class ContextStep(BaseModel):
    """一次模型调用前的上下文准备结果，取自 model_started 事件。"""

    model_config = ConfigDict(extra="forbid")

    stage: str | None = None
    compacted_tool_results: int = 0
    removed_tool_rounds: int = 0
    summary_updated: bool = False
    reached_target: bool | None = None
    prepared_input_tokens: int | None = None
    summary_covered_after: int | None = None


class Reflection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_run", "skipped", "completed", "failed"] = "not_run"
    action: str | None = None
    mutation_applied: bool | None = None
    skip_reason: str | None = None


class TurnOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    say: str
    run_id: str | None = None
    conversation_id: str | None = None
    user_sequence: int | None = None
    answer: str = ""
    stop_reason: str | None = None
    steps: int = 0
    tools: list[ToolUse] = Field(default_factory=list)
    context: list[ContextStep] = Field(default_factory=list)
    approvals: list[str] = Field(default_factory=list)  # 每次审批的决定：approved / denied
    approval_calls: list[ApprovalUse] = Field(default_factory=list)
    received_tool_outputs: list[str] = Field(default_factory=list)
    reflection: Reflection = Field(default_factory=Reflection)
    chargeable_tokens: int = 0
    model_calls: int = 0
    failed_events: list[str] = Field(default_factory=list)  # agent_failed 的错误信息


class TaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    status: str
    steps: list[dict[str, str]] = Field(default_factory=list)  # {title, status}


class ArtifactState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    path: str | None = None
    url: str | None = None


class MemoryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    content: str


class RoundState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    kind: str
    phase: str
    step_id: str | None = None
    audit_status: str | None = None
    integrity: str | None = None
    step_acceptance: str | None = None


class MeaState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    abort_reason: str | None = None
    pending_question: str | None = None
    rounds: list[RoundState] = Field(default_factory=list)
    steps: dict[str, str] = Field(default_factory=dict)  # step id → status
    role_rejections: int = 0


class Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    variant: str
    attempt: int
    status: Literal["ok", "error", "timeout", "skipped"] = "ok"
    error: str | None = None
    duration_seconds: float = 0.0
    conversations: dict[str, str] = Field(default_factory=dict)
    turns: list[TurnOutcome] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)  # 结束时工作区里的文本文件
    initial_file_hashes: dict[str, str] = Field(default_factory=dict)
    file_hashes: dict[str, str] = Field(default_factory=dict)
    tasks: list[TaskState] = Field(default_factory=list)
    artifacts: list[ArtifactState] = Field(default_factory=list)
    memories: list[MemoryState] = Field(default_factory=list)
    core_memory: str = ""
    mea: MeaState | None = None
    chargeable_tokens: int = 0  # 全部运行（含长任务子运行、摘要、反思）的可计费 Token
    model_calls: int = 0

    # 函数说明：Outcome.last
    # 用途：返回 `self.turns[-1] if self.turns else None`，提供 Outcome 的派生值。
    # 返回：类型 `TurnOutcome | None`；返回 `self.turns[-1] if self.turns else None`。
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @property
    def last(self) -> TurnOutcome | None:
        return self.turns[-1] if self.turns else None

    # 函数说明：Outcome.all_tools
    # 用途：返回 `[tool for turn in self.turns for tool in turn.tools]`，提供 Outcome 的
    # 派生值。
    # 返回：类型 `list[ToolUse]`；返回
    # `[tool for turn in self.turns for tool in turn.tools]`。
    def all_tools(self) -> list[ToolUse]:
        return [tool for turn in self.turns for tool in turn.tools]


__all__ = [
    "ArtifactState",
    "ContextStep",
    "MeaState",
    "MemoryState",
    "OUTPUT_KEEP_CHARS",
    "Outcome",
    "Reflection",
    "RoundState",
    "TaskState",
    "ToolUse",
    "TurnOutcome",
]
