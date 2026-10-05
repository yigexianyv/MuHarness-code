
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.models.types import Message, ModelUsage, ToolCall, ToolResult
from app.runtime.context.summary import ConversationSummaryState
from app.runtime.context.tool_views import ToolResultView


class AgentStopReason(StrEnum):

    FINAL_ANSWER = "final_answer"
    CONTEXT_ERROR = "context_error"
    MODEL_ERROR = "model_error"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    MAX_STEPS = "max_steps"
    RUN_BUDGET = "run_budget"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class AgentError(BaseModel):

    model_config = ConfigDict(extra="forbid")

    type: str
    message: str


class ToolCallRecord(BaseModel):

    model_config = ConfigDict(extra="forbid")

    round_index: int
    tool_call: ToolCall
    result: ToolResult


class ToolRound(BaseModel):

    model_config = ConfigDict(extra="forbid")

    round_index: int
    assistant_message: Message
    records: tuple[ToolCallRecord, ...] = ()


class AgentResult(BaseModel):

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    final_message: Message
    messages: tuple[Message, ...]
    steps: int = 0
    stop_reason: AgentStopReason
    tool_rounds: tuple[ToolRound, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()
    usage: ModelUsage = Field(default_factory=ModelUsage)
    error: AgentError | None = None
    summary_state: ConversationSummaryState | None = None
    tool_result_views: tuple[ToolResultView, ...] = ()
    plan_task_id: str | None = None
    model_finish_reason: str | None = None
    unresolved_output_truncation: bool = False

    # 函数说明：AgentResult.ok
    # 用途：返回 `self.stop_reason is AgentStopReason.FINAL_ANSWER`，提供 AgentResult 的
    # 派生值。
    # 返回：类型 `bool`；返回 `self.stop_reason is AgentStopReason.FINAL_ANSWER`。
    @property
    def ok(self) -> bool:
        return self.stop_reason is AgentStopReason.FINAL_ANSWER

    # 函数说明：AgentResult.content
    # 用途：返回 `self.final_message.content`，提供 AgentResult 的派生值。
    # 返回：类型 `str | None`；返回 `self.final_message.content`。
    @property
    def content(self) -> str | None:
        return self.final_message.content

    # 函数说明：AgentResult.role
    # 用途：返回 `self.final_message.role.value`，提供 AgentResult 的派生值。
    # 返回：类型 `str`；返回 `self.final_message.role.value`。
    @property
    def role(self) -> str:
        return self.final_message.role.value
