

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelProvider(StrEnum):
    OPENAI = "openai"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"


class ApiStyle(StrEnum):
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"


class AgentMode(StrEnum):

    NORMAL = "normal"
    PLAN = "plan"
    # MEA 长任务的三个角色；工具白名单见 app/tools/role_boundary.py
    MANAGE = "manage"
    EXECUTE = "execute"
    AUDIT = "audit"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: str | None = None


class ToolPermission(StrEnum):

    ALLOWED = "allowed"
    HUMAN_APPROVAL = "human_approval"
    FORBIDDEN = "forbidden"

    # 函数说明：ToolPermission.model_visible
    # 用途：返回 `self is not ToolPermission.FORBIDDEN`，提供 ToolPermission 的派生值。
    # 返回：类型 `bool`；返回 `self is not ToolPermission.FORBIDDEN`。
    def model_visible(self) -> bool:
        return self is not ToolPermission.FORBIDDEN


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    strict: bool | None = None
    permission: ToolPermission = ToolPermission.ALLOWED
    closing_allowed: bool = False
    record_output: bool = True


class ToolResult(BaseModel):

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    success: bool
    output: str | None = None
    error: str | None = None
    duration_ms: float = Field(ge=0)
    approval_wait_ms: float | None = Field(default=None, ge=0)
    execution_duration_ms: float | None = Field(default=None, ge=0)
    evidence_id: str | None = None
    output_chars: int | None = Field(default=None, ge=0)
    output_sha256: str | None = None
    output_truncated: bool | None = None
    evidence_error: str | None = None


class ModelRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    messages: tuple[Message, ...]
    model: str | None = None
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    # 函数说明：ModelRequest.require_messages
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `ModelRequest`；返回 `self`。
    # 分支与异常：
    #   当 `not self.messages` 时，抛出 `ValueError('messages cannot be empty')`。
    @model_validator(mode="after")
    def require_messages(self) -> ModelRequest:
        if not self.messages:
            raise ValueError("messages cannot be empty")
        return self


class ModelUsage(BaseModel):

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int | None = Field(default=None, ge=0)
    uncached_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_input_tokens: int | None = Field(default=None, ge=0)
    model_calls: int = Field(default=0, ge=0)


# 函数说明：add_model_usage
# 用途：添加模型用量，供模型请求与响应处理使用。
# 参数：
#   left：传给 `_has_model_usage` 的输入，类型 `ModelUsage`。
#   right：传给 `_has_model_usage` 的输入，类型 `ModelUsage`。
# 返回：类型 `ModelUsage`；返回 `ModelUsage(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_has_model_usage` → `ModelUsage` →
# `add_optional`。
def add_model_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:

    left_has_usage = _has_model_usage(left)
    right_has_usage = _has_model_usage(right)

    # 函数说明：add_model_usage.add_optional
    # 用途：添加`optional`，供模型请求与响应处理使用。
    # 参数：
    #   left_value：`left_value`输入或配置值，类型 `int | None`。
    #   right_value：`right_value`输入或配置值，类型 `int | None`。
    # 返回：类型 `int | None`；按分支返回 `right_value`；`left_value`；`None`；
    # `left_value + right_value`。
    # 分支与异常：
    #   当 `not left_has_usage` 时，返回 `right_value`。
    #   当 `not right_has_usage` 时，返回 `left_value`。
    #   当 `left_value is None or right_value is None` 时，返回 `None`。
    # 闭包依赖：从外层读取 `left_has_usage`、`right_has_usage`。
    def add_optional(left_value: int | None, right_value: int | None) -> int | None:
        if not left_has_usage:
            return right_value
        if not right_has_usage:
            return left_value
        if left_value is None or right_value is None:
            return None
        return left_value + right_value

    return ModelUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cached_input_tokens=add_optional(
            left.cached_input_tokens,
            right.cached_input_tokens,
        ),
        uncached_input_tokens=add_optional(
            left.uncached_input_tokens,
            right.uncached_input_tokens,
        ),
        cache_read_input_tokens=add_optional(
            left.cache_read_input_tokens,
            right.cache_read_input_tokens,
        ),
        cache_write_input_tokens=add_optional(
            left.cache_write_input_tokens,
            right.cache_write_input_tokens,
        ),
        model_calls=left.model_calls + right.model_calls,
    )


# 函数说明：_has_model_usage
# 用途：返回 `bool(…)`，提供 模型请求与响应处理 的派生值。
# 参数：
#   usage：模型调用用量统计，类型 `ModelUsage`。
# 返回：类型 `bool`；返回 `bool(…)`。
def _has_model_usage(usage: ModelUsage) -> bool:
    return bool(
        usage.input_tokens
        or usage.output_tokens
        or usage.total_tokens
        or usage.cached_input_tokens is not None
        or usage.uncached_input_tokens is not None
        or usage.cache_read_input_tokens is not None
        or usage.cache_write_input_tokens is not None
        or usage.model_calls
    )


class ModelResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    id: str
    provider: str
    model: str
    message: Message
    finish_reason: str | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    raw: dict[str, Any] | None = None
