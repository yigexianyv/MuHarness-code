
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.types import (
    AgentMode,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

from .approval import ApprovalDecision, ApprovalRequest
from .permissions.models import PermissionRule


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:

    tool_call: ToolCall
    run_id: str | None = None
    conversation_id: str | None = None
    user_input: str | None = None
    step: int | None = None
    tool_definition: ToolDefinition | None = None
    arguments: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    mode: AgentMode | None = None


@dataclass(frozen=True, slots=True)
class ToolHookDecision:

    denied_reason: str | None = None
    approval_request: ApprovalRequest | None = None
    matched_rule: PermissionRule | None = None


class ToolHook:

    critical = False

    # 函数说明：ToolHook.before_execute
    # 用途：在 `execute` 前后执行 ToolHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `ToolHookDecision | None`；无结果值，显式返回 None。
    async def before_execute(
        self,
        context: ToolExecutionContext,
    ) -> ToolHookDecision | None:

        return None

    # 函数说明：ToolHook.on_approval_required
    # 用途：处理 `approval_required` 事件；调用路径和状态变化见下方说明。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def on_approval_required(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
    ) -> None:
        pass

    # 函数说明：ToolHook.on_approval_completed
    # 用途：处理 `approval_completed` 事件；调用路径和状态变化见下方说明。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    #   decision：权限、上下文或审计决策，类型 `ApprovalDecision`。
    #   rule：待匹配的权限规则，类型 `PermissionRule | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def on_approval_completed(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        rule: PermissionRule | None = None,
    ) -> None:
        pass

    # 函数说明：ToolHook.after_execute
    # 用途：在 `execute` 前后执行 ToolHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def after_execute(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        pass


class ToolHookRunner:

    # 函数说明：ToolHookRunner.__init__
    # 用途：初始化 ToolHookRunner；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   *hooks：额外位置参数，按实现向内部调用传递。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._hooks`。
    def __init__(self, *hooks: ToolHook) -> None:
        self._hooks = hooks

    # 函数说明：ToolHookRunner.before_execute
    # 用途：在 `execute` 前后执行 ToolHookRunner 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `ToolHookDecision | None`；按分支返回 `ToolHookDecision(denied_reason=f
    # 'Critical tool hook failed: {type(exc).__name__}: {exc}')`；`decision`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`hook.before_execute` →
    # `ToolHookDecision`。
    # 分支与异常：
    #   捕获 `Exception` 后，返回 `ToolHookDecision(denied_reason=f'Critical tool hook
    # failed: {type(exc).__name__}: {exc}')`。
    #   当 `hook.critical` 时，返回 `ToolHookDecision(…)`。
    async def before_execute(
        self,
        context: ToolExecutionContext,
    ) -> ToolHookDecision | None:
        decision: ToolHookDecision | None = None
        for hook in self._hooks:
            try:
                current = await hook.before_execute(context)
                if current is not None and current.denied_reason is not None:
                    decision = current
                elif decision is None and current is not None:
                    decision = current
            except Exception as exc:
                if hook.critical:
                    return ToolHookDecision(
                        denied_reason=(
                            f"Critical tool hook failed: {type(exc).__name__}: {exc}"
                        )
                    )
                continue
        return decision

    # 函数说明：ToolHookRunner.on_approval_required
    # 用途：处理 `approval_required` 事件；调用路径和状态变化见下方说明。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`hook.on_approval_required`。
    # 分支与异常：
    #   捕获 `Exception` 后，跳过当前循环项，继续处理后续项。
    async def on_approval_required(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
    ) -> None:
        for hook in self._hooks:
            try:
                await hook.on_approval_required(context, request)
            except Exception:
                continue

    # 函数说明：ToolHookRunner.on_approval_completed
    # 用途：处理 `approval_completed` 事件；调用路径和状态变化见下方说明。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    #   decision：权限、上下文或审计决策，类型 `ApprovalDecision`。
    #   rule：待匹配的权限规则，类型 `PermissionRule | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`hook.on_approval_completed`。
    # 分支与异常：
    #   捕获 `Exception` 后，跳过当前循环项，继续处理后续项。
    async def on_approval_completed(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        rule: PermissionRule | None = None,
    ) -> None:
        for hook in self._hooks:
            try:
                await hook.on_approval_completed(context, request, decision, rule)
            except Exception:
                continue

    # 函数说明：ToolHookRunner.after_execute
    # 用途：在 `execute` 前后执行 ToolHookRunner 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`hook.after_execute`。
    # 分支与异常：
    #   捕获 `Exception` 后，跳过当前循环项，继续处理后续项。
    async def after_execute(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        for hook in self._hooks:
            try:
                await hook.after_execute(context, result)
            except Exception:
                continue


__all__ = [
    "ToolExecutionContext",
    "ToolHook",
    "ToolHookDecision",
    "ToolHookRunner",
]
