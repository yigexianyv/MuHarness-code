
from __future__ import annotations

from typing import Any, Protocol

from app.models.types import ToolResult
from app.tools.approval import ApprovalDecision, ApprovalRequest
from app.tools.hooks import ToolExecutionContext, ToolHook
from app.tools.permissions.models import PermissionRule

from .events import AgentEventType


class AgentEventEmitter(Protocol):

    # 函数说明：AgentEventEmitter.emit
    # 用途：发出AgentEventEmitter，供模型与工具执行循环使用。
    # 参数：
    #   event_type：事件输入或配置值，类型 `AgentEventType`。
    #   **payload：额外关键字参数，按实现处理或转交。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def emit(self, event_type: AgentEventType, **payload: Any) -> None:
        pass


class AgentEventHook(ToolHook):

    # 函数说明：AgentEventHook.__init__
    # 用途：初始化 AgentEventHook；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   emitter：执行事件发射器，类型 `AgentEventEmitter`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._emitter`。
    def __init__(self, emitter: AgentEventEmitter) -> None:
        self._emitter = emitter

    # 函数说明：AgentEventHook.before_execute
    # 用途：在 `execute` 前后执行 AgentEventHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._emitter.emit`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def before_execute(self, context: ToolExecutionContext) -> None:
        await self._emitter.emit(
            AgentEventType.TOOL_STARTED,
            step=context.step,
            tool_call=context.tool_call,
        )

    # 函数说明：AgentEventHook.on_approval_required
    # 用途：处理 `approval_required` 事件；调用路径和状态变化见下方说明。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._emitter.emit`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def on_approval_required(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
    ) -> None:
        await self._emitter.emit(
            AgentEventType.TOOL_APPROVAL_REQUIRED,
            step=context.step,
            tool_call=context.tool_call,
        )

    # 函数说明：AgentEventHook.on_approval_completed
    # 用途：处理 `approval_completed` 事件；调用路径和状态变化见下方说明。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    #   decision：权限、上下文或审计决策，类型 `ApprovalDecision`。
    #   rule：待匹配的权限规则，类型 `PermissionRule | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._emitter.emit`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def on_approval_completed(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        rule: PermissionRule | None = None,
    ) -> None:
        await self._emitter.emit(
            AgentEventType.TOOL_APPROVAL_COMPLETED,
            step=context.step,
            tool_call=context.tool_call,
            approval_decision=decision,
            rule_id=rule.id if rule is not None else None,
            rule_description=rule.description if rule is not None else None,
        )

    # 函数说明：AgentEventHook.after_execute
    # 用途：在 `execute` 前后执行 AgentEventHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._emitter.emit`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def after_execute(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        await self._emitter.emit(
            AgentEventType.TOOL_COMPLETED,
            step=context.step,
            tool_call=context.tool_call,
            tool_result=result,
        )


__all__ = ["AgentEventEmitter", "AgentEventHook"]
