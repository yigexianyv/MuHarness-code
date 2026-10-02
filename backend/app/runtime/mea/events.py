"""长任务的前端推送与子 Run 事件接线。

- ``mea.status``：长任务本身每次写库后推送（完整的 MeaRun，不含轮次）；
- ``mea.round``：轮次每次写库后推送摘要（不含计划、执行报告和审计报告全文，前端用 mea.get 取）；
- ``mea.agent_event``：子 Run 的 AgentEvent，附带 mea_id 和角色。子 Run 的事件不走 ``agent.event``，
  这样现有聊天界面不会把角色输出当成对话回复显示。

``MeaRunGateway`` 包一层 RunManager：启动子 Run 时挂上 Trace 记录和上面的事件转发，
其余方法原样转发。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from app.runtime.agent.events import (
    AgentEvent,
    AgentEventHandler,
    AgentEventType,
    CompositeEventHandler,
)

from .models import MeaRound, MeaRun

logger = logging.getLogger("muharness.mea.events")

Broadcaster = Callable[[str, Any], Awaitable[None]]

_ROUND_SUMMARY_FIELDS = (
    "index",
    "kind",
    "phase",
    "route",
    "step_id",
    "subtask",
    "focus",
    "manager_run_id",
    "executor_run_id",
    "auditor_run_id",
    "audit_status",
    "integrity_status",
    "contract_audit_status",
    "step_acceptance",
    "stale_requirements",
    "audit_output_truncated",
    "abandon_reason",
    "interrupt_reason",
    "created_at",
    "updated_at",
)


# 函数说明：round_summary
# 用途：在规划、执行、审计协作中处理 `round_summary`，通过 `rnd.model_dump` 完成首个内部
# 处理步骤。
# 参数：
#   rnd：当前测试模型轮次，类型 `MeaRound`。
# 返回：类型 `dict[str, Any]`；返回 `data`。
def round_summary(rnd: MeaRound) -> dict[str, Any]:
    data = rnd.model_dump(mode="json", include=set(_ROUND_SUMMARY_FIELDS))
    data["ref"] = rnd.ref
    return data


class MeaEvents:
    """在 Host 启动后由服务端设置广播函数；没有设置时所有推送都是空操作。"""

    # 函数说明：MeaEvents.__init__
    # 用途：初始化 MeaEvents；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._broadcaster`。
    def __init__(self) -> None:
        self._broadcaster: Broadcaster | None = None

    # 函数说明：MeaEvents.set_broadcaster
    # 用途：设置`broadcaster`，供规划、执行、审计协作使用。
    # 参数：
    #   broadcaster：通知广播回调，类型 `Broadcaster | None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._broadcaster`。
    def set_broadcaster(self, broadcaster: Broadcaster | None) -> None:
        self._broadcaster = broadcaster

    # 函数说明：MeaEvents.publish
    # 用途：发布MeaEvents，供规划、执行、审计协作使用。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `Any`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._broadcaster` →
    # `logger.exception`。
    # 分支与异常：
    #   当 `self._broadcaster is None` 时，返回 `None`。
    #   捕获 `Exception` 后，执行异常处理调用 `logger.exception`。
    async def publish(self, method: str, params: Any) -> None:
        if self._broadcaster is None:
            return
        try:
            await self._broadcaster(method, params)
        except Exception:
            logger.exception("mea broadcast failed: %s", method)

    # 函数说明：MeaEvents.on_store_change
    # 用途：处理 `store_change` 事件；调用路径和状态变化见下方说明。
    # 参数：
    #   run：当前运行记录，类型 `MeaRun | None`。
    #   rnd：当前测试模型轮次，类型 `MeaRound | None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.publish` → `round_summary`。
    async def on_store_change(self, run: MeaRun | None, rnd: MeaRound | None) -> None:
        if run is not None:
            await self.publish("mea.status", {"mea": run.model_dump(mode="json")})
        if rnd is not None:
            await self.publish(
                "mea.round", {"mea_id": rnd.mea_run_id, "round": round_summary(rnd)}
            )

    # 函数说明：MeaEvents.agent_event_handler
    # 用途：返回 `_ForwardingHandler(self, mea_id, role)`，提供 MeaEvents 的派生值。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str | None`。
    #   role：规划、执行、审计等运行角色，类型 `str | None`。
    # 返回：类型 `AgentEventHandler`；返回 `_ForwardingHandler(self, mea_id, role)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_ForwardingHandler`。
    def agent_event_handler(self, mea_id: str | None, role: str | None) -> AgentEventHandler:
        return _ForwardingHandler(self, mea_id, role)


class _ForwardingHandler(AgentEventHandler):

    # 函数说明：_ForwardingHandler.__init__
    # 用途：初始化 _ForwardingHandler；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   events：事件集合，类型 `MeaEvents`。
    #   mea_id：长任务协作记录标识，类型 `str | None`。
    #   role：规划、执行、审计等运行角色，类型 `str | None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._events`、`self._mea_id`、`self._role`。
    def __init__(self, events: MeaEvents, mea_id: str | None, role: str | None) -> None:
        self._events = events
        self._mea_id = mea_id
        self._role = role

    # 函数说明：_ForwardingHandler.emit
    # 用途：发出_ForwardingHandler，供规划、执行、审计协作使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._events.publish`。
    async def emit(self, event: AgentEvent) -> None:
        await self._events.publish(
            "mea.agent_event",
            {"mea_id": self._mea_id, "role": self._role, "event": event.model_dump(mode="json")},
        )


class _ApprovalTracker(AgentEventHandler):
    """记录子 Run 是否正在等人工审批；等待审批的时间不计入角色的运行时限。"""

    # 函数说明：_ApprovalTracker.__init__
    # 用途：初始化 _ApprovalTracker；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   pending：待处理项输入或配置值，类型 `dict[str, int]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._pending`。
    def __init__(self, pending: dict[str, int]) -> None:
        self._pending = pending

    # 函数说明：_ApprovalTracker.emit
    # 用途：发出_ApprovalTracker，供规划、执行、审计协作使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._pending.pop`。
    async def emit(self, event: AgentEvent) -> None:
        if event.type is AgentEventType.TOOL_APPROVAL_REQUIRED:
            self._pending[event.run_id] = self._pending.get(event.run_id, 0) + 1
        elif event.type is AgentEventType.TOOL_APPROVAL_COMPLETED:
            remaining = self._pending.get(event.run_id, 0) - 1
            if remaining > 0:
                self._pending[event.run_id] = remaining
            else:
                self._pending.pop(event.run_id, None)


class MeaRunGateway:
    """MeaRunner 用到的 RunManager 子集；启动子 Run 时统一挂上事件处理器。"""

    # 函数说明：MeaRunGateway.__init__
    # 用途：初始化 MeaRunGateway；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   run_manager：运行管理者输入或配置值，类型 `Any`。
    #   trace_handler_factory：执行轨迹构造工厂，类型
    # `Callable[[], AgentEventHandler] | None`；默认 `None`。
    #   events：事件集合，类型 `MeaEvents | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._runs`、`self._trace_handler_factory`、`self._events`、
    # `self._pending_approvals`。
    def __init__(
        self,
        run_manager: Any,
        *,
        trace_handler_factory: Callable[[], AgentEventHandler] | None = None,
        events: MeaEvents | None = None,
    ) -> None:
        self._runs = run_manager
        self._trace_handler_factory = trace_handler_factory
        self._events = events
        self._pending_approvals: dict[str, int] = {}

    # 函数说明：MeaRunGateway.awaiting_approval
    # 用途：返回 `self._pending_approvals.get(run_id, 0) > 0`，提供 MeaRunGateway 的派生
    # 值。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `bool`；返回 `self._pending_approvals.get(run_id, 0) > 0`。
    def awaiting_approval(self, run_id: str) -> bool:
        return self._pending_approvals.get(run_id, 0) > 0

    # 函数说明：MeaRunGateway.start
    # 用途：启动MeaRunGateway，供规划、执行、审计协作使用。
    # 参数：
    #   user_message：当前用户消息，类型 `str`。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：类型 `tuple[str, Any]`；返回
    # `await self._runs.start(user_message, event_handler=handler, **kwargs)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_ApprovalTracker` →
    # `self._trace_handler_factory` → `self._events.agent_event_handler` →
    # `CompositeEventHandler` → `self._runs.start`。
    async def start(self, user_message: str, **kwargs: Any) -> tuple[str, Any]:
        handlers: list[AgentEventHandler] = [_ApprovalTracker(self._pending_approvals)]
        if self._trace_handler_factory is not None:
            handlers.append(self._trace_handler_factory())
        if self._events is not None:
            source = kwargs.get("source")
            role = source.split(":", 1)[1] if isinstance(source, str) and ":" in source else source
            handlers.append(self._events.agent_event_handler(kwargs.get("source_id"), role))
        handler: AgentEventHandler | None = None
        if len(handlers) == 1:
            handler = handlers[0]
        elif handlers:
            handler = CompositeEventHandler(*handlers)
        return await self._runs.start(user_message, event_handler=handler, **kwargs)

    # 函数说明：MeaRunGateway.wait
    # 用途：等待MeaRunGateway，供规划、执行、审计协作使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Any`；返回 `await self._runs.wait(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._runs.wait`。
    async def wait(self, run_id: str) -> Any:
        return await self._runs.wait(run_id)

    # 函数说明：MeaRunGateway.result
    # 用途：返回 `self._runs.result(run_id)`，提供 MeaRunGateway 的派生值。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Any`；返回 `self._runs.result(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._runs.result`。
    def result(self, run_id: str) -> Any:
        return self._runs.result(run_id)

    # 函数说明：MeaRunGateway.forget_results
    # 用途：清除结果集合，供规划、执行、审计协作使用。
    # 参数：
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._runs.forget_results` →
    # `self._pending_approvals.pop`。
    def forget_results(self, run_ids: tuple[str, ...]) -> None:
        self._runs.forget_results(run_ids)
        for run_id in run_ids:
            self._pending_approvals.pop(run_id, None)

    # 函数说明：MeaRunGateway.get_run
    # 用途：获取运行，供规划、执行、审计协作使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Any`；返回 `await self._runs.get_run(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._runs.get_run`。
    async def get_run(self, run_id: str) -> Any:
        return await self._runs.get_run(run_id)

    # 函数说明：MeaRunGateway.active_run_ids
    # 用途：运行`ids`，供规划、执行、审计协作使用。
    # 返回：类型 `tuple[str, ...]`；返回 `self._runs.active_run_ids`。
    @property
    def active_run_ids(self) -> tuple[str, ...]:
        return self._runs.active_run_ids

    # 函数说明：MeaRunGateway.cancel
    # 用途：取消MeaRunGateway，供规划、执行、审计协作使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Any`；返回 `await self._runs.cancel(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._runs.cancel`。
    async def cancel(self, run_id: str) -> Any:
        return await self._runs.cancel(run_id)


__all__ = ["MeaEvents", "MeaRunGateway", "round_summary"]
