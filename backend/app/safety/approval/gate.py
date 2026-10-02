
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from app.tools.approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalResponse,
)
from app.tools.approval import (
    ApprovalRequest as ApprovalSubmission,
)

from .models import ApprovalRequest as ApprovalRecord
from .models import ApprovalRequestStatus
from .store import SQLiteApprovalStore

Broadcaster = Callable[[str, Any], Awaitable[None]]


class WebApprovalGate(ApprovalGate):

    # 函数说明：WebApprovalGate.__init__
    # 用途：初始化 WebApprovalGate；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteApprovalStore`。
    #   broadcaster：通知广播回调，类型 `Broadcaster | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`、`self._broadcaster`、`self._pending`。
    def __init__(
        self,
        store: SQLiteApprovalStore,
        *,
        broadcaster: Broadcaster | None = None,
    ) -> None:
        self._store = store
        self._broadcaster = broadcaster
        self._pending: dict[str, asyncio.Future[ApprovalResponse]] = {}

    # 函数说明：WebApprovalGate.set_broadcaster
    # 用途：设置`broadcaster`，供人工审批请求与等待使用。
    # 参数：
    #   broadcaster：通知广播回调，类型 `Broadcaster`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._broadcaster`。
    def set_broadcaster(self, broadcaster: Broadcaster) -> None:

        self._broadcaster = broadcaster

    # 函数说明：WebApprovalGate.pending_count
    # 用途：统计待处理项，供人工审批请求与等待使用。
    # 返回：类型 `int`；返回 `len(self._pending)`。
    @property
    def pending_count(self) -> int:

        return len(self._pending)

    # 函数说明：WebApprovalGate.request_approval
    # 用途：保存待决审批并等待前端回复，将决定返回工具调用方。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalSubmission`。
    # 返回：类型 `ApprovalResponse`；返回 `await future`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.create` →
    # `asyncio.get_running_loop` → `loop.create_future` → `self._notify` →
    # `self._pending.pop`。
    async def request_approval(
        self,
        request: ApprovalSubmission,
    ) -> ApprovalResponse:

        """保存待决审批并等待前端回复，将决定返回工具调用方。"""
        record = await self._store.create(
            run_id=request.run_id,
            conversation_id=request.conversation_id,
            tool_name=request.tool_name,
            tool_call_id=request.tool_call_id,
            arguments=request.arguments,
            reason=request.description,
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalResponse] = loop.create_future()
        self._pending[record.id] = future
        await self._notify("approval.required", {"approval": record})
        try:
            return await future
        finally:
            self._pending.pop(record.id, None)

    # 函数说明：WebApprovalGate.approve
    # 用途：确认待决审批并唤醒等待中的调用。
    # 参数：
    #   approval_id：审批请求标识，类型 `str`。
    # 返回：类型 `ApprovalRecord`；返回 `record`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.resolve` →
    # `self._settle` → `ApprovalResponse` → `self._notify`。
    async def approve(self, approval_id: str) -> ApprovalRecord:

        """确认待决审批并唤醒等待中的调用。"""
        record = await self._store.resolve(
            approval_id,
            ApprovalRequestStatus.APPROVED,
        )
        self._settle(
            approval_id,
            ApprovalResponse(decision=ApprovalDecision.APPROVED),
        )
        await self._notify("approval.resolved", {"approval": record})
        return record

    # 函数说明：WebApprovalGate.deny
    # 用途：拒绝待决审批并唤醒等待中的调用。
    # 参数：
    #   approval_id：审批请求标识，类型 `str`。
    # 返回：类型 `ApprovalRecord`；返回 `record`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.resolve` →
    # `self._settle` → `ApprovalResponse` → `self._notify`。
    async def deny(self, approval_id: str) -> ApprovalRecord:

        """拒绝待决审批并唤醒等待中的调用。"""
        record = await self._store.resolve(
            approval_id,
            ApprovalRequestStatus.DENIED,
        )
        self._settle(
            approval_id,
            ApprovalResponse(decision=ApprovalDecision.DENIED),
        )
        await self._notify("approval.resolved", {"approval": record})
        return record

    # 函数说明：WebApprovalGate._settle
    # 用途：完成WebApprovalGate，供人工审批请求与等待使用。
    # 参数：
    #   approval_id：审批请求标识，类型 `str`。
    #   response：模型、工具或服务返回的响应，类型 `ApprovalResponse`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`future.done` →
    # `future.set_result`。
    def _settle(self, approval_id: str, response: ApprovalResponse) -> None:

        future = self._pending.get(approval_id)
        if future is not None and not future.done():
            future.set_result(response)

    # 函数说明：WebApprovalGate._notify
    # 用途：通知WebApprovalGate，供人工审批请求与等待使用。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `Any`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._broadcaster`。
    async def _notify(self, method: str, params: Any) -> None:
        if self._broadcaster is not None:
            await self._broadcaster(method, params)


__all__ = ["WebApprovalGate"]
