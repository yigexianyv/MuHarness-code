


from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from app.runtime.agent.events import AgentEvent, AgentEventHandler, AgentEventType

if TYPE_CHECKING:
    from .connection import RpcConnection

logger = logging.getLogger("muharness.server.rpc.hub")


class RpcHub:

    # 函数说明：RpcHub.__init__
    # 用途：初始化 RpcHub；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Lock`。
    # 副作用与资源：
    #   更新对象字段：`self._connections`、`self._lock`。
    def __init__(self) -> None:
        self._connections: set[RpcConnection] = set()
        self._lock = asyncio.Lock()

    # 函数说明：RpcHub.register
    # 用途：注册RpcHub，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   connection：传给 `self._connections.add` 的输入，类型 `RpcConnection`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connections.add` →
    # `logger.debug`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def register(self, connection: RpcConnection) -> None:
        async with self._lock:
            self._connections.add(connection)
        logger.debug("rpc connection registered (%d)", len(self._connections))

    # 函数说明：RpcHub.unregister
    # 用途：注销RpcHub，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   connection：传给 `self._connections.discard` 的输入，类型 `RpcConnection`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connections.discard` →
    # `logger.debug`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def unregister(self, connection: RpcConnection) -> None:
        async with self._lock:
            self._connections.discard(connection)
        logger.debug("rpc connection unregistered (%d)", len(self._connections))

    # 函数说明：RpcHub.broadcast
    # 用途：向当前连接广播服务端事件，并隔离单个连接的发送失败。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `Any`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`connection.send_notification`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def broadcast(self, method: str, params: Any) -> None:

        """向当前连接广播服务端事件，并隔离单个连接的发送失败。"""
        async with self._lock:
            connections = tuple(self._connections)
        for connection in connections:
            await connection.send_notification(method, params)

    # 函数说明：RpcHub.connection_count
    # 用途：统计连接，供JSON-RPC 连接与消息分发使用。
    # 返回：类型 `int`；返回 `len(self._connections)`。
    @property
    def connection_count(self) -> int:
        return len(self._connections)


class RpcBroadcastEventHandler(AgentEventHandler):

    # 函数说明：RpcBroadcastEventHandler.__init__
    # 用途：初始化 RpcBroadcastEventHandler；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   hub：`hub`输入或配置值，类型 `RpcHub`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._hub`。
    def __init__(self, hub: RpcHub) -> None:
        self._hub = hub

    # 函数说明：RpcBroadcastEventHandler.emit
    # 用途：发出RpcBroadcastEventHandler，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._hub.broadcast` →
    # `_derived_run_status`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def emit(self, event: AgentEvent) -> None:
        await self._hub.broadcast("agent.event", event.model_dump(mode="json"))
        status = _derived_run_status(event)
        if status is not None:
            await self._hub.broadcast(
                "run.status",
                {"run_id": event.run_id, "status": status},
            )


# 函数说明：_derived_run_status
# 用途：运行状态，供JSON-RPC 连接与消息分发使用。
# 参数：
#   event：待记录或转发的事件，类型 `AgentEvent`。
# 返回：类型 `str | None`；按分支返回 `'running'`；`'completed'`；`'failed'`；`None`。
# 分支与异常：
#   当 `event.type is AgentEventType.AGENT_STARTED` 时，返回 `'running'`。
#   当 `event.type is AgentEventType.AGENT_COMPLETED` 时，返回 `'completed'`。
#   当 `event.type is AgentEventType.AGENT_FAILED` 时，返回 `'failed'`。
def _derived_run_status(event: AgentEvent) -> str | None:

    if event.type is AgentEventType.AGENT_STARTED:
        return "running"
    if event.type is AgentEventType.AGENT_COMPLETED:
        return "completed"
    if event.type is AgentEventType.AGENT_FAILED:
        return "failed"
    return None


__all__ = ["RpcBroadcastEventHandler", "RpcHub"]
