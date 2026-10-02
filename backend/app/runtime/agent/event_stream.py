
from __future__ import annotations

import asyncio
from typing import Any

from .events import AgentEvent, AgentEventHandler, AgentEventType


class EventEmitter:

    # 函数说明：EventEmitter.__init__
    # 用途：初始化 EventEmitter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   handler：请求或事件处理回调，类型 `AgentEventHandler`。
    #   run_id：目标运行标识，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   sequence_offset：`sequence_offset`输入或配置值，类型 `int`；默认 `0`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `sequence_offset < 0` 时，抛出
    # `ValueError('sequence_offset cannot be negative')`。
    # 副作用与资源：
    #   更新对象字段：`self._handler`、`self._run_id`、`self._conversation_id`、
    # `self._sequence`。
    def __init__(
        self,
        *,
        handler: AgentEventHandler,
        run_id: str,
        conversation_id: str | None,
        sequence_offset: int = 0,
    ) -> None:
        if sequence_offset < 0:
            raise ValueError("sequence_offset cannot be negative")
        self._handler = handler
        self._run_id = run_id
        self._conversation_id = conversation_id
        self._sequence = sequence_offset

    # 函数说明：EventEmitter.emit
    # 用途：发出EventEmitter，供模型与工具执行循环使用。
    # 参数：
    #   event_type：事件输入或配置值，类型 `AgentEventType`。
    #   **payload：额外关键字参数，按实现处理或转交。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentEvent` →
    # `self._handler.emit`。
    # 分支与异常：
    #   捕获 `Exception` 后，返回 `None`。
    # 副作用与资源：
    #   更新对象字段：`self._sequence`。
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def emit(
        self,
        event_type: AgentEventType,
        **payload: Any,
    ) -> None:
        event = AgentEvent(
            run_id=self._run_id,
            conversation_id=self._conversation_id,
            sequence=self._sequence,
            type=event_type,
            **payload,
        )
        self._sequence += 1
        try:
            await self._handler.emit(event)
        except Exception:
            return


STREAM_FINISHED = object()


class QueueEventHandler(AgentEventHandler):

    # 函数说明：QueueEventHandler.__init__
    # 用途：初始化 QueueEventHandler；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Queue`。
    # 副作用与资源：
    #   更新对象字段：`self._queue`。
    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentEvent | object] = asyncio.Queue(maxsize=100)

    # 函数说明：QueueEventHandler.emit
    # 用途：发出QueueEventHandler，供模型与工具执行循环使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._queue.put`。
    async def emit(self, event: AgentEvent) -> None:
        await self._queue.put(event)

    # 函数说明：QueueEventHandler.finish
    # 用途：结束QueueEventHandler，供模型与工具执行循环使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._queue.put`。
    async def finish(self) -> None:
        await self._queue.put(STREAM_FINISHED)

    # 函数说明：QueueEventHandler.next
    # 用途：返回 `await self._queue.get()`，提供 QueueEventHandler 的派生值。
    # 返回：类型 `AgentEvent | object`；返回 `await self._queue.get()`。
    async def next(self) -> AgentEvent | object:
        return await self._queue.get()


__all__ = ["EventEmitter", "QueueEventHandler", "STREAM_FINISHED"]
