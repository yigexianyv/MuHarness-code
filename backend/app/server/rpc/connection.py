


from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .dispatcher import RpcContext, RpcDispatcher
from .protocol import JSONRPC_VERSION, JsonRpcError, parse_message

if TYPE_CHECKING:
    from app.application import Application

    from .hub import RpcHub

logger = logging.getLogger("muharness.server.rpc.connection")


# 函数说明：_to_jsonable
# 用途：在JSON-RPC 连接与消息分发中处理 `_to_jsonable`，通过 `value.model_dump` 完成首个
# 内部处理步骤。
# 参数：
#   value：待校验、规范化或转换的值，类型 `Any`。
# 返回：类型 `Any`；按分支返回 `value.model_dump(mode='json')`；
# `[_to_jsonable(item) for item in value]`；
# `{key: _to_jsonable(item) for key, item in value.items()}`；`value`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_to_jsonable`。
# 分支与异常：
#   当 `isinstance(value, BaseModel)` 时，返回 `value.model_dump(mode='json')`。
#   当 `isinstance(value, tuple)` 时，返回 `[_to_jsonable(item) for item in value]`。
#   当 `isinstance(value, list)` 时，返回 `[_to_jsonable(item) for item in value]`。
#   当 `isinstance(value, dict)` 时，返回
# `{key: _to_jsonable(item) for key, item in value.items()}`。
def _to_jsonable(value: Any) -> Any:

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


class RpcConnection:

    # 函数说明：RpcConnection.__init__
    # 用途：初始化 RpcConnection；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   websocket：当前 WebSocket 连接，类型 `WebSocket`。
    #   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
    #   application：已装配的应用依赖，类型 `Application`。
    #   hub：`hub`输入或配置值，类型 `RpcHub`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Lock`。
    # 副作用与资源：
    #   更新对象字段：`self._websocket`、`self._dispatcher`、`self._application`、
    # `self._hub`、`self._send_lock`、`self._tasks`、`self._closed`。
    def __init__(
        self,
        websocket: WebSocket,
        dispatcher: RpcDispatcher,
        application: Application,
        hub: RpcHub,
    ) -> None:
        self._websocket = websocket
        self._dispatcher = dispatcher
        self._application = application
        self._hub = hub
        self._send_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    # 函数说明：RpcConnection.application
    # 用途：返回 `self._application`，提供 RpcConnection 的派生值。
    # 返回：类型 `Application`；返回 `self._application`。
    @property
    def application(self) -> Application:
        return self._application

    # 函数说明：RpcConnection.hub
    # 用途：返回 `self._hub`，提供 RpcConnection 的派生值。
    # 返回：类型 `RpcHub`；返回 `self._hub`。
    @property
    def hub(self) -> RpcHub:
        return self._hub

    # 函数说明：RpcConnection.is_closed
    # 用途：判断`closed`是否满足当前实现的条件。
    # 返回：类型 `bool`；返回 `self._closed`。
    @property
    def is_closed(self) -> bool:
        return self._closed


    # 函数说明：RpcConnection.run
    # 用途：接收 WebSocket 消息并分发 RPC 请求，连接结束时清理订阅。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._websocket.receive_text` →
    # `asyncio.create_task` → `self._handle_message` → `self._tasks.add` →
    # `task.add_done_callback`。
    # 分支与异常：
    #   捕获 `WebSocketDisconnect` 后，忽略该异常并继续当前流程。
    # 副作用与资源：
    #   更新对象字段：`self._closed`。
    #   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
    async def run(self) -> None:

        """接收 WebSocket 消息并分发 RPC 请求，连接结束时清理订阅。"""
        try:
            while not self._closed:
                text = await self._websocket.receive_text()
                task = asyncio.create_task(self._handle_message(text))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        except WebSocketDisconnect:
            pass
        finally:
            self._closed = True

    # 函数说明：RpcConnection._handle_message
    # 用途：处理消息，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_message` →
    # `self.send_error` → `RpcContext` → `self._dispatch_notification` →
    # `self._dispatcher.dispatch` → `self.send_response`。
    # 分支与异常：
    #   `parsed.error is not None` 分支在完成前置处理后返回 `None`。
    #   `parsed.notification is not None` 分支在完成前置处理后返回 `None`。
    #   捕获 `JsonRpcError` 后，返回 `None`。
    async def _handle_message(self, text: str) -> None:
        parsed = parse_message(text)
        if parsed.error is not None:
            await self.send_error(parsed.id, parsed.error)
            return
        context = RpcContext(self._application, self)
        if parsed.notification is not None:
            await self._dispatch_notification(
                parsed.notification.method,
                parsed.notification.params,
                context,
            )
            return
        assert parsed.request is not None
        request = parsed.request
        try:
            result = await self._dispatcher.dispatch(
                request.method,
                request.params,
                context,
            )
        except JsonRpcError as exc:
            await self.send_error(request.id, exc)
            return
        await self.send_response(request.id, result)

    # 函数说明：RpcConnection._dispatch_notification
    # 用途：分发`notification`，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
    #   context：本次操作的上下文对象，类型 `RpcContext`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._dispatcher.dispatch` →
    # `logger.exception`。
    # 分支与异常：
    #   捕获 `JsonRpcError` 后，忽略该异常并继续当前流程。
    #   捕获 `Exception` 后，执行异常处理调用 `logger.exception`。
    async def _dispatch_notification(
        self,
        method: str,
        params: dict[str, Any],
        context: RpcContext,
    ) -> None:
        try:
            await self._dispatcher.dispatch(method, params, context)
        except JsonRpcError:
            pass
        except Exception:
            logger.exception("rpc notification %r failed", method)


    # 函数说明：RpcConnection._send
    # 用途：发送RpcConnection，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   payload：传输或持久化载荷，类型 `dict[str, Any]`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps` → `_to_jsonable` →
    # `self._websocket.send_text`。
    # 资源/并发边界：`self._send_lock`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `self._closed` 时，返回 `None`。
    #   捕获 `Exception` 后，执行异常分支中的状态更新；具体更新见实现。
    # 副作用与资源：
    #   更新对象字段：`self._closed`。
    async def _send(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        text = json.dumps(_to_jsonable(payload), ensure_ascii=False)
        async with self._send_lock:
            if self._closed:
                return
            try:
                await self._websocket.send_text(text)
            except Exception:
                self._closed = True

    # 函数说明：RpcConnection.send_response
    # 用途：发送响应，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   request_id：请求标识，类型 `str | int`。
    #   result：上一步计算或执行得到的结果，类型 `Any`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._send`。
    async def send_response(self, request_id: str | int, result: Any) -> None:
        await self._send(
            {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}
        )

    # 函数说明：RpcConnection.send_error
    # 用途：发送错误，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   request_id：请求标识，类型 `str | int | None`。
    #   error：异常或错误信息，类型 `JsonRpcError`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._send` → `error.to_body`。
    async def send_error(
        self,
        request_id: str | int | None,
        error: JsonRpcError,
    ) -> None:
        await self._send(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "error": error.to_body(),
            }
        )

    # 函数说明：RpcConnection.send_notification
    # 用途：发送`notification`，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `Any`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._send` → `_to_jsonable`。
    async def send_notification(self, method: str, params: Any) -> None:
        await self._send(
            {
                "jsonrpc": JSONRPC_VERSION,
                "method": method,
                "params": _to_jsonable(params),
            }
        )


__all__ = ["RpcConnection"]
