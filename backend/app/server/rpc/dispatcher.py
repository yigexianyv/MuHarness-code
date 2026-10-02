



from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .protocol import JsonRpcError, RpcErrorCode

if TYPE_CHECKING:
    from app.application import Application

    from .connection import RpcConnection

logger = logging.getLogger("muharness.server.rpc.dispatcher")

Handler = Callable[[dict[str, Any], "RpcContext"], Awaitable[Any]]


class RpcContext:

    # 函数说明：RpcContext.__init__
    # 用途：初始化 RpcContext；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   application：已装配的应用依赖，类型 `Application`。
    #   connection：连接输入或配置值，类型 `RpcConnection`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.application`、`self.connection`。
    def __init__(
        self,
        application: Application,
        connection: RpcConnection,
    ) -> None:
        self.application = application
        self.connection = connection


class RpcDispatcher:

    # 函数说明：RpcDispatcher.__init__
    # 用途：初始化 RpcDispatcher；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._methods`。
    def __init__(self) -> None:
        self._methods: dict[str, Handler] = {}

    # 函数说明：RpcDispatcher.register
    # 用途：注册 RPC 方法处理器并拒绝重复方法名。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   handler：请求或事件处理回调，类型 `Handler`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `not method or not method.strip()` 时，抛出
    # `ValueError('rpc method name cannot be empty')`。
    #   当 `method in self._methods` 时，抛出
    # `ValueError(f'duplicate rpc method: {method}')`。
    def register(self, method: str, handler: Handler) -> None:
        """注册 RPC 方法处理器并拒绝重复方法名。"""
        if not method or not method.strip():
            raise ValueError("rpc method name cannot be empty")
        if method in self._methods:
            raise ValueError(f"duplicate rpc method: {method}")
        self._methods[method] = handler

    # 函数说明：RpcDispatcher.register_many
    # 用途：注册`many`，供JSON-RPC 连接与消息分发使用。
    # 参数：
    #   handlers：`handlers`输入或配置值，类型 `dict[str, Handler]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.register`。
    def register_many(self, handlers: dict[str, Handler]) -> None:
        for method_name, handler in handlers.items():
            self.register(method_name, handler)

    # 函数说明：RpcDispatcher.has_method
    # 用途：判断`method`是否满足当前实现的条件。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    # 返回：类型 `bool`；返回 `method in self._methods`。
    def has_method(self, method: str) -> bool:
        return method in self._methods

    # 函数说明：RpcDispatcher.dispatch
    # 用途：查找并调用 RPC 方法，将未知方法与内部异常映射为协议错误。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
    #   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
    # 返回：类型 `Any`；返回 `await handler(params, ctx)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`handler` → `logger.exception`。
    # 分支与异常：
    #   当 `handler is None` 时，抛出 `JsonRpcError(…)`。
    #   捕获 `JsonRpcError` 后，重新抛出原异常。
    #   捕获 `Exception` 后，转换或抛出
    # `JsonRpcError(RpcErrorCode.INTERNAL_ERROR, 'Internal error')`。
    async def dispatch(
        self,
        method: str,
        params: dict[str, Any],
        ctx: RpcContext,
    ) -> Any:

        """查找并调用 RPC 方法，将未知方法与内部异常映射为协议错误。"""
        handler = self._methods.get(method)
        if handler is None:
            raise JsonRpcError(
                RpcErrorCode.METHOD_NOT_FOUND,
                f"Method not found: {method}",
            )
        try:
            return await handler(params, ctx)
        except JsonRpcError:
            raise
        except Exception as exc:
            logger.exception("rpc method %r failed", method)
            raise JsonRpcError(
                RpcErrorCode.INTERNAL_ERROR,
                "Internal error",
            ) from exc


# 函数说明：rpc_method
# 用途：处理JSON-RPC 连接与消息分发中的 `rpc_method` 数据；结果及边界条件见下方说明。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `Callable[[Handler], Handler]`；返回 `decorate`。
def rpc_method(name: str) -> Callable[[Handler], Handler]:

    # 函数说明：rpc_method.decorate
    # 用途：处理JSON-RPC 连接与消息分发中的 `decorate` 数据；结果及边界条件见下方说明。
    # 参数：
    #   fn：`fn`输入或配置值，类型 `Handler`。
    # 返回：类型 `Handler`；返回 `fn`。
    # 副作用与资源：
    #   更新对象字段：`fn._rpc_name`。
    # 闭包依赖：从外层读取 `name`。
    def decorate(fn: Handler) -> Handler:
        fn._rpc_name = name
        return fn

    return decorate


__all__ = ["Handler", "RpcContext", "RpcDispatcher", "rpc_method"]
