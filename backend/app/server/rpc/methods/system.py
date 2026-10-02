
from __future__ import annotations

import asyncio
from typing import Any

from app.server.version import __version__

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import INVALID_STATE, JsonRpcError


# 函数说明：system_info
# 用途：处理JSON-RPC 请求处理中的 `system_info` 数据；结果及边界条件见下方说明。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `status`、`provider`、`model`、`version`、
# `database`。
async def system_info(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:

    application = ctx.application
    return {
        "status": "ok",
        "provider": application.provider,
        "model": application.model,
        "version": __version__,
        "database": str(application.database),
    }


# 函数说明：system_restart
# 用途：在JSON-RPC 请求处理中处理 `system_restart`，通过
# `asyncio.get_running_loop().call_later` 完成首个内部处理步骤。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `accepted`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `asyncio.get_running_loop().call_later` → `asyncio.get_running_loop`。
# 分支与异常：
#   当 `active_run_ids` 时，抛出 `JsonRpcError(…)`。
#   当 `callback is None` 时，抛出
# `JsonRpcError(INVALID_STATE, '当前 Host 启动方式不支持应用内重启')`。
async def system_restart(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:

    application = ctx.application
    active_run_ids = application.run_manager.active_run_ids
    if active_run_ids:
        raise JsonRpcError(
            INVALID_STATE,
            "存在正在执行的 Run，暂时不能重启 Host",
            {"active_run_ids": list(active_run_ids)},
        )
    callback = application.host_restart_callback
    if callback is None:
        raise JsonRpcError(
            INVALID_STATE,
            "当前 Host 启动方式不支持应用内重启",
        )
    asyncio.get_running_loop().call_later(0.25, callback)
    return {"accepted": True}


# 函数说明：register
# 用途：注册JSON-RPC 请求处理，供JSON-RPC 请求处理使用。
# 参数：
#   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dispatcher.register`。
def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("system.info", system_info)
    dispatcher.register("system.restart", system_restart)
