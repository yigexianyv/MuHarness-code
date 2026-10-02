
from __future__ import annotations

from typing import Any

from app.records.trace import summarize_run_usage

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import RESOURCE_NOT_FOUND, JsonRpcError, RpcErrorCode


# 函数说明：trace_get
# 用途：获取执行轨迹，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `run_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `run`、`events`、`usage`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`application.trace_store.load_events`
# → `summarize_run_usage`。
# 分支与异常：
#   当 `not isinstance(run_id, str) or not run_id` 时，抛出 `JsonRpcError(…)`。
#   当 `trace is None` 时，抛出
# `JsonRpcError(RESOURCE_NOT_FOUND, 'run trace not found')`。
async def trace_get(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    run_id = params.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "run_id is required")
    application = ctx.application
    trace = await application.trace_store.get(run_id)
    if trace is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "run trace not found")
    events = await application.trace_store.load_events(run_id)
    return {
        "run": trace,
        "events": events,
        "usage": summarize_run_usage(events),
    }


# 函数说明：register
# 用途：注册JSON-RPC 请求处理，供JSON-RPC 请求处理使用。
# 参数：
#   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dispatcher.register`。
def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("trace.get", trace_get)
