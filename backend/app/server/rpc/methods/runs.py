
from __future__ import annotations

from typing import Any

from app.runtime.run import RunStatus

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import (
    INVALID_STATE,
    RESOURCE_NOT_FOUND,
    JsonRpcError,
    RpcErrorCode,
)


# 函数说明：run_list
# 用途：运行列表，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `conversation_id`、`status`
# 。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `runs`、`count`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RunStatus` → `_positive_int` →
# `ctx.application.run_manager.list_runs`。
# 分支与异常：
#   当 `conversation_id is not None and (not isinstance(…` 时，抛出 `JsonRpcError(…)`。
#   捕获 `ValueError` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, f'invalid status: {status}')`。
async def run_list(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    conversation_id = params.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            "conversation_id must be a string",
        )
    status = params.get("status")
    if status is not None:
        try:
            status = RunStatus(status)
        except ValueError as exc:
            raise JsonRpcError(
                RpcErrorCode.INVALID_PARAMS,
                f"invalid status: {status}",
            ) from exc
    limit = _positive_int(params, "limit", default=50)
    runs = await ctx.application.run_manager.list_runs(
        conversation_id=conversation_id,
        status=status,
        limit=limit,
    )
    return {"runs": runs, "count": len(runs)}


# 函数说明：run_get
# 用途：运行`get`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `run_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `run`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `ctx.application.run_manager.get_run`。
# 分支与异常：
#   当 `run is None` 时，抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'run not found')`。
async def run_get(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    run_id = _require_str(params, "run_id")
    run = await ctx.application.run_manager.get_run(run_id)
    if run is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "run not found")
    return {"run": run}


# 函数说明：run_cancel
# 用途：运行`cancel`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `run_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `run`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `application.run_manager.get_run` → `application.run_manager.cancel` →
# `ctx.connection.hub.broadcast`。
# 分支与异常：
#   当 `run is None` 时，抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'run not found')`。
#   捕获 `ValueError` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`。
# 副作用与资源：
#   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
async def run_cancel(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    run_id = _require_str(params, "run_id")
    application = ctx.application
    run = await application.run_manager.get_run(run_id)
    if run is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "run not found")
    try:
        updated = await application.run_manager.cancel(run.id)
    except ValueError as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    await ctx.connection.hub.broadcast(
        "run.status",
        {"run_id": run_id, "status": updated.status.value},
    )
    return {"run": updated}


# 函数说明：run_interrupt
# 用途：运行`interrupt`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `run_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `run`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `application.run_manager.get_run` → `application.run_manager.interrupt` →
# `ctx.connection.hub.broadcast`。
# 分支与异常：
#   当 `run is None` 时，抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'run not found')`。
#   捕获 `ValueError` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`。
# 副作用与资源：
#   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
async def run_interrupt(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:

    run_id = _require_str(params, "run_id")
    application = ctx.application
    run = await application.run_manager.get_run(run_id)
    if run is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "run not found")
    try:
        updated = await application.run_manager.interrupt(run.id)
    except ValueError as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    await ctx.connection.hub.broadcast(
        "run.status",
        {"run_id": run_id, "status": updated.status.value},
    )
    return {"run": updated}


# 函数说明：run_recover
# 用途：运行`recover`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `run_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `recovered_from_run_id`、`run`、`result`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `application.run_manager.get_run` → `(run.source or '').startswith` →
# `application.conversation_service.recover`。
# 分支与异常：
#   当 `run is None` 时，抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'run not found')`。
#   当 `(run.source or '').startswith('mea:')` 时，抛出
# `JsonRpcError(INVALID_STATE, '长任务的子 Run 由长任务自己恢复，请在长任务面板点继续')`
# 。
#   捕获 `(KeyError, ValueError)` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`
# 。
async def run_recover(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    run_id = _require_str(params, "run_id")
    application = ctx.application
    run = await application.run_manager.get_run(run_id)
    if run is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "run not found")
    if (run.source or "").startswith("mea:"):
        # 长任务的子 Run 不能按普通对话恢复：中断后由长任务先做核查审计再决定
        raise JsonRpcError(
            INVALID_STATE,
            "长任务的子 Run 由长任务自己恢复，请在长任务面板点继续",
        )

    try:
        dispatch = await application.conversation_service.recover(run_id)
    except (KeyError, ValueError) as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {
        "recovered_from_run_id": run.id,
        "run": dispatch.run,
        "result": dispatch.result,
    }


# 函数说明：_require_str
# 用途：获取并校验必需的`str`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
# 返回：类型 `str`；返回 `value`。
# 分支与异常：
#   当 `not isinstance(value, str) or not value` 时，抛出 `JsonRpcError(…)`。
def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, f"{key} is required")
    return value


# 函数说明：_positive_int
# 用途：在JSON-RPC 请求处理中处理 `_positive_int`，通过 `params.get` 完成首个内部处理步
# 骤。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
#   default：未提供有效值时使用的默认值，类型 `int`。
# 返回：类型 `int`；返回 `value`。
# 分支与异常：
#   当 `not isinstance(value, int) or isinstance(value, bool) or…` 时，抛出
# `JsonRpcError(…)`。
def _positive_int(params: dict[str, Any], key: str, *, default: int) -> int:
    value = params.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            f"{key} must be a positive integer",
        )
    return value


# 函数说明：register
# 用途：注册JSON-RPC 请求处理，供JSON-RPC 请求处理使用。
# 参数：
#   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dispatcher.register`。
def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("run.list", run_list)
    dispatcher.register("run.get", run_get)
    dispatcher.register("run.cancel", run_cancel)
    dispatcher.register("run.interrupt", run_interrupt)
    dispatcher.register("run.recover", run_recover)
