
from __future__ import annotations

from typing import Any

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import (
    INVALID_STATE,
    RESOURCE_NOT_FOUND,
    JsonRpcError,
    RpcErrorCode,
)


# 函数说明：task_get
# 用途：校验任务标识并查找任务，未找到时转换为 JSON-RPC 资源不存在错误。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `task_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `task`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `ctx.application.task_store.resolve`。
# 分支与异常：
#   当 `task is None` 时，抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'task not found')`。
async def task_get(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    task_id = _require_str(params, "task_id")
    task = await ctx.application.task_store.resolve(task_id)
    if task is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "task not found")
    return {"task": task}


# 函数说明：task_list
# 用途：按所属会话查询任务，拒绝布尔值或超出 1–100 的 limit，返回任务列表。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `conversation_id`、`limit`
# 。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `tasks`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `ctx.application.task_store.list`。
# 分支与异常：
#   当 `not isinstance(raw_limit, int) or isinstance(raw_limit,…` 时，抛出
# `JsonRpcError(…)`。
#   当 `raw_limit < 1 or raw_limit > 100` 时，抛出 `JsonRpcError(…)`。
async def task_list(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:

    conversation_id = _require_str(params, "conversation_id")
    raw_limit = params.get("limit", 20)
    if not isinstance(raw_limit, int) or isinstance(raw_limit, bool):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "limit must be an integer")
    if raw_limit < 1 or raw_limit > 100:
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            "limit must be between 1 and 100",
        )
    tasks = await ctx.application.task_store.list(
        limit=raw_limit,
        owner_conversation_id=conversation_id,
    )
    return {"tasks": tasks}


# 函数说明：task_plan_accept
# 用途：接受待确认任务计划，将存储层的任务缺失或非法状态映射为对应 JSON-RPC 错误。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `task_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `task`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `ctx.application.task_store.plan_accept`。
# 分支与异常：
#   捕获 `KeyError` 后，转换或抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'task not found')`
# 。
#   捕获 `ValueError` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`。
async def task_plan_accept(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    task_id = _require_str(params, "task_id")
    try:
        task = await ctx.application.task_store.plan_accept(task_id)
    except KeyError as exc:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "task not found") from exc
    except ValueError as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {"task": task}


# 函数说明：task_plan_reject
# 用途：拒绝待确认任务计划，将存储层的任务缺失或非法状态映射为对应 JSON-RPC 错误。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `task_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `task`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `ctx.application.task_store.plan_reject`。
# 分支与异常：
#   捕获 `KeyError` 后，转换或抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'task not found')`
# 。
#   捕获 `ValueError` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`。
async def task_plan_reject(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    task_id = _require_str(params, "task_id")
    try:
        task = await ctx.application.task_store.plan_reject(task_id)
    except KeyError as exc:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "task not found") from exc
    except ValueError as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {"task": task}


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


# 函数说明：register
# 用途：注册JSON-RPC 请求处理，供JSON-RPC 请求处理使用。
# 参数：
#   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dispatcher.register`。
def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("task.list", task_list)
    dispatcher.register("task.get", task_get)
    dispatcher.register("task.plan_accept", task_plan_accept)
    dispatcher.register("task.plan_reject", task_plan_reject)
