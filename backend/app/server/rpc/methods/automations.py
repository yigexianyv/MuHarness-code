
from __future__ import annotations

from typing import Any

from app.domain.automation.tools import build_schedule_and_next

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import (
    INVALID_STATE,
    RESOURCE_NOT_FOUND,
    JsonRpcError,
    RpcErrorCode,
)


# 函数说明：automation_list
# 用途：列出自动化任务，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `conversation_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `automations`、`count`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_positive_int` →
# `ctx.application.automation_scheduler.list`。
# 分支与异常：
#   当 `conversation_id is not None and (not isinstance(…` 时，抛出 `JsonRpcError(…)`。
async def automation_list(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    conversation_id = params.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            "conversation_id must be a string",
        )
    limit = _positive_int(params, "limit", default=50)
    automations = await ctx.application.automation_scheduler.list(
        conversation_id=conversation_id,
        limit=limit,
    )
    return {"automations": automations, "count": len(automations)}


# 函数说明：automation_get
# 用途：获取自动化任务，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `automation_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `automation`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str`。
# 分支与异常：
#   当 `automation is None` 时，抛出
# `JsonRpcError(RESOURCE_NOT_FOUND, 'automation not found')`。
async def automation_get(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    automation_id = _require_str(params, "automation_id")
    automation = await ctx.application.automation_scheduler.get(automation_id)
    if automation is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "automation not found")
    return {"automation": automation}


# 函数说明：automation_create
# 用途：创建自动化任务，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `title`、`prompt`、
# `conversation_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `automation`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `build_schedule_and_next` → `ctx.application.automation_scheduler.create_automation`。
# 分支与异常：
#   当 `conversation_id is not None and (not isinstance(…` 时，抛出 `JsonRpcError(…)`。
#   捕获 `ValueError` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
async def automation_create(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    title = _require_str(params, "title")
    prompt = _require_str(params, "prompt")
    conversation_id = params.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            "conversation_id must be a string",
        )
    try:
        schedule, next_run_at = build_schedule_and_next(params)
    except ValueError as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    automation = await ctx.application.automation_scheduler.create_automation(
        title=title,
        prompt=prompt,
        conversation_id=conversation_id,
        schedule=schedule,
        next_run_at=next_run_at,
    )
    return {"automation": automation}


# 函数说明：_control
# 用途：在JSON-RPC 请求处理中处理 `_control`，通过 `scheduler.get` 完成首个内部处理步骤
# 。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `automation_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
#   action：`action`输入或配置值，类型 `str`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `automation`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` → `scheduler.pause` →
# `scheduler.resume` → `scheduler.cancel`。
# 分支与异常：
#   当 `automation is None` 时，抛出
# `JsonRpcError(RESOURCE_NOT_FOUND, 'automation not found')`。
#   捕获 `ValueError` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`。
async def _control(
    params: dict[str, Any],
    ctx: RpcContext,
    action: str,
) -> dict[str, Any]:
    automation_id = _require_str(params, "automation_id")
    scheduler = ctx.application.automation_scheduler
    automation = await scheduler.get(automation_id)
    if automation is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "automation not found")
    try:
        if action == "pause":
            updated = await scheduler.pause(automation.id)
        elif action == "resume":
            updated = await scheduler.resume(automation.id)
        else:
            updated = await scheduler.cancel(automation.id)
    except ValueError as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {"automation": updated}


# 函数说明：automation_pause
# 用途：暂停自动化任务，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；返回 `await _control(params, ctx, 'pause')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_control`。
async def automation_pause(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    return await _control(params, ctx, "pause")


# 函数说明：automation_resume
# 用途：恢复自动化任务，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；返回 `await _control(params, ctx, 'resume')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_control`。
async def automation_resume(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    return await _control(params, ctx, "resume")


# 函数说明：automation_cancel
# 用途：取消自动化任务，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；返回 `await _control(params, ctx, 'cancel')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_control`。
async def automation_cancel(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    return await _control(params, ctx, "cancel")


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
    dispatcher.register("automation.list", automation_list)
    dispatcher.register("automation.get", automation_get)
    dispatcher.register("automation.create", automation_create)
    dispatcher.register("automation.pause", automation_pause)
    dispatcher.register("automation.resume", automation_resume)
    dispatcher.register("automation.cancel", automation_cancel)
