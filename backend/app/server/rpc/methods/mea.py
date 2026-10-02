"""长任务（MEA）的 RPC 方法。

状态变化通过 ``mea.status`` / ``mea.round`` / ``mea.agent_event`` 推送（见 ``app.runtime.mea.events``），
这里的方法只做参数校验和调用 MeaRunner。
"""

from __future__ import annotations

import json
from typing import Any

from app.domain.task import TaskStatus
from app.runtime.mea import (
    ExtraToolsError,
    MeaStartError,
    MeaStatus,
    optional_executor_tools,
    validate_extra_tools,
)
from app.runtime.mea.requirements import AmendmentKind

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import (
    INVALID_STATE,
    RESOURCE_NOT_FOUND,
    JsonRpcError,
    RpcErrorCode,
)

_MAX_ROUND_BUDGET = 200
_MAX_TEXT_CHARS = 8_000


# 函数说明：mea_start
# 用途：为计划启动长任务。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `conversation_id`、
# `task_id`、`extra_tools`、`original_request`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `mea`、`task`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` → `_int` → `_bool` →
# `_runner` → `application.task_store.resolve` → `validate_extra_tools`；另有 4 个调用点
# 。
# 分支与异常：
#   当 `not isinstance(raw_tools, list) or not all((isinstance(item…` 时，抛出
# `JsonRpcError(…)`。
#   当 `task is None` 时，抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'task not found')`。
#   捕获 `ExtraToolsError` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
#   当 `original_request is not None and (not isinstance(…` 时，抛出 `JsonRpcError(…)`。
#   捕获 `MeaStartError` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`。
#   捕获 `ValueError` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`。
async def mea_start(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    """为计划启动长任务。计划还在待确认时，先检查能否启动，再接受计划。"""

    conversation_id = _require_str(params, "conversation_id")
    task_id = _require_str(params, "task_id")
    round_budget = _int(params, "round_budget", default=25, minimum=1, maximum=_MAX_ROUND_BUDGET)
    accept_plan = _bool(params, "accept_plan", default=True)
    auto_approve_sandbox = _bool(params, "auto_approve_sandbox", default=True)
    raw_tools = params.get("extra_tools", [])
    if not isinstance(raw_tools, list) or not all(isinstance(item, str) for item in raw_tools):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "extra_tools must be a list of strings")

    application = ctx.application
    runner = _runner(ctx)
    task = await application.task_store.resolve(task_id, owner_conversation_id=conversation_id)
    if task is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "task not found")
    try:
        extra_tools = validate_extra_tools(raw_tools, application.tool_registry)
    except ExtraToolsError as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    original_request = params.get("original_request")
    if original_request is not None and (
        not isinstance(original_request, str) or not original_request.strip()
    ):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "original_request must be a non-empty string")
    if original_request is None:
        original_request = await _original_request(application, task)

    try:
        if task.status is TaskStatus.PENDING:
            if not accept_plan:
                raise MeaStartError("计划还没有被接受")
            await runner.preflight(task, original_request)
            task = await application.task_store.plan_accept(task.id)
        mea = await runner.start(
            task_id=task.id,
            conversation_id=conversation_id,
            original_request=original_request,
            round_budget=round_budget,
            extra_tools=extra_tools,
            auto_approve_sandbox=auto_approve_sandbox,
        )
    except MeaStartError as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    except ValueError as exc:  # plan_accept 的状态冲突
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {"mea": mea, "task": task}


# 函数说明：mea_get
# 用途：获取`mea`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `mea`、`rounds`、`requirements`、`task`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_mea` → `store.rounds` →
# `store.requirements` → `json.loads` → `requirements.to_json`。
async def mea_get(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    mea = await _require_mea(params, ctx)
    store = ctx.application.mea_store
    rounds = await store.rounds(mea.id)
    requirements = await store.requirements(mea.id)
    task = await ctx.application.task_store.get(mea.task_id)
    return {
        "mea": mea,
        "rounds": rounds,
        "requirements": json.loads(requirements.to_json()),
        "task": task,
    }


# 函数说明：mea_list
# 用途：列出长任务。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `conversation_id`、`status`
# 。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `meas`、`count`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_int` → `MeaStatus` →
# `ctx.application.mea_store.list_runs`。
# 分支与异常：
#   当 `conversation_id is not None and (not isinstance(…` 时，抛出 `JsonRpcError(…)`。
#   捕获 `ValueError` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, f'invalid status: {status}')`。
async def mea_list(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    """列出长任务。不传 conversation_id 时列出所有会话的（“待处理”和“账本”页面用）。"""

    conversation_id = params.get("conversation_id")
    if conversation_id is not None and (not isinstance(conversation_id, str) or not conversation_id):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "conversation_id must be a non-empty string")
    limit = _int(params, "limit", default=20, minimum=1, maximum=100)
    status = params.get("status")
    if status is not None:
        try:
            status = MeaStatus(status)
        except ValueError as exc:
            raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, f"invalid status: {status}") from exc
    meas = await ctx.application.mea_store.list_runs(
        conversation_id=conversation_id, status=status, limit=limit
    )
    return {"meas": meas, "count": len(meas)}


# 函数说明：mea_tools
# 用途：启动长任务时可以勾选授权给 Executor 的额外工具。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `tools`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`optional_executor_tools`。
async def mea_tools(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    """启动长任务时可以勾选授权给 Executor 的额外工具。"""

    return {"tools": optional_executor_tools(ctx.application.tool_registry)}


# 函数说明：mea_answer
# 用途：处理回复`mea`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；返回 `_amend_result(result)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_mea` → `_text` →
# `_runner(ctx).answer` → `_runner` → `_amend_result`。
async def mea_answer(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    mea = await _require_mea(params, ctx)
    text = _text(params, "text")
    result = await _runner(ctx).answer(mea.id, text)
    return _amend_result(result)


# 函数说明：mea_note
# 用途：用户补充指令。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `kind`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；按分支返回 `_amend_result(result)`；
# `{'accepted': False, 'reason': 'mea_finalized'}`；`{'accepted': True, 'mea': updated}`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_mea` → `_text` → `_bool` →
# `_runner` → `runner.add_amendment` → `_amend_result`；另有 2 个调用点。
# 分支与异常：
#   当 `kind not in ('persistent', 'once')` 时，抛出 `JsonRpcError(…)`。
#   `kind == 'persistent'` 分支在完成前置处理后返回 `_amend_result(result)`。
#   捕获 `ValueError` 后，返回 `{'accepted': False, 'reason': 'mea_finalized'}`。
async def mea_note(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    """用户补充指令。persistent 生成新的要求版本；once 只注入下一轮 Manager。

    长任务已进入终结或已结束时返回 ``accepted=false, reason=mea_finalized``，
    前端应把这条消息作为普通新消息发送。
    """

    mea = await _require_mea(params, ctx)
    text = _text(params, "text")
    kind = params.get("kind", "persistent")
    if kind not in ("persistent", "once"):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "kind must be 'persistent' or 'once'")
    immediate = _bool(params, "immediate", default=False)
    runner = _runner(ctx)
    if kind == "persistent":
        result = await runner.add_amendment(
            mea.id, text, kind=AmendmentKind.NOTE, immediate=immediate
        )
        return _amend_result(result)
    try:
        updated = await runner.add_once_note(mea.id, text)
    except ValueError:
        return {"accepted": False, "reason": "mea_finalized"}
    if immediate:
        await runner.interrupt_child(mea.id)
    return {"accepted": True, "mea": updated}


# 函数说明：mea_pause
# 用途：暂停`mea`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `mea`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_mea` → `_runner(ctx).pause`
# → `_runner`。
async def mea_pause(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    mea = await _require_mea(params, ctx)
    return {"mea": await _runner(ctx).pause(mea.id)}


# 函数说明：mea_resume
# 用途：恢复`mea`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `mea`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_mea` → `_int` →
# `_runner(ctx).resume` → `_runner`。
# 分支与异常：
#   当 `mea.status not in (MeaStatus.PAUSED, MeaStatus.WAITING_USER…` 时，抛出
# `JsonRpcError(…)`。
async def mea_resume(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    mea = await _require_mea(params, ctx)
    extra_rounds = _int(params, "extra_rounds", default=0, minimum=0, maximum=_MAX_ROUND_BUDGET)
    if mea.status not in (MeaStatus.PAUSED, MeaStatus.WAITING_USER, MeaStatus.RUNNING):
        raise JsonRpcError(INVALID_STATE, f"长任务处于 {mea.status.value}，不能继续")
    return {"mea": await _runner(ctx).resume(mea.id, extra_rounds=extra_rounds)}


# 函数说明：mea_cancel
# 用途：取消`mea`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `mea`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_mea` → `_runner(ctx).cancel`
#  → `_runner`。
# 分支与异常：
#   当 `mea.status is MeaStatus.FINALIZING` 时，抛出
# `JsonRpcError(INVALID_STATE, '长任务正在收尾，不能取消')`。
async def mea_cancel(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    mea = await _require_mea(params, ctx)
    if mea.status is MeaStatus.FINALIZING:
        raise JsonRpcError(INVALID_STATE, "长任务正在收尾，不能取消")
    return {"mea": await _runner(ctx).cancel(mea.id)}


# ---------------------------------------------------------------------- helpers


# 函数说明：_runner
# 用途：处理JSON-RPC 请求处理中的 `_runner` 数据；结果及边界条件见下方说明。
# 参数：
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `Any`；返回 `runner`。
# 分支与异常：
#   当 `runner is None` 时，抛出 `RuntimeError('mea runner unavailable')`。
def _runner(ctx: RpcContext) -> Any:
    runner = getattr(ctx.application, "mea_runner", None)
    if runner is None:
        raise RuntimeError("mea runner unavailable")
    return runner


# 函数说明：_require_mea
# 用途：获取并校验必需的`mea`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `mea_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `Any`；返回 `mea`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str`。
# 分支与异常：
#   当 `mea is None` 时，抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'mea not found')`。
async def _require_mea(params: dict[str, Any], ctx: RpcContext) -> Any:
    mea_id = _require_str(params, "mea_id")
    mea = await ctx.application.mea_store.get(mea_id)
    if mea is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "mea not found")
    return mea


# 函数说明：_original_request
# 用途：没有显式传入时，用创建这个计划的那次运行的用户消息。
# 参数：
#   application：已装配的应用依赖，类型 `Any`。
#   task：当前任务记录，类型 `Any`。
# 返回：类型 `str`；按分支返回 `message`；`task.goal or task.title`。
# 分支与异常：
#   当 `isinstance(message, str) and message.strip()` 时，返回 `message`。
async def _original_request(application: Any, task: Any) -> str:
    """没有显式传入时，用创建这个计划的那次运行的用户消息。"""

    run_store = getattr(application, "run_store", None)
    if run_store is not None:
        for run_id in task.run_ids:
            run = await run_store.get(run_id)
            message = getattr(run, "user_message", None) if run is not None else None
            if isinstance(message, str) and message.strip():
                return message
    return task.goal or task.title


# 函数说明：_amend_result
# 用途：修订结果，供JSON-RPC 请求处理使用。
# 参数：
#   result：上一步计算或执行得到的结果，类型 `Any`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `accepted`、`reason`、`revision`、
# `amendment_id`。
def _amend_result(result: Any) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "reason": result.reason,
        "revision": result.revision,
        "amendment_id": result.amendment_id,
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


# 函数说明：_text
# 用途：在JSON-RPC 请求处理中处理 `_text`，通过 `_require_str(params, key).strip` 完成首
# 个内部处理步骤。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
# 返回：类型 `str`；返回 `value`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str`。
# 分支与异常：
#   当 `not value` 时，抛出 `JsonRpcError(…)`。
#   当 `len(value) > _MAX_TEXT_CHARS` 时，抛出 `JsonRpcError(…)`。
def _text(params: dict[str, Any], key: str) -> str:
    value = _require_str(params, key).strip()
    if not value:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, f"{key} must be non-empty")
    if len(value) > _MAX_TEXT_CHARS:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, f"{key} exceeds {_MAX_TEXT_CHARS} characters")
    return value


# 函数说明：_int
# 用途：在JSON-RPC 请求处理中处理 `_int`，通过 `params.get` 完成首个内部处理步骤。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
#   default：未提供有效值时使用的默认值，类型 `int`。
#   minimum：`minimum`输入或配置值，类型 `int`。
#   maximum：`maximum`输入或配置值，类型 `int`。
# 返回：类型 `int`；返回 `value`。
# 分支与异常：
#   当 `not isinstance(value, int) or isinstance(value, bool) or (…` 时，抛出
# `JsonRpcError(…)`。
def _int(params: dict[str, Any], key: str, *, default: int, minimum: int, maximum: int) -> int:
    value = params.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            f"{key} must be an integer between {minimum} and {maximum}",
        )
    return value


# 函数说明：_bool
# 用途：在JSON-RPC 请求处理中处理 `_bool`，通过 `params.get` 完成首个内部处理步骤。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
#   default：未提供有效值时使用的默认值，类型 `bool`。
# 返回：类型 `bool`；返回 `value`。
# 分支与异常：
#   当 `not isinstance(value, bool)` 时，抛出 `JsonRpcError(…)`。
def _bool(params: dict[str, Any], key: str, *, default: bool) -> bool:
    value = params.get(key, default)
    if not isinstance(value, bool):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, f"{key} must be a boolean")
    return value


# 函数说明：register
# 用途：注册JSON-RPC 请求处理，供JSON-RPC 请求处理使用。
# 参数：
#   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dispatcher.register`。
def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("mea.start", mea_start)
    dispatcher.register("mea.get", mea_get)
    dispatcher.register("mea.list", mea_list)
    dispatcher.register("mea.tools", mea_tools)
    dispatcher.register("mea.answer", mea_answer)
    dispatcher.register("mea.note", mea_note)
    dispatcher.register("mea.pause", mea_pause)
    dispatcher.register("mea.resume", mea_resume)
    dispatcher.register("mea.cancel", mea_cancel)
