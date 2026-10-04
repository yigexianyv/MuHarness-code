
from __future__ import annotations

from typing import Any

from app.runtime.rewind.service import RewindConflict, RewindError, RewindService
from app.runtime.run import RunStatus, history_sha256

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
_CONTEXT_MESSAGES_MAX_LIMIT = 200


async def run_context_messages(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    """分页读取一次运行看过的原始消息：继承的会话历史 + 本次运行新增的消息。

    序号与摘要覆盖水位一致，面板据此展示"第 a~b 条已由摘要替代"的原文。
    """
    run_id = _require_str(params, "run_id")
    offset = params.get("offset", 0)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            "offset must be a non-negative integer",
        )
    limit = _positive_int(params, "limit", default=50)
    limit = min(limit, _CONTEXT_MESSAGES_MAX_LIMIT)
    application = ctx.application
    store = application.run_message_store
    if store is None:
        raise JsonRpcError(INVALID_STATE, "run message store is not ready")
    ref = await store.history_ref(run_id)
    if ref is None:
        raise JsonRpcError(
            RESOURCE_NOT_FOUND,
            "该运行没有原文记录（功能上线前的运行，或不属于普通会话）",
        )
    inherited_count = ref.inherited_count
    own_count = await store.count(run_id)
    total = inherited_count + own_count
    end = min(total, offset + limit)
    items: list[dict[str, Any]] = []
    if offset < inherited_count:
        if ref.conversation_id is None:
            raise JsonRpcError(
                INVALID_STATE,
                f"该运行没有关联会话，无法还原它继承的前 {inherited_count} 条消息",
            )
        history = (
            await application.conversation_store.load_messages(ref.conversation_id)
            if await application.conversation_store.get(ref.conversation_id)
            is not None
            else ()
        )
        inherited = tuple(history[:inherited_count])
        if (
            len(inherited) != inherited_count
            or history_sha256(inherited) != ref.inherited_sha256
        ):
            raise JsonRpcError(
                INVALID_STATE,
                "会话历史在该运行之后被改写，无法还原它继承的前 "
                f"{inherited_count} 条消息；本次运行新增的消息"
                f"（从第 {inherited_count + 1} 条开始）仍可查看",
            )
        items.extend(
            {"index": index, "message": inherited[index], "inherited": True}
            for index in range(offset, min(end, inherited_count))
        )
    own_start = max(offset, inherited_count) - inherited_count
    own_limit = end - inherited_count - own_start
    if own_limit > 0:
        items.extend(
            {"index": inherited_count + index, "message": message, "inherited": False}
            for index, message in await store.load(
                run_id,
                offset=own_start,
                limit=own_limit,
            )
        )
    return {
        "run_id": run_id,
        "conversation_id": ref.conversation_id,
        "inherited_count": inherited_count,
        "total": total,
        "offset": offset,
        "messages": items,
    }


_EVIDENCE_PAGE_MAX_CHARS = 12_000
_FORK_CHAIN_LIMIT = 20


async def run_context_evidence(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    """分页读取某次工具调用的完整原文（模型收到的可能是截短版本）。

    只在该运行所属会话、以及它经"重做此步"继承的来源会话里查找，
    不能凭工具调用 ID 读到无关会话的证据。
    """
    run_id = _require_str(params, "run_id")
    tool_call_id = _require_str(params, "tool_call_id")
    offset = params.get("offset", 0)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            "offset must be a non-negative integer",
        )
    limit = min(
        _positive_int(params, "limit", default=_EVIDENCE_PAGE_MAX_CHARS),
        _EVIDENCE_PAGE_MAX_CHARS,
    )
    application = ctx.application
    run = await application.run_manager.get_run(run_id)
    if run is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "run not found")
    if run.conversation_id is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "完整原文不可用：该运行没有关联会话")
    conversation_ids = [run.conversation_id]
    rewind_service = getattr(application, "rewind_service", None)
    current = run.conversation_id
    while rewind_service is not None and len(conversation_ids) < _FORK_CHAIN_LIMIT:
        fork = await rewind_service.fork_for(current)
        if fork is None or fork.source_conversation_id in conversation_ids:
            break
        current = fork.source_conversation_id
        conversation_ids.append(current)
    document = await application.evidence_store.find_for_tool_call(
        tool_call_id,
        conversation_ids=conversation_ids,
    )
    if document is None:
        raise JsonRpcError(
            RESOURCE_NOT_FOUND,
            "完整原文不可用：没有找到这次工具调用的证据记录",
        )
    content = document.content
    end = min(len(content), offset + limit)
    return {
        "run_id": run_id,
        "tool_call_id": tool_call_id,
        "tool_name": document.record.tool_name,
        "evidence_id": document.record.id,
        "total_chars": len(content),
        "offset": offset,
        "content": content[offset:end],
        "next_offset": end if end < len(content) else None,
    }


async def run_context_tool_view(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    """第 step 步请求里，模型实际收到的这次工具输出（逐字，与发给适配器的一致）。"""
    run_id = _require_str(params, "run_id")
    step = _require_step(params)
    tool_call_id = _require_str(params, "tool_call_id")
    store = getattr(ctx.application, "run_step_store", None)
    if store is None:
        raise JsonRpcError(INVALID_STATE, "run step store is not ready")
    view = await store.request_tool_view(run_id, step, tool_call_id)
    if view is None:
        raise JsonRpcError(
            RESOURCE_NOT_FOUND,
            "这一步没有记录请求里的工具视图（功能上线前的运行，或该工具不在这一步之前）",
        )
    return {"run_id": run_id, "step": step, **view}


def _rewind_service(ctx: RpcContext) -> RewindService:
    service = ctx.application.rewind_service
    if service is None:
        raise JsonRpcError(INVALID_STATE, "rewind service is not ready")
    return service


def _require_step(params: dict[str, Any]) -> int:
    step = params.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "step must be >= 1")
    return step


async def run_steps_list(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    """每一步的检查点，以及能否从这一步重做（不能时给出原因）。"""
    run_id = _require_str(params, "run_id")
    steps = await _rewind_service(ctx).list_steps(run_id)
    return {"run_id": run_id, "steps": steps}


async def run_rewind_preview(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    """只读：回到第 N 步之前会恢复、删除哪些文件，哪些操作不会回退。"""
    run_id = _require_str(params, "run_id")
    step = _require_step(params)
    try:
        preview = await _rewind_service(ctx).preview(run_id, step)
    except RewindError as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {"preview": preview}


async def run_rewind_apply(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    """恢复文件并新建分支会话；同一个 rewind_key 只生效一次。"""
    run_id = _require_str(params, "run_id")
    step = _require_step(params)
    preview_id = _require_str(params, "preview_id")
    rewind_key = _require_str(params, "rewind_key")
    correction = params.get("correction")
    if not isinstance(correction, str):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "correction must be a string")
    try:
        result = await _rewind_service(ctx).apply(
            run_id=run_id,
            step=step,
            preview_id=preview_id,
            correction=correction,
            rewind_key=rewind_key,
        )
    except RewindConflict as exc:
        raise JsonRpcError(
            INVALID_STATE, str(exc), {"reason": "preview_outdated"}
        ) from exc
    except RewindError as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {"rewind": result}


async def run_rewind_undo(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    """把工作区文件恢复到回退之前；分支会话保留。"""
    rewind_key = _require_str(params, "rewind_key")
    try:
        result = await _rewind_service(ctx).undo(rewind_key)
    except RewindError as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {"rewind": result}


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
    dispatcher.register("run.context.messages", run_context_messages)
    dispatcher.register("run.context.evidence", run_context_evidence)
    dispatcher.register("run.context.tool_view", run_context_tool_view)
    dispatcher.register("run.steps.list", run_steps_list)
    dispatcher.register("run.rewind.preview", run_rewind_preview)
    dispatcher.register("run.rewind.apply", run_rewind_apply)
    dispatcher.register("run.rewind.undo", run_rewind_undo)
