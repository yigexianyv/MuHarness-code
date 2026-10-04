


from __future__ import annotations

from typing import Any

from app.application import title_from_content
from app.domain.conversation import (
    ConstraintsRevisionConflict,
    ConversationSource,
    TriggerContext,
)
from app.models.types import AgentMode

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import INVALID_STATE, RESOURCE_NOT_FOUND, JsonRpcError, RpcErrorCode


# 函数说明：conversation_list
# 用途：列出会话，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `conversations`、`count`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_positive_int` →
# `ctx.application.conversation_store.list`。
async def conversation_list(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    limit = _positive_int(params, "limit", default=50)
    conversations = await ctx.application.conversation_store.list(limit=limit)
    return {"conversations": conversations, "count": len(conversations)}


# 函数说明：conversation_get
# 用途：获取会话，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `conversation_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `conversation`、`messages`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `application.conversation_store.load_messages`。
# 分支与异常：
#   当 `conversation is None` 时，抛出
# `JsonRpcError(RESOURCE_NOT_FOUND, 'conversation not found')`。
async def conversation_get(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    conversation_id = _require_str(params, "conversation_id")
    application = ctx.application
    conversation = await application.conversation_store.get(conversation_id)
    if conversation is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "conversation not found")
    messages = await application.conversation_store.load_messages(conversation_id)
    return {"conversation": conversation, "messages": messages}


# 函数说明：conversation_create
# 用途：创建会话，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `title`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `conversation`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `ctx.application.conversation_store.create`。
# 分支与异常：
#   当 `title is not None and (not isinstance(title, str) or not…` 时，抛出
# `JsonRpcError(…)`。
async def conversation_create(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    title = params.get("title")
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "title must be a string")
    conversation = await ctx.application.conversation_store.create(
        title=title or "新会话"
    )
    return {"conversation": conversation}


# 函数说明：conversation_send
# 用途：校验发送参数，提交会话运行并返回 RPC 约定的结果字段。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `conversation_id`、
# `content`、`mode`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `conversation_id`、`run`、`result`、
# `content`、`plan_task_id`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` → `AgentMode` →
# `mea_runner.busy_run` → `application.conversation_service.dispatch` → `TriggerContext`
#  → `application.conversation_store.rename`；另有 1 个调用点。
# 分支与异常：
#   当 `not content.strip()` 时，抛出 `JsonRpcError(…)`。
#   捕获 `ValueError` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, f'invalid mode: {raw_mode}')`。
#   当 `mode not in (AgentMode.NORMAL, AgentMode.PLAN)` 时，抛出 `JsonRpcError(…)`。
#   当 `conversation is None` 时，抛出
# `JsonRpcError(RESOURCE_NOT_FOUND, 'conversation not found')`。
async def conversation_send(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    """校验发送参数，提交会话运行并返回 RPC 约定的结果字段。"""
    conversation_id = _require_str(params, "conversation_id")
    content = _require_str(params, "content")
    if not content.strip():
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "content must be non-empty")

    mode = AgentMode.NORMAL
    raw_mode = params.get("mode")
    if raw_mode is not None:
        try:
            mode = AgentMode(raw_mode)
        except ValueError as exc:
            raise JsonRpcError(
                RpcErrorCode.INVALID_PARAMS,
                f"invalid mode: {raw_mode}",
            ) from exc
        # manage / execute / audit 只由长任务内部使用，不接受前端直接指定
        if mode not in (AgentMode.NORMAL, AgentMode.PLAN):
            raise JsonRpcError(
                RpcErrorCode.INVALID_PARAMS,
                f"invalid mode: {raw_mode}",
            )

    application = ctx.application
    conversation = await application.conversation_store.get(conversation_id)
    if conversation is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "conversation not found")
    mea_runner = getattr(application, "mea_runner", None)
    if mea_runner is not None:
        # 长任务推进期间，补充走 mea.note / mea.answer；普通消息会和长任务并发改写同一个 Task
        busy = await mea_runner.busy_run(conversation_id)
        if busy is not None:
            raise JsonRpcError(
                INVALID_STATE,
                "mea_running: 这个会话有正在推进的长任务，请用补充指令或先暂停",
                {"reason": "mea_running", "mea_id": busy.id, "status": busy.status.value},
            )

    dispatch = await application.conversation_service.dispatch(
        conversation_id=conversation_id,
        content=content,
        trigger=TriggerContext(source=ConversationSource.MANUAL),
        mode=mode,
    )
    if conversation.title == "新会话":
        await application.conversation_store.rename(
            conversation_id,
            title_from_content(content),
        )
    return {
        "conversation_id": conversation_id,
        "run": dispatch.run,
        "result": dispatch.result,
        "content": dispatch.result.content,
        "plan_task_id": dispatch.result.plan_task_id,
    }


# 函数说明：conversation_rename
# 用途：重命名会话，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `conversation_id`、`title`
# 。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `conversation`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `ctx.application.conversation_store.rename`。
# 分支与异常：
#   捕获 `KeyError` 后，转换或抛出 `JsonRpcError(RESOURCE_NOT_FOUND, str(exc))`。
async def conversation_rename(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    conversation_id = _require_str(params, "conversation_id")
    title = _require_str(params, "title")
    try:
        conversation = await ctx.application.conversation_store.rename(
            conversation_id, title
        )
    except KeyError as exc:
        raise JsonRpcError(RESOURCE_NOT_FOUND, str(exc)) from exc
    return {"conversation": conversation}


# 函数说明：conversation_delete
# 用途：删除会话，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `conversation_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；返回 `result.model_dump(mode='json')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` → `lifecycle.delete`。
# 分支与异常：
#   当 `lifecycle is None` 时，抛出 `RuntimeError('conversation lifecycle unavailable')`
# 。
#   当 `result is None` 时，抛出
# `JsonRpcError(RESOURCE_NOT_FOUND, 'conversation not found')`。
async def conversation_delete(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    conversation_id = _require_str(params, "conversation_id")
    lifecycle = ctx.application.conversation_lifecycle
    if lifecycle is None:
        raise RuntimeError("conversation lifecycle unavailable")
    result = await lifecycle.delete(conversation_id)
    if result is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "conversation not found")
    return result.model_dump(mode="json")


# 函数说明：_require_str
# 用途：获取并校验必需的`str`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
# 返回：类型 `str`；返回 `value`。
# 分支与异常：
#   当 `not isinstance(value, str) or not value` 时，抛出 `JsonRpcError(…)`。
async def conversation_constraints_get(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    conversation_id = _require_str(params, "conversation_id")
    try:
        constraints = await ctx.application.conversation_store.get_constraints(
            conversation_id
        )
    except KeyError as exc:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "conversation not found") from exc
    return {"constraints": constraints}


async def conversation_constraints_set(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    """保存"必须记住的事项"；运行中保存会在下一次模型请求生效。"""
    conversation_id = _require_str(params, "conversation_id")
    text = params.get("text")
    if not isinstance(text, str):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "text must be a string")
    expected_revision = params.get("expected_revision")
    if expected_revision is not None and (
        not isinstance(expected_revision, int)
        or isinstance(expected_revision, bool)
        or expected_revision < 0
    ):
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            "expected_revision must be a non-negative integer",
        )
    try:
        constraints = await ctx.application.conversation_store.set_constraints(
            conversation_id,
            text,
            expected_revision=expected_revision,
        )
    except KeyError as exc:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "conversation not found") from exc
    except ConstraintsRevisionConflict as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    except ValueError as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"constraints": constraints}


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
    dispatcher.register("conversation.list", conversation_list)
    dispatcher.register("conversation.get", conversation_get)
    dispatcher.register("conversation.create", conversation_create)
    dispatcher.register("conversation.send", conversation_send)
    dispatcher.register("conversation.rename", conversation_rename)
    dispatcher.register("conversation.delete", conversation_delete)
    dispatcher.register("conversation.constraints.get", conversation_constraints_get)
    dispatcher.register("conversation.constraints.set", conversation_constraints_set)
