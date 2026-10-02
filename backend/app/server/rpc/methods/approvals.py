


from __future__ import annotations

from typing import Any

from app.safety.approval import ApprovalRequestStatus

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import (
    INVALID_STATE,
    RESOURCE_NOT_FOUND,
    JsonRpcError,
    RpcErrorCode,
)


# 函数说明：approval_list
# 用途：列出审批，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `status`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `approvals`、`count`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalRequestStatus` →
# `_positive_int` → `ctx.application.approval_store.list`。
# 分支与异常：
#   捕获 `ValueError` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, f'invalid status: {status}')`。
async def approval_list(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    status = params.get("status")
    if status is not None:
        try:
            status = ApprovalRequestStatus(status)
        except ValueError as exc:
            raise JsonRpcError(
                RpcErrorCode.INVALID_PARAMS,
                f"invalid status: {status}",
            ) from exc
    limit = _positive_int(params, "limit", default=50)
    approvals = await ctx.application.approval_store.list(
        status=status,
        limit=limit,
    )
    return {"approvals": approvals, "count": len(approvals)}


# 函数说明：approval_get
# 用途：获取审批，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `approval_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `approval`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str`。
# 分支与异常：
#   当 `approval is None` 时，抛出
# `JsonRpcError(RESOURCE_NOT_FOUND, 'approval not found')`。
async def approval_get(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    approval_id = _require_str(params, "approval_id")
    approval = await ctx.application.approval_store.get(approval_id)
    if approval is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "approval not found")
    return {"approval": approval}


# 函数说明：approval_approve
# 用途：批准审批，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `approval_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `approval`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_gate` → `_require_str` →
# `gate.approve`。
# 分支与异常：
#   捕获 `(KeyError, ValueError)` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`
# 。
async def approval_approve(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    gate = _require_gate(ctx)
    approval_id = _require_str(params, "approval_id")
    try:
        approval = await gate.approve(approval_id)
    except (KeyError, ValueError) as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {"approval": approval}


# 函数说明：approval_deny
# 用途：拒绝审批，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `approval_id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `approval`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_gate` → `_require_str` →
# `gate.deny`。
# 分支与异常：
#   捕获 `(KeyError, ValueError)` 后，转换或抛出 `JsonRpcError(INVALID_STATE, str(exc))`
# 。
async def approval_deny(params: dict[str, Any], ctx: RpcContext) -> dict[str, Any]:
    gate = _require_gate(ctx)
    approval_id = _require_str(params, "approval_id")
    try:
        approval = await gate.deny(approval_id)
    except (KeyError, ValueError) as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    return {"approval": approval}


# 函数说明：_require_gate
# 用途：获取并校验必需的`gate`，供JSON-RPC 请求处理使用。
# 参数：
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `Any`；返回 `gate`。
# 分支与异常：
#   当 `gate is None` 时，抛出 `JsonRpcError(…)`。
def _require_gate(ctx: RpcContext) -> Any:
    gate = ctx.application.web_approval_gate
    if gate is None:
        raise JsonRpcError(INVALID_STATE, "web approval gate not available")
    return gate


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
    dispatcher.register("approval.list", approval_list)
    dispatcher.register("approval.get", approval_get)
    dispatcher.register("approval.approve", approval_approve)
    dispatcher.register("approval.deny", approval_deny)
