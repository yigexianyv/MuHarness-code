
from __future__ import annotations

import re
from typing import Any

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import RESOURCE_NOT_FOUND, JsonRpcError, RpcErrorCode

_ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MAX_LIST_LIMIT = 200


# 函数说明：artifact_list
# 用途：列出交付物，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `run_id`、`conversation_id`
# 、`limit`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；按分支返回 `{'artifacts': [], 'count': 0}`；
# `{'artifacts': payload, 'count': len(payload)}`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.list` → `artifact.public_dict`
# 。
# 分支与异常：
#   当 `run_id is not None and (not isinstance(run_id, str) or not…` 时，抛出
# `JsonRpcError(…)`。
#   当 `conversation_id is not None and (not isinstance(…` 时，抛出 `JsonRpcError(…)`。
#   当 `not isinstance(limit, int) or isinstance(limit, bool) or…` 时，抛出
# `JsonRpcError(…)`。
#   当 `store is None` 时，返回 `{'artifacts': [], 'count': 0}`。
async def artifact_list(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:
    run_id = params.get("run_id")
    conversation_id = params.get("conversation_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "run_id must be a string")
    if conversation_id is not None and (
        not isinstance(conversation_id, str) or not conversation_id
    ):
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS, "conversation_id must be a string"
        )
    limit = params.get("limit")
    if limit is None:
        limit = 50
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or limit > _MAX_LIST_LIMIT
    ):
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            f"limit must be an int between 1 and {_MAX_LIST_LIMIT}",
        )

    store = ctx.application.artifact_store
    if store is None:
        return {"artifacts": [], "count": 0}
    artifacts = await store.list(
        run_id=run_id, conversation_id=conversation_id, limit=limit
    )
    payload = [artifact.public_dict() for artifact in artifacts]
    return {"artifacts": payload, "count": len(payload)}


# 函数说明：artifact_get
# 用途：获取交付物，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `id`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `artifact`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_ARTIFACT_ID_RE.fullmatch` →
# `artifact.public_dict`。
# 分支与异常：
#   当 `not isinstance(artifact_id, str) or not…` 时，抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, 'id is required')`。
#   当 `store is None` 时，抛出 `JsonRpcError(RESOURCE_NOT_FOUND, 'artifact not found')`
# 。
#   当 `artifact is None` 时，抛出
# `JsonRpcError(RESOURCE_NOT_FOUND, 'artifact not found')`。
async def artifact_get(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:
    artifact_id = params.get("id")
    if not isinstance(artifact_id, str) or not _ARTIFACT_ID_RE.fullmatch(
        artifact_id
    ):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, "id is required")

    store = ctx.application.artifact_store
    if store is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "artifact not found")
    artifact = await store.get(artifact_id)
    if artifact is None:
        raise JsonRpcError(RESOURCE_NOT_FOUND, "artifact not found")
    return {"artifact": artifact.public_dict()}


# 函数说明：register
# 用途：注册JSON-RPC 请求处理，供JSON-RPC 请求处理使用。
# 参数：
#   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dispatcher.register`。
def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("artifact.list", artifact_list)
    dispatcher.register("artifact.get", artifact_get)
