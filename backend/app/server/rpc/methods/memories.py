
from __future__ import annotations

from typing import Any

from ..dispatcher import RpcContext, RpcDispatcher


# 函数说明：memory_list
# 用途：列出记忆，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；按分支返回
# `{'core': '', 'active': [], 'archived': [], 'active_count': 0, 'max_active': 0}`；`{'
# core': core, 'active': [record.model_dump(mode='json') for record in active], '
# archived…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`manager.core.load` → `manager.list` →
#  `manager.list_archived`。
# 分支与异常：
#   当 `manager is None` 时，返回
# `{'core': '', 'active': [], 'archived': [], 'active_count':…`。
async def memory_list(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:
    del params
    manager = ctx.application.memory_manager
    if manager is None:
        return {
            "core": "",
            "active": [],
            "archived": [],
            "active_count": 0,
            "max_active": 0,
        }

    core = await manager.core.load()
    active = await manager.list()
    archived = await manager.list_archived()
    return {
        "core": core,
        "active": [record.model_dump(mode="json") for record in active],
        "archived": [record.model_dump(mode="json") for record in archived],
        "active_count": len(active),
        "max_active": manager.max_active,
    }


# 函数说明：register
# 用途：注册JSON-RPC 请求处理，供JSON-RPC 请求处理使用。
# 参数：
#   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dispatcher.register`。
def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("memory.list", memory_list)


__all__ = ["memory_list", "register"]
