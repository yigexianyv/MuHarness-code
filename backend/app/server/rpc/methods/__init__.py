
from __future__ import annotations

from ..dispatcher import RpcDispatcher
from . import (
    approvals,
    artifacts,
    automations,
    conversations,
    extensions,
    mea,
    memories,
    model_settings,
    runs,
    system,
    tasks,
    trace,
)


# 函数说明：build_dispatcher
# 用途：构建`dispatcher`，供JSON-RPC 请求处理使用。
# 返回：类型 `RpcDispatcher`；返回 `dispatcher`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RpcDispatcher` → `system.register` →
# `conversations.register` → `runs.register` → `trace.register` → `automations.register`
# ；另有 7 个调用点。
def build_dispatcher() -> RpcDispatcher:

    dispatcher = RpcDispatcher()
    system.register(dispatcher)
    conversations.register(dispatcher)
    runs.register(dispatcher)
    trace.register(dispatcher)
    automations.register(dispatcher)
    approvals.register(dispatcher)
    artifacts.register(dispatcher)
    tasks.register(dispatcher)
    mea.register(dispatcher)
    memories.register(dispatcher)
    extensions.register(dispatcher)
    model_settings.register(dispatcher)
    return dispatcher


__all__ = ["build_dispatcher"]
