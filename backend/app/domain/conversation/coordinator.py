
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager


class ConversationOperationCoordinator:

    # 函数说明：ConversationOperationCoordinator.__init__
    # 用途：初始化 ConversationOperationCoordinator；参数及实际保存的实例字段见下方说明
    # 。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._locks`、`self._deleting`。
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._deleting: set[str] = set()

    # 函数说明：ConversationOperationCoordinator._lock_for
    # 用途：获取锁`for`，供会话生命周期与历史持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `asyncio.Lock`；返回 `lock`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Lock`。
    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock

    # 函数说明：ConversationOperationCoordinator.execution
    # 用途：在会话生命周期与历史持久化中处理 `execution`，通过 `self._lock_for` 完成首个
    # 内部处理步骤。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`。
    # 返回：异步生成器，逐项产出 `None`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock_for`。
    # 资源/并发边界：`self._lock_for(conversation_id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   `conversation_id is None` 分支在完成前置处理后返回 `None`。
    #   当 `conversation_id in self._deleting` 时，抛出
    # `KeyError(f'会话正在删除：{conversation_id}')`。
    @asynccontextmanager
    async def execution(
        self,
        conversation_id: str | None,
    ) -> AsyncIterator[None]:

        if conversation_id is None:
            yield
            return
        if conversation_id in self._deleting:
            raise KeyError(f"会话正在删除：{conversation_id}")
        async with self._lock_for(conversation_id):
            if conversation_id in self._deleting:
                raise KeyError(f"会话正在删除：{conversation_id}")
            yield

    # 函数说明：ConversationOperationCoordinator.deletion
    # 用途：在会话生命周期与历史持久化中处理 `deletion`，通过 `self._deleting.add` 完成
    # 首个内部处理步骤。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   stop_active_work：活跃项输入或配置值，类型 `Callable[[], Awaitable[None]]`。
    # 返回：异步生成器，逐项产出 `None`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._deleting.add` →
    # `stop_active_work` → `self._lock_for` → `self._deleting.discard` → `lock.locked` →
    #  `self._locks.pop`。
    # 资源/并发边界：`self._lock_for(conversation_id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `conversation_id in self._deleting` 时，抛出
    # `RuntimeError(f'会话删除已在进行：{conversation_id}')`。
    @asynccontextmanager
    async def deletion(
        self,
        conversation_id: str,
        *,
        stop_active_work: Callable[[], Awaitable[None]],
    ) -> AsyncIterator[None]:

        if conversation_id in self._deleting:
            raise RuntimeError(f"会话删除已在进行：{conversation_id}")
        self._deleting.add(conversation_id)
        try:
            await stop_active_work()
            async with self._lock_for(conversation_id):
                yield
        finally:
            self._deleting.discard(conversation_id)
            lock = self._locks.get(conversation_id)
            if lock is not None and not lock.locked():
                self._locks.pop(conversation_id, None)


__all__ = ["ConversationOperationCoordinator"]
