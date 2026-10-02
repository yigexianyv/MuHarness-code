
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger("muharness.post_run")

PostRunJob = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _PostRunOwner:

    conversation_id: str | None
    run_id: str | None


class PostRunProcessor:

    # 函数说明：PostRunProcessor.__init__
    # 用途：初始化 PostRunProcessor；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   drain_timeout：`drain_timeout`输入或配置值，类型 `float`；默认 `10.0`。
    #   max_concurrency：`max_concurrency`输入或配置值，类型 `int`；默认 `32`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._drain_timeout`、`self._max_concurrency`、`self._active`、
    # `self._closed`。
    def __init__(
        self,
        *,
        drain_timeout: float = 10.0,
        max_concurrency: int = 32,
    ) -> None:
        self._drain_timeout = drain_timeout
        self._max_concurrency = max_concurrency
        self._active: dict[asyncio.Task[None], _PostRunOwner] = {}
        self._closed = False

    # 函数说明：PostRunProcessor.active_count
    # 用途：统计活跃项，供模型与工具执行循环使用。
    # 返回：类型 `int`；返回 `len(self._active)`。
    @property
    def active_count(self) -> int:
        return len(self._active)

    # 函数说明：PostRunProcessor.closed
    # 用途：返回 `self._closed`，提供 PostRunProcessor 的派生值。
    # 返回：类型 `bool`；返回 `self._closed`。
    @property
    def closed(self) -> bool:
        return self._closed

    # 函数说明：PostRunProcessor.submit
    # 用途：排队执行后台收尾任务，并追踪其所属会话以便取消。
    # 参数：
    #   job：传给 `self._run_job` 的输入，类型 `PostRunJob`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
    # 返回：类型 `bool`；按分支返回 `False`；`True`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`logger.warning` →
    # `asyncio.get_running_loop().create_task` → `asyncio.get_running_loop` →
    # `self._run_job` → `_PostRunOwner` → `task.add_done_callback`。
    # 分支与异常：
    #   `self._closed` 分支在完成前置处理后返回 `False`。
    #   `len(self._active) >= self._max_concurrency` 分支在完成前置处理后返回 `False`。
    # 副作用与资源：
    #   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
    def submit(
        self,
        job: PostRunJob,
        *,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        """排队执行后台收尾任务，并追踪其所属会话以便取消。"""
        if self._closed:
            logger.warning("post-run processor closed; dropping background job")
            return False
        if len(self._active) >= self._max_concurrency:
            logger.warning("post-run processor saturated; dropping background job")
            return False
        task = asyncio.get_running_loop().create_task(self._run_job(job))
        self._active[task] = _PostRunOwner(
            conversation_id=conversation_id,
            run_id=run_id,
        )
        task.add_done_callback(self._active.pop)
        return True

    # 函数说明：PostRunProcessor.cancel_for_conversation
    # 用途：取消指定会话仍在排队或执行的收尾任务。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `int`；返回 `len(targets)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`task.cancel` → `asyncio.gather`。
    async def cancel_for_conversation(self, conversation_id: str) -> int:

        """取消指定会话仍在排队或执行的收尾任务。"""
        targets = [
            task
            for task, owner in self._active.items()
            if owner.conversation_id == conversation_id
        ]
        for task in targets:
            task.cancel()
        if targets:
            await asyncio.gather(*targets, return_exceptions=True)
        return len(targets)

    # 函数说明：PostRunProcessor._run_job
    # 用途：运行`job`，供模型与工具执行循环使用。
    # 参数：
    #   job：`job`输入或配置值，类型 `PostRunJob`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`job` → `logger.exception`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，重新抛出原异常。
    #   捕获 `Exception` 后，执行异常处理调用 `logger.exception`。
    async def _run_job(self, job: PostRunJob) -> None:
        try:
            await job()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("post-run background job failed")

    # 函数说明：PostRunProcessor.close
    # 用途：停止接收新任务并等待或取消现有收尾任务。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.wait` → `task.cancel` →
    # `asyncio.gather` → `task.done` → `task.cancelled` → `task.exception`；另有 2 个调
    # 用点。
    # 分支与异常：
    #   当 `not self._active` 时，返回 `None`。
    # 副作用与资源：
    #   更新对象字段：`self._closed`。
    async def close(self) -> None:
        """停止接收新任务并等待或取消现有收尾任务。"""
        self._closed = True
        if not self._active:
            return
        tasks = list(self._active)
        _, pending = await asyncio.wait(tasks, timeout=self._drain_timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in tasks:
            if task.done() and not task.cancelled() and task.exception() is not None:
                logger.error(
                    "post-run job raised unexpectedly",
                    exc_info=task.exception(),
                )
        self._active.clear()


__all__ = ["PostRunProcessor"]
