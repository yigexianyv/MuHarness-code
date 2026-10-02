
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from app.models.types import AgentMode
from app.runtime.agent.events import AgentEventHandler
from app.runtime.agent.result import AgentResult
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.checkpoint import CheckpointStatus, SQLiteCheckpointStore
from app.runtime.context import ConversationSummaryState
from app.runtime.context.tool_views import ToolResultView

if TYPE_CHECKING:
    from app.safety.approval import SQLiteApprovalStore

from .models import TERMINAL_STATUSES, Run, RunStatus
from .store import SQLiteRunStore

_STALE_RUN_ERROR = "process restarted; run did not reach a terminal state"
_STALE_PENDING_ERROR = "process restarted; run never started"
logger = logging.getLogger("muharness.run.manager")
RunFinalizer = Callable[[str], Awaitable[object] | object]


class RunManager:

    # 函数说明：RunManager.__init__
    # 用途：初始化 RunManager；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   run_store：运行持久化存储依赖，类型 `SQLiteRunStore`。
    #   checkpoint_store：运行检查点存储，类型 `SQLiteCheckpointStore`。
    #   runtime：执行环境输入或配置值，类型 `AgentRuntime`。
    #   approval_store：审批记录存储，类型 `SQLiteApprovalStore | None`；默认 `None`。
    #   run_finalizers：传给 `tuple` 的输入，类型 `Sequence[RunFinalizer]`；默认 `()`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._run_store`、`self._checkpoint_store`、`self._runtime`、
    # `self._approval_store`、`self._run_finalizers`、`self._active_tasks`、
    # `self._last_results`。
    def __init__(
        self,
        run_store: SQLiteRunStore,
        checkpoint_store: SQLiteCheckpointStore,
        runtime: AgentRuntime,
        approval_store: SQLiteApprovalStore | None = None,
        run_finalizers: Sequence[RunFinalizer] = (),
    ) -> None:
        self._run_store = run_store
        self._checkpoint_store = checkpoint_store
        self._runtime = runtime
        self._approval_store = approval_store
        self._run_finalizers = tuple(run_finalizers)
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._last_results: dict[str, AgentResult] = {}


    # 函数说明：RunManager.initialize
    # 用途：初始化运行存储，并对上次进程留下的运行记录进行状态对账。
    # 返回：类型 `tuple[Run, ...]`；返回 `await self.reconcile()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_store.initialize` →
    # `self.reconcile`。
    async def initialize(self) -> tuple[Run, ...]:

        """初始化运行存储，并对上次进程留下的运行记录进行状态对账。"""
        await self._run_store.initialize()
        return await self.reconcile()

    # 函数说明：RunManager.reconcile
    # 用途：youxiangeigaopingjia根据检查点修复遗留的待运行与运行中记录，保留可恢复状态。
    # 返回：类型 `tuple[Run, ...]`；返回 `tuple(reconciled)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._checkpoint_store.recover_running` → `self._run_store.list_runs` →
    # `self._run_store.mark_failed` → `self._run_store.mark_interrupted` →
    # `self._run_store.mark_completed`。
    async def reconcile(self) -> tuple[Run, ...]:

        """youxiangeigaopingjia根据检查点修复遗留的待运行与运行中记录，保留可恢复状态。"""
        if self._checkpoint_store is not None:
            await self._checkpoint_store.recover_running()
        stale_pending = await self._run_store.list_runs(status=RunStatus.PENDING)
        reconciled: list[Run] = []
        for run in stale_pending:
            updated = await self._run_store.mark_failed(
                run.id,
                error=_STALE_PENDING_ERROR,
            )
            reconciled.append(updated)
        stale = await self._run_store.list_runs(status=RunStatus.RUNNING)
        for run in stale:
            checkpoint = await self._checkpoint_store.get(run.id)
            if checkpoint is None:
                updated = await self._run_store.mark_failed(
                    run.id,
                    error=_STALE_RUN_ERROR
                    + " (no recoverable checkpoint)",
                )
            elif checkpoint.status is CheckpointStatus.INTERRUPTED:
                updated = await self._run_store.mark_interrupted(
                    run.id,
                    error=_STALE_RUN_ERROR
                    + " (recoverable checkpoint preserved)",
                )
            elif checkpoint.status is CheckpointStatus.COMPLETED:
                updated = await self._run_store.mark_completed(
                    run.id,
                    stop_reason=(
                        checkpoint.stop_reason.value
                        if checkpoint.stop_reason is not None
                        else None
                    ),
                )
            else:
                updated = await self._run_store.mark_failed(
                    run.id,
                    error=checkpoint.error,
                )
            reconciled.append(updated)
        return tuple(reconciled)


    # 函数说明：RunManager.start
    # 用途：先创建持久化运行记录，再启动后台任务；返回运行 ID 与任务句柄。
    # 参数：
    #   user_message：当前用户消息，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   history：原始会话历史，类型 `tuple[Any, ...]`；默认 `()`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   tool_result_views：工具结果的固定模型视图，类型 `tuple[ToolResultView, ...]`；默
    # 认 `()`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`；默认 `None`。
    #   recovery_run_id：待恢复的运行标识，类型 `str | None`；默认 `None`。
    #   recovered_from_run_id：运行标识，类型 `str | None`；默认 `None`。
    #   source：输入来源或原始数据，类型 `str | None`；默认 `None`。
    #   source_id：来源记录标识，类型 `str | None`；默认 `None`。
    #   scheduled_for：计划触发时间，类型 `datetime | None`；默认 `None`。
    #   triggered_at：实际触发时间，类型 `datetime | None`；默认 `None`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`；默认 `AgentMode.NORMAL`。
    #   runtime：执行环境输入或配置值，类型 `AgentRuntime | None`；默认 `None`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
    #   tool_context_metadata：本次工具执行关联的元数据，类型 `Mapping[str, Any] | None`
    # ；默认 `None`。
    # 返回：类型 `tuple[str, asyncio.Task[None]]`；返回 `(run.id, task)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_store.create` →
    # `self._run_store.mark_started` → `asyncio.create_task` → `self._execute`。
    # 副作用与资源：
    #   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
    async def start(
        self,
        user_message: str,
        *,
        conversation_id: str | None = None,
        history: tuple[Any, ...] = (),
        summary_state: ConversationSummaryState | None = None,
        tool_result_views: tuple[ToolResultView, ...] = (),
        event_handler: AgentEventHandler | None = None,
        recovery_run_id: str | None = None,
        recovered_from_run_id: str | None = None,
        source: str | None = None,
        source_id: str | None = None,
        scheduled_for: datetime | None = None,
        triggered_at: datetime | None = None,
        mode: AgentMode = AgentMode.NORMAL,
        runtime: AgentRuntime | None = None,
        run_id: str | None = None,
        tool_context_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str, asyncio.Task[None]]:

        """先创建持久化运行记录，再启动后台任务；返回运行 ID 与任务句柄。

        ``runtime`` 为空时使用默认 Runtime；长任务的三个角色各自传入自己的 Runtime。
        ``run_id`` 由调用方预先分配时，重复创建会抛 ``RunAlreadyExists``，
        绝不会启动第二次。``tool_context_metadata`` 是本次运行的可信内存上下文，
        会复制后传给工具；不会持久化到 Run。
        """
        run = await self._run_store.create(
            conversation_id=conversation_id,
            user_message=user_message,
            recovered_from_run_id=recovered_from_run_id,
            source=source,
            source_id=source_id,
            scheduled_for=scheduled_for,
            triggered_at=triggered_at,
            mode=mode,
            run_id=run_id,
        )
        await self._run_store.mark_started(run.id)
        frozen_tool_context_metadata = dict(tool_context_metadata or {})
        task = asyncio.create_task(
            self._execute(
                run.id,
                user_message=user_message,
                conversation_id=conversation_id,
                history=history,
                summary_state=summary_state,
                tool_result_views=tool_result_views,
                event_handler=event_handler,
                recovery_run_id=recovery_run_id,
                mode=mode,
                runtime=runtime or self._runtime,
                tool_context_metadata=frozen_tool_context_metadata,
            )
        )
        self._active_tasks[run.id] = task
        return run.id, task

    # 函数说明：RunManager.wait
    # 用途：等待指定运行结束并读取其最终 Agent 结果。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Run`；返回 `await self._run_store.require(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_store.require`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，忽略该异常并继续当前流程。
    async def wait(self, run_id: str) -> Run:

        """等待指定运行结束并读取其最终 Agent 结果。"""
        task = self._active_tasks.get(run_id)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
        return await self._run_store.require(run_id)

    # 函数说明：RunManager.result
    # 用途：返回 `self._last_results.get(run_id)`，提供 RunManager 的派生值。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `AgentResult | None`；返回 `self._last_results.get(run_id)`。
    def result(self, run_id: str) -> AgentResult | None:

        return self._last_results.get(run_id)

    # 函数说明：RunManager.forget_results
    # 用途：清除结果集合，供运行调度与状态持久化使用。
    # 参数：
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._last_results.pop`。
    # 分支与异常：
    #   当 `run_id in self._active_tasks` 时，抛出
    # `RuntimeError(f'cannot forget active run: {run_id}')`。
    def forget_results(self, run_ids: tuple[str, ...]) -> None:

        for run_id in run_ids:
            if run_id in self._active_tasks:
                raise RuntimeError(f"cannot forget active run: {run_id}")
            self._last_results.pop(run_id, None)

    # 函数说明：RunManager.get_run
    # 用途：获取运行，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Run | None`；返回 `await self._run_store.get(run_id)`。
    async def get_run(self, run_id: str) -> Run | None:
        return await self._run_store.get(run_id)

    # 函数说明：RunManager.list_runs
    # 用途：列出运行集合，供运行调度与状态持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   status：目标状态，类型 `RunStatus | str | None`；默认 `None`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `20`。
    # 返回：类型 `tuple[Run, ...]`；返回 `await self._run_store.list_runs(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_store.list_runs`。
    async def list_runs(
        self,
        *,
        conversation_id: str | None = None,
        status: RunStatus | str | None = None,
        limit: int = 20,
    ) -> tuple[Run, ...]:
        return await self._run_store.list_runs(
            conversation_id=conversation_id,
            status=status,
            limit=limit,
        )

    # 函数说明：RunManager.active_run_ids
    # 用途：运行`ids`，供运行调度与状态持久化使用。
    # 返回：类型 `tuple[str, ...]`；返回 `tuple((run_id for run_id, task in self.
    # _active_tasks.items() if not task.done()))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`task.done`。
    @property
    def active_run_ids(self) -> tuple[str, ...]:
        return tuple(
            run_id
            for run_id, task in self._active_tasks.items()
            if not task.done()
        )


    # 函数说明：RunManager.cancel
    # 用途：取消当前运行，并同步处理待决审批与运行状态。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Run`；按分支返回 `run`；`updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_store.require` →
    # `self._run_store.mark_failed` → `self._cancel_pending_approvals` →
    # `self._run_finalizers_for` → `task.done` → `self._run_store.mark_cancelled`；另有
    # 1 个调用点。
    # 分支与异常：
    #   当 `run.status in TERMINAL_STATUSES` 时，返回 `run`。
    #   当 `run.status not in {RunStatus.PENDING, RunStatus.RUNNING}` 时，抛出
    # `ValueError(f'cannot cancel run in state {run.status.value}')`。
    #   `run.status is RunStatus.PENDING` 分支在完成前置处理后返回 `updated`。
    #   `task is None or task.done()` 分支在完成前置处理后返回 `updated`。
    #   捕获 `asyncio.CancelledError` 后，忽略该异常并继续当前流程。
    async def cancel(self, run_id: str) -> Run:

        """取消当前运行，并同步处理待决审批与运行状态。"""
        run = await self._run_store.require(run_id)
        if run.status in TERMINAL_STATUSES:
            return run
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            raise ValueError(
                f"cannot cancel run in state {run.status.value}"
            )
        if run.status is RunStatus.PENDING:
            updated = await self._run_store.mark_failed(
                run_id,
                error="conversation deleted before execution started",
            )
            await self._cancel_pending_approvals(run_id)
            await self._run_finalizers_for(run_id)
            return updated
        task = self._active_tasks.get(run_id)
        if task is None or task.done():
            updated = await self._run_store.mark_cancelled(
                run_id,
                error="cancelled without active execution",
            )
            await self._cancel_pending_approvals(run_id)
            await self._run_finalizers_for(run_id)
            return updated
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        updated = await self._run_store.require(run_id)
        await self._cancel_pending_approvals(run_id)
        return updated

    # 函数说明：RunManager.cancel_for_conversation
    # 用途：取消会话，供运行调度与状态持久化使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `tuple[Run, ...]`；返回 `tuple(cancelled)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._run_store.list_for_conversation` → `self.cancel`。
    # 分支与异常：
    #   当 `current is None or current.status in TERMINAL_STATUSES` 时，跳过当前循环项。
    async def cancel_for_conversation(
        self,
        conversation_id: str,
    ) -> tuple[Run, ...]:

        active = await self._run_store.list_for_conversation(
            conversation_id,
            active_only=True,
        )
        cancelled: list[Run] = []
        for run in active:
            current = await self._run_store.get(run.id)
            if current is None or current.status in TERMINAL_STATUSES:
                continue
            cancelled.append(await self.cancel(run.id))
        return tuple(cancelled)

    # 函数说明：RunManager.interrupt
    # 用途：中断运行以保留可恢复的检查点。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Run`；按分支返回 `run`；`updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_store.require` →
    # `self._run_store.mark_interrupted` → `task.done` → `task.cancel` →
    # `self._cancel_pending_approvals`。
    # 分支与异常：
    #   当 `run.status in TERMINAL_STATUSES` 时，返回 `run`。
    #   当 `run.status is not RunStatus.RUNNING` 时，抛出 `ValueError(…)`。
    #   捕获 `asyncio.CancelledError` 后，忽略该异常并继续当前流程。
    async def interrupt(self, run_id: str) -> Run:

        """中断运行以保留可恢复的检查点。"""
        run = await self._run_store.require(run_id)
        if run.status in TERMINAL_STATUSES:
            return run
        if run.status is not RunStatus.RUNNING:
            raise ValueError(
                f"cannot interrupt run in state {run.status.value}"
            )
        await self._run_store.mark_interrupted(
            run_id,
            error="interrupted by user",
        )
        task = self._active_tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        updated = await self._run_store.require(run_id)
        await self._cancel_pending_approvals(run_id)
        return updated

    # 函数说明：RunManager._cancel_pending_approvals
    # 用途：取消待处理项，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._approval_store.cancel_pending_for_run`。
    async def _cancel_pending_approvals(self, run_id: str) -> None:

        if self._approval_store is not None:
            await self._approval_store.cancel_pending_for_run(run_id)

    # 函数说明：RunManager._run_finalizers_for
    # 用途：运行`finalizers_for`，供运行调度与状态持久化使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`finalizer` →
    # `inspect.isawaitable` → `logger.exception`。
    # 分支与异常：
    #   捕获 `Exception` 后，执行异常处理调用 `logger.exception`。
    async def _run_finalizers_for(self, run_id: str) -> None:

        for finalizer in self._run_finalizers:
            try:
                result = finalizer(run_id)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("run finalizer failed for %s", run_id)


    # 函数说明：RunManager.recover
    # 用途：从可恢复检查点创建新的运行，延续原始请求。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   history：原始会话历史，类型 `tuple[Any, ...]`；默认 `()`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   tool_result_views：工具结果的固定模型视图，类型 `tuple[ToolResultView, ...]`；默
    # 认 `()`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`；默认 `None`。
    # 返回：类型 `tuple[str, asyncio.Task[None]]`；返回 `await self.start(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_store.require` →
    # `self._checkpoint_store.get_unrecovered` → `self.start`。
    # 分支与异常：
    #   当 `run.status is not RunStatus.INTERRUPTED` 时，抛出 `ValueError(…)`。
    #   当 `checkpoint is None` 时，抛出
    # `ValueError(f'no recoverable checkpoint for run {run_id}')`。
    async def recover(
        self,
        run_id: str,
        *,
        history: tuple[Any, ...] = (),
        summary_state: ConversationSummaryState | None = None,
        tool_result_views: tuple[ToolResultView, ...] = (),
        event_handler: AgentEventHandler | None = None,
    ) -> tuple[str, asyncio.Task[None]]:

        """从可恢复检查点创建新的运行，延续原始请求。"""
        run = await self._run_store.require(run_id)
        if run.status is not RunStatus.INTERRUPTED:
            raise ValueError(
                f"only interrupted run can be recovered: {run_id} "
                f"({run.status.value})"
            )
        checkpoint = await self._checkpoint_store.get_unrecovered(run_id)
        if checkpoint is None:
            raise ValueError(
                f"no recoverable checkpoint for run {run_id}"
            )
        return await self.start(
            checkpoint.user_message.content or run.user_message,
            conversation_id=run.conversation_id,
            history=history,
            summary_state=summary_state,
            tool_result_views=tool_result_views,
            event_handler=event_handler,
            recovery_run_id=run_id,
            recovered_from_run_id=run_id,
            mode=run.mode,
        )


    # 函数说明：RunManager._execute
    # 用途：管理运行状态转换，调用 Agent 并持久化完成、失败或取消结果。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   user_message：当前用户消息，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   history：原始会话历史，类型 `tuple[Any, ...]`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`。
    #   tool_result_views：工具结果的固定模型视图，类型 `tuple[ToolResultView, ...]`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`。
    #   recovery_run_id：待恢复的运行标识，类型 `str | None`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   runtime：执行环境输入或配置值，类型 `AgentRuntime`。
    #   tool_context_metadata：本次工具执行关联的元数据，类型 `Mapping[str, Any] | None`
    # 。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`runtime.run_stream` →
    # `self._run_store.mark_cancelled` → `self._run_store.mark_failed` →
    # `self._run_store.mark_completed` → `self._run_finalizers_for` →
    # `self._active_tasks.pop`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，重新抛出原异常。
    #   捕获 `BaseException` 后，返回 `None`。
    #   `result is None` 分支在完成前置处理后返回 `None`。
    async def _execute(
        self,
        run_id: str,
        *,
        user_message: str,
        conversation_id: str | None,
        history: tuple[Any, ...],
        summary_state: ConversationSummaryState | None,
        tool_result_views: tuple[ToolResultView, ...],
        event_handler: AgentEventHandler | None,
        recovery_run_id: str | None,
        mode: AgentMode,
        runtime: AgentRuntime,
        tool_context_metadata: Mapping[str, Any] | None,
    ) -> None:
        """管理运行状态转换，调用 Agent 并持久化完成、失败或取消结果。"""
        try:
            result: AgentResult | None = None
            try:
                async for event in runtime.run_stream(
                    user_message,
                    history=history,
                    conversation_id=conversation_id,
                    event_handler=event_handler,
                    summary_state=summary_state,
                    tool_result_views=tool_result_views,
                    run_id=run_id,
                    recovery_run_id=recovery_run_id,
                    mode=mode,
                    tool_context_metadata=tool_context_metadata,
                ):
                    if event.result is not None:
                        result = event.result
            except asyncio.CancelledError:
                current = await self._run_store.get(run_id)
                if current is None or current.status is RunStatus.RUNNING:
                    await self._run_store.mark_cancelled(
                        run_id,
                        error="cancelled by user",
                    )
                raise
            except BaseException as exc:
                await self._run_store.mark_failed(
                    run_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return

            if result is None:
                await self._run_store.mark_failed(
                    run_id,
                    error="agent produced no result",
                )
                return
            self._last_results[run_id] = result
            if result.ok:
                await self._run_store.mark_completed(
                    run_id,
                    stop_reason=result.stop_reason.value,
                )
            else:
                await self._run_store.mark_failed(
                    run_id,
                    error=(
                        result.error.message
                        if result.error is not None
                        else result.stop_reason.value
                    ),
                    stop_reason=result.stop_reason.value,
                )
        finally:
            await self._run_finalizers_for(run_id)
            self._active_tasks.pop(run_id, None)


__all__ = ["RunManager"]
