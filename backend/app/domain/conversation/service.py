
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.models.types import AgentMode, Message, MessageRole
from app.records.trace import SQLiteTraceEventHandler, SQLiteTraceStore
from app.runtime.agent.event_stream import EventEmitter
from app.runtime.agent.events import (
    AgentEventHandler,
    AgentEventType,
    CompositeEventHandler,
)
from app.runtime.agent.result import AgentResult, AgentStopReason
from app.runtime.run.models import RunStatus

from .coordinator import ConversationOperationCoordinator
from .inputs import ConversationSource, TriggerContext
from .store import SQLiteConversationStore

if TYPE_CHECKING:
    from app.runtime.context import SQLiteConversationSummaryStore
    from app.runtime.run import Run, RunManager

logger = logging.getLogger("muharness.conversation.service")


@dataclass
class DispatchResult:

    run: Run
    result: AgentResult
    trigger: TriggerContext
    conversation_id: str | None


class ConversationService:

    # 函数说明：ConversationService.__init__
    # 用途：初始化 ConversationService；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   conversation_store：会话历史存储，类型 `SQLiteConversationStore`。
    #   run_manager：运行管理者输入或配置值，类型 `RunManager`。
    #   trace_store：执行轨迹存储，类型 `SQLiteTraceStore`。
    #   summary_store：摘要持久化存储依赖，类型 `SQLiteConversationSummaryStore | None`
    # ；默认 `None`。
    #   shared_event_handler：事件输入或配置值，类型 `AgentEventHandler | None`；默认
    # `None`。
    #   operation_coordinator：`operation_coordinator`输入或配置值，类型
    # `ConversationOperationCoordinator | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationOperationCoordinator`
    # 。
    # 副作用与资源：
    #   更新对象字段：`self._conversation_store`、`self._run_manager`、
    # `self._trace_store`、`self._summary_store`、`self._shared_event_handler`、
    # `self.operation_coordinator`。
    def __init__(
        self,
        conversation_store: SQLiteConversationStore,
        run_manager: RunManager,
        trace_store: SQLiteTraceStore,
        *,
        summary_store: SQLiteConversationSummaryStore | None = None,
        shared_event_handler: AgentEventHandler | None = None,
        operation_coordinator: ConversationOperationCoordinator | None = None,
    ) -> None:
        self._conversation_store = conversation_store
        self._run_manager = run_manager
        self._trace_store = trace_store
        self._summary_store = summary_store
        self._shared_event_handler = shared_event_handler
        self.operation_coordinator = (
            operation_coordinator or ConversationOperationCoordinator()
        )

    # 函数说明：ConversationService.dispatch
    # 用途：在会话级互斥锁下提交用户消息，避免同一会话并发改写历史。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   content：内容正文，类型 `str`。
    #   trigger：`trigger`输入或配置值，类型 `TriggerContext | None`；默认 `None`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`；默认 `None`。
    #   on_run_started：运行启动回调，类型 `Any | None`；默认 `None`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`；默认 `AgentMode.NORMAL`。
    # 返回：类型 `DispatchResult`；返回 `await self._dispatch_locked(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`TriggerContext` →
    # `self.operation_coordinator.execution` → `self._dispatch_locked`。
    async def dispatch(
        self,
        *,
        conversation_id: str | None,
        content: str,
        trigger: TriggerContext | None = None,
        event_handler: AgentEventHandler | None = None,
        on_run_started: Any | None = None,
        mode: AgentMode = AgentMode.NORMAL,
    ) -> DispatchResult:

        """在会话级互斥锁下提交用户消息，避免同一会话并发改写历史。"""
        trigger = trigger or TriggerContext(
            source=ConversationSource.MANUAL
        )
        async with self.operation_coordinator.execution(conversation_id):
            return await self._dispatch_locked(
                conversation_id=conversation_id,
                content=content,
                trigger=trigger,
                event_handler=event_handler,
                on_run_started=on_run_started,
                mode=mode,
            )

    # 函数说明：ConversationService._dispatch_locked
    # 用途：保存输入、启动运行并在完成后持久化回复与摘要状态。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   content：内容正文，类型 `str`。
    #   trigger：`trigger`输入或配置值，类型 `TriggerContext`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`。
    #   on_run_started：运行启动回调，类型 `Any | None`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    # 返回：类型 `DispatchResult`；返回 `DispatchResult(run=run, result=result, trigger=
    # trigger, conversation_id=conversation_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._conversation_store.load_messages` →
    # `self._conversation_store.load_summary_state` →
    # `self._conversation_store.load_tool_result_views` → `SQLiteTraceEventHandler` →
    # `CompositeEventHandler` → `self._run_manager.start`；另有 11 个调用点。
    # 分支与异常：
    #   捕获 `KeyboardInterrupt` 后，重新抛出原异常。
    #   捕获 `(ValueError, KeyError)` 后，忽略该异常并继续当前流程。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def _dispatch_locked(
        self,
        *,
        conversation_id: str | None,
        content: str,
        trigger: TriggerContext,
        event_handler: AgentEventHandler | None,
        on_run_started: Any | None,
        mode: AgentMode,
    ) -> DispatchResult:
        """保存输入、启动运行并在完成后持久化回复与摘要状态。"""
        history: tuple[Any, ...] = ()
        if conversation_id is not None:
            history = tuple(
                await self._conversation_store.load_messages(conversation_id)
            )
        summary_state = (
            await self._conversation_store.load_summary_state(conversation_id)
            if conversation_id is not None
            else None
        )
        tool_result_views = (
            await self._conversation_store.load_tool_result_views(conversation_id)
            if conversation_id is not None
            else ()
        )

        trace_handler = SQLiteTraceEventHandler(self._trace_store)
        handlers: list[AgentEventHandler] = [trace_handler]
        if self._shared_event_handler is not None:
            handlers.append(self._shared_event_handler)
        if event_handler is not None:
            handlers.append(event_handler)
        handler: AgentEventHandler = (
            CompositeEventHandler(*handlers)
            if len(handlers) > 1
            else handlers[0]
        )

        run_id, _ = await self._run_manager.start(
            content,
            conversation_id=conversation_id,
            history=history,
            summary_state=summary_state,
            tool_result_views=tool_result_views,
            event_handler=handler,
            source=trigger.source.value,
            source_id=trigger.automation_id,
            scheduled_for=trigger.scheduled_for,
            triggered_at=trigger.triggered_at,
            mode=mode,
        )
        if on_run_started is not None:
            await on_run_started(run_id)
        try:
            run = await self._run_manager.wait(run_id)
        except KeyboardInterrupt:
            try:
                await self._run_manager.cancel(run_id)
            except (ValueError, KeyError):
                pass
            raise
        result = self._run_manager.result(run_id)
        if result is None:
            emitter = EventEmitter(
                handler=handler,
                run_id=run_id,
                conversation_id=conversation_id,
                sequence_offset=await self._trace_store.next_sequence(run_id),
            )
            if run.status is RunStatus.CANCELLED:
                cancelled_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "Run cancelled：已停止，未生成最终回复。"
                        "（本轮未完成的内容不会显示）"
                    ),
                )
                result = AgentResult(
                    run_id=run_id,
                    final_message=cancelled_message,
                    messages=(
                        *history,
                        Message(role=MessageRole.USER, content=content),
                        cancelled_message,
                    ),
                    steps=0,
                    stop_reason=AgentStopReason.CANCELLED,
                    summary_state=summary_state,
                    tool_result_views=tool_result_views,
                )
                await emitter.emit(
                    AgentEventType.AGENT_CANCELLED,
                    message=cancelled_message,
                    stop_reason=AgentStopReason.CANCELLED,
                    result=result,
                )
            elif run.status is RunStatus.INTERRUPTED:
                interrupted_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "Run interrupted：已暂停，可从断点继续。"
                        "（点击 Recover 从保存的中断点恢复）"
                    ),
                )
                result = AgentResult(
                    run_id=run_id,
                    final_message=interrupted_message,
                    messages=(
                        *history,
                        Message(role=MessageRole.USER, content=content),
                        interrupted_message,
                    ),
                    steps=0,
                    stop_reason=AgentStopReason.INTERRUPTED,
                    summary_state=summary_state,
                    tool_result_views=tool_result_views,
                )
                await emitter.emit(
                    AgentEventType.AGENT_FAILED,
                    message=interrupted_message,
                    stop_reason=AgentStopReason.INTERRUPTED,
                    result=result,
                )
            else:
                raise RuntimeError("RunManager 未返回最终 AgentResult")

        if conversation_id is not None:
            await self._conversation_store.save_history_state(
                conversation_id,
                result.messages,
                summary_state=result.summary_state,
                tool_result_views=result.tool_result_views,
            )

        return DispatchResult(
            run=run,
            result=result,
            trigger=trigger,
            conversation_id=conversation_id,
        )

    # 函数说明：ConversationService.recover
    # 用途：按会话串行恢复被中断的运行。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   trigger：`trigger`输入或配置值，类型 `TriggerContext | None`；默认 `None`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`；默认 `None`。
    #   on_run_started：运行启动回调，类型 `Any | None`；默认 `None`。
    # 返回：类型 `DispatchResult`；返回 `await self._recover_locked(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_manager.get_run` →
    # `TriggerContext` → `self.operation_coordinator.execution` → `self._recover_locked`
    # 。
    # 分支与异常：
    #   当 `run is None` 时，抛出 `KeyError(f'Run 不存在：{run_id}')`。
    async def recover(
        self,
        run_id: str,
        *,
        trigger: TriggerContext | None = None,
        event_handler: AgentEventHandler | None = None,
        on_run_started: Any | None = None,
    ) -> DispatchResult:

        """按会话串行恢复被中断的运行。"""
        run = await self._run_manager.get_run(run_id)
        if run is None:
            raise KeyError(f"Run 不存在：{run_id}")
        conversation_id = run.conversation_id
        trigger = trigger or TriggerContext(source=ConversationSource.MANUAL)
        async with self.operation_coordinator.execution(conversation_id):
            return await self._recover_locked(
                run_id=run_id,
                conversation_id=conversation_id,
                trigger=trigger,
                event_handler=event_handler,
                on_run_started=on_run_started,
            )

    # 函数说明：ConversationService._recover_locked
    # 用途：加载原运行上下文，并将恢复结果写回所属会话。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   trigger：`trigger`输入或配置值，类型 `TriggerContext`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`。
    #   on_run_started：运行启动回调，类型 `Any | None`。
    # 返回：类型 `DispatchResult`；返回 `DispatchResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._conversation_store.load_messages` →
    # `self._conversation_store.load_summary_state` →
    # `self._conversation_store.load_tool_result_views` → `SQLiteTraceEventHandler` →
    # `CompositeEventHandler` → `self._run_manager.recover`；另有 6 个调用点。
    # 分支与异常：
    #   捕获 `KeyboardInterrupt` 后，重新抛出原异常。
    #   捕获 `(ValueError, KeyError)` 后，忽略该异常并继续当前流程。
    #   当 `result is None` 时，抛出 `RuntimeError('RunManager 未返回最终 AgentResult')`
    # 。
    async def _recover_locked(
        self,
        *,
        run_id: str,
        conversation_id: str | None,
        trigger: TriggerContext,
        event_handler: AgentEventHandler | None,
        on_run_started: Any | None,
    ) -> DispatchResult:
        """加载原运行上下文，并将恢复结果写回所属会话。"""
        history: tuple[Any, ...] = ()
        if conversation_id is not None:
            history = tuple(
                await self._conversation_store.load_messages(conversation_id)
            )
        summary_state = (
            await self._conversation_store.load_summary_state(conversation_id)
            if conversation_id is not None
            else None
        )
        tool_result_views = (
            await self._conversation_store.load_tool_result_views(conversation_id)
            if conversation_id is not None
            else ()
        )

        trace_handler = SQLiteTraceEventHandler(self._trace_store)
        handlers: list[AgentEventHandler] = [trace_handler]
        if self._shared_event_handler is not None:
            handlers.append(self._shared_event_handler)
        if event_handler is not None:
            handlers.append(event_handler)
        handler: AgentEventHandler = (
            CompositeEventHandler(*handlers)
            if len(handlers) > 1
            else handlers[0]
        )

        new_run_id, _ = await self._run_manager.recover(
            run_id,
            history=history,
            summary_state=summary_state,
            tool_result_views=tool_result_views,
            event_handler=handler,
        )
        if on_run_started is not None:
            await on_run_started(new_run_id)
        try:
            recovered_run = await self._run_manager.wait(new_run_id)
        except KeyboardInterrupt:
            try:
                await self._run_manager.cancel(new_run_id)
            except (ValueError, KeyError):
                pass
            raise
        result = self._run_manager.result(new_run_id)
        if result is None:
            raise RuntimeError("RunManager 未返回最终 AgentResult")

        if conversation_id is not None:
            await self._conversation_store.save_history_state(
                conversation_id,
                result.messages,
                summary_state=result.summary_state,
                tool_result_views=result.tool_result_views,
            )

        return DispatchResult(
            run=recovered_run,
            result=result,
            trigger=trigger,
            conversation_id=conversation_id,
        )

    # 函数说明：ConversationService.is_run_running
    # 用途：判断运行是否满足当前实现的条件。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `bool`；返回 `run is not None and run.status.value == 'running'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_manager.get_run`。
    async def is_run_running(self, run_id: str) -> bool:

        run = await self._run_manager.get_run(run_id)
        return run is not None and run.status.value == "running"


__all__ = ["ConversationService", "DispatchResult"]
