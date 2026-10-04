
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.domain.memory import (
    MemoryMaintenanceReflector,
    MemoryManager,
    PostRunMemoryReflector,
)
from app.domain.skills import (
    SkillContextProvider,
    SkillStore,
)
from app.domain.task.context import TaskContextProvider
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    AgentMode,
    Message,
    MessageRole,
    ModelProvider,
)
from app.runtime.checkpoint import (
    RunCheckpoint,
    SQLiteCheckpointStore,
)
from app.runtime.context import ContextManager, ConversationSummaryState
from app.runtime.context.tool_views import ToolResultView
from app.runtime.rewind import RunStepRecorder
from app.tools.approval import ApprovalGate
from app.tools.executor import ToolExecutor
from app.tools.hooks import ToolHook
from app.tools.observability import ToolExecutionRecord
from app.tools.output import ToolOutputRecorder
from app.tools.permissions.policy import PermissionPolicyEngine
from app.tools.permissions.store import PermissionRuleStore
from app.tools.registry import ToolRegistry

from .budget import (
    RunBudget,
    RunBudgetConfig,
)
from .event_stream import (
    STREAM_FINISHED as _STREAM_FINISHED,
)
from .event_stream import (
    EventEmitter as _EventEmitter,
)
from .event_stream import (
    QueueEventHandler as _QueueEventHandler,
)
from .events import (
    AgentEvent,
    AgentEventHandler,
    AgentEventType,
    CompositeEventHandler,
    NullEventHandler,
)
from .loop import AgentLoop, ConstraintsProvider
from .memory_post_run import PostRunMemoryCoordinator
from .result import (
    AgentResult,
)

if TYPE_CHECKING:
    from app.runtime.rewind import SQLiteRunStepStore, WorkspaceSnapshotStore
    from app.runtime.run.messages_store import (
        RunMessageRecorder,
        SQLiteRunMessageStore,
    )


class AgentRuntime:

    # 函数说明：AgentRuntime.__init__
    # 用途：初始化 AgentRuntime；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   model_registry：模型适配器注册表，类型 `ModelAdapterRegistry`。
    #   tool_registry：工具注册表，类型 `ToolRegistry`。
    #   provider：模型或搜索服务商，类型 `ModelProvider | str | None`；默认 `None`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   system_prompt：系统提示文本，类型 `str | None`；默认 `None`。
    #   max_steps：模型执行步数上限，类型 `int`；默认 `12`。
    #   max_tool_rounds：工具调用轮次上限，类型 `int | None`；默认 `None`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    #   tool_executor：工具执行器，类型 `ToolExecutor | None`；默认 `None`。
    #   tool_hooks：工具输入或配置值，类型 `Sequence[ToolHook]`；默认 `()`。
    #   approval_gate：工具审批门，类型 `ApprovalGate | None`；默认 `None`。
    #   policy_engine：权限规则决策引擎，类型 `PermissionPolicyEngine | None`；默认
    # `None`。
    #   rule_store：权限规则存储，类型 `PermissionRuleStore | None`；默认 `None`。
    #   context_manager：上下文预算与压缩管理器，类型 `ContextManager | None`；默认
    # `None`。
    #   task_context_provider：任务状态上下文提供器，类型 `TaskContextProvider | None`；
    # 默认 `None`。
    #   checkpoint_store：运行检查点存储，类型 `SQLiteCheckpointStore | None`；默认
    # `None`。
    #   memory_manager：长期记忆管理器，类型 `MemoryManager | None`；默认 `None`。
    #   memory_reflector：记忆输入或配置值，类型 `PostRunMemoryReflector | None`；默认
    # `None`。
    #   memory_maintenance_reflector：记忆维护输入或配置值，类型
    # `MemoryMaintenanceReflector | None`；默认 `None`。
    #   skill_store：技能存储，类型 `SkillStore | None`；默认 `None`。
    #   skill_context_provider：已激活技能上下文提供器，类型
    # `SkillContextProvider | None`；默认 `None`。
    #   tool_output_recorder：工具输出输入或配置值，类型 `ToolOutputRecorder | None`；默
    # 认 `None`。
    #   post_run_submit：运行输入或配置值，类型 `Callable[..., bool] | None`；默认
    # `None`。
    #   run_budget_config：传给 `RunBudget` 的输入，类型 `RunBudgetConfig | None`；默认
    # `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextManager` →
    # `PostRunMemoryCoordinator` → `RunBudget` → `ToolExecutor` → `AgentLoop`。
    # 分支与异常：
    #   当 `max_steps < 1` 时，抛出 `ValueError('max_steps must be at least 1')`。
    #   当 `max_output_tokens is not None and max_output_tokens < 1` 时，抛出
    # `ValueError('max_output_tokens must be at least 1')`。
    #   当 `max_tool_rounds is not None and max_tool_rounds < 1` 时，抛出
    # `ValueError('max_tool_rounds must be at least 1')`。
    #   当 `system_prompt is not None and (not isinstance(system_prompt…` 时，抛出
    # `TypeError('system_prompt must be a string or None')`。
    # 副作用与资源：
    #   更新对象字段：`self._system_prompt`、`self._context_manager`、
    # `self._checkpoint_store`、`self._post_run`、`self._run_budget`、
    # `self._tool_executor`、`self._loop`。
    def __init__(
        self,
        model_registry: ModelAdapterRegistry,
        tool_registry: ToolRegistry,
        *,
        provider: ModelProvider | str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        max_steps: int = 12,
        max_tool_rounds: int | None = None,
        max_output_tokens: int | None = None,
        tool_executor: ToolExecutor | None = None,
        tool_hooks: Sequence[ToolHook] = (),
        approval_gate: ApprovalGate | None = None,
        policy_engine: PermissionPolicyEngine | None = None,
        rule_store: PermissionRuleStore | None = None,
        context_manager: ContextManager | None = None,
        task_context_provider: TaskContextProvider | None = None,
        checkpoint_store: SQLiteCheckpointStore | None = None,
        memory_manager: MemoryManager | None = None,
        memory_reflector: PostRunMemoryReflector | None = None,
        memory_maintenance_reflector: MemoryMaintenanceReflector | None = None,
        skill_store: SkillStore | None = None,
        skill_context_provider: SkillContextProvider | None = None,
        tool_output_recorder: ToolOutputRecorder | None = None,
        post_run_submit: Callable[..., bool] | None = None,
        run_budget_config: RunBudgetConfig | None = None,
        run_message_store: SQLiteRunMessageStore | None = None,
        constraints_provider: ConstraintsProvider | None = None,
        run_step_store: SQLiteRunStepStore | None = None,
        workspace_snapshots: WorkspaceSnapshotStore | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if max_tool_rounds is not None and max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None")
        if memory_reflector is not None and memory_manager is None:
            raise ValueError("memory_reflector requires memory_manager")
        if memory_maintenance_reflector is not None and memory_manager is None:
            raise ValueError("memory_maintenance_reflector requires memory_manager")
        if skill_context_provider is not None and skill_store is None:
            raise ValueError("skill_context_provider requires skill_store")

        self._system_prompt = (
            system_prompt
            if system_prompt is not None and system_prompt.strip()
            else None
        )
        self._context_manager = context_manager or ContextManager()
        self._checkpoint_store = checkpoint_store
        self._run_message_store = run_message_store
        self._run_step_store = run_step_store
        self._workspace_snapshots = workspace_snapshots
        self._post_run = PostRunMemoryCoordinator(
            manager=memory_manager,
            reflector=memory_reflector,
            maintenance_reflector=memory_maintenance_reflector,
            task_context_provider=task_context_provider,
            submit=post_run_submit,
        )
        self._run_budget = RunBudget(run_budget_config)
        self._tool_executor = tool_executor or ToolExecutor(
            tool_registry,
            approval_gate=approval_gate,
            policy_engine=policy_engine,
            rule_store=rule_store,
            hooks=tool_hooks,
            output_recorder=tool_output_recorder,
        )
        self._loop = AgentLoop(
            model_registry=model_registry,
            tool_registry=tool_registry,
            tool_executor=self._tool_executor,
            provider=provider,
            model=model,
            system_prompt=self._system_prompt,
            max_steps=max_steps,
            max_tool_rounds=max_tool_rounds,
            max_output_tokens=max_output_tokens,
            context_manager=self._context_manager,
            task_context_provider=task_context_provider,
            checkpoint_store=checkpoint_store,
            memory_manager=memory_manager,
            skill_store=skill_store,
            skill_context_provider=skill_context_provider,
            run_budget=self._run_budget,
            constraints_provider=constraints_provider,
        )

    # 函数说明：AgentRuntime.tool_executor
    # 用途：返回 `self._tool_executor`，提供 AgentRuntime 的派生值。
    # 返回：类型 `ToolExecutor`；返回 `self._tool_executor`。
    @property
    def tool_executor(self) -> ToolExecutor:
        return self._tool_executor

    # 函数说明：AgentRuntime.tool_records
    # 用途：返回 `self._tool_executor.execution_records`，提供 AgentRuntime 的派生值。
    # 返回：类型 `tuple[ToolExecutionRecord, ...]`；返回
    # `self._tool_executor.execution_records`。
    @property
    def tool_records(self) -> tuple[ToolExecutionRecord, ...]:
        return self._tool_executor.execution_records

    # 函数说明：AgentRuntime.run
    # 用途：执行一次 Agent 运行，复用历史与检查点并返回最终结果。
    # 参数：
    #   user_input：本次用户输入，类型 `str`。
    #   history：原始会话历史，类型 `Sequence[Message]`；默认 `()`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`；默认 `None`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   tool_result_views：工具结果的固定模型视图，类型 `Sequence[ToolResultView]`；默认
    #  `()`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
    #   recovery_run_id：待恢复的运行标识，类型 `str | None`；默认 `None`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`；默认 `AgentMode.NORMAL`。
    #   tool_context_metadata：本次工具执行关联的元数据，类型 `Mapping[str, Any] | None`
    # ；默认 `None`。
    # 返回：类型 `AgentResult`；返回 `result`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`uuid4` → `_EventEmitter` →
    # `NullEventHandler` → `self._checkpoint_store.get_unrecovered` →
    # `self._checkpoint_store.start` → `Message`；另有 9 个调用点。
    # 分支与异常：
    #   捕获 `BaseException` 后，重新抛出原异常。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def run(
        self,
        user_input: str,
        *,
        history: Sequence[Message] = (),
        conversation_id: str | None = None,
        event_handler: AgentEventHandler | None = None,
        summary_state: ConversationSummaryState | None = None,
        tool_result_views: Sequence[ToolResultView] = (),
        run_id: str | None = None,
        recovery_run_id: str | None = None,
        mode: AgentMode = AgentMode.NORMAL,
        tool_context_metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:

        """执行一次 Agent 运行，复用历史与检查点并返回最终结果。

        运行事件由 event_handler 接收。
        """
        run_id = run_id or uuid4().hex
        emitter = _EventEmitter(
            handler=event_handler or NullEventHandler(),
            run_id=run_id,
            conversation_id=conversation_id,
        )
        try:
            recovery_checkpoint: RunCheckpoint | None = None
            if self._checkpoint_store is not None:
                if recovery_run_id is not None:
                    recovery_checkpoint = (
                        await self._checkpoint_store.get_unrecovered(
                            recovery_run_id
                        )
                    )
                await self._checkpoint_store.start(
                    run_id,
                    conversation_id=conversation_id,
                    user_message=Message(
                        role=MessageRole.USER,
                        content=user_input,
                    ),
                )
            recorder: RunMessageRecorder | None = None
            if self._run_message_store is not None:
                with suppress(Exception):
                    recorder = await self._run_message_store.start_recording(
                        run_id,
                        conversation_id=conversation_id,
                        history=history,
                    )
            step_recorder: RunStepRecorder | None = None
            if self._run_step_store is not None:
                step_recorder = RunStepRecorder(
                    self._run_step_store,
                    self._workspace_snapshots,
                    run_id,
                )
            try:
                result = await self._loop.run(
                    run_id,
                    user_input,
                    history=history,
                    conversation_id=conversation_id,
                    emitter=emitter,
                    summary_state=summary_state,
                    tool_result_views=tool_result_views,
                    recovery_checkpoint=recovery_checkpoint,
                    mode=mode,
                    tool_context_metadata=tool_context_metadata,
                    message_recorder=recorder,
                    step_recorder=step_recorder,
                )
            except BaseException as exc:
                if self._checkpoint_store is not None:
                    with suppress(Exception):
                        await self._checkpoint_store.interrupt(
                            run_id,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                raise

            if recorder is not None:
                await recorder.sync(result.messages)
            if step_recorder is not None:
                await step_recorder.finish()
            if self._checkpoint_store is not None:
                if result.ok:
                    await self._checkpoint_store.complete(
                        run_id,
                        stop_reason=result.stop_reason,
                    )
                    if recovery_checkpoint is not None:
                        with suppress(Exception):
                            await self._checkpoint_store.mark_recovered(
                                recovery_checkpoint.run_id,
                                recovered_by_run_id=run_id,
                            )
                else:
                    await self._checkpoint_store.fail(
                        run_id,
                        stop_reason=result.stop_reason,
                        error=(
                            result.error.message
                            if result.error is not None
                            else None
                        ),
                    )
            await emitter.emit(
                (
                    AgentEventType.AGENT_COMPLETED
                    if result.ok
                    else AgentEventType.AGENT_FAILED
                ),
                step=result.steps or None,
                message=result.final_message,
                usage=result.usage,
                stop_reason=result.stop_reason,
                error=result.error,
                result=result,
            )
            await self._post_run.schedule(
                result,
                user_input=user_input,
                conversation_id=conversation_id,
                emitter=emitter,
            )
            return result
        finally:
            with suppress(Exception):
                await self._tool_executor.clear_run_rules(run_id)

    # 函数说明：AgentRuntime.run_stream
    # 用途：启动流式运行，按生成顺序输出事件并在结束时清理事件通道。
    # 参数：
    #   user_input：本次用户输入，类型 `str`。
    #   history：原始会话历史，类型 `Sequence[Message]`；默认 `()`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`；默认 `None`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   tool_result_views：工具结果的固定模型视图，类型 `Sequence[ToolResultView]`；默认
    #  `()`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
    #   recovery_run_id：待恢复的运行标识，类型 `str | None`；默认 `None`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`；默认 `AgentMode.NORMAL`。
    #   tool_context_metadata：本次工具执行关联的元数据，类型 `Mapping[str, Any] | None`
    # ；默认 `None`。
    # 返回：异步生成器，逐项产出 `item`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_QueueEventHandler` →
    # `CompositeEventHandler` → `asyncio.create_task` → `execute` → `queue_handler.next`
    #  → `task.done`；另有 2 个调用点。
    # 分支与异常：
    #   当 `item is _STREAM_FINISHED` 时，结束当前循环。
    # 副作用与资源：
    #   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
    async def run_stream(
        self,
        user_input: str,
        *,
        history: Sequence[Message] = (),
        conversation_id: str | None = None,
        event_handler: AgentEventHandler | None = None,
        summary_state: ConversationSummaryState | None = None,
        tool_result_views: Sequence[ToolResultView] = (),
        run_id: str | None = None,
        recovery_run_id: str | None = None,
        mode: AgentMode = AgentMode.NORMAL,
        tool_context_metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:

        """启动流式运行，按生成顺序输出事件并在结束时清理事件通道。"""
        queue_handler = _QueueEventHandler()
        handler: AgentEventHandler = queue_handler
        if event_handler is not None:
            handler = CompositeEventHandler(queue_handler, event_handler)

        # 函数说明：AgentRuntime.run_stream.execute
        # 用途：执行AgentRuntime，供模型与工具执行循环使用。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.run` →
        # `queue_handler.finish`。
        # 闭包依赖：从外层读取 `conversation_id`、`handler`、`history`、`mode`、
        # `queue_handler`、`recovery_run_id`、`run_id`、`summary_state`、
        # `tool_context_metadata`、`tool_result_views`。
        async def execute() -> None:
            try:
                await self.run(
                    user_input,
                    history=history,
                    conversation_id=conversation_id,
                    event_handler=handler,
                    summary_state=summary_state,
                    tool_result_views=tool_result_views,
                    run_id=run_id,
                    recovery_run_id=recovery_run_id,
                    mode=mode,
                    tool_context_metadata=tool_context_metadata,
                )
            finally:
                await queue_handler.finish()

        task = asyncio.create_task(execute())
        try:
            while True:
                item = await queue_handler.next()
                if item is _STREAM_FINISHED:
                    break
                if isinstance(item, AgentEvent):
                    yield item
            await task
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
