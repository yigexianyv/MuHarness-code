
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.domain.skills import SKILL_READ_TOOL_NAME
from app.models.types import AgentMode, Message, MessageRole, ToolCall, ToolResult
from app.runtime.checkpoint import SQLiteCheckpointStore
from app.tools.catalog import TOOL_SEARCH_NAME, activated_tool_names
from app.tools.executor import ToolExecutor
from app.tools.hooks import ToolExecutionContext, ToolHook
from app.tools.registry import ToolRegistry
from app.tools.role_boundary import is_role_mode, role_rejection_message

from .context_session import RuntimeContextSession
from .errors import RepeatedToolCallError
from .event_stream import EventEmitter
from .result import ToolCallRecord
from .runtime_helpers import plan_task_id_from_output, tool_call_signature


@dataclass(frozen=True, slots=True)
class ToolRoundOutcome:

    records: tuple[ToolCallRecord, ...]
    result_messages: tuple[Message, ...]
    pending_activations: frozenset[str]
    previous_signature: str | None
    repeated_count: int
    plan_task_created: bool
    plan_task_id: str | None
    repeated_error: RepeatedToolCallError | None = None


class ToolRoundExecutor:

    # 函数说明：ToolRoundExecutor.__init__
    # 用途：初始化 ToolRoundExecutor；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
    #   executor：执行者输入或配置值，类型 `ToolExecutor`。
    #   checkpoint_store：运行检查点存储，类型 `SQLiteCheckpointStore | None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._registry`、`self._executor`、`self._checkpoint_store`。
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        executor: ToolExecutor,
        checkpoint_store: SQLiteCheckpointStore | None,
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._checkpoint_store = checkpoint_store

    # 函数说明：ToolRoundExecutor.execute
    # 用途：执行当前模型响应中的工具调用，保存检查点并汇总可回传给模型的结果。
    # 参数：
    #   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`。
    #   run_id：目标运行标识，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   user_input：本次用户输入，类型 `str`。
    #   step：当前任务步骤，类型 `int`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   round_index：当前执行轮次索引，类型 `int`。
    #   closing_can_deliver：`closing_can_deliver`输入或配置值，类型 `bool`。
    #   activated_tools：工具集合输入或配置值，类型 `set[str]`。
    #   context_session：上下文输入或配置值，类型 `RuntimeContextSession`。
    #   previous_signature：`previous_signature`输入或配置值，类型 `str | None`。
    #   repeated_count：`repeated_count`输入或配置值，类型 `int`。
    #   emitter：执行事件发射器，类型 `EventEmitter`。
    #   hook：`hook`输入或配置值，类型 `ToolHook`。
    #   tool_context_metadata：本次工具执行关联的元数据，类型 `Mapping[str, Any] | None`
    # ；默认 `None`。
    # 返回：类型 `ToolRoundOutcome`；返回 `ToolRoundOutcome(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._checkpoint_store.before_tools` → `tool_call_signature` → `ToolRoundOutcome`
    #  → `frozenset` → `ToolExecutionContext` → `self._execute_one`；另有 8 个调用点。
    # 分支与异常：
    #   当 `repeated_count >= 3` 时，返回 `ToolRoundOutcome(…)`。
    async def execute(
        self,
        tool_calls: tuple[ToolCall, ...],
        *,
        run_id: str,
        conversation_id: str | None,
        user_input: str,
        step: int,
        mode: AgentMode,
        round_index: int,
        closing_can_deliver: bool,
        activated_tools: set[str],
        context_session: RuntimeContextSession,
        previous_signature: str | None,
        repeated_count: int,
        emitter: EventEmitter,
        hook: ToolHook,
        tool_context_metadata: Mapping[str, Any] | None = None,
    ) -> ToolRoundOutcome:

        """执行当前模型响应中的工具调用，保存检查点并汇总可回传给模型的结果。"""
        records: list[ToolCallRecord] = []
        result_messages: list[Message] = []
        pending_activations: set[str] = set()
        plan_task_created = False
        plan_task_id: str | None = None

        if self._checkpoint_store is not None:
            await self._checkpoint_store.before_tools(
                run_id,
                step=step,
                tool_calls=tool_calls,
            )

        for tool_call in tool_calls:
            signature = tool_call_signature(tool_call)
            if signature == previous_signature:
                repeated_count += 1
            else:
                previous_signature = signature
                repeated_count = 1
            if repeated_count >= 3:
                return ToolRoundOutcome(
                    records=tuple(records),
                    result_messages=tuple(result_messages),
                    pending_activations=frozenset(pending_activations),
                    previous_signature=previous_signature,
                    repeated_count=repeated_count,
                    plan_task_created=plan_task_created,
                    plan_task_id=plan_task_id,
                    repeated_error=RepeatedToolCallError(tool_call.name),
                )

            context = ToolExecutionContext(
                run_id=run_id,
                conversation_id=conversation_id,
                user_input=user_input,
                step=step,
                tool_call=tool_call,
                metadata={
                    **dict(tool_context_metadata or {}),
                    "active_skill_names": context_session.active_skill_names,
                },
                mode=mode,
            )
            result = await self._execute_one(
                tool_call,
                context=context,
                hook=hook,
                mode=mode,
                closing_can_deliver=closing_can_deliver,
                activated_tools=activated_tools,
            )
            if self._checkpoint_store is not None:
                await self._checkpoint_store.complete_tool(run_id, result)

            if (
                mode is AgentMode.PLAN
                and result.success
                and tool_call.name in ("task_create", "task_update")
            ):
                plan_task_created = True
                task_id = plan_task_id_from_output(result.output)
                if task_id:
                    plan_task_id = task_id

            records.append(
                ToolCallRecord(
                    round_index=round_index,
                    tool_call=tool_call,
                    result=result,
                )
            )
            result_messages.append(self._result_message(result))

            if tool_call.name == TOOL_SEARCH_NAME and result.success:
                pending_activations.update(
                    name
                    for name in activated_tool_names(result.output)
                    if self._registry.is_deferred(name)
                )
            if tool_call.name == SKILL_READ_TOOL_NAME and result.success:
                await context_session.activate_skill(
                    result,
                    emitter=emitter,
                    step=step,
                )

        return ToolRoundOutcome(
            records=tuple(records),
            result_messages=tuple(result_messages),
            pending_activations=frozenset(pending_activations),
            previous_signature=previous_signature,
            repeated_count=repeated_count,
            plan_task_created=plan_task_created,
            plan_task_id=plan_task_id,
        )

    # 函数说明：ToolRoundExecutor._execute_one
    # 用途：执行单个工具调用，处理重复调用保护、技能激活和结果事件。
    # 参数：
    #   tool_call：单次结构化工具调用，类型 `ToolCall`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   hook：`hook`输入或配置值，类型 `ToolHook`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   closing_can_deliver：`closing_can_deliver`输入或配置值，类型 `bool`。
    #   activated_tools：工具集合输入或配置值，类型 `set[str]`。
    # 返回：类型 `ToolResult`；按分支返回 `result`；
    # `await self._executor.execute(tool_call, context=context, hooks=(hook,))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._rejection_reason` →
    # `hook.before_execute` → `ToolResult` → `hook.after_execute` →
    # `self._executor.execute`。
    # 分支与异常：
    #   `rejection is not None` 分支在完成前置处理后返回 `result`。
    #   捕获 `Exception` 后，返回 `result`。
    async def _execute_one(
        self,
        tool_call: ToolCall,
        *,
        context: ToolExecutionContext,
        hook: ToolHook,
        mode: AgentMode,
        closing_can_deliver: bool,
        activated_tools: set[str],
    ) -> ToolResult:

        """执行单个工具调用，处理重复调用保护、技能激活和结果事件。"""
        rejection = self._rejection_reason(
            tool_call,
            mode=mode,
            closing_can_deliver=closing_can_deliver,
            activated_tools=activated_tools,
        )
        if rejection is not None:
            await hook.before_execute(context)
            result = ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                error=rejection,
                duration_ms=0,
            )
            await hook.after_execute(context, result)
            return result
        try:
            return await self._executor.execute(
                tool_call,
                context=context,
                hooks=(hook,),
            )
        except Exception as exc:
            result = ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=0.0,
            )
            await hook.after_execute(context, result)
            return result

    # 函数说明：ToolRoundExecutor._rejection_reason
    # 用途：在模型与工具执行循环中处理 `_rejection_reason`，通过
    # `self._registry.is_allowed_for_mode` 完成首个内部处理步骤。
    # 参数：
    #   tool_call：单次结构化工具调用，类型 `ToolCall`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   closing_can_deliver：`closing_can_deliver`输入或配置值，类型 `bool`。
    #   activated_tools：工具集合输入或配置值，类型 `set[str]`。
    # 返回：类型 `str | None`；按分支返回 `role_rejection_message(mode, tool_call.name)`
    # ；`'Tool is not allowed during budget closing (delivery tools only).'`；
    # `'Tool is not allowed in plan mode (read-only / planning tools only).'`；
    # `'Deferred tool is not active. Call tool_search first.'` 等 5 种表达式。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`is_role_mode` →
    # `self._registry.is_allowed_for_mode` → `role_rejection_message` →
    # `self._registry.is_allowed_during_closing` → `self._registry.is_deferred` →
    # `self._registry.is_available_for_mode`。
    # 分支与异常：
    #   当 `is_role_mode(mode) and (not…` 时，返回
    # `role_rejection_message(mode, tool_call.name)`。
    #   当 `closing_can_deliver and (not (…` 时，返回
    # `'Tool is not allowed during budget closing (delivery tools…`。
    #   当 `mode is not AgentMode.NORMAL and (not…` 时，返回
    # `'Tool is not allowed in plan mode (read-only / planning…`。
    #   当 `not self._registry.is_available_for_mode(tool_call.name,…` 时，返回
    # `'Deferred tool is not active. Call tool_search first.'`。
    def _rejection_reason(
        self,
        tool_call: ToolCall,
        *,
        mode: AgentMode,
        closing_can_deliver: bool,
        activated_tools: set[str],
    ) -> str | None:
        if is_role_mode(mode) and not self._registry.is_allowed_for_mode(
            tool_call.name,
            mode,
        ):
            return role_rejection_message(mode, tool_call.name)
        if closing_can_deliver and not (
            self._registry.is_allowed_during_closing(tool_call.name, mode)
            and (
                not self._registry.is_deferred(tool_call.name)
                or tool_call.name in activated_tools
            )
        ):
            return (
                "Tool is not allowed during budget closing "
                "(delivery tools only)."
            )
        if mode is not AgentMode.NORMAL and not self._registry.is_allowed_for_mode(
            tool_call.name,
            mode,
        ):
            return (
                "Tool is not allowed in plan mode "
                "(read-only / planning tools only)."
            )
        if not self._registry.is_available_for_mode(
            tool_call.name,
            mode,
            activated_names=activated_tools,
        ):
            return "Deferred tool is not active. Call tool_search first."
        return None

    # 函数说明：ToolRoundExecutor._result_message
    # 用途：返回 `Message(…)`，提供 ToolRoundExecutor 的派生值。
    # 参数：
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    # 返回：类型 `Message`；返回 `Message(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` →
    # `result.model_dump_json`。
    @staticmethod
    def _result_message(result: ToolResult) -> Message:
        return Message(
            role=MessageRole.TOOL,
            name=result.tool_name,
            tool_call_id=result.tool_call_id,
            content=result.model_dump_json(exclude_none=True),
        )


__all__ = ["ToolRoundExecutor", "ToolRoundOutcome"]
