from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.domain.memory import (
    MemoryManager,
    MemoryRecallQueryInputs,
    recent_user_message_texts,
)
from app.domain.skills import SkillContextProvider, SkillStore
from app.domain.task.context import TaskContextProvider
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    AgentMode,
    Message,
    MessageRole,
    ModelProvider,
    ModelRequest,
    ModelUsage,
)
from app.runtime.checkpoint import RunCheckpoint, SQLiteCheckpointStore
from app.runtime.context import (
    ContextManager,
    ConversationSummaryState,
)
from app.runtime.context.tool_views import (
    ToolResultView,
    ToolResultViewState,
    tool_output_excerpts,
)
from app.tools.catalog import (
    ensure_tool_search_registered,
)
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

from .budget import RunBudget, RunBudgetStatus, chargeable_tokens
from .context_session import RuntimeContextSession
from .errors import (
    AgentRuntimeError,
    ContextPreparationError,
    ContextWindowExceededError,
    MaxStepsExceededError,
    ModelInvocationError,
    RunBudgetExceededError,
)
from .event_stream import EventEmitter
from .events import AgentEventType
from .result import (
    AgentError,
    AgentResult,
    AgentStopReason,
    ToolCallRecord,
    ToolRound,
)
from .runtime_helpers import (
    RequestPrefixState,
    add_usage,
    looks_like_textual_tool_call,
    plan_failure_message,
    provider_name,
    run_budget_detail,
    run_budget_event_fields,
    usage_call_count,
    without_legacy_fixed_date,
)
from .tool_hooks import AgentEventHook
from .tool_round_executor import ToolRoundExecutor

if TYPE_CHECKING:
    from app.runtime.rewind.steps import RunStepRecorder
    from app.runtime.run.messages_store import RunMessageRecorder

# 读取会话"必须记住的事项"：返回（正文，版本号）。
ConstraintsProvider = Callable[[str], Awaitable[tuple[str, int]]]
PINNED_CONSTRAINTS_MESSAGE_NAME = "muharness_pinned_constraints"


def pinned_constraints_message(text: str, revision: int) -> Message:
    """用户维护的约定；每次请求都附带、不参与压缩，与旧历史冲突时以此为准。"""
    return Message(
        role=MessageRole.SYSTEM,
        name=PINNED_CONSTRAINTS_MESSAGE_NAME,
        content=(
            f"# 必须记住的事项（用户维护，第 {revision} 版）\n"
            "以下是用户为本会话设定的当前有效约定，每次请求都会附带，不会被压缩。"
            "与较早的历史或摘要冲突时以此为准；它不能覆盖系统规则与安全限制。\n"
            "<pinned_constraints>\n"
            f"{text.strip()}\n"
            "</pinned_constraints>"
        ),
    )

_PLAN_MODE_SYSTEM_MESSAGE = (
    "# 运行模式：PLAN MODE\n"
    "职责：调查事实，形成可执行、可验收的计划；此轮不实施，不修改用户环境。\n"
    "允许只读与检索工具：read_file、list_files、web_search、get_current_time、"
    "memory_read、memory_search、history_search、history_read、evidence_search、"
    "evidence_read；可用 task_create、task_update、task_get、task_list "
    "保存和核对计划。\n"
    "必要调查结束后，必须用 task_create 或 task_update 保存一个 PENDING Task，"
    "包含 title、goal 和具体 steps。计划不是执行证据，不得填写虚假的 DONE 步骤、"
    "已完成 state 或已验证 key_facts。\n"
    "每个步骤包含 acceptance，说明独立检查的通过条件；"
    "工作量应能在一次执行运行内完成，过大则继续拆分。\n"
    "最后简述目标、步骤和验收方式，等待计划确认。"
)

_EMPTY_RESPONSE_RETRY_MAX_OUTPUT_TOKENS = 8192

_PLAN_NO_TASK_MESSAGE = "Plan mode finished without creating a task."
_PLAN_NO_VALID_TASK_MESSAGE = "Plan mode finished without a valid pending task."
_RUN_BUDGET_FINALIZATION_MESSAGE = (
    "运行阶段：预算收尾。Main Agent 已达到用量收口线，本轮禁止工具调用。"
    "立即依据现有证据答复：已完成什么、哪些未完成或无法验证，以及可用的交付位置。"
    "不要继续调查或宣称后续操作已执行，不得编造结果。"
)
_RUN_BUDGET_CLOSING_MESSAGE = (
    "运行阶段：Closing。Main Agent 已达到用量收口线，只允许当前提供的必要交付工具。"
    "停止搜索、调查和范围扩展。已有结果需要落盘、发布或 task_update 写回时，"
    "利用这一次交付机会完成；无待交付结果则直接答复。只报告成功回执确认的结果，"
    "同时说明未完成项。"
)
_RUN_BUDGET_WARNING_MESSAGE = (
    "运行阶段：预算预警。累计 Main Agent 用量接近收尾区，优先处理当前目标的"
    "关键缺口、验证和交付，复用已有事实并减少重复调用。必要工具仍可使用；"
    "不要因预警跳过必需验证或提前宣称完成。"
)
_TOOL_ROUND_LIMIT_FALLBACK_MESSAGE = (
    "本次运行已达到工具轮次上限，工具执行已停止。已取得的结果保存在运行记录中，"
    "但模型未能生成可靠的最终总结，因此不能据此认定任务完成。需要继续时，"
    "请结合这些记录明确下一步要求。"
)
_EMPTY_FINAL_RETRY_MESSAGE = (
    "响应修正：上一条响应既无可展示文本，也无工具调用。请给出有效的下一步："
    "尚有必要工作且当前模式允许时，发起所需工具调用；已完成或无法继续时，"
    "依据证据直接答复并说明缺口。不要重复分析或仅输出内部思考。"
)
_TEXTUAL_TOOL_CALL_RETRY_MESSAGE = (
    "响应修正：普通文本里的调用协议不会被执行。需要工具时使用 Provider 的"
    "结构化 tool_calls，并遵守当前模式和工具权限；无需工具时输出完整的用户答复。"
    "不要在正文中输出 DSML、XML 或其他伪工具协议，也不要把这些文本当成执行记录。"
)


class AgentLoop:
    # 函数说明：AgentLoop.__init__
    # 用途：初始化 AgentLoop；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   model_registry：模型适配器注册表，类型 `ModelAdapterRegistry`。
    #   tool_registry：工具注册表，类型 `ToolRegistry`。
    #   tool_executor：工具执行器，类型 `ToolExecutor`。
    #   provider：模型或搜索服务商，类型 `ModelProvider | str | None`。
    #   model：模型名称，类型 `str | None`。
    #   system_prompt：系统提示文本，类型 `str | None`。
    #   max_steps：模型执行步数上限，类型 `int`。
    #   max_tool_rounds：工具调用轮次上限，类型 `int | None`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`。
    #   context_manager：上下文预算与压缩管理器，类型 `ContextManager`。
    #   task_context_provider：任务状态上下文提供器，类型 `TaskContextProvider | None`。
    #   checkpoint_store：运行检查点存储，类型 `SQLiteCheckpointStore | None`。
    #   memory_manager：长期记忆管理器，类型 `MemoryManager | None`。
    #   skill_store：技能存储，类型 `SkillStore | None`。
    #   skill_context_provider：已激活技能上下文提供器，类型
    # `SkillContextProvider | None`。
    #   run_budget：本轮累计运行预算配置，类型 `RunBudget`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRoundExecutor`。
    # 副作用与资源：
    #   更新对象字段：`self._model_registry`、`self._tool_registry`、
    # `self._tool_executor`、`self._provider`、`self._model`、`self._system_prompt`、
    # `self._max_steps`、`self._max_tool_rounds` 等 17 个字段。
    def __init__(
        self,
        *,
        model_registry: ModelAdapterRegistry,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        provider: ModelProvider | str | None,
        model: str | None,
        system_prompt: str | None,
        max_steps: int,
        max_tool_rounds: int | None,
        max_output_tokens: int | None,
        context_manager: ContextManager,
        task_context_provider: TaskContextProvider | None,
        checkpoint_store: SQLiteCheckpointStore | None,
        memory_manager: MemoryManager | None,
        skill_store: SkillStore | None,
        skill_context_provider: SkillContextProvider | None,
        run_budget: RunBudget,
        constraints_provider: ConstraintsProvider | None = None,
    ) -> None:
        self._model_registry = model_registry
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._provider = provider
        self._model = model
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._max_tool_rounds = max_tool_rounds
        self._max_output_tokens = max_output_tokens
        self._context_manager = context_manager
        self._task_context_provider = task_context_provider
        self._checkpoint_store = checkpoint_store
        self._memory_manager = memory_manager
        self._skill_store = skill_store
        self._skill_context_provider = skill_context_provider
        self._run_budget = run_budget
        self._constraints_provider = constraints_provider
        self._tool_round_executor = ToolRoundExecutor(
            registry=tool_registry,
            executor=tool_executor,
            checkpoint_store=checkpoint_store,
        )

    # 函数说明：AgentLoop.run
    # 用途：驱动一次完整模型—工具循环，维护原始历史、固定模型视图、运行预算和检查点，直
    # 到答复或受控终止。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   user_input：本次用户输入，类型 `str`。
    #   history：原始会话历史，类型 `Sequence[Message]`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   emitter：执行事件发射器，类型 `EventEmitter`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`。
    #   recovery_checkpoint：恢复信息检查点输入或配置值，类型 `RunCheckpoint | None`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   tool_context_metadata：本次工具执行关联的元数据，类型 `Mapping[str, Any] | None`
    # ；默认 `None`。
    #   tool_result_views：工具结果的固定模型视图，类型 `Sequence[ToolResultView]`；默认
    #  `()`。
    # 返回：类型 `AgentResult`；按分支返回 `await stop_with_error(…)`；
    # `stop_at_tool_round_limit(step=step)`；`self._result(…)`。
    # 设计约束：原始会话与给模型的投影视图分开维护；普通轮次追加消息，压缩边界才允许重建
    # 请求。
    # 设计约束：模型输出普通文本形式的伪工具调用不会直接执行；工具权限、预算和模式仍由执
    # 行层约束。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentEventHook` → `Message` →
    # `without_legacy_fixed_date` → `ToolResultViewState` → `ModelUsage` →
    # `RuntimeContextSession`；另有 35 个调用点。
    # 分支与异常：
    #   当 `finalization_step and (not finalization_pending) and (not…` 时，结束当前循环
    # 。
    #   `budget_decision.exceeded` 分支在完成前置处理后返回 `await stop_with_error(…)`。
    #   `budget_closing_reporting_attempted` 分支在完成前置处理后返回
    # `await stop_with_error(…)`。
    #   捕获 `Exception` 后，返回 `await stop_with_error(…)`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def run(
        self,
        run_id: str,
        user_input: str,
        *,
        history: Sequence[Message],
        conversation_id: str | None,
        emitter: EventEmitter,
        summary_state: ConversationSummaryState | None,
        recovery_checkpoint: RunCheckpoint | None,
        mode: AgentMode,
        tool_context_metadata: Mapping[str, Any] | None = None,
        tool_result_views: Sequence[ToolResultView] = (),
        message_recorder: RunMessageRecorder | None = None,
        step_recorder: RunStepRecorder | None = None,
    ) -> AgentResult:
        """驱动模型与工具；原始历史保持完整，模型视图只在压缩边界重建。"""
        tool_event_hook = AgentEventHook(emitter)
        user_message = Message(role=MessageRole.USER, content=user_input)
        historical_message_count = len(history)
        messages = [*history, user_message]
        transformed_history = tuple(
            without_legacy_fixed_date(message) for message in history
        )
        request_system_message = (
            Message(role=MessageRole.SYSTEM, content=self._system_prompt)
            if self._system_prompt is not None
            and not any(
                message.role is MessageRole.SYSTEM
                and message.content == self._system_prompt
                for message in transformed_history
            )
            else None
        )
        request_historical_message_count = historical_message_count + (
            1 if request_system_message is not None else 0
        )
        tool_view_state = ToolResultViewState(
            history,
            tool_result_views,
            **self._context_manager.tool_view_limits,
        )
        previous_signature: str | None = None
        repeated_count = 0
        tool_rounds: list[ToolRound] = []
        tool_calls: list[ToolCallRecord] = []
        usage = ModelUsage()
        main_model_calls = 0
        budget_chargeable_tokens = 0
        current_summary_state = summary_state
        context_session = RuntimeContextSession(
            memory_manager=self._memory_manager,
            skill_store=self._skill_store,
            skill_context_provider=self._skill_context_provider,
            task_context_provider=self._task_context_provider,
            recall_query=(
                MemoryRecallQueryInputs(
                    user_message=user_input,
                    recent_user_messages=recent_user_message_texts(history),
                    summary_objective=(
                        summary_state.summary.current_objective
                        if summary_state is not None
                        and summary_state.summary.current_objective
                        else None
                    ),
                )
                if self._memory_manager is not None
                else None
            ),
        )
        activated_tools: set[str] = set()
        reported_excerpt_ids: set[str] = set()
        ensure_tool_search_registered(self._tool_registry)
        plan_task_created = False
        plan_task_id: str | None = None
        finalization_pending = False
        budget_warning_emitted = False
        budget_closing_started = False
        budget_closing_delivery_used = False
        budget_closing_reporting_attempted = False
        audit_report_recovery = False
        audit_report_recovery_used = False
        empty_final_retry_used = False
        retry_max_output_tokens: int | None = None
        textual_tool_call_retry_used = False
        response_repair_message: Message | None = None
        request_prefix_state: RequestPrefixState | None = None
        failed_compaction_source: tuple[Message, ...] | None = None
        pinned_constraints: tuple[str, int] | None = None
        summary_baseline_emitted = False

        async def record_messages() -> None:
            if message_recorder is not None:
                await message_recorder.sync(messages)

        async def load_pinned_constraints() -> tuple[str, int] | None:
            # 读取失败时沿用上一次读到的版本，不中断运行。
            if self._constraints_provider is None or conversation_id is None:
                return None
            try:
                return await self._constraints_provider(conversation_id)
            except Exception:
                return pinned_constraints

        await emitter.emit(
            AgentEventType.AGENT_STARTED,
            message=user_message,
            provider=provider_name(self._provider),
            model=self._model,
        )

        # 函数说明：AgentLoop.run.stop_with_error
        # 用途：停止错误，供模型与工具执行循环使用。
        # 参数：
        #   error：异常或错误信息，类型 `AgentRuntimeError`。
        #   stop_reason：运行终止原因，类型 `AgentStopReason`。
        #   step：当前任务步骤，类型 `int`。
        # 返回：类型 `AgentResult`；返回 `self._result(…)`。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._result` →
        # `self._error_message` → `tool_view_state.snapshot`。
        # 闭包依赖：从外层读取 `current_summary_state`、`messages`、`run_id`、
        # `tool_calls`、`tool_rounds`、`tool_view_state`、`usage`。
        async def stop_with_error(
            error: AgentRuntimeError,
            stop_reason: AgentStopReason,
            *,
            step: int,
        ) -> AgentResult:

            return self._result(
                run_id=run_id,
                final_message=self._error_message(error),
                messages=messages,
                steps=step,
                stop_reason=stop_reason,
                tool_rounds=tool_rounds,
                tool_calls=tool_calls,
                usage=usage,
                error=error,
                summary_state=current_summary_state,
                tool_result_views=tool_view_state.snapshot(messages),
            )

        # 函数说明：AgentLoop.run.stop_at_tool_round_limit
        # 用途：停止工具上限，供模型与工具执行循环使用。
        # 参数：
        #   step：当前任务步骤，类型 `int`。
        # 返回：类型 `AgentResult`；返回 `self._result(…)`。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `self._result` →
        # `tool_view_state.snapshot`。
        # 闭包依赖：从外层读取 `current_summary_state`、`messages`、`run_id`、
        # `tool_calls`、`tool_rounds`、`tool_view_state`、`usage`。
        def stop_at_tool_round_limit(*, step: int) -> AgentResult:

            final_message = Message(
                role=MessageRole.ASSISTANT,
                content=_TOOL_ROUND_LIMIT_FALLBACK_MESSAGE,
            )
            messages[-1] = final_message
            return self._result(
                run_id=run_id,
                final_message=final_message,
                messages=messages,
                steps=step,
                stop_reason=AgentStopReason.FINAL_ANSWER,
                tool_rounds=tool_rounds,
                tool_calls=tool_calls,
                usage=usage,
                summary_state=current_summary_state,
                tool_result_views=tool_view_state.snapshot(messages),
            )

        for step in range(1, self._max_steps + 2):
            finalization_step = step > self._max_steps
            if (
                finalization_step
                and not finalization_pending
                and not audit_report_recovery
            ):
                break
            budget_decision = self._run_budget.evaluate(
                usage,
                chargeable_tokens_override=budget_chargeable_tokens,
                model_calls_override=main_model_calls,
            )
            budget_warning_in_request = budget_decision.should_warn
            budget_config = self._run_budget.config
            if budget_decision.exceeded:
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_EXCEEDED,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )
                return await stop_with_error(
                    RunBudgetExceededError(run_budget_detail(budget_decision)),
                    AgentStopReason.RUN_BUDGET,
                    step=max(0, step - 1),
                )
            if budget_decision.should_warn and not budget_warning_emitted:
                budget_warning_emitted = True
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_WARNING,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )
            budget_forces_final = budget_decision.should_finalize
            if budget_forces_final:
                if budget_closing_delivery_used:
                    if budget_closing_reporting_attempted:
                        await emitter.emit(
                            AgentEventType.RUN_BUDGET_EXCEEDED,
                            step=step,
                            **run_budget_event_fields(
                                budget_decision,
                                budget_config,
                                status=RunBudgetStatus.EXCEEDED,
                            ),
                        )
                        return await stop_with_error(
                            RunBudgetExceededError(
                                "dedicated closing report call was already used"
                            ),
                            AgentStopReason.RUN_BUDGET,
                            step=max(0, step - 1),
                        )
                    budget_closing_reporting_attempted = True
                elif not budget_closing_started:
                    budget_closing_started = True
                    await emitter.emit(
                        AgentEventType.RUN_BUDGET_FINALIZING,
                        step=step,
                        **run_budget_event_fields(budget_decision, budget_config),
                    )
            await record_messages()
            if self._checkpoint_store is not None:
                await self._checkpoint_store.before_model(run_id, step=step)
            tool_round_limit_reached = (
                self._max_tool_rounds is not None
                and len(tool_rounds) >= self._max_tool_rounds
            )
            forced_without_budget = (
                finalization_step or tool_round_limit_reached or audit_report_recovery
            )
            closing_can_deliver = (
                budget_forces_final
                and not budget_closing_delivery_used
                and not forced_without_budget
            )
            force_final_answer = forced_without_budget or (
                budget_forces_final and not closing_can_deliver
            )
            raw_source_messages = tuple(messages)
            view_messages = tool_view_state.project(raw_source_messages)
            # 只报本次运行执行的工具；继承的历史结果在它们自己的运行里已经报过
            run_tool_call_ids = {record.tool_call.id for record in tool_calls}
            new_excerpts = tuple(
                excerpt
                for excerpt in tool_output_excerpts(raw_source_messages, view_messages)
                if excerpt.tool_call_id in run_tool_call_ids
                and excerpt.tool_call_id not in reported_excerpt_ids
            )
            reported_excerpt_ids.update(item.tool_call_id for item in new_excerpts)
            projected_messages = tuple(
                without_legacy_fixed_date(message) for message in view_messages
            )
            source_messages = (
                (request_system_message, *projected_messages)
                if request_system_message is not None
                else projected_messages
            )
            request_messages = source_messages
            if closing_can_deliver:
                request_tools = self._tool_registry.closing_definitions_for_mode(
                    mode,
                    activated_names=activated_tools,
                )
                if not request_tools:
                    closing_can_deliver = False
                    force_final_answer = True
            elif force_final_answer:
                request_tools = ()
            else:
                request_tools = self._tool_registry.model_definitions_for_mode(
                    mode,
                    activated_names=activated_tools,
                )
            try:
                adapter = self._model_registry.get(self._provider)
                resolved_model = self._model or adapter.default_model
                resolved_provider = adapter.provider
                effective_max_output_tokens = (
                    self._max_output_tokens or adapter.config.default_max_output_tokens
                )
                if retry_max_output_tokens is not None:
                    effective_max_output_tokens = retry_max_output_tokens
                    retry_max_output_tokens = None
                if budget_forces_final and force_final_answer:
                    effective_max_output_tokens = min(
                        effective_max_output_tokens,
                        budget_config.finalization_max_output_tokens,
                    )
            except Exception as exc:
                return await stop_with_error(
                    ModelInvocationError(f"{type(exc).__name__}: {exc}"),
                    AgentStopReason.MODEL_ERROR,
                    step=step,
                )

            try:
                trailing_system_messages: list[Message] = []
                if mode is AgentMode.PLAN:
                    trailing_system_messages.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            name="muharness_plan_mode",
                            content=_PLAN_MODE_SYSTEM_MESSAGE,
                        )
                    )
                if budget_warning_in_request:
                    trailing_system_messages.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=_RUN_BUDGET_WARNING_MESSAGE,
                        )
                    )
                context_injection = await context_session.build(
                    conversation_id=conversation_id,
                    recovery_checkpoint=recovery_checkpoint,
                    trailing_system_messages=tuple(trailing_system_messages),
                )
                context_messages = context_injection.messages
                pinned_constraints = await load_pinned_constraints()
                if pinned_constraints is not None and pinned_constraints[0].strip():
                    context_messages = (
                        pinned_constraints_message(*pinned_constraints),
                        *context_messages,
                    )
                if context_messages:
                    request_messages = (
                        *request_messages[:request_historical_message_count],
                        *context_messages,
                        *request_messages[request_historical_message_count:],
                    )
                if closing_can_deliver:
                    request_messages = (
                        *request_messages,
                        Message(
                            role=MessageRole.SYSTEM,
                            content=_RUN_BUDGET_CLOSING_MESSAGE,
                        ),
                    )
                elif force_final_answer:
                    if budget_forces_final:
                        final_instruction = _RUN_BUDGET_FINALIZATION_MESSAGE
                    else:
                        final_instruction = (
                            "运行阶段：工具轮次用尽。禁止继续调用工具；依据最后的"
                            "实际结果直接答复，明确已完成、未完成与无法验证的内容。"
                        )
                    request_messages = (
                        *request_messages,
                        Message(
                            role=MessageRole.SYSTEM,
                            content=final_instruction,
                        ),
                    )
                if response_repair_message is not None:
                    request_messages = (*request_messages, response_repair_message)
            except Exception as exc:
                return await stop_with_error(
                    ContextPreparationError(f"{type(exc).__name__}: {exc}"),
                    AgentStopReason.CONTEXT_ERROR,
                    step=step,
                )
            prefix_decision = "rebuild"
            prefix_rebuild_reason = "initial_request"
            request_config = (
                resolved_provider,
                resolved_model,
                effective_max_output_tokens,
            )

            continuation_messages = (
                request_prefix_state.extend(
                    source_messages=source_messages,
                    context_messages=context_messages,
                    tools=request_tools,
                    request_config=request_config,
                )
                if request_prefix_state is not None
                else None
            )
            if continuation_messages is not None:
                # Closing/repair instructions are generated, never canonical
                # messages, and are included in the final budget calculation.
                if closing_can_deliver or force_final_answer:
                    continuation_messages = (
                        *continuation_messages,
                        Message(
                            role=MessageRole.SYSTEM,
                            content=(
                                _RUN_BUDGET_CLOSING_MESSAGE
                                if closing_can_deliver
                                else final_instruction
                            ),
                        ),
                    )
                if response_repair_message is not None:
                    continuation_messages = (
                        *continuation_messages,
                        response_repair_message,
                    )
                prefix_decision = "append"
                prefix_rebuild_reason = None
            elif request_prefix_state is not None:
                prefix_rebuild_reason = (
                    "tool_schema_changed"
                    if request_tools != request_prefix_state.tools
                    else "request_config_changed"
                    if request_config != request_prefix_state.request_config
                    else "instruction_or_source_changed"
                )
            cache_prefix_reused = continuation_messages is not None
            cache_prefix_message_count = (
                len(request_prefix_state.sent_messages)
                if cache_prefix_reused and request_prefix_state is not None
                else 0
            )
            context_input_messages = continuation_messages or request_messages
            summary_state_before = current_summary_state
            try:
                context_decision = await self._context_manager.prepare(
                    context_input_messages,
                    tools=request_tools,
                    model=resolved_model,
                    provider=resolved_provider,
                    max_output_tokens=effective_max_output_tokens,
                    source_messages=raw_source_messages,
                    protected_source_indices=(historical_message_count,),
                    summary_state=current_summary_state,
                    allow_compaction=raw_source_messages != failed_compaction_source,
                )
                if context_decision.summary_updated:
                    prefix_decision = "compact"
                    prefix_rebuild_reason = "context_pressure"
                    failed_compaction_source = None
                elif context_decision.summary_error:
                    failed_compaction_source = raw_source_messages
            except Exception as exc:
                return await stop_with_error(
                    ContextPreparationError(f"{type(exc).__name__}: {exc}"),
                    AgentStopReason.CONTEXT_ERROR,
                    step=step,
                )

            if request_prefix_state is not None:
                previous_prefix = request_prefix_state.sent_messages
                cache_prefix_reused = (
                    request_tools == request_prefix_state.tools
                    and context_decision.messages[: len(previous_prefix)]
                    == previous_prefix
                )
                if not cache_prefix_reused:
                    cache_prefix_message_count = 0

            current_summary_state = context_decision.summary_state
            usage = add_usage(usage, context_decision.summary_usage)
            main_model_calls += usage_call_count(context_decision.summary_usage)
            budget_chargeable_tokens += chargeable_tokens(
                context_decision.summary_usage
            )
            request_messages = context_decision.messages
            request_tools = context_decision.tools
            prepared_budget_decision = self._run_budget.evaluate(
                usage,
                chargeable_tokens_override=budget_chargeable_tokens,
                model_calls_override=main_model_calls,
            )
            budget_decision = prepared_budget_decision
            if budget_decision.exceeded:
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_EXCEEDED,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )
                return await stop_with_error(
                    RunBudgetExceededError(run_budget_detail(budget_decision)),
                    AgentStopReason.RUN_BUDGET,
                    step=max(0, step - 1),
                )
            if budget_decision.should_warn and not budget_warning_emitted:
                budget_warning_emitted = True
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_WARNING,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )
            if budget_decision.should_warn and not budget_warning_in_request:
                request_messages = (
                    *request_messages,
                    Message(
                        role=MessageRole.SYSTEM,
                        content=_RUN_BUDGET_WARNING_MESSAGE,
                    ),
                )
            if budget_decision.should_finalize and not budget_forces_final:
                budget_forces_final = True
                budget_closing_started = True
                closing_can_deliver = not forced_without_budget
                if closing_can_deliver:
                    request_tools = self._tool_registry.closing_definitions_for_mode(
                        mode,
                        activated_names=activated_tools,
                    )
                    closing_can_deliver = bool(request_tools)
                force_final_answer = forced_without_budget or not closing_can_deliver
                if force_final_answer:
                    request_tools = ()
                    effective_max_output_tokens = min(
                        effective_max_output_tokens,
                        budget_config.finalization_max_output_tokens,
                    )
                request_messages = (
                    *request_messages,
                    Message(
                        role=MessageRole.SYSTEM,
                        content=(
                            _RUN_BUDGET_CLOSING_MESSAGE
                            if closing_can_deliver
                            else _RUN_BUDGET_FINALIZATION_MESSAGE
                        ),
                    ),
                )
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_FINALIZING,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )
            if (
                request_prefix_state is not None
                and request_tools != request_prefix_state.tools
            ):
                cache_prefix_reused = False
                cache_prefix_message_count = 0
                if prefix_decision != "compact":
                    prefix_decision = "rebuild"
                    prefix_rebuild_reason = "tool_schema_changed"
            # The summary call can cross a run-budget boundary. Check the
            # actual payload after any warning/closing instructions or schema
            # change without invoking a second compression pass.
            context_decision = self._context_manager.remeasure(
                context_decision,
                messages=request_messages,
                tools=request_tools,
                model=resolved_model,
                provider=resolved_provider,
                max_output_tokens=effective_max_output_tokens,
            )
            final_request_config = (
                resolved_provider,
                resolved_model,
                effective_max_output_tokens,
            )
            if (
                request_prefix_state is not None
                and final_request_config != request_prefix_state.request_config
            ):
                cache_prefix_reused = False
                cache_prefix_message_count = 0
                if prefix_decision == "append":
                    prefix_decision = "rebuild"
                    prefix_rebuild_reason = "request_config_changed"
            summary_fields = _summary_event_fields(
                before=summary_state_before,
                after=context_decision.summary_state,
                summary_updated=context_decision.summary_updated,
                include_baseline=not summary_baseline_emitted,
            )
            summary_baseline_emitted = True
            await emitter.emit(
                AgentEventType.MODEL_STARTED,
                step=step,
                provider=resolved_provider,
                model=resolved_model,
                original_estimated_input_tokens=(
                    context_decision.original_estimated_input_tokens
                ),
                prepared_input_tokens=context_decision.prepared_input_tokens,
                estimated_input_tokens=context_decision.estimated_input_tokens,
                context_trimmed=context_decision.trimmed,
                context_window=context_decision.context_window,
                input_budget=context_decision.input_budget,
                working_input_budget=context_decision.working_input_budget,
                hard_trigger_tokens=context_decision.hard_trigger_tokens,
                hard_target_tokens=context_decision.hard_target_tokens,
                usage_ratio=context_decision.usage_ratio,
                trigger_tokens=context_decision.trigger_tokens,
                target_tokens=context_decision.target_tokens,
                tool_result_budget_tokens=(context_decision.tool_result_budget_tokens),
                tool_result_tokens_before=(context_decision.tool_result_tokens_before),
                tool_result_tokens_after=(context_decision.tool_result_tokens_after),
                tool_schema_tokens=context_decision.tool_schema_tokens,
                message_tokens_before=context_decision.message_tokens_before,
                message_tokens_after=context_decision.message_tokens_after,
                unsummarized_conversation_blocks=(
                    context_decision.unsummarized_conversation_blocks
                ),
                conversation_block_limit=(context_decision.conversation_block_limit),
                conversation_block_triggered=(
                    context_decision.conversation_block_triggered
                ),
                requires_compaction=context_decision.requires_compaction,
                exceeds_input_budget=context_decision.exceeds_input_budget,
                capability_source=context_decision.capability_source,
                original_usage_ratio=context_decision.original_usage_ratio,
                prepared_usage_ratio=context_decision.prepared_usage_ratio,
                compaction_stage=context_decision.compaction_stage.value,
                compacted_tool_results=context_decision.compacted_tool_results,
                tool_output_excerpts=new_excerpts,
                removed_tool_rounds=context_decision.removed_tool_rounds,
                reached_target=context_decision.reached_target,
                needs_next_compaction_stage=(
                    context_decision.needs_next_compaction_stage
                ),
                summary_updated=context_decision.summary_updated,
                summarized_conversation_blocks=(
                    context_decision.summarized_conversation_blocks
                ),
                summary_usage=context_decision.summary_usage,
                summary_provider=context_decision.summary_provider,
                summary_model=context_decision.summary_model,
                summary_duration_ms=context_decision.summary_duration_ms,
                summary_error=context_decision.summary_error,
                cache_prefix_reused=cache_prefix_reused,
                cache_prefix_message_count=cache_prefix_message_count,
                available_skill_count=context_injection.available_skill_count,
                skill_catalog_tokens=context_injection.skill_catalog_tokens,
                active_skill_names=context_injection.active_skill_names,
                active_skill_tokens=context_injection.active_skill_tokens,
                active_skill_message_names=(
                    context_injection.active_skill_message_names
                ),
                recall_candidate_ids=context_injection.recall_candidate_ids,
                recall_mode=context_injection.recall_mode,
                prefix_decision=prefix_decision,
                prefix_rebuild_reason=prefix_rebuild_reason,
                compact_ceiling_tokens=context_decision.compact_ceiling_tokens,
                forced_target_tokens=context_decision.forced_target_tokens,
                source_message_count=len(raw_source_messages),
                constraints_revision=(
                    pinned_constraints[1] if pinned_constraints is not None else None
                ),
                **summary_fields,
                **run_budget_event_fields(budget_decision, budget_config),
            )
            if context_decision.exceeds_input_budget:
                return await stop_with_error(
                    ContextWindowExceededError(
                        context_decision.estimated_input_tokens or 0,
                        context_decision.input_budget or 0,
                    ),
                    AgentStopReason.CONTEXT_ERROR,
                    step=step,
                )

            if step_recorder is not None:
                # 回退检查点：上一步已完成、这一步还没请求模型
                await step_recorder.record(
                    step=step,
                    message_count=len(raw_source_messages),
                    summary_state=current_summary_state,
                    tool_result_views=tool_view_state.snapshot(raw_source_messages),
                    constraints_revision=(
                        pinned_constraints[1]
                        if pinned_constraints is not None
                        else None
                    ),
                    task_context_text=next(
                        (
                            message.content
                            for message in context_messages
                            if message.name == "muharness_active_task"
                        ),
                        None,
                    ),
                )

            request_prefix_state = RequestPrefixState(
                source_messages=source_messages,
                context_messages=context_messages,
                tools=request_tools,
                sent_messages=request_messages,
                request_config=final_request_config,
            )

            try:

                # 函数说明：AgentLoop.run.emit_text_delta
                # 用途：发出文本，供模型与工具执行循环使用。
                # 参数：
                #   delta：`delta`输入或配置值，类型 `str`。
                # 返回：类型 `None`；无结果值，显式返回 None。
                # 关键调用（按源码出现顺序，实际执行取决于分支）：`emitter.emit`。
                # 分支与异常：
                #   当 `not delta` 时，返回 `None`。
                # 副作用与资源：
                #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
                # 闭包依赖：从外层读取 `emitter`、`resolved_model`、`resolved_provider`
                # 、`step`。
                async def emit_text_delta(delta: str) -> None:
                    if not delta:
                        return
                    await emitter.emit(
                        AgentEventType.MODEL_OUTPUT_DELTA,
                        step=step,
                        provider=resolved_provider,
                        model=resolved_model,
                        delta=delta,
                    )

                model_started_at = perf_counter()
                response = await adapter.complete_stream(
                    ModelRequest(
                        messages=request_messages,
                        model=resolved_model,
                        tools=request_tools,
                        max_output_tokens=effective_max_output_tokens,
                    ),
                    on_text_delta=emit_text_delta,
                    on_reasoning_delta=None,
                )
            except Exception as exc:
                return await stop_with_error(
                    ModelInvocationError(f"{type(exc).__name__}: {exc}"),
                    AgentStopReason.MODEL_ERROR,
                    step=step,
                )

            usage = add_usage(usage, response.usage)
            main_model_calls += max(1, response.usage.model_calls)
            budget_chargeable_tokens += chargeable_tokens(response.usage)
            assistant_message = response.message.model_copy(update={"reasoning": None})
            response_repair_message = None
            messages.append(assistant_message)
            await emitter.emit(
                AgentEventType.MODEL_COMPLETED,
                step=step,
                provider=response.provider,
                model=response.model,
                message=assistant_message,
                usage=response.usage,
                model_finish_reason=response.finish_reason,
                model_duration_ms=(perf_counter() - model_started_at) * 1000,
                requested_max_output_tokens=effective_max_output_tokens,
                reasoning_chars=len(response.message.reasoning or ""),
            )
            tool_calls_in_message = assistant_message.tool_calls
            if (
                mode is AgentMode.AUDIT
                and response.finish_reason in {"max_tokens", "length"}
                and not tool_calls_in_message
                and not audit_report_recovery_used
                and not budget_forces_final
            ):
                # Preserve tool evidence; recover the report without repeating tools.
                audit_report_recovery_used = True
                audit_report_recovery = True
                messages.pop()
                request_prefix_state = None
                retry_max_output_tokens = min(
                    self._max_output_tokens or 8192,
                    self._context_manager.registry.lookup(
                        resolved_provider,
                        resolved_model,
                    ).max_output_tokens,
                )
                response_repair_message = Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "审计报告因输出预算耗尽而截断。保留本轮已取得的工具证据，"
                        "现在禁止调用工具，也不要重新核验或复述背景。"
                        "重新输出完整简短报告，约 500 字：前四行严格写状态、完整性、"
                        "契约审计、步骤验收；随后写各验收项结论与证据编号、阻断约束、"
                        "给任务管理器的状态更新。证据不足明确写 unknown，不得猜测通过。"
                    ),
                )
                continue
            if force_final_answer and tool_calls_in_message:
                if budget_forces_final:
                    return await stop_with_error(
                        RunBudgetExceededError(
                            "model attempted a tool call during budget finalization"
                        ),
                        AgentStopReason.RUN_BUDGET,
                        step=step,
                    )
                if tool_round_limit_reached:
                    return stop_at_tool_round_limit(step=step)
                return await stop_with_error(
                    ModelInvocationError(
                        "model attempted a tool call during forced finalization"
                    ),
                    AgentStopReason.MODEL_ERROR,
                    step=step,
                )
            if not tool_calls_in_message:
                if not (assistant_message.content or "").strip():
                    if tool_round_limit_reached:
                        return stop_at_tool_round_limit(step=step)
                    messages.pop()
                    if budget_forces_final:
                        return await stop_with_error(
                            RunBudgetExceededError(
                                "model returned empty content during budget "
                                "finalization"
                            ),
                            AgentStopReason.RUN_BUDGET,
                            step=step,
                        )
                    if force_final_answer:
                        return await stop_with_error(
                            ModelInvocationError(
                                "model returned empty content during forced "
                                "finalization"
                            ),
                            AgentStopReason.MODEL_ERROR,
                            step=step,
                        )
                    if not empty_final_retry_used:
                        empty_final_retry_used = True
                        # Only recover a confirmed truncation; preserve explicit limits.
                        if (
                            response.finish_reason in {"max_tokens", "length"}
                            and self._max_output_tokens is None
                            and effective_max_output_tokens
                            < _EMPTY_RESPONSE_RETRY_MAX_OUTPUT_TOKENS
                        ):
                            retry_max_output_tokens = min(
                                effective_max_output_tokens * 2,
                                _EMPTY_RESPONSE_RETRY_MAX_OUTPUT_TOKENS,
                                self._context_manager.registry.lookup(
                                    resolved_provider,
                                    resolved_model,
                                ).max_output_tokens,
                            )
                        response_repair_message = Message(
                            role=MessageRole.SYSTEM,
                            content=_EMPTY_FINAL_RETRY_MESSAGE,
                        )
                        continue
                    return await stop_with_error(
                        ModelInvocationError(
                            "model returned empty content twice without tool calls"
                        ),
                        AgentStopReason.MODEL_ERROR,
                        step=step,
                    )
                if looks_like_textual_tool_call(assistant_message.content):
                    if force_final_answer:
                        if budget_forces_final:
                            return await stop_with_error(
                                RunBudgetExceededError(
                                    "model emitted a textual tool call during "
                                    "budget finalization"
                                ),
                                AgentStopReason.RUN_BUDGET,
                                step=step,
                            )
                        if tool_round_limit_reached:
                            return stop_at_tool_round_limit(step=step)
                        return await stop_with_error(
                            ModelInvocationError(
                                "model emitted a textual tool call during forced "
                                "finalization"
                            ),
                            AgentStopReason.MODEL_ERROR,
                            step=step,
                        )
                    messages.pop()
                    if not textual_tool_call_retry_used:
                        textual_tool_call_retry_used = True
                        response_repair_message = Message(
                            role=MessageRole.SYSTEM,
                            content=_TEXTUAL_TOOL_CALL_RETRY_MESSAGE,
                        )
                        continue
                    return await stop_with_error(
                        ModelInvocationError(
                            "model emitted a textual tool call twice without "
                            "structured tool_calls"
                        ),
                        AgentStopReason.MODEL_ERROR,
                        step=step,
                    )
                final_message = assistant_message
                if mode is AgentMode.PLAN:
                    plan_valid = False
                    if (
                        plan_task_id is not None
                        and self._task_context_provider is not None
                    ):
                        plan_valid = (
                            await self._task_context_provider.pending_plan_is_valid(
                                conversation_id,
                                plan_task_id,
                            )
                        )
                    if not plan_valid:
                        prefix = (
                            _PLAN_NO_TASK_MESSAGE
                            if not plan_task_created
                            else _PLAN_NO_VALID_TASK_MESSAGE
                        )
                        final_message = plan_failure_message(
                            assistant_message,
                            prefix,
                        )
                        plan_task_id = None
                        messages[-1] = final_message
                return self._result(
                    run_id=run_id,
                    final_message=final_message,
                    messages=messages,
                    steps=step,
                    stop_reason=AgentStopReason.FINAL_ANSWER,
                    tool_rounds=tool_rounds,
                    tool_calls=tool_calls,
                    usage=usage,
                    plan_task_id=plan_task_id,
                    model_finish_reason=response.finish_reason,
                    summary_state=current_summary_state,
                    tool_result_views=tool_view_state.snapshot(messages),
                )

            await record_messages()
            round_outcome = await self._tool_round_executor.execute(
                tool_calls_in_message,
                run_id=run_id,
                conversation_id=conversation_id,
                user_input=user_input,
                step=step,
                mode=mode,
                round_index=len(tool_rounds),
                closing_can_deliver=closing_can_deliver,
                activated_tools=activated_tools,
                context_session=context_session,
                previous_signature=previous_signature,
                repeated_count=repeated_count,
                emitter=emitter,
                hook=tool_event_hook,
                tool_context_metadata=tool_context_metadata,
            )
            tool_calls.extend(round_outcome.records)
            messages.extend(round_outcome.result_messages)
            await record_messages()
            previous_signature = round_outcome.previous_signature
            repeated_count = round_outcome.repeated_count
            plan_task_created = plan_task_created or round_outcome.plan_task_created
            if round_outcome.plan_task_id is not None:
                plan_task_id = round_outcome.plan_task_id
            if round_outcome.repeated_error is not None:
                return self._result(
                    run_id=run_id,
                    final_message=self._error_message(round_outcome.repeated_error),
                    messages=messages,
                    steps=step,
                    stop_reason=AgentStopReason.REPEATED_TOOL_CALL,
                    tool_rounds=tool_rounds,
                    tool_calls=tool_calls,
                    usage=usage,
                    error=round_outcome.repeated_error,
                    summary_state=current_summary_state,
                    tool_result_views=tool_view_state.snapshot(messages),
                )
            pending_activations = set(round_outcome.pending_activations)
            round_records = list(round_outcome.records)
            tool_rounds.append(
                ToolRound(
                    round_index=len(tool_rounds),
                    assistant_message=assistant_message,
                    records=tuple(round_records),
                )
            )
            if closing_can_deliver:
                budget_closing_delivery_used = True
            activated_tools.update(pending_activations)
            finalization_pending = step == self._max_steps and (
                self._run_budget.evaluate(
                    usage,
                    chargeable_tokens_override=budget_chargeable_tokens,
                    model_calls_override=main_model_calls,
                ).should_finalize
            )

        error = MaxStepsExceededError(self._max_steps)
        return self._result(
            run_id=run_id,
            final_message=self._error_message(error),
            messages=messages,
            steps=self._max_steps,
            stop_reason=AgentStopReason.MAX_STEPS,
            tool_rounds=tool_rounds,
            tool_calls=tool_calls,
            usage=usage,
            error=error,
            summary_state=current_summary_state,
            tool_result_views=tool_view_state.snapshot(messages),
        )

    # 函数说明：AgentLoop._error_message
    # 用途：返回
    # `Message(role=MessageRole.ASSISTANT, content=f'Agent stopped: {error}')`，提供
    # AgentLoop 的派生值。
    # 参数：
    #   error：异常或错误信息，类型 `AgentRuntimeError`。
    # 返回：类型 `Message`；返回
    # `Message(role=MessageRole.ASSISTANT, content=f'Agent stopped: {error}')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
    @staticmethod
    def _error_message(error: AgentRuntimeError) -> Message:
        return Message(
            role=MessageRole.ASSISTANT,
            content=f"Agent stopped: {error}",
        )

    # 函数说明：AgentLoop._result
    # 用途：处理模型与工具执行循环中的 `_result` 数据；结果及边界条件见下方说明。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   final_message：消息输入或配置值，类型 `Message`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   steps：任务步骤集合，类型 `int`。
    #   stop_reason：运行终止原因，类型 `AgentStopReason`。
    #   tool_rounds：传给 `tuple` 的输入，类型 `list[ToolRound]`。
    #   tool_calls：待执行的结构化工具调用，类型 `list[ToolCallRecord]`。
    #   usage：模型调用用量统计，类型 `ModelUsage`。
    #   error：异常或错误信息，类型 `AgentRuntimeError | None`；默认 `None`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   plan_task_id：计划任务标识，类型 `str | None`；默认 `None`。
    #   model_finish_reason：模型原因输入或配置值，类型 `str | None`；默认 `None`。
    #   tool_result_views：工具结果的固定模型视图，类型 `Sequence[ToolResultView]`；默认
    #  `()`。
    # 返回：类型 `AgentResult`；返回 `AgentResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentResult`。
    @staticmethod
    def _result(
        *,
        run_id: str,
        final_message: Message,
        messages: Sequence[Message],
        steps: int,
        stop_reason: AgentStopReason,
        tool_rounds: list[ToolRound],
        tool_calls: list[ToolCallRecord],
        usage: ModelUsage,
        error: AgentRuntimeError | None = None,
        summary_state: ConversationSummaryState | None = None,
        plan_task_id: str | None = None,
        model_finish_reason: str | None = None,
        tool_result_views: Sequence[ToolResultView] = (),
    ) -> AgentResult:
        complete_messages = tuple(messages)
        if not complete_messages or complete_messages[-1] != final_message:
            complete_messages = (*complete_messages, final_message)

        return AgentResult(
            run_id=run_id,
            final_message=final_message,
            messages=complete_messages,
            steps=steps,
            stop_reason=stop_reason,
            tool_rounds=tuple(tool_rounds),
            tool_calls=tuple(tool_calls),
            usage=usage,
            error=(
                AgentError(type=type(error).__name__, message=str(error))
                if error is not None
                else None
            ),
            summary_state=summary_state,
            plan_task_id=plan_task_id,
            model_finish_reason=model_finish_reason,
            tool_result_views=tuple(tool_result_views),
        )


def _summary_event_fields(
    *,
    before: ConversationSummaryState | None,
    after: ConversationSummaryState | None,
    summary_updated: bool,
    include_baseline: bool,
) -> dict[str, Any]:
    """上下文面板需要的压缩数据：覆盖范围、摘要快照和可能丢失的约束。

    快照只在摘要更新时附带；每次运行的第一步附带一份初始摘要作为对比基线。
    """
    fields: dict[str, Any] = {
        "summary_covered_before": (
            before.covered_message_count if before is not None else 0
        ),
        "summary_covered_after": (
            after.covered_message_count if after is not None else 0
        ),
    }
    if after is not None and (summary_updated or include_baseline):
        fields["summary_snapshot"] = after.summary.model_dump(mode="json")
    if summary_updated and before is not None and after is not None:
        fields["summary_previous_snapshot"] = before.summary.model_dump(mode="json")
        kept = set(after.summary.user_constraints)
        fields["constraints_possibly_dropped"] = tuple(
            entry for entry in before.summary.user_constraints if entry not in kept
        )
    return fields


__all__ = [
    "PINNED_CONSTRAINTS_MESSAGE_NAME",
    "AgentLoop",
    "ConstraintsProvider",
    "pinned_constraints_message",
]
