
from __future__ import annotations

from collections.abc import Callable

from app.domain.memory import (
    MaintenanceAction,
    MemoryMaintenanceCandidate,
    MemoryMaintenanceInput,
    MemoryMaintenanceReflector,
    MemoryManager,
    MemoryReflectionInput,
    PostRunMemoryReflector,
    ReflectionAction,
    decide_reflection_gate,
)
from app.domain.task.context import TaskContextProvider
from app.models.types import ModelUsage

from .event_stream import EventEmitter
from .events import AgentEventType
from .result import AgentResult, AgentStopReason
from .runtime_helpers import (
    recalled_memory_revisions,
    reflection_tool_context,
)


class PostRunMemoryCoordinator:

    # 函数说明：PostRunMemoryCoordinator.__init__
    # 用途：初始化 PostRunMemoryCoordinator；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager | None`。
    #   reflector：`reflector`输入或配置值，类型 `PostRunMemoryReflector | None`。
    #   maintenance_reflector：维护输入或配置值，类型
    # `MemoryMaintenanceReflector | None`。
    #   task_context_provider：任务状态上下文提供器，类型 `TaskContextProvider | None`。
    #   submit：`submit`输入或配置值，类型 `Callable[..., bool] | None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`、`self._reflector`、`self._maintenance_reflector`
    # 、`self._task_context_provider`、`self._submit`。
    def __init__(
        self,
        *,
        manager: MemoryManager | None,
        reflector: PostRunMemoryReflector | None,
        maintenance_reflector: MemoryMaintenanceReflector | None,
        task_context_provider: TaskContextProvider | None,
        submit: Callable[..., bool] | None,
    ) -> None:
        self._manager = manager
        self._reflector = reflector
        self._maintenance_reflector = maintenance_reflector
        self._task_context_provider = task_context_provider
        self._submit = submit

    # 函数说明：PostRunMemoryCoordinator.schedule
    # 用途：在运行结束后提交记忆反思任务，不阻塞当前答复。
    # 参数：
    #   result：上一步计算或执行得到的结果，类型 `AgentResult`。
    #   user_input：本次用户输入，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   emitter：执行事件发射器，类型 `EventEmitter`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.ensure_capacity` →
    # `emitter.emit` → `recalled_memory_revisions` → `decide_reflection_gate` →
    # `manager.reflection_context` → `self._task_context_provider.message_for`；另有 5
    # 个调用点。
    # 分支与异常：
    #   `reflector is None` 分支在完成前置处理后返回 `None`。
    #   `not reflector.enabled` 分支在完成前置处理后返回 `None`。
    #   `result.stop_reason is not AgentStopReason.FINAL_ANSWER` 分支在完成前置处理后返
    # 回 `None`。
    #   `not gate.should_reflect` 分支在完成前置处理后返回 `None`。
    #   捕获 `Exception` 后，返回 `None`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def schedule(
        self,
        result: AgentResult,
        *,
        user_input: str,
        conversation_id: str | None,
        emitter: EventEmitter,
    ) -> None:

        """在运行结束后提交记忆反思任务，不阻塞当前答复。"""
        reflector = self._reflector
        manager = self._manager

        if reflector is None:
            if (
                self._maintenance_reflector is not None
                and result.stop_reason is AgentStopReason.FINAL_ANSWER
            ):
                await self.ensure_capacity(required_slots=0, emitter=emitter)
            return
        if not reflector.enabled:
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_SKIPPED,
                reflection_triggered=False,
                reflection_skip_reason="disabled",
            )
            if (
                self._maintenance_reflector is not None
                and result.stop_reason is AgentStopReason.FINAL_ANSWER
            ):
                await self.ensure_capacity(required_slots=0, emitter=emitter)
            return
        if result.stop_reason is not AgentStopReason.FINAL_ANSWER:
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_SKIPPED,
                reflection_triggered=False,
                reflection_skip_reason=f"stop_reason={result.stop_reason.value}",
            )
            return

        recalled_revisions = recalled_memory_revisions(result.tool_calls)
        if reflector.config.gate_enabled:
            gate = decide_reflection_gate(
                user_input,
                recalled_memory_ids=tuple(recalled_revisions),
            )
            if not gate.should_reflect:
                await emitter.emit(
                    AgentEventType.MEMORY_REFLECTION_SKIPPED,
                    reflection_triggered=False,
                    reflection_skip_reason=f"gate:{gate.reason.value}",
                )
                if manager is not None:
                    await self.ensure_capacity(required_slots=0, emitter=emitter)
                return

        try:
            if manager is None:
                raise RuntimeError("memory manager unavailable")
            core_memory, memory_index = await manager.reflection_context()
            task_context = ""
            if self._task_context_provider is not None:
                task_message = await self._task_context_provider.message_for(
                    conversation_id
                )
                if task_message is not None:
                    task_context = task_message.content or ""
            reflection_input = MemoryReflectionInput(
                run_id=result.run_id,
                conversation_id=conversation_id,
                user_input=user_input[:8_000],
                final_answer=(result.final_message.content or "")[:12_000],
                tool_context=reflection_tool_context(
                    result.tool_calls,
                    max_chars=reflector.config.max_tool_context_chars,
                ),
                recalled_memory_ids=tuple(recalled_revisions),
                core_memory=core_memory,
                memory_index=memory_index,
                task_context=task_context,
            )
        except Exception as exc:
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_FAILED,
                reflection_triggered=True,
                reflection_error=f"{type(exc).__name__}: {exc}",
                reflection_mutation_applied=False,
                provider=reflector.provider_hint,
                model=reflector.model_hint,
                usage=ModelUsage(),
            )
            if manager is not None:
                await self.ensure_capacity(required_slots=0, emitter=emitter)
            return

        # 函数说明：PostRunMemoryCoordinator.schedule.job
        # 用途：在模型与工具执行循环中处理 `job`，通过 `self._run_reflection` 完成首个内
        # 部处理步骤。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run_reflection`。
        # 闭包依赖：从外层读取 `emitter`、`manager`、`recalled_revisions`、
        # `reflection_input`、`reflector`。
        async def job() -> None:
            await self._run_reflection(
                reflector=reflector,
                manager=manager,
                reflection_input=reflection_input,
                recalled_revisions=recalled_revisions,
                emitter=emitter,
            )

        if self._submit is not None:
            self._submit(
                job,
                conversation_id=conversation_id,
                run_id=result.run_id,
            )
        else:
            await job()

    # 函数说明：PostRunMemoryCoordinator._run_reflection
    # 用途：运行反思，供模型与工具执行循环使用。
    # 参数：
    #   reflector：`reflector`输入或配置值，类型 `PostRunMemoryReflector`。
    #   manager：当前业务管理器，类型 `MemoryManager`。
    #   reflection_input：传给 `reflector.decide` 的输入，类型 `MemoryReflectionInput`。
    #   recalled_revisions：`recalled_revisions`输入或配置值，类型 `dict[str, int]`。
    #   emitter：执行事件发射器，类型 `EventEmitter`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`emitter.emit` →
    # `reflector.decide` → `manager.update_if_revision` → `manager.validate_create` →
    # `self.ensure_capacity` → `manager.create_if_capacity`；另有 3 个调用点。
    # 分支与异常：
    #   当 `proposal.error is not None` 时，抛出 `RuntimeError(proposal.error)`。
    #   当 `proposal.decision is None` 时，抛出
    # `RuntimeError('reflection model returned no decision')`。
    #   当 `expected_revision is None` 时，抛出 `ValueError(…)`。
    #   当 `not capacity_ready` 时，抛出 `RuntimeError(…)`。
    #   捕获 `Exception` 后，返回 `None`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def _run_reflection(
        self,
        *,
        reflector: PostRunMemoryReflector,
        manager: MemoryManager,
        reflection_input: MemoryReflectionInput,
        recalled_revisions: dict[str, int],
        emitter: EventEmitter,
    ) -> None:

        await emitter.emit(
            AgentEventType.MEMORY_REFLECTION_STARTED,
            reflection_triggered=True,
            provider=reflector.provider_hint,
            model=reflector.model_hint,
        )
        proposal = None
        memory_id: str | None = None
        mutation_applied = False
        try:
            proposal = await reflector.decide(reflection_input)
            if proposal.error is not None:
                raise RuntimeError(proposal.error)
            if proposal.decision is None:
                raise RuntimeError("reflection model returned no decision")

            decision = proposal.decision
            if decision.action is ReflectionAction.UPDATE:
                memory_id = decision.memory_id
                expected_revision = recalled_revisions.get(memory_id or "")
                if expected_revision is None:
                    raise ValueError(
                        "reflection update requires memory_read success in current run"
                    )
                record = await manager.update_if_revision(
                    memory_id or "",
                    expected_revision=expected_revision,
                    title=decision.title or "",
                    summary=decision.summary or "",
                    content=decision.content or "",
                    reason=decision.reason,
                )
                memory_id = record.id
                mutation_applied = True
            elif decision.action is ReflectionAction.CREATE:
                manager.validate_create(
                    title=decision.title or "",
                    summary=decision.summary or "",
                    content=decision.content or "",
                )
                capacity_ready = await self.ensure_capacity(
                    required_slots=1,
                    emitter=emitter,
                )
                if not capacity_ready:
                    raise RuntimeError(
                        "memory create skipped because capacity is unavailable"
                    )
                record = await manager.create_if_capacity(
                    title=decision.title or "",
                    summary=decision.summary or "",
                    content=decision.content or "",
                )
                if record is None:
                    raise RuntimeError(
                        "memory create lost the available slot to a concurrent run"
                    )
                memory_id = record.id
                mutation_applied = True
        except Exception as exc:
            provider = (
                proposal.provider
                if proposal is not None
                else reflector.provider_hint
            )
            model = proposal.model if proposal is not None else reflector.model_hint
            usage = proposal.usage if proposal is not None else ModelUsage()
            duration_ms = proposal.duration_ms if proposal is not None else None
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_FAILED,
                reflection_triggered=True,
                reflection_action=(
                    proposal.decision.action.value
                    if proposal is not None and proposal.decision is not None
                    else None
                ),
                reflection_duration_ms=duration_ms,
                reflection_attempts=(
                    proposal.attempts if proposal is not None else None
                ),
                reflection_finish_reason=(
                    proposal.finish_reason if proposal is not None else None
                ),
                reflection_error=f"{type(exc).__name__}: {exc}",
                reflection_memory_id=(
                    proposal.decision.memory_id
                    if proposal is not None and proposal.decision is not None
                    else None
                ),
                reflection_mutation_applied=False,
                reflection_input_json=(
                    proposal.input_json if proposal is not None else None
                ),
                reflection_raw_output=(
                    proposal.raw_output if proposal is not None else None
                ),
                provider=provider,
                model=model,
                usage=usage,
            )
            if (
                proposal is None
                or proposal.decision is None
                or proposal.decision.action is not ReflectionAction.CREATE
            ):
                await self.ensure_capacity(required_slots=0, emitter=emitter)
            return

        maintenance_required = await manager.maintenance_required()
        candidate_ids: tuple[str, ...] = ()
        if maintenance_required:
            candidates = await manager.retention_candidates()
            candidate_ids = tuple(item.id for item in candidates)
        await emitter.emit(
            AgentEventType.MEMORY_REFLECTION_COMPLETED,
            reflection_triggered=True,
            reflection_action=proposal.decision.action.value,
            reflection_duration_ms=proposal.duration_ms,
            reflection_attempts=proposal.attempts,
            reflection_finish_reason=proposal.finish_reason,
            reflection_memory_id=memory_id,
            reflection_mutation_applied=mutation_applied,
            reflection_maintenance_required=maintenance_required,
            reflection_retention_candidate_ids=candidate_ids,
            reflection_input_json=proposal.input_json,
            reflection_raw_output=proposal.raw_output,
            provider=proposal.provider,
            model=proposal.model,
            usage=proposal.usage,
        )
        if maintenance_required:
            await self.ensure_capacity(required_slots=0, emitter=emitter)

    # 函数说明：PostRunMemoryCoordinator.ensure_capacity
    # 用途：在写入新记忆前检查容量，并按维护决策释放空间。
    # 参数：
    #   required_slots：`required_slots`输入或配置值，类型 `int`。
    #   emitter：执行事件发射器，类型 `EventEmitter`。
    # 返回：类型 `bool`；按分支返回
    # `await self._ensure_capacity_impl(required_slots=required_slots, emitter=emitter)`
    # ；`False`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._ensure_capacity_impl` →
    # `emitter.emit`。
    # 分支与异常：
    #   捕获 `Exception` 后，返回 `False`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def ensure_capacity(
        self,
        *,
        required_slots: int,
        emitter: EventEmitter,
    ) -> bool:

        """在写入新记忆前检查容量，并按维护决策释放空间。"""
        try:
            return await self._ensure_capacity_impl(
                required_slots=required_slots,
                emitter=emitter,
            )
        except Exception as exc:
            await emitter.emit(
                AgentEventType.MEMORY_MAINTENANCE_FAILED,
                maintenance_triggered=True,
                maintenance_error=f"{type(exc).__name__}: {exc}",
            )
            return False

    # 函数说明：PostRunMemoryCoordinator._ensure_capacity_impl
    # 用途：确保`capacity_impl`，供模型与工具执行循环使用。
    # 参数：
    #   required_slots：`required_slots`输入或配置值，类型 `int`。
    #   emitter：执行事件发射器，类型 `EventEmitter`。
    # 返回：类型 `bool`；按分支返回 `False`；`True`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`manager.active_count` →
    # `emitter.emit` → `manager.retention_candidates` → `MemoryMaintenanceCandidate` →
    # `maintainer.decide` → `MemoryMaintenanceInput`；另有 1 个调用点。
    # 分支与异常：
    #   当 `manager is None` 时，返回 `False`。
    #   当 `active_count + required_slots <= manager.max_active` 时，返回 `True`。
    #   `maintainer is None or not maintainer.enabled` 分支在完成前置处理后返回 `False`
    # 。
    #   当 `remaining <= 0` 时，返回 `True`。
    #   捕获 `Exception` 后，返回 `False`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def _ensure_capacity_impl(
        self,
        *,
        required_slots: int,
        emitter: EventEmitter,
    ) -> bool:

        manager = self._manager
        if manager is None:
            return False
        active_count = await manager.active_count()
        if active_count + required_slots <= manager.max_active:
            return True

        maintainer = self._maintenance_reflector
        if maintainer is None or not maintainer.enabled:
            await emitter.emit(
                AgentEventType.MEMORY_MAINTENANCE_SKIPPED,
                maintenance_triggered=False,
                maintenance_skip_reason=(
                    "unavailable" if maintainer is None else "disabled"
                ),
                maintenance_active_count=active_count,
                maintenance_max_active=manager.max_active,
                maintenance_remaining_overflow=max(
                    0,
                    active_count + required_slots - manager.max_active,
                ),
            )
            return False

        for _ in range(maintainer.config.max_actions):
            active_count = await manager.active_count()
            remaining = active_count + required_slots - manager.max_active
            if remaining <= 0:
                return True
            records = await manager.retention_candidates(
                limit=maintainer.config.candidate_limit
            )
            record_snapshots = {record.id: record for record in records}
            candidates = tuple(
                MemoryMaintenanceCandidate(
                    id=record.id,
                    title=record.title,
                    summary=record.summary,
                    content=record.content,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    last_accessed_at=record.last_accessed_at,
                    access_count=record.access_count,
                )
                for record in records
            )
            candidate_ids = tuple(candidate.id for candidate in candidates)
            if not candidates:
                await emitter.emit(
                    AgentEventType.MEMORY_MAINTENANCE_FAILED,
                    maintenance_triggered=True,
                    maintenance_error="no active maintenance candidates",
                    maintenance_active_count=active_count,
                    maintenance_max_active=manager.max_active,
                    maintenance_remaining_overflow=max(0, remaining),
                )
                return False
            await emitter.emit(
                AgentEventType.MEMORY_MAINTENANCE_STARTED,
                maintenance_triggered=True,
                maintenance_active_count=active_count,
                maintenance_max_active=manager.max_active,
                maintenance_candidate_ids=candidate_ids,
                maintenance_remaining_overflow=max(0, remaining),
                provider=maintainer.provider_hint,
                model=maintainer.model_hint,
            )
            proposal = await maintainer.decide(
                MemoryMaintenanceInput(
                    active_count=active_count,
                    max_active=manager.max_active,
                    required_slots=required_slots,
                    candidates=candidates,
                )
            )
            if proposal.error is not None or proposal.decision is None:
                await emitter.emit(
                    AgentEventType.MEMORY_MAINTENANCE_FAILED,
                    maintenance_triggered=True,
                    maintenance_duration_ms=proposal.duration_ms,
                    maintenance_error=(
                        proposal.error or "maintenance model returned no decision"
                    ),
                    maintenance_active_count=active_count,
                    maintenance_max_active=manager.max_active,
                    maintenance_candidate_ids=candidate_ids,
                    maintenance_remaining_overflow=max(0, remaining),
                    provider=proposal.provider,
                    model=proposal.model,
                    usage=proposal.usage,
                )
                return False

            decision = proposal.decision
            if decision.action is MaintenanceAction.DEFER:
                await emitter.emit(
                    AgentEventType.MEMORY_MAINTENANCE_COMPLETED,
                    maintenance_triggered=True,
                    maintenance_action=decision.action.value,
                    maintenance_duration_ms=proposal.duration_ms,
                    maintenance_reason=decision.reason,
                    maintenance_active_count=active_count,
                    maintenance_max_active=manager.max_active,
                    maintenance_candidate_ids=candidate_ids,
                    maintenance_remaining_overflow=max(0, remaining),
                    provider=proposal.provider,
                    model=proposal.model,
                    usage=proposal.usage,
                )
                return False

            memory_id = decision.memory_id or ""
            snapshots = {candidate.id: candidate for candidate in candidates}
            if snapshots.get(memory_id) is None:
                await emitter.emit(
                    AgentEventType.MEMORY_MAINTENANCE_FAILED,
                    maintenance_triggered=True,
                    maintenance_action=decision.action.value,
                    maintenance_duration_ms=proposal.duration_ms,
                    maintenance_error=(
                        "maintenance selected an ID outside the candidate set"
                    ),
                    maintenance_memory_id=memory_id,
                    maintenance_active_count=active_count,
                    maintenance_max_active=manager.max_active,
                    maintenance_candidate_ids=candidate_ids,
                    maintenance_remaining_overflow=max(0, remaining),
                    provider=proposal.provider,
                    model=proposal.model,
                    usage=proposal.usage,
                )
                return False
            try:
                archived = await manager.archive_if_unchanged(
                    memory_id,
                    expected_record=record_snapshots[memory_id],
                    reason=decision.reason,
                )
            except Exception as exc:
                await emitter.emit(
                    AgentEventType.MEMORY_MAINTENANCE_FAILED,
                    maintenance_triggered=True,
                    maintenance_action=decision.action.value,
                    maintenance_duration_ms=proposal.duration_ms,
                    maintenance_error=f"{type(exc).__name__}: {exc}",
                    maintenance_memory_id=memory_id,
                    maintenance_reason=decision.reason,
                    maintenance_active_count=active_count,
                    maintenance_max_active=manager.max_active,
                    maintenance_candidate_ids=candidate_ids,
                    maintenance_remaining_overflow=max(0, remaining),
                    provider=proposal.provider,
                    model=proposal.model,
                    usage=proposal.usage,
                )
                return False
            after_count = await manager.active_count()
            after_remaining = max(
                0,
                after_count + required_slots - manager.max_active,
            )
            await emitter.emit(
                AgentEventType.MEMORY_MAINTENANCE_COMPLETED,
                maintenance_triggered=True,
                maintenance_action=decision.action.value,
                maintenance_duration_ms=proposal.duration_ms,
                maintenance_memory_id=archived.id,
                maintenance_reason=decision.reason,
                maintenance_active_count=after_count,
                maintenance_max_active=manager.max_active,
                maintenance_candidate_ids=candidate_ids,
                maintenance_remaining_overflow=after_remaining,
                provider=proposal.provider,
                model=proposal.model,
                usage=proposal.usage,
            )

        active_count = await manager.active_count()
        remaining = max(0, active_count + required_slots - manager.max_active)
        if remaining == 0:
            return True
        await emitter.emit(
            AgentEventType.MEMORY_MAINTENANCE_SKIPPED,
            maintenance_triggered=True,
            maintenance_skip_reason="max_actions_reached",
            maintenance_active_count=active_count,
            maintenance_max_active=manager.max_active,
            maintenance_remaining_overflow=remaining,
        )
        return False


__all__ = ["PostRunMemoryCoordinator"]
