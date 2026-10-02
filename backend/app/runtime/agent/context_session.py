from __future__ import annotations

from dataclasses import dataclass

from app.domain.memory import (
    MemoryManager,
    MemoryRecallQueryInputs,
    MemoryRecallSnapshot,
    SearchMode,
)
from app.domain.skills import Skill, SkillContextProvider, SkillMetadata, SkillStore
from app.domain.task.context import TaskContextProvider
from app.models.types import Message, MessageRole, ToolResult
from app.runtime.checkpoint import RunCheckpoint, render_checkpoint_context

from .event_stream import EventEmitter
from .events import AgentEventType
from .runtime_helpers import skill_read_outcome


@dataclass(frozen=True, slots=True)
class RuntimeContextInjection:
    messages: tuple[Message, ...]
    available_skill_count: int | None
    skill_catalog_tokens: int | None
    active_skill_names: tuple[str, ...]
    active_skill_tokens: int | None
    active_skill_message_names: tuple[str, ...]
    recall_candidate_ids: tuple[str, ...] = ()
    recall_mode: str | None = None


class RuntimeContextSession:
    # 函数说明：RuntimeContextSession.__init__
    # 用途：初始化 RuntimeContextSession；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   memory_manager：长期记忆管理器，类型 `MemoryManager | None`。
    #   skill_store：技能存储，类型 `SkillStore | None`。
    #   skill_context_provider：已激活技能上下文提供器，类型
    # `SkillContextProvider | None`。
    #   task_context_provider：任务状态上下文提供器，类型 `TaskContextProvider | None`。
    #   recall_query：查询输入或配置值，类型 `MemoryRecallQueryInputs | None`；默认
    # `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._memory_manager`、`self._skill_store`、
    # `self._skill_context_provider`、`self._task_context_provider`、
    # `self._recall_query`、`self._memory_messages`、`self._memory_loaded`、
    # `self._recall_snapshot` 等 11 个字段。
    def __init__(
        self,
        *,
        memory_manager: MemoryManager | None,
        skill_store: SkillStore | None,
        skill_context_provider: SkillContextProvider | None,
        task_context_provider: TaskContextProvider | None,
        recall_query: MemoryRecallQueryInputs | None = None,
    ) -> None:
        self._memory_manager = memory_manager
        self._skill_store = skill_store
        self._skill_context_provider = skill_context_provider
        self._task_context_provider = task_context_provider
        self._recall_query = recall_query
        self._memory_messages: tuple[Message, ...] = ()
        self._memory_loaded = False
        self._recall_snapshot: MemoryRecallSnapshot | None = None
        self._catalog: tuple[SkillMetadata, ...] = ()
        self._catalog_loaded = False
        self._active_skills: dict[str, Skill] = {}

    # 函数说明：RuntimeContextSession.active_skill_names
    # 用途：返回 `tuple(self._active_skills)`，提供 RuntimeContextSession 的派生值。
    # 返回：类型 `tuple[str, ...]`；返回 `tuple(self._active_skills)`。
    @property
    def active_skill_names(self) -> tuple[str, ...]:
        return tuple(self._active_skills)

    # 函数说明：RuntimeContextSession.build
    # 用途：组合系统提示、会话历史、记忆与任务上下文，形成模型请求消息。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   recovery_checkpoint：传给 `render_checkpoint_context` 的输入，类型
    # `RunCheckpoint | None`。
    #   trailing_system_messages：传给 `messages.extend` 的输入，类型
    # `tuple[Message, ...]`。
    # 返回：类型 `RuntimeContextInjection`；返回 `RuntimeContextInjection(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._load_memory_messages` →
    # `self._skill_store.catalog` → `provider.catalog_message` →
    # `provider.active_messages` → `render_checkpoint_context` →
    # `self._task_context_provider.message_for`；另有 3 个调用点。
    # 分支与异常：
    #   捕获 `Exception` 后，执行异常分支中的状态更新；具体更新见实现。
    # 副作用与资源：
    #   更新对象字段：`self._memory_loaded`、`self._memory_messages`、
    # `self._catalog_loaded`、`self._catalog`。
    async def build(
        self,
        *,
        conversation_id: str | None,
        recovery_checkpoint: RunCheckpoint | None,
        trailing_system_messages: tuple[Message, ...],
    ) -> RuntimeContextInjection:
        """组合系统提示、会话历史、记忆与任务上下文，形成模型请求消息。"""
        messages: list[Message] = []
        if self._memory_manager is not None and not self._memory_loaded:
            self._memory_loaded = True
            try:
                self._memory_messages = await self._load_memory_messages(
                    conversation_id
                )
            except Exception:
                self._memory_messages = ()
        messages.extend(self._memory_messages)

        provider = self._skill_context_provider
        if provider is not None and self._skill_store is not None:
            if not self._catalog_loaded:
                self._catalog_loaded = True
                try:
                    self._catalog = await self._skill_store.catalog()
                except Exception:
                    self._catalog = ()
            catalog_message = provider.catalog_message(self._catalog)
            if catalog_message is not None:
                messages.append(catalog_message)

        injected_active_names: tuple[str, ...] = ()
        if provider is not None and self._active_skills:
            active_messages = provider.active_messages(
                tuple(self._active_skills.values())
            )
            if active_messages:
                injected_active_names = tuple(self._active_skills)
                messages.extend(active_messages)

        if recovery_checkpoint is not None:
            messages.append(render_checkpoint_context(recovery_checkpoint))
        if self._task_context_provider is not None:
            task_message = await self._task_context_provider.message_for(
                conversation_id
            )
            if task_message is not None:
                # Never use a changing Task revision as a top-level SYSTEM
                # segment (Anthropic hoists those before the entire history).
                # Permission and mode checks remain enforced by the runtime.
                messages.append(
                    task_message.model_copy(
                        update={
                            "role": MessageRole.USER,
                        }
                    )
                )
        messages.extend(trailing_system_messages)

        return RuntimeContextInjection(
            messages=tuple(messages),
            available_skill_count=(
                len(self._catalog) if provider is not None else None
            ),
            skill_catalog_tokens=(
                provider.catalog_tokens(self._catalog) if provider is not None else None
            ),
            active_skill_names=tuple(self._active_skills),
            active_skill_tokens=(
                provider.active_tokens(tuple(self._active_skills.values()))
                if provider is not None
                else None
            ),
            active_skill_message_names=injected_active_names,
            recall_candidate_ids=tuple(
                candidate.memory_id
                for candidate in (
                    self._recall_snapshot.candidates
                    if self._recall_snapshot is not None
                    else ()
                )
            ),
            recall_mode=(
                self._recall_snapshot.mode.value
                if self._recall_snapshot is not None
                else None
            ),
        )

    # 函数说明：RuntimeContextSession._load_memory_messages
    # 用途：加载记忆消息序列，供模型与工具执行循环使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`。
    # 返回：类型 `tuple[Message, ...]`；按分支返回 `()`；
    # `await _legacy_context_messages(manager)`；
    # `await manager.context_messages(recall=snapshot)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_legacy_context_messages` →
    # `self._recall_query_with_task` → `manager.recall` → `manager.context_messages`。
    # 分支与异常：
    #   当 `manager is None` 时，返回 `()`。
    #   当 `not hybrid or self._recall_query is None` 时，返回
    # `await _legacy_context_messages(manager)`。
    #   捕获 `Exception` 后，返回 `await _legacy_context_messages(manager)`。
    #   当 `snapshot.mode is SearchMode.UNAVAILABLE` 时，返回
    # `await _legacy_context_messages(manager)`。
    # 副作用与资源：
    #   更新对象字段：`self._recall_snapshot`。
    async def _load_memory_messages(
        self,
        conversation_id: str | None,
    ) -> tuple[Message, ...]:

        manager = self._memory_manager
        if manager is None:
            return ()
        hybrid = bool(getattr(manager, "hybrid_recall_enabled", False))
        if not hybrid or self._recall_query is None:
            return await _legacy_context_messages(manager)
        try:
            query = await self._recall_query_with_task(conversation_id)
            snapshot = await manager.recall(query)
        except Exception:
            return await _legacy_context_messages(manager)
        self._recall_snapshot = snapshot
        if snapshot.mode is SearchMode.UNAVAILABLE:
            return await _legacy_context_messages(manager)
        return await manager.context_messages(recall=snapshot)

    # 函数说明：RuntimeContextSession._recall_query_with_task
    # 用途：召回查询任务，供模型与工具执行循环使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str | None`。
    # 返回：类型 `MemoryRecallQueryInputs`；返回
    # `self._recall_query.with_task(task_title, task_steps)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`callable` → `recall_fields` →
    # `self._recall_query.with_task`。
    async def _recall_query_with_task(
        self,
        conversation_id: str | None,
    ) -> MemoryRecallQueryInputs:

        assert self._recall_query is not None
        task_title: str | None = None
        task_steps: tuple[str, ...] = ()
        provider = self._task_context_provider
        recall_fields = getattr(provider, "recall_fields_for", None)
        if callable(recall_fields):
            task_title, task_steps = await recall_fields(conversation_id)
        return self._recall_query.with_task(task_title, task_steps)

    # 函数说明：RuntimeContextSession.activate_skill
    # 用途：加载并激活技能内容，同时更新本次运行可见的技能状态。
    # 参数：
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    #   emitter：执行事件发射器，类型 `EventEmitter`。
    #   step：当前任务步骤，类型 `int`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_read_outcome` →
    # `self._emit_activation_failed` → `self._skill_store.load` →
    # `provider.would_exceed_budget` → `emitter.emit` → `provider.active_tokens`。
    # 分支与异常：
    #   当 `self._skill_store is None or provider is None` 时，返回 `None`。
    #   当 `not skill_name` 时，返回 `None`。
    #   `not found` 分支在完成前置处理后返回 `None`。
    #   当 `skill_name in self._active_skills` 时，返回 `None`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def activate_skill(
        self,
        result: ToolResult,
        *,
        emitter: EventEmitter,
        step: int,
    ) -> None:
        """加载并激活技能内容，同时更新本次运行可见的技能状态。"""
        provider = self._skill_context_provider
        if self._skill_store is None or provider is None:
            return
        skill_name, found = skill_read_outcome(result.output)
        if not skill_name:
            return
        if not found:
            await self._emit_activation_failed(
                emitter,
                step=step,
                skill_name=skill_name,
                error="skill not found",
            )
            return
        if skill_name in self._active_skills:
            return
        skill = await self._skill_store.load(skill_name)
        if skill is None:
            await self._emit_activation_failed(
                emitter,
                step=step,
                skill_name=skill_name,
                error="skill not found",
            )
            return
        if provider.would_exceed_budget(
            tuple(self._active_skills.values()),
            skill,
        ):
            await self._emit_activation_failed(
                emitter,
                step=step,
                skill_name=skill_name,
                error="active skill context budget exceeded",
            )
            return
        self._active_skills[skill.metadata.name] = skill
        await emitter.emit(
            AgentEventType.SKILL_ACTIVATED,
            step=step,
            skill_name=skill.metadata.name,
            skill_scope=skill.metadata.scope.value,
            active_skill_names=tuple(self._active_skills),
            active_skill_tokens=provider.active_tokens(
                tuple(self._active_skills.values())
            ),
        )

    # 函数说明：RuntimeContextSession._emit_activation_failed
    # 用途：发出`activation_failed`，供模型与工具执行循环使用。
    # 参数：
    #   emitter：执行事件发射器，类型 `EventEmitter`。
    #   step：当前任务步骤，类型 `int`。
    #   skill_name：技能名称输入或配置值，类型 `str`。
    #   error：异常或错误信息，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`emitter.emit`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    @staticmethod
    async def _emit_activation_failed(
        emitter: EventEmitter,
        *,
        step: int,
        skill_name: str,
        error: str,
    ) -> None:
        await emitter.emit(
            AgentEventType.SKILL_ACTIVATION_FAILED,
            step=step,
            skill_name=skill_name,
            skill_error=error,
        )


# 函数说明：_legacy_context_messages
# 用途：返回 `await manager.context_messages()`，提供 模型与工具执行循环 的派生值。
# 参数：
#   manager：当前业务管理器，类型 `MemoryManager`。
# 返回：类型 `tuple[Message, ...]`；返回 `await manager.context_messages()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`manager.context_messages`。
async def _legacy_context_messages(manager: MemoryManager) -> tuple[Message, ...]:

    return await manager.context_messages()


__all__ = ["RuntimeContextInjection", "RuntimeContextSession"]
