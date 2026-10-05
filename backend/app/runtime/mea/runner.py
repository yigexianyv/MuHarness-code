
"""好评，MeaRunner：按阶段推进的 Manage → Execute → Audit 主循环。

每轮最多三个子 Run（Manager / Executor / Auditor），都是普通的 RunManager Run，
只是 mode、输入文本和 Runtime 不同。所有会影响外部状态的步骤都遵守两条原则：

- 子 Run 的 ID 先落盘再启动；Executor 是否启动以 runs 表里是否已有这个 Run 为准；
- 会写 Task 的操作内容先确定、先落盘，恢复时只重放、不按当前状态重算。

阶段每切换一次都先写库，进程在任何位置退出后，``run()`` 都能从已保存的阶段继续。
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from app.domain.task import (
    CLOSED_STEP_STATUSES,
    FileTaskStore,
    OpIdReusedError,
    OpPrecondition,
    OpResult,
    StepStatusChange,
    StepSupersede,
    Task,
    TaskOp,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
    steps_missing_acceptance,
)
from app.models.types import AgentMode
from app.runtime.agent.result import AgentResult, AgentStopReason
from app.runtime.run import RunAlreadyExists, RunStatus
from app.tools.role_boundary import count_role_rejections

from .models import (
    CLOSED_MEA_STATUSES,
    CLOSED_ROUND_PHASES,
    STEP_AUDIT_KINDS,
    TERMINAL_MEA_STATUSES,
    CompletionDecision,
    MeaRound,
    MeaRun,
    MeaStatus,
    OnceNote,
    RoundKind,
    RoundPhase,
    StepAuditPatch,
    now_utc,
)
from .parsing import (
    AuditStatus,
    ContractAudit,
    ControlHeader,
    Integrity,
    ManagerOutput,
    Route,
    StepAcceptance,
    apply_blocking_guard,
    clip_preserve,
    parse_control_header,
    parse_manager_output,
    strip_control_header,
)
from .prompts import (
    AuditKind,
    RoundView,
    StepView,
    SupersededView,
    boundary_rejection_note,
    build_auditor_prompt,
    build_executor_prompt,
    build_final_response_prompt,
    build_format_repair_prompt,
    build_manager_prompt,
    format_audit_findings,
    format_related_reports,
    invalid_final_audit_feedback,
    invalid_header_report,
    invalid_plan_feedback,
    no_report_placeholder,
    readonly_violation_report,
    recovery_record,
    requirements_updated_feedback,
    round_abandoned_feedback,
    supersede_rejected_feedback,
)
from .requirements import (
    AmendmentKind,
    Requirements,
    RequirementsTooLongError,
    add_amendment,
    build_requirements,
    render_requirements,
)
from .snapshot import snapshot_diff, snapshot_digest, snapshot_workspace
from .store import SQLiteMeaStore

logger = logging.getLogger("muharness.mea.runner")

MAX_INVALID_ROUTES = 3
MAX_ABANDONED_ROUNDS = 5
MAX_ROLE_FAILURES = 2
# 没有进展：同一个步骤（或最终验收）连续多轮没通过
STALL_SAME_WORKSPACE_ROUNDS = 3  # 且审计时的工作区快照完全相同：什么都没改变
STALL_MAX_ATTEMPTS = 6  # 工作区在变，但一直通不过验收
MAX_TRUNCATED_AUDITS = 2
MAX_TOTAL_INVALID_ROUTES = 6
REJECTIONS_CAP_INTEGRITY = 3
FINAL_RESPONSE_ATTEMPTS = 3
_MAX_STATE_ENTRIES = 100
_MAX_STATE_ENTRY_CHARS = 2_000
_MAX_CONTRACT_CHARS = 12_000
_TODO_OR_RUNNING = (TaskStepStatus.TODO, TaskStepStatus.IN_PROGRESS)
_FATAL_STOP_REASONS = frozenset({AgentStopReason.MODEL_ERROR, AgentStopReason.CONTEXT_ERROR})
ROLE_NAMES = {AgentMode.MANAGE: "manager", AgentMode.EXECUTE: "executor", AgentMode.AUDIT: "auditor"}
# 子 Run 只在这些状态下启动；取消、暂停到位、失败之后不再启动新的子 Run
_CHILD_START_STATUSES = frozenset({MeaStatus.RUNNING, MeaStatus.FINALIZING})
_LOOP_STOP_TIMEOUT = 30.0
_TIMED_OUT = object()


class RunGateway(Protocol):
    """RunManager 中 runner 用到的部分；测试里可以换成假的实现。"""

    # 函数说明：RunGateway.start
    # 用途：启动RunGateway，供规划、执行、审计协作使用。
    # 参数：
    #   user_message：当前用户消息，类型 `str`。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：类型 `tuple[str, Any]`；不返回结果值（隐式 None）。
    async def start(self, user_message: str, **kwargs: Any) -> tuple[str, Any]: ...

    # 函数说明：RunGateway.wait
    # 用途：等待RunGateway，供规划、执行、审计协作使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def wait(self, run_id: str) -> Any: ...

    # 函数说明：RunGateway.result
    # 用途：处理规划、执行、审计协作中的 `result` 数据；结果及边界条件见下方说明。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `AgentResult | None`；不返回结果值（隐式 None）。
    def result(self, run_id: str) -> AgentResult | None: ...

    # 函数说明：RunGateway.forget_results
    # 用途：清除结果集合，供规划、执行、审计协作使用。
    # 参数：
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def forget_results(self, run_ids: tuple[str, ...]) -> None: ...

    # 函数说明：RunGateway.get_run
    # 用途：获取运行，供规划、执行、审计协作使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def get_run(self, run_id: str) -> Any: ...

    # 函数说明：RunGateway.active_run_ids
    # 用途：运行`ids`，供规划、执行、审计协作使用。
    # 返回：类型 `tuple[str, ...]`；不返回结果值（隐式 None）。
    @property
    def active_run_ids(self) -> tuple[str, ...]: ...

    # 函数说明：RunGateway.cancel
    # 用途：取消RunGateway，供规划、执行、审计协作使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def cancel(self, run_id: str) -> Any: ...


FinalMessageSink = Callable[[str, str, str], Awaitable[object]]
RecoveryInfo = Callable[[str], Awaitable[tuple[list[str], list[str]]]]
ExecutorEvidenceInfo = Callable[[str, str, str], Awaitable[str]]


class MeaStartError(ValueError):
    pass


class RoundConflict(RuntimeError):
    pass


class MeaFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AmendResult:

    accepted: bool
    reason: str | None = None
    revision: int | None = None
    amendment_id: str | None = None


@dataclass(frozen=True, slots=True)
class SubRunResult:

    run_id: str
    text: str
    ok: bool
    cancelled: bool
    stop_reason: str | None = None
    rejections: Counter[str] = field(default_factory=Counter)
    timed_out: bool = False
    model_finish_reason: str | None = None

    # 函数说明：SubRunResult.truncated
    # 用途：返回 `self.model_finish_reason in {'max_tokens', 'length'}`，提供
    # SubRunResult 的派生值。
    # 返回：类型 `bool`；返回 `self.model_finish_reason in {'max_tokens', 'length'}`。
    @property
    def truncated(self) -> bool:
        return self.model_finish_reason in {"max_tokens", "length"}


@dataclass(frozen=True, slots=True)
class _PlanDecision:

    route: Route
    kind: RoundKind
    step_id: str | None
    proposal: TaskOp | None
    feedback: tuple[str, ...]
    question: str | None = None
    choices: tuple[str, ...] = ()


class MeaRunner:

    # 函数说明：MeaRunner.__init__
    # 用途：初始化 MeaRunner；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteMeaStore`。
    #   tasks：任务集合输入或配置值，类型 `FileTaskStore`。
    #   runs：运行集合输入或配置值，类型 `RunGateway`。
    #   runtimes：传给 `set` 的输入，类型 `Mapping[AgentMode, Any]`。
    #   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path`。
    #   snapshot_exclude：传给 `tuple` 的输入，类型 `Sequence[str | Path]`；默认 `()`。
    #   final_message_sink：消息输入或配置值，类型 `FinalMessageSink | None`；默认
    # `None`。
    #   recovery_info：恢复信息输入或配置值，类型 `RecoveryInfo | None`；默认 `None`。
    #   role_timeouts：角色输入或配置值，类型 `Mapping[AgentMode, float | None] | None`
    # ；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(workspace_root).expanduser().resolve` → `Path(workspace_root).expanduser` →
    # `Path` → `Counter`。
    # 分支与异常：
    #   当 `missing` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   更新对象字段：`self._store`、`self._tasks`、`self._runs`、`self._runtimes`、
    # `self._runtimes_source`、`self._workspace`、`self._exclude`、`self._final_sink` 等
    #  15 个字段。
    def __init__(
        self,
        *,
        store: SQLiteMeaStore,
        tasks: FileTaskStore,
        runs: RunGateway,
        runtimes: Mapping[AgentMode, Any],
        workspace_root: str | Path,
        snapshot_exclude: Sequence[str | Path] = (),
        final_message_sink: FinalMessageSink | None = None,
        recovery_info: RecoveryInfo | None = None,
        executor_evidence: ExecutorEvidenceInfo | None = None,
        role_timeouts: Mapping[AgentMode, float | None] | None = None,
    ) -> None:
        missing = {AgentMode.MANAGE, AgentMode.EXECUTE, AgentMode.AUDIT} - set(runtimes)
        if missing:
            raise ValueError(f"missing role runtimes: {sorted(m.value for m in missing)}")
        self._store = store
        self._tasks = tasks
        self._runs = runs
        self._runtimes = dict(runtimes)
        self._runtimes_source = runtimes  # RoleRuntimes 还能按额外工具取 Executor Runtime
        self._workspace = Path(workspace_root).expanduser().resolve()
        self._exclude = tuple(snapshot_exclude)
        self._final_sink = final_message_sink
        self._recovery_info = recovery_info
        self._executor_evidence = executor_evidence
        self._timeouts = dict(role_timeouts or {})
        self._start_lock = asyncio.Lock()
        self._loop_locks: dict[str, asyncio.Lock] = {}
        self._amend_locks: dict[str, asyncio.Lock] = {}
        self._loops: dict[str, asyncio.Task[MeaRun]] = {}
        self._active_child: dict[str, str] = {}
        self._role_failures: Counter[str] = Counter()

    # ================================================================ 公开接口

    # 函数说明：MeaRunner.start
    # 用途：为一个已接受的计划启动长任务。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str`。
    #   original_request：传给 `self.preflight` 的输入，类型 `str`。
    #   round_budget：预算输入或配置值，类型 `int`；默认 `25`。
    #   extra_tools：传给 `tuple` 的输入，类型 `Sequence[str]`；默认 `()`。
    #   auto_approve_sandbox：`auto_approve_sandbox`输入或配置值，类型 `bool`；默认
    # `True`。
    #   spawn：`spawn`输入或配置值，类型 `bool`；默认 `True`。
    # 返回：类型 `MeaRun`；返回 `run`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._tasks.resolve` →
    # `self.preflight` → `now_utc` → `MeaRun` → `uuid4` → `self._store.create_run`；另有
    #  1 个调用点。
    # 分支与异常：
    #   当 `task is None` 时，抛出 `MeaStartError(f'任务不存在：{task_id}')`。
    #   当 `task.status is not TaskStatus.ACTIVE` 时，抛出
    # `MeaStartError(f'只有进行中的任务可以启动长任务（当前 {task.status.value}）')`。
    async def start(
        self,
        *,
        task_id: str,
        conversation_id: str,
        original_request: str,
        round_budget: int = 25,
        extra_tools: Sequence[str] = (),
        auto_approve_sandbox: bool = True,
        spawn: bool = True,
        restart: bool = False,
    ) -> MeaRun:
        """为一个已接受的计划启动长任务。缺验收标准或要求超长时拒绝启动。"""

        async with self._start_lock:
            task = await self._tasks.resolve(
                task_id, owner_conversation_id=conversation_id,
            )
            if task is None:
                raise MeaStartError(f"任务不存在：{task_id}")
            if await self.busy_run(conversation_id) is not None:
                raise MeaStartError("当前会话有正在执行的长任务，请先等待结束或取消")
            if restart:
                await self.preflight(task, original_request)
                task = await self._tasks.restart(task.id)
            if task.status is not TaskStatus.ACTIVE:
                raise MeaStartError(f"只有进行中的任务可以启动长任务（当前 {task.status.value}）")
            requirements = await self.preflight(task, original_request)
            now = now_utc()
            run = MeaRun(
                id=uuid4().hex,
                task_id=task.id,
                conversation_id=conversation_id,
                round_budget=round_budget,
                requirements_revision=requirements.revision,
                extra_tools=tuple(extra_tools),
                auto_approve_sandbox=auto_approve_sandbox,
                created_at=now,
                updated_at=now,
            )
            await self._store.create_run(run, requirements)
            if spawn:
                self.spawn(run.id)
            return run

    # 函数说明：MeaRunner.preflight
    # 用途：启动前检查（不看 Task 状态，方便调用方先检查、再接受计划）。
    # 参数：
    #   task：当前任务记录，类型 `Task`。
    #   original_request：请求输入或配置值，类型 `str`。
    # 返回：类型 `Requirements`；返回 `build_requirements(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`steps_missing_acceptance` →
    # `self._store.list_runs` → `build_requirements`。
    # 分支与异常：
    #   当 `not task.steps` 时，抛出 `MeaStartError('任务没有步骤')`。
    #   当 `missing` 时，抛出
    # `MeaStartError('以下步骤缺少验收标准，请重新规划：' + '、'.join(missing))`。
    #   当 `any(…)` 时，抛出 `MeaStartError('这个任务已经有一个未结束的长任务')`。
    #   捕获 `RequirementsTooLongError` 后，转换或抛出 `MeaStartError(str(exc))`。
    async def preflight(self, task: Task, original_request: str) -> Requirements:
        """启动前检查（不看 Task 状态，方便调用方先检查、再接受计划）。返回初始权威要求。"""

        if not task.steps:
            raise MeaStartError("任务没有步骤")
        missing = steps_missing_acceptance(task)
        if missing:
            raise MeaStartError("以下步骤缺少验收标准，请重新规划：" + "、".join(missing))
        existing = await self._store.list_runs(conversation_id=task.owner_conversation_id)
        if any(run.task_id == task.id and run.status not in CLOSED_MEA_STATUSES for run in existing):
            raise MeaStartError("这个任务已经有一个未结束的长任务")
        try:
            return build_requirements(
                original_request=original_request,
                goal=task.goal,
                description=task.description,
                constraints=task.constraints,
            )
        except RequirementsTooLongError as exc:  # 超过 16,000 字符：拒绝启动，不截断
            raise MeaStartError(str(exc)) from exc

    # 函数说明：MeaRunner.open_runs
    # 用途：会话里还没有结束（含 finalizing）的长任务。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `tuple[MeaRun, ...]`；返回
    # `tuple((run for run in runs if run.status not in TERMINAL_MEA_STATUSES))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.list_runs`。
    async def open_runs(self, conversation_id: str) -> tuple[MeaRun, ...]:
        """会话里还没有结束（含 finalizing）的长任务。"""

        runs = await self._store.list_runs(conversation_id=conversation_id, limit=1_000)
        return tuple(run for run in runs if run.status not in TERMINAL_MEA_STATUSES)

    # 函数说明：MeaRunner.busy_run
    # 用途：正在推进（running / finalizing）的长任务；这时会话不接受普通消息。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `MeaRun | None`；按分支返回 `run`；`None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.open_runs`。
    # 分支与异常：
    #   当 `run.status in _CHILD_START_STATUSES` 时，返回 `run`。
    async def busy_run(self, conversation_id: str) -> MeaRun | None:
        """正在推进（running / finalizing）的长任务；这时会话不接受普通消息。"""

        for run in await self.open_runs(conversation_id):
            if run.status in _CHILD_START_STATUSES:
                return run
        return None

    # 函数说明：MeaRunner.cancel_for_conversation
    # 用途：删除会话前调用：取消所有未结束的长任务，并等循环退出。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `int`；返回 `len(runs)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.open_runs` → `self.cancel` →
    #  `self._stop_loop`。
    async def cancel_for_conversation(self, conversation_id: str) -> int:
        """删除会话前调用：取消所有未结束的长任务，并等循环退出。"""

        runs = await self.open_runs(conversation_id)
        for run in runs:
            await self.cancel(run.id)
        for run in runs:
            await self._stop_loop(run.id)
        return len(runs)

    # 函数说明：MeaRunner.shutdown
    # 用途：Host 关闭：停止循环，不改状态。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`task.done` → `task.cancel` →
    # `self._loops.clear`。
    # 分支与异常：
    #   捕获 `(asyncio.CancelledError, Exception)` 后，忽略该异常并继续当前流程。
    async def shutdown(self) -> None:
        """Host 关闭：停止循环，不改状态。下次启动时 reconcile 把 running 改为 paused。"""

        for mea_id in list(self._loops):
            task = self._loops.get(mea_id)
            if task is not None and not task.done():
                task.cancel()
        for task in list(self._loops.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._loops.clear()

    # 函数说明：MeaRunner._stop_loop
    # 用途：停止`loop`，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`task.done` → `asyncio.wait_for` →
    #  `asyncio.shield` → `task.cancel`。
    # 分支与异常：
    #   当 `task is None or task.done()` 时，返回 `None`。
    #   捕获 `(TimeoutError, asyncio.CancelledError, Exception)` 后，执行异常处理调用
    # `task.cancel`。
    #   捕获 `(asyncio.CancelledError, Exception)` 后，忽略该异常并继续当前流程。
    async def _stop_loop(self, mea_id: str) -> None:
        task = self._loops.get(mea_id)
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), _LOOP_STOP_TIMEOUT)
        except (TimeoutError, asyncio.CancelledError, Exception):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # 函数说明：MeaRunner.spawn
    # 用途：启动MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `asyncio.Task[MeaRun]`；按分支返回 `current`；`task`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`current.done` →
    # `asyncio.create_task` → `self._run_logged`。
    # 分支与异常：
    #   当 `current is not None and (not current.done())` 时，返回 `current`。
    # 副作用与资源：
    #   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
    def spawn(self, mea_id: str) -> asyncio.Task[MeaRun]:
        current = self._loops.get(mea_id)
        if current is not None and not current.done():
            return current
        task = asyncio.create_task(self._run_logged(mea_id))
        self._loops[mea_id] = task
        return task

    # 函数说明：MeaRunner.wait
    # 用途：等待MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `MeaRun`；返回 `await self._store.require(mea_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.shield` →
    # `self._store.require`。
    async def wait(self, mea_id: str) -> MeaRun:
        task = self._loops.get(mea_id)
        if task is not None:
            await asyncio.shield(task)
        return await self._store.require(mea_id)

    # 函数说明：MeaRunner.run
    # 用途：推进一个长任务直到它离开 running（完成、请示、暂停、失败……）。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `MeaRun`；返回 `mea`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock` →
    # `self._store.require` → `self._current_round` → `self._set_status` →
    # `self._counted_rounds` → `self._pause_out_of_rounds`；另有 8 个调用点。
    # 资源/并发边界：`self._lock(self._loop_locks, mea_id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   `mea.pause_requested` 分支在完成前置处理后结束当前循环。
    #   `await self._counted_rounds(mea.id) >= mea.round_budget` 分支在完成前置处理后结
    # 束当前循环。
    #   捕获 `RoundConflict` 后，跳过当前循环项，继续处理后续项。
    #   捕获 `(OpIdReusedError, MeaFailure)` 后，结束所在循环。
    async def run(self, mea_id: str) -> MeaRun:
        """推进一个长任务直到它离开 running（完成、请示、暂停、失败……）。"""

        async with self._lock(self._loop_locks, mea_id):
            mea = await self._store.require(mea_id)
            while mea.status is MeaStatus.RUNNING:
                rnd = await self._current_round(mea.id)
                if rnd is None:
                    if mea.pause_requested:
                        mea = await self._set_status(mea, MeaStatus.PAUSED, pause_requested=False)
                        break
                    if await self._counted_rounds(mea.id) >= mea.round_budget:
                        mea = await self._pause_out_of_rounds(mea)
                        break
                    rnd = await self._open_next_round(mea)
                try:
                    rnd = await self._advance(mea, rnd)
                except RoundConflict as exc:
                    rnd = await self._store.save_round(
                        rnd.append_feedback(f"步骤状态冲突: {exc}").model_copy(
                            update={"phase": RoundPhase.APPLIED}
                        )
                    )
                    mea = await self._wait_for_user(
                        mea, f"步骤状态和长任务记录不一致（{exc}），需要你确认后再继续。"
                    )
                    continue
                except (OpIdReusedError, MeaFailure) as exc:
                    mea = await self._fail(mea, f"{type(exc).__name__}: {exc}")
                    break
                mea = await self._after_round(mea, rnd)
            if mea.status is MeaStatus.FINALIZING:
                mea = await self._finalize(mea)
            return mea

    # 函数说明：MeaRunner.add_amendment
    # 用途：追加持续有效的要求。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    #   text：待处理的文本，类型 `str`。
    #   kind：`kind`输入或配置值，类型 `AmendmentKind`；默认 `AmendmentKind.NOTE`。
    #   immediate：`immediate`输入或配置值，类型 `bool`；默认 `False`。
    # 返回：类型 `AmendResult`；按分支返回
    # `AmendResult(accepted=False, reason='mea_finalized')`；
    # `AmendResult(accepted=False, reason=f'requirements_too_long: {exc}')`；
    # `AmendResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock` →
    # `self._store.require` → `AmendResult` → `self._store.requirements` →
    # `add_amendment` → `self._store.save_requirements`；另有 5 个调用点。
    # 资源/并发边界：`self._lock(self._amend_locks, mea_id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `mea.status in CLOSED_MEA_STATUSES` 时，返回
    # `AmendResult(accepted=False, reason='mea_finalized')`。
    #   捕获 `RequirementsTooLongError` 后，返回
    # `AmendResult(accepted=False, reason=f'requirements_too_long: {exc}')`。
    async def add_amendment(
        self,
        mea_id: str,
        text: str,
        *,
        kind: AmendmentKind = AmendmentKind.NOTE,
        immediate: bool = False,
    ) -> AmendResult:
        """追加持续有效的要求。终结决定之后返回 mea_finalized，由前端作为新请求发送。"""

        resumed = False
        async with self._lock(self._amend_locks, mea_id):
            mea = await self._store.require(mea_id)
            if mea.status in CLOSED_MEA_STATUSES:
                return AmendResult(accepted=False, reason="mea_finalized")
            current = await self._store.requirements(mea_id)
            try:
                updated = add_amendment(current, kind=kind, text=text)
            except RequirementsTooLongError as exc:
                return AmendResult(accepted=False, reason=f"requirements_too_long: {exc}")
            mea = await self._store.save_requirements(mea, updated)
            # 要求变了就是新情况：停滞检测从这里重新计数
            mea = await self._store.save_run(
                mea.model_copy(update={"stall_guard_after": await self._last_round_index(mea_id)})
            )
            if kind is AmendmentKind.ANSWER and mea.status is MeaStatus.WAITING_USER:
                mea = await self._set_status(
                    mea, MeaStatus.RUNNING, pending_question=None, pending_choices=()
                )
                resumed = True
            child = self._active_child.get(mea_id)
        if immediate and child is not None:
            await self._cancel_child(child)
        if resumed:
            self.spawn(mea_id)
        return AmendResult(
            accepted=True,
            revision=updated.revision,
            amendment_id=updated.amendments[-1].id,
        )

    # 函数说明：MeaRunner.answer
    # 用途：处理回复MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `AmendResult`；返回
    # `await self.add_amendment(mea_id, text, kind=AmendmentKind.ANSWER)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.add_amendment`。
    async def answer(self, mea_id: str, text: str) -> AmendResult:
        return await self.add_amendment(mea_id, text, kind=AmendmentKind.ANSWER)

    # 函数说明：MeaRunner.add_once_note
    # 用途：添加`once_note`，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `MeaRun`；返回
    # `await self._store.save_run(mea.model_copy(update={'once_notes': notes}))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock` →
    # `self._store.require` → `OnceNote` → `self._store.save_run`。
    # 资源/并发边界：`self._lock(self._amend_locks, mea_id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `mea.status in CLOSED_MEA_STATUSES` 时，抛出 `ValueError('mea_finalized')`。
    async def add_once_note(self, mea_id: str, text: str) -> MeaRun:
        async with self._lock(self._amend_locks, mea_id):
            mea = await self._store.require(mea_id)
            if mea.status in CLOSED_MEA_STATUSES:
                raise ValueError("mea_finalized")
            notes = (*mea.once_notes, OnceNote(text=text.strip()))
            return await self._store.save_run(mea.model_copy(update={"once_notes": notes}))

    # 函数说明：MeaRunner.pause
    # 用途：暂停MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `MeaRun`；按分支返回
    # `await self._store.save_run(mea.model_copy(update={'pause_requested': True}))`；
    # `await self._set_status(mea, MeaStatus.PAUSED)`；`mea`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock` →
    # `self._store.require` → `self._store.save_run` → `self._set_status`。
    # 资源/并发边界：`self._lock(self._amend_locks, mea_id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `mea.status is MeaStatus.RUNNING` 时，返回 `await self._store.save_run(…)`。
    #   当 `mea.status is MeaStatus.WAITING_USER` 时，返回
    # `await self._set_status(mea, MeaStatus.PAUSED)`。
    async def pause(self, mea_id: str) -> MeaRun:
        async with self._lock(self._amend_locks, mea_id):
            mea = await self._store.require(mea_id)
            if mea.status is MeaStatus.RUNNING:
                return await self._store.save_run(mea.model_copy(update={"pause_requested": True}))
            if mea.status is MeaStatus.WAITING_USER:
                return await self._set_status(mea, MeaStatus.PAUSED)
            return mea

    # 函数说明：MeaRunner.resume
    # 用途：恢复MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    #   extra_rounds：传给 `max` 的输入，类型 `int`；默认 `0`。
    # 返回：类型 `MeaRun`；返回 `mea`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock` →
    # `self._store.require` → `self._set_status` → `self._last_round_index` →
    # `self.spawn`。
    # 资源/并发边界：`self._lock(self._amend_locks, mea_id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `mea.status is not MeaStatus.FINALIZING and mea.status is…` 时，返回 `mea`。
    async def resume(self, mea_id: str, *, extra_rounds: int = 0) -> MeaRun:
        async with self._lock(self._amend_locks, mea_id):
            mea = await self._store.require(mea_id)
            if mea.status in (MeaStatus.PAUSED, MeaStatus.WAITING_USER):
                # 不回答直接继续：清掉待回答的问题，Manager 下一轮按现有要求重新判断
                mea = await self._set_status(
                    mea,
                    MeaStatus.RUNNING,
                    pause_requested=False,
                    pending_question=None,
                    pending_choices=(),
                    round_budget=mea.round_budget + max(0, extra_rounds),
                    stall_guard_after=await self._last_round_index(mea_id),
                )
            elif mea.status is not MeaStatus.FINALIZING and mea.status is not MeaStatus.RUNNING:
                return mea
        self.spawn(mea_id)
        return mea

    # 函数说明：MeaRunner.cancel
    # 用途：取消MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `MeaRun`；返回 `mea`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock` →
    # `self._store.require` → `self._set_status` → `self._cancel_child`。
    # 资源/并发边界：`self._lock(self._amend_locks, mea_id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `mea.status in CLOSED_MEA_STATUSES` 时，返回 `mea`。
    async def cancel(self, mea_id: str) -> MeaRun:
        async with self._lock(self._amend_locks, mea_id):
            mea = await self._store.require(mea_id)
            if mea.status in CLOSED_MEA_STATUSES:
                return mea
            mea = await self._set_status(mea, MeaStatus.CANCELLED, abort_reason="用户取消")
            child = self._active_child.get(mea_id)
        if child is not None:
            await self._cancel_child(child)
        return mea

    # 函数说明：MeaRunner.interrupt_child
    # 用途：取消当前子 Run（一次性补充 + 立即生效）。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `bool`；按分支返回 `False`；`True`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._cancel_child`。
    # 分支与异常：
    #   当 `child is None` 时，返回 `False`。
    async def interrupt_child(self, mea_id: str) -> bool:
        """取消当前子 Run（一次性补充 + 立即生效）。Executor 被取消后本轮进入核查审计。"""

        child = self._active_child.get(mea_id)
        if child is None:
            return False
        await self._cancel_child(child)
        return True

    # 函数说明：MeaRunner.reconcile
    # 用途：Host 启动时调用：running 改为 paused 等用户点继续；finalizing 直接补齐。
    # 返回：类型 `tuple[MeaRun, ...]`；返回 `tuple(changed)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.list_runs` →
    # `self._set_status` → `self.spawn`。
    async def reconcile(self) -> tuple[MeaRun, ...]:
        """Host 启动时调用：running 改为 paused 等用户点继续；finalizing 直接补齐。"""

        changed: list[MeaRun] = []
        for mea in await self._store.list_runs(status=MeaStatus.RUNNING, limit=1_000):
            changed.append(await self._set_status(mea, MeaStatus.PAUSED))
        for mea in await self._store.list_runs(status=MeaStatus.FINALIZING, limit=1_000):
            self.spawn(mea.id)
            changed.append(mea)
        return tuple(changed)

    # ================================================================ 轮次推进

    # 函数说明：MeaRunner._advance
    # 用途：推进MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `MeaRound`；按分支返回 `await self._manage(mea, rnd)`；
    # `await self._apply_plan(mea, rnd)`；`await self._execute(mea, rnd)`；
    # `await self._audit(mea, rnd)` 等 6 种表达式。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._manage` →
    # `self._apply_plan` → `self._execute` → `self._audit` → `self._apply_verdict` →
    # `self._store.save_round`。
    async def _advance(self, mea: MeaRun, rnd: MeaRound) -> MeaRound:
        match rnd.phase:
            case RoundPhase.MANAGING:
                return await self._manage(mea, rnd)
            case RoundPhase.PLANNED:
                return await self._apply_plan(mea, rnd)
            case RoundPhase.EXECUTING:
                return await self._execute(mea, rnd)
            case RoundPhase.EXECUTED | RoundPhase.AUDITING:
                return await self._audit(mea, rnd)
            case RoundPhase.AUDITED:
                return await self._apply_verdict(mea, rnd)
            case RoundPhase.INTERRUPTED:
                return await self._store.save_round(rnd.model_copy(update={"phase": RoundPhase.APPLIED}))
        raise AssertionError(f"round {rnd.ref} cannot advance from {rnd.phase}")

    # 函数说明：MeaRunner._manage
    # 用途：管理MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `MeaRound`；按分支返回
    # `await self._interrupt(rnd, 'Manager 运行被取消')`；
    # `await self._store.save_round(rnd)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.requirements` →
    # `self._task` → `self._store.rounds` → `build_manager_prompt` →
    # `render_requirements` → `_step_views`；另有 13 个调用点。
    # 分支与异常：
    #   当 `result.cancelled` 时，返回
    # `await self._interrupt(rnd, 'Manager 运行被取消')`。
    async def _manage(self, mea: MeaRun, rnd: MeaRound) -> MeaRound:
        requirements = await self._store.requirements(mea.id)
        task = await self._task(mea)
        rounds = await self._store.rounds(mea.id)
        notes = [note.text for note in mea.once_notes if note.consumed_in_round is None]
        prompt = build_manager_prompt(
            requirements_text=render_requirements(requirements),
            steps=_step_views(task, rounds),
            contract=task.contract,
            state="\n".join(task.state),
            rounds=_round_views(rounds),
            round_index=await self._counted_rounds(mea.id),
            round_budget=mea.round_budget,
            once_notes=notes,
        )
        run_id = uuid4().hex  # Manager 没有副作用，恢复时总是换新 ID 重跑
        rnd = await self._store.save_round(rnd.model_copy(update={"manager_run_id": run_id}))
        result = await self._run_role(mea, run_id, AgentMode.MANAGE, prompt)
        if result.cancelled:
            return await self._interrupt(rnd, "Manager 运行被取消")
        output = parse_manager_output(result.text)
        if result.truncated or not result.ok:
            decision = _PlanDecision(
                Route.INVALID, RoundKind.NORMAL, None, None,
                (invalid_plan_feedback("被截断或未正常结束；缩短状态和契约，先输出路由，禁止应用半份计划"),),
            )
        else:
            decision = self._decide_plan(output, task, requirements, rnd)
        if notes:
            consumed = tuple(
                note.model_copy(update={"consumed_in_round": rnd.index})
                if note.consumed_in_round is None
                else note
                for note in mea.once_notes
            )
            await self._store.save_run(mea.model_copy(update={"once_notes": consumed}))
        update: dict[str, Any] = {
            "phase": RoundPhase.PLANNED,
            "route": decision.route.value,
            "kind": decision.kind,
            "step_id": decision.step_id,
            "plan_text": output.plan_text or result.text,
            "subtask": output.subtask,
            "focus": output.focus,
            "related_refs": output.related_refs,
            "plan_proposal": decision.proposal,
            "plan_base_task_revision": task.revision,
            "plan_requirements_revision": requirements.revision,
        }
        rnd = rnd.model_copy(update=update)
        for text in decision.feedback:
            rnd = rnd.append_feedback(text)
        if decision.route is Route.ASK:
            rnd = rnd.model_copy(update={"focus": decision.question})
            await self._store.save_run(
                (await self._store.require(mea.id)).model_copy(
                    update={"pending_question": decision.question,
                            "pending_choices": decision.choices}
                )
            )
        return await self._store.save_round(rnd)

    # 函数说明：MeaRunner._apply_plan
    # 用途：应用计划，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `MeaRound`；按分支返回
    # `await self._abandon(mea, rnd, plan_applied=False, stale_revision=current)`；
    # `await self._abandon(…)`；
    # `await self._store.save_round(rnd.model_copy(update={'phase': phase}))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._revision` → `self._abandon`
    #  → `self._tasks.apply_op` → `Route` → `self._store.save_round`。
    # 分支与异常：
    #   当 `rnd.plan_requirements_revision != current` 时，返回 `await self._abandon(…)`
    # 。
    #   当 `outcome.result is OpResult.CONFLICT` 时，返回 `await self._abandon(…)`。
    async def _apply_plan(self, mea: MeaRun, rnd: MeaRound) -> MeaRound:
        current = await self._revision(mea.id)
        if rnd.plan_requirements_revision != current:
            return await self._abandon(mea, rnd, plan_applied=False, stale_revision=current)
        if rnd.plan_proposal is not None:
            outcome = await self._tasks.apply_op(mea.task_id, rnd.plan_proposal)
            if outcome.result is OpResult.CONFLICT:
                return await self._abandon(
                    mea, rnd, plan_applied=False, reason=f"Task 在本轮期间被修改（{outcome.reason}）"
                )
        route = Route(rnd.route or Route.INVALID.value)
        if route is Route.EXECUTE:
            phase = RoundPhase.EXECUTING
        elif route in (Route.AUDIT_ONLY, Route.FINAL_AUDIT):
            phase = RoundPhase.EXECUTED
        else:
            phase = RoundPhase.APPLIED  # 请示 / 阻塞 / 路由无效：交给 _after_round
        return await self._store.save_round(rnd.model_copy(update={"phase": phase}))

    # 函数说明：MeaRunner._execute
    # 用途：执行MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `MeaRound`；按分支返回 `await self._abandon(…)`；`rnd`；
    # `await self._interrupt(rnd, '进程重启时 Executor 没有正常结束')`；
    # `await self._interrupt(rnd, 'Executor 被取消')` 等 6 种表达式。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._runs.get_run` →
    # `self._lock` → `self._revision` → `self._abandon` → `self._store.require` →
    # `self._task`；另有 15 个调用点。
    # 资源/并发边界：`self._lock(self._amend_locks, mea.id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `rnd.plan_requirements_revision != current` 时，返回 `await self._abandon(…)`
    # 。
    #   当 `(await self._store.require(mea.id)).status is not…` 时，返回 `rnd`。
    #   捕获 `RunAlreadyExists` 后，忽略该异常并继续当前流程。
    #   当 `run_id not in self._runs.active_run_ids` 时，返回
    # `await self._interrupt(rnd, '进程重启时 Executor 没有正常结束')`。
    async def _execute(self, mea: MeaRun, rnd: MeaRound) -> MeaRound:
        run_id = rnd.executor_run_id
        existing = await self._runs.get_run(run_id) if run_id else None
        if existing is None:
            # 含“ID 已绑定、Run 未创建”：锁内重新检查要求版本，只保护“检查 + 创建 Run 记录”
            async with self._lock(self._amend_locks, mea.id):
                current = await self._revision(mea.id)
                if rnd.plan_requirements_revision != current:
                    return await self._abandon(
                        mea, rnd, plan_applied=rnd.plan_proposal is not None,
                        stale_revision=current, release=True,
                    )
                if (await self._store.require(mea.id)).status is not MeaStatus.RUNNING:
                    return rnd
                task = await self._task(mea)
                await self._begin_step(mea, rnd)
                if run_id is None:
                    run_id = uuid4().hex
                    rnd = await self._store.save_round(
                        rnd.model_copy(
                            update={"executor_run_id": run_id,
                                    "exec_requirements_revision": current}
                        )
                    )
                requirements = await self._store.requirements(mea.id, current)
                step = _step(task, rnd.step_id)
                rounds = await self._store.rounds(mea.id)
                prompt = build_executor_prompt(
                    requirements_text=render_requirements(requirements),
                    contract=task.contract,
                    state="\n".join(task.state),
                    step=_step_view(step, rounds),
                    subtask=rnd.subtask or "",
                    related_reports=format_related_reports(_round_views(rounds), rnd.related_refs),
                    workspace_root=str(self._workspace),
                )
                try:
                    await self._start_role(
                        mea,
                        run_id,
                        AgentMode.EXECUTE,
                        prompt,
                        task_step_id=rnd.step_id,
                    )
                except RunAlreadyExists:
                    pass  # 已经创建过：绝不再启动，下面按已有 Run 等待
        elif run_id not in self._runs.active_run_ids:
            return await self._interrupt(rnd, "进程重启时 Executor 没有正常结束")
        assert run_id is not None
        result = await self._wait_role(mea, run_id, AgentMode.EXECUTE)
        if result.cancelled:
            return await self._interrupt(rnd, "Executor 被取消")
        if result.timed_out:  # 被强制停止，和中断一样先做核查审计
            return await self._interrupt(rnd, result.text)
        self._note_role_result(mea, result)  # 连续两次模型 / 上下文错误才终止；预算类停止照常交给审计
        return await self._store.save_round(
            rnd.model_copy(
                update={
                    "phase": RoundPhase.EXECUTED,
                    "executor_output": result.text,
                    "executor_rejections": dict(result.rejections),
                }
            )
        )

    # 函数说明：MeaRunner._audit
    # 用途：审计MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `MeaRound`；按分支返回 `await self._store.save_round(rnd.model_copy(
    # update={'phase': RoundPhase.EXECUTED}))`；`await self._store.save_round(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._begin_step` →
    # `self._revision` → `self._store.requirements` → `self._task` →
    # `self._store.rounds` → `snapshot_workspace`；另有 6 个调用点。
    # 分支与异常：
    #   当 `result.cancelled` 时，返回 `await self._store.save_round(…)`。
    async def _audit(self, mea: MeaRun, rnd: MeaRound) -> MeaRound:
        if rnd.kind in STEP_AUDIT_KINDS:
            await self._begin_step(mea, rnd)  # 幂等：接管 todo 或中断轮留下的 in_progress
        current = await self._revision(mea.id)
        requirements = await self._store.requirements(mea.id, current)
        task = await self._task(mea)
        rounds = await self._store.rounds(mea.id)
        before = snapshot_workspace(self._workspace, exclude=self._exclude)
        run_id = uuid4().hex
        rnd = await self._store.save_round(
            rnd.model_copy(
                update={
                    "phase": RoundPhase.AUDITING,
                    "auditor_run_id": run_id,
                    "audit_requirements_revision": current,
                    "snapshot_before": snapshot_digest(before),
                }
            )
        )
        prompt = await self._auditor_prompt(mea, rnd, task, requirements, rounds)
        result = await self._run_role(
            mea,
            run_id,
            AgentMode.AUDIT,
            prompt,
            task_step_id=rnd.step_id,
        )
        if result.cancelled:
            # 审计只读：取消后回到 executed，下次换新 ID 按最新要求重审
            return await self._store.save_round(rnd.model_copy(update={"phase": RoundPhase.EXECUTED}))
        header, report = await self._finalize_audit(mea, rnd, result, before)
        return await self._store.save_round(
            rnd.model_copy(
                update={
                    "phase": RoundPhase.AUDITED,
                    "auditor_report": report,
                    "audit_output_truncated": result.truncated,
                    "audit_status": header.status.value,
                    "integrity_status": header.integrity.value,
                    "contract_audit_status": header.contract_audit.value,
                    "step_acceptance": header.step_acceptance.value,
                }
            )
        )

    # 函数说明：MeaRunner._apply_verdict
    # 用途：应用`verdict`，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `MeaRound`；按分支返回 `rnd`；`await self._store.save_round(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock` →
    # `self._store.require` → `_passes` → `CompletionDecision` → `now_utc` →
    # `self._store.save_run_and_round`；另有 9 个调用点。
    # 资源/并发边界：`self._lock(self._amend_locks, mea.id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   `rnd.kind is RoundKind.FINAL_AUDIT` 分支在完成前置处理后返回
    # `await self._store.save_round(…)`。
    #   `not stale and _passes(rnd, final=True) and (mea.status is…` 分支在完成前置处理
    # 后返回 `rnd`。
    async def _apply_verdict(self, mea: MeaRun, rnd: MeaRound) -> MeaRound:
        async with self._lock(self._amend_locks, mea.id):  # 与 add_amendment 互斥
            mea = await self._store.require(mea.id)
            current = mea.requirements_revision
            if rnd.kind is RoundKind.FINAL_AUDIT:  # 不写步骤；终结决定是单个事务，重算安全
                stale = rnd.audit_requirements_revision != current
                if not stale and _passes(rnd, final=True) and mea.status is MeaStatus.RUNNING:
                    decision = CompletionDecision(
                        outcome="completed",
                        round_index=rnd.index,
                        auditor_run_id=rnd.auditor_run_id,
                        requirements_revision=current,
                        decided_at=now_utc(),
                    )
                    _, rnd = await self._store.save_run_and_round(
                        mea.model_copy(
                            update={"status": MeaStatus.FINALIZING,
                                    "completion_decision": decision}
                        ),
                        rnd.model_copy(update={"phase": RoundPhase.APPLIED}),
                    )
                    return rnd
                if stale:
                    rnd = rnd.append_feedback(await self._stale_feedback(mea, rnd, current))
                return await self._store.save_round(
                    rnd.model_copy(update={"phase": RoundPhase.APPLIED, "stale_requirements": stale})
                )
            if rnd.verdict_patch is None:  # 首次：锁内确定内容，先落盘
                stale = rnd.audit_requirements_revision != current
                done = not stale and _passes(rnd, final=False)
                patch = StepAuditPatch(
                    op=TaskOp(
                        op_id=rnd.op_id("verdict"),
                        precondition=OpPrecondition(
                            step_id=rnd.step_id, step_in=(TaskStepStatus.IN_PROGRESS,)
                        ),
                        step_status=StepStatusChange(
                            step_id=rnd.step_id or "",
                            status=TaskStepStatus.DONE if done else TaskStepStatus.TODO,
                            note=f"{'依据' if done else '见'} {rnd.ref}",
                        ),
                    ),
                    stale=stale,
                )
                if stale:
                    rnd = rnd.append_feedback(await self._stale_feedback(mea, rnd, current))
                rnd = await self._store.save_round(rnd.model_copy(update={"verdict_patch": patch}))
            assert rnd.verdict_patch is not None
            outcome = await self._tasks.apply_op(mea.task_id, rnd.verdict_patch.op)  # 恢复时原样重放
            if outcome.result is OpResult.CONFLICT:
                rnd = rnd.append_feedback(f"步骤状态没有写入: {outcome.reason}")
            return await self._store.save_round(
                rnd.model_copy(
                    update={"phase": RoundPhase.APPLIED,
                            "stale_requirements": rnd.verdict_patch.stale}
                )
            )

    # 函数说明：MeaRunner._after_round
    # 用途：在规划、执行、审计协作中处理 `_after_round`，通过 `self._store.require` 完成
    # 首个内部处理步骤。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `MeaRun`；按分支返回 `mea`；`await self._wait_for_user(mea, f'要求连续
    # 变化，已有 {MAX_ABANDONED_ROUNDS} 轮计划作废。请确认要求后再继续。')`；
    # `await self._wait_for_user(…)`；
    # `await self._set_status(mea, MeaStatus.WAITING_USER)` 等 9 种表达式。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.require` →
    # `self._store.rounds` → `_trailing` → `self._wait_for_user` → `self._set_status` →
    # `self._decide_blocked`；另有 1 个调用点。
    # 分支与异常：
    #   当 `mea.status is not MeaStatus.RUNNING` 时，返回 `mea`。
    #   当 `_trailing(rounds, lambda r: r.phase is RoundPhase.ABANDONED…` 时，返回
    # `await self._wait_for_user(…)`。
    #   当 `_trailing(audited, lambda r: r.audit_output_truncated) >=…` 时，返回
    # `await self._wait_for_user(…)`。
    #   当 `route == Route.ASK.value` 时，返回
    # `await self._set_status(mea, MeaStatus.WAITING_USER)`。
    async def _after_round(self, mea: MeaRun, rnd: MeaRound) -> MeaRun:
        mea = await self._store.require(mea.id)
        if mea.status is not MeaStatus.RUNNING:
            return mea
        all_rounds = await self._store.rounds(mea.id)
        # 各种“连续 / 累计”保护只看你上次继续、回答或补充之后的轮次
        rounds = [r for r in all_rounds if r.index > mea.stall_guard_after]
        if rnd.phase is RoundPhase.ABANDONED:
            if _trailing(rounds, lambda r: r.phase is RoundPhase.ABANDONED) >= MAX_ABANDONED_ROUNDS:
                return await self._wait_for_user(
                    mea, f"要求连续变化，已有 {MAX_ABANDONED_ROUNDS} 轮计划作废。请确认要求后再继续。"
                )
        elif rnd.phase is RoundPhase.APPLIED:
            audited = [r for r in rounds if r.auditor_report is not None]
            if _trailing(audited, lambda r: r.audit_output_truncated) >= MAX_TRUNCATED_AUDITS:
                return await self._wait_for_user(
                    mea, "连续两次审计输出被截断，已停止推进；工具证据已保留。"
                    "请检查输出预算（MEA_ROLE_MAX_OUTPUT_TOKENS）或缩短审计报告后继续。"
                )
            route = rnd.route
            if route == Route.ASK.value:
                return await self._set_status(mea, MeaStatus.WAITING_USER)
            if route == Route.BLOCKED.value:
                return await self._decide_blocked(mea, rnd)
            if route == Route.INVALID.value and (
                _trailing(rounds, lambda r: r.route == Route.INVALID.value) >= MAX_INVALID_ROUTES
            ):
                return await self._wait_for_user(
                    mea, f"任务管理器连续 {MAX_INVALID_ROUTES} 轮没有给出有效计划，需要你介入。"
                )
            if route == Route.INVALID.value and sum(
                r.route == Route.INVALID.value for r in rounds
            ) >= MAX_TOTAL_INVALID_ROUTES:
                return await self._wait_for_user(
                    mea, f"累计 {MAX_TOTAL_INVALID_ROUTES} 轮计划无效，已停止重试。"
                    "请检查模型输出截断和协议格式后再继续。"
                )
            if rnd.kind is RoundKind.RECOVERY_AUDIT and (
                rnd.integrity_status != Integrity.CLEAN.value
                or rnd.audit_status == AuditStatus.BLOCKED.value
            ):
                return await self._wait_for_user(
                    mea,
                    f"{rnd.ref} 中断核查无法确认重做是否安全，请查看审计报告后决定是否继续。",
                )
            stalled = _stall_reason(rounds)
            if stalled is not None:
                return await self._wait_for_user(mea, stalled)
        if mea.pause_requested:
            return await self._set_status(mea, MeaStatus.PAUSED, pause_requested=False)
        return mea

    # ================================================================ 计划校验

    # 函数说明：MeaRunner._decide_plan
    # 用途：作出决策计划，供规划、执行、审计协作使用。
    # 参数：
    #   output：工具、模型或转换步骤的输出，类型 `ManagerOutput`。
    #   task：当前任务记录，类型 `Task`。
    #   requirements：当前生效的任务要求，类型 `Requirements`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `_PlanDecision`；按分支返回 `invalid('没有有效路由')`；`invalid('的 `步
    # 骤更新:
    # ` 含有无法识别的行（' + '；'.join(output.step_updates.invalid_lines) + '）')`；
    # `invalid(f"的 `步骤:` 不是一个未完成的步骤（写的是 {step_id or '空'}）")`；`
    # invalid(text=invalid_final_audit_feedback([(sid, '尚未通过步骤验收') for sid in
    # pending]))` 等 6 种表达式。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`invalid` →
    # `requirements.amendment` → `supersede_rejected_feedback` → `StepSupersede` →
    # `item.basis.upper` → `_next_step_id`；另有 8 个调用点。
    # 分支与异常：
    #   当 `output.route is Route.INVALID` 时，返回 `invalid('没有有效路由')`。
    #   当 `output.step_updates.invalid_lines` 时，返回 `invalid(…)`。
    #   `reason is not None` 分支在完成前置处理后跳过当前循环项。
    #   当 `step_id not in statuses or statuses[step_id] not in…` 时，返回
    # `invalid(f"的 `步骤:` 不是一个未完成的步骤（写的是 {step_id or '空'}）")`。
    def _decide_plan(
        self,
        output: ManagerOutput,
        task: Task,
        requirements: Requirements,
        rnd: MeaRound,
    ) -> _PlanDecision:
        feedback: list[str] = []

        # 函数说明：MeaRunner._decide_plan.invalid
        # 用途：处理规划、执行、审计协作中的 `invalid` 数据；结果及边界条件见下方说明。
        # 参数：
        #   detail：`detail`输入或配置值，类型 `str | None`；默认 `None`。
        #   text：待处理的文本，类型 `str | None`；默认 `None`。
        # 返回：类型 `_PlanDecision`；返回 `_PlanDecision(Route.INVALID, RoundKind.
        # NORMAL, None, None, (*feedback, message))`。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`invalid_plan_feedback` →
        # `_PlanDecision`。
        # 闭包依赖：从外层读取 `feedback`。
        def invalid(detail: str | None = None, *, text: str | None = None) -> _PlanDecision:
            message = text or invalid_plan_feedback(detail or "没有有效路由")
            return _PlanDecision(Route.INVALID, RoundKind.NORMAL, None, None, (*feedback, message))

        if output.route is Route.INVALID:
            return invalid("没有有效路由")
        if output.step_updates.invalid_lines:
            return invalid("的 `步骤更新:` 含有无法识别的行（" + "；".join(output.step_updates.invalid_lines) + "）")

        statuses = {step.id: step.status for step in task.steps}
        steps_by_id = {step.id: step for step in task.steps}
        supersedes: list[StepSupersede] = []
        for item in output.step_updates.superseded:
            step = steps_by_id.get(item.step_id)
            amendment = requirements.amendment(item.basis)
            reason: str | None = None
            if step is None:
                reason = f"步骤 {item.step_id} 不存在"
            elif step.status is TaskStepStatus.SUPERSEDED:
                reason = f"步骤 {item.step_id} 已经被取代"
            elif amendment is None:
                reason = f"权威要求里不存在修订 {item.basis}，依据必须是用户修订，不能是原始请求"
            elif amendment.revision <= step.origin_requirements_revision:
                reason = (
                    f"{item.basis} 生效于 v{amendment.revision}，"
                    f"不晚于 {item.step_id} 的产生版本 v{step.origin_requirements_revision}"
                )
            if reason is not None:
                feedback.append(supersede_rejected_feedback(item.step_id, item.basis, reason))
                continue
            supersedes.append(
                StepSupersede(step_id=item.step_id, amendment_id=item.basis.upper(), reason=item.reason)
            )
            statuses[item.step_id] = TaskStepStatus.SUPERSEDED

        new_steps: list[TaskStep] = []
        for added in output.step_updates.added:
            step_id = _next_step_id(statuses)
            statuses[step_id] = TaskStepStatus.TODO
            new_steps.append(
                TaskStep(
                    id=step_id,
                    title=added.title[:500],
                    acceptance=added.acceptance[:_MAX_STATE_ENTRY_CHARS],
                    origin_requirements_revision=requirements.revision,
                )
            )

        route = output.route
        step_id = output.step_id
        question: str | None = None
        kind = RoundKind.NORMAL
        if route in (Route.EXECUTE, Route.AUDIT_ONLY):
            if step_id not in statuses or statuses[step_id] not in _TODO_OR_RUNNING:
                return invalid(f"的 `步骤:` 不是一个未完成的步骤（写的是 {step_id or '空'}）")
            kind = RoundKind.NORMAL if route is Route.EXECUTE else RoundKind.AUDIT_ONLY
        elif route is Route.FINAL_AUDIT:
            pending = [sid for sid, status in statuses.items() if status not in CLOSED_STEP_STATUSES]
            if pending:
                return invalid(text=invalid_final_audit_feedback([(sid, "尚未通过步骤验收") for sid in pending]))
            kind = RoundKind.FINAL_AUDIT
            step_id = None
        elif route is Route.ASK:
            question = (output.question or "").strip()
            if not question:
                return invalid("了 `下一步: 请示用户`，但没有 `问题:`")
            step_id = None
        else:  # BLOCKED
            step_id = None

        state = _state_entries(output.state)
        contract = output.contract
        if contract is not None and len(contract) > _MAX_CONTRACT_CHARS:
            contract = clip_preserve(contract, _MAX_CONTRACT_CHARS)
            feedback.append(f"任务契约超过 {_MAX_CONTRACT_CHARS} 字符，已保留首尾后写入；请精简契约。")
        proposal: TaskOp | None = None
        if state is not None or contract is not None or new_steps or supersedes:
            proposal = TaskOp(
                op_id=rnd.op_id("plan"),
                precondition=OpPrecondition(revision=task.revision),
                state=state,
                set_contract=contract is not None,
                contract=contract,
                add_steps=tuple(new_steps),
                supersede=tuple(supersedes),
            )
        return _PlanDecision(
            route=route,
            kind=kind,
            step_id=step_id,
            proposal=proposal,
            feedback=tuple(feedback),
            question=question,
            choices=output.choices,
        )

    # ================================================================ 审计

    # 函数说明：MeaRunner._auditor_prompt
    # 用途：在规划、执行、审计协作中处理 `_auditor_prompt`，通过
    # `requirements.amendment` 完成首个内部处理步骤。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    #   task：当前任务记录，类型 `Task`。
    #   requirements：当前生效的任务要求，类型 `Requirements`。
    #   rounds：已完成的模型或工具轮次，类型 `Sequence[MeaRound]`。
    # 返回：类型 `str`；返回 `build_auditor_prompt(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_round_views` → `AuditKind` →
    # `_step_view` → `_step` → `format_related_reports` → `requirements.amendment`；另有
    #  8 个调用点。
    # 分支与异常：
    #   捕获 `Exception` 后，执行异常处理调用 `logger.exception`。
    async def _auditor_prompt(
        self,
        mea: MeaRun,
        rnd: MeaRound,
        task: Task,
        requirements: Requirements,
        rounds: Sequence[MeaRound],
    ) -> str:
        views = _round_views(rounds)
        kind = AuditKind(rnd.kind.value)
        step = _step_view(_step(task, rnd.step_id), rounds) if rnd.step_id else None
        superseded: list[SupersededView] = []
        related = format_related_reports(views, rnd.related_refs)
        record = ""
        executor_run_id = rnd.executor_run_id
        candidates = [
            item for item in rounds
            if item.index <= rnd.index and item.executor_run_id
            and (kind is AuditKind.FINAL_AUDIT or item.step_id == rnd.step_id)
        ]
        if executor_run_id is None and candidates:
            executor_run_id = candidates[-1].executor_run_id
        records: list[str] = []
        if self._executor_evidence is not None:
            # Keep referenced original runs when retries only sought evidence.
            refs = set(rnd.related_refs)
            selected = sorted(
                sorted(candidates, key=lambda item: (
                    item.index == rnd.index, item.ref in refs, item.index,
                ), reverse=True)[:3],
                key=lambda item: item.index,
            )
            if len(candidates) > len(selected):
                records.append(
                    f"另有 {len(candidates) - len(selected)} 个 Executor Run 未附录；"
                    "以下记录不能代表本步骤或整个任务的全部历史。"
                )
            for item in selected:
                try:
                    records.append(await self._executor_evidence(
                        item.executor_run_id, mea.conversation_id, mea.id,
                    ))
                except Exception:
                    logger.exception(
                        "executor evidence unavailable for %s", item.executor_run_id,
                    )
                    records.append(
                        f"run_id={item.executor_run_id}: "
                        "运行证据不可用，不能据此证明调用或回复。"
                    )
        if kind is AuditKind.FINAL_AUDIT:
            for item in task.steps:
                if item.status is TaskStepStatus.SUPERSEDED and item.superseded_by:
                    amendment = requirements.amendment(item.superseded_by)
                    superseded.append(
                        SupersededView(
                            step_id=item.id,
                            acceptance=item.acceptance,
                            amendment_id=item.superseded_by,
                            amendment_text=amendment.text if amendment else "(未找到该修订)",
                            reason=item.superseded_reason,
                        )
                    )
            related = format_related_reports(views, _last_satisfied_refs(task, rounds))
        if kind is AuditKind.RECOVERY_AUDIT:
            interrupted = next(
                (r for r in reversed(rounds) if r.index < rnd.index and r.interrupt_reason), None
            )
            completed: list[str] = []
            pending: list[str] = []
            executor_run_id = interrupted.executor_run_id if interrupted else None
            if interrupted is not None and interrupted.executor_run_id and self._recovery_info:
                try:
                    completed, pending = await self._recovery_info(interrupted.executor_run_id)
                except Exception:
                    logger.exception("recovery info failed for %s", interrupted.executor_run_id)
            record = recovery_record(
                round_index=interrupted.index if interrupted else rnd.index,
                reason=(interrupted.interrupt_reason if interrupted else None) or "未知",
                subtask=rnd.subtask or "",
                completed_calls=completed,
                pending_calls=pending,
            )
        return build_auditor_prompt(
            kind=kind,
            requirements_text=render_requirements(requirements),
            contract=task.contract,
            state="\n".join(task.state),
            task_id=task.id,
            workspace_root=str(self._workspace),
            audit_revision=requirements.revision,
            step=step,
            other_steps=[_step_view(s, rounds) for s in task.steps],
            superseded=superseded,
            subtask=rnd.subtask or "",
            executor_output=rnd.executor_output or "",
            focus=rnd.focus,
            recovery_record=record,
            related_reports=related,
            executor_run_id=executor_run_id,
            executor_records="\n".join(records),
        )

    # 函数说明：MeaRunner._finalize_audit
    # 用途：结束`audit`，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    #   result：上一步计算或执行得到的结果，类型 `SubRunResult`。
    #   before：传给 `snapshot_diff` 的输入，类型 `Any`。
    # 返回：类型 `tuple[ControlHeader, str]`；返回 `(header, report)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`no_report_placeholder` →
    # `parse_control_header` → `snapshot_workspace` → `self._run_role` → `uuid4` →
    # `build_format_repair_prompt`；另有 9 个调用点。
    async def _finalize_audit(
        self,
        mea: MeaRun,
        rnd: MeaRound,
        result: SubRunResult,
        before: Any,
    ) -> tuple[ControlHeader, str]:
        report = result.text.strip()
        if not report:
            report = no_report_placeholder()
        if result.truncated or not result.ok:
            # Partial evidence must never become a passing verdict via header repair.
            report = (
                "状态: blocked\n完整性: suspect\n"
                "契约审计: unknown\n步骤验收: not_satisfied\n"
                "审计事实: 审计输出被截断或未正常结束，本轮不能形成通过裁决。\n"
                "缺口: 需要完整报告；已有工具证据保留，可用仅审计核实。\n"
                "下一步: 缩短报告，只列本步骤证据与缺口，再进行完整审计。\n\n"
                + "原始输出（不作为通过依据）:\n" + report[:1800]
            )
        header = parse_control_header(report)
        kind = AuditKind(rnd.kind.value)
        scope_valid = _header_matches_scope(header, kind)
        if (not scope_valid and not result.timed_out
                and result.ok and not result.truncated):
            repair_before = snapshot_workspace(self._workspace, exclude=self._exclude)
            repaired = await self._run_role(
                mea, uuid4().hex, AgentMode.MANAGE,
                build_format_repair_prompt(report, kind=kind),
            )
            repair_diff = snapshot_diff(
                repair_before, snapshot_workspace(self._workspace, exclude=self._exclude)
            )
            candidate = repaired.text.strip()
            candidate_header = parse_control_header(candidate)
            if (repaired.ok and not repaired.truncated and not repair_diff.mutated
                    and _header_matches_scope(candidate_header, kind)):
                # A scope correction may change the header, never the recorded findings.
                report = (
                    candidate_header.render() + "\n" + strip_control_header(report)
                    if header is not None else candidate
                )
                header = candidate_header
        if not _header_matches_scope(header, kind):
            if header is not None:
                report += (
                    f"\n\nharness 审计类型不匹配: {kind.value} "
                    f"不能使用步骤验收 {header.step_acceptance.value}；本轮不通过。"
                )
            report = invalid_header_report(report)
            header = parse_control_header(report)
        assert header is not None
        diff = snapshot_diff(before, snapshot_workspace(self._workspace, exclude=self._exclude))
        if diff.mutated:
            report = readonly_violation_report(diff.paths(), report)
            header = parse_control_header(report)
            assert header is not None
        guarded = apply_blocking_guard(header, report)
        header, report = guarded.header, guarded.report
        rejections = Counter(rnd.executor_rejections) + result.rejections
        if rejections:
            report = f"{report.rstrip()}\n\n" + "\n".join(
                boundary_rejection_note(rnd.index, role, counts)
                for role, counts in (("executor", Counter(rnd.executor_rejections)),
                                     ("auditor", result.rejections))
                if counts
            )
            if sum(rejections.values()) >= REJECTIONS_CAP_INTEGRITY and header.integrity is Integrity.CLEAN:
                header = ControlHeader(
                    header.status, Integrity.SUSPECT, header.contract_audit, header.step_acceptance
                )
        return header, report

    # 函数说明：MeaRunner._stale_feedback
    # 用途：在规划、执行、审计协作中处理 `_stale_feedback`，通过
    # `self._store.requirements` 完成首个内部处理步骤。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    #   current：当前值或状态，类型 `int`。
    # 返回：类型 `str`；返回 `requirements_updated_feedback(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.requirements` →
    # `requirements_updated_feedback`。
    async def _stale_feedback(self, mea: MeaRun, rnd: MeaRound, current: int) -> str:
        requirements = await self._store.requirements(mea.id, current)
        old = rnd.audit_requirements_revision or current
        return requirements_updated_feedback(
            round_index=rnd.index,
            old_revision=old,
            new_revision=current,
            new_amendments=[(a.id, a.text) for a in requirements.amendments if a.revision > old],
            step_ids=[rnd.step_id] if rnd.step_id else [],
        )

    # ================================================================ 状态写入辅助

    # 函数说明：MeaRunner._begin_step
    # 用途：开始步骤，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`RoundConflict` →
    # `self._tasks.apply_op` → `TaskOp` → `rnd.op_id` → `OpPrecondition` →
    # `StepStatusChange`。
    # 分支与异常：
    #   当 `not rnd.step_id` 时，抛出 `RoundConflict(f'{rnd.ref} 没有关联步骤')`。
    #   当 `outcome.result is OpResult.CONFLICT` 时，抛出
    # `RoundConflict(outcome.reason or '步骤已不是未完成状态')`。
    async def _begin_step(self, mea: MeaRun, rnd: MeaRound) -> None:
        if not rnd.step_id:
            raise RoundConflict(f"{rnd.ref} 没有关联步骤")
        outcome = await self._tasks.apply_op(
            mea.task_id,
            TaskOp(
                op_id=rnd.op_id("begin"),
                precondition=OpPrecondition(step_id=rnd.step_id, step_in=_TODO_OR_RUNNING),
                step_status=StepStatusChange(
                    step_id=rnd.step_id, status=TaskStepStatus.IN_PROGRESS, note=f"{rnd.ref} 处理中"
                ),
            ),
        )
        if outcome.result is OpResult.CONFLICT:
            raise RoundConflict(outcome.reason or "步骤已不是未完成状态")

    # 函数说明：MeaRunner._abandon
    # 用途：放弃MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    #   plan_applied：计划输入或配置值，类型 `bool`。
    #   stale_revision：传给 `self._store.requirements` 的输入，类型 `int | None`；默认
    # `None`。
    #   reason：状态变化、拒绝或降级原因，类型 `str | None`；默认 `None`。
    #   release：`release`输入或配置值，类型 `bool`；默认 `False`。
    # 返回：类型 `MeaRound`；返回 `await self._store.save_round(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._tasks.op_applied` →
    # `rnd.op_id` → `self._tasks.apply_op` → `TaskOp` → `OpPrecondition` →
    # `StepStatusChange`；另有 5 个调用点。
    async def _abandon(
        self,
        mea: MeaRun,
        rnd: MeaRound,
        *,
        plan_applied: bool,
        stale_revision: int | None = None,
        reason: str | None = None,
        release: bool = False,
    ) -> MeaRound:
        if release and rnd.step_id and await self._tasks.op_applied(mea.task_id, rnd.op_id("begin")):
            outcome = await self._tasks.apply_op(
                mea.task_id,
                TaskOp(
                    op_id=rnd.op_id("release"),
                    precondition=OpPrecondition(
                        step_id=rnd.step_id, step_in=(TaskStepStatus.IN_PROGRESS,)
                    ),
                    step_status=StepStatusChange(
                        step_id=rnd.step_id, status=TaskStepStatus.TODO, note=f"{rnd.ref} 作废"
                    ),
                ),
            )
            if outcome.result is OpResult.CONFLICT:
                logger.warning("release conflict for %s: %s", rnd.ref, outcome.reason)
        amendment_ids: list[str] = []
        if stale_revision is not None:
            requirements = await self._store.requirements(mea.id, stale_revision)
            base = rnd.plan_requirements_revision or 0
            new = [a for a in requirements.amendments if a.revision > base]
            amendment_ids = [a.id for a in new]
            added = "；".join(f"{a.id}: {a.text}" for a in new) or "(无)"
            reason = f"权威要求已更新到 v{stale_revision}（新增 {added}）"
        added_steps = (
            [s.id for s in rnd.plan_proposal.add_steps] if plan_applied and rnd.plan_proposal else []
        )
        feedback = round_abandoned_feedback(
            round_index=rnd.index,
            plan_applied=plan_applied,
            reason=reason or "未知原因",
            added_step_ids=added_steps,
            amendment_ids=amendment_ids,
        )
        return await self._store.save_round(
            rnd.append_feedback(feedback).model_copy(
                update={"phase": RoundPhase.ABANDONED, "abandon_reason": reason}
            )
        )

    # 函数说明：MeaRunner._interrupt
    # 用途：中断MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    # 返回：类型 `MeaRound`；返回 `await self._store.save_round(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.save_round`。
    async def _interrupt(self, rnd: MeaRound, reason: str) -> MeaRound:
        return await self._store.save_round(
            rnd.model_copy(update={"phase": RoundPhase.INTERRUPTED, "interrupt_reason": reason})
        )

    # 函数说明：MeaRunner._decide_blocked
    # 用途：作出决策`blocked`，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `MeaRun`；返回 `mea`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock` →
    # `self._store.require` → `CompletionDecision` → `clip_preserve` → `now_utc` →
    # `self._store.save_run_and_round`。
    # 资源/并发边界：`self._lock(self._amend_locks, mea.id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `mea.status is not MeaStatus.RUNNING` 时，返回 `mea`。
    async def _decide_blocked(self, mea: MeaRun, rnd: MeaRound) -> MeaRun:
        async with self._lock(self._amend_locks, mea.id):
            mea = await self._store.require(mea.id)
            if mea.status is not MeaStatus.RUNNING:
                return mea
            decision = CompletionDecision(
                outcome="blocked",
                round_index=rnd.index,
                requirements_revision=mea.requirements_revision,
                reason=clip_preserve(rnd.plan_text or "", 2_000) or None,
                decided_at=now_utc(),
            )
            mea, _ = await self._store.save_run_and_round(
                mea.model_copy(update={"status": MeaStatus.FINALIZING, "completion_decision": decision}),
                rnd,
            )
            return mea

    # 函数说明：MeaRunner._finalize
    # 用途：只执行已落盘的终结决定，每一步都可重放。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    # 返回：类型 `MeaRun`；按分支返回
    # `await self._fail(mea, 'finalizing without a completion decision')`；
    # `await self._fail(mea, f'task_changed_after_decision: {outcome.reason}')`；
    # `await self._set_status(mea, status)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._fail` →
    # `self._tasks.apply_op` → `TaskOp` → `OpPrecondition` → `self._final_response` →
    # `self._store.save_run`；另有 2 个调用点。
    # 分支与异常：
    #   当 `decision is None` 时，返回 `await self._fail(…)`。
    #   当 `outcome.result is OpResult.CONFLICT` 时，返回 `await self._fail(…)`。
    async def _finalize(self, mea: MeaRun) -> MeaRun:
        """只执行已落盘的终结决定，每一步都可重放。"""

        decision = mea.completion_decision
        if decision is None:
            return await self._fail(mea, "finalizing without a completion decision")
        if decision.outcome == "completed":
            outcome = await self._tasks.apply_op(
                mea.task_id,
                TaskOp(
                    op_id=f"{mea.id}/complete",
                    precondition=OpPrecondition(all_steps_closed=True),
                    complete_task=True,
                ),
            )
            if outcome.result is OpResult.CONFLICT:
                return await self._fail(mea, f"task_changed_after_decision: {outcome.reason}")
        if mea.final_response is None:
            text = await self._final_response(mea, decision)
            mea = await self._store.save_run(mea.model_copy(update={"final_response": text}))
        if self._final_sink is not None:
            await self._final_sink(mea.conversation_id, mea.final_response or "", f"{mea.id}/final")
        status = MeaStatus.COMPLETED if decision.outcome == "completed" else MeaStatus.BLOCKED
        return await self._set_status(mea, status)

    # 函数说明：MeaRunner._final_response
    # 用途：在规划、执行、审计协作中处理 `_final_response`，通过
    # `self._store.requirements` 完成首个内部处理步骤。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   decision：权限、上下文或审计决策，类型 `CompletionDecision`。
    # 返回：类型 `str`；按分支返回 `result.text.strip()`；`'\n'.join(lines)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.requirements` →
    # `self._task` → `self._store.rounds` → `build_final_response_prompt` →
    # `render_requirements` → `format_audit_findings`；另有 4 个调用点。
    # 分支与异常：
    #   当 `result.ok and (not result.truncated) and result.text.strip()` 时，返回
    # `result.text.strip()`。
    async def _final_response(self, mea: MeaRun, decision: CompletionDecision) -> str:
        requirements = await self._store.requirements(mea.id, decision.requirements_revision)
        task = await self._task(mea)
        rounds = await self._store.rounds(mea.id)
        outcome = "complete" if decision.outcome == "completed" else "blocked"
        deliverables = [
            r.executor_output or ""
            for r in rounds
            if r.kind is RoundKind.NORMAL
            and r.verdict_patch is not None
            and r.verdict_patch.op.step_status is not None
            and r.verdict_patch.op.step_status.status is TaskStepStatus.DONE
        ]
        prompt = build_final_response_prompt(
            requirements_text=render_requirements(requirements),
            outcome=outcome,
            abort_reason=decision.reason,
            state="\n".join(task.state),
            findings=format_audit_findings(
                _round_views(rounds),
                final_round_index=decision.round_index if outcome == "complete" else None,
            ),
            deliverables=deliverables,
        )
        for _ in range(FINAL_RESPONSE_ATTEMPTS):
            result = await self._run_role(mea, uuid4().hex, AgentMode.MANAGE, prompt, count_failure=False)
            if result.ok and not result.truncated and result.text.strip():
                return result.text.strip()
        done = [s for s in task.steps if s.status is TaskStepStatus.DONE]
        lines = [
            "长任务已完成并通过最终验收。" if outcome == "complete" else "长任务无法继续推进，已停止。",
            *(f"- {s.title}：已完成" for s in done),
        ]
        if decision.reason:
            lines.append(f"原因：{clip_preserve(decision.reason, 1_000)}")
        return "\n".join(lines)

    # 函数说明：MeaRunner._pause_out_of_rounds
    # 用途：暂停`out_of_rounds`，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    # 返回：类型 `MeaRun`；返回 `mea`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._set_status` →
    # `self._counted_rounds` → `self._final_sink`。
    async def _pause_out_of_rounds(self, mea: MeaRun) -> MeaRun:
        text = (
            f"长任务已用完 {mea.round_budget} 轮预算，暂停在当前进度。"
            "可以追加轮次后继续，或查看每轮的审计报告。"
        )
        mea = await self._set_status(mea, MeaStatus.PAUSED, abort_reason="max_rounds")
        if self._final_sink is not None:
            key = f"{mea.id}/paused/{await self._counted_rounds(mea.id)}"
            await self._final_sink(mea.conversation_id, text, key)
        return mea

    # 函数说明：MeaRunner._wait_for_user
    # 用途：等待`for_user`，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   question：`question`输入或配置值，类型 `str`。
    # 返回：类型 `MeaRun`；返回 `await self._set_status(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._set_status`。
    async def _wait_for_user(self, mea: MeaRun, question: str) -> MeaRun:
        return await self._set_status(
            mea, MeaStatus.WAITING_USER, pending_question=question, pending_choices=()
        )

    # 函数说明：MeaRunner._fail
    # 用途：记录失败MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    # 返回：类型 `MeaRun`；返回
    # `await self._set_status(mea, MeaStatus.FAILED, abort_reason=reason)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`logger.error` →
    # `self._store.require` → `self._set_status`。
    async def _fail(self, mea: MeaRun, reason: str) -> MeaRun:
        logger.error("mea %s failed: %s", mea.id, reason)
        mea = await self._store.require(mea.id)
        return await self._set_status(mea, MeaStatus.FAILED, abort_reason=reason)

    # 函数说明：MeaRunner._set_status
    # 用途：设置状态，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   status：目标状态，类型 `MeaStatus`。
    #   **update：额外关键字参数，按实现处理或转交。
    # 返回：类型 `MeaRun`；返回
    # `await self._store.save_run(mea.model_copy(update={'status': status, **update}))`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.save_run`。
    async def _set_status(self, mea: MeaRun, status: MeaStatus, **update: Any) -> MeaRun:
        return await self._store.save_run(mea.model_copy(update={"status": status, **update}))

    # ================================================================ 子 Run

    # 函数说明：MeaRunner._run_role
    # 用途：运行角色，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   run_id：目标运行标识，类型 `str`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   prompt：本次调用使用的提示文本，类型 `str`。
    #   count_failure：`count_failure`输入或配置值，类型 `bool`；默认 `True`。
    #   task_step_id：任务步骤标识，类型 `str | None`；默认 `None`。
    # 返回：类型 `SubRunResult`；按分支返回
    # `SubRunResult(run_id, '', ok=False, cancelled=True, stop_reason='stopped')`；
    # `SubRunResult(run_id, '', ok=False, cancelled=True, stop_reason='duplicate')`；
    # `result`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.require` →
    # `SubRunResult` → `self._start_role` → `self._wait_role` → `self._note_role_result`
    # 。
    # 分支与异常：
    #   当 `(await self._store.require(mea.id)).status not in…` 时，返回
    # `SubRunResult(…)`。
    #   捕获 `RunAlreadyExists` 后，返回
    # `SubRunResult(run_id, '', ok=False, cancelled=True, stop_reason='duplicate')`。
    #   当 `run_id not in self._runs.active_run_ids` 时，返回 `SubRunResult(…)`。
    async def _run_role(
        self,
        mea: MeaRun,
        run_id: str,
        mode: AgentMode,
        prompt: str,
        *,
        count_failure: bool = True,
        task_step_id: str | None = None,
    ) -> SubRunResult:
        if (await self._store.require(mea.id)).status not in _CHILD_START_STATUSES:
            return SubRunResult(run_id, "", ok=False, cancelled=True, stop_reason="stopped")
        try:
            await self._start_role(
                mea,
                run_id,
                mode,
                prompt,
                task_step_id=task_step_id,
            )
        except RunAlreadyExists:
            if run_id not in self._runs.active_run_ids:
                return SubRunResult(run_id, "", ok=False, cancelled=True, stop_reason="duplicate")
        result = await self._wait_role(mea, run_id, mode)
        if count_failure:
            self._note_role_result(mea, result)
        return result

    # 函数说明：MeaRunner._start_role
    # 用途：启动角色，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   run_id：目标运行标识，类型 `str`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   prompt：本次调用使用的提示文本，类型 `str`。
    #   task_step_id：任务步骤标识，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._runs.start` →
    # `self._runtime_for`。
    async def _start_role(
        self,
        mea: MeaRun,
        run_id: str,
        mode: AgentMode,
        prompt: str,
        *,
        task_step_id: str | None = None,
    ) -> None:
        self._active_child[mea.id] = run_id
        tool_context_metadata = {"task_id": mea.task_id, "mea_id": mea.id}
        if task_step_id is not None:
            tool_context_metadata["task_step_id"] = task_step_id
        await self._runs.start(
            prompt,
            conversation_id=mea.conversation_id,
            mode=mode,
            runtime=self._runtime_for(mea, mode),
            run_id=run_id,
            source=f"mea:{ROLE_NAMES[mode]}",
            source_id=mea.id,
            tool_context_metadata=tool_context_metadata,
        )

    # 函数说明：MeaRunner._runtime_for
    # 用途：处理规划、执行、审计协作中的 `_runtime_for` 数据；结果及边界条件见下方说明。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    # 返回：类型 `Any`；按分支返回
    # `executor_for(mea.extra_tools, mea.auto_approve_sandbox)`；`self._runtimes[mode]`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`executor_for`。
    # 分支与异常：
    #   当 `mode is AgentMode.EXECUTE and executor_for is not None` 时，返回
    # `executor_for(mea.extra_tools, mea.auto_approve_sandbox)`。
    def _runtime_for(self, mea: MeaRun, mode: AgentMode) -> Any:
        executor_for = getattr(self._runtimes_source, "executor_for", None)
        if mode is AgentMode.EXECUTE and executor_for is not None:
            return executor_for(mea.extra_tools, mea.auto_approve_sandbox)
        return self._runtimes[mode]

    # 函数说明：MeaRunner._wait_role
    # 用途：等待角色，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   run_id：目标运行标识，类型 `str`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode | None`；默认 `None`。
    # 返回：类型 `SubRunResult`；返回 `SubRunResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.ensure_future` →
    # `self._runs.wait` → `self._wait_with_budget` → `self._cancel_child` →
    # `self._forget` → `SubRunResult`；另有 4 个调用点。
    # 分支与异常：
    #   `run is _TIMED_OUT` 分支在完成前置处理后返回 `SubRunResult(…)`。
    #   当 `result is None` 时，返回 `SubRunResult(…)`。
    # 副作用与资源：
    #   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
    async def _wait_role(
        self, mea: MeaRun, run_id: str, mode: AgentMode | None = None
    ) -> SubRunResult:
        self._active_child[mea.id] = run_id
        timeout = self._timeouts.get(mode) if mode is not None else None
        # RunManager.wait 会吞掉等待期间收到的 CancelledError。把它放进单独的任务再等：循环本身
        # 被取消时（Host 关闭、删除会话）取消异常照常抛出，不会把仍在运行的子 Run 当成已结束继续推进。
        waiter = asyncio.ensure_future(self._runs.wait(run_id))
        try:
            if timeout:
                run = await self._wait_with_budget(run_id, waiter, timeout)
                if run is _TIMED_OUT:
                    await self._cancel_child(run_id)
                    await waiter
                    self._forget(run_id)
                    role = ROLE_NAMES.get(mode, mode.value) if mode is not None else "子 Run"
                    minutes = f"{timeout / 60:g}"
                    return SubRunResult(
                        run_id,
                        f"{role} 运行超过 {minutes} 分钟没有结束，已被强制停止，没有最终回复。",
                        ok=False,
                        cancelled=False,
                        timed_out=True,
                    )
            else:
                run = await asyncio.shield(waiter)
        finally:
            if self._active_child.get(mea.id) == run_id:
                self._active_child.pop(mea.id, None)
        result = self._runs.result(run_id)
        self._forget(run_id)
        cancelled = getattr(run, "status", None) is RunStatus.CANCELLED
        if result is None:
            return SubRunResult(
                run_id,
                str(getattr(run, "error", None) or ""),
                ok=False,
                cancelled=cancelled,
                stop_reason=None,
            )
        return SubRunResult(
            run_id,
            result.final_message.content or "",
            ok=result.ok,
            cancelled=cancelled or result.stop_reason is AgentStopReason.CANCELLED,
            stop_reason=result.stop_reason.value,
            model_finish_reason=result.model_finish_reason,
            rejections=count_role_rejections(record.result for record in result.tool_calls),
        )

    # 函数说明：MeaRunner._wait_with_budget
    # 用途：等子 Run 结束；超过 ``timeout`` 秒返回 ``_TIMED_OUT``。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   waiter：`waiter`输入或配置值，类型 `asyncio.Future[Any]`。
    #   timeout：等待超时配置，类型 `float`。
    # 返回：类型 `Any`；按分支返回 `waiter.result()`；`_TIMED_OUT`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.wait` → `waiter.result` →
    #  `awaiting`。
    # 分支与异常：
    #   当 `done` 时，返回 `waiter.result()`。
    #   当 `elapsed >= timeout` 时，返回 `_TIMED_OUT`。
    async def _wait_with_budget(
        self, run_id: str, waiter: asyncio.Future[Any], timeout: float
    ) -> Any:
        """等子 Run 结束；超过 ``timeout`` 秒返回 ``_TIMED_OUT``。等待人工审批的时间不计时。

        ``asyncio.wait`` 超时或本任务被取消时都不会取消 ``waiter``。
        """

        awaiting = getattr(self._runs, "awaiting_approval", None)
        tick = min(timeout, 1.0)
        elapsed = 0.0
        while True:
            done, _ = await asyncio.wait({waiter}, timeout=tick)
            if done:
                return waiter.result()
            if awaiting is None or not awaiting(run_id):
                elapsed += tick
            if elapsed >= timeout:
                return _TIMED_OUT

    # 函数说明：MeaRunner._forget
    # 用途：清除MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._runs.forget_results`。
    # 分支与异常：
    #   捕获 `RuntimeError` 后，忽略该异常并继续当前流程。
    def _forget(self, run_id: str) -> None:
        try:
            self._runs.forget_results((run_id,))
        except RuntimeError:
            pass

    # 函数说明：MeaRunner._note_role_result
    # 用途：在规划、执行、审计协作中处理 `_note_role_result`，通过
    # `self._role_failures.pop` 完成首个内部处理步骤。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    #   result：上一步计算或执行得到的结果，类型 `SubRunResult`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentStopReason` →
    # `self._role_failures.pop` → `MeaFailure`。
    # 分支与异常：
    #   当 `result.cancelled or result.timed_out` 时，返回 `None`。
    #   `self._role_failures[mea.id] >= MAX_ROLE_FAILURES` 分支在完成前置处理后抛出
    # `MeaFailure(…)`。
    def _note_role_result(self, mea: MeaRun, result: SubRunResult) -> None:
        if result.cancelled or result.timed_out:  # 超时按预算类停止处理，不算模型错误
            return
        fatal = result.stop_reason is None or AgentStopReason(result.stop_reason) in _FATAL_STOP_REASONS
        if fatal:
            self._role_failures[mea.id] += 1
            if self._role_failures[mea.id] >= MAX_ROLE_FAILURES:
                self._role_failures.pop(mea.id, None)
                raise MeaFailure(f"子 Run 连续 {MAX_ROLE_FAILURES} 次模型或上下文错误：{result.text[:300]}")
        else:
            self._role_failures.pop(mea.id, None)

    # 函数说明：MeaRunner._cancel_child
    # 用途：取消`child`，供规划、执行、审计协作使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._runs.cancel`。
    # 分支与异常：
    #   捕获 `(KeyError, ValueError)` 后，忽略该异常并继续当前流程。
    async def _cancel_child(self, run_id: str) -> None:
        try:
            await self._runs.cancel(run_id)
        except (KeyError, ValueError):
            pass

    # ================================================================ 杂项

    async def _run_logged(self, mea_id: str) -> MeaRun:
        """Persist the exception reason; keep the full traceback in server logs."""
        try:
            return await self.run(mea_id)
        except Exception as exc:
            logger.exception("mea loop crashed: %s", mea_id)
            mea = await self._store.require(mea_id)
            reason = f"runner crashed: {type(exc).__name__}: {exc}"
            return await self._fail(mea, reason)

    # 函数说明：MeaRunner._task
    # 用途：在规划、执行、审计协作中处理 `_task`，通过 `self._tasks.get` 完成首个内部处
    # 理步骤。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    # 返回：类型 `Task`；返回 `task`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MeaFailure`。
    # 分支与异常：
    #   当 `task is None` 时，抛出 `MeaFailure(f'任务不存在：{mea.task_id}')`。
    async def _task(self, mea: MeaRun) -> Task:
        task = await self._tasks.get(mea.task_id)
        if task is None:
            raise MeaFailure(f"任务不存在：{mea.task_id}")
        return task

    # 函数说明：MeaRunner._revision
    # 用途：返回 `(await self._store.require(mea_id)).requirements_revision`，提供
    # MeaRunner 的派生值。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `int`；返回 `(await self._store.require(mea_id)).requirements_revision`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.require`。
    async def _revision(self, mea_id: str) -> int:
        return (await self._store.require(mea_id)).requirements_revision

    # 函数说明：MeaRunner._current_round
    # 用途：在规划、执行、审计协作中处理 `_current_round`，通过 `self._store.last_round`
    #  完成首个内部处理步骤。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `MeaRound | None`；按分支返回 `None`；`last`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.last_round`。
    # 分支与异常：
    #   当 `last is None or last.phase in CLOSED_ROUND_PHASES` 时，返回 `None`。
    async def _current_round(self, mea_id: str) -> MeaRound | None:
        last = await self._store.last_round(mea_id)
        if last is None or last.phase in CLOSED_ROUND_PHASES:
            return None
        return last

    # 函数说明：MeaRunner._last_round_index
    # 用途：在规划、执行、审计协作中处理 `_last_round_index`，通过
    # `self._store.last_round` 完成首个内部处理步骤。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `int`；返回 `last.index if last is not None else 0`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.last_round`。
    async def _last_round_index(self, mea_id: str) -> int:
        last = await self._store.last_round(mea_id)
        return last.index if last is not None else 0

    # 函数说明：MeaRunner._counted_rounds
    # 用途：返回 `sum((1 for r in await self._store.rounds(mea_id) if r.phase is not
    # RoundPhase.ABANDONED))`，提供 MeaRunner 的派生值。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `int`；返回 `sum((1 for r in await self._store.rounds(mea_id) if r.
    # phase is not RoundPhase.ABANDONED))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.rounds`。
    async def _counted_rounds(self, mea_id: str) -> int:
        return sum(1 for r in await self._store.rounds(mea_id) if r.phase is not RoundPhase.ABANDONED)

    # 函数说明：MeaRunner._open_next_round
    # 用途：打开`next_round`，供规划、执行、审计协作使用。
    # 参数：
    #   mea：当前长任务协作记录，类型 `MeaRun`。
    # 返回：类型 `MeaRound`；返回 `await self._store.save_round(rnd)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.last_round` →
    # `now_utc` → `self._runs.get_run` → `MeaRound` → `self._store.save_round`。
    async def _open_next_round(self, mea: MeaRun) -> MeaRound:
        last = await self._store.last_round(mea.id)
        index = (last.index + 1) if last else 1
        now = now_utc()
        if (
            last is not None
            and last.interrupt_reason
            and last.executor_run_id
            and await self._runs.get_run(last.executor_run_id) is not None
        ):
            rnd = MeaRound(
                mea_run_id=mea.id,
                index=index,
                kind=RoundKind.RECOVERY_AUDIT,
                phase=RoundPhase.EXECUTED,
                route="recovery_audit",
                step_id=last.step_id,
                subtask=last.subtask,
                executor_output=last.executor_output,
                plan_requirements_revision=mea.requirements_revision,
                created_at=now,
                updated_at=now,
            )
        else:
            rnd = MeaRound(mea_run_id=mea.id, index=index, created_at=now, updated_at=now)
        return await self._store.save_round(rnd)

    # 函数说明：MeaRunner._lock
    # 用途：获取锁MeaRunner，供规划、执行、审计协作使用。
    # 参数：
    #   locks：`locks`输入或配置值，类型 `dict[str, asyncio.Lock]`。
    #   key：字段名或查询键，类型 `str`。
    # 返回：类型 `asyncio.Lock`；返回 `lock`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Lock`。
    @staticmethod
    def _lock(locks: dict[str, asyncio.Lock], key: str) -> asyncio.Lock:
        lock = locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            locks[key] = lock
        return lock


# ==================================================================== 纯函数


# 函数说明：_passes
# 用途：处理规划、执行、审计协作中的 `_passes` 数据；结果及边界条件见下方说明。
# 参数：
#   rnd：当前测试模型轮次，类型 `MeaRound`。
#   final：`final`输入或配置值，类型 `bool`。
# 返回：类型 `bool`；按分支返回 `False`；
# `rnd.audit_status == AuditStatus.COMPLETE.value`；
# `rnd.step_acceptance == StepAcceptance.SATISFIED.value`。
# 分支与异常：
#   当 `rnd.integrity_status != Integrity.CLEAN.value` 时，返回 `False`。
#   当 `rnd.contract_audit_status != ContractAudit.ALIGNED.value` 时，返回 `False`。
#   当 `final` 时，返回 `rnd.audit_status == AuditStatus.COMPLETE.value`。
def _header_matches_scope(header: ControlHeader | None, kind: AuditKind) -> bool:
    if header is None:
        return False
    return ((header.step_acceptance is StepAcceptance.NOT_APPLICABLE)
            == (kind is AuditKind.FINAL_AUDIT))


def _passes(rnd: MeaRound, *, final: bool) -> bool:
    if rnd.integrity_status != Integrity.CLEAN.value:
        return False
    if rnd.contract_audit_status != ContractAudit.ALIGNED.value:
        return False
    if final:
        return (rnd.audit_status == AuditStatus.COMPLETE.value
                and rnd.step_acceptance == StepAcceptance.NOT_APPLICABLE.value)
    return rnd.step_acceptance == StepAcceptance.SATISFIED.value


# 函数说明：_round_passed
# 用途：处理规划、执行、审计协作中的 `_round_passed` 数据；结果及边界条件见下方说明。
# 参数：
#   rnd：当前测试模型轮次，类型 `MeaRound`。
# 返回：类型 `bool`；返回 `patch is not None and (not patch.stale) and (patch.op.
# step_status is not None) and (…`。
def _round_passed(rnd: MeaRound) -> bool:
    patch = rnd.verdict_patch
    return (
        patch is not None
        and not patch.stale
        and patch.op.step_status is not None
        and patch.op.step_status.status is TaskStepStatus.DONE
    )


# 函数说明：_stall_reason
# 用途：同一个目标（步骤或最终验收）连续没通过时返回说明，否则 None。
# 参数：
#   rounds：已完成的模型或工具轮次，类型 `Sequence[MeaRound]`。
# 返回：类型 `str | None`；按分支返回 `None`；`f'{name} 连续 {
# STALL_SAME_WORKSPACE_ROUNDS} 轮没有通过审计，而且这几轮工作区完全没有变化（{refs}），
# 继续重试大概率是同样的结果。请查…`；`f'{name} 已连续 {len(streak)} 轮没有通过审计。请
# 查看最近的审计报告，确认验收标准是否合理、是否缺少条件，补充说明后再继续。'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_round_passed`。
# 分支与异常：
#   当 `not attempts` 时，返回 `None`。
#   当 `_round_passed(last)` 时，返回 `None`。
#   当 `rnd.step_id != target or _round_passed(rnd)` 时，结束当前循环。
#   当 `len(same_workspace) >= STALL_SAME_WORKSPACE_ROUNDS and len(…` 时，返回
# `f'{name} 连续 {STALL_SAME_WORKSPACE_ROUNDS} 轮没有通过审计，而且这几轮工作区完…`。
def _stall_reason(rounds: Sequence[MeaRound]) -> str | None:
    """同一个目标（步骤或最终验收）连续没通过时返回说明，否则 None。

    只看做过审计的已完成轮次；作废、请示、阻塞、计划无效的轮次跳过，它们各有自己的保护。
    工作区快照连续不变说明这几轮什么都没改变（例如审计反复被截断、执行者原地打转），
    快照在变但一直通不过则多给几轮再停。
    """

    attempts = [
        r
        for r in rounds
        if r.phase is RoundPhase.APPLIED and r.auditor_run_id and r.snapshot_before is not None
    ]
    if not attempts:
        return None
    last = attempts[-1]
    if _round_passed(last):
        return None
    target = last.step_id
    streak: list[MeaRound] = []
    for rnd in reversed(attempts):
        if rnd.step_id != target or _round_passed(rnd):
            break
        streak.append(rnd)
    name = f"步骤 {target}" if target else "最终验收"
    refs = "、".join(r.ref for r in reversed(streak[:STALL_SAME_WORKSPACE_ROUNDS]))
    same_workspace = streak[:STALL_SAME_WORKSPACE_ROUNDS]
    if (
        len(same_workspace) >= STALL_SAME_WORKSPACE_ROUNDS
        and len({r.snapshot_before for r in same_workspace}) == 1
    ):
        return (
            f"{name} 连续 {STALL_SAME_WORKSPACE_ROUNDS} 轮没有通过审计，而且这几轮工作区完全没有变化"
            f"（{refs}），继续重试大概率是同样的结果。请查看这几轮的审计报告，"
            "补充说明或调整要求后再继续。"
        )
    if len(streak) >= STALL_MAX_ATTEMPTS:
        return (
            f"{name} 已连续 {len(streak)} 轮没有通过审计。请查看最近的审计报告，"
            "确认验收标准是否合理、是否缺少条件，补充说明后再继续。"
        )
    return None


# 函数说明：_trailing
# 用途：处理规划、执行、审计协作中的 `_trailing` 数据；结果及边界条件见下方说明。
# 参数：
#   rounds：已完成的模型或工具轮次，类型 `Sequence[MeaRound]`。
#   predicate：`predicate`输入或配置值，类型 `Callable[[MeaRound], bool]`。
# 返回：类型 `int`；返回 `count`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`predicate`。
# 分支与异常：
#   当 `not predicate(rnd)` 时，结束当前循环。
def _trailing(rounds: Sequence[MeaRound], predicate: Callable[[MeaRound], bool]) -> int:
    count = 0
    for rnd in reversed(rounds):
        if not predicate(rnd):
            break
        count += 1
    return count


# 函数说明：_state_entries
# 用途：在规划、执行、审计协作中处理 `_state_entries`，通过 `line.strip` 完成首个内部处
# 理步骤。
# 参数：
#   state：当前状态快照，类型 `str | None`。
# 返回：类型 `tuple[str, ...] | None`；按分支返回 `None`；
# `tuple((line[:_MAX_STATE_ENTRY_CHARS] for line in lines[:_MAX_STATE_ENTRIES]))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`state.splitlines`。
# 分支与异常：
#   当 `state is None` 时，返回 `None`。
def _state_entries(state: str | None) -> tuple[str, ...] | None:
    if state is None:
        return None
    lines = [line.strip() for line in state.splitlines() if line.strip()]
    return tuple(line[:_MAX_STATE_ENTRY_CHARS] for line in lines[:_MAX_STATE_ENTRIES])


# 函数说明：_next_step_id
# 用途：在规划、执行、审计协作中处理 `_next_step_id`，通过 `sid.startswith` 完成首个内部
# 处理步骤。
# 参数：
#   existing：已经存在的值或记录，类型 `Mapping[str, object]`。
# 返回：类型 `str`；返回 `f's{candidate}'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`sid.startswith` → `sid[1:].isdigit`。
def _next_step_id(existing: Mapping[str, object]) -> str:
    numbers = [int(sid[1:]) for sid in existing if sid.startswith("s") and sid[1:].isdigit()]
    candidate = max(numbers, default=0) + 1
    while f"s{candidate}" in existing:
        candidate += 1
    return f"s{candidate}"


# 函数说明：_step
# 用途：处理规划、执行、审计协作中的 `_step` 数据；结果及边界条件见下方说明。
# 参数：
#   task：当前任务记录，类型 `Task`。
#   step_id：目标步骤标识，类型 `str | None`。
# 返回：类型 `TaskStep`；返回 `step`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RoundConflict`。
# 分支与异常：
#   当 `step.id == step_id` 时，返回 `step`。
def _step(task: Task, step_id: str | None) -> TaskStep:
    for step in task.steps:
        if step.id == step_id:
            return step
    raise RoundConflict(f"任务步骤不存在：{step_id}")


# 函数说明：_step_view
# 用途：处理规划、执行、审计协作中的 `_step_view` 数据；结果及边界条件见下方说明。
# 参数：
#   step：当前任务步骤，类型 `TaskStep`。
#   rounds：已完成的模型或工具轮次，类型 `Sequence[MeaRound]`。
# 返回：类型 `StepView`；返回 `StepView(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StepView`。
# 分支与异常：
#   `rnd.step_id == step.id and rnd.auditor_report` 分支在完成前置处理后结束当前循环。
def _step_view(step: TaskStep, rounds: Sequence[MeaRound]) -> StepView:
    ref: str | None = None
    for rnd in reversed(rounds):
        if rnd.step_id == step.id and rnd.auditor_report:
            ref = rnd.ref
            break
    return StepView(
        id=step.id,
        title=step.title,
        status=step.status.value,
        acceptance=step.acceptance,
        status_ref=ref,
        origin_revision=step.origin_requirements_revision,
        superseded_by=step.superseded_by,
        superseded_reason=step.superseded_reason,
    )


# 函数说明：_step_views
# 用途：返回 `[_step_view(step, rounds) for step in task.steps]`，提供 规划、执行、审计
# 协作 的派生值。
# 参数：
#   task：当前任务记录，类型 `Task`。
#   rounds：已完成的模型或工具轮次，类型 `Sequence[MeaRound]`。
# 返回：类型 `list[StepView]`；返回 `[_step_view(step, rounds) for step in task.steps]`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_step_view`。
def _step_views(task: Task, rounds: Sequence[MeaRound]) -> list[StepView]:
    return [_step_view(step, rounds) for step in task.steps]


# 函数说明：_round_views
# 用途：返回 `[RoundView(index=rnd.index, kind=AuditKind(rnd.kind.value), step_id=rnd.
# step_id, subtask=…`，提供 规划、执行、审计协作 的派生值。
# 参数：
#   rounds：已完成的模型或工具轮次，类型 `Sequence[MeaRound]`。
# 返回：类型 `list[RoundView]`；返回 `[RoundView(index=rnd.index, kind=AuditKind(rnd.
# kind.value), step_id=rnd.step_id, subtask=…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RoundView` → `AuditKind`。
def _round_views(rounds: Sequence[MeaRound]) -> list[RoundView]:
    return [
        RoundView(
            index=rnd.index,
            kind=AuditKind(rnd.kind.value),
            step_id=rnd.step_id,
            subtask=rnd.subtask or "",
            auditor_report=rnd.auditor_report or "",
            harness_feedback=rnd.harness_feedback or "",
        )
        for rnd in rounds
    ]


# 函数说明：_last_satisfied_refs
# 用途：在规划、执行、审计协作中处理 `_last_satisfied_refs`，通过 `refs.append` 完成首个
# 内部处理步骤。
# 参数：
#   task：当前任务记录，类型 `Task`。
#   rounds：已完成的模型或工具轮次，类型 `Sequence[MeaRound]`。
# 返回：类型 `list[str]`；返回 `refs`。
# 分支与异常：
#   `rnd.step_id == step.id and rnd.step_acceptance ==…` 分支在完成前置处理后结束当前循
# 环。
def _last_satisfied_refs(task: Task, rounds: Sequence[MeaRound]) -> list[str]:
    refs: list[str] = []
    for step in task.steps:
        for rnd in reversed(rounds):
            if rnd.step_id == step.id and rnd.step_acceptance == StepAcceptance.SATISFIED.value:
                refs.append(rnd.ref)
                break
    return refs


__all__ = [
    "ROLE_NAMES",
    "AmendResult",
    "MeaFailure",
    "MeaRunner",
    "MeaStartError",
    "RoundConflict",
    "RunGateway",
    "SubRunResult",
]
