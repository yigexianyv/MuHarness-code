
"""MeaRunner 的离线测试：假的子 Run 网关按角色返回脚本文本，覆盖正常流程、
请示、取代、要求版本、终结决定，以及每个恢复窗口的崩溃注入。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from app.domain.conversation.store import SQLiteConversationStore
from app.domain.task import FileTaskStore, TaskOp, TaskStatus, TaskStep, TaskStepStatus
from app.models.types import AgentMode, Message, MessageRole
from app.runtime.agent.result import AgentResult, AgentStopReason
from app.runtime.mea.models import MeaStatus, RoundKind, RoundPhase
from app.runtime.mea.prompts import FINAL_RESPONSE_INSTRUCTIONS
from app.runtime.mea.requirements import AmendmentKind
from app.runtime.mea.runner import MeaRunner, MeaStartError
from app.runtime.mea.store import SQLiteMeaStore
from app.runtime.run import RunAlreadyExists, RunStatus

# ---------------------------------------------------------------------------
# 假的子 Run 网关
# ---------------------------------------------------------------------------


class Crash(Exception):
    """模拟进程在某个位置退出。"""


@dataclass
class FakeRun:
    status: RunStatus
    error: str | None = None


Reply = str | Callable[[str], Any]


@dataclass
class Script:
    manager: list[Reply] = field(default_factory=list)
    executor: list[Reply] = field(default_factory=list)
    auditor: list[Reply] = field(default_factory=list)
    final: Reply = "长任务已完成：CSV 已导入。"
    repair: Reply | None = None
    prompts: dict[AgentMode, list[str]] = field(default_factory=dict)

    # 函数说明：Script.reply
    # 用途：回复Script，供回归测试与测试辅助使用。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `str(item)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.prompts.setdefault` →
    # `prompt.startswith` → `self.manager.pop` → `self.executor.pop` →
    # `self.auditor.pop` → `callable`；另有 2 个调用点。
    async def reply(self, mode: AgentMode, prompt: str) -> str:
        self.prompts.setdefault(mode, []).append(prompt)
        if mode is AgentMode.MANAGE:
            if prompt.startswith(FINAL_RESPONSE_INSTRUCTIONS):
                item = self.final
            elif prompt.startswith("上一份 auditor 报告缺少有效的前四行控制头"):
                item = self.repair or ""
            else:
                item = self.manager.pop(0)
        elif mode is AgentMode.EXECUTE:
            item = self.executor.pop(0)
        else:
            item = self.auditor.pop(0)
        if callable(item):
            item = item(prompt)
            if inspect.isawaitable(item):
                item = await item
        return str(item)

    # 函数说明：Script.count
    # 用途：统计Script，供回归测试与测试辅助使用。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    # 返回：类型 `int`；返回 `len(self.prompts.get(mode, []))`。
    def count(self, mode: AgentMode) -> int:
        return len(self.prompts.get(mode, []))


class FakeRuns:
    """RunManager 的替身。table 模拟 runs 表，跨“进程”共享；任务句柄只属于当前进程。"""

    # 函数说明：FakeRuns.__init__
    # 用途：初始化 FakeRuns；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   table：`table`输入或配置值，类型 `dict[str, FakeRun]`。
    #   script：`script`输入或配置值，类型 `Script`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.table`、`self.script`、`self.started`、`self.start_metadata`
    # 、`self.fail_before_create`、`self._tasks`、`self._results`。
    def __init__(self, table: dict[str, FakeRun], script: Script) -> None:
        self.table = table
        self.script = script
        self.started: list[tuple[AgentMode, str]] = []
        self.start_metadata: dict[str, dict[str, Any]] = {}
        self.fail_before_create: Callable[[AgentMode], bool] | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._results: dict[str, AgentResult] = {}

    # 函数说明：FakeRuns.start
    # 用途：启动FakeRuns，供回归测试与测试辅助使用。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   run_id：目标运行标识，类型 `str`。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：返回 `(run_id, task)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.fail_before_create` →
    # `Crash` → `RunAlreadyExists` → `FakeRun` → `asyncio.create_task` → `self._execute`
    # 。
    # 分支与异常：
    #   当 `self.fail_before_create is not None and…` 时，抛出
    # `Crash(f'crash before creating {mode.value} run')`。
    #   当 `run_id in self.table` 时，抛出 `RunAlreadyExists(run_id)`。
    # 副作用与资源：
    #   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
    async def start(self, prompt: str, *, mode: AgentMode, run_id: str, **kwargs: Any):
        if self.fail_before_create is not None and self.fail_before_create(mode):
            raise Crash(f"crash before creating {mode.value} run")
        if run_id in self.table:
            raise RunAlreadyExists(run_id)
        self.table[run_id] = FakeRun(RunStatus.RUNNING)
        self.started.append((mode, run_id))
        self.start_metadata[run_id] = dict(
            kwargs.get("tool_context_metadata") or {}
        )
        task = asyncio.create_task(self._execute(run_id, mode, prompt))
        self._tasks[run_id] = task
        return run_id, task

    # 函数说明：FakeRuns._execute
    # 用途：执行FakeRuns，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.script.reply` → `FakeRun` →
    # `self._tasks.pop` → `AgentResult` → `Message`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，重新抛出原异常。
    #   捕获 `Exception` 后，返回 `None`。
    # 副作用与资源：
    #   更新对象字段：`self.table[run_id].status`。
    async def _execute(self, run_id: str, mode: AgentMode, prompt: str) -> None:
        try:
            text = await self.script.reply(mode, prompt)
        except asyncio.CancelledError:
            self.table[run_id].status = RunStatus.CANCELLED
            raise
        except Exception as exc:  # 和 RunManager 一样：记为失败，没有 AgentResult
            self.table[run_id] = FakeRun(RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            return
        finally:
            self._tasks.pop(run_id, None)
        self._results[run_id] = AgentResult(
            run_id=run_id,
            final_message=Message(role=MessageRole.ASSISTANT, content=text),
            messages=(),
            stop_reason=AgentStopReason.FINAL_ANSWER,
        )
        self.table[run_id].status = RunStatus.COMPLETED

    # 函数说明：FakeRuns.wait
    # 用途：等待FakeRuns，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `FakeRun`；返回 `self.table[run_id]`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，忽略该异常并继续当前流程。
    async def wait(self, run_id: str) -> FakeRun:
        task = self._tasks.get(run_id)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
        return self.table[run_id]

    # 函数说明：FakeRuns.result
    # 用途：返回 `self._results.get(run_id)`，提供 FakeRuns 的派生值。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `AgentResult | None`；返回 `self._results.get(run_id)`。
    def result(self, run_id: str) -> AgentResult | None:
        return self._results.get(run_id)

    # 函数说明：FakeRuns.forget_results
    # 用途：清除结果集合，供回归测试与测试辅助使用。
    # 参数：
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._results.pop`。
    def forget_results(self, run_ids: tuple[str, ...]) -> None:
        for run_id in run_ids:
            self._results.pop(run_id, None)

    # 函数说明：FakeRuns.get_run
    # 用途：获取运行，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `FakeRun | None`；返回 `self.table.get(run_id)`。
    async def get_run(self, run_id: str) -> FakeRun | None:
        return self.table.get(run_id)

    # 函数说明：FakeRuns.active_run_ids
    # 用途：运行`ids`，供回归测试与测试辅助使用。
    # 返回：类型 `tuple[str, ...]`；返回
    # `tuple((run_id for run_id, task in self._tasks.items() if not task.done()))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`task.done`。
    @property
    def active_run_ids(self) -> tuple[str, ...]:
        return tuple(run_id for run_id, task in self._tasks.items() if not task.done())

    # 函数说明：FakeRuns.cancel
    # 用途：取消FakeRuns，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `FakeRun`；返回 `self.table[run_id]`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`task.cancel`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，忽略该异常并继续当前流程。
    async def cancel(self, run_id: str) -> FakeRun:
        task = self._tasks.get(run_id)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return self.table[run_id]

    # 函数说明：FakeRuns.executor_starts
    # 用途：返回 `sum((1 for mode, _ in self.started if mode is AgentMode.EXECUTE))`，提
    # 供 FakeRuns 的派生值。
    # 返回：类型 `int`；返回
    # `sum((1 for mode, _ in self.started if mode is AgentMode.EXECUTE))`。
    def executor_starts(self) -> int:
        return sum(1 for mode, _ in self.started if mode is AgentMode.EXECUTE)


class CrashingTasks(FileTaskStore):
    """在指定操作写入 Task 之后抛出 Crash：模拟“Task 已写、round 未更新”。"""

    # 函数说明：CrashingTasks.__init__
    # 用途：初始化 CrashingTasks；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   path：目标文件或目录路径，类型 `Path`。
    #   crash_after：`crash_after`输入或配置值，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.crash_after`。
    def __init__(self, path: Path, crash_after: str | None = None) -> None:
        super().__init__(path)
        self.crash_after = crash_after

    # 函数说明：CrashingTasks.apply_op
    # 用途：应用`op`，供回归测试与测试辅助使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   op：传给 `super().apply_op` 的输入，类型 `TaskOp`。
    # 返回：返回 `outcome`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().apply_op` → `super` →
    # `op.op_id.endswith` → `Crash`。
    # 分支与异常：
    #   `self.crash_after and op.op_id.endswith(self.crash_after)` 分支在完成前置处理后
    # 抛出 `Crash(f'crash after {op.op_id}')`。
    # 副作用与资源：
    #   更新对象字段：`self.crash_after`。
    async def apply_op(self, task_id: str, op: TaskOp):
        outcome = await super().apply_op(task_id, op)
        if self.crash_after and op.op_id.endswith(self.crash_after):
            self.crash_after = None
            raise Crash(f"crash after {op.op_id}")
        return outcome


# ---------------------------------------------------------------------------
# 文本模板
# ---------------------------------------------------------------------------


# 函数说明：manager
# 用途：返回 `f'当前任务状态:\n{state}\n\n任务契约:\n- 目标状态: users 表包含 CSV 全部数
# 据\n\n步骤更新:\n{updates}\n\n依赖判断:\n- 本…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   route：路由输入或配置值，类型 `str`。
#   updates：`updates`输入或配置值，类型 `str`；默认 `'无'`。
#   state：当前状态快照，类型 `str`；默认 `'- 已完成: 见审计'`。
# 返回：类型 `str`；返回 `f'当前任务状态:\n{state}\n\n任务契约:\n- 目标状态: users 表包
# 含 CSV 全部数据\n\n步骤更新:\n{updates}\n\n依赖判断:\n- 本…`。
def manager(route: str, *, updates: str = "无", state: str = "- 已完成: 见审计") -> str:
    return (
        f"当前任务状态:\n{state}\n\n任务契约:\n- 目标状态: users 表包含 CSV 全部数据\n\n"
        f"步骤更新:\n{updates}\n\n依赖判断:\n- 本轮路由理由: 按依赖推进\n\n{route}"
    )


# 函数说明：execute
# 用途：执行回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   step：当前任务步骤，类型 `str`。
# 返回：类型 `str`；返回 `f'下一步: 执行任务\n步骤: {step}\n任务: 完成 {step}\n验收标准:
#  见步骤验收\n相关审计报告: 无\n相关已审计状态: 无\n边界: 不改表结构'`。
def execute(step: str) -> str:
    return (
        f"下一步: 执行任务\n步骤: {step}\n任务: 完成 {step}\n验收标准: 见步骤验收\n"
        "相关审计报告: 无\n相关已审计状态: 无\n边界: 不改表结构"
    )


# 函数说明：audit_only
# 用途：审计`only`，供回归测试与测试辅助使用。
# 参数：
#   step：当前任务步骤，类型 `str`。
# 返回：类型 `str`；返回 `f'下一步: 仅审计\n步骤: {step}\n核实重点: 数据是否已存在'`。
def audit_only(step: str) -> str:
    return f"下一步: 仅审计\n步骤: {step}\n核实重点: 数据是否已存在"


FINAL = "下一步: 最终验收\n验收重点: 行数与编码"


# 函数说明：report
# 用途：返回 `f'状态: {status}\n完整性: {integrity}\n契约审计: {contract}\n步骤验收: {
# step}\n审计事实: 已独立核验。\n\n验收约束反查:…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   status：目标状态，类型 `str`；默认 `'complete'`。
#   integrity：`integrity`输入或配置值，类型 `str`；默认 `'clean'`。
#   contract：`contract`输入或配置值，类型 `str`；默认 `'aligned'`。
#   step：当前任务步骤，类型 `str`；默认 `'satisfied'`。
#   blocking：`blocking`输入或配置值，类型 `str`；默认 `'无'`。
# 返回：类型 `str`；返回 `f'状态: {status}\n完整性: {integrity}\n契约审计: {contract}\n
# 步骤验收: {step}\n审计事实: 已独立核验。\n\n验收约束反查:…`。
def report(
    status: str = "complete",
    integrity: str = "clean",
    contract: str = "aligned",
    step: str = "satisfied",
    blocking: str = "无",
) -> str:
    return (
        f"状态: {status}\n完整性: {integrity}\n契约审计: {contract}\n步骤验收: {step}\n"
        "审计事实: 已独立核验。\n\n验收约束反查:\n"
        f"契约结论: {contract}\n阻断约束: {blocking}\n范围外约束: 无\n"
        "给任务管理器的状态更新: 见上"
    )


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------


@dataclass
class Env:
    root: Path
    script: Script
    table: dict[str, FakeRun]
    store: SQLiteMeaStore
    tasks: FileTaskStore
    workspace: Path
    sink: list[tuple[str, str, str]]
    runs: FakeRuns | None = None

    # 函数说明：Env.runner
    # 用途：处理回归测试与测试辅助中的 `runner` 数据；结果及边界条件见下方说明。
    # 参数：
    #   tasks：任务集合输入或配置值，类型 `FileTaskStore | None`；默认 `None`。
    #   sink：`sink`输入或配置值；默认 `None`。
    #   timeouts：`timeouts`输入或配置值；默认 `None`。
    # 返回：类型 `MeaRunner`；返回 `MeaRunner(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeRuns` → `MeaRunner` →
    # `object`。
    # 副作用与资源：
    #   更新对象字段：`self.runs`。
    def runner(
        self, *, tasks: FileTaskStore | None = None, sink=None, timeouts=None,
        executor_evidence=None,
    ) -> MeaRunner:
        self.runs = FakeRuns(self.table, self.script)

        # 函数说明：Env.runner.record
        # 用途：记录Env，供回归测试与测试辅助使用。
        # 参数：
        #   conversation_id：目标会话标识，类型 `str`。
        #   text：待处理的文本，类型 `str`。
        #   key：字段名或查询键，类型 `str`。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        async def record(conversation_id: str, text: str, key: str) -> None:
            self.sink.append((conversation_id, text, key))

        return MeaRunner(
            store=self.store,
            tasks=tasks or self.tasks,
            runs=self.runs,
            runtimes={AgentMode.MANAGE: object(), AgentMode.EXECUTE: object(),
                      AgentMode.AUDIT: object()},
            workspace_root=self.workspace,
            final_message_sink=sink or record,
            role_timeouts=timeouts,
            executor_evidence=executor_evidence,
        )


# 函数说明：_env
# 用途：在回归测试与测试辅助中处理 `_env`，通过 `store.initialize` 完成首个内部处理步骤
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   script：传给 `Env` 的输入，类型 `Script`。
# 返回：类型 `Env`；返回 `Env(tmp_path, script, {}, store, tasks, workspace, [])`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteMeaStore` → `store.initialize`
# → `FileTaskStore` → `tasks.initialize` → `workspace.mkdir` →
# `(workspace / 'users.csv').write_text`；另有 1 个调用点。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / 'users.csv').write_text`。
async def _env(tmp_path: Path, script: Script) -> Env:
    store = SQLiteMeaStore(tmp_path / "mea.db")
    await store.initialize()
    tasks = FileTaskStore(tmp_path / "tasks")
    await tasks.initialize()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "users.csv").write_text("id,name\n1,a\n", encoding="utf-8")
    return Env(tmp_path, script, {}, store, tasks, workspace, [])


# 函数说明：_task
# 用途：在回归测试与测试辅助中处理 `_task`，通过 `env.tasks.create` 完成首个内部处理步骤
# 。
# 参数：
#   env：环境变量映射，类型 `Env`。
#   acceptance：`acceptance`输入或配置值，类型 `bool`；默认 `True`。
# 返回：返回 `await env.tasks.set_status(task.id, TaskStatus.ACTIVE)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`env.tasks.create` → `TaskStep` →
# `env.tasks.apply_patch` → `_constraint_patch` → `env.tasks.set_status`。
async def _task(env: Env, *, acceptance: bool = True):
    task = await env.tasks.create(
        owner_conversation_id="conv-1",
        title="导入用户",
        goal="users 表包含 CSV 全部数据",
        description="CSV 是 UTF-8",
        steps=(
            TaskStep(id="s1", title="读取 CSV", acceptance="读到表头和数据" if acceptance else None),
            TaskStep(id="s2", title="导入数据库", acceptance="users 表行数正确"),
        ),
    )
    task = await env.tasks.apply_patch(task.id, _constraint_patch())
    return await env.tasks.set_status(task.id, TaskStatus.ACTIVE)


# 函数说明：_constraint_patch
# 用途：处理回归测试与测试辅助中的 `_constraint_patch` 数据；结果及边界条件见下方说明。
# 返回：返回 `TaskPatch(add_constraints=('不要修改数据库结构',))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskPatch`。
def _constraint_patch():
    from app.domain.task import TaskPatch

    return TaskPatch(add_constraints=("不要修改数据库结构",))


# 函数说明：_start
# 用途：启动回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   env：环境变量映射，类型 `Env`。
#   runner：`runner`输入或配置值，类型 `MeaRunner`。
# 返回：类型 `str`；返回 `mea.id`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_task` → `runner.start`。
async def _start(env: Env, runner: MeaRunner) -> str:
    task = await _task(env)
    mea = await runner.start(
        task_id=task.id, conversation_id="conv-1", original_request="把 users.csv 导入数据库",
        spawn=False,
    )
    return mea.id


# 函数说明：_steps
# 用途：执行 `_steps` 测试辅助流程并检查预期结果。
# 参数：
#   env：环境变量映射，类型 `Env`。
#   mea_id：长任务协作记录标识，类型 `str`。
# 返回：类型 `dict[str, TaskStepStatus]`；返回
# `{step.id: step.status for step in task.steps}`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`env.store.require`。
# 分支与异常：
#   验证条件：`task is not None`。
async def _steps(env: Env, mea_id: str) -> dict[str, TaskStepStatus]:
    mea = await env.store.require(mea_id)
    task = await env.tasks.get(mea.task_id)
    assert task is not None
    return {step.id: step.status for step in task.steps}


# ---------------------------------------------------------------------------
# 正常流程
# ---------------------------------------------------------------------------


# 函数说明：test_full_run_completes_task_through_final_audit
# 用途：回归验证回归测试与测试辅助中的 `full_run_completes_task_through_final_audit` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.COMPLETED`。
#   验证条件：`mea.final_response == '长任务已完成：CSV 已导入。'`。
#   验证条件：`task is not None and task.status is TaskStatus.COMPLETED`。
#   验证条件：
# `[s.status for s in task.steps] == [TaskStepStatus.DONE, TaskStepStatus.DONE]`。
async def test_full_run_completes_task_through_final_audit(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(execute("s1")), manager(execute("s2")), manager(FINAL)],
        executor=["读到 1 行数据", "imported 1"],
        auditor=[report(), report(), report(step="not_applicable")],
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    mea = await runner.run(mea_id)

    assert mea.status is MeaStatus.COMPLETED
    assert mea.final_response == "长任务已完成：CSV 已导入。"
    task = await env.tasks.get(mea.task_id)
    assert task is not None and task.status is TaskStatus.COMPLETED
    assert [s.status for s in task.steps] == [TaskStepStatus.DONE, TaskStepStatus.DONE]
    assert task.contract == "- 目标状态: users 表包含 CSV 全部数据"
    assert env.sink == [("conv-1", "长任务已完成：CSV 已导入。", f"{mea_id}/final")]
    # 权威要求原文进入三个角色的输入
    for mode in (AgentMode.MANAGE, AgentMode.EXECUTE, AgentMode.AUDIT):
        assert "- 不要修改数据库结构" in script.prompts[mode][0]
        assert "[原始请求]\n把 users.csv 导入数据库" in script.prompts[mode][0]
    rounds = await env.store.rounds(mea_id)
    assert [r.kind for r in rounds] == [RoundKind.NORMAL, RoundKind.NORMAL, RoundKind.FINAL_AUDIT]
    assert env.runs is not None
    for audit_index, rnd in enumerate(rounds[:2]):
        expected = {
            "task_id": mea.task_id,
            "mea_id": mea_id,
            "task_step_id": rnd.step_id,
        }
        assert env.runs.start_metadata[rnd.executor_run_id] == expected
        assert env.runs.start_metadata[rnd.auditor_run_id] == expected
        assert (
            f'artifact_list(task_id="{mea.task_id}", run_id="{rnd.executor_run_id}")'
            in script.prompts[AgentMode.AUDIT][audit_index]
        )


# 函数说明：test_exploration_round_keeps_step_open
# 用途：回归验证回归测试与测试辅助中的 `exploration_round_keeps_step_open` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.BLOCKED`。
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.TODO`。
#   验证条件：`task is not None and task.status is TaskStatus.ACTIVE`。
async def test_exploration_round_keeps_step_open(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞\n原因: 测试到此为止")],
        executor=["只看了目录结构"],
        auditor=[report(step="not_satisfied")],  # 子任务 complete，但步骤未满足
        final="阻塞了",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    mea = await runner.run(mea_id)

    assert mea.status is MeaStatus.BLOCKED
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.TODO
    task = await env.tasks.get(mea.task_id)
    assert task is not None and task.status is TaskStatus.ACTIVE


# 函数说明：test_audit_only_marks_todo_step_done_without_executor
# 用途：回归验证回归测试与测试辅助中的
# `audit_only_marks_todo_step_done_without_executor` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `report` → `_env` → `env.runner`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.DONE`。
#   验证条件：`env.runs is not None and env.runs.executor_starts() == 0`。
async def test_audit_only_marks_todo_step_done_without_executor(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(audit_only("s1")), manager("下一步: 阻塞")],
        auditor=[report()],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    await runner.run(mea_id)

    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.DONE
    assert env.runs is not None and env.runs.executor_starts() == 0


# 函数说明：test_blocking_constraints_downgrade_step_verdict
# 用途：回归验证回归测试与测试辅助中的 `blocking_constraints_downgrade_step_verdict` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.TODO`。
#   验证条件：`(rnd.audit_status, rnd.step_acceptance, rnd.contract_audit_status) == ('
# incomplete', 'not_satisfied', 'unknown')`。
async def test_blocking_constraints_downgrade_step_verdict(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=["done"],
        auditor=[report(blocking="表结构是否保持不变: unknown")],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    await runner.run(mea_id)

    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.TODO
    rnd = (await env.store.rounds(mea_id))[0]
    assert (rnd.audit_status, rnd.step_acceptance, rnd.contract_audit_status) == (
        "incomplete", "not_satisfied", "unknown"
    )


# 函数说明：test_three_invalid_routes_wait_for_user
# 用途：回归验证回归测试与测试辅助中的 `three_invalid_routes_wait_for_user` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `_env` → `env.runner` →
# `_start` → `runner.run` → `env.store.rounds`。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.WAITING_USER`。
#   验证条件：`[r.route for r in rounds] == ['invalid'] * 3`。
#   验证条件：`'没有有效路由' in (rounds[0].harness_feedback or '')`。
#   验证条件：`'没有有效路由' in script.prompts[AgentMode.MANAGE][1]`。
async def test_three_invalid_routes_wait_for_user(tmp_path: Path) -> None:
    script = Script(manager=["随便说说"] * 3)
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    mea = await runner.run(mea_id)

    assert mea.status is MeaStatus.WAITING_USER
    rounds = await env.store.rounds(mea_id)
    assert [r.route for r in rounds] == ["invalid"] * 3
    assert "没有有效路由" in (rounds[0].harness_feedback or "")
    assert "没有有效路由" in script.prompts[AgentMode.MANAGE][1]  # 反馈进入下一轮输入


# 函数说明：test_final_audit_with_open_steps_is_invalid
# 用途：回归验证回归测试与测试辅助中的 `final_audit_with_open_steps_is_invalid` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `_env` →
# `env.runner` → `_start` → `runner.run`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`first.route == 'invalid'`。
#   验证条件：`'s1（尚未通过步骤验收）' in (first.harness_feedback or '')`。
async def test_final_audit_with_open_steps_is_invalid(tmp_path: Path) -> None:
    script = Script(manager=[manager(FINAL), manager("下一步: 阻塞")], final="停")
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    await runner.run(mea_id)

    first = (await env.store.rounds(mea_id))[0]
    assert first.route == "invalid"
    assert "s1（尚未通过步骤验收）" in (first.harness_feedback or "")


# 函数说明：test_ask_then_answer_adds_amendment_for_every_role
# 用途：回归验证回归测试与测试辅助中的 `ask_then_answer_adds_amendment_for_every_role`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `report` → `_env` → `env.runner`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`waiting.status is MeaStatus.WAITING_USER`。
#   验证条件：`waiting.pending_question == '用测试库还是生产库？'`。
#   验证条件：`waiting.pending_choices == ('测试库', '生产库')`。
#   验证条件：
# `result.accepted and result.amendment_id == 'A1' and (result.revision == 2)`。
@pytest.mark.parametrize(
    ("question", "choices", "answer"),
    [
        ("用测试库还是生产库？", "测试库 | 生产库", "测试库"),
        ("是否删除 demo/b.txt？", "授权删除 `demo/b.txt` | 不授权并终止任务",
         "授权删除 `demo/b.txt`"),
    ],
)
async def test_ask_then_answer_adds_amendment_for_every_role(
    tmp_path: Path, question: str, choices: str, answer: str
) -> None:
    script = Script(
        manager=[
            manager(f"下一步: 请示用户\n问题: {question}\n选项: {choices}"),
            manager(audit_only("s1")),
            manager("下一步: 阻塞"),
        ],
        auditor=[report()],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    waiting = await runner.run(mea_id)
    assert waiting.status is MeaStatus.WAITING_USER
    assert waiting.pending_question == question
    assert waiting.pending_choices == tuple(choices.split(" | "))

    result = await runner.answer(mea_id, waiting.pending_choices[0])
    assert result.accepted and result.amendment_id == "A1" and result.revision == 2
    await runner.wait(mea_id)

    assert "A1（生效于要求 v2" in script.prompts[AgentMode.MANAGE][1]
    assert f"请示回答）: {answer}" in script.prompts[AgentMode.MANAGE][1]
    assert f"请示回答）: {answer}" in script.prompts[AgentMode.AUDIT][0]


# 函数说明：test_start_rejects_steps_without_acceptance
# 用途：回归验证回归测试与测试辅助中的 `start_rejects_steps_without_acceptance` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_env` → `Script` → `env.runner` →
# `_task` → `pytest.raises` → `runner.start`。
# 分支与异常：
#   预期异常：`pytest.raises(MeaStartError, match='s1')`。
async def test_start_rejects_steps_without_acceptance(tmp_path: Path) -> None:
    env = await _env(tmp_path, Script())
    runner = env.runner()
    task = await _task(env, acceptance=False)

    with pytest.raises(MeaStartError, match="s1"):
        await runner.start(task_id=task.id, conversation_id="conv-1", original_request="x", spawn=False)


# ---------------------------------------------------------------------------
# 取代
# ---------------------------------------------------------------------------


# 函数说明：test_supersede_requires_a_later_user_amendment
# 用途：回归验证回归测试与测试辅助中的 `supersede_requires_a_later_user_amendment` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `_env` →
# `env.runner` → `_start` → `runner.run`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`(await _steps(env, mea_id))['s2'] is TaskStepStatus.TODO`。
#   验证条件：`'权威要求里不存在修订 A9' in (first.harness_feedback or '')`。
async def test_supersede_requires_a_later_user_amendment(tmp_path: Path) -> None:
    script = Script(
        manager=[
            manager("下一步: 阻塞", updates="- 取代: s2 | 依据: A9 | 原因: 不想做了"),
        ],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    await runner.run(mea_id)

    assert (await _steps(env, mea_id))["s2"] is TaskStepStatus.TODO
    first = (await env.store.rounds(mea_id))[0]
    assert "权威要求里不存在修订 A9" in (first.harness_feedback or "")


# 函数说明：test_supersede_with_amendment_then_complete
# 用途：回归验证回归测试与测试辅助中的 `supersede_with_amendment_then_complete` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `execute` → `report` → `_env`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.COMPLETED`。
#   验证条件：`steps == {'s1': TaskStepStatus.DONE, 's2': TaskStepStatus.SUPERSEDED, 's3
# ': TaskStepStatus.DONE}`。
#   验证条件：`task is not None and task.steps[2].origin_requirements_revision == 2`。
#   验证条件：`' - s2 — 原验收: users 表行数正确 — 依据 A1: 不导入数据库了，改为导出 CSV
# ' in final_prompt`。
async def test_supersede_with_amendment_then_complete(tmp_path: Path) -> None:
    script = Script(
        manager=[
            manager(
                audit_only("s1"),
                updates="- 取代: s2 | 依据: A1 | 原因: 用户改为只导出 CSV\n"
                        "- 新增: 导出 CSV | 验收: out/users.csv 存在",
            ),
            manager(execute("s3")),
            manager(FINAL),
        ],
        executor=["exported"],
        auditor=[report(), report(), report(step="not_applicable")],
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    await runner.add_amendment(mea_id, "不导入数据库了，改为导出 CSV")

    mea = await runner.run(mea_id)

    assert mea.status is MeaStatus.COMPLETED
    steps = await _steps(env, mea_id)
    assert steps == {"s1": TaskStepStatus.DONE, "s2": TaskStepStatus.SUPERSEDED,
                     "s3": TaskStepStatus.DONE}
    task = await env.tasks.get(mea.task_id)
    assert task is not None and task.steps[2].origin_requirements_revision == 2
    final_prompt = script.prompts[AgentMode.AUDIT][-1]
    assert "  - s2 — 原验收: users 表行数正确 — 依据 A1: 不导入数据库了，改为导出 CSV" in final_prompt


# ---------------------------------------------------------------------------
# 要求版本与终结决定
# ---------------------------------------------------------------------------


# 函数说明：test_amendment_during_final_audit_blocks_completion
# 用途：回归验证回归测试与测试辅助中的 `amendment_during_final_audit_blocks_completion`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `report` → `_env` → `env.runner`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`task is not None and task.status is TaskStatus.ACTIVE`。
#   验证条件：`final_round.stale_requirements is True`。
#   验证条件：`'期间权威要求已更新到 v2（新增 A1: 还要保留原文件）' in (final_round.
# harness_feedback or '')`。
#   验证条件：`'A1（生效于要求 v2' in script.prompts[AgentMode.MANAGE][3]`。
async def test_amendment_during_final_audit_blocks_completion(tmp_path: Path) -> None:
    holder: dict[str, Any] = {}

    # 函数说明：test_amendment_during_final_audit_blocks_completion.final_report
    # 用途：在回归测试与测试辅助中处理 `final_report`，通过
    # `holder['runner'].add_amendment` 完成首个内部处理步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `report(step='not_applicable')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`holder['runner'].add_amendment` →
    #  `report`。
    # 闭包依赖：从外层读取 `holder`。
    async def final_report(prompt: str) -> str:
        await holder["runner"].add_amendment(holder["mea_id"], "还要保留原文件")
        return report(step="not_applicable")

    script = Script(
        manager=[manager(audit_only("s1")), manager(audit_only("s2")), manager(FINAL),
                 manager("下一步: 阻塞")],
        auditor=[report(), report(), final_report],
        final="停",
    )
    env = await _env(tmp_path, script)
    holder["runner"] = runner = env.runner()
    holder["mea_id"] = mea_id = await _start(env, runner)

    mea = await runner.run(mea_id)

    task = await env.tasks.get(mea.task_id)
    assert task is not None and task.status is TaskStatus.ACTIVE  # 旧要求下的结论不完成任务
    final_round = (await env.store.rounds(mea_id))[2]
    assert final_round.stale_requirements is True
    assert "期间权威要求已更新到 v2（新增 A1: 还要保留原文件）" in (final_round.harness_feedback or "")
    assert "A1（生效于要求 v2" in script.prompts[AgentMode.MANAGE][3]
    assert mea.status is MeaStatus.BLOCKED


# 函数说明：test_amendment_after_completion_decision_is_rejected
# 用途：回归验证回归测试与测试辅助中的 `amendment_after_completion_decision_is_rejected`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Event` → `Script` → `manager`
#  → `audit_only` → `report` → `_env`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`gate.is_set()`。
#   验证条件：`replies[0].accepted is False and replies[0].reason == 'mea_finalized'`。
#   验证条件：`(await env.store.requirements(mea_id)).revision == 1`。
#   验证条件：`mea.status is MeaStatus.COMPLETED`。
async def test_amendment_after_completion_decision_is_rejected(tmp_path: Path) -> None:
    gate = asyncio.Event()
    replies: list[Any] = []

    # 函数说明：test_amendment_after_completion_decision_is_rejected.slow_final
    # 用途：在回归测试与测试辅助中处理 `slow_final`，通过 `replies.append` 完成首个内部
    # 处理步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `'完成了'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`runner.add_amendment` →
    # `gate.set`。
    # 闭包依赖：从外层读取 `gate`、`mea_id`、`replies`、`runner`。
    async def slow_final(prompt: str) -> str:
        replies.append(await runner.add_amendment(mea_id, "再加一条"))
        gate.set()
        return "完成了"

    script = Script(
        manager=[manager(audit_only("s1")), manager(audit_only("s2")), manager(FINAL)],
        auditor=[report(), report(), report(step="not_applicable")],
        final=slow_final,
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    mea = await runner.run(mea_id)

    assert gate.is_set()
    assert replies[0].accepted is False and replies[0].reason == "mea_finalized"
    assert (await env.store.requirements(mea_id)).revision == 1
    assert mea.status is MeaStatus.COMPLETED
    assert "再加一条" not in (mea.final_response or "")


# 函数说明：test_crash_between_decision_and_task_write_finishes_once
# 用途：回归验证回归测试与测试辅助中的
# `crash_between_decision_and_task_write_finishes_once` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `report` → `_env` → `SQLiteConversationStore`；另有 13 个调用点。
# 分支与异常：
#   验证条件：`(await env.store.require(mea_id)).status is MeaStatus.FINALIZING`。
#   验证条件：`(await restarted.add_amendment(mea_id, '晚到的补充')).accepted is False`
# 。
#   验证条件：`mea.status is MeaStatus.COMPLETED`。
#   验证条件：`task is not None and task.status is TaskStatus.COMPLETED`。
#   预期异常：`pytest.raises(Crash)`。
async def test_crash_between_decision_and_task_write_finishes_once(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(audit_only("s1")), manager(audit_only("s2")), manager(FINAL)],
        auditor=[report(), report(), report(step="not_applicable")],
        final="最终回复一次",
    )
    env = await _env(tmp_path, script)
    conversations = SQLiteConversationStore(tmp_path / "conv.db")
    await conversations.initialize()
    conversation = await conversations.create(title="t")

    # 函数说明：test_crash_between_decision_and_task_write_finishes_once.sink
    # 用途：在回归测试与测试辅助中处理 `sink`，通过 `conversations.append_messages` 完成
    # 首个内部处理步骤。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   text：待处理的文本，类型 `str`。
    #   key：字段名或查询键，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`conversations.append_messages` →
    # `Message`。
    # 闭包依赖：从外层读取 `conversation`、`conversations`。
    async def sink(conversation_id: str, text: str, key: str) -> None:
        await conversations.append_messages(
            conversation.id,
            (Message(role=MessageRole.ASSISTANT, content=text),),
            idempotency_key=key,
        )

    crashing = CrashingTasks(tmp_path / "tasks", crash_after="/complete")
    runner = env.runner(tasks=crashing, sink=sink)
    mea_id = await _start(env, runner)

    with pytest.raises(Crash):
        await runner.run(mea_id)
    assert (await env.store.require(mea_id)).status is MeaStatus.FINALIZING

    # 新进程：Host 启动对账直接补齐 finalizing，不跑新轮次，也不接受修订
    restarted = env.runner(sink=sink)
    assert (await restarted.add_amendment(mea_id, "晚到的补充")).accepted is False
    await restarted.reconcile()
    mea = await restarted.wait(mea_id)
    await restarted.reconcile()  # 再对账一次也不会重复

    assert mea.status is MeaStatus.COMPLETED
    task = await env.tasks.get(mea.task_id)
    assert task is not None and task.status is TaskStatus.COMPLETED
    assert list(task.applied_ops).count(f"{mea_id}/complete") == 1
    messages = await conversations.load_messages(conversation.id)
    assert [m.content for m in messages] == ["最终回复一次"]


# ---------------------------------------------------------------------------
# 恢复窗口（崩溃注入）
# ---------------------------------------------------------------------------


# 函数说明：test_bound_executor_id_without_run_rechecks_requirements
# 用途：回归验证回归测试与测试辅助中的
# `bound_executor_id_without_run_rechecks_requirements` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `_env` → `env.runner` → `_start`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`env.runs is not None`。
#   验证条件：
# `crashed.phase is RoundPhase.EXECUTING and crashed.executor_run_id is not None`。
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.IN_PROGRESS`。
#   验证条件：`env.runs is not None and env.runs.executor_starts() == 0`。
#   预期异常：`pytest.raises(Crash)`。
# 副作用与资源：
#   更新对象字段：`env.runs.fail_before_create`。
async def test_bound_executor_id_without_run_rechecks_requirements(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=["不应该运行"],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    assert env.runs is not None
    env.runs.fail_before_create = lambda mode: mode is AgentMode.EXECUTE

    with pytest.raises(Crash):
        await runner.run(mea_id)
    crashed = (await env.store.rounds(mea_id))[0]
    assert crashed.phase is RoundPhase.EXECUTING and crashed.executor_run_id is not None
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.IN_PROGRESS

    restarted = env.runner()
    await restarted.add_amendment(mea_id, "改用测试库")  # 暂停期间修改要求
    await restarted.run(mea_id)

    assert env.runs is not None and env.runs.executor_starts() == 0
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.TODO  # begin 已释放
    rounds = await env.store.rounds(mea_id)
    assert rounds[0].phase is RoundPhase.ABANDONED
    assert "A1（生效于要求 v2" in script.prompts[AgentMode.MANAGE][1]


# 函数说明：test_bound_executor_id_without_run_starts_with_same_id
# 用途：回归验证回归测试与测试辅助中的
# `bound_executor_id_without_run_starts_with_same_id` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`env.runs is not None`。
#   验证条件：`[run_id for mode, run_id in env.runs.started if mode is AgentMode.EXECUTE
# ] == [bound]`。
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.DONE`。
#   预期异常：`pytest.raises(Crash)`。
# 副作用与资源：
#   更新对象字段：`env.runs.fail_before_create`。
async def test_bound_executor_id_without_run_starts_with_same_id(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=["done"],
        auditor=[report()],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    assert env.runs is not None
    env.runs.fail_before_create = lambda mode: mode is AgentMode.EXECUTE
    with pytest.raises(Crash):
        await runner.run(mea_id)
    bound = (await env.store.rounds(mea_id))[0].executor_run_id

    restarted = env.runner()
    await restarted.run(mea_id)

    assert env.runs is not None
    assert [run_id for mode, run_id in env.runs.started if mode is AgentMode.EXECUTE] == [bound]
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.DONE


# 函数说明：test_process_restart_during_executor_goes_to_recovery_audit
# 用途：回归验证回归测试与测试辅助中的
# `process_restart_during_executor_goes_to_recovery_audit` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`first_runs is not None`。
#   验证条件：`log.read_text(encoding='utf-8') == 'line\n'`。
#   验证条件：`env.runs is not None and env.runs.executor_starts() == 0`。
#   验证条件：`rounds[0].phase is RoundPhase.APPLIED and rounds[0].interrupt_reason`。
#   预期异常：`pytest.raises(Crash)`。
# 副作用与资源：
#   更新对象字段：`first_runs.wait`。
#   文件或资源访问：`log.read_text`。
async def test_process_restart_during_executor_goes_to_recovery_audit(tmp_path: Path) -> None:
    log = tmp_path / "workspace" / "log.txt"

    # 函数说明：test_process_restart_during_executor_goes_to_recovery_audit.append_line
    # 用途：追加`line`，供回归测试与测试辅助使用。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `'写了一行'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`log.open` → `handle.write`。
    # 副作用与资源：
    #   文件或资源访问：`log.open`。
    # 闭包依赖：从外层读取 `log`。
    def append_line(prompt: str) -> str:
        with log.open("a", encoding="utf-8") as handle:
            handle.write("line\n")
        return "写了一行"

    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=[append_line],
        auditor=[report()],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    first_runs = env.runs
    assert first_runs is not None

    # 函数说明：
    # test_process_restart_during_executor_goes_to_recovery_audit.crash_on_executor_wait
    # 用途：等待执行者，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：返回 `await FakeRuns.wait(first_runs, run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.sleep` → `Crash` →
    # `FakeRuns.wait`。
    # 分支与异常：
    #   `any(…)` 分支在完成前置处理后抛出
    # `Crash('process died while waiting for executor')`。
    # 闭包依赖：从外层读取 `first_runs`。
    async def crash_on_executor_wait(run_id: str):
        if any(mode is AgentMode.EXECUTE and rid == run_id for mode, rid in first_runs.started):
            await asyncio.sleep(0.05)  # Executor 已经跑过（写了一行），进程随即退出
            raise Crash("process died while waiting for executor")
        return await FakeRuns.wait(first_runs, run_id)

    first_runs.wait = crash_on_executor_wait  # type: ignore[method-assign]
    with pytest.raises(Crash):
        await runner.run(mea_id)

    restarted = env.runner()  # 新进程：Run 记录存在，但不在本进程运行
    await restarted.run(mea_id)

    assert log.read_text(encoding="utf-8") == "line\n"  # 只追加了一次
    assert env.runs is not None and env.runs.executor_starts() == 0
    rounds = await env.store.rounds(mea_id)
    assert rounds[0].phase is RoundPhase.APPLIED and rounds[0].interrupt_reason
    assert rounds[1].kind is RoundKind.RECOVERY_AUDIT
    assert "中断说明" in script.prompts[AgentMode.AUDIT][0]
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.DONE


# 函数说明：test_verdict_written_before_round_update_is_replayed_not_recomputed
# 用途：回归验证回归测试与测试辅助中的
# `verdict_written_before_round_update_is_replayed_not_recomputed` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `CrashingTasks`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`(await env.store.rounds(mea_id))[0].phase is RoundPhase.AUDITED`。
#   验证条件：`mea.status is not MeaStatus.FAILED`。
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.DONE`。
#   验证条件：`'A1（生效于要求 v2' in script.prompts[AgentMode.MANAGE][1]`。
#   预期异常：`pytest.raises(Crash)`。
async def test_verdict_written_before_round_update_is_replayed_not_recomputed(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=["done"],
        auditor=[report()],
        final="停",
    )
    env = await _env(tmp_path, script)
    crashing = CrashingTasks(tmp_path / "tasks", crash_after="/verdict")
    runner = env.runner(tasks=crashing)
    mea_id = await _start(env, runner)

    with pytest.raises(Crash):
        await runner.run(mea_id)
    assert (await env.store.rounds(mea_id))[0].phase is RoundPhase.AUDITED

    restarted = env.runner()
    await restarted.add_amendment(mea_id, "新增要求")  # 恢复前要求更新
    mea = await restarted.run(mea_id)

    assert mea.status is not MeaStatus.FAILED  # 没有 OpIdReused
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.DONE  # 保持首次确定的结论
    assert "A1（生效于要求 v2" in script.prompts[AgentMode.MANAGE][1]


# 函数说明：test_immediate_amendment_cancels_executor_and_recovers
# 用途：回归验证回归测试与测试辅助中的
# `immediate_amendment_cancels_executor_and_recovers` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Event` → `Script` → `manager`
#  → `execute` → `report` → `_env`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`rounds[0].interrupt_reason == 'Executor 被取消'`。
#   验证条件：`rounds[1].kind is RoundKind.RECOVERY_AUDIT`。
#   验证条件：`'A1（生效于要求 v2' in script.prompts[AgentMode.AUDIT][0]`。
async def test_immediate_amendment_cancels_executor_and_recovers(tmp_path: Path) -> None:
    started = asyncio.Event()

    # 函数说明：test_immediate_amendment_cancels_executor_and_recovers.slow_executor
    # 用途：在回归测试与测试辅助中处理 `slow_executor`，通过 `started.set` 完成首个内部
    # 处理步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `'never'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`started.set` → `asyncio.sleep`。
    # 闭包依赖：从外层读取 `started`。
    async def slow_executor(prompt: str) -> str:
        started.set()
        await asyncio.sleep(3600)
        return "never"

    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=[slow_executor],
        auditor=[report(step="not_satisfied")],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    loop = runner.spawn(mea_id)

    await asyncio.wait_for(started.wait(), 5)
    await runner.add_amendment(mea_id, "立即改用测试库", immediate=True)
    await asyncio.wait_for(loop, 10)

    rounds = await env.store.rounds(mea_id)
    assert rounds[0].interrupt_reason == "Executor 被取消"
    assert rounds[1].kind is RoundKind.RECOVERY_AUDIT
    assert "A1（生效于要求 v2" in script.prompts[AgentMode.AUDIT][0]


# ---------------------------------------------------------------------------
# 审计闸门
# ---------------------------------------------------------------------------


# 函数说明：test_auditor_writing_workspace_voids_the_audit
# 用途：回归验证回归测试与测试辅助中的 `auditor_writing_workspace_voids_the_audit` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `_env` → `env.runner` → `_start`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`rnd.integrity_status == 'violation'`。
#   验证条件：`'+ auditor.txt' in (rnd.auditor_report or '')`。
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.TODO`。
async def test_auditor_writing_workspace_voids_the_audit(tmp_path: Path) -> None:
    # 函数说明：test_auditor_writing_workspace_voids_the_audit.writing_auditor
    # 用途：在回归测试与测试辅助中处理 `writing_auditor`，通过
    # `(tmp_path / 'workspace' / 'auditor.txt').write_text` 完成首个内部处理步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `report()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `(tmp_path / 'workspace' / 'auditor.txt').write_text` → `report`。
    # 副作用与资源：
    #   文件或资源访问：`(tmp_path / 'workspace' / 'auditor.txt').write_text`。
    # 闭包依赖：从外层读取 `tmp_path`。
    def writing_auditor(prompt: str) -> str:
        (tmp_path / "workspace" / "auditor.txt").write_text("oops", encoding="utf-8")
        return report()

    script = Script(
        manager=[manager(audit_only("s1")), manager("下一步: 阻塞")],
        auditor=[writing_auditor],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    await runner.run(mea_id)

    rnd = (await env.store.rounds(mea_id))[0]
    assert rnd.integrity_status == "violation"
    assert "+ auditor.txt" in (rnd.auditor_report or "")
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.TODO


# 函数说明：test_missing_header_is_repaired_once
# 用途：回归验证回归测试与测试辅助中的 `missing_header_is_repaired_once` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `report` → `_env` → `env.runner`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.DONE`。
async def test_missing_header_is_repaired_once(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(audit_only("s1")), manager("下一步: 阻塞")],
        auditor=["看起来都好了"],
        repair=report(),
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    await runner.run(mea_id)

    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.DONE


# 函数说明：test_unrepairable_header_is_synthesized_as_blocked
# 用途：回归验证回归测试与测试辅助中的 `unrepairable_header_is_synthesized_as_blocked`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `_env` → `env.runner` → `_start`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`(rnd.audit_status, rnd.integrity_status, rnd.step_acceptance) == ('
# blocked', 'suspect', 'not_satisfied')`。
async def test_unrepairable_header_is_synthesized_as_blocked(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(audit_only("s1")), manager("下一步: 阻塞")],
        auditor=["没有控制头"],
        repair="还是没有",
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    await runner.run(mea_id)

    rnd = (await env.store.rounds(mea_id))[0]
    assert (rnd.audit_status, rnd.integrity_status, rnd.step_acceptance) == (
        "blocked", "suspect", "not_satisfied"
    )


@pytest.mark.parametrize("route", [execute("s1"), audit_only("s1")])
async def test_step_audit_wrong_scope_is_repaired_without_reexecuting(tmp_path, route):
    script = Script(
        manager=[manager(route), manager("下一步: 阻塞")],
        executor=["读取内容 v2"],
        auditor=[report(step="not_applicable") + "\n实际读取结果: v2"],
        repair=report(),
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    await runner.run(mea_id)
    rnd = (await env.store.rounds(mea_id))[0]
    assert rnd.step_acceptance == "satisfied"
    assert "实际读取结果: v2" in rnd.auditor_report
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.DONE
    assert env.runs.executor_starts() == (1 if route == execute("s1") else 0)
    repairs = [
        p for p in script.prompts[AgentMode.MANAGE] if p.startswith("上一份 auditor")
    ]
    assert len(repairs) == 1
    assert "禁止 not_applicable" in repairs[0]


@pytest.mark.parametrize("repair", [report(step="not_applicable"), report()])
async def test_scope_repair_cannot_erase_blocking_findings(tmp_path, repair):
    script = Script(
        manager=[manager(audit_only("s1")), manager("下一步: 阻塞")],
        auditor=[report(step="not_applicable", blocking="unknown: 缺少真实读取结果")],
        repair=repair,
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    await runner.run(mea_id)
    rnd = (await env.store.rounds(mea_id))[0]
    assert rnd.step_acceptance == "not_satisfied"
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.TODO


async def test_audit_only_receives_original_executor_run_records(tmp_path):
    received = []

    async def evidence(run_id, conversation_id, mea_id):
        received.append((run_id, conversation_id, mea_id))
        return (
            '{"run_id":"' + run_id + '","final_reply":"v2","call_list_complete":true}'
        )

    def recheck(prompt):
        assert '"final_reply":"v2"' in prompt
        assert "来自持久化 Trace" in prompt
        assert "不得据此证明未执行" in prompt
        return report()

    script = Script(
        manager=[
            manager(execute("s1")),
            manager(audit_only("s1")),
            manager("下一步: 阻塞"),
        ],
        executor=["v2"],
        auditor=[report(status="blocked", step="not_satisfied"), recheck],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner(executor_evidence=evidence)
    mea_id = await _start(env, runner)
    await runner.run(mea_id)
    rounds = await env.store.rounds(mea_id)
    assert received == [(rounds[0].executor_run_id, "conv-1", mea_id)] * 2
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.DONE
    assert env.runs.executor_starts() == 1


@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_final_audit_requires_final_scope_header(tmp_path, repair_succeeds):
    script = Script(
        manager=[
            manager(execute("s1")),
            manager(execute("s2")),
            manager(FINAL),
            manager("下一步: 阻塞"),
        ],
        executor=["done", "done"],
        auditor=[report(), report(), report()],
        repair=report(step="not_applicable") if repair_succeeds else report(),
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    mea = await runner.run(mea_id)
    assert mea.status is (MeaStatus.COMPLETED if repair_succeeds else MeaStatus.BLOCKED)


async def test_audit_keeps_referenced_original_run_when_retries_exceed_limit(tmp_path):
    from app.runtime.mea.models import MeaRound, now_utc

    received = []

    async def evidence(run_id, conversation_id, mea_id):
        received.append(run_id)
        return '{"run_id":"' + run_id + '"}'

    env = await _env(tmp_path, Script())
    runner = env.runner(executor_evidence=evidence)
    mea_id = await _start(env, runner)
    mea = await env.store.require(mea_id)
    task = await env.tasks.get(mea.task_id)
    requirements = await env.store.requirements(mea_id)
    now = now_utc()
    rounds = [
        MeaRound(
            mea_run_id=mea_id,
            index=i,
            step_id="s1",
            executor_run_id=f"exec-{i}",
            created_at=now,
            updated_at=now,
        )
        for i in range(1, 5)
    ]
    current = MeaRound(
        mea_run_id=mea_id,
        index=5,
        step_id="s1",
        kind=RoundKind.AUDIT_ONLY,
        related_refs=("round_001",),
        created_at=now,
        updated_at=now,
    )
    prompt = await runner._auditor_prompt(mea, current, task, requirements, rounds)
    assert received == ["exec-1", "exec-3", "exec-4"]
    assert "不能代表本步骤或整个任务的全部历史" in prompt


# 函数说明：test_once_note_reaches_only_the_next_manager
# 用途：回归验证回归测试与测试辅助中的 `once_note_reaches_only_the_next_manager` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `report` → `_env` → `env.runner`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`'- 这一轮先核查 s1' in first`。
#   验证条件：`'这一轮先核查 s1' not in second`。
async def test_once_note_reaches_only_the_next_manager(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(audit_only("s1")), manager("下一步: 阻塞")],
        auditor=[report()],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    await runner.add_once_note(mea_id, "这一轮先核查 s1")

    await runner.run(mea_id)

    first, second = script.prompts[AgentMode.MANAGE][:2]
    assert "- 这一轮先核查 s1" in first
    assert "这一轮先核查 s1" not in second


# 函数说明：test_amendment_kind_answer_only_resumes_waiting
# 用途：回归验证回归测试与测试辅助中的 `amendment_kind_answer_only_resumes_waiting` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_env` → `Script` → `env.runner` →
# `_start` → `runner.add_amendment` → `env.store.require`。
# 分支与异常：
#   验证条件：
# `result.accepted and (await env.store.require(mea_id)).status is MeaStatus.RUNNING`。
async def test_amendment_kind_answer_only_resumes_waiting(tmp_path: Path) -> None:
    env = await _env(tmp_path, Script())
    runner = env.runner()
    mea_id = await _start(env, runner)

    result = await runner.add_amendment(mea_id, "补充", kind=AmendmentKind.NOTE)

    assert result.accepted and (await env.store.require(mea_id)).status is MeaStatus.RUNNING


# ---------------------------------------------------------------------------
# 超时、取消与会话级操作（第 5 步）
# ---------------------------------------------------------------------------


# 函数说明：_hang
# 用途：在回归测试与测试辅助中处理 `_hang`，通过 `asyncio.sleep` 完成首个内部处理步骤。
# 参数：
#   prompt：本次调用使用的提示文本，类型 `str`。
# 返回：类型 `str`；返回 `'never'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.sleep`。
async def _hang(prompt: str) -> str:
    await asyncio.sleep(3600)
    return "never"


# 函数说明：test_executor_timeout_interrupts_round_and_recovers
# 用途：回归验证回归测试与测试辅助中的 `executor_timeout_interrupts_round_and_recovers`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`'强制停止' in (rounds[0].interrupt_reason or '')`。
#   验证条件：`rounds[1].kind is RoundKind.RECOVERY_AUDIT`。
#   验证条件：`env.runs.executor_starts() == 1`。
#   验证条件：`env.table[rounds[0].executor_run_id].status is RunStatus.CANCELLED`。
async def test_executor_timeout_interrupts_round_and_recovers(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=[_hang],
        auditor=[report(step="not_satisfied")],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner(timeouts={AgentMode.EXECUTE: 0.05})
    mea_id = await _start(env, runner)

    mea = await asyncio.wait_for(runner.run(mea_id), 10)

    rounds = await env.store.rounds(mea_id)
    assert "强制停止" in (rounds[0].interrupt_reason or "")
    assert rounds[1].kind is RoundKind.RECOVERY_AUDIT
    assert env.runs.executor_starts() == 1
    assert env.table[rounds[0].executor_run_id].status is RunStatus.CANCELLED
    assert mea.status is MeaStatus.BLOCKED


# 函数说明：test_auditor_timeout_fails_the_audit_without_repair
# 用途：回归验证回归测试与测试辅助中的 `auditor_timeout_fails_the_audit_without_repair`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `report` → `_env` → `env.runner`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`rnd.audit_status == 'blocked' and rnd.integrity_status == 'suspect'`。
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.TODO`。
#   验证条件：`repairs == []`。
async def test_auditor_timeout_fails_the_audit_without_repair(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(audit_only("s1")), manager("下一步: 阻塞")],
        auditor=[_hang],
        repair=report(),  # 不应被调用
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner(timeouts={AgentMode.AUDIT: 0.05})
    mea_id = await _start(env, runner)

    await asyncio.wait_for(runner.run(mea_id), 10)

    rnd = (await env.store.rounds(mea_id))[0]
    assert rnd.audit_status == "blocked" and rnd.integrity_status == "suspect"
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.TODO
    repairs = [p for p in script.prompts[AgentMode.MANAGE] if p.startswith("上一份 auditor")]
    assert repairs == []


# 函数说明：test_cancel_stops_the_loop_without_starting_new_children
# 用途：回归验证回归测试与测试辅助中的
# `cancel_stops_the_loop_without_starting_new_children` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Event` → `Script` → `_env` →
# `env.runner` → `_start` → `runner.spawn`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.CANCELLED`。
#   验证条件：`len(env.runs.started) == 1`。
#   验证条件：`(await env.store.rounds(mea_id))[0].phase is RoundPhase.INTERRUPTED`。
async def test_cancel_stops_the_loop_without_starting_new_children(tmp_path: Path) -> None:
    started = asyncio.Event()

    # 函数说明：test_cancel_stops_the_loop_without_starting_new_children.slow_manager
    # 用途：在回归测试与测试辅助中处理 `slow_manager`，通过 `started.set` 完成首个内部处
    # 理步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `'never'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`started.set` → `asyncio.sleep`。
    # 闭包依赖：从外层读取 `started`。
    async def slow_manager(prompt: str) -> str:
        started.set()
        await asyncio.sleep(3600)
        return "never"

    script = Script(manager=[slow_manager])
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    loop = runner.spawn(mea_id)
    await asyncio.wait_for(started.wait(), 5)

    await runner.cancel(mea_id)
    mea = await asyncio.wait_for(loop, 10)

    assert mea.status is MeaStatus.CANCELLED
    assert len(env.runs.started) == 1
    assert (await env.store.rounds(mea_id))[0].phase is RoundPhase.INTERRUPTED


# 函数说明：test_busy_run_and_cancel_for_conversation
# 用途：回归验证回归测试与测试辅助中的 `busy_run_and_cancel_for_conversation` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Event` → `Script` → `manager`
#  → `execute` → `_env` → `env.runner`；另有 9 个调用点。
# 分支与异常：
#   验证条件：`(await runner.busy_run('conv-1')).id == mea_id`。
#   验证条件：`await runner.busy_run('conv-2') is None`。
#   验证条件：`await runner.cancel_for_conversation('conv-1') == 1`。
#   验证条件：`(await env.store.require(mea_id)).status is MeaStatus.CANCELLED`。
async def test_busy_run_and_cancel_for_conversation(tmp_path: Path) -> None:
    started = asyncio.Event()

    # 函数说明：test_busy_run_and_cancel_for_conversation.slow_executor
    # 用途：在回归测试与测试辅助中处理 `slow_executor`，通过 `started.set` 完成首个内部
    # 处理步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `'never'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`started.set` → `asyncio.sleep`。
    # 闭包依赖：从外层读取 `started`。
    async def slow_executor(prompt: str) -> str:
        started.set()
        await asyncio.sleep(3600)
        return "never"

    script = Script(manager=[manager(execute("s1"))], executor=[slow_executor])
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    assert (await runner.busy_run("conv-1")).id == mea_id
    assert await runner.busy_run("conv-2") is None
    runner.spawn(mea_id)
    await asyncio.wait_for(started.wait(), 5)

    assert await runner.cancel_for_conversation("conv-1") == 1

    assert (await env.store.require(mea_id)).status is MeaStatus.CANCELLED
    assert await runner.busy_run("conv-1") is None
    assert env.runs.active_run_ids == ()
    assert await env.store.delete_for_conversation("conv-1") == 1
    assert await env.store.get(mea_id) is None and await env.store.rounds(mea_id) == ()


# 函数说明：test_paused_mea_does_not_block_the_conversation
# 用途：回归验证回归测试与测试辅助中的 `paused_mea_does_not_block_the_conversation` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_env` → `Script` → `env.runner` →
# `_start` → `runner.pause` → `env.store.require`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`(await env.store.require(mea_id)).pause_requested`。
#   验证条件：`mea.status is MeaStatus.PAUSED`。
#   验证条件：`await runner.busy_run('conv-1') is None`。
#   验证条件：`[run.id for run in await runner.open_runs('conv-1')] == [mea_id]`。
async def test_paused_mea_does_not_block_the_conversation(tmp_path: Path) -> None:
    env = await _env(tmp_path, Script())
    runner = env.runner()
    mea_id = await _start(env, runner)
    await runner.pause(mea_id)
    assert (await env.store.require(mea_id)).pause_requested
    mea = await runner.run(mea_id)  # 还没开轮就暂停
    assert mea.status is MeaStatus.PAUSED
    assert await runner.busy_run("conv-1") is None
    assert [run.id for run in await runner.open_runs("conv-1")] == [mea_id]


# 函数说明：test_once_note_with_interrupt_child_cancels_executor
# 用途：回归验证回归测试与测试辅助中的 `once_note_with_interrupt_child_cancels_executor`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Event` → `Script` → `manager`
#  → `execute` → `report` → `_env`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`await runner.interrupt_child(mea_id) is True`。
#   验证条件：`rounds[0].interrupt_reason == 'Executor 被取消'`。
#   验证条件：`rounds[1].kind is RoundKind.RECOVERY_AUDIT`。
#   验证条件：`'先停一下，换个思路' in script.prompts[AgentMode.MANAGE][1]`。
async def test_once_note_with_interrupt_child_cancels_executor(tmp_path: Path) -> None:
    started = asyncio.Event()

    # 函数说明：test_once_note_with_interrupt_child_cancels_executor.slow_executor
    # 用途：在回归测试与测试辅助中处理 `slow_executor`，通过 `started.set` 完成首个内部
    # 处理步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `'never'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`started.set` → `asyncio.sleep`。
    # 闭包依赖：从外层读取 `started`。
    async def slow_executor(prompt: str) -> str:
        started.set()
        await asyncio.sleep(3600)
        return "never"

    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=[slow_executor],
        auditor=[report(step="not_satisfied")],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    loop = runner.spawn(mea_id)
    await asyncio.wait_for(started.wait(), 5)

    await runner.add_once_note(mea_id, "先停一下，换个思路")
    assert await runner.interrupt_child(mea_id) is True
    await asyncio.wait_for(loop, 10)

    rounds = await env.store.rounds(mea_id)
    assert rounds[0].interrupt_reason == "Executor 被取消"
    assert rounds[1].kind is RoundKind.RECOVERY_AUDIT
    assert "先停一下，换个思路" in script.prompts[AgentMode.MANAGE][1]


# 函数说明：test_resume_from_waiting_clears_the_question
# 用途：回归验证回归测试与测试辅助中的 `resume_from_waiting_clears_the_question` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `_env` →
# `env.runner` → `_start` → `runner.run`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.WAITING_USER and mea.pending_question`。
#   验证条件：`mea.pending_question is None and mea.status is MeaStatus.RUNNING`。
#   验证条件：`(await env.store.require(mea_id)).status is MeaStatus.BLOCKED`。
async def test_resume_from_waiting_clears_the_question(tmp_path: Path) -> None:
    script = Script(
        manager=[manager("下一步: 请示用户\n问题: 用哪个库？\n选项: A | B"), manager("下一步: 阻塞")],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    mea = await runner.run(mea_id)
    assert mea.status is MeaStatus.WAITING_USER and mea.pending_question

    mea = await runner.resume(mea_id)
    assert mea.pending_question is None and mea.status is MeaStatus.RUNNING
    await runner.wait(mea_id)
    assert (await env.store.require(mea_id)).status is MeaStatus.BLOCKED


# 函数说明：test_child_runs_are_tagged_with_their_role
# 用途：回归验证回归测试与测试辅助中的 `child_runs_are_tagged_with_their_role` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`[source for source, _ in sources][:3] == ['mea:manager', 'mea:executor',
# 'mea:auditor']`。
#   验证条件：`{source_id for _, source_id in sources} == {mea_id}`。
# 副作用与资源：
#   更新对象字段：`env.runs.start`。
async def test_child_runs_are_tagged_with_their_role(tmp_path: Path) -> None:
    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=["done"],
        auditor=[report()],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    sources: list[tuple[str, str]] = []
    original = env.runs.start

    # 函数说明：test_child_runs_are_tagged_with_their_role.recording_start
    # 用途：启动`recording`，供回归测试与测试辅助使用。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：返回 `await original(prompt, **kwargs)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`original`。
    # 闭包依赖：从外层读取 `original`、`sources`。
    async def recording_start(prompt: str, **kwargs):
        sources.append((kwargs["source"], kwargs["source_id"]))
        return await original(prompt, **kwargs)

    env.runs.start = recording_start
    mea_id = await _start(env, runner)
    await runner.run(mea_id)

    assert [source for source, _ in sources][:3] == ["mea:manager", "mea:executor", "mea:auditor"]
    assert {source_id for _, source_id in sources} == {mea_id}


# 函数说明：test_shutdown_while_executor_runs_writes_nothing
# 用途：回归验证回归测试与测试辅助中的 `shutdown_while_executor_runs_writes_nothing` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Event` → `Script` → `manager`
#  → `execute` → `_env` → `env.runner`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`rnd.phase is RoundPhase.EXECUTING and rnd.executor_output is None`。
#   验证条件：`(await env.store.require(mea_id)).status is MeaStatus.RUNNING`。
#   验证条件：`env.runs.active_run_ids`。
async def test_shutdown_while_executor_runs_writes_nothing(tmp_path: Path) -> None:
    started = asyncio.Event()

    # 函数说明：test_shutdown_while_executor_runs_writes_nothing.slow_executor
    # 用途：在回归测试与测试辅助中处理 `slow_executor`，通过 `started.set` 完成首个内部
    # 处理步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `'never'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`started.set` → `asyncio.sleep`。
    # 闭包依赖：从外层读取 `started`。
    async def slow_executor(prompt: str) -> str:
        started.set()
        await asyncio.sleep(3600)
        return "never"

    script = Script(manager=[manager(execute("s1"))], executor=[slow_executor])
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    runner.spawn(mea_id)
    await asyncio.wait_for(started.wait(), 5)

    await runner.shutdown()

    rnd = (await env.store.rounds(mea_id))[0]
    assert rnd.phase is RoundPhase.EXECUTING and rnd.executor_output is None
    assert (await env.store.require(mea_id)).status is MeaStatus.RUNNING  # 下次启动对账改为 paused
    assert env.runs.active_run_ids  # 子 Run 本身没有被当成结束
    for handle in list(env.runs._tasks.values()):
        handle.cancel()


# 函数说明：test_time_waiting_for_approval_does_not_count_toward_timeout
# 用途：回归验证回归测试与测试辅助中的
# `time_waiting_for_approval_does_not_count_toward_timeout` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Event` → `Script` → `manager`
#  → `execute` → `report` → `_env`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`rnd.interrupt_reason is None and rnd.executor_output == '等审批后完成'`。
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.DONE`。
# 副作用与资源：
#   更新对象字段：`env.runs.awaiting_approval`。
async def test_time_waiting_for_approval_does_not_count_toward_timeout(tmp_path: Path) -> None:
    release = asyncio.Event()

    # 函数说明：
    # test_time_waiting_for_approval_does_not_count_toward_timeout.approval_executor
    # 用途：在回归测试与测试辅助中处理 `approval_executor`，通过 `release.wait` 完成首个
    # 内部处理步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `'等审批后完成'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`release.wait`。
    # 闭包依赖：从外层读取 `release`。
    async def approval_executor(prompt: str) -> str:
        await release.wait()
        return "等审批后完成"

    script = Script(
        manager=[manager(execute("s1")), manager("下一步: 阻塞")],
        executor=[approval_executor],
        auditor=[report()],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner(timeouts={AgentMode.EXECUTE: 0.05})
    env.runs.awaiting_approval = lambda run_id: not release.is_set()  # 一直在等人工审批
    mea_id = await _start(env, runner)
    loop = runner.spawn(mea_id)

    await asyncio.sleep(0.3)  # 远超 0.05 秒时限
    release.set()
    await asyncio.wait_for(loop, 10)

    rnd = (await env.store.rounds(mea_id))[0]
    assert rnd.interrupt_reason is None and rnd.executor_output == "等审批后完成"
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.DONE


# 函数说明：test_markdown_no_blocker_can_complete
# 用途：回归验证回归测试与测试辅助中的 `markdown_no_blocker_can_complete` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report().replace` → `report` → `_env`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`(await runner.run(mea_id)).status is MeaStatus.COMPLETED`。
async def test_markdown_no_blocker_can_complete(tmp_path):
    script = Script(
        manager=[manager(execute("s1")), manager(execute("s2")), manager(FINAL)],
        executor=["done", "done"],
        auditor=[report().replace("阻断约束: 无", "**阻断约束:** 无"), report(), report(step="not_applicable")],
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    assert (await runner.run(mea_id)).status is MeaStatus.COMPLETED


# 函数说明：test_truncated_role_cannot_commit_success
# 用途：回归验证回归测试与测试辅助中的 `truncated_role_cannot_commit_success` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   role：规划、执行、审计等运行角色。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`mea.status in (MeaStatus.PAUSED, MeaStatus.WAITING_USER)`。
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.TODO`。
#   验证条件：`env.runs.executor_starts() == 0`。
#   验证条件：`rounds[0].step_acceptance != 'satisfied'`。
# 副作用与资源：
#   更新对象字段：`env.runs.result`。
@pytest.mark.parametrize(
    ("role", "truncation_field"),
    [
        (AgentMode.MANAGE, "model_finish_reason"),
        (AgentMode.AUDIT, "model_finish_reason"),
        (AgentMode.AUDIT, "unresolved_output_truncation"),
    ],
)
async def test_truncated_role_cannot_commit_success(tmp_path, role, truncation_field):
    script = Script(
        manager=[manager(execute("s1"))] * 3,
        executor=["done"] * 3,
        auditor=[report()] * 3,
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    original = env.runs.result

    # 函数说明：test_truncated_role_cannot_commit_success.truncated_result
    # 用途：在回归测试与测试辅助中处理 `truncated_result`，通过 `result.model_copy` 完成
    # 首个内部处理步骤。
    # 参数：
    #   run_id：目标运行标识。
    # 返回：按分支返回 `result.model_copy(update={'model_finish_reason': 'max_tokens'})`
    # ；`result`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`original` → `next`。
    # 分支与异常：
    #   当 `mode is role and result is not None` 时，返回 `result.model_copy(…)`。
    # 闭包依赖：从外层读取 `env`、`original`、`role`。
    def truncated_result(run_id):
        result = original(run_id)
        mode = next(mode for mode, rid in env.runs.started if rid == run_id)
        if mode is role and result is not None:
            update = (
                {"model_finish_reason": "end_turn", truncation_field: True}
                if truncation_field == "unresolved_output_truncation"
                else {truncation_field: "max_tokens"}
            )
            return result.model_copy(update=update)
        return result

    env.runs.result = truncated_result
    if role is AgentMode.AUDIT:
        mea = await env.store.require(mea_id)
        await env.store.save_run(mea.model_copy(update={"round_budget": 3}))
    mea = await runner.run(mea_id)
    assert mea.status in (MeaStatus.PAUSED, MeaStatus.WAITING_USER)
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.TODO
    if role is AgentMode.MANAGE:
        assert env.runs.executor_starts() == 0
    else:
        rounds = await env.store.rounds(mea_id)
        assert rounds[0].audit_output_truncated is True
        assert rounds[0].step_acceptance != "satisfied"
        assert "截断" in rounds[0].auditor_report


# 函数说明：test_scattered_invalid_plans_have_total_limit
# 用途：回归验证回归测试与测试辅助中的 `scattered_invalid_plans_have_total_limit` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.WAITING_USER`。
#   验证条件：`'累计 6' in mea.pending_question`。
#   验证条件：`env.runs.executor_starts() == 2`。
async def test_scattered_invalid_plans_have_total_limit(tmp_path):
    script = Script(
        manager=["invalid", "invalid", manager(execute("s1")),
                 "invalid", "invalid", manager(execute("s1")),
                 "invalid", "invalid"],
        executor=["unfinished"] * 2,
        auditor=[report(status="incomplete", step="not_satisfied")] * 2,
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    mea = await runner.run(mea_id)
    assert mea.status is MeaStatus.WAITING_USER
    assert "累计 6" in mea.pending_question
    assert env.runs.executor_starts() == 2


# ---------------------------------------------------------------------------
# 没有进展检测
# ---------------------------------------------------------------------------


# 函数说明：test_stall_when_workspace_never_changes
# 用途：回归验证回归测试与测试辅助中的 `stall_when_workspace_never_changes` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `audit_only` →
# `report` → `_env` → `env.runner`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.WAITING_USER`。
#   验证条件：`'工作区完全没有变化' in (mea.pending_question or '')`。
#   验证条件：`'round_001、round_002、round_003' in (mea.pending_question or '')`。
#   验证条件：`len(await env.store.rounds(mea_id)) == 3`。
async def test_stall_when_workspace_never_changes(tmp_path: Path) -> None:
    # 审计反复不通过、工作区一点没变（例如审计输出被截断后原地重试）
    script = Script(
        manager=[manager(audit_only("s1"))] * 6,
        auditor=[report(status="incomplete", step="not_satisfied")] * 6,
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    mea = await runner.run(mea_id)

    assert mea.status is MeaStatus.WAITING_USER
    assert "工作区完全没有变化" in (mea.pending_question or "")
    assert "round_001、round_002、round_003" in (mea.pending_question or "")
    assert len(await env.store.rounds(mea_id)) == 3

    # 你点继续之后重新计数：再给 3 轮机会，而不是下一轮立刻又停
    await runner.resume(mea_id)
    mea = await runner.wait(mea_id)
    assert mea.status is MeaStatus.WAITING_USER
    assert len(await env.store.rounds(mea_id)) == 6


# 函数说明：test_stall_after_many_attempts_even_if_workspace_changes
# 用途：回归验证回归测试与测试辅助中的
# `stall_after_many_attempts_even_if_workspace_changes` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Script` → `manager` → `execute` →
# `report` → `_env` → `env.runner`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.WAITING_USER`。
#   验证条件：`'已连续 6 轮没有通过审计' in (mea.pending_question or '')`。
#   验证条件：`counter['n'] == 6`。
async def test_stall_after_many_attempts_even_if_workspace_changes(tmp_path: Path) -> None:
    counter = {"n": 0}

    # 函数说明：test_stall_after_many_attempts_even_if_workspace_changes.busy_executor
    # 用途：在回归测试与测试辅助中处理 `busy_executor`，通过
    # `(tmp_path / 'workspace' / f"try_{counter['n']}.txt").write_text` 完成首个内部处理
    # 步骤。
    # 参数：
    #   prompt：本次调用使用的提示文本，类型 `str`。
    # 返回：类型 `str`；返回 `'又改了一版'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `(tmp_path / 'workspace' / f"try_{counter['n']}.txt").write_text`。
    # 副作用与资源：
    #   文件或资源访问：
    # `(tmp_path / 'workspace' / f"try_{counter['n']}.txt").write_text`。
    # 闭包依赖：从外层读取 `counter`、`tmp_path`。
    def busy_executor(prompt: str) -> str:
        counter["n"] += 1
        (tmp_path / "workspace" / f"try_{counter['n']}.txt").write_text("x", encoding="utf-8")
        return "又改了一版"

    script = Script(
        manager=[manager(execute("s1"))] * 7,
        executor=[busy_executor] * 7,
        auditor=[report(status="incomplete", step="not_satisfied")] * 7,
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    mea = await runner.run(mea_id)

    assert mea.status is MeaStatus.WAITING_USER
    assert "已连续 6 轮没有通过审计" in (mea.pending_question or "")
    assert counter["n"] == 6


# 函数说明：test_passing_a_step_resets_the_stall_count
# 用途：回归验证回归测试与测试辅助中的 `passing_a_step_resets_the_stall_count` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`report` → `Script` → `manager` →
# `audit_only` → `_env` → `env.runner`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.BLOCKED`。
#   验证条件：`(await _steps(env, mea_id))['s1'] is TaskStepStatus.DONE`。
async def test_passing_a_step_resets_the_stall_count(tmp_path: Path) -> None:
    failing = report(status="incomplete", step="not_satisfied")
    script = Script(
        manager=[manager(audit_only("s1")), manager(audit_only("s1")), manager(audit_only("s1")),
                 manager(audit_only("s2")), manager(audit_only("s2")), manager("下一步: 阻塞")],
        auditor=[failing, failing, report(), failing, failing],
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)

    mea = await runner.run(mea_id)

    assert mea.status is MeaStatus.BLOCKED  # 两次失败 + 通过 + 两次失败，都没到 3 轮
    assert (await _steps(env, mea_id))["s1"] is TaskStepStatus.DONE


# 函数说明：test_amendment_resets_the_stall_guard
# 用途：回归验证回归测试与测试辅助中的 `amendment_resets_the_stall_guard` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`report` → `Script` → `manager` →
# `audit_only` → `_env` → `env.runner`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`(await runner.run(mea_id)).status is MeaStatus.WAITING_USER`。
#   验证条件：`result.accepted`。
#   验证条件：`(await env.store.require(mea_id)).stall_guard_after == 3`。
#   验证条件：`mea.status is MeaStatus.BLOCKED`。
async def test_amendment_resets_the_stall_guard(tmp_path: Path) -> None:
    failing = report(status="incomplete", step="not_satisfied")
    script = Script(
        manager=[manager(audit_only("s1"))] * 4 + [manager("下一步: 阻塞")],
        auditor=[failing] * 4,
        final="停",
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    assert (await runner.run(mea_id)).status is MeaStatus.WAITING_USER

    result = await runner.answer(mea_id, "表头允许有 BOM，按 utf-8-sig 读取")
    assert result.accepted
    mea = await runner.wait(mea_id)

    assert (await env.store.require(mea_id)).stall_guard_after == 3
    assert mea.status is MeaStatus.BLOCKED


@pytest.mark.parametrize("persistent", [False, True])
async def test_verdict_replace_failure_is_retried_or_reported(
    tmp_path: Path, monkeypatch, persistent: bool,
) -> None:
    import json

    from app.domain.task import store as task_store

    script = Script(
        manager=[manager(execute("s1")), manager(execute("s2")), manager(FINAL)],
        executor=["read", "imported"],
        auditor=[report(), report(), report(step="not_applicable")],
    )
    env = await _env(tmp_path, script)
    runner = env.runner()
    mea_id = await _start(env, runner)
    replace = task_store.os.replace
    attempts = 0

    def replace_verdict(source, target):
        nonlocal attempts
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        if f"{mea_id}/r002/verdict" in payload["applied_ops"]:
            attempts += 1
            if persistent or attempts == 1:
                error = PermissionError("task file is busy")
                error.winerror = 32
                raise error
        replace(source, target)

    monkeypatch.setattr(task_store.os, "replace", replace_verdict)
    monkeypatch.setattr(task_store.time, "sleep", lambda _: None)
    result = await runner._run_logged(mea_id)
    saved = await env.store.require(mea_id)
    task = await env.tasks.get(result.task_id)
    assert script.count(AgentMode.EXECUTE) == 2
    if persistent:
        assert attempts == 4
        assert saved.status is MeaStatus.FAILED
        assert "PermissionError: task file is busy" in saved.abort_reason
        assert task.steps[-1].status is TaskStepStatus.IN_PROGRESS
        last = (await env.store.rounds(mea_id))[-1]
        assert last.phase is RoundPhase.AUDITED
        assert last.verdict_patch is not None
    else:
        assert saved.status is MeaStatus.COMPLETED
        assert task.status is TaskStatus.COMPLETED
        assert saved.abort_reason is None
