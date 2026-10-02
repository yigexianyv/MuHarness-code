
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.paths import runtime_data_path

from .models import (
    CLOSED_STEP_STATUSES,
    MAX_APPLIED_OPS,
    TASK_ID_LENGTH,
    Task,
    TaskPatch,
    TaskPriority,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
)
from .ops import OpIdReusedError, OpOutcome, OpPrecondition, OpResult, TaskOp

DEFAULT_TASKS_DIR = runtime_data_path("tasks")
MAX_TASK_FILE_BYTES = 1_000_000
logger = logging.getLogger("muharness.task.store")

_TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)
_CONTEXT_TASK_STATUSES = frozenset({TaskStatus.ACTIVE, TaskStatus.PAUSED})
_TASK_ID_RE = re.compile(rf"^[0-9a-f]{{{TASK_ID_LENGTH}}}$")
_TASK_PREFIX_RE = re.compile(rf"^[0-9a-f]{{4,{TASK_ID_LENGTH}}}$")


class LegacyTaskOwnershipError(ValueError):
    pass


class FileTaskStore:

    # 函数说明：FileTaskStore.__init__
    # 用途：初始化 FileTaskStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   tasks_dir：相关数据的根目录或保存目录，类型 `str | Path`；默认
    # `DEFAULT_TASKS_DIR`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(tasks_dir).expanduser().resolve` → `Path(tasks_dir).expanduser` → `Path`。
    # 副作用与资源：
    #   更新对象字段：`self.tasks_dir`、`self._locks`。
    def __init__(self, tasks_dir: str | Path = DEFAULT_TASKS_DIR) -> None:
        self.tasks_dir = Path(tasks_dir).expanduser().resolve()
        self._locks: dict[str, asyncio.Lock] = {}

    # 函数说明：FileTaskStore.initialize
    # 用途：初始化FileTaskStore，供任务状态与步骤管理使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def initialize(self) -> None:

        await asyncio.to_thread(
            self.tasks_dir.mkdir,
            parents=True,
            exist_ok=True,
        )

    # 函数说明：FileTaskStore.create
    # 用途：创建任务文件并建立初始步骤与会话关联。
    # 参数：
    #   title：面向用户的标题，类型 `str`。
    #   description：补充描述，类型 `str | None`；默认 `None`。
    #   goal：`goal`输入或配置值，类型 `str | None`；默认 `None`。
    #   priority：`priority`输入或配置值，类型 `TaskPriority`；默认
    # `TaskPriority.NORMAL`。
    #   steps：任务步骤集合，类型 `Sequence[TaskStep]`；默认 `()`。
    #   owner_conversation_id：记录所属会话标识，类型 `str`。
    #   run_ids：待处理的运行标识集合，类型 `Sequence[str]`；默认 `()`。
    # 返回：类型 `Task`；返回 `task`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` → `Task` → `uuid4`
    # → `_normalize_required_entry` → `_merge_entries` → `self._write`。
    async def create(
        self,
        *,
        title: str,
        description: str | None = None,
        goal: str | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        steps: Sequence[TaskStep] = (),
        owner_conversation_id: str,
        run_ids: Sequence[str] = (),
    ) -> Task:

        """创建任务文件并建立初始步骤与会话关联。"""
        now = datetime.now(UTC)
        task = Task(
            id=uuid4().hex,
            title=title,
            description=description,
            goal=goal,
            status=TaskStatus.PENDING,
            priority=priority,
            steps=tuple(steps),
            owner_conversation_id=_normalize_required_entry(
                owner_conversation_id,
                field_name="owner_conversation_id",
            ),
            run_ids=_merge_entries((), run_ids),
            created_at=now,
            updated_at=now,
        )
        await self._write(task)
        return task

    # 函数说明：FileTaskStore.get
    # 用途：获取FileTaskStore，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    # 返回：类型 `Task | None`；按分支返回 `None`；
    # `await asyncio.to_thread(_read_task, path)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_validate_task_id` → `self._path`
    #  → `asyncio.to_thread`。
    # 分支与异常：
    #   当 `not await asyncio.to_thread(path.is_file)` 时，返回 `None`。
    #   当 `await asyncio.to_thread(path.is_symlink)` 时，返回 `None`。
    #   捕获 `LegacyTaskOwnershipError` 后，返回 `None`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def get(self, task_id: str) -> Task | None:

        normalized = _validate_task_id(task_id)
        path = self._path(normalized)
        if not await asyncio.to_thread(path.is_file):
            return None
        if await asyncio.to_thread(path.is_symlink):
            return None
        try:
            return await asyncio.to_thread(_read_task, path)
        except LegacyTaskOwnershipError:
            return None

    # 函数说明：FileTaskStore.resolve
    # 用途：按任务标识解析文件并加载当前状态。
    # 参数：
    #   identifier：待规范化的标识，类型 `str`。
    #   owner_conversation_id：记录所属会话标识，类型 `str | None`；默认 `None`。
    # 返回：类型 `Task | None`；按分支返回 `None`；`exact`；
    # `matches[0] if matches else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`identifier.strip().lower` →
    # `_validate_task_prefix` → `_normalize_required_entry` → `self._all_tasks` →
    # `task.id.startswith`。
    # 分支与异常：
    #   当 `not normalized` 时，返回 `None`。
    #   `len(normalized) == TASK_ID_LENGTH` 分支在完成前置处理后返回 `None`。
    #   当 `exact is not None and (owner is None or…` 时，返回 `exact`。
    #   当 `len(matches) > 1` 时，抛出 `ValueError(f'任务 ID 前缀不唯一：{identifier}')`
    # 。
    async def resolve(
        self,
        identifier: str,
        *,
        owner_conversation_id: str | None = None,
    ) -> Task | None:

        """按任务标识解析文件并加载当前状态。"""
        normalized = identifier.strip().lower()
        if not normalized:
            return None

        _validate_task_prefix(normalized)

        owner = (
            _normalize_required_entry(
                owner_conversation_id,
                field_name="owner_conversation_id",
            )
            if owner_conversation_id is not None
            else None
        )
        if len(normalized) == TASK_ID_LENGTH:
            exact = await self.get(normalized)
            if exact is not None and (
                owner is None or exact.owner_conversation_id == owner
            ):
                return exact
            return None

        matches = [
            task
            for task in await self._all_tasks()
            if task.id.startswith(normalized)
            and (owner is None or task.owner_conversation_id == owner)
        ]
        if len(matches) > 1:
            raise ValueError(f"任务 ID 前缀不唯一：{identifier}")
        return matches[0] if matches else None

    # 函数说明：FileTaskStore.list
    # 用途：列出FileTaskStore，供任务状态与步骤管理使用。
    # 参数：
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `50`。
    #   status：目标状态，类型 `TaskStatus | None`；默认 `None`。
    #   owner_conversation_id：记录所属会话标识，类型 `str | None`；默认 `None`。
    # 返回：类型 `tuple[Task, ...]`；返回 `tuple(tasks[:limit])`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._all_tasks` →
    # `_normalize_required_entry` → `tasks.sort`。
    # 分支与异常：
    #   当 `limit < 1` 时，抛出 `ValueError('limit must be at least 1')`。
    async def list(
        self,
        *,
        limit: int = 50,
        status: TaskStatus | None = None,
        owner_conversation_id: str | None = None,
    ) -> tuple[Task, ...]:

        if limit < 1:
            raise ValueError("limit must be at least 1")
        tasks = await self._all_tasks()
        if status is not None:
            tasks = [task for task in tasks if task.status is status]
        if owner_conversation_id is not None:
            normalized = _normalize_required_entry(
                owner_conversation_id,
                field_name="owner_conversation_id",
            )
            tasks = [
                task
                for task in tasks
                if normalized == task.owner_conversation_id
            ]
        tasks.sort(key=lambda task: task.updated_at, reverse=True)
        return tuple(tasks[:limit])

    # 函数说明：FileTaskStore.delete
    # 用途：删除FileTaskStore，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    # 返回：类型 `bool`；按分支返回 `False`；`True`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_validate_task_id` →
    # `self._lock_for` → `self._path` → `asyncio.to_thread`。
    # 资源/并发边界：`self._lock_for(normalized)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `not await asyncio.to_thread(path.is_file)` 时，返回 `False`。
    #   当 `await asyncio.to_thread(path.is_symlink)` 时，返回 `False`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def delete(self, task_id: str) -> bool:

        normalized = _validate_task_id(task_id)
        async with self._lock_for(normalized):
            path = self._path(normalized)
            if not await asyncio.to_thread(path.is_file):
                return False
            if await asyncio.to_thread(path.is_symlink):
                return False
            await asyncio.to_thread(path.unlink)
            return True

    # 函数说明：FileTaskStore.delete_for_conversation
    # 用途：删除会话，供任务状态与步骤管理使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `tuple[str, ...]`；返回 `tuple(deleted)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.list_for_conversation` →
    # `self.delete`。
    async def delete_for_conversation(
        self,
        conversation_id: str,
    ) -> tuple[str, ...]:

        tasks = await self.list_for_conversation(conversation_id)
        deleted: list[str] = []
        for task in tasks:
            if await self.delete(task.id):
                deleted.append(task.id)
        return tuple(deleted)

    # 函数说明：FileTaskStore.list_for_conversation
    # 用途：列出会话，供任务状态与步骤管理使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `tuple[Task, ...]`；返回 `tuple((task for task in await self._all_tasks
    # () if task.owner_conversation_id == owner))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_required_entry` →
    # `self._all_tasks`。
    async def list_for_conversation(
        self,
        conversation_id: str,
    ) -> tuple[Task, ...]:

        owner = _normalize_required_entry(
            conversation_id,
            field_name="conversation_id",
        )
        return tuple(
            task
            for task in await self._all_tasks()
            if task.owner_conversation_id == owner
        )

    # 函数说明：FileTaskStore.apply_patch
    # 用途：校验并原子应用任务变更，防止写入不一致状态。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   patch：传给 `_apply_patch` 的输入，类型 `TaskPatch`。
    #   owner_conversation_id：记录所属会话标识，类型 `str | None`；默认 `None`。
    # 返回：类型 `Task`；返回 `updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_validate_task_id` →
    # `self._lock_for` → `self._require` → `_normalize_required_entry` → `_apply_patch`
    # → `datetime.now`；另有 1 个调用点。
    # 资源/并发边界：`self._lock_for(normalized)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `not patch.has_changes` 时，抛出
    # `ValueError('task patch must contain at least one change')`。
    #   当 `task.owner_conversation_id != owner` 时，抛出
    # `KeyError(f'任务不存在：{task_id}')`。
    #   当 `patch.expected_revision is not None and…` 时，抛出 `ValueError(…)`。
    async def apply_patch(
        self,
        task_id: str,
        patch: TaskPatch,
        *,
        owner_conversation_id: str | None = None,
    ) -> Task:

        """校验并原子应用任务变更，防止写入不一致状态。"""
        normalized = _validate_task_id(task_id)
        if not patch.has_changes:
            raise ValueError("task patch must contain at least one change")
        async with self._lock_for(normalized):
            task = await self._require(normalized)
            if owner_conversation_id is not None:
                owner = _normalize_required_entry(
                    owner_conversation_id,
                    field_name="owner_conversation_id",
                )
                if task.owner_conversation_id != owner:
                    raise KeyError(f"任务不存在：{task_id}")
            if (
                patch.expected_revision is not None
                and patch.expected_revision != task.revision
            ):
                raise ValueError(
                    "task revision conflict: "
                    f"expected {patch.expected_revision}, current {task.revision}"
                )
            updated = _apply_patch(task, patch, datetime.now(UTC))
            await self._write(updated)
            return updated

    # 函数说明：FileTaskStore.apply_op
    # 用途：MEA 的幂等写入：同一操作 ID 只应用一次，前置条件不成立时返回 conflict。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   op：传给 `_apply_op` 的输入，类型 `TaskOp`。
    # 返回：类型 `OpOutcome`；按分支返回 `OpOutcome(OpResult.ALREADY_APPLIED, task)`；
    # `OpOutcome(OpResult.CONFLICT, task, reason)`；
    # `OpOutcome(OpResult.CONFLICT, task, f'{type(exc).__name__}: {exc}')`；
    # `OpOutcome(OpResult.APPLIED, updated)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_validate_task_id` → `op.digest`
    # → `self._lock_for` → `self._require` → `OpOutcome` → `_precondition_failure`；另有
    #  3 个调用点。
    # 资源/并发边界：`self._lock_for(normalized)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   `recorded is not None` 分支在完成前置处理后返回
    # `OpOutcome(OpResult.ALREADY_APPLIED, task)`。
    #   当 `recorded != digest` 时，抛出 `OpIdReusedError(op.op_id)`。
    #   当 `reason is not None` 时，返回 `OpOutcome(OpResult.CONFLICT, task, reason)`。
    #   捕获 `(KeyError, ValueError)` 后，返回
    # `OpOutcome(OpResult.CONFLICT, task, f'{type(exc).__name__}: {exc}')`。
    async def apply_op(self, task_id: str, op: TaskOp) -> OpOutcome:

        """MEA 的幂等写入：同一操作 ID 只应用一次，前置条件不成立时返回 conflict。"""
        normalized = _validate_task_id(task_id)
        digest = op.digest()
        async with self._lock_for(normalized):
            task = await self._require(normalized)
            recorded = task.applied_ops.get(op.op_id)
            if recorded is not None:
                if recorded != digest:
                    raise OpIdReusedError(op.op_id)
                return OpOutcome(OpResult.ALREADY_APPLIED, task)
            reason = _precondition_failure(task, op.precondition)
            if reason is not None:
                return OpOutcome(OpResult.CONFLICT, task, reason)
            try:
                updated = _apply_op(task, op, digest, datetime.now(UTC))
            except (KeyError, ValueError) as exc:
                return OpOutcome(OpResult.CONFLICT, task, f"{type(exc).__name__}: {exc}")
            await self._write(updated)
            return OpOutcome(OpResult.APPLIED, updated)

    # 函数说明：FileTaskStore.op_applied
    # 用途：在任务状态与步骤管理中处理 `op_applied`，通过 `self._require` 完成首个内部处
    # 理步骤。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   op_id：幂等操作标识，类型 `str`。
    # 返回：类型 `bool`；返回 `op_id in task.applied_ops`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._require` →
    # `_validate_task_id`。
    async def op_applied(self, task_id: str, op_id: str) -> bool:

        task = await self._require(_validate_task_id(task_id))
        return op_id in task.applied_ops

    # 函数说明：FileTaskStore.update_goal
    # 用途：更新`goal`，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   goal：`goal`输入或配置值，类型 `str | None`。
    # 返回：类型 `Task`；返回 `await self._update(task_id, TaskPatch(goal=goal))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update` → `TaskPatch`。
    async def update_goal(self, task_id: str, goal: str | None) -> Task:

        return await self._update(
            task_id,
            TaskPatch(goal=goal),
        )

    # 函数说明：FileTaskStore.update_state
    # 用途：更新状态，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   *state：额外位置参数，按实现向内部调用传递。
    # 返回：类型 `Task`；返回
    # `await self._update(task_id, TaskPatch(state=tuple(state)))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update` → `TaskPatch`。
    async def update_state(self, task_id: str, *state: str) -> Task:

        return await self._update(
            task_id,
            TaskPatch(state=tuple(state)),
        )

    # 函数说明：FileTaskStore.add_constraints
    # 用途：添加`constraints`，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   *constraints：额外位置参数，按实现向内部调用传递。
    # 返回：类型 `Task`；返回
    # `await self._update(task_id, TaskPatch(add_constraints=tuple(constraints)))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update` → `TaskPatch`。
    async def add_constraints(self, task_id: str, *constraints: str) -> Task:

        return await self._update(
            task_id,
            TaskPatch(add_constraints=tuple(constraints)),
        )

    # 函数说明：FileTaskStore.add_key_facts
    # 用途：添加键，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   *facts：额外位置参数，按实现向内部调用传递。
    # 返回：类型 `Task`；返回
    # `await self._update(task_id, TaskPatch(add_key_facts=tuple(facts)))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update` → `TaskPatch`。
    async def add_key_facts(self, task_id: str, *facts: str) -> Task:

        return await self._update(
            task_id,
            TaskPatch(add_key_facts=tuple(facts)),
        )

    # 函数说明：FileTaskStore.replace_steps
    # 用途：替换步骤集合，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   steps：任务步骤集合，类型 `Sequence[TaskStep]`。
    # 返回：类型 `Task`；返回
    # `await self._update(task_id, TaskPatch(replace_steps=tuple(steps)))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update` → `TaskPatch`。
    async def replace_steps(
        self,
        task_id: str,
        steps: Sequence[TaskStep],
    ) -> Task:

        return await self._update(
            task_id,
            TaskPatch(replace_steps=tuple(steps)),
        )

    # 函数说明：FileTaskStore.set_step_status
    # 用途：设置步骤状态，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   step_id：目标步骤标识，类型 `str`。
    #   status：目标状态，类型 `TaskStepStatus`。
    #   note：`note`输入或配置值，类型 `str | None`；默认 `None`。
    # 返回：类型 `Task`；返回 `await self._update(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update` → `TaskPatch`。
    async def set_step_status(
        self,
        task_id: str,
        step_id: str,
        status: TaskStepStatus,
        *,
        note: str | None = None,
    ) -> Task:

        return await self._update(
            task_id,
            TaskPatch(step_id=step_id, step_status=status, step_note=note),
        )

    # 函数说明：FileTaskStore.set_status
    # 用途：设置状态，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   status：目标状态，类型 `TaskStatus`。
    # 返回：类型 `Task`；返回 `await self._update(task_id, TaskPatch(status=status))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update` → `TaskPatch`。
    async def set_status(self, task_id: str, status: TaskStatus) -> Task:

        return await self._update(task_id, TaskPatch(status=status))

    # 函数说明：FileTaskStore.plan_accept
    # 用途：接受计划，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    # 返回：类型 `Task`；返回
    # `await self._transition_from_pending(task_id, TaskStatus.ACTIVE)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._transition_from_pending`。
    async def plan_accept(self, task_id: str) -> Task:

        return await self._transition_from_pending(
            task_id,
            TaskStatus.ACTIVE,
        )

    # 函数说明：FileTaskStore.plan_reject
    # 用途：拒绝计划，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    # 返回：类型 `Task`；返回
    # `await self._transition_from_pending(task_id, TaskStatus.CANCELLED)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._transition_from_pending`。
    async def plan_reject(self, task_id: str) -> Task:

        return await self._transition_from_pending(
            task_id,
            TaskStatus.CANCELLED,
        )

    # 函数说明：FileTaskStore._transition_from_pending
    # 用途：在任务状态与步骤管理中处理 `_transition_from_pending`，通过 `self._lock_for`
    #  完成首个内部处理步骤。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   status：目标状态，类型 `TaskStatus`。
    # 返回：类型 `Task`；返回 `updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_validate_task_id` →
    # `self._lock_for` → `self._require` → `datetime.now` → `Task.model_validate` →
    # `self._write`。
    # 资源/并发边界：`self._lock_for(normalized)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `task.status is not TaskStatus.PENDING` 时，抛出 `ValueError(…)`。
    async def _transition_from_pending(
        self,
        task_id: str,
        status: TaskStatus,
    ) -> Task:

        normalized = _validate_task_id(task_id)
        async with self._lock_for(normalized):
            task = await self._require(normalized)
            if task.status is not TaskStatus.PENDING:
                raise ValueError(
                    f"only pending task can be transitioned: {task_id} "
                    f"({task.status.value})"
                )
            now = datetime.now(UTC)
            task_values = task.model_dump(mode="python")
            task_values["status"] = status
            if status in _TERMINAL_STATUSES:
                task_values["completed_at"] = now
            task_values["revision"] = task.revision + 1
            task_values["updated_at"] = now
            updated = Task.model_validate(task_values)
            await self._write(updated)
            return updated

    # 函数说明：FileTaskStore.attach_run
    # 用途：关联运行，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Task`；返回 `await self._update(task_id, TaskPatch(run_id=run_id))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._update` → `TaskPatch`。
    async def attach_run(self, task_id: str, run_id: str) -> Task:

        return await self._update(
            task_id,
            TaskPatch(run_id=run_id),
        )

    # 函数说明：FileTaskStore.active_for_conversation
    # 用途：在任务状态与步骤管理中处理 `active_for_conversation`，通过 `self._all_tasks`
    #  完成首个内部处理步骤。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `Task | None`；返回 `tasks[0] if tasks else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_required_entry` →
    # `self._all_tasks` → `tasks.sort`。
    async def active_for_conversation(
        self,
        conversation_id: str,
    ) -> Task | None:

        normalized = _normalize_required_entry(
            conversation_id,
            field_name="conversation_id",
        )
        tasks = [
            task
            for task in await self._all_tasks()
            if normalized == task.owner_conversation_id
            and task.status in _CONTEXT_TASK_STATUSES
        ]
        tasks.sort(key=lambda task: task.updated_at, reverse=True)
        return tasks[0] if tasks else None

    # 函数说明：FileTaskStore._require
    # 用途：获取并校验必需的FileTaskStore，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    # 返回：类型 `Task`；返回 `task`。
    # 分支与异常：
    #   当 `task is None` 时，抛出 `KeyError(f'任务不存在：{task_id}')`。
    async def _require(self, task_id: str) -> Task:
        task = await self.get(task_id)
        if task is None:
            raise KeyError(f"任务不存在：{task_id}")
        return task

    # 函数说明：FileTaskStore._update
    # 用途：更新FileTaskStore，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    #   patch：传给 `self.apply_patch` 的输入，类型 `TaskPatch`。
    # 返回：类型 `Task`；返回 `await self.apply_patch(task_id, patch)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.apply_patch`。
    async def _update(
        self,
        task_id: str,
        patch: TaskPatch,
    ) -> Task:
        return await self.apply_patch(task_id, patch)

    # 函数说明：FileTaskStore._all_tasks
    # 用途：返回 `await asyncio.to_thread(_scan_tasks, self.tasks_dir)`，提供
    # FileTaskStore 的派生值。
    # 返回：类型 `list[Task]`；返回
    # `await asyncio.to_thread(_scan_tasks, self.tasks_dir)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def _all_tasks(self) -> list[Task]:

        return await asyncio.to_thread(_scan_tasks, self.tasks_dir)

    # 函数说明：FileTaskStore._write
    # 用途：写入FileTaskStore，供任务状态与步骤管理使用。
    # 参数：
    #   task：当前任务记录，类型 `Task`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._path` → `asyncio.to_thread`
    # 。
    # 分支与异常：
    #   当 `await asyncio.to_thread(path.is_symlink)` 时，抛出
    # `ValueError('task path cannot be a symbolic link')`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def _write(self, task: Task) -> None:
        path = self._path(task.id)
        if await asyncio.to_thread(path.is_symlink):
            raise ValueError("task path cannot be a symbolic link")
        await asyncio.to_thread(_write_task, path, task)

    # 函数说明：FileTaskStore._path
    # 用途：在任务状态与步骤管理中处理 `_path`，通过 `path.parent.resolve` 完成首个内部
    # 处理步骤。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    # 返回：类型 `Path`；返回 `path`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_validate_task_id` →
    # `path.parent.resolve`。
    # 分支与异常：
    #   当 `path.parent.resolve() != self.tasks_dir` 时，抛出
    # `ValueError('task path escapes tasks directory')`。
    def _path(self, task_id: str) -> Path:
        normalized = _validate_task_id(task_id)
        path = self.tasks_dir / f"{normalized}.json"
        if path.parent.resolve() != self.tasks_dir:
            raise ValueError("task path escapes tasks directory")
        return path

    # 函数说明：FileTaskStore._lock_for
    # 用途：获取锁`for`，供任务状态与步骤管理使用。
    # 参数：
    #   task_id：目标任务标识，类型 `str`。
    # 返回：类型 `asyncio.Lock`；返回 `lock`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Lock`。
    def _lock_for(self, task_id: str) -> asyncio.Lock:
        lock = self._locks.get(task_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[task_id] = lock
        return lock


# 函数说明：_scan_tasks
# 用途：在任务状态与步骤管理中处理 `_scan_tasks`，通过 `tasks_dir.is_dir` 完成首个内部处
# 理步骤。
# 参数：
#   tasks_dir：相关数据的根目录或保存目录，类型 `Path`。
# 返回：类型 `list[Task]`；按分支返回 `[]`；`tasks`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`tasks_dir.is_dir` → `tasks_dir.glob`
# → `path.is_symlink` → `_TASK_ID_RE.fullmatch` → `_read_task` → `logger.warning`。
# 分支与异常：
#   当 `not tasks_dir.is_dir()` 时，返回 `[]`。
#   当 `path.is_symlink() or not _TASK_ID_RE.fullmatch(path.stem)` 时，跳过当前循环项。
#   当 `task.id != path.stem` 时，跳过当前循环项。
#   捕获 `LegacyTaskOwnershipError` 后，跳过当前循环项，继续处理后续项。
#   捕获 `(OSError, ValueError, TypeError)` 后，跳过当前循环项，继续处理后续项。
def _scan_tasks(tasks_dir: Path) -> list[Task]:
    if not tasks_dir.is_dir():
        return []
    tasks: list[Task] = []
    for path in sorted(tasks_dir.glob("*.json")):
        if path.is_symlink() or not _TASK_ID_RE.fullmatch(path.stem):
            continue
        try:
            task = _read_task(path)
            if task.id != path.stem:
                continue
            tasks.append(task)
        except LegacyTaskOwnershipError:
            continue
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "Skipping invalid task file path=%s error=%s: %s",
                path,
                type(exc).__name__,
                exc,
            )
            continue
    return tasks


# 函数说明：_read_task
# 用途：读取任务，供任务状态与步骤管理使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
# 返回：类型 `Task`；按分支返回 `task`；`Task.model_validate(task_values)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.stat` → `json.loads` →
# `path.read_text` → `task_values.pop` → `logger.warning` → `Task.model_validate`；另有
# 2 个调用点。
# 分支与异常：
#   当 `path.stat().st_size > MAX_TASK_FILE_BYTES` 时，抛出 `ValueError(…)`。
#   `'owner_conversation_id' not in task_values` 分支在完成前置处理后返回 `task`。
#   `not isinstance(legacy_owners, list) or len(legacy_owners) !…` 分支在完成前置处理后
# 抛出 `LegacyTaskOwnershipError(…)`。
#   当 `'conversation_ids' in task_values` 时，抛出 `ValueError(…)`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
def _read_task(path: Path) -> Task:
    if path.stat().st_size > MAX_TASK_FILE_BYTES:
        raise ValueError(
            f"task file exceeds {MAX_TASK_FILE_BYTES} byte safety limit"
        )
    task_values = json.loads(path.read_text(encoding="utf-8"))
    if "owner_conversation_id" not in task_values:
        legacy_owners = task_values.pop("conversation_ids", None)
        if not isinstance(legacy_owners, list) or len(legacy_owners) != 1:
            logger.warning(
                "Legacy task has ambiguous owner and is inaccessible path=%s owners=%r",
                path,
                legacy_owners,
            )
            raise LegacyTaskOwnershipError(
                "legacy task must have exactly one conversation owner"
            )
        task_values["owner_conversation_id"] = legacy_owners[0]
        task = Task.model_validate(task_values)
        _write_task(path, task)
        logger.info(
            "Migrated legacy task owner path=%s owner_conversation_id=%s",
            path,
            task.owner_conversation_id,
        )
        return task
    if "conversation_ids" in task_values:
        raise ValueError("task cannot contain both owner ownership schemas")
    return Task.model_validate(task_values)


# 函数说明：_write_task
# 用途：写入任务，供任务状态与步骤管理使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
#   task：当前任务记录，类型 `Task`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.parent.mkdir` → `json.dumps` →
# `path.with_name` → `uuid4` → `temporary.open` → `file.write`；另有 6 个调用点。
# 副作用与资源：
#   文件或资源访问：`path.parent.mkdir`、`temporary.open`、`os.replace`、
# `temporary.unlink`。
def _write_task(path: Path, task: Task) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        task.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# 函数说明：_apply_patch
# 用途：应用`patch`，供任务状态与步骤管理使用。
# 参数：
#   task：当前任务记录，类型 `Task`。
#   patch：`patch`输入或配置值，类型 `TaskPatch`。
#   now：`now`输入或配置值，类型 `datetime`。
# 返回：类型 `Task`；返回 `Task.model_validate(task_values)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_merge_entries` →
# `_inherit_step_fields` → `_validate_step_replacement` → `TaskStep.model_validate` →
# `Task.model_validate`。
# 分支与异常：
#   当 `patch.status is not None and patch.status is not task.status` 时，抛出
# `ValueError(…)`。
#   当 `step.status is TaskStepStatus.DONE and patch.step_status is…` 时，抛出
# `ValueError('done task step cannot be rolled back')`。
#   当 `TaskStepStatus.SUPERSEDED in (step.status,…` 时，抛出 `ValueError(…)`。
#   当 `not has_matching_step` 时，抛出 `KeyError(f'任务步骤不存在：{patch.step_id}')`。
def _apply_patch(task: Task, patch: TaskPatch, now: datetime) -> Task:

    task_values = task.model_dump(mode="python")
    if task.status in _TERMINAL_STATUSES:
        if patch.status is not None and patch.status is not task.status:
            raise ValueError("terminal task cannot be reopened by ordinary update")
    if "goal" in patch.model_fields_set:
        task_values["goal"] = patch.goal
    if patch.status is not None:
        task_values["status"] = patch.status
        if patch.status in _TERMINAL_STATUSES:
            if task.status not in _TERMINAL_STATUSES:
                task_values["completed_at"] = now
        elif task.status in _TERMINAL_STATUSES:
            task_values["completed_at"] = None
    if "state" in patch.model_fields_set:
        task_values["state"] = patch.state or ()
    if patch.add_constraints:
        task_values["constraints"] = _merge_entries(
            task.constraints,
            patch.add_constraints,
        )
    if patch.add_key_facts:
        task_values["key_facts"] = _merge_entries(
            task.key_facts,
            patch.add_key_facts,
        )
    if "replace_steps" in patch.model_fields_set:
        replacement_steps = _inherit_step_fields(task.steps, patch.replace_steps or ())
        _validate_step_replacement(task.steps, replacement_steps)
        task_values["steps"] = replacement_steps
    if patch.step_id is not None and patch.step_status is not None:
        updated_steps: list[TaskStep] = []
        has_matching_step = False
        for step in task.steps:
            if step.id == patch.step_id:
                has_matching_step = True
                if (
                    step.status is TaskStepStatus.DONE
                    and patch.step_status is not TaskStepStatus.DONE
                ):
                    raise ValueError("done task step cannot be rolled back")
                if TaskStepStatus.SUPERSEDED in (step.status, patch.step_status):
                    raise ValueError(
                        "superseded steps can only be written by a long-running task"
                    )
                step_values = step.model_dump(mode="python")
                step_values["status"] = patch.step_status
                if "step_note" in patch.model_fields_set:
                    step_values["note"] = patch.step_note
                updated_steps.append(TaskStep.model_validate(step_values))
            else:
                updated_steps.append(step)
        if not has_matching_step:
            raise KeyError(f"任务步骤不存在：{patch.step_id}")
        task_values["steps"] = tuple(updated_steps)
    if patch.run_id is not None:
        task_values["run_ids"] = _merge_entries(task.run_ids, (patch.run_id,))
    task_values["revision"] = task.revision + 1
    task_values["updated_at"] = now
    return Task.model_validate(task_values)


# 函数说明：_validate_step_replacement
# 用途：校验步骤，供任务状态与步骤管理使用。
# 参数：
#   existing：已经存在的值或记录，类型 `Sequence[TaskStep]`。
#   replacement：`replacement`输入或配置值，类型 `Sequence[TaskStep]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   `step.status is TaskStepStatus.SUPERSEDED` 分支在完成前置处理后跳过当前循环项。
#   当 `replacement_by_id.get(step.id) != step` 时，抛出 `ValueError(…)`。
#   当 `step.status not in {TaskStepStatus.DONE,…` 时，跳过当前循环项。
#   当 `candidate is None` 时，抛出 `ValueError(…)`。
def _validate_step_replacement(
    existing: Sequence[TaskStep],
    replacement: Sequence[TaskStep],
) -> None:

    replacement_by_id = {step.id: step for step in replacement}
    for step in existing:
        if step.status is TaskStepStatus.SUPERSEDED:
            if replacement_by_id.get(step.id) != step:
                raise ValueError(
                    f"replace_steps cannot change superseded step: {step.id}"
                )
            continue
        if step.status not in {
            TaskStepStatus.DONE,
            TaskStepStatus.IN_PROGRESS,
        }:
            continue
        candidate = replacement_by_id.get(step.id)
        if candidate is None:
            raise ValueError(
                f"replace_steps cannot delete protected step: {step.id}"
            )
        if step.status is TaskStepStatus.DONE:
            if candidate.status is not TaskStepStatus.DONE:
                raise ValueError(
                    f"replace_steps cannot roll back done step: {step.id}"
                )
        elif candidate.status is TaskStepStatus.TODO:
            raise ValueError(
                f"replace_steps cannot roll back in_progress step: {step.id}"
            )


# 函数说明：_inherit_step_fields
# 用途：整体替换步骤时，同 id 步骤没写 acceptance 就沿用原来的，产生版本也沿用。
# 参数：
#   existing：已经存在的值或记录，类型 `Sequence[TaskStep]`。
#   replacement：`replacement`输入或配置值，类型 `Sequence[TaskStep]`。
# 返回：类型 `tuple[TaskStep, ...]`；返回 `tuple(merged)`。
# 分支与异常：
#   `previous is None` 分支在完成前置处理后跳过当前循环项。
def _inherit_step_fields(
    existing: Sequence[TaskStep],
    replacement: Sequence[TaskStep],
) -> tuple[TaskStep, ...]:

    """整体替换步骤时，同 id 步骤没写 acceptance 就沿用原来的，产生版本也沿用。"""
    existing_by_id = {step.id: step for step in existing}
    merged: list[TaskStep] = []
    for step in replacement:
        previous = existing_by_id.get(step.id)
        if previous is None:
            merged.append(step)
            continue
        update: dict[str, object] = {
            "origin_requirements_revision": previous.origin_requirements_revision
        }
        if step.acceptance is None and previous.acceptance is not None:
            update["acceptance"] = previous.acceptance
        merged.append(step.model_copy(update=update))
    return tuple(merged)


# 函数说明：_precondition_failure
# 用途：在任务状态与步骤管理中处理 `_precondition_failure`，通过 `'/'.join` 完成首个内部
# 处理步骤。
# 参数：
#   task：当前任务记录，类型 `Task`。
#   condition：`condition`输入或配置值，类型 `OpPrecondition`。
# 返回：类型 `str | None`；按分支返回
# `f'task revision is {task.revision}, expected {condition.revision}'`；
# `f'task step not found: {condition.step_id}'`；
# `f'task step {step.id} is {step.status.value}, expected {expected}'`；
# `'task steps not done or superseded: ' + ', '.join(open_steps)` 等 5 种表达式。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`next`。
# 分支与异常：
#   当 `condition.revision is not None and task.revision !=…` 时，返回
# `f'task revision is {task.revision}, expected {…`。
#   当 `step is None` 时，返回 `f'task step not found: {condition.step_id}'`。
#   `step.status not in condition.step_in` 分支在完成前置处理后返回
# `f'task step {step.id} is {step.status.value}, expected {…`。
#   当 `open_steps` 时，返回 `'task steps not done or superseded: ' + ', '.join(…`。
def _precondition_failure(task: Task, condition: OpPrecondition) -> str | None:

    if condition.revision is not None and task.revision != condition.revision:
        return f"task revision is {task.revision}, expected {condition.revision}"
    if condition.step_id is not None:
        step = next((s for s in task.steps if s.id == condition.step_id), None)
        if step is None:
            return f"task step not found: {condition.step_id}"
        if step.status not in condition.step_in:
            expected = "/".join(status.value for status in condition.step_in)
            return (
                f"task step {step.id} is {step.status.value}, expected {expected}"
            )
    if condition.all_steps_closed:
        open_steps = [s.id for s in task.steps if s.status not in CLOSED_STEP_STATUSES]
        if open_steps:
            return "task steps not done or superseded: " + ", ".join(open_steps)
    return None


# 函数说明：_apply_op
# 用途：应用`op`，供任务状态与步骤管理使用。
# 参数：
#   task：当前任务记录，类型 `Task`。
#   op：`op`输入或配置值，类型 `TaskOp`。
#   digest：`digest`输入或配置值，类型 `str`。
#   now：`now`输入或配置值，类型 `datetime`。
# 返回：类型 `Task`；返回 `Task.model_validate(task_values)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskStep.model_validate` →
# `applied.pop` → `next` → `iter` → `Task.model_validate`。
# 分支与异常：
#   当 `task.status in _TERMINAL_STATUSES` 时，抛出
# `ValueError(f'task is {task.status.value}')`。
#   当 `item.step_id not in index_by_id` 时，抛出
# `KeyError(f'任务步骤不存在：{item.step_id}')`。
#   当 `step.status is TaskStepStatus.SUPERSEDED` 时，抛出
# `ValueError(f'task step already superseded: {step.id}')`。
#   当 `new_step.id in index_by_id` 时，抛出
# `ValueError(f'task step id already exists: {new_step.id}')`。
def _apply_op(task: Task, op: TaskOp, digest: str, now: datetime) -> Task:

    if task.status in _TERMINAL_STATUSES:
        raise ValueError(f"task is {task.status.value}")
    task_values = task.model_dump(mode="python")
    if op.state is not None:
        task_values["state"] = op.state
    if op.set_contract:
        task_values["contract"] = op.contract

    steps = list(task.steps)
    index_by_id = {step.id: index for index, step in enumerate(steps)}
    for item in op.supersede:
        if item.step_id not in index_by_id:
            raise KeyError(f"任务步骤不存在：{item.step_id}")
        index = index_by_id[item.step_id]
        step = steps[index]
        if step.status is TaskStepStatus.SUPERSEDED:
            raise ValueError(f"task step already superseded: {step.id}")
        steps[index] = TaskStep.model_validate(
            {
                **step.model_dump(mode="python"),
                "status": TaskStepStatus.SUPERSEDED,
                "superseded_by": item.amendment_id,
                "superseded_reason": item.reason,
            }
        )
    for new_step in op.add_steps:
        if new_step.id in index_by_id:
            raise ValueError(f"task step id already exists: {new_step.id}")
        index_by_id[new_step.id] = len(steps)
        steps.append(new_step)
    if op.step_status is not None:
        change = op.step_status
        if change.step_id not in index_by_id:
            raise KeyError(f"任务步骤不存在：{change.step_id}")
        index = index_by_id[change.step_id]
        step = steps[index]
        if step.status is TaskStepStatus.SUPERSEDED:
            raise ValueError(f"superseded task step cannot change: {step.id}")
        if step.status is TaskStepStatus.DONE and change.status is not TaskStepStatus.DONE:
            raise ValueError("done task step cannot be rolled back")
        update: dict[str, object] = {"status": change.status}
        if change.note is not None:
            update["note"] = change.note
        steps[index] = TaskStep.model_validate(
            {**step.model_dump(mode="python"), **update}
        )
    task_values["steps"] = tuple(steps)

    if op.complete_task:
        task_values["status"] = TaskStatus.COMPLETED
        task_values["completed_at"] = now

    applied = dict(task.applied_ops)
    applied[op.op_id] = digest
    while len(applied) > MAX_APPLIED_OPS:
        applied.pop(next(iter(applied)))
    task_values["applied_ops"] = applied
    task_values["revision"] = task.revision + 1
    task_values["updated_at"] = now
    return Task.model_validate(task_values)


# 函数说明：_merge_entries
# 用途：合并条目，供任务状态与步骤管理使用。
# 参数：
#   existing：已经存在的值或记录，类型 `Sequence[str]`。
#   new：`new`输入或配置值，类型 `Sequence[str]`。
# 返回：类型 `tuple[str, ...]`；返回 `tuple(merged)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`seen.add`。
def _merge_entries(
    existing: Sequence[str],
    new: Sequence[str],
) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for entry in (*existing, *new):
        text = " ".join(entry.split()).strip()
        if text and text not in seen:
            merged.append(text)
            seen.add(text)
    return tuple(merged)


# 函数说明：_validate_task_id
# 用途：校验任务标识，供任务状态与步骤管理使用。
# 参数：
#   task_id：目标任务标识，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`task_id.strip().lower` →
# `_TASK_ID_RE.fullmatch`。
# 分支与异常：
#   当 `not isinstance(task_id, str)` 时，抛出 `TypeError('task_id must be a string')`。
#   当 `not _TASK_ID_RE.fullmatch(normalized)` 时，抛出 `ValueError(…)`。
def _validate_task_id(task_id: str) -> str:
    if not isinstance(task_id, str):
        raise TypeError("task_id must be a string")
    normalized = task_id.strip().lower()
    if not _TASK_ID_RE.fullmatch(normalized):
        raise ValueError("task_id must be a 32-character hexadecimal string")
    return normalized


# 函数说明：_validate_task_prefix
# 用途：校验任务，供任务状态与步骤管理使用。
# 参数：
#   identifier：待规范化的标识，类型 `str`。
# 返回：类型 `str`；返回 `identifier`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_TASK_PREFIX_RE.fullmatch`。
# 分支与异常：
#   当 `not _TASK_PREFIX_RE.fullmatch(identifier)` 时，抛出 `ValueError(…)`。
def _validate_task_prefix(identifier: str) -> str:
    if not _TASK_PREFIX_RE.fullmatch(identifier):
        raise ValueError(
            "task identifier must be a 4-32 character hexadecimal prefix"
        )
    return identifier


# 函数说明：_normalize_required_entry
# 用途：规范化条目，供任务状态与步骤管理使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
#   field_name：待校验的字段名称，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 分支与异常：
#   当 `not isinstance(value, str)` 时，抛出
# `TypeError(f'{field_name} must be a string')`。
#   当 `not normalized` 时，抛出 `ValueError(f'{field_name} cannot be empty')`。
def _normalize_required_entry(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


__all__ = ["DEFAULT_TASKS_DIR", "MAX_TASK_FILE_BYTES", "FileTaskStore", "OpIdReusedError"]
