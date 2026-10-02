
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.domain.task import (
    FileTaskStore,
    OpIdReusedError,
    OpPrecondition,
    OpResult,
    StepStatusChange,
    StepSupersede,
    TaskCreateTool,
    TaskGetTool,
    TaskOp,
    TaskPatch,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
    TaskUpdateTool,
    steps_missing_acceptance,
)
from app.domain.task.models import MAX_APPLIED_OPS
from app.models.types import ToolCall
from app.tools import ToolExecutionContext

_OWNER = "conv-test"
_MEA = "mea1"
_TODO_OR_RUNNING = (TaskStepStatus.TODO, TaskStepStatus.IN_PROGRESS)


# 函数说明：_store
# 用途：保存回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `FileTaskStore`；返回 `store`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` → `store.initialize`。
async def _store(tmp_path: Path) -> FileTaskStore:
    store = FileTaskStore(tmp_path / "tasks")
    await store.initialize()
    return store


# 函数说明：_active_task
# 用途：在回归测试与测试辅助中处理 `_active_task`，通过 `store.create` 完成首个内部处理
# 步骤。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：返回 `await store.set_status(task.id, TaskStatus.ACTIVE)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `store.set_status`。
async def _active_task(store: FileTaskStore):
    task = await store.create(
        owner_conversation_id=_OWNER,
        title="导入用户数据",
        goal="users 表包含 CSV 全部数据",
        steps=(
            TaskStep(id="s1", title="读取 CSV", acceptance="读到表头和 120 行"),
            TaskStep(id="s2", title="导入数据库", acceptance="users 表 120 行"),
        ),
    )
    return await store.set_status(task.id, TaskStatus.ACTIVE)


# 函数说明：_begin
# 用途：开始回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   op_id：幂等操作标识，类型 `str`。
#   step_id：目标步骤标识，类型 `str`。
# 返回：类型 `TaskOp`；返回 `TaskOp(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskOp` → `OpPrecondition` →
# `StepStatusChange`。
def _begin(op_id: str, step_id: str) -> TaskOp:
    return TaskOp(
        op_id=op_id,
        precondition=OpPrecondition(step_id=step_id, step_in=_TODO_OR_RUNNING),
        step_status=StepStatusChange(step_id=step_id, status=TaskStepStatus.IN_PROGRESS),
    )


# 函数说明：_verdict
# 用途：返回 `TaskOp(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   op_id：幂等操作标识，类型 `str`。
#   step_id：目标步骤标识，类型 `str`。
#   status：目标状态，类型 `TaskStepStatus`。
#   note：`note`输入或配置值，类型 `str`。
# 返回：类型 `TaskOp`；返回 `TaskOp(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskOp` → `OpPrecondition` →
# `StepStatusChange`。
def _verdict(op_id: str, step_id: str, status: TaskStepStatus, note: str) -> TaskOp:
    return TaskOp(
        op_id=op_id,
        precondition=OpPrecondition(step_id=step_id, step_in=(TaskStepStatus.IN_PROGRESS,)),
        step_status=StepStatusChange(step_id=step_id, status=status, note=note),
    )


# ---------------------------------------------------------------------------
# 模型兼容
# ---------------------------------------------------------------------------


# 函数说明：test_task_file_written_before_mea_fields_still_loads
# 用途：回归验证回归测试与测试辅助中的 `task_file_written_before_mea_fields_still_loads`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` →
# `(store.tasks_dir / f'{task_id}.json').write_text` → `json.dumps` →
# `steps_missing_acceptance`。
# 分支与异常：
#   验证条件：`task is not None`。
#   验证条件：`task.contract is None`。
#   验证条件：`task.applied_ops == {}`。
#   验证条件：`task.steps[0].acceptance is None`。
# 副作用与资源：
#   文件或资源访问：`(store.tasks_dir / f'{task_id}.json').write_text`。
async def test_task_file_written_before_mea_fields_still_loads(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task_id = "a" * 32
    legacy = {
        "id": task_id,
        "title": "旧任务",
        "status": "active",
        "priority": "normal",
        "steps": [{"id": "s1", "title": "旧步骤", "status": "todo", "note": None}],
        "owner_conversation_id": _OWNER,
        "created_at": "2026-09-01T00:00:00Z",
        "updated_at": "2026-09-01T00:00:00Z",
        "revision": 3,
    }
    (store.tasks_dir / f"{task_id}.json").write_text(json.dumps(legacy), encoding="utf-8")

    task = await store.get(task_id)

    assert task is not None
    assert task.contract is None
    assert task.applied_ops == {}
    assert task.steps[0].acceptance is None
    assert task.steps[0].origin_requirements_revision == 1
    assert steps_missing_acceptance(task) == ("s1",)


# 函数说明：test_superseded_step_requires_amendment_reference
# 用途：回归验证回归测试与测试辅助中的 `superseded_step_requires_amendment_reference` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `TaskStep`。
# 分支与异常：
#   验证条件：`step.superseded_by == 'A1'`。
#   预期异常：`pytest.raises(ValidationError)`。
def test_superseded_step_requires_amendment_reference() -> None:
    with pytest.raises(ValidationError):
        TaskStep(id="s1", title="x", status=TaskStepStatus.SUPERSEDED)
    with pytest.raises(ValidationError):
        TaskStep(id="s1", title="x", superseded_by="A1")
    step = TaskStep(id="s1", title="x", status=TaskStepStatus.SUPERSEDED, superseded_by="a1")
    assert step.superseded_by == "A1"


# ---------------------------------------------------------------------------
# 幂等：同一操作 ID 只应用一次
# ---------------------------------------------------------------------------


# 函数说明：_plan_op
# 用途：返回 `TaskOp(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   task_revision：任务输入或配置值，类型 `int`。
#   op_id：幂等操作标识，类型 `str`；默认 `f'{_MEA}/r001/plan'`。
# 返回：类型 `TaskOp`；返回 `TaskOp(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskOp` → `OpPrecondition` →
# `TaskStep`。
def _plan_op(task_revision: int, op_id: str = f"{_MEA}/r001/plan") -> TaskOp:
    return TaskOp(
        op_id=op_id,
        precondition=OpPrecondition(revision=task_revision),
        state=("已完成: 无", "未完成: s1、s2"),
        set_contract=True,
        contract="目标状态: users 表 120 行\n验收约束: 不修改表结构",
        add_steps=(TaskStep(id="s3", title="校验行数", acceptance="count(*) = 120"),),
    )


# 函数说明：test_plan_op_applies_once_and_replay_is_a_noop
# 用途：回归验证回归测试与测试辅助中的 `plan_op_applies_once_and_replay_is_a_noop` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` → `_plan_op`
#  → `store.apply_op` → `op.digest`。
# 分支与异常：
#   验证条件：`first.result is OpResult.APPLIED`。
#   验证条件：`replay.result is OpResult.ALREADY_APPLIED`。
#   验证条件：`stored is not None`。
#   验证条件：`[step.id for step in stored.steps] == ['s1', 's2', 's3']`。
async def test_plan_op_applies_once_and_replay_is_a_noop(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)
    op = _plan_op(task.revision)

    first = await store.apply_op(task.id, op)
    replay = await store.apply_op(task.id, op)  # 模拟：写 Task 后、保存 round 前崩溃，恢复时重放

    assert first.result is OpResult.APPLIED
    assert replay.result is OpResult.ALREADY_APPLIED
    stored = await store.get(task.id)
    assert stored is not None
    assert [step.id for step in stored.steps] == ["s1", "s2", "s3"]  # 新增步骤只出现一次
    assert stored.revision == task.revision + 1
    assert stored.contract == "目标状态: users 表 120 行\n验收约束: 不修改表结构"  # 保留换行
    assert stored.state == ("已完成: 无", "未完成: s1、s2")
    assert stored.applied_ops == {op.op_id: op.digest()}


# 函数说明：test_revision_conflict_is_not_mistaken_for_applied
# 用途：回归验证回归测试与测试辅助中的 `revision_conflict_is_not_mistaken_for_applied`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` → `_plan_op`
#  → `store.update_goal` → `store.apply_op`。
# 分支与异常：
#   验证条件：`outcome.result is OpResult.CONFLICT`。
#   验证条件：`'expected' in (outcome.reason or '')`。
#   验证条件：`stored is not None`。
#   验证条件：`op.op_id not in stored.applied_ops`。
async def test_revision_conflict_is_not_mistaken_for_applied(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)
    op = _plan_op(task.revision)
    await store.update_goal(task.id, "别处修改了任务")  # revision +1，本次操作并未执行

    outcome = await store.apply_op(task.id, op)

    assert outcome.result is OpResult.CONFLICT
    assert "expected" in (outcome.reason or "")
    stored = await store.get(task.id)
    assert stored is not None
    assert op.op_id not in stored.applied_ops
    assert [step.id for step in stored.steps] == ["s1", "s2"]


# 函数说明：test_same_op_id_with_different_content_raises
# 用途：回归验证回归测试与测试辅助中的 `same_op_id_with_different_content_raises` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.apply_op` → `_plan_op` → `pytest.raises`。
# 分支与异常：
#   预期异常：`pytest.raises(OpIdReusedError)`。
async def test_same_op_id_with_different_content_raises(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)
    await store.apply_op(task.id, _plan_op(task.revision))

    different = _plan_op(task.revision).model_copy(update={"state": ("另一份计划",)})
    with pytest.raises(OpIdReusedError):
        await store.apply_op(task.id, different)


# 函数说明：test_new_steps_require_acceptance
# 用途：回归验证回归测试与测试辅助中的 `new_steps_require_acceptance` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `TaskOp` →
# `TaskStep`。
# 分支与异常：
#   预期异常：`pytest.raises(ValidationError)`。
async def test_new_steps_require_acceptance() -> None:
    with pytest.raises(ValidationError):
        TaskOp(op_id="x", add_steps=(TaskStep(id="s9", title="没有验收"),))
    with pytest.raises(ValidationError):
        TaskOp(op_id="x")  # 没有任何改动


# 函数说明：test_applied_ops_are_trimmed_to_limit
# 用途：回归验证回归测试与测试辅助中的 `applied_ops_are_trimmed_to_limit` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.apply_op` → `TaskOp`。
# 分支与异常：
#   验证条件：`stored is not None`。
#   验证条件：`len(stored.applied_ops) == MAX_APPLIED_OPS`。
#   验证条件：`f'{_MEA}/r000/state' not in stored.applied_ops`。
#   验证条件：`f'{_MEA}/r{MAX_APPLIED_OPS + 4:03d}/state' in stored.applied_ops`。
async def test_applied_ops_are_trimmed_to_limit(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)
    for index in range(MAX_APPLIED_OPS + 5):
        await store.apply_op(task.id, TaskOp(op_id=f"{_MEA}/r{index:03d}/state", state=(str(index),)))

    stored = await store.get(task.id)
    assert stored is not None
    assert len(stored.applied_ops) == MAX_APPLIED_OPS
    assert f"{_MEA}/r000/state" not in stored.applied_ops
    assert f"{_MEA}/r{MAX_APPLIED_OPS + 4:03d}/state" in stored.applied_ops


# ---------------------------------------------------------------------------
# 步骤状态：begin / verdict / release
# ---------------------------------------------------------------------------


# 函数说明：test_begin_verdict_and_takeover_by_recovery_round
# 用途：回归验证回归测试与测试辅助中的 `begin_verdict_and_takeover_by_recovery_round` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.apply_op` → `_begin` → `_verdict`。
# 分支与异常：
#   验证条件：
# `(await store.apply_op(task.id, _begin(f'{_MEA}/r001/begin', 's1'))).applied`。
#   验证条件：`takeover.result is OpResult.APPLIED`。
#   验证条件：`done.result is OpResult.APPLIED`。
#   验证条件：`step.status is TaskStepStatus.DONE and step.note == '依据 round_002'`。
async def test_begin_verdict_and_takeover_by_recovery_round(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)

    assert (await store.apply_op(task.id, _begin(f"{_MEA}/r001/begin", "s1"))).applied
    # 中断轮留下 in_progress，recovery_audit 用自己的 begin 接管
    takeover = await store.apply_op(task.id, _begin(f"{_MEA}/r002/begin", "s1"))
    assert takeover.result is OpResult.APPLIED

    done = await store.apply_op(
        task.id, _verdict(f"{_MEA}/r002/verdict", "s1", TaskStepStatus.DONE, "依据 round_002")
    )
    assert done.result is OpResult.APPLIED
    step = done.task.steps[0]
    assert step.status is TaskStepStatus.DONE and step.note == "依据 round_002"


# 函数说明：test_audit_only_round_can_move_todo_step_to_done
# 用途：回归验证回归测试与测试辅助中的 `audit_only_round_can_move_todo_step_to_done` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.apply_op` → `_begin` → `_verdict`。
# 分支与异常：
#   验证条件：`outcome.result is OpResult.APPLIED`。
#   验证条件：`outcome.task.steps[1].status is TaskStepStatus.DONE`。
async def test_audit_only_round_can_move_todo_step_to_done(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)

    await store.apply_op(task.id, _begin(f"{_MEA}/r001/begin", "s2"))
    outcome = await store.apply_op(
        task.id, _verdict(f"{_MEA}/r001/verdict", "s2", TaskStepStatus.DONE, "依据 round_001")
    )

    assert outcome.result is OpResult.APPLIED
    assert outcome.task.steps[1].status is TaskStepStatus.DONE


# 函数说明：test_only_one_step_can_be_in_progress
# 用途：回归验证回归测试与测试辅助中的 `only_one_step_can_be_in_progress` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.apply_op` → `_begin`。
# 分支与异常：
#   验证条件：`outcome.result is OpResult.CONFLICT`。
#   验证条件：`'in_progress' in (outcome.reason or '')`。
async def test_only_one_step_can_be_in_progress(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)
    await store.apply_op(task.id, _begin(f"{_MEA}/r001/begin", "s1"))

    outcome = await store.apply_op(task.id, _begin(f"{_MEA}/r002/begin", "s2"))

    assert outcome.result is OpResult.CONFLICT
    assert "in_progress" in (outcome.reason or "")


# 函数说明：test_release_returns_step_to_todo
# 用途：回归验证回归测试与测试辅助中的 `release_returns_step_to_todo` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.apply_op` → `_begin` → `TaskOp` → `OpPrecondition`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`outcome.task.steps[0].status is TaskStepStatus.TODO`。
#   验证条件：
# `(await store.apply_op(task.id, _begin(f'{_MEA}/r004/begin', 's2'))).applied`。
async def test_release_returns_step_to_todo(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)
    await store.apply_op(task.id, _begin(f"{_MEA}/r003/begin", "s1"))

    release = TaskOp(
        op_id=f"{_MEA}/r003/release",
        precondition=OpPrecondition(step_id="s1", step_in=(TaskStepStatus.IN_PROGRESS,)),
        step_status=StepStatusChange(step_id="s1", status=TaskStepStatus.TODO, note="r003 作废"),
    )
    outcome = await store.apply_op(task.id, release)

    assert outcome.task.steps[0].status is TaskStepStatus.TODO
    assert (await store.apply_op(task.id, _begin(f"{_MEA}/r004/begin", "s2"))).applied


# 函数说明：test_verdict_precondition_requires_in_progress
# 用途：回归验证回归测试与测试辅助中的 `verdict_precondition_requires_in_progress` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.apply_op` → `_verdict`。
# 分支与异常：
#   验证条件：`outcome.result is OpResult.CONFLICT`。
#   验证条件：`'is todo, expected in_progress' in (outcome.reason or '')`。
async def test_verdict_precondition_requires_in_progress(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)

    outcome = await store.apply_op(
        task.id, _verdict(f"{_MEA}/r001/verdict", "s1", TaskStepStatus.DONE, "依据 round_001")
    )

    assert outcome.result is OpResult.CONFLICT
    assert "is todo, expected in_progress" in (outcome.reason or "")


# ---------------------------------------------------------------------------
# 取代与完成
# ---------------------------------------------------------------------------


# 函数说明：test_supersede_done_step_and_block_further_changes
# 用途：回归验证回归测试与测试辅助中的 `supersede_done_step_and_block_further_changes`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.apply_op` → `_begin` → `_verdict` → `TaskOp`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`outcome.result is OpResult.APPLIED`。
#   验证条件：`(s1.status, s1.superseded_by, s1.superseded_reason) == (TaskStepStatus.
# SUPERSEDED, 'A1', '改为导出 CSV')`。
#   验证条件：`outcome.task.steps[2].origin_requirements_revision == 2`。
#   验证条件：`blocked.result is OpResult.CONFLICT`。
#   预期异常：`pytest.raises(ValueError, match='superseded')`。
async def test_supersede_done_step_and_block_further_changes(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)
    await store.apply_op(task.id, _begin(f"{_MEA}/r001/begin", "s1"))
    await store.apply_op(
        task.id, _verdict(f"{_MEA}/r001/verdict", "s1", TaskStepStatus.DONE, "依据 round_001")
    )

    supersede = TaskOp(
        op_id=f"{_MEA}/r002/plan",
        supersede=(StepSupersede(step_id="s1", amendment_id="a1", reason="改为导出 CSV"),),
        add_steps=(TaskStep(id="s3", title="导出 CSV", acceptance="out/users.csv 存在",
                            origin_requirements_revision=2),),
    )
    outcome = await store.apply_op(task.id, supersede)
    assert outcome.result is OpResult.APPLIED
    s1 = outcome.task.steps[0]
    assert (s1.status, s1.superseded_by, s1.superseded_reason) == (
        TaskStepStatus.SUPERSEDED, "A1", "改为导出 CSV"
    )
    assert outcome.task.steps[2].origin_requirements_revision == 2

    blocked = await store.apply_op(task.id, _begin(f"{_MEA}/r003/begin", "s1"))
    assert blocked.result is OpResult.CONFLICT
    again = TaskOp(
        op_id=f"{_MEA}/r003/plan",
        supersede=(StepSupersede(step_id="s1", amendment_id="A2", reason="再取代"),),
    )
    assert (await store.apply_op(task.id, again)).result is OpResult.CONFLICT

    # 普通任务工具不能写或改 superseded
    with pytest.raises(ValueError, match="superseded"):
        await store.set_step_status(task.id, "s1", TaskStepStatus.TODO)
    with pytest.raises(ValueError, match="superseded"):
        await store.set_step_status(task.id, "s2", TaskStepStatus.SUPERSEDED)
    with pytest.raises(ValueError, match="superseded"):
        await store.apply_patch(
            task.id, TaskPatch(replace_steps=(TaskStep(id="s2", title="只剩一步"),))
        )


# 函数说明：_complete_op
# 用途：完成`op`，供回归测试与测试辅助使用。
# 参数：
#   op_id：幂等操作标识，类型 `str`；默认 `f'{_MEA}/complete'`。
# 返回：类型 `TaskOp`；返回 `TaskOp(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskOp` → `OpPrecondition`。
def _complete_op(op_id: str = f"{_MEA}/complete") -> TaskOp:
    return TaskOp(
        op_id=op_id,
        precondition=OpPrecondition(all_steps_closed=True),
        complete_task=True,
    )


# 函数说明：test_complete_requires_every_step_done_or_superseded
# 用途：回归验证回归测试与测试辅助中的 `complete_requires_every_step_done_or_superseded`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.apply_op` → `_complete_op` → `_begin` → `_verdict`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`early.result is OpResult.CONFLICT`。
#   验证条件：`'s1, s2' in (early.reason or '')`。
#   验证条件：`done.result is OpResult.APPLIED`。
#   验证条件：`done.task.status is TaskStatus.COMPLETED`。
async def test_complete_requires_every_step_done_or_superseded(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)

    early = await store.apply_op(task.id, _complete_op())
    assert early.result is OpResult.CONFLICT
    assert "s1, s2" in (early.reason or "")

    await store.apply_op(task.id, _begin(f"{_MEA}/r001/begin", "s1"))
    await store.apply_op(
        task.id, _verdict(f"{_MEA}/r001/verdict", "s1", TaskStepStatus.DONE, "依据 round_001")
    )
    await store.apply_op(
        task.id,
        TaskOp(op_id=f"{_MEA}/r002/plan",
               supersede=(StepSupersede(step_id="s2", amendment_id="A1", reason="不再需要"),)),
    )

    done = await store.apply_op(task.id, _complete_op())
    replay = await store.apply_op(task.id, _complete_op())

    assert done.result is OpResult.APPLIED
    assert done.task.status is TaskStatus.COMPLETED
    assert done.task.completed_at is not None
    assert replay.result is OpResult.ALREADY_APPLIED


# 函数说明：test_ops_on_terminal_task_conflict
# 用途：回归验证回归测试与测试辅助中的 `ops_on_terminal_task_conflict` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` → `_active_task` →
# `store.set_status` → `store.apply_op` → `_begin`。
# 分支与异常：
#   验证条件：`outcome.result is OpResult.CONFLICT`。
#   验证条件：`'cancelled' in (outcome.reason or '')`。
async def test_ops_on_terminal_task_conflict(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    task = await _active_task(store)
    await store.set_status(task.id, TaskStatus.CANCELLED)

    outcome = await store.apply_op(task.id, _begin(f"{_MEA}/r001/begin", "s1"))

    assert outcome.result is OpResult.CONFLICT
    assert "cancelled" in (outcome.reason or "")


# ---------------------------------------------------------------------------
# 任务工具
# ---------------------------------------------------------------------------


# 函数说明：_context
# 用途：返回 `ToolExecutionContext(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `ToolExecutionContext`；返回 `ToolExecutionContext(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolExecutionContext` → `ToolCall`。
def _context(name: str) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_call=ToolCall(id=f"{name}-call", name=name, arguments={}),
        conversation_id=_OWNER,
        run_id="run-test",
    )


# 函数说明：test_task_tools_carry_acceptance_and_hide_mea_internals
# 用途：回归验证回归测试与测试辅助中的
# `task_tools_carry_acceptance_and_hide_mea_internals` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_store` →
# `TaskCreateTool(store).execute_with_context` → `TaskCreateTool` → `_context` →
# `steps_missing_acceptance` → `TaskUpdateTool(store).execute_with_context`；另有 3 个调
# 用点。
# 分支与异常：
#   验证条件：
# `[step['acceptance'] for step in created['steps']] == ['读到 120 行', None]`。
#   验证条件：`'applied_ops' not in created`。
#   验证条件：`task is not None`。
#   验证条件：`steps_missing_acceptance(task) == (created['steps'][1]['id'],)`。
async def test_task_tools_carry_acceptance_and_hide_mea_internals(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    created = await TaskCreateTool(store).execute_with_context(
        {
            "title": "导入",
            "goal": "导入 CSV",
            "steps": [
                {"title": "读取", "acceptance": "读到 120 行"},
                {"title": "导入"},
            ],
        },
        _context("task_create"),
    )
    assert [step["acceptance"] for step in created["steps"]] == ["读到 120 行", None]
    assert "applied_ops" not in created

    task = await store.get(created["id"])
    assert task is not None
    assert steps_missing_acceptance(task) == (created["steps"][1]["id"],)

    # 整体替换步骤时，同 id 步骤没写 acceptance 就沿用原来的
    first_id = created["steps"][0]["id"]
    updated = await TaskUpdateTool(store).execute_with_context(
        {"task_id": task.id, "steps": [{"id": first_id, "title": "读取 CSV"}]},
        _context("task_update"),
    )
    assert updated["steps"][0]["acceptance"] == "读到 120 行"

    fetched = await TaskGetTool(store).execute_with_context({"task_id": task.id}, _context("task_get"))
    assert "applied_ops" not in fetched

    enum_values = TaskUpdateTool(store).definition.parameters["properties"]["step_status"]["enum"]
    assert "superseded" not in enum_values
