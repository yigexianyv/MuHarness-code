from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest

from app.domain.task import (
    DEFAULT_TASKS_DIR,
    FileTaskStore,
    TaskContextProvider,
    TaskPatch,
    TaskPriority,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
)
from app.paths import runtime_data_path

_OWNER = "conv-test"


# 函数说明：store
# 用途：保存回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `FileTaskStore`；返回 `instance`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` →
# `instance.initialize`。
@pytest.fixture
async def store(tmp_path) -> FileTaskStore:
    instance = FileTaskStore(tmp_path / "tasks")
    await instance.initialize()
    return instance


# 函数说明：test_default_tasks_dir_uses_runtime_data_path
# 用途：回归验证回归测试与测试辅助中的 `default_tasks_dir_uses_runtime_data_path` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`runtime_data_path`。
# 分支与异常：
#   验证条件：`DEFAULT_TASKS_DIR.name == 'tasks'`。
#   验证条件：`DEFAULT_TASKS_DIR == runtime_data_path('tasks')`。
async def test_default_tasks_dir_uses_runtime_data_path() -> None:
    assert DEFAULT_TASKS_DIR.name == "tasks"
    assert DEFAULT_TASKS_DIR == runtime_data_path("tasks")


# 函数说明：test_create_and_get_round_trip
# 用途：回归验证回归测试与测试辅助中的 `create_and_get_round_trip` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `task_file.is_file` → `json.loads` → `task_file.read_text`。
# 分支与异常：
#   验证条件：`loaded is not None`。
#   验证条件：`loaded.title == '构建 API 层'`。
#   验证条件：`loaded.description == '把 Agent 暴露为可调用服务'`。
#   验证条件：`loaded.goal == '完成 /chat SSE 端点'`。
# 副作用与资源：
#   文件或资源访问：`task_file.read_text`。
async def test_create_and_get_round_trip(store: FileTaskStore) -> None:
    task = await store.create(
        owner_conversation_id=_OWNER,
        title="构建 API 层",
        description="把 Agent 暴露为可调用服务",
        goal="完成 /chat SSE 端点",
        priority=TaskPriority.HIGH,
        steps=(
            TaskStep(id="s1", title="设计端点"),
            TaskStep(id="s2", title="实现 SSE"),
        ),
    )

    loaded = await store.get(task.id)
    assert loaded is not None
    assert loaded.title == "构建 API 层"
    assert loaded.description == "把 Agent 暴露为可调用服务"
    assert loaded.goal == "完成 /chat SSE 端点"
    assert loaded.priority is TaskPriority.HIGH
    assert loaded.status is TaskStatus.PENDING
    assert len(loaded.steps) == 2
    assert loaded.completed_at is None

    task_file = store.tasks_dir / f"{task.id}.json"
    assert task_file.is_file()
    payload = json.loads(task_file.read_text(encoding="utf-8"))
    assert payload["title"] == "构建 API 层"
    assert payload["steps"][0]["title"] == "设计端点"


# 函数说明：test_create_writes_pretty_printed_json
# 用途：回归验证回归测试与测试辅助中的 `create_writes_pretty_printed_json` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `task_file.read_text`
# 。
# 分支与异常：
#   验证条件：`'\n ' in content`。
# 副作用与资源：
#   文件或资源访问：`task_file.read_text`。
async def test_create_writes_pretty_printed_json(store: FileTaskStore) -> None:
    task = await store.create(title="可读性", owner_conversation_id=_OWNER)
    task_file = store.tasks_dir / f"{task.id}.json"
    content = task_file.read_text(encoding="utf-8")
    assert "\n  " in content


# 函数说明：test_normalizes_whitespace_and_deduplicates
# 用途：回归验证回归测试与测试辅助中的 `normalizes_whitespace_and_deduplicates` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` →
# `store.add_constraints`。
# 分支与异常：
#   验证条件：`task.title == '压缩 目标'`。
#   验证条件：`task.goal == '完成 压缩'`。
#   验证条件：`updated.constraints == ('只读', '安全优先')`。
async def test_normalizes_whitespace_and_deduplicates(store: FileTaskStore) -> None:
    task = await store.create(
        owner_conversation_id=_OWNER,
        title="  压缩  目标  ",
        goal="  完成  压缩  ",
    )
    assert task.title == "压缩 目标"
    assert task.goal == "完成 压缩"

    updated = await store.add_constraints(task.id, "  只读  ", "只读", "安全优先")
    assert updated.constraints == ("只读", "安全优先")


# 函数说明：test_resolve_by_id_prefix
# 用途：回归验证回归测试与测试辅助中的 `resolve_by_id_prefix` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.resolve`。
# 分支与异常：
#   验证条件：`resolved is not None and resolved.id == task.id`。
async def test_resolve_by_id_prefix(store: FileTaskStore) -> None:
    task = await store.create(title="前缀测试", owner_conversation_id=_OWNER)
    resolved = await store.resolve(task.id[:8], owner_conversation_id=_OWNER)
    assert resolved is not None and resolved.id == task.id


# 函数说明：test_resolve_ambiguous_prefix_raises
# 用途：回归验证回归测试与测试辅助中的 `resolve_ambiguous_prefix_raises` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`f'{common}{suffix}'.ljust` →
# `path.write_text` → `json.dumps` → `pytest.raises` → `store.resolve`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='前缀不唯一')`。
# 副作用与资源：
#   文件或资源访问：`path.write_text`。
async def test_resolve_ambiguous_prefix_raises(store: FileTaskStore) -> None:

    common = "abcd1234"
    for suffix, title in (("aa", "任务 A"), ("bb", "任务 B")):
        task_id = f"{common}{suffix}".ljust(32, "0")
        path = store.tasks_dir / f"{task_id}.json"
        path.write_text(
            json.dumps(
                {
                    "id": task_id,
                    "title": title,
                    "status": "pending",
                    "priority": "normal",
                    "constraints": [],
                    "state": [],
                    "key_facts": [],
                    "steps": [],
                    "owner_conversation_id": _OWNER,
                    "run_ids": [],
                    "created_at": "2026-08-06T00:00:00+00:00",
                    "updated_at": "2026-08-06T00:00:00+00:00",
                    "completed_at": None,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="前缀不唯一"):
        await store.resolve(common, owner_conversation_id=_OWNER)


# 函数说明：test_status_lifecycle_sets_and_clears_completed_at
# 用途：回归验证回归测试与测试辅助中的 `status_lifecycle_sets_and_clears_completed_at`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.set_status` →
# `pytest.raises`。
# 分支与异常：
#   验证条件：`active.completed_at is None`。
#   验证条件：`completed.status is TaskStatus.COMPLETED`。
#   验证条件：`completed.completed_at is not None`。
#   预期异常：`pytest.raises(ValueError, match='terminal task')`。
async def test_status_lifecycle_sets_and_clears_completed_at(
    store: FileTaskStore,
) -> None:
    task = await store.create(title="生命周期", owner_conversation_id=_OWNER)
    active = await store.set_status(task.id, TaskStatus.ACTIVE)
    assert active.completed_at is None

    completed = await store.set_status(task.id, TaskStatus.COMPLETED)
    assert completed.status is TaskStatus.COMPLETED
    assert completed.completed_at is not None

    with pytest.raises(ValueError, match="terminal task"):
        await store.set_status(task.id, TaskStatus.ACTIVE)


# 函数说明：test_step_status_advance
# 用途：回归验证回归测试与测试辅助中的 `step_status_advance` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `store.set_step_status` → `next` → `pytest.raises`。
# 分支与异常：
#   验证条件：`step.status is TaskStepStatus.IN_PROGRESS`。
#   验证条件：`step.note == '开始'`。
#   预期异常：`pytest.raises(KeyError, match='步骤不存在')`。
async def test_step_status_advance(store: FileTaskStore) -> None:
    task = await store.create(
        owner_conversation_id=_OWNER,
        title="步骤推进",
        steps=(
            TaskStep(id="s1", title="步骤一"),
            TaskStep(id="s2", title="步骤二"),
        ),
    )

    updated = await store.set_step_status(
        task.id,
        "s1",
        TaskStepStatus.IN_PROGRESS,
        note="开始",
    )
    step = next(step for step in updated.steps if step.id == "s1")
    assert step.status is TaskStepStatus.IN_PROGRESS
    assert step.note == "开始"

    with pytest.raises(KeyError, match="步骤不存在"):
        await store.set_step_status(
            task.id,
            "missing",
            TaskStepStatus.DONE,
            note="不存在",
        )


# 函数说明：test_replace_steps
# 用途：回归验证回归测试与测试辅助中的 `replace_steps` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.replace_steps`
#  → `TaskStep`。
# 分支与异常：
#   验证条件：`[step.id for step in updated.steps] == ['x']`。
#   验证条件：`updated.steps[0].status is TaskStepStatus.DONE`。
async def test_replace_steps(store: FileTaskStore) -> None:
    task = await store.create(title="重排步骤", owner_conversation_id=_OWNER)
    updated = await store.replace_steps(
        task.id,
        (
            TaskStep(
                id="x",
                title="新步骤",
                status=TaskStepStatus.DONE,
                note="已经完成",
            ),
        ),
    )
    assert [step.id for step in updated.steps] == ["x"]
    assert updated.steps[0].status is TaskStepStatus.DONE


# 函数说明：test_update_goal_state_and_key_facts
# 用途：回归验证回归测试与测试辅助中的 `update_goal_state_and_key_facts` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.update_goal` →
#  `store.update_state` → `store.add_key_facts`。
# 分支与异常：
#   验证条件：`goal_updated.goal == '新目标'`。
#   验证条件：`state_updated.state == ('设计完成', '开始实现')`。
#   验证条件：`facts_updated.key_facts == ('使用 pydantic v2',)`。
async def test_update_goal_state_and_key_facts(store: FileTaskStore) -> None:
    task = await store.create(title="目标更新", owner_conversation_id=_OWNER)
    goal_updated = await store.update_goal(task.id, "新目标")
    assert goal_updated.goal == "新目标"

    state_updated = await store.update_state(task.id, "设计完成", "开始实现")
    assert state_updated.state == ("设计完成", "开始实现")

    facts_updated = await store.add_key_facts(task.id, "使用 pydantic v2")
    assert facts_updated.key_facts == ("使用 pydantic v2",)


# 函数说明：test_owner_is_fixed_and_run_can_be_attached
# 用途：回归验证回归测试与测试辅助中的 `owner_is_fixed_and_run_can_be_attached` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.attach_run`。
# 分支与异常：
#   验证条件：`with_run.owner_conversation_id == 'conv-1'`。
#   验证条件：`with_run.run_ids == ('run-1',)`。
#   验证条件：`duplicated.run_ids == ('run-1',)`。
async def test_owner_is_fixed_and_run_can_be_attached(store: FileTaskStore) -> None:
    task = await store.create(title="关联", owner_conversation_id="conv-1")
    with_run = await store.attach_run(task.id, "run-1")
    assert with_run.owner_conversation_id == "conv-1"
    assert with_run.run_ids == ("run-1",)

    duplicated = await store.attach_run(with_run.id, "run-1")
    assert duplicated.run_ids == ("run-1",)


# 函数说明：test_list_filters_by_status_and_orders_by_update
# 用途：回归验证回归测试与测试辅助中的 `list_filters_by_status_and_orders_by_update` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.set_status` →
# `store.list`。
# 分支与异常：
#   验证条件：`{task.title for task in actives} == {'进行中', '进行中二'}`。
#   验证条件：`all_tasks[0].title == '已完成'`。
#   验证条件：`pending.id in {task.id for task in all_tasks}`。
async def test_list_filters_by_status_and_orders_by_update(
    store: FileTaskStore,
) -> None:
    pending = await store.create(title="待办", owner_conversation_id=_OWNER)
    in_progress = await store.create(title="进行中", owner_conversation_id=_OWNER)
    await store.set_status(in_progress.id, TaskStatus.ACTIVE)
    active = await store.create(title="进行中二", owner_conversation_id=_OWNER)
    await store.set_status(active.id, TaskStatus.ACTIVE)
    completed = await store.create(title="已完成", owner_conversation_id=_OWNER)
    await store.set_status(completed.id, TaskStatus.COMPLETED)

    actives = await store.list(status=TaskStatus.ACTIVE)
    assert {task.title for task in actives} == {"进行中", "进行中二"}

    all_tasks = await store.list(limit=10)
    assert all_tasks[0].title == "已完成"
    assert pending.id in {task.id for task in all_tasks}


# 函数说明：test_list_filters_by_conversation
# 用途：回归验证回归测试与测试辅助中的 `list_filters_by_conversation` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.list`。
# 分支与异常：
#   验证条件：`[task.id for task in in_a] == [task_a.id]`。
#   验证条件：`[task.id for task in in_b] == [task_b.id]`。
#   验证条件：`{task.id for task in all_tasks} == {task_a.id, task_b.id}`。
async def test_list_filters_by_conversation(store: FileTaskStore) -> None:
    task_a = await store.create(title="A 任务", owner_conversation_id="conv-a")
    task_b = await store.create(title="B 任务", owner_conversation_id="conv-b")

    in_a = await store.list(owner_conversation_id="conv-a")
    assert [task.id for task in in_a] == [task_a.id]

    in_b = await store.list(owner_conversation_id="conv-b")
    assert [task.id for task in in_b] == [task_b.id]

    all_tasks = await store.list(limit=10)
    assert {task.id for task in all_tasks} == {
        task_a.id,
        task_b.id,
    }


# 函数说明：test_list_skips_corrupt_files
# 用途：回归验证回归测试与测试辅助中的 `list_skips_corrupt_files` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `corrupt.write_text`
# → `store.list`。
# 分支与异常：
#   验证条件：`[task.title for task in tasks] == ['正常任务']`。
# 副作用与资源：
#   文件或资源访问：`corrupt.write_text`。
async def test_list_skips_corrupt_files(store: FileTaskStore) -> None:
    await store.create(title="正常任务", owner_conversation_id=_OWNER)
    corrupt = store.tasks_dir / "corrupt.json"
    corrupt.write_text("{ not valid json", encoding="utf-8")

    tasks = await store.list(limit=10)
    assert [task.title for task in tasks] == ["正常任务"]


# 函数说明：test_delete
# 用途：回归验证回归测试与测试辅助中的 `delete` 场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.delete`。
# 分支与异常：
#   验证条件：`await store.delete(task.id) is True`。
#   验证条件：`await store.get(task.id) is None`。
#   验证条件：`await store.delete(task.id) is False`。
async def test_delete(store: FileTaskStore) -> None:
    task = await store.create(title="删除", owner_conversation_id=_OWNER)
    assert await store.delete(task.id) is True
    assert await store.get(task.id) is None
    assert await store.delete(task.id) is False


# 函数说明：test_missing_task_raises_key_error
# 用途：回归验证回归测试与测试辅助中的 `missing_task_raises_key_error` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `store.update_goal`
# 。
# 分支与异常：
#   预期异常：`pytest.raises(KeyError, match='任务不存在')`。
async def test_missing_task_raises_key_error(store: FileTaskStore) -> None:
    with pytest.raises(KeyError, match="任务不存在"):
        await store.update_goal("0" * 32, "x")


# 函数说明：test_progress_summary
# 用途：回归验证回归测试与测试辅助中的 `progress_summary` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep`。
# 分支与异常：
#   验证条件：`task.progress_summary == '[pending] 进度 (1/2 步骤完成)'`。
async def test_progress_summary(store: FileTaskStore) -> None:
    task = await store.create(
        owner_conversation_id=_OWNER,
        title="进度",
        steps=(
            TaskStep(
                id="a",
                title="A",
                status=TaskStepStatus.DONE,
                note="完成",
            ),
            TaskStep(id="b", title="B"),
        ),
    )
    assert task.progress_summary == "[pending] 进度 (1/2 步骤完成)"


# 函数说明：test_rejects_path_traversal_and_absolute_identifiers
# 用途：回归验证回归测试与测试辅助中的 `rejects_path_traversal_and_absolute_identifiers`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`；读取键 `../outside`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `store.delete` →
# `store.resolve`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='task_id')`。
#   预期异常：`pytest.raises(ValueError, match='identifier')`。
async def test_rejects_path_traversal_and_absolute_identifiers(
    store: FileTaskStore,
) -> None:
    with pytest.raises(ValueError, match="task_id"):
        await store.get("../outside")
    with pytest.raises(ValueError, match="task_id"):
        await store.delete("/tmp/outside")
    with pytest.raises(ValueError, match="identifier"):
        await store.resolve("../../")


# 函数说明：test_rejects_symlinked_task_file
# 用途：回归验证回归测试与测试辅助中的 `rejects_symlinked_task_file` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `task_file.replace` →
#  `task_file.symlink_to` → `pytest.skip`。
# 分支与异常：
#   捕获 `OSError` 后，执行异常处理调用 `pytest.skip`。
#   验证条件：`await store.get(task.id) is None`。
async def test_rejects_symlinked_task_file(store: FileTaskStore, tmp_path) -> None:
    task = await store.create(title="外部文件", owner_conversation_id=_OWNER)
    task_file = store.tasks_dir / f"{task.id}.json"
    external = tmp_path / "external.json"
    task_file.replace(external)
    try:
        task_file.symlink_to(external)
    except OSError:
        pytest.skip("当前 Windows 账户没有创建符号链接的权限")

    assert await store.get(task.id) is None


# 函数说明：test_concurrent_updates_do_not_lose_facts
# 用途：回归验证回归测试与测试辅助中的 `concurrent_updates_do_not_lose_facts` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `asyncio.gather` →
# `store.add_key_facts`。
# 分支与异常：
#   验证条件：`updated is not None`。
#   验证条件：`set(updated.key_facts) == {'事实 A', '事实 B'}`。
#   验证条件：`updated.revision == 3`。
async def test_concurrent_updates_do_not_lose_facts(store: FileTaskStore) -> None:
    task = await store.create(title="并发更新", owner_conversation_id=_OWNER)

    await asyncio.gather(
        store.add_key_facts(task.id, "事实 A"),
        store.add_key_facts(task.id, "事实 B"),
    )

    updated = await store.get(task.id)
    assert updated is not None
    assert set(updated.key_facts) == {"事实 A", "事实 B"}
    assert updated.revision == 3


# 函数说明：test_revision_conflict_does_not_overwrite_task
# 用途：回归验证回归测试与测试辅助中的 `revision_conflict_does_not_overwrite_task` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.apply_patch` →
#  `TaskPatch` → `pytest.raises`。
# 分支与异常：
#   验证条件：`current == updated`。
#   预期异常：`pytest.raises(ValueError, match='revision conflict')`。
async def test_revision_conflict_does_not_overwrite_task(
    store: FileTaskStore,
) -> None:
    task = await store.create(title="版本检查", owner_conversation_id=_OWNER)
    updated = await store.apply_patch(
        task.id,
        TaskPatch(goal="第一版", expected_revision=1),
    )

    with pytest.raises(ValueError, match="revision conflict"):
        await store.apply_patch(
            task.id,
            TaskPatch(goal="过期覆盖", expected_revision=1),
        )

    current = await store.get(task.id)
    assert current == updated


# 函数说明：test_active_task_is_latest_active_or_paused_for_conversation
# 用途：回归验证回归测试与测试辅助中的
# `active_task_is_latest_active_or_paused_for_conversation` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.set_status` →
# `store.active_for_conversation` → `store.plan_accept`。
# 分支与异常：
#   验证条件：`await store.active_for_conversation('conv-1') is None`。
#   验证条件：`active is not None`。
#   验证条件：`active.id == second.id`。
#   验证条件：`active.status is TaskStatus.ACTIVE`。
async def test_active_task_is_latest_active_or_paused_for_conversation(
    store: FileTaskStore,
) -> None:
    first = await store.create(
        title="较早任务",
        owner_conversation_id="conv-1",
    )
    second = await store.create(
        title="当前任务",
        owner_conversation_id="conv-1",
    )
    await store.set_status(first.id, TaskStatus.COMPLETED)

    assert await store.active_for_conversation("conv-1") is None

    await store.plan_accept(second.id)
    active = await store.active_for_conversation("conv-1")
    assert active is not None
    assert active.id == second.id
    assert active.status is TaskStatus.ACTIVE

    pending_only = await store.create(
        title="未接受计划",
        owner_conversation_id="conv-2",
    )
    assert pending_only.status is TaskStatus.PENDING
    assert await store.active_for_conversation("conv-2") is None


# 函数说明：test_task_context_provider_reads_only_current_owner
# 用途：回归验证回归测试与测试辅助中的 `task_context_provider_reads_only_current_owner`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.plan_accept` →
#  `TaskContextProvider` → `provider.message_for`。
# 分支与异常：
#   验证条件：`message_a is not None and task_a.id in (message_a.content or '')`。
#   验证条件：`task_b.id not in (message_a.content or '')`。
#   验证条件：`message_b is not None and task_b.id in (message_b.content or '')`。
#   验证条件：`task_a.id not in (message_b.content or '')`。
async def test_task_context_provider_reads_only_current_owner(
    store: FileTaskStore,
) -> None:
    task_a = await store.create(
        title="A 私有任务",
        owner_conversation_id="conv-a",
    )
    task_b = await store.create(
        title="B 私有任务",
        owner_conversation_id="conv-b",
    )
    await store.plan_accept(task_a.id)
    await store.plan_accept(task_b.id)
    provider = TaskContextProvider(store)

    message_a = await provider.message_for("conv-a")
    message_b = await provider.message_for("conv-b")

    assert message_a is not None and task_a.id in (message_a.content or "")
    assert task_b.id not in (message_a.content or "")
    assert message_b is not None and task_b.id in (message_b.content or "")
    assert task_a.id not in (message_b.content or "")
    assert await provider.message_for(None) is None


# 函数说明：test_pending_task_not_injected_as_active_context
# 用途：回归验证回归测试与测试辅助中的 `pending_task_not_injected_as_active_context` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskContextProvider`
#  → `provider.message_for` → `store.plan_accept`。
# 分支与异常：
#   验证条件：`task.status is TaskStatus.PENDING`。
#   验证条件：`await provider.message_for('conv-a') is None`。
#   验证条件：`message is not None and task.id in (message.content or '')`。
async def test_pending_task_not_injected_as_active_context(
    store: FileTaskStore,
) -> None:

    task = await store.create(
        title="未接受计划",
        owner_conversation_id="conv-a",
    )
    assert task.status is TaskStatus.PENDING
    provider = TaskContextProvider(store)

    assert await provider.message_for("conv-a") is None

    await store.plan_accept(task.id)
    message = await provider.message_for("conv-a")
    assert message is not None and task.id in (message.content or "")


# 函数说明：test_task_context_folds_old_done_steps_but_keeps_working_steps
# 用途：回归验证回归测试与测试辅助中的
# `task_context_folds_old_done_steps_but_keeps_working_steps` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `store.plan_accept` → `TaskContextProvider` → `provider.message_for`。
# 分支与异常：
#   验证条件：`message is not None`。
#   验证条件：`'"omitted_done_steps":3' in content`。
#   验证条件：`'"omitted_pending_steps":0' in content`。
#   验证条件：`'"step-3"' in content`。
async def test_task_context_folds_old_done_steps_but_keeps_working_steps(
    store: FileTaskStore,
) -> None:
    task = await store.create(
        title="长任务",
        owner_conversation_id="conv-a",
        steps=tuple(
            TaskStep(
                id=f"step-{index}",
                title=f"步骤 {index}",
                status=TaskStepStatus.DONE,
                note=f"证据 {index}",
            )
            for index in range(5)
        )
        + (TaskStep(id="step-next", title="下一步"),),
    )
    await store.plan_accept(task.id)
    provider = TaskContextProvider(store, recent_done_steps=2)

    message = await provider.message_for("conv-a")

    assert message is not None
    content = message.content or ""
    assert '"omitted_done_steps":3' in content
    assert '"omitted_pending_steps":0' in content
    assert '"step-3"' in content
    assert '"step-4"' in content
    assert '"step-next"' in content
    assert '"step-0"' not in content
    assert "in_progress" in content
    assert task.id in content


# 函数说明：test_task_context_keeps_late_active_step_and_acceptance
# 用途：回归验证回归测试与测试辅助中的
# `task_context_keeps_late_active_step_and_acceptance` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `store.plan_accept` → `TaskContextProvider(store).message_for` → `TaskContextProvider`
#  → `json.loads`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`message is not None`。
#   验证条件：`active['acceptance'] == '运行测试且返回零'`。
#   验证条件：`len(payload['steps']) == 12`。
#   验证条件：`payload['omitted_pending_steps'] == 3`。
async def test_task_context_keeps_late_active_step_and_acceptance(
    store: FileTaskStore,
) -> None:
    task = await store.create(
        title="保留正在执行的步骤",
        owner_conversation_id="conv-many-steps",
        steps=tuple(TaskStep(id=f"s{i}", title=f"步骤 {i}") for i in range(14))
        + (
            TaskStep(
                id="active-last",
                title="正在执行",
                status=TaskStepStatus.IN_PROGRESS,
                acceptance="运行测试且返回零",
            ),
        ),
    )
    await store.plan_accept(task.id)
    message = await TaskContextProvider(store).message_for("conv-many-steps")
    assert message is not None
    payload = json.loads(
        (message.content or "").split("<active_task>")[1].split("</active_task>")[0]
    )
    active = next(step for step in payload["steps"] if step["id"] == "active-last")
    assert active["acceptance"] == "运行测试且返回零"
    assert len(payload["steps"]) == 12
    assert payload["omitted_pending_steps"] == 3


# 函数说明：test_resolve_prefix_filters_owner_before_ambiguity
# 用途：回归验证回归测试与测试辅助中的 `resolve_prefix_filters_owner_before_ambiguity`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`f'{common}aa'.ljust` →
# `f'{common}bb'.ljust` → `_write_task_payload` → `store.resolve` → `pytest.raises`。
# 分支与异常：
#   验证条件：`resolved_a is not None and resolved_a.id == task_ids[0]`。
#   验证条件：`resolved_b is not None and resolved_b.id == task_ids[1]`。
#   预期异常：`pytest.raises(ValueError, match='前缀不唯一')`。
async def test_resolve_prefix_filters_owner_before_ambiguity(
    store: FileTaskStore,
) -> None:

    common = "abcd1234"
    task_ids = (
        f"{common}aa".ljust(32, "0"),
        f"{common}bb".ljust(32, "0"),
    )
    for task_id, owner in zip(task_ids, ("conv-a", "conv-b"), strict=True):
        _write_task_payload(
            store,
            task_id=task_id,
            owner_conversation_id=owner,
        )

    resolved_a = await store.resolve(
        common,
        owner_conversation_id="conv-a",
    )
    resolved_b = await store.resolve(
        common,
        owner_conversation_id="conv-b",
    )
    assert resolved_a is not None and resolved_a.id == task_ids[0]
    assert resolved_b is not None and resolved_b.id == task_ids[1]

    with pytest.raises(ValueError, match="前缀不唯一"):
        await store.resolve(common)


# 函数说明：test_legacy_single_owner_is_migrated_atomically
# 用途：回归验证回归测试与测试辅助中的 `legacy_single_owner_is_migrated_atomically` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_write_task_payload` → `json.loads` →
#  `path.read_text`。
# 分支与异常：
#   验证条件：`loaded is not None`。
#   验证条件：`loaded.owner_conversation_id == 'conv-legacy'`。
#   验证条件：`migrated['owner_conversation_id'] == 'conv-legacy'`。
#   验证条件：`'conversation_ids' not in migrated`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
async def test_legacy_single_owner_is_migrated_atomically(
    store: FileTaskStore,
) -> None:
    task_id = "a" * 32
    path = _write_task_payload(
        store,
        task_id=task_id,
        conversation_ids=["conv-legacy"],
    )

    loaded = await store.get(task_id)

    assert loaded is not None
    assert loaded.owner_conversation_id == "conv-legacy"
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["owner_conversation_id"] == "conv-legacy"
    assert "conversation_ids" not in migrated
    assert migrated["revision"] == 1


# 函数说明：test_legacy_ambiguous_owner_warns_and_is_inaccessible
# 用途：回归验证回归测试与测试辅助中的
# `legacy_ambiguous_owner_warns_and_is_inaccessible` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
#   caplog：`caplog`输入或配置值。
#   legacy_owners：`legacy_owners`输入或配置值，类型 `list[str]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_write_task_payload` →
# `path.read_bytes` → `store.resolve` → `store.list`。
# 分支与异常：
#   验证条件：`await store.get(task_id) is None`。
#   验证条件：`await store.resolve(task_id, owner_conversation_id='conv-a') is None`。
#   验证条件：`await store.list(owner_conversation_id='conv-a') == ()`。
#   验证条件：`path.read_bytes() == original`。
# 副作用与资源：
#   文件或资源访问：`path.read_bytes`。
@pytest.mark.parametrize("legacy_owners", [[], ["conv-a", "conv-b"]])
async def test_legacy_ambiguous_owner_warns_and_is_inaccessible(
    store: FileTaskStore,
    caplog,
    legacy_owners: list[str],
) -> None:
    task_id = "b" * 32
    path = _write_task_payload(
        store,
        task_id=task_id,
        conversation_ids=legacy_owners,
    )
    original = path.read_bytes()

    assert await store.get(task_id) is None
    assert (
        await store.resolve(
            task_id,
            owner_conversation_id="conv-a",
        )
        is None
    )
    assert await store.list(owner_conversation_id="conv-a") == ()
    assert path.read_bytes() == original
    assert "ambiguous owner" in caplog.text


# 函数说明：test_task_rejects_multiple_in_progress_steps
# 用途：回归验证回归测试与测试辅助中的 `task_rejects_multiple_in_progress_steps` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `store.create` →
# `TaskStep` → `store.tasks_dir.glob`。
# 分支与异常：
#   验证条件：`list(store.tasks_dir.glob('*.json')) == []`。
#   预期异常：`pytest.raises(ValueError, match='at most one in_progress')`。
async def test_task_rejects_multiple_in_progress_steps(
    store: FileTaskStore,
) -> None:
    with pytest.raises(ValueError, match="at most one in_progress"):
        await store.create(
            title="校验",
            owner_conversation_id=_OWNER,
            steps=(
                TaskStep(id="a", title="A", status=TaskStepStatus.IN_PROGRESS),
                TaskStep(id="b", title="B", status=TaskStepStatus.IN_PROGRESS),
            ),
        )
    assert list(store.tasks_dir.glob("*.json")) == []


# 函数说明：test_done_and_blocked_step_require_note
# 用途：回归验证回归测试与测试辅助中的 `done_and_blocked_step_require_note` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   status：目标状态，类型 `TaskStepStatus`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `TaskStep`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='requires a note')`。
@pytest.mark.parametrize("status", [TaskStepStatus.DONE, TaskStepStatus.BLOCKED])
def test_done_and_blocked_step_require_note(status: TaskStepStatus) -> None:
    with pytest.raises(ValueError, match="requires a note"):
        TaskStep(id="step", title="步骤", status=status)


# 函数说明：test_paused_task_rejects_in_progress_step_without_writing
# 用途：回归验证回归测试与测试辅助中的
# `paused_task_rejects_in_progress_step_without_writing` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `_assert_patch_rejected_without_file_change` → `TaskPatch`。
async def test_paused_task_rejects_in_progress_step_without_writing(
    store: FileTaskStore,
) -> None:
    task = await store.create(
        title="暂停约束",
        owner_conversation_id=_OWNER,
        steps=(TaskStep(id="s1", title="执行", status=TaskStepStatus.IN_PROGRESS),),
    )
    await _assert_patch_rejected_without_file_change(
        store,
        task.id,
        TaskPatch(status=TaskStatus.PAUSED),
        "paused task",
    )


# 函数说明：test_completed_task_requires_every_step_done_without_writing
# 用途：回归验证回归测试与测试辅助中的
# `completed_task_requires_every_step_done_without_writing` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `_assert_patch_rejected_without_file_change` → `TaskPatch`。
async def test_completed_task_requires_every_step_done_without_writing(
    store: FileTaskStore,
) -> None:
    task = await store.create(
        title="完成约束",
        owner_conversation_id=_OWNER,
        steps=(TaskStep(id="s1", title="未完成"),),
    )
    await _assert_patch_rejected_without_file_change(
        store,
        task.id,
        TaskPatch(status=TaskStatus.COMPLETED),
        "all steps to be done",
    )


# 函数说明：test_done_step_cannot_be_rolled_back_without_writing
# 用途：回归验证回归测试与测试辅助中的 `done_step_cannot_be_rolled_back_without_writing`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `_assert_patch_rejected_without_file_change` → `TaskPatch`。
async def test_done_step_cannot_be_rolled_back_without_writing(
    store: FileTaskStore,
) -> None:
    task = await store.create(
        title="完成步骤不可回退",
        owner_conversation_id=_OWNER,
        steps=(
            TaskStep(
                id="s1",
                title="已完成",
                status=TaskStepStatus.DONE,
                note="通过测试",
            ),
        ),
    )
    await _assert_patch_rejected_without_file_change(
        store,
        task.id,
        TaskPatch(step_id="s1", step_status=TaskStepStatus.TODO),
        "cannot be rolled back",
    )


# 函数说明：test_replace_steps_cannot_delete_or_rollback_started_steps
# 用途：回归验证回归测试与测试辅助中的
# `replace_steps_cannot_delete_or_rollback_started_steps` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
#   initial_status：状态输入或配置值，类型 `TaskStepStatus`。
#   replacement：`replacement`输入或配置值，类型 `tuple[TaskStep, ...]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `_assert_patch_rejected_without_file_change` → `TaskPatch`。
@pytest.mark.parametrize(
    ("initial_status", "replacement"),
    [
        (TaskStepStatus.DONE, ()),
        (
            TaskStepStatus.DONE,
            (TaskStep(id="s1", title="回退", status=TaskStepStatus.TODO),),
        ),
        (TaskStepStatus.IN_PROGRESS, ()),
        (
            TaskStepStatus.IN_PROGRESS,
            (TaskStep(id="s1", title="回退", status=TaskStepStatus.TODO),),
        ),
    ],
)
async def test_replace_steps_cannot_delete_or_rollback_started_steps(
    store: FileTaskStore,
    initial_status: TaskStepStatus,
    replacement: tuple[TaskStep, ...],
) -> None:
    note = "已经完成" if initial_status is TaskStepStatus.DONE else None
    task = await store.create(
        title="重排保护",
        owner_conversation_id=_OWNER,
        steps=(TaskStep(id="s1", title="受保护", status=initial_status, note=note),),
    )
    await _assert_patch_rejected_without_file_change(
        store,
        task.id,
        TaskPatch(replace_steps=replacement),
        "replace_steps cannot",
    )


# 函数说明：test_terminal_task_cannot_be_reopened_without_writing
# 用途：回归验证回归测试与测试辅助中的
# `terminal_task_cannot_be_reopened_without_writing` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
#   terminal_status：传给 `store.set_status` 的输入，类型 `TaskStatus`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.set_status` →
# `_assert_patch_rejected_without_file_change` → `TaskPatch`。
@pytest.mark.parametrize(
    "terminal_status",
    [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED],
)
async def test_terminal_task_cannot_be_reopened_without_writing(
    store: FileTaskStore,
    terminal_status: TaskStatus,
) -> None:
    task = await store.create(
        title="终态保护",
        owner_conversation_id=_OWNER,
    )
    terminal = await store.set_status(task.id, terminal_status)
    await _assert_patch_rejected_without_file_change(
        store,
        terminal.id,
        TaskPatch(status=TaskStatus.ACTIVE),
        "terminal task",
    )


# 函数说明：test_replace_steps_and_single_step_update_are_mutually_exclusive
# 用途：回归验证回归测试与测试辅助中的
# `replace_steps_and_single_step_update_are_mutually_exclusive` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `path.read_bytes` → `pytest.raises` → `TaskPatch`。
# 分支与异常：
#   验证条件：`path.read_bytes() == original`。
#   预期异常：`pytest.raises(ValueError, match='cannot be combined')`。
# 副作用与资源：
#   文件或资源访问：`path.read_bytes`。
async def test_replace_steps_and_single_step_update_are_mutually_exclusive(
    store: FileTaskStore,
) -> None:
    task = await store.create(
        title="更新互斥",
        owner_conversation_id=_OWNER,
        steps=(TaskStep(id="s1", title="步骤"),),
    )
    path = store.tasks_dir / f"{task.id}.json"
    original = path.read_bytes()

    with pytest.raises(ValueError, match="cannot be combined"):
        TaskPatch(
            replace_steps=task.steps,
            step_id="s1",
            step_status=TaskStepStatus.IN_PROGRESS,
        )
    assert path.read_bytes() == original


# 函数说明：_assert_patch_rejected_without_file_change
# 用途：执行 `_assert_patch_rejected_without_file_change` 测试辅助流程并检查预期结果。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
#   task_id：目标任务标识，类型 `str`。
#   patch：传给 `store.apply_patch` 的输入，类型 `TaskPatch`。
#   message：单条消息或通知，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.read_bytes` → `pytest.raises` →
# `store.apply_patch`。
# 分支与异常：
#   验证条件：`path.read_bytes() == original`。
#   预期异常：`pytest.raises(ValueError, match=message)`。
# 副作用与资源：
#   文件或资源访问：`path.read_bytes`。
async def _assert_patch_rejected_without_file_change(
    store: FileTaskStore,
    task_id: str,
    patch: TaskPatch,
    message: str,
) -> None:
    path = store.tasks_dir / f"{task_id}.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match=message):
        await store.apply_patch(task_id, patch)
    assert path.read_bytes() == original


# 函数说明：_write_task_payload
# 用途：写入任务载荷，供回归测试与测试辅助使用。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
#   task_id：目标任务标识，类型 `str`。
#   owner_conversation_id：记录所属会话标识，类型 `str | None`；默认 `None`。
#   conversation_ids：会话输入或配置值，类型 `list[str] | None`；默认 `None`。
# 返回：返回 `path`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `datetime(2026, 8, 6, tzinfo=UTC).isoformat` → `datetime` → `path.write_text` →
# `json.dumps`。
# 副作用与资源：
#   文件或资源访问：`path.write_text`。
def _write_task_payload(
    store: FileTaskStore,
    *,
    task_id: str,
    owner_conversation_id: str | None = None,
    conversation_ids: list[str] | None = None,
):
    payload = {
        "id": task_id,
        "title": "兼容任务",
        "status": "pending",
        "priority": "normal",
        "constraints": [],
        "state": [],
        "key_facts": [],
        "steps": [],
        "run_ids": [],
        "created_at": datetime(2026, 8, 6, tzinfo=UTC).isoformat(),
        "updated_at": datetime(2026, 8, 6, tzinfo=UTC).isoformat(),
        "completed_at": None,
        "revision": 1,
    }
    if owner_conversation_id is not None:
        payload["owner_conversation_id"] = owner_conversation_id
    else:
        payload["conversation_ids"] = conversation_ids or []
    path = store.tasks_dir / f"{task_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("winerror", [5, 32, 33, None])
@pytest.mark.parametrize("persistent", [False, True])
async def test_atomic_replace_failures_preserve_task(
    store: FileTaskStore, monkeypatch, winerror, persistent,
) -> None:
    from app.domain.task import store as task_store

    task = await store.create(owner_conversation_id=_OWNER, title="original")
    path = store.tasks_dir / f"{task.id}.json"
    original = path.read_bytes()
    replace = task_store.os.replace
    calls = []
    delays = []
    error = PermissionError("task file is busy")
    if winerror is not None:
        error.winerror = winerror

    def flaky_replace(source, target):
        calls.append(target)
        if persistent or len(calls) == 1:
            raise error
        replace(source, target)

    monkeypatch.setattr(task_store.os, "replace", flaky_replace)
    monkeypatch.setattr(task_store.time, "sleep", delays.append)
    if persistent or winerror is None:
        with pytest.raises(PermissionError) as caught:
            await store._write(task.model_copy(update={"title": "updated"}))
        assert caught.value is error
        assert path.read_bytes() == original
        assert len(calls) == (4 if winerror is not None else 1)
    else:
        await store._write(task.model_copy(update={"title": "updated"}))
        assert (await store.get(task.id)).title == "updated"
        assert len(calls) == 2
    assert len(delays) == len(calls) - 1
    assert not list(store.tasks_dir.glob("*.tmp"))
