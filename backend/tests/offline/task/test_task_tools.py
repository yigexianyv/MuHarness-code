
from __future__ import annotations

import json

import pytest

from app.domain.task import (
    FileTaskStore,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
    TaskUpdateTool,
    register_task_tools,
)
from app.models.types import ToolCall
from app.tools import ToolExecutionContext, ToolExecutor, ToolRegistry

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


# 函数说明：_registry
# 用途：处理回归测试与测试辅助中的 `_registry` 数据；结果及边界条件见下方说明。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `ToolRegistry`；返回 `registry`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `register_task_tools`
# 。
def _registry(store: FileTaskStore) -> ToolRegistry:
    registry = ToolRegistry()
    register_task_tools(registry, store)
    return registry


# 函数说明：_context
# 用途：处理回归测试与测试辅助中的 `_context` 数据；结果及边界条件见下方说明。
# 参数：
#   tool_name：工具名称，类型 `str`。
#   conversation_id：目标会话标识，类型 `str`；默认 `_OWNER`。
#   run_id：目标运行标识，类型 `str`；默认 `'run-test'`。
# 返回：类型 `ToolExecutionContext`；返回
# `ToolExecutionContext(tool_call=call, conversation_id=conversation_id, run_id=run_id)`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `ToolExecutionContext`。
def _context(
    tool_name: str,
    *,
    conversation_id: str = _OWNER,
    run_id: str = "run-test",
) -> ToolExecutionContext:
    call = ToolCall(id=f"{tool_name}-call", name=tool_name, arguments={})
    return ToolExecutionContext(
        tool_call=call,
        conversation_id=conversation_id,
        run_id=run_id,
    )


# 函数说明：_execute
# 用途：执行回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   tool：目标工具实例。
#   arguments：工具调用的参数对象或 JSON 文本。
#   conversation_id：目标会话标识，类型 `str`；默认 `_OWNER`。
# 返回：返回 `await tool.execute_with_context(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`tool.execute_with_context` →
# `_context`。
async def _execute(
    tool,
    arguments,
    *,
    conversation_id: str = _OWNER,
):
    return await tool.execute_with_context(
        arguments,
        _context(tool.definition.name, conversation_id=conversation_id),
    )


# 函数说明：test_registers_four_task_tools
# 用途：回归验证回归测试与测试辅助中的 `registers_four_task_tools` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` → `registry.names` →
# `registry.definitions` → `definition.permission.model_visible`。
# 分支与异常：
#   验证条件：
# `set(registry.names()) == {'task_create', 'task_update', 'task_get', 'task_list'}`。
#   验证条件：`{definition.name for definition in definitions} == set(registry.names())`
# 。
#   验证条件：
# `all((definition.permission.model_visible() for definition in definitions))`。
async def test_registers_four_task_tools(store: FileTaskStore) -> None:
    registry = _registry(store)
    assert set(registry.names()) == {
        "task_create",
        "task_update",
        "task_get",
        "task_list",
    }
    definitions = registry.definitions(for_model=True)
    assert {definition.name for definition in definitions} == set(registry.names())
    assert all(
        definition.permission.model_visible() for definition in definitions
    )


# 函数说明：test_task_create_with_steps
# 用途：回归验证回归测试与测试辅助中的 `task_create_with_steps` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskCreateTool` → `_execute` →
# `(store.tasks_dir / f"{result['id']}.json").is_file`。
# 分支与异常：
#   验证条件：`result['title'] == '构建 API 层'`。
#   验证条件：`result['goal'] == '完成 /chat SSE 端点'`。
#   验证条件：`result['priority'] == 'high'`。
#   验证条件：`result['status'] == 'pending'`。
async def test_task_create_with_steps(store: FileTaskStore) -> None:
    tool = TaskCreateTool(store)
    result = await _execute(
        tool,
        {
            "title": "构建 API 层",
            "goal": "完成 /chat SSE 端点",
            "priority": "high",
            "steps": [
                {"title": "设计端点"},
                {"title": "实现 SSE", "note": "复用事件队列"},
            ],
        }
    )
    assert result["title"] == "构建 API 层"
    assert result["goal"] == "完成 /chat SSE 端点"
    assert result["priority"] == "high"
    assert result["status"] == "pending"
    assert len(result["steps"]) == 2
    assert result["steps"][0]["title"] == "设计端点"
    assert result["steps"][0]["status"] == "todo"

    loaded = await store.get(result["id"])
    assert loaded is not None and loaded.title == "构建 API 层"
    assert (store.tasks_dir / f"{result['id']}.json").is_file()


# 函数说明：test_task_create_requires_title
# 用途：回归验证回归测试与测试辅助中的 `task_create_requires_title` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskCreateTool` → `pytest.raises` →
# `_execute`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='title')`。
async def test_task_create_requires_title(store: FileTaskStore) -> None:
    tool = TaskCreateTool(store)
    with pytest.raises(ValueError, match="title"):
        await _execute(tool, {})


# 函数说明：test_task_create_automatically_binds_execution_context
# 用途：回归验证回归测试与测试辅助中的
# `task_create_automatically_binds_execution_context` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` → `ToolCall` →
# `ToolExecutor(registry).execute` → `ToolExecutor` → `ToolExecutionContext` →
# `store.list`。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`len(tasks) == 1`。
#   验证条件：`tasks[0].owner_conversation_id == 'conv-current'`。
#   验证条件：`tasks[0].run_ids == ('run-current',)`。
async def test_task_create_automatically_binds_execution_context(
    store: FileTaskStore,
) -> None:
    registry = _registry(store)
    call = ToolCall(
        id="create-1",
        name="task_create",
        arguments={"title": "长任务", "goal": "完成全部步骤"},
    )

    result = await ToolExecutor(registry).execute(
        call,
        context=ToolExecutionContext(
            tool_call=call,
            conversation_id="conv-current",
            run_id="run-current",
        ),
    )

    assert result.success is True
    tasks = await store.list()
    assert len(tasks) == 1
    assert tasks[0].owner_conversation_id == "conv-current"
    assert tasks[0].run_ids == ("run-current",)


# 函数说明：test_task_update_advances_step
# 用途：回归验证回归测试与测试辅助中的 `task_update_advances_step` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `TaskStep` → `_execute` → `next`。
# 分支与异常：
#   验证条件：`step['status'] == 'done'`。
#   验证条件：`step['note'] == '已完成'`。
async def test_task_update_advances_step(store: FileTaskStore) -> None:
    tool = TaskUpdateTool(store)
    created = await store.create(
        owner_conversation_id=_OWNER,
        title="任务",
        steps=(
            TaskStep(id="s1", title="步骤一"),
            TaskStep(id="s2", title="步骤二"),
        ),
    )
    result = await _execute(
        tool,
        {
            "task_id": created.id,
            "step_id": "s1",
            "step_status": "done",
            "step_note": "已完成",
        }
    )
    step = next(step for step in result["steps"] if step["id"] == "s1")
    assert step["status"] == "done"
    assert step["note"] == "已完成"


# 函数说明：test_task_update_done_requires_note
# 用途：回归验证回归测试与测试辅助中的 `task_update_done_requires_note` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `TaskStep` → `pytest.raises` → `_execute`。
# 分支与异常：
#   验证条件：`unchanged is not None`。
#   验证条件：`unchanged.steps[0].status is TaskStepStatus.TODO`。
#   预期异常：`pytest.raises(ValueError, match='step_note')`。
async def test_task_update_done_requires_note(store: FileTaskStore) -> None:

    tool = TaskUpdateTool(store)
    created = await store.create(
        owner_conversation_id=_OWNER,
        title="完成依据",
        steps=(TaskStep(id="s1", title="步骤一"),),
    )

    with pytest.raises(ValueError, match="step_note"):
        await _execute(
            tool,
            {
                "task_id": created.id,
                "step_id": "s1",
                "step_status": "done",
            }
        )
    with pytest.raises(ValueError, match="step_note"):
        await _execute(
            tool,
            {
                "task_id": created.id,
                "step_id": "s1",
                "step_status": "done",
                "step_note": "   ",
            }
        )

    unchanged = await store.get(created.id)
    assert unchanged is not None
    assert unchanged.steps[0].status is TaskStepStatus.TODO


# 函数说明：test_task_update_in_progress_without_note_allowed
# 用途：回归验证回归测试与测试辅助中的 `task_update_in_progress_without_note_allowed` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `TaskStep` → `_execute`。
# 分支与异常：
#   验证条件：`result['steps'][0]['status'] == 'in_progress'`。
async def test_task_update_in_progress_without_note_allowed(
    store: FileTaskStore,
) -> None:

    tool = TaskUpdateTool(store)
    created = await store.create(
        owner_conversation_id=_OWNER,
        title="执行中",
        steps=(TaskStep(id="s1", title="步骤一"),),
    )

    result = await _execute(
        tool,
        {
            "task_id": created.id,
            "step_id": "s1",
            "step_status": "in_progress",
        }
    )
    assert result["steps"][0]["status"] == "in_progress"


# 函数说明：test_task_update_blocked_requires_note
# 用途：回归验证回归测试与测试辅助中的 `task_update_blocked_requires_note` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `TaskStep` → `pytest.raises` → `_execute`。
# 资源/并发边界：`pytest.raises(ValueError, match='blocked')`，上下文退出时执行相应清理
# 。
# 分支与异常：
#   验证条件：`step['status'] == 'blocked'`。
#   验证条件：`step['note'] == '缺少用户提供的实验结果文件'`。
#   预期异常：`pytest.raises(ValueError, match='blocked')`。
async def test_task_update_blocked_requires_note(store: FileTaskStore) -> None:

    tool = TaskUpdateTool(store)
    created = await store.create(
        owner_conversation_id=_OWNER,
        title="阻塞原因",
        steps=(TaskStep(id="s1", title="步骤一"),),
    )

    with pytest.raises(ValueError, match="blocked"):
        await _execute(
            tool,
            {
                "task_id": created.id,
                "step_id": "s1",
                "step_status": "blocked",
            }
        )
    with pytest.raises(ValueError, match="blocked"):
        await _execute(
            tool,
            {
                "task_id": created.id,
                "step_id": "s1",
                "step_status": "blocked",
                "step_note": "  ",
            }
        )

    result = await _execute(
        tool,
        {
            "task_id": created.id,
            "step_id": "s1",
            "step_status": "blocked",
            "step_note": "缺少用户提供的实验结果文件",
        }
    )
    step = result["steps"][0]
    assert step["status"] == "blocked"
    assert step["note"] == "缺少用户提供的实验结果文件"


# 函数说明：test_task_update_can_pause_and_resume
# 用途：回归验证回归测试与测试辅助中的 `task_update_can_pause_and_resume` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `_execute`。
# 分支与异常：
#   验证条件：`paused['status'] == 'paused'`。
#   验证条件：`resumed['status'] == 'active'`。
async def test_task_update_can_pause_and_resume(store: FileTaskStore) -> None:

    tool = TaskUpdateTool(store)
    created = await store.create(
        title="等待输入", owner_conversation_id=_OWNER
    )

    paused = await _execute(
        tool, {"task_id": created.id, "status": "paused"}
    )
    assert paused["status"] == "paused"

    resumed = await _execute(
        tool, {"task_id": created.id, "status": "active"}
    )
    assert resumed["status"] == "active"


# 函数说明：test_task_update_can_replace_plan_and_preserve_existing_step_id
# 用途：回归验证回归测试与测试辅助中的
# `task_update_can_replace_plan_and_preserve_existing_step_id` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `TaskStep` → `_execute`。
# 分支与异常：
#   验证条件：`result['revision'] == 2`。
#   验证条件：`result['steps'][0]['id'] == 'existing'`。
#   验证条件：`result['steps'][0]['status'] == 'in_progress'`。
#   验证条件：`len(result['steps'][1]['id']) == 32`。
async def test_task_update_can_replace_plan_and_preserve_existing_step_id(
    store: FileTaskStore,
) -> None:
    tool = TaskUpdateTool(store)
    created = await store.create(
        owner_conversation_id=_OWNER,
        title="动态计划",
        steps=(TaskStep(id="existing", title="原步骤"),),
    )

    result = await _execute(
        tool,
        {
            "task_id": created.id,
            "expected_revision": created.revision,
            "steps": [
                {
                    "id": "existing",
                    "title": "调整后的步骤",
                    "status": "in_progress",
                },
                {"title": "新增步骤"},
            ],
        }
    )

    assert result["revision"] == 2
    assert result["steps"][0]["id"] == "existing"
    assert result["steps"][0]["status"] == "in_progress"
    assert len(result["steps"][1]["id"]) == 32


# 函数说明：test_task_update_status_goal_state_constraints_facts
# 用途：回归验证回归测试与测试辅助中的 `task_update_status_goal_state_constraints_facts`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `_execute`。
# 分支与异常：
#   验证条件：`result['status'] == 'active'`。
#   验证条件：`result['goal'] == '新目标'`。
#   验证条件：`result['state'] == ['设计完成', '开始实现']`。
#   验证条件：`result['constraints'] == ['只读', '安全优先']`。
async def test_task_update_status_goal_state_constraints_facts(
    store: FileTaskStore,
) -> None:
    tool = TaskUpdateTool(store)
    created = await store.create(
        title="更新测试", owner_conversation_id=_OWNER
    )
    result = await _execute(
        tool,
        {
            "task_id": created.id,
            "status": "active",
            "goal": "新目标",
            "state": ["设计完成", "开始实现"],
            "constraints": ["只读", "安全优先"],
            "facts": ["使用 pydantic v2"],
        }
    )
    assert result["status"] == "active"
    assert result["goal"] == "新目标"
    assert result["state"] == ["设计完成", "开始实现"]
    assert result["constraints"] == ["只读", "安全优先"]
    assert result["key_facts"] == ["使用 pydantic v2"]


# 函数说明：test_task_update_attaches_run_and_conversation
# 用途：回归验证回归测试与测试辅助中的 `task_update_attaches_run_and_conversation` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` → `store.create` →
# `ToolCall` → `ToolExecutor(registry).execute` → `ToolExecutor` →
# `ToolExecutionContext`。
# 分支与异常：
#   验证条件：`executed.success is True`。
#   验证条件：`updated is not None`。
#   验证条件：`result['owner_conversation_id'] == 'conv-1'`。
#   验证条件：`result['run_ids'] == ['run-1']`。
async def test_task_update_attaches_run_and_conversation(
    store: FileTaskStore,
) -> None:
    registry = _registry(store)
    created = await store.create(
        title="关联", owner_conversation_id="conv-1"
    )
    call = ToolCall(
        id="update-1",
        name="task_update",
        arguments={"task_id": created.id, "status": "active"},
    )
    executed = await ToolExecutor(registry).execute(
        call,
        context=ToolExecutionContext(
            tool_call=call,
            conversation_id="conv-1",
            run_id="run-1",
        ),
    )
    assert executed.success is True
    updated = await store.get(created.id)
    assert updated is not None
    result = updated.model_dump(mode="json")
    assert result["owner_conversation_id"] == "conv-1"
    assert result["run_ids"] == ["run-1"]


# 函数说明：test_task_update_requires_update_field
# 用途：回归验证回归测试与测试辅助中的 `task_update_requires_update_field` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `pytest.raises` → `_execute`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='update field')`。
async def test_task_update_requires_update_field(store: FileTaskStore) -> None:
    tool = TaskUpdateTool(store)
    created = await store.create(title="无更新", owner_conversation_id=_OWNER)
    with pytest.raises(ValueError, match="update field"):
        await _execute(tool, {"task_id": created.id})


# 函数说明：test_task_update_step_requires_pair
# 用途：回归验证回归测试与测试辅助中的 `task_update_step_requires_pair` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `pytest.raises` → `_execute`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='together')`。
async def test_task_update_step_requires_pair(store: FileTaskStore) -> None:
    tool = TaskUpdateTool(store)
    created = await store.create(title="成对校验", owner_conversation_id=_OWNER)
    with pytest.raises(ValueError, match="together"):
        await _execute(tool, {"task_id": created.id, "step_id": "s1"})


# 函数说明：test_task_update_is_atomic_when_later_field_is_invalid
# 用途：回归验证回归测试与测试辅助中的
# `task_update_is_atomic_when_later_field_is_invalid` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `store.create` →
# `TaskStep` → `pytest.raises` → `_execute`。
# 分支与异常：
#   验证条件：`unchanged == created`。
#   预期异常：`pytest.raises(ValueError, match='state')`。
async def test_task_update_is_atomic_when_later_field_is_invalid(
    store: FileTaskStore,
) -> None:
    tool = TaskUpdateTool(store)
    created = await store.create(
        owner_conversation_id=_OWNER,
        title="原子更新",
        steps=(TaskStep(id="s1", title="步骤一"),),
    )

    with pytest.raises(ValueError, match="state"):
        await _execute(
            tool,
            {
                "task_id": created.id,
                "step_id": "s1",
                "step_status": "done",
                "state": 123,
            }
        )

    unchanged = await store.get(created.id)
    assert unchanged == created


# 函数说明：test_task_update_missing_task
# 用途：回归验证回归测试与测试辅助中的 `task_update_missing_task` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskUpdateTool` → `pytest.raises` →
# `_execute`。
# 分支与异常：
#   预期异常：`pytest.raises(KeyError, match='任务不存在')`。
async def test_task_update_missing_task(store: FileTaskStore) -> None:
    tool = TaskUpdateTool(store)
    with pytest.raises(KeyError, match="任务不存在"):
        await _execute(tool, {"task_id": "0" * 32, "goal": "x"})


# 函数说明：test_task_get_returns_full_details
# 用途：回归验证回归测试与测试辅助中的 `task_get_returns_full_details` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskGetTool` → `store.create` →
# `TaskStep` → `_execute`。
# 分支与异常：
#   验证条件：`result['id'] == created.id`。
#   验证条件：`result['goal'] == '目标'`。
#   验证条件：`result['steps'][0]['title'] == '步骤一'`。
async def test_task_get_returns_full_details(store: FileTaskStore) -> None:
    tool = TaskGetTool(store)
    created = await store.create(
        owner_conversation_id=_OWNER,
        title="详情",
        steps=(TaskStep(id="s1", title="步骤一"),),
        goal="目标",
    )
    result = await _execute(tool, {"task_id": created.id})
    assert result["id"] == created.id
    assert result["goal"] == "目标"
    assert result["steps"][0]["title"] == "步骤一"


# 函数说明：test_task_get_missing
# 用途：回归验证回归测试与测试辅助中的 `task_get_missing` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskGetTool` → `pytest.raises` →
# `_execute`。
# 分支与异常：
#   预期异常：`pytest.raises(KeyError, match='任务不存在')`。
async def test_task_get_missing(store: FileTaskStore) -> None:
    tool = TaskGetTool(store)
    with pytest.raises(KeyError, match="任务不存在"):
        await _execute(tool, {"task_id": "0" * 32})


# 函数说明：test_task_current_handle_resolves_only_conversation_active_task
# 用途：回归验证回归测试与测试辅助中的
# `task_current_handle_resolves_only_conversation_active_task` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `store.set_status` →
# `TaskGetTool` → `_execute`。
# 分支与异常：
#   验证条件：`result_a['id'] == task_a.id`。
#   验证条件：`result_b['id'] == task_b.id`。
async def test_task_current_handle_resolves_only_conversation_active_task(
    store: FileTaskStore,
) -> None:

    task_a = await store.create(
        title="A 当前任务",
        owner_conversation_id="conv-a",
    )
    task_b = await store.create(
        title="B 当前任务",
        owner_conversation_id="conv-b",
    )
    await store.set_status(task_a.id, TaskStatus.ACTIVE)
    await store.set_status(task_b.id, TaskStatus.ACTIVE)
    tool = TaskGetTool(store)

    result_a = await _execute(
        tool,
        {"task_id": "current"},
        conversation_id="conv-a",
    )
    result_b = await _execute(
        tool,
        {"task_id": "CURRENT"},
        conversation_id="conv-b",
    )

    assert result_a["id"] == task_a.id
    assert result_b["id"] == task_b.id


# 函数说明：test_task_update_current_handle_advances_active_task
# 用途：回归验证回归测试与测试辅助中的 `task_update_current_handle_advances_active_task`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `store.set_status` → `store.set_step_status` → `_execute` → `TaskUpdateTool`。
# 分支与异常：
#   验证条件：`result['id'] == created.id`。
#   验证条件：`result['status'] == 'completed'`。
#   验证条件：`result['steps'][0]['status'] == 'done'`。
async def test_task_update_current_handle_advances_active_task(
    store: FileTaskStore,
) -> None:

    created = await store.create(
        title="当前任务",
        owner_conversation_id=_OWNER,
        steps=(TaskStep(id="s1", title="完成实现"),),
    )
    await store.set_status(created.id, TaskStatus.ACTIVE)
    await store.set_step_status(
        created.id,
        "s1",
        TaskStepStatus.IN_PROGRESS,
    )

    result = await _execute(
        TaskUpdateTool(store),
        {
            "task_id": "current",
            "status": "completed",
            "step_id": "s1",
            "step_status": "done",
            "step_note": "相关测试通过",
        },
    )

    assert result["id"] == created.id
    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "done"


# 函数说明：test_task_current_handle_missing_without_active_task
# 用途：回归验证回归测试与测试辅助中的 `task_current_handle_missing_without_active_task`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `pytest.raises` →
# `_execute` → `TaskGetTool`。
# 分支与异常：
#   预期异常：`pytest.raises(KeyError, match='任务不存在')`。
async def test_task_current_handle_missing_without_active_task(
    store: FileTaskStore,
) -> None:

    await store.create(title="待接受计划", owner_conversation_id=_OWNER)

    with pytest.raises(KeyError, match="任务不存在"):
        await _execute(TaskGetTool(store), {"task_id": "current"})


# 函数说明：test_task_list_filters_and_briefs
# 用途：回归验证回归测试与测试辅助中的 `task_list_filters_and_briefs` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskListTool` → `store.create` →
# `store.set_status` → `_execute` → `briefs['进行中']['progress'].startswith`。
# 分支与异常：
#   验证条件：`all_result['count'] == 2`。
#   验证条件：`set(briefs) == {'进行中', '待办'}`。
#   验证条件：`'steps' not in all_result['tasks'][0]`。
#   验证条件：`briefs['进行中']['progress'].startswith('[active]')`。
async def test_task_list_filters_and_briefs(store: FileTaskStore) -> None:
    tool = TaskListTool(store)
    active = await store.create(title="进行中", owner_conversation_id=_OWNER)
    await store.set_status(active.id, TaskStatus.ACTIVE)
    await store.create(title="待办", owner_conversation_id=_OWNER)

    all_result = await _execute(tool, {})
    assert all_result["count"] == 2
    briefs = {item["title"]: item for item in all_result["tasks"]}
    assert set(briefs) == {"进行中", "待办"}
    assert "steps" not in all_result["tasks"][0]
    assert briefs["进行中"]["progress"].startswith("[active]")

    active_result = await _execute(tool, {"status": "active"})
    assert [item["title"] for item in active_result["tasks"]] == ["进行中"]


# 函数说明：test_task_list_invalid_limit
# 用途：回归验证回归测试与测试辅助中的 `task_list_invalid_limit` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskListTool` → `pytest.raises` →
# `_execute`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='limit')`。
async def test_task_list_invalid_limit(store: FileTaskStore) -> None:
    tool = TaskListTool(store)
    with pytest.raises(ValueError, match="limit"):
        await _execute(tool, {"limit": 0})


# 函数说明：test_task_list_scoped_to_current_conversation
# 用途：回归验证回归测试与测试辅助中的 `task_list_scoped_to_current_conversation` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskListTool` →
# `tool.execute_with_context` → `ToolExecutionContext` → `ToolCall`。
# 分支与异常：
#   验证条件：`[item['title'] for item in result_a['tasks']] == ['A 会话任务']`。
#   验证条件：`[item['title'] for item in result_b['tasks']] == ['B 会话任务']`。
async def test_task_list_scoped_to_current_conversation(
    store: FileTaskStore,
) -> None:

    await store.create(title="A 会话任务", owner_conversation_id="conv-a")
    await store.create(title="B 会话任务", owner_conversation_id="conv-b")
    tool = TaskListTool(store)

    result_a = await tool.execute_with_context(
        {},
        ToolExecutionContext(
            tool_call=ToolCall(id="list-1", name="task_list", arguments={}),
            conversation_id="conv-a",
            run_id="run-a",
        ),
    )
    assert [item["title"] for item in result_a["tasks"]] == ["A 会话任务"]

    result_b = await tool.execute_with_context(
        {},
        ToolExecutionContext(
            tool_call=ToolCall(id="list-2", name="task_list", arguments={}),
            conversation_id="conv-b",
            run_id="run-b",
        ),
    )
    assert [item["title"] for item in result_b["tasks"]] == ["B 会话任务"]


# 函数说明：test_task_get_hidden_from_other_conversation
# 用途：回归验证回归测试与测试辅助中的 `task_get_hidden_from_other_conversation` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskGetTool` →
# `pytest.raises` → `tool.execute_with_context` → `ToolExecutionContext` → `ToolCall`。
# 分支与异常：
#   验证条件：`result['id'] == created.id`。
#   预期异常：`pytest.raises(KeyError, match='任务不存在')`。
async def test_task_get_hidden_from_other_conversation(
    store: FileTaskStore,
) -> None:

    created = await store.create(
        title="A 会话任务", owner_conversation_id="conv-a"
    )
    tool = TaskGetTool(store)

    with pytest.raises(KeyError, match="任务不存在"):
        await tool.execute_with_context(
            {"task_id": created.id},
            ToolExecutionContext(
                tool_call=ToolCall(id="get-1", name="task_get", arguments={}),
                conversation_id="conv-b",
                run_id="run-b",
            ),
        )

    result = await tool.execute_with_context(
        {"task_id": created.id},
        ToolExecutionContext(
            tool_call=ToolCall(id="get-2", name="task_get", arguments={}),
            conversation_id="conv-a",
            run_id="run-a",
        ),
    )
    assert result["id"] == created.id


# 函数说明：test_task_update_rejected_from_other_conversation
# 用途：回归验证回归测试与测试辅助中的 `task_update_rejected_from_other_conversation` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskUpdateTool` →
# `pytest.raises` → `tool.execute_with_context` → `ToolExecutionContext` → `ToolCall`。
# 分支与异常：
#   验证条件：`unchanged is not None and unchanged.status is TaskStatus.PENDING`。
#   预期异常：`pytest.raises(KeyError, match='任务不存在')`。
async def test_task_update_rejected_from_other_conversation(
    store: FileTaskStore,
) -> None:

    created = await store.create(
        title="A 会话任务", owner_conversation_id="conv-a"
    )
    tool = TaskUpdateTool(store)

    with pytest.raises(KeyError, match="任务不存在"):
        await tool.execute_with_context(
            {"task_id": created.id, "status": "completed"},
            ToolExecutionContext(
                tool_call=ToolCall(id="upd-1", name="task_update", arguments={}),
                conversation_id="conv-b",
                run_id="run-b",
            ),
        )

    unchanged = await store.get(created.id)
    assert unchanged is not None and unchanged.status is TaskStatus.PENDING


# 函数说明：test_executor_path_rejects_other_conversation_update
# 用途：回归验证回归测试与测试辅助中的 `executor_path_rejects_other_conversation_update`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` → `store.create` →
# `ToolCall` → `ToolExecutor(registry).execute` → `ToolExecutor` →
# `ToolExecutionContext`。
# 分支与异常：
#   验证条件：`executed.success is False`。
#   验证条件：`'任务不存在' in (executed.error or '')`。
#   验证条件：`unchanged is not None and unchanged.status is TaskStatus.PENDING`。
async def test_executor_path_rejects_other_conversation_update(
    store: FileTaskStore,
) -> None:

    registry = _registry(store)
    created = await store.create(title="A 任务", owner_conversation_id="conv-a")
    call = ToolCall(
        id="upd-exec-1",
        name="task_update",
        arguments={"task_id": created.id, "status": "completed"},
    )

    executed = await ToolExecutor(registry).execute(
        call,
        context=ToolExecutionContext(
            tool_call=call,
            conversation_id="conv-b",
            run_id="run-b",
        ),
    )

    assert executed.success is False
    assert "任务不存在" in (executed.error or "")
    unchanged = await store.get(created.id)
    assert unchanged is not None and unchanged.status is TaskStatus.PENDING


# 函数说明：test_task_tools_reject_missing_conversation_context
# 用途：回归验证回归测试与测试辅助中的 `task_tools_reject_missing_conversation_context`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
#   tool_factory：工具构造工厂。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`tool_factory` → `pytest.raises` →
# `tool.execute`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='conversation context')`。
@pytest.mark.parametrize(
    ("tool_factory", "arguments"),
    [
        (TaskCreateTool, {"title": "无上下文"}),
        (TaskUpdateTool, {"task_id": "0" * 32, "goal": "目标"}),
        (TaskGetTool, {"task_id": "0" * 32}),
        (TaskListTool, {}),
    ],
)
async def test_task_tools_reject_missing_conversation_context(
    store: FileTaskStore,
    tool_factory,
    arguments: dict,
) -> None:

    tool = tool_factory(store)
    with pytest.raises(ValueError, match="conversation context"):
        await tool.execute(arguments)


# 函数说明：test_same_prefix_in_other_conversation_does_not_cause_ambiguity
# 用途：回归验证回归测试与测试辅助中的
# `same_prefix_in_other_conversation_does_not_cause_ambiguity` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `f'{common}aa'.ljust`
#  → `f'{common}bb'.ljust` → `first_path.unlink` → `second_path.unlink` →
# `(store.tasks_dir / f"{first_data['id']}.json").write_text`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`result_a['id'] == first_data['id']`。
#   验证条件：`result_b['id'] == second_data['id']`。
# 副作用与资源：
#   文件或资源访问：`first_path.unlink`、`second_path.unlink`、
# `(store.tasks_dir / f"{first_data['id']}.json").write_text`、
# `(store.tasks_dir / f"{second_data['id']}.json").write_text`。
async def test_same_prefix_in_other_conversation_does_not_cause_ambiguity(
    store: FileTaskStore,
) -> None:

    common = "abcd1234"
    first = await store.create(
        title="A 任务",
        owner_conversation_id="conv-a",
    )
    second = await store.create(
        title="B 任务",
        owner_conversation_id="conv-b",
    )
    first_path = store.tasks_dir / f"{first.id}.json"
    second_path = store.tasks_dir / f"{second.id}.json"
    first_data = first.model_dump(mode="json")
    second_data = second.model_dump(mode="json")
    first_data["id"] = f"{common}aa".ljust(32, "0")
    second_data["id"] = f"{common}bb".ljust(32, "0")
    first_path.unlink()
    second_path.unlink()
    (store.tasks_dir / f"{first_data['id']}.json").write_text(
        json.dumps(first_data, ensure_ascii=False),
        encoding="utf-8",
    )
    (store.tasks_dir / f"{second_data['id']}.json").write_text(
        json.dumps(second_data, ensure_ascii=False),
        encoding="utf-8",
    )

    tool = TaskGetTool(store)
    result_a = await _execute(tool, {"task_id": common}, conversation_id="conv-a")
    result_b = await _execute(tool, {"task_id": common}, conversation_id="conv-b")
    assert result_a["id"] == first_data["id"]
    assert result_b["id"] == second_data["id"]


# 函数说明：test_task_update_rejects_plan_and_single_step_together
# 用途：回归验证回归测试与测试辅助中的
# `task_update_rejects_plan_and_single_step_together` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `TaskStep` →
# `path.read_bytes` → `pytest.raises` → `_execute` → `TaskUpdateTool`。
# 分支与异常：
#   验证条件：`path.read_bytes() == original`。
#   预期异常：`pytest.raises(ValueError, match='cannot be combined')`。
# 副作用与资源：
#   文件或资源访问：`path.read_bytes`。
async def test_task_update_rejects_plan_and_single_step_together(
    store: FileTaskStore,
) -> None:
    task = await store.create(
        title="互斥更新",
        owner_conversation_id=_OWNER,
        steps=(TaskStep(id="s1", title="步骤"),),
    )
    path = store.tasks_dir / f"{task.id}.json"
    original = path.read_bytes()

    with pytest.raises(ValueError, match="cannot be combined"):
        await _execute(
            TaskUpdateTool(store),
            {
                "task_id": task.id,
                "steps": [{"id": "s1", "title": "步骤"}],
                "step_id": "s1",
                "step_status": "in_progress",
            },
        )
    assert path.read_bytes() == original
