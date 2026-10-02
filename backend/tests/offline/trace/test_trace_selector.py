
from __future__ import annotations

from datetime import UTC, datetime

from app.domain.skill_learning.trace_selector import TaskTraceSelector
from app.domain.task import Task
from app.models.types import ToolCall, ToolResult
from app.runtime.agent.events import AgentEvent, AgentEventType

TASK_A = "a" * 32
TASK_B = "b" * 32


# 函数说明：_task
# 用途：返回 `Task(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   task_id：目标任务标识，类型 `str`；默认 `TASK_A`。
#   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`；默认 `('r1',)`。
# 返回：类型 `Task`；返回 `Task(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Task` → `datetime.now`。
def _task(task_id: str = TASK_A, run_ids: tuple[str, ...] = ("r1",)) -> Task:
    return Task(
        id=task_id,
        title="测试任务",
        owner_conversation_id="conv",
        run_ids=run_ids,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


# 函数说明：_tool_event
# 用途：返回 `AgentEvent(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   run_id：目标运行标识，类型 `str`。
#   sequence：事件或记录顺序号，类型 `int`。
#   step：当前任务步骤，类型 `int`。
#   name：目标对象、工具或配置项名称，类型 `str`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
#   success：执行是否成功，类型 `bool`；默认 `True`。
# 返回：类型 `AgentEvent`；返回 `AgentEvent(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentEvent` → `ToolCall` →
# `ToolResult`。
def _tool_event(
    run_id: str,
    sequence: int,
    step: int,
    name: str,
    arguments: dict,
    *,
    success: bool = True,
) -> AgentEvent:
    return AgentEvent(
        run_id=run_id,
        sequence=sequence,
        step=step,
        type=AgentEventType.TOOL_COMPLETED,
        tool_call=ToolCall(id=f"c{sequence}", name=name, arguments=arguments),
        tool_result=ToolResult(
            tool_call_id=f"c{sequence}",
            tool_name=name,
            success=success,
            duration_ms=0.0,
        ),
    )


# 函数说明：_run
# 用途：运行回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   *events：额外位置参数，按实现向内部调用传递。
# 返回：类型 `tuple[AgentEvent, ...]`；返回 `events`。
def _run(*events: AgentEvent) -> tuple[AgentEvent, ...]:
    return events


# 函数说明：_selected_steps
# 用途：返回 `[event.step for event in selector.select(task, run_events, max_events=
# max_events) if…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   selector：`selector`输入或配置值，类型 `TaskTraceSelector`。
#   task：当前任务记录，类型 `Task`。
#   run_events：传给 `selector.select` 的输入，类型 `dict[str, tuple[AgentEvent, ...]]`
# 。
#   max_events：事件序列输入或配置值，类型 `int | None`；默认 `None`。
# 返回：类型 `list[int]`；返回 `[event.step for event in selector.select(task,
# run_events, max_events=max_events) if…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`selector.select`。
def _selected_steps(
    selector: TaskTraceSelector,
    task: Task,
    run_events: dict[str, tuple[AgentEvent, ...]],
    *,
    max_events: int | None = None,
) -> list[int]:
    return [
        event.step
        for event in selector.select(task, run_events, max_events=max_events)
        if event.step is not None
    ]




# 函数说明：test_same_run_exact_range
# 用途：回归验证回归测试与测试辅助中的 `same_run_exact_range` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_run` → `_tool_event` →
# `TaskTraceSelector` → `_selected_steps` → `_task`。
# 分支与异常：
#   验证条件：`_selected_steps(selector, _task(), {'r1': run}) == [2, 3, 4, 5, 6]`。
def test_same_run_exact_range() -> None:
    run = _run(
        _tool_event("r1", 1, 1, "read_file", {}),  
        _tool_event(
            "r1", 2, 2, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "in_progress"},
        ),
        _tool_event("r1", 3, 3, "run_pytest", {}, success=False),
        _tool_event("r1", 4, 4, "edit_file", {}),
        _tool_event("r1", 5, 5, "run_pytest", {}),
        _tool_event(
            "r1", 6, 6, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "done",
             "step_note": "ok"},
        ),
        _tool_event("r1", 7, 7, "read_file", {}),  
    )
    selector = TaskTraceSelector()
    assert _selected_steps(selector, _task(), {"r1": run}) == [2, 3, 4, 5, 6]




# 函数说明：test_only_current_task_anchors
# 用途：回归验证回归测试与测试辅助中的 `only_current_task_anchors` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_run` → `_tool_event` →
# `TaskTraceSelector` → `_selected_steps` → `_task`。
# 分支与异常：
#   验证条件：`_selected_steps(selector, _task(TASK_A), {'r1': run}) == [4, 5, 6, 7]`。
def test_only_current_task_anchors() -> None:
    run = _run(
        _tool_event("r1", 1, 1, "read_file", {}),
        _tool_event(
            "r1", 2, 2, "task_update",
            {"task_id": TASK_B, "step_id": "s1", "step_status": "in_progress"},
        ),  
        _tool_event("r1", 3, 3, "read_file", {}),
        _tool_event(
            "r1", 4, 4, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "in_progress"},
        ),
        _tool_event("r1", 5, 5, "edit_file", {}),
        _tool_event("r1", 6, 6, "run_pytest", {}),
        _tool_event(
            "r1", 7, 7, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "done",
             "step_note": "ok"},
        ),
    )
    selector = TaskTraceSelector()
    assert _selected_steps(selector, _task(TASK_A), {"r1": run}) == [4, 5, 6, 7]




# 函数说明：test_failed_task_update_is_not_anchor
# 用途：回归验证回归测试与测试辅助中的 `failed_task_update_is_not_anchor` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_run` → `_tool_event` →
# `TaskTraceSelector` → `selector.select` → `_task`。
# 分支与异常：
#   验证条件：`selector.select(_task(), {'r1': run}) == ()`。
def test_failed_task_update_is_not_anchor() -> None:
    run = _run(
        _tool_event(
            "r1", 1, 1, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "in_progress"},
            success=False,
        ),
        _tool_event("r1", 2, 2, "edit_file", {}),
    )
    selector = TaskTraceSelector()
    assert selector.select(_task(), {"r1": run}) == ()




# 函数说明：test_missing_in_progress_uses_backward_window
# 用途：回归验证回归测试与测试辅助中的 `missing_in_progress_uses_backward_window` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_run` → `_tool_event` →
# `TaskTraceSelector` → `_selected_steps` → `_task`。
# 分支与异常：
#   验证条件：`_selected_steps(selector, _task(), {'r1': run}) == [1, 2, 3, 4]`。
def test_missing_in_progress_uses_backward_window() -> None:
    run = _run(
        _tool_event("r1", 1, 1, "read_file", {}),
        _tool_event("r1", 2, 2, "run_pytest", {}, success=False),
        _tool_event("r1", 3, 3, "edit_file", {}),
        _tool_event(
            "r1", 4, 4, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "done",
             "step_note": "ok"},
        ),
    )
    selector = TaskTraceSelector(backward_window_steps=5)
    assert _selected_steps(selector, _task(), {"r1": run}) == [1, 2, 3, 4]




# 函数说明：test_cross_run_step_span
# 用途：回归验证回归测试与测试辅助中的 `cross_run_step_span` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_task` → `_run` → `_tool_event` →
# `TaskTraceSelector` → `_selected_steps`。
# 分支与异常：
#   验证条件：
# `_selected_steps(selector, task, {'r1': run1, 'r4': run4}) == [5, 6, 7, 1, 2, 3, 4]`。
def test_cross_run_step_span() -> None:
    task = _task(run_ids=("r1", "r4"))
    run1 = _run(
        _tool_event(
            "r1", 1, 5, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "in_progress"},
        ),
        _tool_event("r1", 2, 6, "read_file", {}),
        _tool_event("r1", 3, 7, "run_pytest", {}, success=False),
    )
    run4 = _run(
        _tool_event("r4", 1, 1, "read_file", {}),
        _tool_event("r4", 2, 2, "edit_file", {}),
        _tool_event("r4", 3, 3, "run_pytest", {}),
        _tool_event(
            "r4", 4, 4, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "done",
             "step_note": "ok"},
        ),
    )
    selector = TaskTraceSelector()
    assert _selected_steps(
        selector, task, {"r1": run1, "r4": run4}
    ) == [5, 6, 7, 1, 2, 3, 4]




# 函数说明：test_unrelated_before_and_after_excluded
# 用途：回归验证回归测试与测试辅助中的 `unrelated_before_and_after_excluded` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_run` → `_tool_event` →
# `TaskTraceSelector` → `_selected_steps` → `_task`。
# 分支与异常：
#   验证条件：`_selected_steps(selector, _task(), {'r1': run}) == [2, 3, 4]`。
def test_unrelated_before_and_after_excluded() -> None:
    run = _run(
        _tool_event("r1", 1, 1, "read_file", {}),  
        _tool_event(
            "r1", 2, 2, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "in_progress"},
        ),
        _tool_event("r1", 3, 3, "edit_file", {}),
        _tool_event(
            "r1", 4, 4, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "done",
             "step_note": "ok"},
        ),
        _tool_event("r1", 5, 5, "read_file", {}),  
    )
    selector = TaskTraceSelector()
    assert _selected_steps(selector, _task(), {"r1": run}) == [2, 3, 4]




# 函数说明：test_plain_update_uses_nearby_window
# 用途：回归验证回归测试与测试辅助中的 `plain_update_uses_nearby_window` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_run` → `_tool_event` →
# `TaskTraceSelector` → `_selected_steps` → `_task`。
# 分支与异常：
#   验证条件：`_selected_steps(selector, _task(), {'r1': run}) == [1, 2]`。
def test_plain_update_uses_nearby_window() -> None:
    run = _run(
        _tool_event("r1", 1, 1, "read_file", {}),
        _tool_event("r1", 2, 2, "task_update", {"task_id": TASK_A, "goal": "新目标"}),
        _tool_event("r1", 3, 3, "run_pytest", {}),
    )
    selector = TaskTraceSelector(backward_window_steps=5)
    assert _selected_steps(selector, _task(), {"r1": run}) == [1, 2]




# 函数说明：test_no_valid_anchor_returns_empty
# 用途：回归验证回归测试与测试辅助中的 `no_valid_anchor_returns_empty` 场景，下方断言说
# 明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_run` → `_tool_event` →
# `TaskTraceSelector` → `selector.select` → `_task`。
# 分支与异常：
#   验证条件：`selector.select(_task(), {'r1': run}) == ()`。
def test_no_valid_anchor_returns_empty() -> None:
    run = _run(
        _tool_event("r1", 1, 1, "read_file", {}),
        _tool_event("r1", 2, 2, "run_pytest", {}),
    )
    selector = TaskTraceSelector()
    assert selector.select(_task(), {"r1": run}) == ()


# 函数说明：test_missing_run_returns_empty
# 用途：回归验证回归测试与测试辅助中的 `missing_run_returns_empty` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskTraceSelector` →
# `selector.select` → `_task`。
# 分支与异常：
#   验证条件：`selector.select(_task(), {}) == ()`。
def test_missing_run_returns_empty() -> None:
    selector = TaskTraceSelector()
    assert selector.select(_task(), {}) == ()




# 函数说明：test_max_events_is_hard_limit
# 用途：回归验证回归测试与测试辅助中的 `max_events_is_hard_limit` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_event` → `TaskTraceSelector` →
# `selector.select` → `_task`。
# 分支与异常：
#   验证条件：`len(selected) == 5`。
#   验证条件：`[event.step for event in selected] == [1, 2, 3, 4, 5]`。
def test_max_events_is_hard_limit() -> None:
    events: list[AgentEvent] = []
    for step in range(1, 21):
        if step == 1:
            events.append(
                _tool_event(
                    "r1", step, step, "task_update",
                    {"task_id": TASK_A, "step_id": "s1",
                     "step_status": "in_progress"},
                )
            )
        elif step == 20:
            events.append(
                _tool_event(
                    "r1", step, step, "task_update",
                    {"task_id": TASK_A, "step_id": "s1",
                     "step_status": "done", "step_note": "ok"},
                )
            )
        else:
            events.append(_tool_event("r1", step, step, "run_pytest", {}))
    selector = TaskTraceSelector()
    selected = selector.select(_task(), {"r1": tuple(events)}, max_events=5)
    assert len(selected) == 5
    assert [event.step for event in selected] == [1, 2, 3, 4, 5]




# 函数说明：test_overlapping_ranges_deduplicate
# 用途：回归验证回归测试与测试辅助中的 `overlapping_ranges_deduplicate` 场景，下方断言说
# 明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_run` → `_tool_event` →
# `TaskTraceSelector` → `_selected_steps` → `_task`。
# 分支与异常：
#   验证条件：`selected == [2, 3, 4, 5, 6]`。
#   验证条件：`len(selected) == len(set(selected))`。
def test_overlapping_ranges_deduplicate() -> None:
    run = _run(
        _tool_event("r1", 1, 1, "read_file", {}),
        _tool_event(
            "r1", 2, 2, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "in_progress"},
        ),
        _tool_event(
            "r1", 3, 3, "task_update",
            {"task_id": TASK_A, "step_id": "s2", "step_status": "in_progress"},
        ),  
        _tool_event("r1", 4, 4, "edit_file", {}),
        _tool_event(
            "r1", 5, 5, "task_update",
            {"task_id": TASK_A, "step_id": "s2", "step_status": "done",
             "step_note": "a"},
        ),
        _tool_event(
            "r1", 6, 6, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "done",
             "step_note": "b"},
        ),
    )
    selector = TaskTraceSelector()
    selected = _selected_steps(selector, _task(), {"r1": run})
    assert selected == [2, 3, 4, 5, 6]
    assert len(selected) == len(set(selected))




# 函数说明：test_repeated_in_progress_keeps_earliest_segment
# 用途：回归验证回归测试与测试辅助中的 `repeated_in_progress_keeps_earliest_segment` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_task` → `_run` → `_tool_event` →
# `TaskTraceSelector` → `_selected_steps`。
# 分支与异常：
#   验证条件：`_selected_steps(selector, task, {'r1': run1, 'r2': run2, 'r3': run3}) ==
# [2, 3, 1, 2, 3, 1, 2, 3, 4]`。
def test_repeated_in_progress_keeps_earliest_segment() -> None:
    task = _task(run_ids=("r1", "r2", "r3"))
    run1 = _run(
        _tool_event(
            "r1", 1, 2, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "in_progress"},
        ),  
        _tool_event("r1", 2, 3, "read_file", {}),
    )
    run2 = _run(
        _tool_event("r2", 1, 1, "read_file", {}),
        _tool_event(
            "r2", 2, 2, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "in_progress"},
        ),  
        _tool_event("r2", 3, 3, "run_pytest", {}, success=False),
    )
    run3 = _run(
        _tool_event("r3", 1, 1, "read_file", {}),
        _tool_event("r3", 2, 2, "edit_file", {}),
        _tool_event("r3", 3, 3, "run_pytest", {}),
        _tool_event(
            "r3", 4, 4, "task_update",
            {"task_id": TASK_A, "step_id": "s1", "step_status": "done",
             "step_note": "ok"},
        ),
    )
    selector = TaskTraceSelector()
    assert _selected_steps(
        selector, task, {"r1": run1, "r2": run2, "r3": run3}
    ) == [2, 3, 1, 2, 3, 1, 2, 3, 4]
