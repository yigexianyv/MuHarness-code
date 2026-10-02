
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.domain.task import Task
from app.runtime.agent.events import AgentEvent, AgentEventType

_DEFAULT_BACKWARD_WINDOW_STEPS = 5

_TASK_UPDATE_TOOL = "task_update"
_STEP_LIFECYCLE_STATUSES = frozenset({"in_progress", "done", "blocked"})


@dataclass(frozen=True)
class _Anchor:

    run_id: str
    step: int
    step_id: str | None
    step_status: str | None


class TaskTraceSelector:

    # 函数说明：TaskTraceSelector.__init__
    # 用途：初始化 TaskTraceSelector；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   backward_window_steps：传给 `max` 的输入，类型 `int`；默认
    # `_DEFAULT_BACKWARD_WINDOW_STEPS`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.backward_window_steps`。
    def __init__(
        self,
        *,
        backward_window_steps: int = _DEFAULT_BACKWARD_WINDOW_STEPS,
    ) -> None:
        self.backward_window_steps = max(1, backward_window_steps)

    # 函数说明：TaskTraceSelector.select
    # 用途：选取TaskTraceSelector，供技能候选提炼与审核使用。
    # 参数：
    #   task：当前任务记录，类型 `Task`。
    #   run_events：传给 `self._find_task_update_anchors` 的输入，类型
    # `dict[str, tuple[AgentEvent, ...]]`。
    #   max_events：事件序列输入或配置值，类型 `int | None`；默认 `None`。
    # 返回：类型 `tuple[AgentEvent, ...]`；按分支返回 `()`；`self.
    # _select_events_for_ranges(task, coverage, run_events, max_events=max_events)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._find_task_update_anchors` →
    #  `self._build_relevant_step_ranges` → `self._select_events_for_ranges`。
    # 分支与异常：
    #   当 `not anchors` 时，返回 `()`。
    #   当 `not coverage` 时，返回 `()`。
    def select(
        self,
        task: Task,
        run_events: dict[str, tuple[AgentEvent, ...]],
        *,
        max_events: int | None = None,
    ) -> tuple[AgentEvent, ...]:

        anchors = self._find_task_update_anchors(task, run_events)
        if not anchors:
            return ()
        coverage = self._build_relevant_step_ranges(anchors, run_events)
        if not coverage:
            return ()
        return self._select_events_for_ranges(
            task, coverage, run_events, max_events=max_events
        )


    # 函数说明：TaskTraceSelector._find_task_update_anchors
    # 用途：更新`anchors`，供技能候选提炼与审核使用。
    # 参数：
    #   task：当前任务记录，类型 `Task`。
    #   run_events：运行事件序列输入或配置值，类型 `dict[str, tuple[AgentEvent, ...]]`。
    # 返回：类型 `tuple[_Anchor, ...]`；返回 `tuple(anchors)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`task.id.startswith` → `_Anchor`。
    # 分支与异常：
    #   当 `event.type is not AgentEventType.TOOL_COMPLETED` 时，跳过当前循环项。
    #   当 `call is None or result is None` 时，跳过当前循环项。
    #   当 `call.name != _TASK_UPDATE_TOOL` 时，跳过当前循环项。
    #   当 `not result.success` 时，跳过当前循环项。
    def _find_task_update_anchors(
        self,
        task: Task,
        run_events: dict[str, tuple[AgentEvent, ...]],
    ) -> tuple[_Anchor, ...]:

        anchors: list[_Anchor] = []
        for run_id, events in run_events.items():
            for event in events:
                if event.type is not AgentEventType.TOOL_COMPLETED:
                    continue
                call = event.tool_call
                result = event.tool_result
                if call is None or result is None:
                    continue
                if call.name != _TASK_UPDATE_TOOL:
                    continue
                if not result.success:
                    continue
                if event.step is None:
                    continue
                arguments = call.arguments
                if not isinstance(arguments, dict):
                    continue
                task_id_arg = arguments.get("task_id")
                if not isinstance(task_id_arg, str) or not task_id_arg.strip():
                    continue
                if not task.id.startswith(task_id_arg.strip()):
                    continue
                step_id = arguments.get("step_id")
                step_status = arguments.get("step_status")
                anchors.append(
                    _Anchor(
                        run_id=run_id,
                        step=event.step,
                        step_id=(
                            step_id.strip()
                            if isinstance(step_id, str) and step_id.strip()
                            else None
                        ),
                        step_status=(
                            step_status.strip()
                            if isinstance(step_status, str) and step_status.strip()
                            else None
                        ),
                    )
                )
        return tuple(anchors)


    # 函数说明：TaskTraceSelector._build_relevant_step_ranges
    # 用途：构建步骤，供技能候选提炼与审核使用。
    # 参数：
    #   anchors：`anchors`输入或配置值，类型 `tuple[_Anchor, ...]`。
    #   run_events：传给 `tuple` 的输入，类型 `dict[str, tuple[AgentEvent, ...]]`。
    # 返回：类型 `dict[str, set[int]]`；返回 `dict(coverage)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`defaultdict` → `items.sort` →
    # `self._add_backward_window` → `coverage[run_id].update` →
    # `coverage[mid_run_id].update`。
    def _build_relevant_step_ranges(
        self,
        anchors: tuple[_Anchor, ...],
        run_events: dict[str, tuple[AgentEvent, ...]],
    ) -> dict[str, set[int]]:

        run_order = tuple(run_events)
        run_max_step: dict[str, int] = {}
        for run_id, events in run_events.items():
            steps = [event.step for event in events if event.step is not None]
            run_max_step[run_id] = max(steps) if steps else 0

        coverage: dict[str, set[int]] = defaultdict(set)
        step_anchors: dict[str, list[tuple[int, str, int, str]]] = defaultdict(list)
        non_step_anchors: list[_Anchor] = []
        run_indices = {run_id: index for index, run_id in enumerate(run_order)}

        for anchor in anchors:
            if (
                anchor.step_id
                and anchor.step_status in _STEP_LIFECYCLE_STATUSES
            ):
                step_anchors[anchor.step_id].append(
                    (
                        run_indices[anchor.run_id],
                        anchor.run_id,
                        anchor.step,
                        anchor.step_status,
                    )
                )
            else:
                non_step_anchors.append(anchor)

        for items in step_anchors.values():
            items.sort(key=lambda item: (item[0], item[2]))
            open_segment: tuple[int, str, int] | None = None
            for run_index, run_id, step, status in items:
                if status == "in_progress":
                    if open_segment is None:
                        open_segment = (run_index, run_id, step)
                else:
                    if open_segment is None:
                        self._add_backward_window(coverage, run_id, step)
                    else:
                        open_index, open_run_id, open_step = open_segment
                        if open_index == run_index:
                            coverage[run_id].update(range(open_step, step + 1))
                        else:
                            for index in range(open_index, run_index + 1):
                                mid_run_id = run_order[index]
                                if index == open_index:
                                    coverage[mid_run_id].update(
                                        range(
                                            open_step,
                                            run_max_step[mid_run_id] + 1,
                                        )
                                    )
                                elif index == run_index:
                                    coverage[mid_run_id].update(
                                        range(1, step + 1)
                                    )
                                else:
                                    coverage[mid_run_id].update(
                                        range(1, run_max_step[mid_run_id] + 1)
                                    )
                        open_segment = None
            if open_segment is not None:
                _, run_id, step = open_segment
                coverage[run_id].update(range(step, run_max_step[run_id] + 1))

        for anchor in non_step_anchors:
            self._add_backward_window(coverage, anchor.run_id, anchor.step)

        return dict(coverage)

    # 函数说明：TaskTraceSelector._add_backward_window
    # 用途：添加`backward_window`，供技能候选提炼与审核使用。
    # 参数：
    #   coverage：`coverage`输入或配置值，类型 `dict[str, set[int]]`。
    #   run_id：目标运行标识，类型 `str`。
    #   anchor_step：步骤输入或配置值，类型 `int`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`coverage[run_id].update`。
    def _add_backward_window(
        self,
        coverage: dict[str, set[int]],
        run_id: str,
        anchor_step: int,
    ) -> None:

        start = max(1, anchor_step - self.backward_window_steps + 1)
        coverage[run_id].update(range(start, anchor_step + 1))


    # 函数说明：TaskTraceSelector._select_events_for_ranges
    # 用途：选取事件序列，供技能候选提炼与审核使用。
    # 参数：
    #   task：当前任务记录，类型 `Task`。
    #   coverage：`coverage`输入或配置值，类型 `dict[str, set[int]]`。
    #   run_events：运行事件序列输入或配置值，类型 `dict[str, tuple[AgentEvent, ...]]`。
    #   max_events：事件序列输入或配置值，类型 `int | None`。
    # 返回：类型 `tuple[AgentEvent, ...]`；返回 `tuple(selected)`。
    # 分支与异常：
    #   当 `not steps` 时，跳过当前循环项。
    #   当 `max_events is not None and len(selected) >= max_events` 时，返回
    # `tuple(selected)`。
    def _select_events_for_ranges(
        self,
        task: Task,
        coverage: dict[str, set[int]],
        run_events: dict[str, tuple[AgentEvent, ...]],
        *,
        max_events: int | None,
    ) -> tuple[AgentEvent, ...]:

        selected: list[AgentEvent] = []
        for run_id in task.run_ids:
            steps = coverage.get(run_id)
            if not steps:
                continue
            for event in run_events[run_id]:
                if event.step is not None and event.step in steps:
                    selected.append(event)
                    if max_events is not None and len(selected) >= max_events:
                        return tuple(selected)
        return tuple(selected)


__all__ = ["TaskTraceSelector"]
