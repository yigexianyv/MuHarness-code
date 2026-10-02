
from __future__ import annotations

import logging
from typing import Any

from app.domain.task import Task
from app.runtime.agent.events import AgentEvent, AgentEventType

from .config import SkillLearningSettings

logger = logging.getLogger("muharness.skill_learning.evidence")

_TASK_TOOL_NAMES = frozenset({"task_create", "task_update"})
_OBSERVABLE_TOOL_START = AgentEventType.TOOL_STARTED
_OBSERVABLE_TOOL_COMPLETE = AgentEventType.TOOL_COMPLETED
_OBSERVABLE_TYPES = frozenset(
    {
        AgentEventType.TOOL_STARTED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.AGENT_COMPLETED,
        AgentEventType.AGENT_FAILED,
    }
)


class TraceEvidenceBuilder:

    # 函数说明：TraceEvidenceBuilder.__init__
    # 用途：初始化 TraceEvidenceBuilder；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   settings：业务或模型设置，类型 `SkillLearningSettings`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.settings`。
    def __init__(self, settings: SkillLearningSettings) -> None:
        self.settings = settings

    # 函数说明：TraceEvidenceBuilder.build
    # 用途：构建TraceEvidenceBuilder，供技能候选提炼与审核使用。
    # 参数：
    #   task：当前任务记录，类型 `Task`。
    #   events：事件集合，类型 `tuple[AgentEvent, ...] | list[AgentEvent]`。
    # 返回：类型 `str`；按分支返回 `self._task_only(task, '没有可用的 Trace 事件')`；
    # `_truncate(text, self.settings.skill_learning_max_evidence_chars)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._task_only` →
    # `self._task_tool_summary` → `_dedupe_consecutive` → `_truncate`。
    # 分支与异常：
    #   当 `not events` 时，返回 `self._task_only(task, '没有可用的 Trace 事件')`。
    #   当 `event.type not in _OBSERVABLE_TYPES` 时，跳过当前循环项。
    def build(
        self,
        task: Task,
        events: tuple[AgentEvent, ...] | list[AgentEvent],
    ) -> str:

        if not events:
            return self._task_only(task, "没有可用的 Trace 事件")
        lines: list[str] = []
        tool_sequence: list[str] = []
        failed_calls: list[str] = []
        task_updates: list[str] = []
        completion: list[str] = []

        for event in events:
            if event.type not in _OBSERVABLE_TYPES:
                continue
            if event.type is _OBSERVABLE_TOOL_START:
                name = event.tool_call.name if event.tool_call else "?"
                tool_sequence.append(name)
            elif event.type is _OBSERVABLE_TOOL_COMPLETE:
                result = event.tool_result
                tool_name = event.tool_call.name if event.tool_call else "?"
                if result is not None and not result.success:
                    failed_calls.append(
                        f"{tool_name}: {result.error or 'failed'}"
                    )
                if (
                    tool_name in _TASK_TOOL_NAMES
                    and result is not None
                    and result.success
                ):
                    summary = self._task_tool_summary(event)
                    if summary:
                        task_updates.append(summary)
            elif event.type is AgentEventType.AGENT_COMPLETED:
                completion.append("agent completed")
            elif event.type is AgentEventType.AGENT_FAILED:
                error = event.error or "agent failed"
                completion.append(f"agent failed: {error}")

        if tool_sequence:
            sequence = " → ".join(_dedupe_consecutive(tool_sequence))
            lines.append(f"工具调用序列: {sequence}")
        if failed_calls:
            lines.append("失败工具调用:")
            lines.extend(f"- {item}" for item in failed_calls[:20])
        if task_updates:
            lines.append("Task 变更:")
            lines.extend(f"- {item}" for item in task_updates[:30])
        if completion:
            lines.append("完成证据: " + "; ".join(completion))
        if not lines:
            lines.append("无工具调用（可能为纯问答 Run）")

        header = (
            f"Task: {task.title}\n"
            f"Goal: {task.goal or '（无）'}\n"
            "状态: "
            f"{task.status.value} · 步骤 {len(task.steps)} · run 数 {len(task.run_ids)}"
        )
        text = header + "\n" + "\n".join(lines)
        return _truncate(text, self.settings.skill_learning_max_evidence_chars)


    # 函数说明：TraceEvidenceBuilder._task_tool_summary
    # 用途：在技能候选提炼与审核中处理 `_task_tool_summary`，通过
    # `self._task_create_summary` 完成首个内部处理步骤。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `str | None`；按分支返回 `None`；`self._task_create_summary(arguments)`
    # ；`self._task_update_summary(arguments)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._task_create_summary` →
    # `self._task_update_summary`。
    # 分支与异常：
    #   当 `call is None or result is None` 时，返回 `None`。
    #   当 `not isinstance(arguments, dict)` 时，返回 `None`。
    #   当 `call.name == 'task_create'` 时，返回 `self._task_create_summary(arguments)`
    # 。
    #   当 `call.name == 'task_update'` 时，返回 `self._task_update_summary(arguments)`
    # 。
    def _task_tool_summary(self, event: AgentEvent) -> str | None:

        call = event.tool_call
        result = event.tool_result
        if call is None or result is None:
            return None
        arguments = call.arguments or {}
        if not isinstance(arguments, dict):
            return None
        if call.name == "task_create":
            return self._task_create_summary(arguments)
        if call.name == "task_update":
            return self._task_update_summary(arguments)
        return None

    # 函数说明：TraceEvidenceBuilder._task_create_summary
    # 用途：创建摘要，供技能候选提炼与审核使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `title`
    # 、`goal`、`steps`。
    # 返回：类型 `str`；返回 `'\n'.join(lines)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_clip` → `_step_titles`。
    def _task_create_summary(self, arguments: dict[str, Any]) -> str:
        lines = ["task_create:"]
        title = arguments.get("title")
        if isinstance(title, str) and title.strip():
            lines.append(f"- title: {_clip(title)}")
        goal = arguments.get("goal")
        if isinstance(goal, str) and goal.strip():
            lines.append(f"- goal: {_clip(goal)}")
        steps = _step_titles(arguments.get("steps"))
        if steps:
            lines.append("- steps: " + "; ".join(steps))
        return "\n".join(lines)

    # 函数说明：TraceEvidenceBuilder._task_update_summary
    # 用途：更新摘要，供技能候选提炼与审核使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `status`、`goal`、`state`、`constraints`、`facts`、`steps`、`step_id`、
    # `step_status`、`step_note`。
    # 返回：类型 `str | None`；按分支返回 `None`；
    # `'task_update:\n' + '\n'.join((' ' + line for line in lines))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_clip` → `_string_entries` →
    # `_step_titles`。
    # 分支与异常：
    #   当 `not lines` 时，返回 `None`。
    def _task_update_summary(self, arguments: dict[str, Any]) -> str | None:

        lines: list[str] = []
        status = arguments.get("status")
        if isinstance(status, str) and status:
            lines.append(f"status: {status}")
        goal = arguments.get("goal")
        if isinstance(goal, str) and goal.strip():
            lines.append(f"goal: {_clip(goal)}")
        state = _string_entries(arguments.get("state"))
        if state:
            lines.append("state replaced: " + " | ".join(state))
        constraints = _string_entries(arguments.get("constraints"))
        if constraints:
            lines.append("constraints added: " + " | ".join(constraints))
        facts = _string_entries(arguments.get("facts"))
        if facts:
            lines.append("facts added: " + " | ".join(facts))
        plan_steps = _step_titles(arguments.get("steps"))
        if plan_steps:
            lines.append("plan replaced:")
            lines.extend(f"  - {item}" for item in plan_steps)
        step_id = arguments.get("step_id")
        step_status = arguments.get("step_status")
        if isinstance(step_id, str) and step_id and isinstance(step_status, str):
            note = arguments.get("step_note")
            note_text = (
                f": {_clip(note)}" if isinstance(note, str) and note else ""
            )
            lines.append(f"step {step_id} -> {step_status}{note_text}")
        if not lines:
            return None
        return "task_update:\n" + "\n".join("  " + line for line in lines)

    # 函数说明：TraceEvidenceBuilder._task_only
    # 用途：在技能候选提炼与审核中处理 `_task_only`，通过 `', '.join` 完成首个内部处理步
    # 骤。
    # 参数：
    #   task：当前任务记录，类型 `Task`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    # 返回：类型 `str`；返回
    # `_truncate('\n'.join(parts), self.settings.skill_learning_max_evidence_chars)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_task_steps` → `_truncate`。
    def _task_only(self, task: Task, reason: str) -> str:
        final_steps = _task_steps(task)
        parts = [
            f"Task: {task.title}",
            f"Goal: {task.goal or '（无）'}",
            f"约束: {', '.join(task.constraints) or '（无）'}",
            f"关键事实: {', '.join(task.key_facts) or '（无）'}",
            f"最终步骤: {', '.join(final_steps) or '（无步骤）'}",
            f"run 数: {len(task.run_ids)}",
            f"备注: {reason}",
        ]
        return _truncate(
            "\n".join(parts),
            self.settings.skill_learning_max_evidence_chars,
        )


# 函数说明：_task_steps
# 用途：返回 `[step.title for step in task.steps if step.status.value == 'done']`，提供
# 技能候选提炼与审核 的派生值。
# 参数：
#   task：当前任务记录，类型 `Task`。
# 返回：类型 `list[str]`；返回
# `[step.title for step in task.steps if step.status.value == 'done']`。
def _task_steps(task: Task) -> list[str]:
    return [
        step.title for step in task.steps if step.status.value == "done"
    ]


# 函数说明：_step_titles
# 用途：在技能候选提炼与审核中处理 `_step_titles`，通过 `item.get` 完成首个内部处理步骤
# 。
# 参数：
#   value：待校验、规范化或转换的值，类型 `Any`。
# 返回：类型 `list[str]`；按分支返回 `[]`；`titles`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_clip`。
# 分支与异常：
#   当 `not isinstance(value, (list, tuple))` 时，返回 `[]`。
#   当 `not isinstance(item, dict)` 时，跳过当前循环项。
#   当 `len(titles) >= 20` 时，结束当前循环。
def _step_titles(value: Any) -> list[str]:

    if not isinstance(value, (list, tuple)):
        return []
    titles: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            titles.append(_clip(title))
        if len(titles) >= 20:
            break
    return titles


# 函数说明：_string_entries
# 用途：在技能候选提炼与审核中处理 `_string_entries`，通过 `item.strip` 完成首个内部处理
# 步骤。
# 参数：
#   value：待校验、规范化或转换的值，类型 `Any`。
# 返回：类型 `list[str]`；按分支返回 `[]`；`entries`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_clip`。
# 分支与异常：
#   当 `not isinstance(value, (list, tuple))` 时，返回 `[]`。
#   当 `len(entries) >= 20` 时，结束当前循环。
def _string_entries(value: Any) -> list[str]:

    if not isinstance(value, (list, tuple)):
        return []
    entries: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            entries.append(_clip(item))
        if len(entries) >= 20:
            break
    return entries


# 函数说明：_clip
# 用途：限制长度技能候选提炼与审核，供技能候选提炼与审核使用。
# 参数：
#   text：待处理的文本，类型 `str`。
#   limit：本次返回或处理的数量上限，类型 `int`；默认 `200`。
# 返回：类型 `str`；返回 `text if len(text) <= limit else text[:limit] + '…'`。
def _clip(text: str, limit: int = 200) -> str:

    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


# 函数说明：_dedupe_consecutive
# 用途：去重`consecutive`，供技能候选提炼与审核使用。
# 参数：
#   sequence：事件或记录顺序号，类型 `list[str]`。
# 返回：类型 `list[str]`；返回 `result`。
def _dedupe_consecutive(sequence: list[str]) -> list[str]:
    result: list[str] = []
    for item in sequence:
        if not result or result[-1] != item:
            result.append(item)
    return result


# 函数说明：_truncate
# 用途：截断技能候选提炼与审核，供技能候选提炼与审核使用。
# 参数：
#   text：待处理的文本，类型 `str`。
#   limit：本次返回或处理的数量上限，类型 `int`。
# 返回：类型 `str`；按分支返回 `text`；`text[:limit] + '…[截断]'`。
# 分支与异常：
#   当 `len(text) <= limit` 时，返回 `text`。
def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…[截断]"


__all__ = ["TraceEvidenceBuilder"]
