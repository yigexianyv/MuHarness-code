
from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.models.types import AgentMode, ToolDefinition
from app.tools.base import BaseTool
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry

from .models import (
    Task,
    TaskPatch,
    TaskPriority,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
)
from .store import FileTaskStore

_MAX_LIST_LIMIT = 100
# superseded 只能由长任务写入，不暴露给普通任务工具
_TOOL_STEP_STATUSES = [
    status.value for status in TaskStepStatus if status is not TaskStepStatus.SUPERSEDED
]
_ACCEPTANCE_SCHEMA = {
    "type": "string",
    "description": "可独立核对的完成条件，例如目标文件及内容、命令结果或数据要求；描述验收方式，不把计划或工具调用本身当成已通过验收。",
}


class TaskCreateTool(BaseTool):

    # 函数说明：TaskCreateTool.__init__
    # 用途：初始化 TaskCreateTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `FileTaskStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: FileTaskStore) -> None:
        self._store = store

    # 函数说明：TaskCreateTool.definition
    # 用途：提供 TaskCreateTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_create",
            record_output=False,
            description=(
                "为当前会话创建可跨轮保存的整体任务及步骤计划，仅在目标"
                "确实需要跨步骤或跨轮持续跟踪时使用。简单问答、单次操作不建"
                "Task；已有任务用 task_update，同一目标的阶段放在 steps，只有"
                "可独立完成和关闭的目标才分别建 Task。task_list 发现已有"
                "任务，task_get 读取详情；本工具不执行计划。成功返回新建的"
                "pending 任务记录，不代表工作已开始或完成；保留返回 ID，启动"
                "任务时用它更新状态，current 仅指已 active/paused 的活动任务。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "非空的整体目标标题；同一目标的阶段写入 steps，不另起任务。",
                    },
                    "description": {
                        "type": "string",
                        "description": "可选的任务范围、背景和交付要求；不要填写尚未发生的完成记录。",
                    },
                    "goal": {
                        "type": "string",
                        "description": "可选的预期结果，说明最终要达成什么，而非已经做了什么。",
                    },
                    "priority": {
                        "type": "string",
                        "enum": [p.value for p in TaskPriority],
                        "description": "可选的跟踪优先级 low/normal/high/urgent，默认 normal；不改变任务状态。",
                    },
                    "steps": {
                        "type": "array",
                        "description": "可选的同一目标下的阶段计划；创建时均为 todo，不表示已经执行。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "非空的单个阶段或动作标题，与其他步骤共同完成整体目标。",
                                },
                                "note": {
                                    "type": "string",
                                    "description": "可选的实施范围、依赖或说明；这里只记录计划，不编造完成依据。",
                                },
                                "acceptance": _ACCEPTANCE_SCHEMA,
                            },
                            "required": ["title"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            strict=False,
            closing_allowed=True,
        )

    # 函数说明：TaskCreateTool.execute
    # 用途：执行TaskCreateTool，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；返回 `await self._execute(arguments, context=None)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._execute`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute(arguments, context=None)

    # 函数说明：TaskCreateTool.execute_with_context
    # 用途：执行上下文，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；返回
    # `await self._execute(arguments, context=context)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._execute`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return await self._execute(arguments, context=context)

    # 函数说明：TaskCreateTool._execute
    # 用途：执行TaskCreateTool，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `title`
    # 、`description`、`goal`、`priority`、`steps`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext | None`。
    # 返回：类型 `dict[str, Any]`；返回 `_task_full(task)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_conversation_id` →
    # `TaskPriority` → `_build_steps` → `self._store.create` → `_task_full`。
    # 分支与异常：
    #   当 `not isinstance(title, str) or not title.strip()` 时，抛出
    # `ValueError("'title' must be a non-empty string")`。
    #   当 `description is not None and (not isinstance(description,…` 时，抛出
    # `ValueError("'description' must be a string")`。
    #   当 `goal is not None and (not isinstance(goal, str))` 时，抛出
    # `ValueError("'goal' must be a string")`。
    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        conversation_id = _require_conversation_id(context)
        title = arguments.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("'title' must be a non-empty string")

        description = arguments.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("'description' must be a string")
        goal = arguments.get("goal")
        if goal is not None and not isinstance(goal, str):
            raise ValueError("'goal' must be a string")

        priority = TaskPriority.NORMAL
        raw_priority = arguments.get("priority")
        if raw_priority is not None:
            priority = TaskPriority(raw_priority)

        steps = _build_steps(arguments.get("steps"))

        task = await self._store.create(
            title=title,
            description=description,
            goal=goal,
            priority=priority,
            steps=steps,
            owner_conversation_id=conversation_id,
            run_ids=(
                (context.run_id,)
                if context is not None and context.run_id
                else ()
            ),
        )
        return _task_full(task)


class TaskUpdateTool(BaseTool):

    # 函数说明：TaskUpdateTool.__init__
    # 用途：初始化 TaskUpdateTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `FileTaskStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: FileTaskStore) -> None:
        self._store = store

    # 函数说明：TaskUpdateTool.definition
    # 用途：提供 TaskUpdateTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_update",
            record_output=False,
            description=(
                "将当前会话已有任务的真实进展、阻塞、计划变化或完成依据"
                "写回持久记录。仅在这些事实变化时使用，至少提供 task_id 和"
                "一个更新字段；不为展示进度重复写相同状态，不把调用成功"
                "当作工作完成；done/blocked 步骤必须附依据/原因。PLAN 模式只"
                "整理计划，不推进任务或步骤状态。确认已有记录用 task_get，"
                "独立新目标用 task_create。成功返回写回后的任务及 revision，"
                "仅证明状态已保存，实际完成仍须满足验收；current 仅指当前会话"
                "最近的 active/paused 任务，其他任务用 ID 或唯一前缀。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": (
                            "当前会话任务的完整 ID、4–32 位唯一十六进制前缀，或"
                            "活动任务句柄 current；新建 pending 任务使用返回的 ID。"
                        ),
                    },
                    "status": {
                        "type": "string",
                        "enum": [s.value for s in TaskStatus],
                        "description": (
                            "依据实际状态设置；等待用户/外部条件可用 paused，此时"
                            "不能保留 in_progress 步骤。completed 要求所有步骤已"
                            "done 或被正式取代；终态任务不能由本工具重新打开，"
                            "PLAN 模式不提供此字段。"
                        ),
                    },
                    "goal": {
                        "type": "string",
                        "description": "整体替换当前目标；仅在目标已明确调整时提供，不追加旧目标。",
                    },
                    "state": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "整体替换当前状态事实列表；填写已观察的进展/阻塞，保留仍有效事实，空列表会清空。",
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "追加新的用户约束并去重，不覆盖已有约束；仅记录用户实际提出的要求。",
                    },
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "追加已确认的关键事实或决策并去重，不覆盖已有记录，也不写入未经验证的推断。",
                    },
                    "steps": {
                        "type": "array",
                        "description": (
                            "整体替换步骤计划，不能与 step_id/step_status/step_note"
                            "同时提供。已有步骤保留 id、状态和有效依据；不能删除或"
                            "回退 done/in_progress 步骤，不能更改 superseded 步骤。"
                            "新步骤可省略 id；未指定状态默认 todo。"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "已有步骤使用原 ID；新步骤可省略，由系统生成。",
                                },
                                "title": {
                                    "type": "string",
                                    "description": "非空的步骤标题，描述该步骤应完成的具体工作。",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": _TOOL_STEP_STATUSES,
                                    "description": "依据真实进展设置，默认 todo；最多一个 in_progress，done/blocked 必须有 note。",
                                },
                                "note": {
                                    "type": "string",
                                    "description": "步骤依据或说明；done 写完成证据，blocked 写阻塞原因，不编造已执行结果。",
                                },
                                "acceptance": _ACCEPTANCE_SCHEMA,
                            },
                            "required": ["title"],
                            "additionalProperties": False,
                        },
                    },
                    "step_id": {
                        "type": "string",
                        "description": "已有步骤的精确 ID，必须与 step_status 一起提供；不用于新增步骤。",
                    },
                    "step_status": {
                        "type": "string",
                        "enum": _TOOL_STEP_STATUSES,
                        "description": (
                            "与 step_id 成对设置真实状态；done 必须同时提供"
                            "有完成依据的 step_note，blocked 必须附阻塞原因。"
                            "不能回退 done，最多一个 in_progress；PLAN 模式不提供。"
                        ),
                    },
                    "step_note": {
                        "type": "string",
                        "description": (
                            "配合 step_id/step_status 的备注；done/blocked 必填"
                            "非空内容，分别说明验收证据/具体阻塞原因。工具调用成功"
                            "不能单独充当完成依据。"
                        ),
                    },
                    "expected_revision": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "可选的已知 revision；与当前版本不同会拒绝写入。"
                            "发生冲突先 task_get 阅读新状态，再决定是否重新更新，"
                            "不要通过删除版本检查盲目覆盖。"
                        ),
                    },
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            strict=False,
            closing_allowed=True,
        )

    # 函数说明：TaskUpdateTool.execute
    # 用途：执行TaskUpdateTool，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；返回 `await self._execute(arguments, context=None)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._execute`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute(arguments, context=None)

    # 函数说明：TaskUpdateTool.execute_with_context
    # 用途：执行上下文，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；返回
    # `await self._execute(arguments, context=context)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._execute`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return await self._execute(arguments, context=context)

    # 函数说明：TaskUpdateTool._execute
    # 用途：执行TaskUpdateTool，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `task_id`、`step_id`、`step_status`、`goal`、`state`、`constraints`、`facts`、
    # `steps`、`status`、`expected_revision`、`step_note`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext | None`。
    # 返回：类型 `dict[str, Any]`；返回 `_task_full(task)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_update_steps` →
    # `TaskStatus` → `_require_conversation_id` → `_resolve_owned` → `TaskStepStatus` →
    # `self._store.apply_patch`；另有 2 个调用点。
    # 分支与异常：
    #   当 `not isinstance(task_id, str) or not task_id.strip()` 时，抛出
    # `ValueError("'task_id' must be a non-empty string")`。
    #   当 `not has_update` 时，抛出 `ValueError(…)`。
    #   当 `'status' in arguments` 时，抛出
    # `ValueError('plan mode 下不能直接改变任务状态；任务由用户接受后才开始')`。
    #   当 `arguments.get('step_id') is not None or arguments.get('…` 时，抛出
    # `ValueError('plan mode 下不能推进步骤状态；只允许更新计划内容')`。
    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        task_id = arguments.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("'task_id' must be a non-empty string")

        has_update = any(
            key in arguments
            for key in (
                "status",
                "goal",
                "state",
                "constraints",
                "facts",
                "steps",
                "step_id",
                "step_status",
            )
        )
        if not has_update:
            raise ValueError(
                "task_update requires at least one update field besides task_id"
            )

        if context is not None and context.mode is AgentMode.PLAN:
            if "status" in arguments:
                raise ValueError(
                    "plan mode 下不能直接改变任务状态；任务由用户接受后才开始"
                )
            if (
                arguments.get("step_id") is not None
                or arguments.get("step_status") is not None
            ):
                raise ValueError(
                    "plan mode 下不能推进步骤状态；只允许更新计划内容"
                )

        step_id = arguments.get("step_id")
        step_status = arguments.get("step_status")
        if step_id is not None or step_status is not None:
            if step_id is None or step_status is None:
                raise ValueError(
                    "'step_id' and 'step_status' must be provided together"
                )
        if "steps" in arguments and (
            step_id is not None
            or step_status is not None
            or "step_note" in arguments
        ):
            raise ValueError(
                "'steps' cannot be combined with step_id/step_status/step_note"
            )
        goal: str | None = None
        if "goal" in arguments:
            goal = arguments["goal"]
            if goal is not None and not isinstance(goal, str):
                raise ValueError("'goal' must be a string")
        state: tuple[str, ...] | None = None
        if "state" in arguments:
            raw_state = arguments["state"]
            if not isinstance(raw_state, list) or not all(
                isinstance(item, str) for item in raw_state
            ):
                raise ValueError("'state' must be a list of strings")
            state = tuple(raw_state)
        constraints: tuple[str, ...] = ()
        if "constraints" in arguments:
            raw_constraints = arguments["constraints"]
            if not isinstance(raw_constraints, list) or not all(
                isinstance(item, str) for item in raw_constraints
            ):
                raise ValueError("'constraints' must be a list of strings")
            constraints = tuple(raw_constraints)
        facts: tuple[str, ...] = ()
        if "facts" in arguments:
            raw_facts = arguments["facts"]
            if not isinstance(raw_facts, list) or not all(
                isinstance(item, str) for item in raw_facts
            ):
                raise ValueError("'facts' must be a list of strings")
            facts = tuple(raw_facts)
        replacement_steps: tuple[TaskStep, ...] | None = None
        if "steps" in arguments:
            replacement_steps = _build_update_steps(arguments["steps"])
        status = (
            TaskStatus(arguments["status"])
            if "status" in arguments
            else None
        )
        expected_revision = arguments.get("expected_revision")
        if expected_revision is not None and (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 1
        ):
            raise ValueError("'expected_revision' must be a positive integer")
        step_note = arguments.get("step_note")
        if step_note is not None and not isinstance(step_note, str):
            raise ValueError("'step_note' must be a string")
        if step_status in ("done", "blocked") and (
            step_note is None or not step_note.strip()
        ):
            if step_status == "done":
                raise ValueError(
                    "将步骤标记为 done 时必须提供 step_note 说明完成依据"
                )
            raise ValueError(
                "将步骤标记为 blocked 时必须提供 step_note 说明阻塞原因"
            )

        conversation_id = _require_conversation_id(context)
        task = await _resolve_owned(self._store, task_id, conversation_id)

        patch_data: dict[str, Any] = {
            "status": status,
            "add_constraints": constraints,
            "add_key_facts": facts,
            "step_id": step_id,
            "step_status": (
                TaskStepStatus(step_status) if step_status is not None else None
            ),
            "expected_revision": expected_revision,
            "run_id": context.run_id if context is not None else None,
        }
        if "goal" in arguments:
            patch_data["goal"] = goal
        if "state" in arguments:
            patch_data["state"] = state
        if "steps" in arguments:
            patch_data["replace_steps"] = replacement_steps
        if "step_note" in arguments:
            patch_data["step_note"] = step_note

        task = await self._store.apply_patch(
            task.id,
            TaskPatch.model_validate(patch_data),
            owner_conversation_id=conversation_id,
        )

        return _task_full(task)


class TaskGetTool(BaseTool):

    # 函数说明：TaskGetTool.__init__
    # 用途：初始化 TaskGetTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `FileTaskStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: FileTaskStore) -> None:
        self._store = store

    # 函数说明：TaskGetTool.definition
    # 用途：提供 TaskGetTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_get",
            record_output=False,
            description=(
                "读取当前会话一个任务的完整目标、步骤、状态、约束、事实"
                "和关联记录。注入信息被折叠、需要精确步骤/revision，或"
                "更新冲突后重新确认时使用；已有完整且有效的记录时不重复读取，"
                "不查其他会话的任务。task_list 只给总览，本工具给详情；"
                "task_update 才会写回。成功返回保存的任务快照，不执行步骤，"
                "也不重新验证记录中的完成结论。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": (
                            "当前会话任务的完整 ID、4–32 位唯一十六进制前缀，"
                            "或 current（最近的 active/paused 任务）；pending/"
                            "已关闭任务使用 ID。"
                        ),
                    }
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：TaskGetTool.execute
    # 用途：执行TaskGetTool，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；返回 `await self._execute(arguments, context=None)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._execute`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute(arguments, context=None)

    # 函数说明：TaskGetTool.execute_with_context
    # 用途：执行上下文，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；返回
    # `await self._execute(arguments, context=context)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._execute`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return await self._execute(arguments, context=context)

    # 函数说明：TaskGetTool._execute
    # 用途：执行TaskGetTool，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `task_id`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext | None`。
    # 返回：类型 `dict[str, Any]`；返回 `_task_full(task)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_conversation_id` →
    # `_resolve_owned` → `_task_full`。
    # 分支与异常：
    #   当 `not isinstance(task_id, str) or not task_id.strip()` 时，抛出
    # `ValueError("'task_id' must be a non-empty string")`。
    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        task_id = arguments.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("'task_id' must be a non-empty string")
        conversation_id = _require_conversation_id(context)
        task = await _resolve_owned(self._store, task_id, conversation_id)
        return _task_full(task)


class TaskListTool(BaseTool):

    # 函数说明：TaskListTool.__init__
    # 用途：初始化 TaskListTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `FileTaskStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: FileTaskStore) -> None:
        self._store = store

    # 函数说明：TaskListTool.definition
    # 用途：提供 TaskListTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_list",
            record_output=False,
            description=(
                "按可选状态列出当前会话任务的简要记录。用户要求"
                "任务总览，或确实需要定位已有任务时使用；已知任务 ID 并需要"
                "详情时直接 task_get，不靠反复列表确认已完成工作。列表"
                "只有概要、进度和更新时间，完整步骤/约束用 task_get，变更用"
                "task_update。成功返回最多 limit 条已保存记录；不代表全部"
                "任务都已列出，也不证明实际工作通过验收。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [s.value for s in TaskStatus],
                        "description": "可选的已保存任务状态过滤；省略时返回各状态的任务，不改变状态。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_LIST_LIMIT,
                        "description": (
                            f"返回数量上限，默认 50，范围 1–{_MAX_LIST_LIMIT}；"
                            "按总览需要取值，不为查单个已知任务扩大数量。"
                        ),
                    },
                },
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：TaskListTool.execute
    # 用途：执行TaskListTool，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；返回 `await self._execute(arguments, context=None)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._execute`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute(arguments, context=None)

    # 函数说明：TaskListTool.execute_with_context
    # 用途：执行上下文，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；返回
    # `await self._execute(arguments, context=context)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._execute`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return await self._execute(arguments, context=context)

    # 函数说明：TaskListTool._execute
    # 用途：执行TaskListTool，供任务状态与步骤管理使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `limit`
    # 、`status`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext | None`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `count`、`tasks`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskStatus` →
    # `_require_conversation_id` → `self._store.list` → `_task_brief`。
    # 分支与异常：
    #   当 `not isinstance(limit, int) or not 1 <= limit <=…` 时，抛出 `ValueError(…)`。
    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        limit = arguments.get("limit", 50)
        if not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ValueError(
                f"'limit' must be an integer between 1 and {_MAX_LIST_LIMIT}"
            )

        status: TaskStatus | None = None
        raw_status = arguments.get("status")
        if raw_status is not None:
            status = TaskStatus(raw_status)

        conversation_id = _require_conversation_id(context)
        tasks = await self._store.list(
            limit=limit,
            status=status,
            owner_conversation_id=conversation_id,
        )
        return {
            "count": len(tasks),
            "tasks": [_task_brief(task) for task in tasks],
        }


# 函数说明：register_task_tools
# 用途：注册任务工具集合，供任务状态与步骤管理使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   store：持久化存储依赖，类型 `FileTaskStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.register` → `TaskCreateTool`
#  → `TaskUpdateTool` → `TaskGetTool` → `TaskListTool`。
def register_task_tools(registry: ToolRegistry, store: FileTaskStore) -> None:

    registry.register(TaskCreateTool(store))
    registry.register(TaskUpdateTool(store))
    registry.register(TaskGetTool(store))
    registry.register(TaskListTool(store))


# 函数说明：_task_full
# 用途：返回 `task.model_dump(mode='json', exclude={'applied_ops'})`，提供 任务状态与步
# 骤管理 的派生值。
# 参数：
#   task：当前任务记录，类型 `Task`。
# 返回：类型 `dict[str, Any]`；返回
# `task.model_dump(mode='json', exclude={'applied_ops'})`。
def _task_full(task: Task) -> dict[str, Any]:
    # applied_ops 是长任务的内部幂等记录，对模型没有用处
    return task.model_dump(mode="json", exclude={"applied_ops"})


# 函数说明：_resolve_owned
# 用途：解析或定位`owned`，供任务状态与步骤管理使用。
# 参数：
#   store：持久化存储依赖，类型 `FileTaskStore`。
#   task_id：目标任务标识，类型 `str`。
#   conversation_id：目标会话标识，类型 `str`。
# 返回：类型 `Task`；返回 `task`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`normalized.lower` →
# `store.active_for_conversation` → `store.resolve`。
# 分支与异常：
#   当 `task is None` 时，抛出 `KeyError(f'任务不存在：{task_id}')`。
async def _resolve_owned(
    store: FileTaskStore,
    task_id: str,
    conversation_id: str,
) -> Task:

    normalized = task_id.strip()
    if normalized.lower() == "current":
        task = await store.active_for_conversation(conversation_id)
    else:
        task = await store.resolve(
            normalized,
            owner_conversation_id=conversation_id,
        )
    if task is None:
        raise KeyError(f"任务不存在：{task_id}")
    return task


# 函数说明：_require_conversation_id
# 用途：获取并校验必需的会话标识，供任务状态与步骤管理使用。
# 参数：
#   context：本次操作的上下文对象，类型 `ToolExecutionContext | None`。
# 返回：类型 `str`；返回 `context.conversation_id`。
# 分支与异常：
#   当 `context is None or not context.conversation_id` 时，抛出
# `ValueError('task tool requires conversation context')`。
def _require_conversation_id(
    context: ToolExecutionContext | None,
) -> str:

    if context is None or not context.conversation_id:
        raise ValueError("task tool requires conversation context")
    return context.conversation_id


# 函数说明：_task_brief
# 用途：返回
# `{'id': task.id, 'title': task.title, 'status': task.status.value, 'priority':…`，提供
#  任务状态与步骤管理 的派生值。
# 参数：
#   task：当前任务记录，类型 `Task`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `id`、`title`、`status`、`priority`、
# `goal`、`progress`、`updated_at`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`task.updated_at.isoformat`。
def _task_brief(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status.value,
        "priority": task.priority.value,
        "goal": task.goal,
        "progress": task.progress_summary,
        "updated_at": task.updated_at.isoformat(),
    }


# 函数说明：_build_steps
# 用途：构建步骤集合，供任务状态与步骤管理使用。
# 参数：
#   raw_steps：传给 `isinstance` 的输入，类型 `object`。
# 返回：类型 `tuple[TaskStep, ...]`；按分支返回 `()`；`tuple(steps)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskStep` → `_optional_acceptance`。
# 分支与异常：
#   当 `raw_steps is None` 时，返回 `()`。
#   当 `not isinstance(raw_steps, list)` 时，抛出 `ValueError("'steps' must be a list")`
# 。
#   当 `not isinstance(item, dict)` 时，抛出
# `ValueError(f'steps[{index}] must be an object')`。
#   当 `not isinstance(title, str) or not title.strip()` 时，抛出 `ValueError(…)`。
def _build_steps(raw_steps: object) -> tuple[TaskStep, ...]:
    if raw_steps is None:
        return ()
    if not isinstance(raw_steps, list):
        raise ValueError("'steps' must be a list")
    steps: list[TaskStep] = []
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            raise ValueError(f"steps[{index}] must be an object")
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"steps[{index}].title must be a non-empty string")
        note = item.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError(f"steps[{index}].note must be a string")
        steps.append(
            TaskStep(
                # 短编号：长任务的 Manager 和审计报告按步骤 ID 引用，32 位随机串容易抄错
                id=f"s{index + 1}",
                title=title,
                note=note,
                acceptance=_optional_acceptance(item, index),
            )
        )
    return tuple(steps)


# 函数说明：_build_update_steps
# 用途：构建步骤集合，供任务状态与步骤管理使用。
# 参数：
#   raw_steps：传给 `isinstance` 的输入，类型 `object`。
# 返回：类型 `tuple[TaskStep, ...]`；返回 `tuple(steps)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`uuid4` → `TaskStep` →
# `TaskStepStatus` → `_optional_acceptance`。
# 分支与异常：
#   当 `not isinstance(raw_steps, list)` 时，抛出 `ValueError("'steps' must be a list")`
# 。
#   当 `not isinstance(item, dict)` 时，抛出
# `ValueError(f'steps[{index}] must be an object')`。
#   当 `not isinstance(title, str) or not title.strip()` 时，抛出 `ValueError(…)`。
#   当 `not isinstance(step_id, str) or not step_id.strip()` 时，抛出
# `ValueError(f'steps[{index}].id must be a non-empty string')`。
def _build_update_steps(raw_steps: object) -> tuple[TaskStep, ...]:

    if not isinstance(raw_steps, list):
        raise ValueError("'steps' must be a list")
    steps: list[TaskStep] = []
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            raise ValueError(f"steps[{index}] must be an object")
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"steps[{index}].title must be a non-empty string")
        step_id = item.get("id") or uuid4().hex
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError(f"steps[{index}].id must be a non-empty string")
        note = item.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError(f"steps[{index}].note must be a string")
        raw_status = item.get("status", TaskStepStatus.TODO.value)
        if raw_status not in _TOOL_STEP_STATUSES:
            raise ValueError(f"steps[{index}].status is not allowed: {raw_status}")
        steps.append(
            TaskStep(
                id=step_id,
                title=title,
                status=TaskStepStatus(raw_status),
                note=note,
                acceptance=_optional_acceptance(item, index),
            )
        )
    return tuple(steps)


# 函数说明：_optional_acceptance
# 用途：在任务状态与步骤管理中处理 `_optional_acceptance`，通过 `item.get` 完成首个内部
# 处理步骤。
# 参数：
#   item：当前集合元素，类型 `dict[str, Any]`；读取键 `acceptance`。
#   index：当前位置或索引，类型 `int`。
# 返回：类型 `str | None`；返回 `acceptance`。
# 分支与异常：
#   当 `acceptance is not None and (not isinstance(acceptance, str))` 时，抛出
# `ValueError(f'steps[{index}].acceptance must be a string')`。
def _optional_acceptance(item: dict[str, Any], index: int) -> str | None:
    acceptance = item.get("acceptance")
    if acceptance is not None and not isinstance(acceptance, str):
        raise ValueError(f"steps[{index}].acceptance must be a string")
    return acceptance


__all__ = [
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskUpdateTool",
    "register_task_tools",
]
