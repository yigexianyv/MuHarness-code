
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from app.models.types import ToolDefinition
from app.tools.base import BaseTool
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry

from .models import Schedule, ScheduleKind
from .scheduler import AutomationScheduler

_KIND_NAMES = {kind.value for kind in ScheduleKind}


class AutomationCreateTool(BaseTool):

    # 函数说明：AutomationCreateTool.__init__
    # 用途：初始化 AutomationCreateTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   scheduler：`scheduler`输入或配置值，类型 `AutomationScheduler`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._scheduler`。
    def __init__(self, scheduler: AutomationScheduler) -> None:
        self._scheduler = scheduler

    # 函数说明：AutomationCreateTool.definition
    # 用途：提供 AutomationCreateTool 的模型可见定义，包含名称、说明、参数结构及权限声明
    # 。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="automation_create",
            record_output=False,
            description=(
                "保存未来触发 Agent Run 的自动化，支持一次性、固定间隔或 cron。"
                "仅在用户要求定时、周期执行或稍后跟进时使用；当前就要完成的工作"
                "直接执行，跨步骤进度用 Task 跟踪，不为普通计划自动创建调度。"
                "prompt 只写触发时的执行内容，时间、频率、时区放在对应调度参数，"
                "避免触发后再次创建自动化。成功表示配置已保存且下一次触发已安排，"
                "不表示目标已执行或未来运行必然成功；核对已有配置先用 automation_list "
                "或 automation_get，"
                "不要重复创建相同调度。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "用于识别这条调度的简短标题，不代替实际执行指令。",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "触发时真正要执行的完整指令，只含执行内容，不重复时间或频率。"
                            "例如每天 22:00 总结进度，prompt 只写“总结项目进度”，"
                            "时间写入调度参数，避免递归创建自动化。"
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "enum": [kind.value for kind in ScheduleKind],
                        "description": (
                            "once 为一次性，需 run_at；interval 为固定间隔，需 "
                            "interval_seconds；cron 为周期日程，需 cron_expr。"
                        ),
                    },
                    "run_at": {
                        "type": "string",
                        "description": (
                            "仅 once 使用：未来的 ISO8601 时间，必须带时区偏移，"
                            "如 2026-10-01T09:00:00+08:00；不能用 timezone 替代此偏移。"
                        ),
                    },
                    "interval_seconds": {
                        "type": "number",
                        "description": (
                            "仅 interval 使用：大于 0 的间隔秒数，首次触发在创建后的该间隔。"
                        ),
                    },
                    "cron_expr": {
                        "type": "string",
                        "description": (
                            "仅 cron 使用：五段 crontab 表达式，如 \"0 9 * * *\"，"
                            "按 timezone 解释；不填自然语言或带秒的六段表达式。"
                        ),
                    },
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA 时区名，如 Asia/Shanghai，默认 UTC；cron 的日程按此解释。"
                        ),
                    },
                },
                "required": ["title", "prompt", "kind"],
                "additionalProperties": False,
            },
            strict=True,
        )

    # 函数说明：AutomationCreateTool.execute
    # 用途：执行AutomationCreateTool，供定时任务调度使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("automation_create requires conversation context")

    # 函数说明：AutomationCreateTool.execute_with_context
    # 用途：执行上下文，供定时任务调度使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `title`
    # 、`prompt`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；返回 `_automation_brief(automation)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_non_empty` →
    # `build_schedule_and_next` → `self._scheduler.create_automation` →
    # `_automation_brief`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        conversation_id = context.conversation_id
        title = _require_non_empty(arguments, "title")
        prompt = _require_non_empty(arguments, "prompt")
        schedule, next_run_at = build_schedule_and_next(arguments)

        automation = await self._scheduler.create_automation(
            title=title,
            prompt=prompt,
            conversation_id=conversation_id,
            schedule=schedule,
            next_run_at=next_run_at,
        )
        return _automation_brief(automation)


class AutomationListTool(BaseTool):

    # 函数说明：AutomationListTool.__init__
    # 用途：初始化 AutomationListTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   scheduler：`scheduler`输入或配置值，类型 `AutomationScheduler`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._scheduler`。
    def __init__(self, scheduler: AutomationScheduler) -> None:
        self._scheduler = scheduler

    # 函数说明：AutomationListTool.definition
    # 用途：提供 AutomationListTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="automation_list",
            record_output=False,
            description=(
                "列出当前会话的自动化摘要（最多 50 条），含状态、下一次触发及最近 Run ID。"
                "需要找到已有调度、避免重复创建或核对状态时使用；已知 ID 且需要完整"
                "prompt 或计划时用 automation_get。列表不会触发或修改调度，返回记录"
                "不等于自动化执行内容已完成；没有相关问题时不重复轮询。"
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            strict=True,
        )

    # 函数说明：AutomationListTool.execute
    # 用途：执行AutomationListTool，供定时任务调度使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("automation_list requires conversation context")

    # 函数说明：AutomationListTool.execute_with_context
    # 用途：执行上下文，供定时任务调度使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `automations`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._scheduler.list` →
    # `_automation_brief`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        automations = await self._scheduler.list(
            conversation_id=context.conversation_id,
        )
        return {"automations": [_automation_brief(item) for item in automations]}


class AutomationGetTool(BaseTool):

    # 函数说明：AutomationGetTool.__init__
    # 用途：初始化 AutomationGetTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   scheduler：`scheduler`输入或配置值，类型 `AutomationScheduler`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._scheduler`。
    def __init__(self, scheduler: AutomationScheduler) -> None:
        self._scheduler = scheduler

    # 函数说明：AutomationGetTool.definition
    # 用途：提供 AutomationGetTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="automation_get",
            record_output=False,
            description=(
                "按已知自动化 ID 读取完整配置和状态，用于核对 prompt、计划或下一次触发。"
                "未知 ID 先用 automation_list；无需详情时不要反复读取。读取不会"
                "触发或恢复运行，状态和 last_run_id 只是调度记录，不代替该 Run 的结果验收。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "从创建回执或列表取得的完整自动化 ID，不是 Task/Run ID。"},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
            strict=True,
        )

    # 函数说明：AutomationGetTool.execute
    # 用途：执行AutomationGetTool，供定时任务调度使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `id`。
    # 返回：类型 `dict[str, Any]`；返回 `_automation_full(automation)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_non_empty` →
    # `_automation_full`。
    # 分支与异常：
    #   当 `automation is None` 时，抛出 `KeyError(f'自动化不存在：{automation_id}')`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        automation_id = _require_non_empty(arguments, "id")
        automation = await self._scheduler.get(automation_id)
        if automation is None:
            raise KeyError(f"自动化不存在：{automation_id}")
        return _automation_full(automation)


class AutomationCancelTool(BaseTool):

    # 函数说明：AutomationCancelTool.__init__
    # 用途：初始化 AutomationCancelTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   scheduler：`scheduler`输入或配置值，类型 `AutomationScheduler`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._scheduler`。
    def __init__(self, scheduler: AutomationScheduler) -> None:
        self._scheduler = scheduler

    # 函数说明：AutomationCancelTool.definition
    # 用途：提供 AutomationCancelTool 的模型可见定义，包含名称、说明、参数结构及权限声明
    # 。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="automation_cancel",
            record_output=False,
            description=(
                "将指定自动化设为 cancelled 并移除后续触发。用户明确不再需要该调度"
                "时使用；暂时停用后还要恢复时用 automation_pause，不用取消代替暂停。"
                "成功只确认调度已取消（重复取消可返回已有状态），不撤销既有结果、"
                "不终止已启动的 Run，也不证明原目标已经完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要停止后续触发的完整自动化 ID，先确认与用户目标对应。"},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
            strict=True,
        )

    # 函数说明：AutomationCancelTool.execute
    # 用途：执行AutomationCancelTool，供定时任务调度使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `id`。
    # 返回：类型 `dict[str, Any]`；返回 `_automation_brief(automation)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_non_empty` →
    # `self._scheduler.cancel` → `_automation_brief`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        automation_id = _require_non_empty(arguments, "id")
        automation = await self._scheduler.cancel(automation_id)
        return _automation_brief(automation)


class AutomationPauseTool(BaseTool):

    # 函数说明：AutomationPauseTool.__init__
    # 用途：初始化 AutomationPauseTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   scheduler：`scheduler`输入或配置值，类型 `AutomationScheduler`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._scheduler`。
    def __init__(self, scheduler: AutomationScheduler) -> None:
        self._scheduler = scheduler

    # 函数说明：AutomationPauseTool.definition
    # 用途：提供 AutomationPauseTool 的模型可见定义，包含名称、说明、参数结构及权限声明
    # 。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="automation_pause",
            record_output=False,
            description=(
                "将 active 自动化设为 paused，保留配置并移除未来触发。用户暂时停用"
                "日程且可能恢复时使用；永久不再需要用 automation_cancel，不对"
                "其他状态重复暂停。成功只确认调度暂停，不终止已启动的 Run；"
                "重新安排触发用 automation_resume，暂停不代表工作已完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "已确认处于 active 状态的完整自动化 ID。"},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
            strict=True,
        )

    # 函数说明：AutomationPauseTool.execute
    # 用途：执行AutomationPauseTool，供定时任务调度使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `id`。
    # 返回：类型 `dict[str, Any]`；返回 `_automation_brief(automation)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_non_empty` →
    # `self._scheduler.pause` → `_automation_brief`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        automation_id = _require_non_empty(arguments, "id")
        automation = await self._scheduler.pause(automation_id)
        return _automation_brief(automation)


class AutomationResumeTool(BaseTool):

    # 函数说明：AutomationResumeTool.__init__
    # 用途：初始化 AutomationResumeTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   scheduler：`scheduler`输入或配置值，类型 `AutomationScheduler`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._scheduler`。
    def __init__(self, scheduler: AutomationScheduler) -> None:
        self._scheduler = scheduler

    # 函数说明：AutomationResumeTool.definition
    # 用途：提供 AutomationResumeTool 的模型可见定义，包含名称、说明、参数结构及权限声明
    # 。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="automation_resume",
            record_output=False,
            description=(
                "将 paused 自动化恢复为 active，并按原计划重新计算、安排下一次触发。"
                "用户要求恢复已有调度时使用，不重新创建相同日程；只接受 paused，"
                "不能用它恢复 cancelled 或已完成记录。错过时间的一次性调度可能立即"
                "安排执行。成功表示调度恢复，不表示触发已完成或执行内容通过验收。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "已确认处于 paused 状态的完整自动化 ID。"},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
            strict=True,
        )

    # 函数说明：AutomationResumeTool.execute
    # 用途：执行AutomationResumeTool，供定时任务调度使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `id`。
    # 返回：类型 `dict[str, Any]`；返回 `_automation_brief(automation)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_non_empty` →
    # `self._scheduler.resume` → `_automation_brief`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        automation_id = _require_non_empty(arguments, "id")
        automation = await self._scheduler.resume(automation_id)
        return _automation_brief(automation)


# 函数说明：register_automation_tools
# 用途：注册自动化任务工具集合，供定时任务调度使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   scheduler：传给 `AutomationCreateTool` 的输入，类型 `AutomationScheduler`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.register` →
# `AutomationCreateTool` → `AutomationListTool` → `AutomationGetTool` →
# `AutomationCancelTool` → `AutomationPauseTool`；另有 1 个调用点。
def register_automation_tools(
    registry: ToolRegistry,
    scheduler: AutomationScheduler,
) -> None:
    registry.register(AutomationCreateTool(scheduler))
    registry.register(AutomationListTool(scheduler))
    registry.register(AutomationGetTool(scheduler))
    registry.register(AutomationCancelTool(scheduler))
    registry.register(AutomationPauseTool(scheduler))
    registry.register(AutomationResumeTool(scheduler))




# 函数说明：build_schedule_and_next
# 用途：构建调度，供定时任务调度使用。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `kind`、
# `timezone`、`run_at`、`interval_seconds`、`cron_expr`。
# 返回：类型 `tuple[Schedule, datetime]`；按分支返回 `(schedule, run_at)`；
# `(schedule, now + timedelta(seconds=interval))`；
# `(schedule, next_local.astimezone(UTC))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ScheduleKind` → `ZoneInfo` →
# `datetime.now` → `datetime.fromisoformat` → `run_at.utcoffset` → `run_at.astimezone`；
# 另有 6 个调用点。
# 分支与异常：
#   当 `not isinstance(raw_kind, str) or raw_kind not in _KIND_NAMES` 时，抛出
# `ValueError(…)`。
#   当 `not isinstance(timezone, str)` 时，抛出
# `ValueError("'timezone' must be a string")`。
#   捕获 `Exception` 后，转换或抛出 `ValueError(f'invalid timezone: {timezone}')`。
#   `kind is ScheduleKind.ONCE` 分支在完成前置处理后返回 `(schedule, run_at)`。
#   捕获 `ValueError` 后，转换或抛出 `ValueError(f'invalid run_at: {raw_run_at!r}')`。
#   捕获 `Exception` 后，转换或抛出 `ValueError(f'invalid cron_expr: {raw_cron!r}')`。
def build_schedule_and_next(
    arguments: dict[str, Any],
) -> tuple[Schedule, datetime]:

    raw_kind = arguments.get("kind")
    if not isinstance(raw_kind, str) or raw_kind not in _KIND_NAMES:
        raise ValueError(
            f"'kind' must be one of: {', '.join(sorted(_KIND_NAMES))}"
        )
    kind = ScheduleKind(raw_kind)
    timezone = arguments.get("timezone") or "UTC"
    if not isinstance(timezone, str):
        raise ValueError("'timezone' must be a string")
    try:
        tz = ZoneInfo(timezone)
    except Exception as exc:
        raise ValueError(f"invalid timezone: {timezone}") from exc

    now = datetime.now(UTC)

    if kind is ScheduleKind.ONCE:
        raw_run_at = arguments.get("run_at")
        if not isinstance(raw_run_at, str) or not raw_run_at.strip():
            raise ValueError("'run_at' is required for kind=once")
        try:
            run_at = datetime.fromisoformat(raw_run_at)
        except ValueError as exc:
            raise ValueError(f"invalid run_at: {raw_run_at!r}") from exc
        if run_at.tzinfo is None or run_at.utcoffset() is None:
            raise ValueError(
                "'run_at' must include a timezone offset "
                "(e.g. 2026-08-20T09:00:00+08:00)"
            )
        run_at = run_at.astimezone(UTC)
        if run_at <= now:
            raise ValueError("'run_at' must be in the future")
        schedule = Schedule(
            kind=kind,
            run_at=run_at,
            timezone=timezone,
        )
        return schedule, run_at

    if kind is ScheduleKind.INTERVAL:
        raw_interval = arguments.get("interval_seconds")
        if not isinstance(raw_interval, (int, float)):
            raise ValueError("'interval_seconds' is required for kind=interval")
        interval = float(raw_interval)
        if interval <= 0:
            raise ValueError("'interval_seconds' must be > 0")
        schedule = Schedule(
            kind=kind,
            interval_seconds=interval,
            timezone=timezone,
        )
        return schedule, now + timedelta(seconds=interval)

    raw_cron = arguments.get("cron_expr")
    if not isinstance(raw_cron, str) or not raw_cron.strip():
        raise ValueError("'cron_expr' is required for kind=cron")
    try:
        trigger = CronTrigger.from_crontab(raw_cron, timezone=tz)
    except Exception as exc:
        raise ValueError(f"invalid cron_expr: {raw_cron!r}") from exc
    next_local = trigger.get_next_fire_time(None, now.astimezone(tz))
    if next_local is None:
        raise ValueError(f"cron_expr has no future fire time: {raw_cron!r}")
    schedule = Schedule(
        kind=kind,
        cron_expr=raw_cron,
        timezone=timezone,
    )
    return schedule, next_local.astimezone(UTC)


# 函数说明：_require_non_empty
# 用途：获取并校验必需的`non_empty`，供定时任务调度使用。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
# 返回：类型 `str`；返回 `value.strip()`。
# 分支与异常：
#   当 `not isinstance(value, str) or not value.strip()` 时，抛出
# `ValueError(f"'{key}' must be a non-empty string")`。
def _require_non_empty(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value.strip()


# 函数说明：_automation_brief
# 用途：返回 `{'id': automation.id, 'title': automation.title, 'status': automation.
# status.value, 'kind…`，提供 定时任务调度 的派生值。
# 参数：
#   automation：自动化任务输入或配置值。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `id`、`title`、`status`、`kind`、
# `next_run_at`、`last_run_id`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`automation.next_run_at.isoformat`。
def _automation_brief(automation) -> dict[str, Any]:
    return {
        "id": automation.id,
        "title": automation.title,
        "status": automation.status.value,
        "kind": automation.schedule.kind.value,
        "next_run_at": (
            automation.next_run_at.isoformat()
            if automation.next_run_at is not None
            else None
        ),
        "last_run_id": automation.last_run_id,
    }


# 函数说明：_automation_full
# 用途：返回 `json.loads(automation.model_dump_json())`，提供 定时任务调度 的派生值。
# 参数：
#   automation：自动化任务输入或配置值。
# 返回：类型 `dict[str, Any]`；返回 `json.loads(automation.model_dump_json())`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` →
# `automation.model_dump_json`。
def _automation_full(automation) -> dict[str, Any]:
    return json.loads(automation.model_dump_json())


__all__ = [
    "AutomationCancelTool",
    "AutomationCreateTool",
    "AutomationGetTool",
    "AutomationListTool",
    "AutomationPauseTool",
    "AutomationResumeTool",
    "build_schedule_and_next",
    "register_automation_tools",
]
