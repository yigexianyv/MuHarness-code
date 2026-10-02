
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.domain.conversation import ConversationSource, TriggerContext
from app.domain.conversation.service import ConversationService

from .models import Automation, AutomationStatus, Schedule, ScheduleKind
from .store import SQLiteAutomationStore

logger = logging.getLogger("muharness.automation.scheduler")

_JOB_PREFIX = "automation-"


class AutomationScheduler:

    # 函数说明：AutomationScheduler.__init__
    # 用途：初始化 AutomationScheduler；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteAutomationStore`。
    #   conversation_service：会话输入或配置值，类型 `ConversationService`。
    #   timezone：传给 `ZoneInfo` 的输入，类型 `str`；默认 `'UTC'`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ZoneInfo` → `AsyncIOScheduler`。
    # 副作用与资源：
    #   更新对象字段：`self._store`、`self._conversation_service`、`self._timezone`、
    # `self._scheduler`、`self._job_ids`、`self._running`。
    def __init__(
        self,
        store: SQLiteAutomationStore,
        conversation_service: ConversationService,
        *,
        timezone: str = "UTC",
    ) -> None:
        self._store = store
        self._conversation_service = conversation_service
        self._timezone = ZoneInfo(timezone)
        self._scheduler = AsyncIOScheduler(timezone=self._timezone)
        self._job_ids: dict[str, str] = {}
        self._running: set[str] = set()


    # 函数说明：AutomationScheduler.start
    # 用途：启动调度器并从存储中恢复仍有效的自动化任务。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.initialize` →
    # `self._store.list` → `self._restore` → `self._scheduler.start`。
    async def start(self) -> None:

        """启动调度器并从存储中恢复仍有效的自动化任务。"""
        await self._store.initialize()
        active = await self._store.list(status=AutomationStatus.ACTIVE)
        for automation in active:
            await self._restore(automation)
        self._scheduler.start()

    # 函数说明：AutomationScheduler.shutdown
    # 用途：关闭AutomationScheduler，供定时任务调度使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._scheduler.shutdown` →
    # `logger.debug` → `self._running.clear` → `self._job_ids.clear`。
    # 分支与异常：
    #   捕获 `Exception` 后，执行异常处理调用 `logger.debug`。
    async def shutdown(self) -> None:

        try:
            self._scheduler.shutdown(wait=False)
        except Exception:
            logger.debug("automation scheduler already shut down", exc_info=True)
        self._running.clear()
        self._job_ids.clear()


    # 函数说明：AutomationScheduler.create_automation
    # 用途：保存自动化配置并安排下一次触发。
    # 参数：
    #   title：面向用户的标题，类型 `str`。
    #   prompt：本次调用使用的提示文本，类型 `str`。
    #   conversation_id：目标会话标识，类型 `str | None`。
    #   schedule：调度输入或配置值，类型 `Schedule`。
    #   next_run_at：下一次运行时间，类型 `datetime`。
    # 返回：类型 `Automation`；返回 `automation`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.create` →
    # `self._schedule`。
    async def create_automation(
        self,
        *,
        title: str,
        prompt: str,
        conversation_id: str | None,
        schedule: Schedule,
        next_run_at: datetime,
    ) -> Automation:
        """保存自动化配置并安排下一次触发。"""
        automation = await self._store.create(
            title=title,
            prompt=prompt,
            conversation_id=conversation_id,
            schedule=schedule,
            next_run_at=next_run_at,
        )
        self._schedule(automation, automation.next_run_at)
        return automation

    # 函数说明：AutomationScheduler.get
    # 用途：获取AutomationScheduler，供定时任务调度使用。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    # 返回：类型 `Automation | None`；返回 `await self._store.get(automation_id)`。
    async def get(self, automation_id: str) -> Automation | None:
        return await self._store.get(automation_id)

    # 函数说明：AutomationScheduler.resolve
    # 用途：解析或定位AutomationScheduler，供定时任务调度使用。
    # 参数：
    #   identifier：待规范化的标识，类型 `str`。
    # 返回：类型 `Automation | None`；返回 `await self._store.resolve(identifier)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.resolve`。
    async def resolve(self, identifier: str) -> Automation | None:

        return await self._store.resolve(identifier)

    # 函数说明：AutomationScheduler.list
    # 用途：列出AutomationScheduler，供定时任务调度使用。
    # 参数：
    #   status：目标状态，类型 `AutomationStatus | str | None`；默认 `None`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `50`。
    # 返回：类型 `tuple[Automation, ...]`；返回 `await self._store.list(status=status,
    # conversation_id=conversation_id, limit=limit)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.list`。
    async def list(
        self,
        *,
        status: AutomationStatus | str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
    ) -> tuple[Automation, ...]:
        return await self._store.list(
            status=status,
            conversation_id=conversation_id,
            limit=limit,
        )

    # 函数说明：AutomationScheduler.cancel
    # 用途：取消AutomationScheduler，供定时任务调度使用。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    # 返回：类型 `Automation`；按分支返回 `automation`；`updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.require` →
    # `self._store.update_status` → `self._remove_job`。
    # 分支与异常：
    #   当 `automation.status is AutomationStatus.CANCELLED` 时，返回 `automation`。
    async def cancel(self, automation_id: str) -> Automation:
        automation = await self._store.require(automation_id)
        if automation.status is AutomationStatus.CANCELLED:
            return automation
        updated = await self._store.update_status(
            automation_id,
            AutomationStatus.CANCELLED,
        )
        self._remove_job(automation_id)
        return updated

    # 函数说明：AutomationScheduler.delete_for_conversation
    # 用途：删除会话，供定时任务调度使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `int`；返回
    # `await self._store.delete_for_conversation(conversation_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._store.list_for_conversation` → `self._remove_job` →
    # `self._store.delete_for_conversation`。
    async def delete_for_conversation(self, conversation_id: str) -> int:

        automations = await self._store.list_for_conversation(conversation_id)
        for automation in automations:
            self._remove_job(automation.id)
        return await self._store.delete_for_conversation(conversation_id)

    # 函数说明：AutomationScheduler.pause
    # 用途：暂停已安排的触发，同时保存暂停状态。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    # 返回：类型 `Automation`；返回 `updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.require` →
    # `self._store.update_status` → `self._remove_job`。
    # 分支与异常：
    #   当 `automation.status is not AutomationStatus.ACTIVE` 时，抛出 `ValueError(…)`。
    async def pause(self, automation_id: str) -> Automation:
        """暂停已安排的触发，同时保存暂停状态。"""
        automation = await self._store.require(automation_id)
        if automation.status is not AutomationStatus.ACTIVE:
            raise ValueError(
                f"only active automation can be paused: {automation_id} "
                f"({automation.status.value})"
            )
        updated = await self._store.update_status(
            automation_id,
            AutomationStatus.PAUSED,
        )
        self._remove_job(automation_id)
        return updated

    # 函数说明：AutomationScheduler.resume
    # 用途：恢复自动化并重新计算下一次触发时间。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    # 返回：类型 `Automation`；返回 `updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.require` →
    # `datetime.now` → `self._next_future` → `self._store.update_status` →
    # `self._schedule`。
    # 分支与异常：
    #   当 `automation.status is not AutomationStatus.PAUSED` 时，抛出 `ValueError(…)`。
    async def resume(self, automation_id: str) -> Automation:
        """恢复自动化并重新计算下一次触发时间。"""
        automation = await self._store.require(automation_id)
        if automation.status is not AutomationStatus.PAUSED:
            raise ValueError(
                f"only paused automation can be resumed: {automation_id} "
                f"({automation.status.value})"
            )
        now = datetime.now(UTC)
        if automation.schedule.kind is ScheduleKind.ONCE:
            next_run = automation.next_run_at or now
            if next_run < now:
                next_run = now
        else:
            next_run = self._next_future(automation, now)
        updated = await self._store.update_status(
            automation_id,
            AutomationStatus.ACTIVE,
            next_run_at=next_run,
        )
        self._schedule(updated, next_run)
        return updated


    # 函数说明：AutomationScheduler._restore
    # 用途：恢复AutomationScheduler，供定时任务调度使用。
    # 参数：
    #   automation：传给 `self._schedule` 的输入，类型 `Automation`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` → `self._schedule`
    # → `self._store.set_next_run_at` → `self._store.update_status` →
    # `self._next_future`。
    # 分支与异常：
    #   `automation.schedule.kind is ScheduleKind.ONCE` 分支在完成前置处理后返回 `None`
    # 。
    #   `not expired` 分支在完成前置处理后返回 `None`。
    async def _restore(self, automation: Automation) -> None:

        now = datetime.now(UTC)
        if automation.schedule.kind is ScheduleKind.ONCE:
            expired = (
                automation.next_run_at is None
                or automation.next_run_at <= now
            )
            if not expired:
                self._schedule(automation, automation.next_run_at)
                return
            if automation.last_run_id is None:
                refreshed = await self._store.set_next_run_at(
                    automation.id,
                    now,
                )
                self._schedule(refreshed, now)
            else:
                await self._store.update_status(
                    automation.id,
                    AutomationStatus.COMPLETED,
                )
            return

        next_run = self._next_future(automation, now)
        if automation.next_run_at is None or next_run != automation.next_run_at:
            refreshed = await self._store.set_next_run_at(automation.id, next_run)
            automation = refreshed
        self._schedule(automation, next_run)

    # 函数说明：AutomationScheduler._schedule
    # 用途：在定时任务调度中处理 `_schedule`，通过 `self._scheduler.reschedule_job` 完成
    # 首个内部处理步骤。
    # 参数：
    #   automation：自动化任务输入或配置值，类型 `Automation`。
    #   run_at：传给 `DateTrigger` 的输入，类型 `datetime`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._scheduler.reschedule_job` →
    #  `DateTrigger` → `self._job_ids.pop` → `self._scheduler.add_job` →
    # `self._job_func`。
    # 分支与异常：
    #   捕获 `Exception` 后，执行异常处理调用 `self._job_ids.pop`。
    def _schedule(self, automation: Automation, run_at: datetime) -> None:

        automation_key = automation.id
        if automation_key in self._job_ids:
            try:
                self._scheduler.reschedule_job(
                    self._job_ids[automation_key],
                    trigger=DateTrigger(run_at),
                )
                return
            except Exception:
                self._job_ids.pop(automation_key, None)
        job = self._scheduler.add_job(
            self._job_func(automation.id),
            trigger=DateTrigger(run_at),
            id=_JOB_PREFIX + automation.id,
            replace_existing=True,
        )
        self._job_ids[automation.id] = job.id

    # 函数说明：AutomationScheduler._remove_job
    # 用途：移除`job`，供定时任务调度使用。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._job_ids.pop` →
    # `self._scheduler.remove_job`。
    # 分支与异常：
    #   当 `job_id is None` 时，返回 `None`。
    #   捕获 `Exception` 后，忽略该异常并继续当前流程。
    def _remove_job(self, automation_id: str) -> None:
        job_id = self._job_ids.pop(automation_id, None)
        if job_id is None:
            return
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass

    # 函数说明：AutomationScheduler._job_func
    # 用途：处理定时任务调度中的 `_job_func` 数据；结果及边界条件见下方说明。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    # 返回：类型 `Any`；返回 `_run`。
    def _job_func(self, automation_id: str) -> Any:
        # 函数说明：AutomationScheduler._job_func._run
        # 用途：运行AutomationScheduler，供定时任务调度使用。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._trigger` →
        # `logger.exception`。
        # 分支与异常：
        #   捕获 `Exception` 后，执行异常处理调用 `logger.exception`。
        # 闭包依赖：从外层读取 `automation_id`。
        async def _run() -> None:
            try:
                await self._trigger(automation_id)
            except Exception:
                logger.exception(
                    "automation trigger failed: %s",
                    automation_id,
                )

        return _run


    # 函数说明：AutomationScheduler._trigger
    # 用途：处理一次到期触发，创建运行并更新调度记录。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` →
    # `self._conversation_service.is_run_running` → `self._next_future` →
    # `self._store.mark_triggered` → `self._schedule` → `self._running.add`；另有 6 个调
    # 用点。
    # 分支与异常：
    #   当 `automation_id in self._running` 时，返回 `None`。
    #   当 `automation is None or automation.status is not…` 时，返回 `None`。
    #   `await self._conversation_service.is_run_running(…)` 分支在完成前置处理后返回
    # `None`。
    #   `automation.schedule.kind is ScheduleKind.ONCE` 分支在完成前置处理后返回 `None`
    # 。
    async def _trigger(self, automation_id: str) -> None:

        """处理一次到期触发，创建运行并更新调度记录。"""
        if automation_id in self._running:
            return
        automation = await self._store.get(automation_id)
        if automation is None or automation.status is not AutomationStatus.ACTIVE:
            return

        now = datetime.now(UTC)
        if automation.last_run_id is not None:
            if await self._conversation_service.is_run_running(
                automation.last_run_id
            ):
                next_run = self._next_future(automation, now)
                await self._store.mark_triggered(
                    automation_id,
                    last_run_id=automation.last_run_id,
                    last_run_at=automation.last_run_at or now,
                    next_run_at=next_run,
                )
                self._schedule(automation, next_run)
                return

        self._running.add(automation_id)
        try:
            dispatch = await self._conversation_service.dispatch(
                conversation_id=automation.conversation_id,
                content=automation.prompt,
                trigger=TriggerContext(
                    source=ConversationSource.AUTOMATION,
                    automation_id=automation.id,
                    scheduled_for=automation.next_run_at,
                    triggered_at=now,
                ),
                on_run_started=self._record_run_started(
                    automation_id,
                    triggered_at=now,
                ),
            )
            run_id = dispatch.run.id
            now = datetime.now(UTC)
            if automation.schedule.kind is ScheduleKind.ONCE:
                await self._store.update_status(
                    automation_id,
                    AutomationStatus.COMPLETED,
                )
                self._remove_job(automation_id)
                return

            next_run = self._next_future(automation, now)
            await self._store.mark_triggered(
                automation_id,
                last_run_id=run_id,
                last_run_at=now,
                next_run_at=next_run,
            )
            self._schedule(automation, next_run)
        finally:
            self._running.discard(automation_id)

    # 函数说明：AutomationScheduler._record_run_started
    # 用途：记录运行，供定时任务调度使用。
    # 参数：
    #   automation_id：自动化任务标识，类型 `str`。
    #   triggered_at：实际触发时间，类型 `datetime`。
    # 返回：类型 `Any`；返回 `_record`。
    def _record_run_started(
        self,
        automation_id: str,
        *,
        triggered_at: datetime,
    ) -> Any:

        # 函数说明：AutomationScheduler._record_run_started._record
        # 用途：记录AutomationScheduler，供定时任务调度使用。
        # 参数：
        #   run_id：目标运行标识，类型 `str`。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.mark_triggered`。
        # 闭包依赖：从外层读取 `automation_id`、`triggered_at`。
        async def _record(run_id: str) -> None:
            await self._store.mark_triggered(
                automation_id,
                last_run_id=run_id,
                last_run_at=triggered_at,
                next_run_at=None,
            )

        return _record


    # 函数说明：AutomationScheduler._build_trigger
    # 用途：构建`trigger`，供定时任务调度使用。
    # 参数：
    #   schedule：调度输入或配置值，类型 `Schedule`。
    # 返回：按分支返回 `IntervalTrigger(seconds=schedule.interval_seconds, timezone=tz)`
    # ；`CronTrigger.from_crontab(schedule.cron_expr, timezone=tz)`；
    # `DateTrigger(schedule.run_at)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ZoneInfo` → `IntervalTrigger` →
    # `CronTrigger.from_crontab` → `DateTrigger`。
    # 分支与异常：
    #   当 `schedule.kind is ScheduleKind.INTERVAL` 时，返回 `IntervalTrigger(…)`。
    #   当 `schedule.kind is ScheduleKind.CRON` 时，返回
    # `CronTrigger.from_crontab(schedule.cron_expr, timezone=tz)`。
    def _build_trigger(self, schedule: Schedule):
        tz = ZoneInfo(schedule.timezone)
        if schedule.kind is ScheduleKind.INTERVAL:
            return IntervalTrigger(
                seconds=schedule.interval_seconds,
                timezone=tz,
            )
        if schedule.kind is ScheduleKind.CRON:
            return CronTrigger.from_crontab(schedule.cron_expr, timezone=tz)
        return DateTrigger(schedule.run_at)

    # 函数说明：AutomationScheduler._next_future
    # 用途：在定时任务调度中处理 `_next_future`，通过 `self._build_trigger` 完成首个内部
    # 处理步骤。
    # 参数：
    #   automation：自动化任务输入或配置值，类型 `Automation`。
    #   now：`now`输入或配置值，类型 `datetime`。
    # 返回：类型 `datetime`；返回 `next_local.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._build_trigger` →
    # `now.astimezone` → `ZoneInfo` → `trigger.get_next_fire_time` →
    # `next_local.astimezone`。
    # 分支与异常：
    #   当 `next_local is None` 时，抛出 `ValueError(…)`。
    def _next_future(self, automation: Automation, now: datetime) -> datetime:

        trigger = self._build_trigger(automation.schedule)
        local_now = now.astimezone(ZoneInfo(automation.schedule.timezone))
        next_local = trigger.get_next_fire_time(None, local_now)
        if next_local is None:
            raise ValueError(
                f"automation {automation.id} has no future trigger time"
            )
        return next_local.astimezone(UTC)


__all__ = ["AutomationScheduler"]
