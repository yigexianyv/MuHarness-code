
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.domain.automation.models import (
    AutomationStatus,
    Schedule,
    ScheduleKind,
)
from app.domain.automation.scheduler import AutomationScheduler
from app.domain.automation.store import SQLiteAutomationStore
from app.domain.automation.tools import (
    AutomationCreateTool,
    build_schedule_and_next,
)
from app.domain.conversation import ConversationSource

_USER_MESSAGE = "提醒我交作业"


class _FakeRunRef:
    # 函数说明：_FakeRunRef.__init__
    # 用途：初始化 _FakeRunRef；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.id`。
    def __init__(self, run_id: str) -> None:
        self.id = run_id


class _FakeDispatch:
    # 函数说明：_FakeDispatch.__init__
    # 用途：初始化 _FakeDispatch；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_FakeRunRef`。
    # 副作用与资源：
    #   更新对象字段：`self.run`。
    def __init__(self, run_id: str) -> None:
        self.run = _FakeRunRef(run_id)


class FakeConversationService:

    # 函数说明：FakeConversationService.__init__
    # 用途：初始化 FakeConversationService；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.dispatched`、`self.running_run_ids`、`self.fail_on_dispatch`
    # 。
    def __init__(self) -> None:
        self.dispatched: list[dict] = []
        self.running_run_ids: set[str] = set()
        self.fail_on_dispatch: bool = False

    # 函数说明：FakeConversationService.dispatch
    # 用途：分发FakeConversationService，供回归测试与测试辅助使用。
    # 参数：
    #   conversation_id：目标会话标识；默认 `None`。
    #   content：内容正文，类型 `str`。
    #   trigger：`trigger`输入或配置值；默认 `None`。
    #   event_handler：事件回调；默认 `None`。
    #   on_run_started：运行启动回调；默认 `None`。
    # 返回：类型 `_FakeDispatch`；返回 `_FakeDispatch(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.running_run_ids.add` →
    # `on_run_started` → `_FakeDispatch`。
    # 分支与异常：
    #   当 `self.fail_on_dispatch` 时，抛出 `RuntimeError('dispatch failed')`。
    async def dispatch(
        self,
        *,
        conversation_id=None,
        content: str,
        trigger=None,
        event_handler=None,
        on_run_started=None,
    ) -> _FakeDispatch:
        if self.fail_on_dispatch:
            raise RuntimeError("dispatch failed")
        run_id = f"run-{len(self.dispatched) + 1}"
        self.dispatched.append(
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "content": content,
                "trigger": trigger,
            }
        )
        self.running_run_ids.add(run_id)
        if on_run_started is not None:
            await on_run_started(run_id)
        return _FakeDispatch(run_id)

    # 函数说明：FakeConversationService.is_run_running
    # 用途：判断运行是否满足当前实现的条件。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `bool`；返回 `run_id in self.running_run_ids`。
    async def is_run_running(self, run_id: str) -> bool:
        return run_id in self.running_run_ids

    # 函数说明：FakeConversationService.finish_run
    # 用途：结束运行，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.running_run_ids.discard`。
    def finish_run(self, run_id: str) -> None:
        self.running_run_ids.discard(run_id)


class CrashAfterStartService(FakeConversationService):

    # 函数说明：CrashAfterStartService.dispatch
    # 用途：分发CrashAfterStartService，供回归测试与测试辅助使用。
    # 参数：
    #   conversation_id：目标会话标识；默认 `None`。
    #   content：内容正文，类型 `str`。
    #   trigger：`trigger`输入或配置值；默认 `None`。
    #   event_handler：事件回调；默认 `None`。
    #   on_run_started：运行启动回调；默认 `None`。
    # 返回：不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`on_run_started`。
    async def dispatch(
        self,
        *,
        conversation_id=None,
        content: str,
        trigger=None,
        event_handler=None,
        on_run_started=None,
    ):
        run_id = "run-crash"
        if on_run_started is not None:
            await on_run_started(run_id)
        raise RuntimeError("process crashed after run started")


# 函数说明：make_scheduler
# 用途：构造`scheduler`，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：异步生成器，逐项产出 `_make`；资源与结束处理遵循生成器流程。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`scheduler.shutdown`。
# 分支与异常：
#   捕获 `Exception` 后，忽略该异常并继续当前流程。
@pytest.fixture
async def make_scheduler(tmp_path):

    schedulers: list[AutomationScheduler] = []

    # 函数说明：make_scheduler._make
    # 用途：构造回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   service：业务服务依赖，类型 `FakeConversationService | None`；默认 `None`。
    # 返回：类型
    # `tuple[SQLiteAutomationStore, AutomationScheduler, FakeConversationService]`；返回
    #  `(store, scheduler, service)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteAutomationStore` →
    # `store.initialize` → `FakeConversationService` → `AutomationScheduler`。
    # 闭包依赖：从外层读取 `schedulers`、`tmp_path`。
    async def _make(
        service: FakeConversationService | None = None,
    ) -> tuple[SQLiteAutomationStore, AutomationScheduler, FakeConversationService]:
        store = SQLiteAutomationStore(tmp_path / "muharness.db")
        await store.initialize()
        service = service or FakeConversationService()
        scheduler = AutomationScheduler(store, service)
        schedulers.append(scheduler)
        return store, scheduler, service

    yield _make

    for scheduler in schedulers:
        try:
            await scheduler.shutdown()
        except Exception:  
            pass


# 函数说明：_future
# 用途：返回 `datetime.now(UTC) + timedelta(days=days, hours=hours)`，提供 回归测试与测
# 试辅助 的派生值。
# 参数：
#   days：`days`输入或配置值，类型 `int`；默认 `1`。
#   hours：`hours`输入或配置值，类型 `int`；默认 `0`。
# 返回：类型 `datetime`；返回 `datetime.now(UTC) + timedelta(days=days, hours=hours)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` → `timedelta`。
def _future(days: int = 1, hours: int = 0) -> datetime:
    return datetime.now(UTC) + timedelta(days=days, hours=hours)


# 函数说明：_past
# 用途：返回 `datetime.now(UTC) - timedelta(hours=hours)`，提供 回归测试与测试辅助 的派
# 生值。
# 参数：
#   hours：`hours`输入或配置值，类型 `int`；默认 `2`。
# 返回：类型 `datetime`；返回 `datetime.now(UTC) - timedelta(hours=hours)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` → `timedelta`。
def _past(hours: int = 2) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)




# 函数说明：test_create_once_automation_and_trigger_dispatch
# 用途：回归验证回归测试与测试辅助中的 `create_once_automation_and_trigger_dispatch` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   make_scheduler：构造测试调度器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_scheduler` → `Schedule` →
# `_future` → `scheduler.create_automation` → `datetime.now` → `scheduler._trigger`。
# 分支与异常：
#   验证条件：`automation.status is AutomationStatus.ACTIVE`。
#   验证条件：`automation.next_run_at is not None`。
#   验证条件：`automation.next_run_at > datetime.now(UTC)`。
#   验证条件：`len(service.dispatched) == 1`。
async def test_create_once_automation_and_trigger_dispatch(make_scheduler) -> None:
    store, scheduler, service = await make_scheduler()
    schedule = Schedule(
        kind=ScheduleKind.ONCE,
        run_at=_future(hours=1),
        timezone="UTC",
    )
    automation = await scheduler.create_automation(
        title="交作业提醒",
        prompt=_USER_MESSAGE,
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=schedule.run_at,
    )

    assert automation.status is AutomationStatus.ACTIVE
    assert automation.next_run_at is not None
    assert automation.next_run_at > datetime.now(UTC)

    await scheduler._trigger(automation.id)

    assert len(service.dispatched) == 1
    dispatched = service.dispatched[0]
    assert dispatched["content"] == _USER_MESSAGE
    assert dispatched["conversation_id"] == "conv-1"




# 函数说明：test_once_automation_becomes_completed_after_trigger
# 用途：回归验证回归测试与测试辅助中的 `once_automation_becomes_completed_after_trigger`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   make_scheduler：构造测试调度器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_scheduler` → `Schedule` →
# `_future` → `scheduler.create_automation` → `scheduler._trigger`。
# 分支与异常：
#   验证条件：`updated is not None`。
#   验证条件：`updated.status is AutomationStatus.COMPLETED`。
#   验证条件：`updated.last_run_id is not None`。
#   验证条件：`updated.next_run_at is None`。
async def test_once_automation_becomes_completed_after_trigger(make_scheduler) -> None:
    store, scheduler, _ = await make_scheduler()
    schedule = Schedule(
        kind=ScheduleKind.ONCE,
        run_at=_future(hours=1),
        timezone="UTC",
    )
    automation = await scheduler.create_automation(
        title="一次性",
        prompt="做一件事",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=schedule.run_at,
    )

    await scheduler._trigger(automation.id)

    updated = await store.get(automation.id)
    assert updated is not None
    assert updated.status is AutomationStatus.COMPLETED
    assert updated.last_run_id is not None
    assert updated.next_run_at is None




# 函数说明：test_interval_automation_stays_active_and_updates_next
# 用途：回归验证回归测试与测试辅助中的
# `interval_automation_stays_active_and_updates_next` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   make_scheduler：构造测试调度器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_scheduler` → `Schedule` →
# `scheduler.create_automation` → `_future` → `scheduler._trigger` → `datetime.now`。
# 分支与异常：
#   验证条件：`updated is not None`。
#   验证条件：`updated.status is AutomationStatus.ACTIVE`。
#   验证条件：`updated.last_run_id is not None`。
#   验证条件：`updated.next_run_at is not None`。
async def test_interval_automation_stays_active_and_updates_next(
    make_scheduler,
) -> None:
    store, scheduler, _ = await make_scheduler()
    schedule = Schedule(
        kind=ScheduleKind.INTERVAL,
        interval_seconds=3600,
        timezone="UTC",
    )
    automation = await scheduler.create_automation(
        title="每小时总结",
        prompt="总结进度",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=_future(hours=1),
    )
    old_next = automation.next_run_at

    await scheduler._trigger(automation.id)

    updated = await store.get(automation.id)
    assert updated is not None
    assert updated.status is AutomationStatus.ACTIVE
    assert updated.last_run_id is not None
    assert updated.next_run_at is not None
    assert updated.next_run_at > datetime.now(UTC)
    assert updated.next_run_at != old_next




# 函数说明：test_pause_resume_cancel_lifecycle
# 用途：回归验证回归测试与测试辅助中的 `pause_resume_cancel_lifecycle` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   make_scheduler：构造测试调度器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_scheduler` → `Schedule` →
# `scheduler.create_automation` → `_future` → `scheduler.pause` → `scheduler._trigger`；
# 另有 3 个调用点。
# 分支与异常：
#   验证条件：`paused.status is AutomationStatus.PAUSED`。
#   验证条件：`service.dispatched == []`。
#   验证条件：`resumed.status is AutomationStatus.ACTIVE`。
#   验证条件：`len(service.dispatched) == 1`。
#   预期异常：`pytest.raises(ValueError, match='only active')`。
async def test_pause_resume_cancel_lifecycle(make_scheduler) -> None:
    store, scheduler, service = await make_scheduler()
    schedule = Schedule(
        kind=ScheduleKind.INTERVAL,
        interval_seconds=3600,
        timezone="UTC",
    )
    automation = await scheduler.create_automation(
        title="定时任务",
        prompt="执行",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=_future(hours=1),
    )

    paused = await scheduler.pause(automation.id)
    assert paused.status is AutomationStatus.PAUSED
    await scheduler._trigger(automation.id)
    assert service.dispatched == []

    resumed = await scheduler.resume(automation.id)
    assert resumed.status is AutomationStatus.ACTIVE
    await scheduler._trigger(automation.id)
    assert len(service.dispatched) == 1

    cancelled = await scheduler.cancel(automation.id)
    assert cancelled.status is AutomationStatus.CANCELLED
    await scheduler._trigger(automation.id)
    assert len(service.dispatched) == 1
    with pytest.raises(ValueError, match="only active"):
        await scheduler.pause(automation.id)




# 函数说明：test_restart_reloads_active_automations
# 用途：回归验证回归测试与测试辅助中的 `restart_reloads_active_automations` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteAutomationStore` →
# `store.initialize` → `FakeConversationService` → `Schedule` → `store.create` →
# `_future`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`any((job.id == f'automation-{automation.id}' for job in jobs))`。
async def test_restart_reloads_active_automations(tmp_path) -> None:
    store = SQLiteAutomationStore(tmp_path / "muharness.db")
    await store.initialize()
    service = FakeConversationService()
    schedule = Schedule(
        kind=ScheduleKind.INTERVAL,
        interval_seconds=3600,
        timezone="UTC",
    )
    automation = await store.create(
        title="重启后任务",
        prompt="继续",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=_future(hours=2),
    )

    restarted = AutomationScheduler(store, service)
    try:
        await restarted.start()
        jobs = restarted._scheduler.get_jobs()
        assert any(job.id == f"automation-{automation.id}" for job in jobs)
    finally:
        await restarted.shutdown()




# 函数说明：test_missed_once_automation_runs_only_once
# 用途：回归验证回归测试与测试辅助中的 `missed_once_automation_runs_only_once` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteAutomationStore` →
# `store.initialize` → `FakeConversationService` → `Schedule` → `_past` → `store.create`
# ；另有 6 个调用点。
# 分支与异常：
#   验证条件：`updated is not None`。
#   验证条件：`updated.status is AutomationStatus.ACTIVE`。
#   验证条件：`updated.next_run_at is not None`。
#   验证条件：`updated.next_run_at <= datetime.now(UTC) + timedelta(seconds=5)`。
async def test_missed_once_automation_runs_only_once(tmp_path) -> None:
    store = SQLiteAutomationStore(tmp_path / "muharness.db")
    await store.initialize()
    service = FakeConversationService()
    schedule = Schedule(
        kind=ScheduleKind.ONCE,
        run_at=_past(),
        timezone="UTC",
    )
    automation = await store.create(
        title="错过的一次性",
        prompt="补跑",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=_past(hours=3),
    )

    scheduler = AutomationScheduler(store, service)
    try:
        await scheduler._restore(automation)
        updated = await store.get(automation.id)
        assert updated is not None
        assert updated.status is AutomationStatus.ACTIVE
        assert updated.next_run_at is not None
        assert updated.next_run_at <= datetime.now(UTC) + timedelta(seconds=5)

        await scheduler._trigger(automation.id)
        assert len(service.dispatched) == 1
        completed = await store.get(automation.id)
        assert completed is not None
        assert completed.status is AutomationStatus.COMPLETED

        await scheduler._trigger(automation.id)
        assert len(service.dispatched) == 1
    finally:
        await scheduler.shutdown()




# 函数说明：test_recurring_misfire_does_not_batch_catchup
# 用途：回归验证回归测试与测试辅助中的 `recurring_misfire_does_not_batch_catchup` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteAutomationStore` →
# `store.initialize` → `FakeConversationService` → `Schedule` → `store.create` → `_past`
# ；另有 5 个调用点。
# 分支与异常：
#   验证条件：`updated is not None`。
#   验证条件：`updated.next_run_at is not None`。
#   验证条件：`updated.next_run_at > datetime.now(UTC)`。
#   验证条件：`service.dispatched == []`。
async def test_recurring_misfire_does_not_batch_catchup(tmp_path) -> None:
    store = SQLiteAutomationStore(tmp_path / "muharness.db")
    await store.initialize()
    service = FakeConversationService()
    schedule = Schedule(
        kind=ScheduleKind.INTERVAL,
        interval_seconds=3600,
        timezone="UTC",
    )
    automation = await store.create(
        title="每小时任务",
        prompt="执行",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=_past(hours=5),
    )

    scheduler = AutomationScheduler(store, service)
    try:
        await scheduler.start()
        updated = await store.get(automation.id)
        assert updated is not None
        assert updated.next_run_at is not None
        assert updated.next_run_at > datetime.now(UTC)
        assert service.dispatched == []
        jobs = scheduler._scheduler.get_jobs()
        automation_jobs = [
            job for job in jobs if job.id == f"automation-{automation.id}"
        ]
        assert len(automation_jobs) == 1
    finally:
        await scheduler.shutdown()




# 函数说明：test_no_overlap_when_previous_run_still_running
# 用途：回归验证回归测试与测试辅助中的 `no_overlap_when_previous_run_still_running` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   make_scheduler：构造测试调度器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_scheduler` → `Schedule` →
# `scheduler.create_automation` → `_future` → `scheduler._trigger` →
# `service.finish_run`。
# 分支与异常：
#   验证条件：`len(service.dispatched) == 1`。
#   验证条件：`refreshed is not None and refreshed.last_run_id is not None`。
#   验证条件：`len(service.dispatched) == 2`。
async def test_no_overlap_when_previous_run_still_running(make_scheduler) -> None:
    store, scheduler, service = await make_scheduler()
    schedule = Schedule(
        kind=ScheduleKind.INTERVAL,
        interval_seconds=3600,
        timezone="UTC",
    )
    automation = await scheduler.create_automation(
        title="每小时任务",
        prompt="执行",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=_future(hours=1),
    )

    await scheduler._trigger(automation.id)
    assert len(service.dispatched) == 1
    refreshed = await store.get(automation.id)
    assert refreshed is not None and refreshed.last_run_id is not None

    await scheduler._trigger(automation.id)
    assert len(service.dispatched) == 1

    service.finish_run(refreshed.last_run_id)
    await scheduler._trigger(automation.id)
    assert len(service.dispatched) == 2




# 函数说明：test_dispatch_failure_does_not_crash_scheduler
# 用途：回归验证回归测试与测试辅助中的 `dispatch_failure_does_not_crash_scheduler` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   make_scheduler：构造测试调度器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_scheduler` → `Schedule` →
# `scheduler.create_automation` → `_future` → `scheduler._job_func(automation.id)` →
# `scheduler._job_func`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`len(service.dispatched) == 1`。
# 副作用与资源：
#   更新对象字段：`service.fail_on_dispatch`。
async def test_dispatch_failure_does_not_crash_scheduler(make_scheduler) -> None:
    store, scheduler, service = await make_scheduler()
    schedule = Schedule(
        kind=ScheduleKind.INTERVAL,
        interval_seconds=3600,
        timezone="UTC",
    )
    automation = await scheduler.create_automation(
        title="失败任务",
        prompt="执行",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=_future(hours=1),
    )

    service.fail_on_dispatch = True
    await scheduler._job_func(automation.id)()
    service.fail_on_dispatch = False

    await scheduler._trigger(automation.id)
    assert len(service.dispatched) == 1




# 函数说明：test_automation_create_argument_validation
# 用途：回归验证回归测试与测试辅助中的 `automation_create_argument_validation` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `build_schedule_and_next`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='kind')`。
#   预期异常：`pytest.raises(ValueError, match='run_at')`。
#   预期异常：`pytest.raises(ValueError, match='timezone offset')`。
#   预期异常：`pytest.raises(ValueError, match='future')`。
#   预期异常：`pytest.raises(ValueError, match='interval_seconds')`。
#   预期异常：`pytest.raises(ValueError, match='cron_expr')`。
#   预期异常：`pytest.raises(ValueError, match='timezone')`。
def test_automation_create_argument_validation() -> None:
    with pytest.raises(ValueError, match="kind"):
        build_schedule_and_next({"kind": "hourly", "prompt": "x", "title": "t"})
    with pytest.raises(ValueError, match="run_at"):
        build_schedule_and_next({"kind": "once", "prompt": "x", "title": "t"})
    with pytest.raises(ValueError, match="timezone offset"):
        build_schedule_and_next(
            {
                "kind": "once",
                "run_at": "2026-08-20T09:00:00",
                "prompt": "x",
                "title": "t",
            }
        )
    with pytest.raises(ValueError, match="future"):
        build_schedule_and_next(
            {
                "kind": "once",
                "run_at": "2020-01-01T00:00:00+08:00",
                "prompt": "x",
                "title": "t",
            }
        )
    with pytest.raises(ValueError, match="interval_seconds"):
        build_schedule_and_next(
            {"kind": "interval", "interval_seconds": 0, "prompt": "x", "title": "t"}
        )
    with pytest.raises(ValueError, match="interval_seconds"):
        build_schedule_and_next({"kind": "interval", "prompt": "x", "title": "t"})
    with pytest.raises(ValueError, match="cron_expr"):
        build_schedule_and_next(
            {"kind": "cron", "cron_expr": "not a cron", "prompt": "x", "title": "t"}
        )
    with pytest.raises(ValueError, match="timezone"):
        build_schedule_and_next(
            {
                "kind": "interval",
                "interval_seconds": 60,
                "timezone": "Mars/Olympus",
                "prompt": "x",
                "title": "t",
            }
        )


# 函数说明：test_automation_create_valid_schedules
# 用途：回归验证回归测试与测试辅助中的 `automation_create_valid_schedules` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_schedule_and_next` →
# `datetime.now`。
# 分支与异常：
#   验证条件：`schedule.kind is ScheduleKind.ONCE`。
#   验证条件：`next_run.tzinfo is not None`。
#   验证条件：`schedule.kind is ScheduleKind.INTERVAL`。
#   验证条件：`next_run > datetime.now(UTC)`。
def test_automation_create_valid_schedules() -> None:
    schedule, next_run = build_schedule_and_next(
        {
            "kind": "once",
            "run_at": "2099-08-20T09:00:00+08:00",
            "prompt": "x",
            "title": "t",
        }
    )
    assert schedule.kind is ScheduleKind.ONCE
    assert next_run.tzinfo is not None
    schedule, next_run = build_schedule_and_next(
        {"kind": "interval", "interval_seconds": 7200, "prompt": "x", "title": "t"}
    )
    assert schedule.kind is ScheduleKind.INTERVAL
    assert next_run > datetime.now(UTC)
    schedule, next_run = build_schedule_and_next(
        {
            "kind": "cron",
            "cron_expr": "0 9 * * *",
            "timezone": "Asia/Shanghai",
            "prompt": "x",
            "title": "t",
        }
    )
    assert schedule.kind is ScheduleKind.CRON
    assert schedule.timezone == "Asia/Shanghai"
    assert next_run > datetime.now(UTC)


# 函数说明：test_automation_create_prompt_excludes_schedule_rule
# 用途：回归验证回归测试与测试辅助中的 `automation_create_prompt_excludes_schedule_rule`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AutomationCreateTool`。
# 分支与异常：
#   验证条件：`'只写触发时的执行内容' in tool_desc and '时间、频率、时区' in tool_desc`
# 。
#   验证条件：`'对应调度参数' in tool_desc`。
#   验证条件：`'再次创建' in tool_desc`。
#   验证条件：`'只含执行内容，不重复时间或频率' in prompt_desc`。
def test_automation_create_prompt_excludes_schedule_rule() -> None:

    definition = AutomationCreateTool(scheduler=None).definition
    tool_desc = definition.description
    prompt_desc = definition.parameters["properties"]["prompt"]["description"]

    assert "只写触发时的执行内容" in tool_desc and "时间、频率、时区" in tool_desc
    assert "对应调度参数" in tool_desc
    assert "再次创建" in tool_desc

    assert "只含执行内容，不重复时间或频率" in prompt_desc
    assert "总结项目进度" in prompt_desc
    assert "避免递归创建自动化" in prompt_desc




# 函数说明：test_automation_dispatch_carries_provenance
# 用途：回归验证回归测试与测试辅助中的 `automation_dispatch_carries_provenance` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   make_scheduler：构造测试调度器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_scheduler` → `Schedule` →
# `scheduler.create_automation` → `_future` → `scheduler._trigger`。
# 分支与异常：
#   验证条件：`len(service.dispatched) == 1`。
#   验证条件：`trigger is not None`。
#   验证条件：`trigger.source is ConversationSource.AUTOMATION`。
#   验证条件：`trigger.automation_id == automation.id`。
async def test_automation_dispatch_carries_provenance(make_scheduler) -> None:
    store, scheduler, service = await make_scheduler()
    schedule = Schedule(
        kind=ScheduleKind.INTERVAL,
        interval_seconds=3600,
        timezone="UTC",
    )
    automation = await scheduler.create_automation(
        title="带来源",
        prompt="继续",
        conversation_id="conv-9",
        schedule=schedule,
        next_run_at=_future(hours=1),
    )

    await scheduler._trigger(automation.id)

    assert len(service.dispatched) == 1
    trigger = service.dispatched[0]["trigger"]
    assert trigger is not None
    assert trigger.source is ConversationSource.AUTOMATION
    assert trigger.automation_id == automation.id
    assert trigger.scheduled_for is not None
    assert trigger.triggered_at is not None




# 函数说明：test_invalid_automation_transitions_rejected
# 用途：回归验证回归测试与测试辅助中的 `invalid_automation_transitions_rejected` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   make_scheduler：构造测试调度器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_scheduler` → `Schedule` →
# `scheduler.create_automation` → `_future` → `scheduler.cancel` → `pytest.raises`；另有
#  2 个调用点。
# 分支与异常：
#   验证条件：`(await store.get(once.id)).status is AutomationStatus.COMPLETED`。
#   预期异常：`pytest.raises(ValueError, match='invalid automation transition')`。
async def test_invalid_automation_transitions_rejected(make_scheduler) -> None:
    store, scheduler, service = await make_scheduler()
    schedule = Schedule(
        kind=ScheduleKind.INTERVAL,
        interval_seconds=3600,
        timezone="UTC",
    )
    automation = await scheduler.create_automation(
        title="状态机",
        prompt="执行",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=_future(hours=1),
    )

    await scheduler.cancel(automation.id)
    with pytest.raises(ValueError, match="invalid automation transition"):
        await store.update_status(automation.id, AutomationStatus.ACTIVE)
    with pytest.raises(ValueError, match="invalid automation transition"):
        await store.update_status(automation.id, AutomationStatus.COMPLETED)

    second = Schedule(
        kind=ScheduleKind.ONCE,
        run_at=_future(hours=2),
        timezone="UTC",
    )
    once = await scheduler.create_automation(
        title="一次性状态机",
        prompt="执行",
        conversation_id="conv-1",
        schedule=second,
        next_run_at=second.run_at,
    )
    await scheduler._trigger(once.id)
    assert (await store.get(once.id)).status is AutomationStatus.COMPLETED
    with pytest.raises(ValueError, match="invalid automation transition"):
        await store.update_status(once.id, AutomationStatus.CANCELLED)
    with pytest.raises(ValueError, match="invalid automation transition"):
        await store.update_status(once.id, AutomationStatus.PAUSED)


# 函数说明：test_completed_and_cancelled_automations_do_not_dispatch
# 用途：回归验证回归测试与测试辅助中的
# `completed_and_cancelled_automations_do_not_dispatch` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   make_scheduler：构造测试调度器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_scheduler` → `Schedule` →
# `_future` → `scheduler.create_automation` → `scheduler._trigger` → `scheduler.cancel`
# 。
# 分支与异常：
#   验证条件：`len(service.dispatched) == 1`。
async def test_completed_and_cancelled_automations_do_not_dispatch(
    make_scheduler,
) -> None:
    store, scheduler, service = await make_scheduler()
    once_schedule = Schedule(
        kind=ScheduleKind.ONCE,
        run_at=_future(hours=1),
        timezone="UTC",
    )
    once = await scheduler.create_automation(
        title="已完成",
        prompt="执行",
        conversation_id="conv-1",
        schedule=once_schedule,
        next_run_at=once_schedule.run_at,
    )
    await scheduler._trigger(once.id)
    assert len(service.dispatched) == 1
    await scheduler._trigger(once.id)
    assert len(service.dispatched) == 1

    interval_schedule = Schedule(
        kind=ScheduleKind.INTERVAL,
        interval_seconds=3600,
        timezone="UTC",
    )
    cancelled = await scheduler.create_automation(
        title="已取消",
        prompt="执行",
        conversation_id="conv-1",
        schedule=interval_schedule,
        next_run_at=_future(hours=1),
    )
    await scheduler.cancel(cancelled.id)
    await scheduler._trigger(cancelled.id)
    assert len(service.dispatched) == 1




class _IntegrationRunManager:

    # 函数说明：_IntegrationRunManager.__init__
    # 用途：初始化 _IntegrationRunManager；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_integration_result`。
    # 副作用与资源：
    #   更新对象字段：`self.started`、`self._result`。
    def __init__(self) -> None:
        self.started: list[str] = []
        self._result = _integration_result()

    # 函数说明：_IntegrationRunManager.start
    # 用途：启动_IntegrationRunManager，供回归测试与测试辅助使用。
    # 参数：
    #   user_message：当前用户消息，类型 `str`。
    #   conversation_id：目标会话标识；默认 `None`。
    #   history：原始会话历史；默认 `()`。
    #   summary_state：会话摘要及覆盖水位；默认 `None`。
    #   tool_result_views：工具结果的固定模型视图；默认 `()`。
    #   event_handler：事件回调；默认 `None`。
    #   recovery_run_id：待恢复的运行标识；默认 `None`。
    #   source：输入来源或原始数据；默认 `None`。
    #   source_id：来源记录标识；默认 `None`。
    #   scheduled_for：计划触发时间；默认 `None`。
    #   triggered_at：实际触发时间；默认 `None`。
    #   mode：Agent 执行模式或检索模式；默认 `None`。
    # 返回：类型 `tuple[str, None]`；返回 `('run-auto', None)`。
    async def start(
        self,
        user_message: str,
        *,
        conversation_id=None,
        history=(),
        summary_state=None,
        tool_result_views=(),
        event_handler=None,
        recovery_run_id=None,
        source=None,
        source_id=None,
        scheduled_for=None,
        triggered_at=None,
        mode=None,
    ) -> tuple[str, None]:
        self.started.append(user_message)
        return "run-auto", None

    # 函数说明：_IntegrationRunManager.wait
    # 用途：等待_IntegrationRunManager，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：返回 `SimpleNamespace(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace`。
    async def wait(self, run_id: str):
        return SimpleNamespace(
            id=run_id,
            stop_reason="final_answer",
            status=SimpleNamespace(value="completed"),
        )

    # 函数说明：_IntegrationRunManager.result
    # 用途：返回 `self._result`，提供 _IntegrationRunManager 的派生值。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：返回 `self._result`。
    def result(self, run_id: str):
        return self._result

    # 函数说明：_IntegrationRunManager.get_run
    # 用途：获取运行，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：无结果值，显式返回 None。
    async def get_run(self, run_id: str):
        return None


# 函数说明：_integration_result
# 用途：处理回归测试与测试辅助中的 `_integration_result` 数据；结果及边界条件见下方说明
# 。
# 返回：返回 `AgentResult(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `AgentResult` →
# `ModelUsage`。
def _integration_result():
    from app.models.types import Message, MessageRole, ModelUsage
    from app.runtime.agent.result import AgentResult, AgentStopReason

    final = Message(role=MessageRole.ASSISTANT, content="自动完成")
    return AgentResult(
        run_id="run-auto",
        final_message=final,
        messages=(Message(role=MessageRole.USER, content="自动"), final),
        steps=1,
        stop_reason=AgentStopReason.FINAL_ANSWER,
        usage=ModelUsage(),
    )


# 函数说明：test_scheduler_auto_triggers_without_manual_trigger
# 用途：回归验证回归测试与测试辅助中的 `scheduler_auto_triggers_without_manual_trigger`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `conversation_store.initialize` → `conversation_store.create` → `SQLiteTraceStore` →
# `trace_store.initialize` → `SQLiteConversationSummaryStore`；另有 16 个调用点。
# 分支与异常：
#   验证条件：`len(manager.started) >= 1`。
#   验证条件：`any((item.title == '自动触发' for item in completed))`。
async def test_scheduler_auto_triggers_without_manual_trigger(tmp_path) -> None:
    import asyncio

    from app.domain.conversation.service import ConversationService
    from app.domain.conversation.store import SQLiteConversationStore
    from app.records.trace import SQLiteTraceStore
    from app.runtime.context import SQLiteConversationSummaryStore
    from app.runtime.run import SQLiteRunStore

    database = tmp_path / "muharness.db"
    conversation_store = SQLiteConversationStore(database)
    await conversation_store.initialize()
    conversation = await conversation_store.create()
    trace_store = SQLiteTraceStore(database)
    await trace_store.initialize()
    summary_store = SQLiteConversationSummaryStore(database)
    await summary_store.initialize()
    await SQLiteRunStore(database).initialize()

    manager = _IntegrationRunManager()
    service = ConversationService(
        conversation_store,
        manager,
        trace_store,
        summary_store=summary_store,
    )
    store = SQLiteAutomationStore(database)
    await store.initialize()
    scheduler = AutomationScheduler(store, service)

    run_at = datetime.now(UTC) + timedelta(milliseconds=50)
    schedule = Schedule(
        kind=ScheduleKind.ONCE,
        run_at=run_at,
        timezone="UTC",
    )
    await scheduler.create_automation(
        title="自动触发",
        prompt="自动执行",
        conversation_id=conversation.id,
        schedule=schedule,
        next_run_at=run_at,
    )
    await scheduler.start()

    await asyncio.sleep(0.3)
    try:
        assert len(manager.started) >= 1
        completed = await store.list(status=AutomationStatus.COMPLETED)
        assert any(item.title == "自动触发" for item in completed)
    finally:
        await scheduler.shutdown()




# 函数说明：test_last_run_id_persisted_before_crash
# 用途：回归验证回归测试与测试辅助中的 `last_run_id_persisted_before_crash` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteAutomationStore` →
# `store.initialize` → `Schedule` → `_future` → `store.create` →
# `CrashAfterStartService`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`updated is not None`。
#   验证条件：`updated.last_run_id == 'run-crash'`。
#   验证条件：`updated.last_run_at is not None`。
#   验证条件：`after is not None`。
async def test_last_run_id_persisted_before_crash(tmp_path) -> None:
    store = SQLiteAutomationStore(tmp_path / "muharness.db")
    await store.initialize()
    schedule = Schedule(
        kind=ScheduleKind.ONCE,
        run_at=_future(hours=1),
        timezone="UTC",
    )
    automation = await store.create(
        title="崩溃任务",
        prompt="执行",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=schedule.run_at,
    )

    crash_service = CrashAfterStartService()
    scheduler = AutomationScheduler(store, crash_service)
    try:
        await scheduler._job_func(automation.id)()
    finally:
        await scheduler.shutdown()

    updated = await store.get(automation.id)
    assert updated is not None
    assert updated.last_run_id == "run-crash"
    assert updated.last_run_at is not None

    restarted_service = FakeConversationService()
    restarted = AutomationScheduler(store, restarted_service)
    try:
        await restarted._restore(await store.get(automation.id))
        after = await store.get(automation.id)
        assert after is not None
        assert after.status is AutomationStatus.COMPLETED
        assert restarted_service.dispatched == []
    finally:
        await restarted.shutdown()




# 函数说明：test_restart_does_not_rerun_started_once_automation
# 用途：回归验证回归测试与测试辅助中的 `restart_does_not_rerun_started_once_automation`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteAutomationStore` →
# `store.initialize` → `Schedule` → `_future` → `store.create` → `_past`；另有 6 个调用
# 点。
# 分支与异常：
#   验证条件：`after is not None`。
#   验证条件：`after.status is AutomationStatus.COMPLETED`。
#   验证条件：`service.dispatched == []`。
async def test_restart_does_not_rerun_started_once_automation(tmp_path) -> None:
    store = SQLiteAutomationStore(tmp_path / "muharness.db")
    await store.initialize()
    schedule = Schedule(
        kind=ScheduleKind.ONCE,
        run_at=_future(hours=1),
        timezone="UTC",
    )
    automation = await store.create(
        title="已启动的一次性",
        prompt="执行",
        conversation_id="conv-1",
        schedule=schedule,
        next_run_at=_past(hours=1),
    )
    await store.mark_triggered(
        automation.id,
        last_run_id="run-crash",
        last_run_at=datetime.now(UTC),
        next_run_at=None,
    )

    service = FakeConversationService()
    scheduler = AutomationScheduler(store, service)
    try:
        await scheduler._restore(await store.get(automation.id))
        after = await store.get(automation.id)
        assert after is not None
        assert after.status is AutomationStatus.COMPLETED
        assert service.dispatched == []
    finally:
        await scheduler.shutdown()
