
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.domain.conversation import (
    ConversationSource,
    SQLiteConversationStore,
    TriggerContext,
)
from app.domain.conversation.service import ConversationService
from app.models.types import Message, MessageRole, ModelUsage, ToolCall
from app.records.trace import SQLiteTraceStore
from app.runtime.agent.events import AgentEvent, AgentEventType
from app.runtime.agent.result import AgentResult, AgentStopReason
from app.runtime.context import (
    ConversationSummaryState,
    RollingConversationSummary,
    SQLiteConversationSummaryStore,
)
from app.runtime.context.tool_views import ToolResultViewState
from app.runtime.run import SQLiteRunStore
from app.runtime.run.models import RunStatus


# 函数说明：_message
# 用途：返回 `Message(role=role, content=content)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   role：规划、执行、审计等运行角色，类型 `MessageRole`。
#   content：内容正文，类型 `str`。
# 返回：类型 `Message`；返回 `Message(role=role, content=content)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
def _message(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content)


class StubRunManager:

    # 函数说明：StubRunManager.__init__
    # 用途：初始化 StubRunManager；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   agent_result：结果输入或配置值，类型 `AgentResult`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._agent_result`、`self.started`、`self.recovered`、
    # `self.run_status`、`self.conversation_id`、`self.missing_run`、`self.cancelled`、
    # `self.interrupted` 等 9 个字段。
    def __init__(self, agent_result: AgentResult) -> None:
        self._agent_result = agent_result
        self.started: list[dict] = []
        self.recovered: list[dict] = []
        self.run_status = "completed"
        self.conversation_id: str | None = "conv-stub"
        self.missing_run = False
        self.cancelled = False
        self.interrupted = False
        self.emit_started = True

    # 函数说明：StubRunManager.start
    # 用途：启动StubRunManager，供回归测试与测试辅助使用。
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
    # 返回：类型 `tuple[str, None]`；返回 `('run-1', None)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`event_handler.emit` →
    # `AgentEvent`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
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
        self.started.append(
            {
                "conversation_id": conversation_id,
                "content": user_message,
                "history": history,
                "summary_state": summary_state,
                "tool_result_views": tool_result_views,
                "event_handler": event_handler,
                "source": source,
                "source_id": source_id,
                "scheduled_for": scheduled_for,
                "triggered_at": triggered_at,
                "mode": mode,
            }
        )
        if self.emit_started and event_handler is not None:
            await event_handler.emit(
                AgentEvent(
                    run_id="run-1",
                    conversation_id=conversation_id,
                    sequence=0,
                    type=AgentEventType.AGENT_STARTED,
                )
            )
        return "run-1", None

    # 函数说明：StubRunManager.recover
    # 用途：恢复StubRunManager，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    #   history：原始会话历史；默认 `()`。
    #   summary_state：会话摘要及覆盖水位；默认 `None`。
    #   tool_result_views：工具结果的固定模型视图；默认 `()`。
    #   event_handler：事件回调；默认 `None`。
    # 返回：类型 `tuple[str, None]`；返回 `('run-2', None)`。
    async def recover(
        self,
        run_id: str,
        *,
        history=(),
        summary_state=None,
        tool_result_views=(),
        event_handler=None,
    ) -> tuple[str, None]:
        self.recovered.append(
            {
                "run_id": run_id,
                "history": history,
                "summary_state": summary_state,
                "tool_result_views": tool_result_views,
                "event_handler": event_handler,
            }
        )
        return "run-2", None

    # 函数说明：StubRunManager.wait
    # 用途：等待StubRunManager，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：返回 `SimpleNamespace(id=run_id, stop_reason='final_answer', status=status)`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace`。
    async def wait(self, run_id: str):
        if self.cancelled:
            status = RunStatus.CANCELLED
        elif self.interrupted:
            status = RunStatus.INTERRUPTED
        else:
            status = SimpleNamespace(value=self.run_status)
        return SimpleNamespace(
            id=run_id,
            stop_reason="final_answer",
            status=status,
        )

    # 函数说明：StubRunManager.result
    # 用途：处理回归测试与测试辅助中的 `result` 数据；结果及边界条件见下方说明。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `AgentResult | None`；按分支返回 `None`；`self._agent_result`。
    # 分支与异常：
    #   当 `self.cancelled or self.interrupted` 时，返回 `None`。
    def result(self, run_id: str) -> AgentResult | None:
        if self.cancelled or self.interrupted:
            return None
        return self._agent_result

    # 函数说明：StubRunManager.get_run
    # 用途：获取运行，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：按分支返回 `None`；`SimpleNamespace(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace`。
    # 分支与异常：
    #   当 `self.missing_run` 时，返回 `None`。
    async def get_run(self, run_id: str):
        if self.missing_run:
            return None
        return SimpleNamespace(
            id=run_id,
            conversation_id=self.conversation_id,
            status=SimpleNamespace(value=self.run_status),
        )


# 函数说明：_result_with
# 用途：返回 `AgentResult(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   messages：本次处理的消息序列，类型 `tuple[Message, ...]`。
#   summary：已有或新生成的摘要；默认 `None`。
#   views：`views`输入或配置值；默认 `()`。
# 返回：类型 `AgentResult`；返回 `AgentResult(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentResult` → `ModelUsage`。
def _result_with(
    messages: tuple[Message, ...], *, summary=None, views=(),
) -> AgentResult:
    return AgentResult(
        run_id="run-1",
        final_message=messages[-1],
        messages=messages,
        steps=1,
        stop_reason=AgentStopReason.FINAL_ANSWER,
        usage=ModelUsage(),
        summary_state=summary,
        tool_result_views=views,
    )


# 函数说明：service_factory
# 用途：处理回归测试与测试辅助中的 `service_factory` 数据；结果及边界条件见下方说明。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：返回 `build`。
@pytest.fixture
async def service_factory(tmp_path):

    # 函数说明：service_factory.build
    # 用途：构建回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   result：上一步计算或执行得到的结果，类型 `AgentResult`。
    #   run_status：运行状态输入或配置值，类型 `str`；默认 `'completed'`。
    # 返回：返回 `(service, conversation_store, summary_store, trace_store, manager)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
    # `SQLiteConversationSummaryStore` → `SQLiteTraceStore` → `SQLiteRunStore` →
    # `store.initialize` → `StubRunManager`；另有 1 个调用点。
    # 副作用与资源：
    #   更新对象字段：`manager.run_status`。
    # 闭包依赖：从外层读取 `tmp_path`。
    async def build(result: AgentResult, run_status: str = "completed"):
        database = tmp_path / "muharness.db"
        conversation_store = SQLiteConversationStore(database)
        summary_store = SQLiteConversationSummaryStore(database)
        trace_store = SQLiteTraceStore(database)
        run_store = SQLiteRunStore(database)
        for store in (
            conversation_store,
            summary_store,
            trace_store,
            run_store,
        ):
            await store.initialize()
        manager = StubRunManager(result)
        manager.run_status = run_status
        service = ConversationService(
            conversation_store,
            manager,
            trace_store,
            summary_store=summary_store,
        )
        return service, conversation_store, summary_store, trace_store, manager

    return build




# 函数说明：test_dispatch_loads_latest_history_and_writes_back
# 用途：回归验证回归测试与测试辅助中的 `dispatch_loads_latest_history_and_writes_back`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `conversation_store.create` → `service.dispatch` → `TriggerContext`；另有
#  1 个调用点。
# 分支与异常：
#   验证条件：`started['conversation_id'] == conversation.id`。
#   验证条件：`[m.content for m in started['history']] == ['A', 'B']`。
#   验证条件：`started['content'] == 'C'`。
#   验证条件：`[m.content for m in persisted] == ['A', 'B', 'C', 'D']`。
async def test_dispatch_loads_latest_history_and_writes_back(service_factory) -> None:
    service, conversation_store, _, _, manager = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "B"),
                _message(MessageRole.USER, "C"),
                _message(MessageRole.ASSISTANT, "D"),
            )
        )
    )
    conversation = await conversation_store.create(
        messages=(
            _message(MessageRole.USER, "A"),
            _message(MessageRole.ASSISTANT, "B"),
        )
    )

    dispatch = await service.dispatch(
        conversation_id=conversation.id,
        content="C",
        trigger=TriggerContext(source=ConversationSource.MANUAL),
    )

    started = manager.started[0]
    assert started["conversation_id"] == conversation.id
    assert [m.content for m in started["history"]] == ["A", "B"]
    assert started["content"] == "C"
    persisted = await conversation_store.load_messages(conversation.id)
    assert [m.content for m in persisted] == ["A", "B", "C", "D"]
    assert dispatch.run.id == "run-1"




# 函数说明：test_dispatch_saves_latest_summary
# 用途：回归验证回归测试与测试辅助中的 `dispatch_saves_latest_summary` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationSummaryState` →
# `RollingConversationSummary` → `service_factory` → `_result_with` → `_message` →
# `conversation_store.create`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`saved is not None`。
#   验证条件：`saved.summary.current_objective == '继续任务'`。
#   验证条件：`saved.covered_message_count == 2`。
async def test_dispatch_saves_latest_summary(service_factory) -> None:
    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="继续任务"),
        covered_message_count=2,
    )
    service, conversation_store, summary_store, _, _ = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "D"),
            ),
            summary=state,
        )
    )
    conversation = await conversation_store.create(
        messages=(_message(MessageRole.USER, "A"),)
    )

    await service.dispatch(
        conversation_id=conversation.id,
        content="C",
        trigger=TriggerContext(source=ConversationSource.AUTOMATION),
    )

    saved = await summary_store.load(conversation.id)
    assert saved is not None
    assert saved.summary.current_objective == "继续任务"
    assert saved.covered_message_count == 2




# 函数说明：test_dispatch_injects_trace_handler
# 用途：回归验证回归测试与测试辅助中的 `dispatch_injects_trace_handler` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `conversation_store.create` → `service.dispatch` → `TriggerContext`；另有
#  2 个调用点。
# 分支与异常：
#   验证条件：`isinstance(handler, SQLiteTraceEventHandler) or hasattr(handler, 'emit')`
# 。
#   验证条件：`trace is not None`。
#   验证条件：`trace.conversation_id == conversation.id`。
# 副作用与资源：
#   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
async def test_dispatch_injects_trace_handler(service_factory) -> None:
    service, conversation_store, _, trace_store, manager = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "D"),
            )
        )
    )
    conversation = await conversation_store.create(
        messages=(_message(MessageRole.USER, "A"),)
    )

    await service.dispatch(
        conversation_id=conversation.id,
        content="C",
        trigger=TriggerContext(source=ConversationSource.AUTOMATION),
    )

    handler = manager.started[0]["event_handler"]
    from app.records.trace import SQLiteTraceEventHandler

    assert isinstance(handler, SQLiteTraceEventHandler) or hasattr(
        handler, "emit"
    )
    await handler.emit(
        AgentEvent(
            run_id="run-1",
            conversation_id=conversation.id,
            type=AgentEventType.AGENT_STARTED,
        )
    )
    trace = await trace_store.get("run-1")
    assert trace is not None
    assert trace.conversation_id == conversation.id




# 函数说明：test_dispatch_preserves_trigger_provenance
# 用途：回归验证回归测试与测试辅助中的 `dispatch_preserves_trigger_provenance` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `conversation_store.create` → `TriggerContext` → `datetime.now`；另有 1
# 个调用点。
# 分支与异常：
#   验证条件：`dispatch.trigger is trigger`。
#   验证条件：`dispatch.trigger.source is ConversationSource.AUTOMATION`。
#   验证条件：`dispatch.trigger.automation_id == 'auto-123'`。
async def test_dispatch_preserves_trigger_provenance(service_factory) -> None:
    service, conversation_store, _, _, _ = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "D"),
            )
        )
    )
    conversation = await conversation_store.create()
    trigger = TriggerContext(
        source=ConversationSource.AUTOMATION,
        automation_id="auto-123",
        scheduled_for=datetime.now(UTC),
        triggered_at=datetime.now(UTC),
    )

    dispatch = await service.dispatch(
        conversation_id=conversation.id,
        content="C",
        trigger=trigger,
    )

    assert dispatch.trigger is trigger
    assert dispatch.trigger.source is ConversationSource.AUTOMATION
    assert dispatch.trigger.automation_id == "auto-123"




# 函数说明：test_is_run_running
# 用途：回归验证回归测试与测试辅助中的 `is_run_running` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `conversation_store.create` → `service.is_run_running`。
# 分支与异常：
#   验证条件：`await service.is_run_running('run-1') is True`。
#   验证条件：`await service.is_run_running('run-1') is False`。
# 副作用与资源：
#   更新对象字段：`manager.run_status`。
async def test_is_run_running(service_factory) -> None:
    service, conversation_store, _, _, manager = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "D"),
            )
        )
    )
    await conversation_store.create()

    manager.run_status = "running"
    assert await service.is_run_running("run-1") is True
    manager.run_status = "completed"
    assert await service.is_run_running("run-1") is False




# 函数说明：test_dispatch_cancelled_run_synthesizes_result
# 用途：回归验证回归测试与测试辅助中的 `dispatch_cancelled_run_synthesizes_result` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `conversation_store.create` → `service.dispatch` → `TriggerContext`；另有
#  2 个调用点。
# 分支与异常：
#   验证条件：`[m.content for m in persisted] == ['A', 'C', 'Run cancelled：已停止，未生
# 成最终回复。（本轮未完成的内容不会显示）']`。
#   验证条件：`dispatch.result.stop_reason is AgentStopReason.CANCELLED`。
#   验证条件：`trace is not None`。
#   验证条件：`trace.status is RunStatus.CANCELLED`。
# 副作用与资源：
#   更新对象字段：`manager.cancelled`。
async def test_dispatch_cancelled_run_synthesizes_result(service_factory) -> None:
    service, conversation_store, _, trace_store, manager = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "B"),
            )
        )
    )
    manager.cancelled = True
    conversation = await conversation_store.create(
        messages=(_message(MessageRole.USER, "A"),)
    )

    dispatch = await service.dispatch(
        conversation_id=conversation.id,
        content="C",
        trigger=TriggerContext(source=ConversationSource.MANUAL),
    )

    persisted = await conversation_store.load_messages(conversation.id)
    assert [m.content for m in persisted] == [
        "A",
        "C",
        "Run cancelled：已停止，未生成最终回复。（本轮未完成的内容不会显示）",
    ]
    assert dispatch.result.stop_reason is AgentStopReason.CANCELLED
    trace = await trace_store.get("run-1")
    assert trace is not None
    assert trace.status is RunStatus.CANCELLED
    assert trace.completed_at is not None
    events = await trace_store.load_events("run-1")
    assert [event.sequence for event in events] == [0, 1]
    assert events[0].type is AgentEventType.AGENT_STARTED
    assert events[1].type is AgentEventType.AGENT_CANCELLED
    assert events[1].stop_reason is AgentStopReason.CANCELLED


# 函数说明：test_dispatch_interrupted_run_synthesizes_result
# 用途：回归验证回归测试与测试辅助中的 `dispatch_interrupted_run_synthesizes_result` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `conversation_store.create` → `service.dispatch` → `TriggerContext`；另有
#  2 个调用点。
# 分支与异常：
#   验证条件：`[m.content for m in persisted] == ['A', 'C', 'Run interrupted：已暂停，可
# 从断点继续。（点击 Recover 从保存的中断点恢复）']`。
#   验证条件：`dispatch.result.stop_reason is AgentStopReason.INTERRUPTED`。
#   验证条件：`trace is not None`。
#   验证条件：`trace.status is RunStatus.INTERRUPTED`。
# 副作用与资源：
#   更新对象字段：`manager.interrupted`。
async def test_dispatch_interrupted_run_synthesizes_result(service_factory) -> None:
    service, conversation_store, _, trace_store, manager = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "B"),
            )
        )
    )
    manager.interrupted = True
    conversation = await conversation_store.create(
        messages=(_message(MessageRole.USER, "A"),)
    )

    dispatch = await service.dispatch(
        conversation_id=conversation.id,
        content="C",
        trigger=TriggerContext(source=ConversationSource.MANUAL),
    )

    persisted = await conversation_store.load_messages(conversation.id)
    assert [m.content for m in persisted] == [
        "A",
        "C",
        "Run interrupted：已暂停，可从断点继续。（点击 Recover 从保存的中断点恢复）",
    ]
    assert dispatch.result.stop_reason is AgentStopReason.INTERRUPTED
    trace = await trace_store.get("run-1")
    assert trace is not None
    assert trace.status is RunStatus.INTERRUPTED
    events = await trace_store.load_events("run-1")
    assert [event.sequence for event in events] == [0, 1]
    assert events[1].type is AgentEventType.AGENT_FAILED
    assert events[1].stop_reason is AgentStopReason.INTERRUPTED




# 函数说明：test_next_dispatch_sees_previous_automation_result
# 用途：回归验证回归测试与测试辅助中的 `next_dispatch_sees_previous_automation_result`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `conversation_store.create` → `service.dispatch` → `TriggerContext`；另有
#  1 个调用点。
# 分支与异常：
#   验证条件：
# `[m.content for m in manager2.started[0]['history']] == ['A', 'B', 'C', 'D']`。
async def test_next_dispatch_sees_previous_automation_result(
    service_factory,
) -> None:
    service, conversation_store, _, _, manager = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "B"),
                _message(MessageRole.USER, "C"),
                _message(MessageRole.ASSISTANT, "D"),
            )
        )
    )
    conversation = await conversation_store.create(
        messages=(
            _message(MessageRole.USER, "A"),
            _message(MessageRole.ASSISTANT, "B"),
        )
    )
    await service.dispatch(
        conversation_id=conversation.id,
        content="C",
        trigger=TriggerContext(source=ConversationSource.AUTOMATION),
    )

    service2, _, _, _, manager2 = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "B"),
                _message(MessageRole.USER, "C"),
                _message(MessageRole.ASSISTANT, "D"),
                _message(MessageRole.USER, "E"),
                _message(MessageRole.ASSISTANT, "F"),
            )
        )
    )
    await service2.dispatch(
        conversation_id=conversation.id,
        content="E",
        trigger=TriggerContext(source=ConversationSource.AUTOMATION),
    )
    assert [m.content for m in manager2.started[0]["history"]] == [
        "A",
        "B",
        "C",
        "D",
    ]




class ConcurrentRunManager:

    # 函数说明：ConcurrentRunManager.__init__
    # 用途：初始化 ConcurrentRunManager；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.active`、`self.max_active`、`self.calls`。
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: list[dict] = []

    # 函数说明：ConcurrentRunManager.start
    # 用途：启动ConcurrentRunManager，供回归测试与测试辅助使用。
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
    # 返回：类型 `tuple[str, None]`；返回 `(f'run-{len(self.calls)}', None)`。
    # 副作用与资源：
    #   更新对象字段：`self.active`、`self.max_active`。
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
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "content": user_message,
                "history": tuple(history),
                "mode": mode,
            }
        )
        return f"run-{len(self.calls)}", None

    # 函数说明：ConcurrentRunManager.wait
    # 用途：等待ConcurrentRunManager，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：返回 `SimpleNamespace(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.sleep` →
    # `SimpleNamespace`。
    # 副作用与资源：
    #   更新对象字段：`self.active`。
    async def wait(self, run_id: str):
        await asyncio.sleep(0.02)
        self.active -= 1
        return SimpleNamespace(
            id=run_id,
            stop_reason="final_answer",
            status=SimpleNamespace(value="completed"),
        )

    # 函数说明：ConcurrentRunManager.result
    # 用途：在回归测试与测试辅助中处理 `result`，通过 `run_id.split` 完成首个内部处理步
    # 骤。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `AgentResult`；返回 `_result_with(messages)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_message` → `_result_with`。
    def result(self, run_id: str) -> AgentResult:
        index = int(run_id.split("-")[1]) - 1
        call = self.calls[index]
        messages = (
            *call["history"],
            _message(MessageRole.USER, call["content"]),
            _message(MessageRole.ASSISTANT, f"out-{index + 1}"),
        )
        return _result_with(messages)

    # 函数说明：ConcurrentRunManager.get_run
    # 用途：获取运行，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：无结果值，显式返回 None。
    async def get_run(self, run_id: str):
        return None


# 函数说明：test_same_conversation_dispatch_is_serialized
# 用途：回归验证回归测试与测试辅助中的 `same_conversation_dispatch_is_serialized` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `conversation_store.create` → `ConcurrentRunManager` →
# `asyncio.create_task`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`manager.max_active == 1`。
#   验证条件：`len(manager.calls) == 2`。
#   验证条件：
# `[m.content for m in manager.calls[1]['history']] == ['A', 'B', 'C', 'out-1']`。
#   验证条件：`[m.content for m in persisted] == ['A', 'B', 'C', 'out-1', 'D', 'out-2']`
# 。
# 副作用与资源：
#   更新对象字段：`service._run_manager`。
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
async def test_same_conversation_dispatch_is_serialized(service_factory) -> None:
    service, conversation_store, _, _, _ = await service_factory(
        _result_with((_message(MessageRole.USER, "x"),))
    )
    conversation = await conversation_store.create(
        messages=(
            _message(MessageRole.USER, "A"),
            _message(MessageRole.ASSISTANT, "B"),
        )
    )
    manager = ConcurrentRunManager()
    service._run_manager = manager  

    task1 = asyncio.create_task(
        service.dispatch(
            conversation_id=conversation.id,
            content="C",
            trigger=TriggerContext(source=ConversationSource.MANUAL),
        )
    )
    task2 = asyncio.create_task(
        service.dispatch(
            conversation_id=conversation.id,
            content="D",
            trigger=TriggerContext(source=ConversationSource.MANUAL),
        )
    )
    await asyncio.gather(task1, task2)

    assert manager.max_active == 1
    assert len(manager.calls) == 2
    assert [m.content for m in manager.calls[1]["history"]] == [
        "A",
        "B",
        "C",
        "out-1",
    ]
    persisted = await conversation_store.load_messages(conversation.id)
    assert [m.content for m in persisted] == [
        "A",
        "B",
        "C",
        "out-1",
        "D",
        "out-2",
    ]


# 函数说明：test_different_conversations_dispatch_in_parallel
# 用途：回归验证回归测试与测试辅助中的 `different_conversations_dispatch_in_parallel` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `conversation_store.create` → `ConcurrentRunManager` →
# `asyncio.create_task`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`manager.max_active == 2`。
#   验证条件：`len(manager.calls) == 2`。
# 副作用与资源：
#   更新对象字段：`service._run_manager`。
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
async def test_different_conversations_dispatch_in_parallel(service_factory) -> None:
    service, conversation_store, _, _, _ = await service_factory(
        _result_with((_message(MessageRole.USER, "x"),))
    )
    conv_a = await conversation_store.create()
    conv_b = await conversation_store.create()
    manager = ConcurrentRunManager()
    service._run_manager = manager  

    task_a = asyncio.create_task(
        service.dispatch(
            conversation_id=conv_a.id,
            content="A-input",
            trigger=TriggerContext(source=ConversationSource.MANUAL),
        )
    )
    task_b = asyncio.create_task(
        service.dispatch(
            conversation_id=conv_b.id,
            content="B-input",
            trigger=TriggerContext(source=ConversationSource.MANUAL),
        )
    )
    await asyncio.gather(task_a, task_b)

    assert manager.max_active == 2
    assert len(manager.calls) == 2




class ProvenanceRuntime:

    # 函数说明：ProvenanceRuntime.run_stream
    # 用途：运行事件流，供回归测试与测试辅助使用。
    # 参数：
    #   user_input：本次用户输入，类型 `str`。
    #   history：原始会话历史；默认 `()`。
    #   conversation_id：目标会话标识；默认 `None`。
    #   event_handler：事件回调；默认 `None`。
    #   summary_state：会话摘要及覆盖水位；默认 `None`。
    #   tool_result_views：工具结果的固定模型视图；默认 `()`。
    #   run_id：目标运行标识；默认 `None`。
    #   recovery_run_id：待恢复的运行标识；默认 `None`。
    #   mode：Agent 执行模式或检索模式；默认 `None`。
    #   tool_context_metadata：本次工具执行关联的元数据；默认 `None`。
    # 返回：异步生成器，逐项产出 `event`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_message` → `_result_with` →
    # `AgentEvent` → `event_handler.emit`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def run_stream(
        self,
        user_input: str,
        *,
        history=(),
        conversation_id=None,
        event_handler=None,
        summary_state=None,
        tool_result_views=(),
        run_id=None,
        recovery_run_id=None,
        mode=None,
        tool_context_metadata=None,
    ):
        final = _message(MessageRole.ASSISTANT, "完成")
        user = _message(MessageRole.USER, user_input)
        result = _result_with((*history, user, final))
        result = result.model_copy(update={"run_id": run_id})
        event = AgentEvent(
            run_id=run_id,
            conversation_id=conversation_id,
            type=AgentEventType.AGENT_COMPLETED,
            stop_reason=result.stop_reason,
            result=result,
        )
        if event_handler is not None:
            await event_handler.emit(event)
        yield event


# 函数说明：test_run_provenance_persisted_across_restart
# 用途：回归验证回归测试与测试辅助中的 `run_provenance_persisted_across_restart` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `conversation_store.initialize` → `conversation_store.create` → `SQLiteTraceStore` →
# `trace_store.initialize` → `SQLiteConversationSummaryStore`；另有 12 个调用点。
# 分支与异常：
#   验证条件：`run is not None`。
#   验证条件：`run.source == 'automation'`。
#   验证条件：`run.source_id == 'auto-9'`。
#   验证条件：`run.scheduled_for is not None`。
async def test_run_provenance_persisted_across_restart(
    tmp_path,
) -> None:
    from app.runtime.checkpoint import SQLiteCheckpointStore
    from app.runtime.run import RunManager

    database = tmp_path / "muharness.db"
    conversation_store = SQLiteConversationStore(database)
    await conversation_store.initialize()
    conversation = await conversation_store.create()
    trace_store = SQLiteTraceStore(database)
    await trace_store.initialize()
    summary_store = SQLiteConversationSummaryStore(database)
    await summary_store.initialize()
    run_store = SQLiteRunStore(database)
    await run_store.initialize()
    checkpoint_store = SQLiteCheckpointStore(database)
    await checkpoint_store.initialize()

    run_manager = RunManager(run_store, checkpoint_store, ProvenanceRuntime())
    service = ConversationService(
        conversation_store,
        run_manager,
        trace_store,
        summary_store=summary_store,
    )
    scheduled_for = datetime.now(UTC)
    triggered_at = datetime.now(UTC)
    dispatch = await service.dispatch(
        conversation_id=conversation.id,
        content="C",
        trigger=TriggerContext(
            source=ConversationSource.AUTOMATION,
            automation_id="auto-9",
            scheduled_for=scheduled_for,
            triggered_at=triggered_at,
        ),
    )

    reopened = SQLiteRunStore(database)
    await reopened.initialize()
    run = await reopened.get(dispatch.run.id)
    assert run is not None
    assert run.source == "automation"
    assert run.source_id == "auto-9"
    assert run.scheduled_for is not None
    assert run.triggered_at is not None




# 函数说明：test_recover_writes_back_conversation_and_summary
# 用途：回归验证回归测试与测试辅助中的 `recover_writes_back_conversation_and_summary` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationSummaryState` →
# `RollingConversationSummary` → `service_factory` → `_result_with` → `_message` →
# `conversation_store.create`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`recovered['run_id'] == 'old-run'`。
#   验证条件：`[m.content for m in recovered['history']] == ['A', 'B']`。
#   验证条件：`[m.content for m in persisted] == ['A', 'B', 'C', 'D-恢复']`。
#   验证条件：`saved is not None`。
# 副作用与资源：
#   更新对象字段：`manager.conversation_id`。
async def test_recover_writes_back_conversation_and_summary(
    service_factory,
) -> None:

    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="恢复后继续"),
        covered_message_count=4,
    )
    service, conversation_store, summary_store, _, manager = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "B"),
                _message(MessageRole.USER, "C"),
                _message(MessageRole.ASSISTANT, "D-恢复"),
            ),
            summary=state,
        )
    )
    conversation = await conversation_store.create(
        messages=(
            _message(MessageRole.USER, "A"),
            _message(MessageRole.ASSISTANT, "B"),
        )
    )
    manager.conversation_id = conversation.id

    dispatch = await service.recover("old-run")

    recovered = manager.recovered[0]
    assert recovered["run_id"] == "old-run"
    assert [m.content for m in recovered["history"]] == ["A", "B"]

    persisted = await conversation_store.load_messages(conversation.id)
    assert [m.content for m in persisted] == [
        "A",
        "B",
        "C",
        "D-恢复",
    ]
    saved = await summary_store.load(conversation.id)
    assert saved is not None
    assert saved.summary.current_objective == "恢复后继续"
    assert saved.covered_message_count == 4
    assert dispatch.run.id == "run-2"


# 函数说明：test_recover_missing_run_raises
# 用途：回归验证回归测试与测试辅助中的 `recover_missing_run_raises` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`service_factory` → `_result_with` →
# `_message` → `pytest.raises` → `service.recover`。
# 分支与异常：
#   验证条件：`manager.recovered == []`。
#   预期异常：`pytest.raises(KeyError)`。
# 副作用与资源：
#   更新对象字段：`manager.missing_run`。
async def test_recover_missing_run_raises(service_factory) -> None:
    service, _, _, _, manager = await service_factory(
        _result_with((_message(MessageRole.USER, "A"),))
    )
    manager.missing_run = True

    with pytest.raises(KeyError):
        await service.recover("no-such-run")
    assert manager.recovered == []


# 函数说明：test_recover_uses_latest_summary
# 用途：回归验证回归测试与测试辅助中的 `recover_uses_latest_summary` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationSummaryState` →
# `RollingConversationSummary` → `service_factory` → `_result_with` → `_message` →
# `conversation_store.create`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`recovered['summary_state'] is not None`。
#   验证条件：`recovered['summary_state'].summary.current_objective == '旧目标'`。
# 副作用与资源：
#   更新对象字段：`manager.conversation_id`。
async def test_recover_uses_latest_summary(service_factory) -> None:

    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="旧目标"),
        covered_message_count=2,
    )
    service, conversation_store, summary_store, _, manager = await service_factory(
        _result_with(
            (
                _message(MessageRole.USER, "A"),
                _message(MessageRole.ASSISTANT, "B"),
            )
        )
    )
    conversation = await conversation_store.create(
        messages=(_message(MessageRole.USER, "A"),)
    )
    await summary_store.save(conversation.id, state)
    manager.conversation_id = conversation.id

    await service.recover("old-run")

    recovered = manager.recovered[0]
    assert recovered["summary_state"] is not None
    assert recovered["summary_state"].summary.current_objective == "旧目标"


# 函数说明：_raw_tool_history
# 用途：返回 `(_message(MessageRole.USER, 'run'), Message(role=MessageRole.ASSISTANT,
# tool_calls=(…`，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `tuple[Message, ...]`；返回 `(_message(MessageRole.USER, 'run'), Message(
# role=MessageRole.ASSISTANT, tool_calls=(…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_message` → `Message` → `ToolCall` →
# `json.dumps`。
def _raw_tool_history() -> tuple[Message, ...]:
    return (
        _message(MessageRole.USER, "run"),
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=(ToolCall(id="call", name="execute", arguments={}),),
        ),
        Message(
            role=MessageRole.TOOL, tool_call_id="call", name="execute",
            content=json.dumps({"output": "raw" * 1000, "success": True}),
        ),
        _message(MessageRole.ASSISTANT, "done"),
    )


# 函数说明：_frozen_views
# 用途：在回归测试与测试辅助中处理 `_frozen_views`，通过 `state.project` 完成首个内部处
# 理步骤。
# 参数：
#   raw：传给 `state.project` 的输入。
# 返回：返回 `state.snapshot(raw)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolResultViewState` →
# `state.project` → `state.snapshot`。
def _frozen_views(raw):
    state = ToolResultViewState((), max_output_chars=100, head_chars=10, tail_chars=10)
    state.project(raw)
    return state.snapshot(raw)


# 函数说明：test_next_dispatch_restores_views_without_changing_raw
# 用途：回归验证回归测试与测试辅助中的
# `next_dispatch_restores_views_without_changing_raw` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_raw_tool_history` → `_frozen_views`
# → `service_factory` → `_result_with` → `store.create` → `service.dispatch`；另有 4 个
# 调用点。
# 分支与异常：
#   验证条件：`manager.started[0]['tool_result_views'] == views`。
#   验证条件：`manager.started[0]['history'] == raw`。
#   验证条件：`(await reopened.load_messages(conversation.id))[2] == raw[2]`。
#   验证条件：`await reopened.load_tool_result_views(conversation.id) == views`。
async def test_next_dispatch_restores_views_without_changing_raw(
    service_factory,
) -> None:
    raw = _raw_tool_history()
    views = _frozen_views(raw)
    service, store, _, _, _ = await service_factory(_result_with(raw, views=views))
    conversation = await store.create()
    await service.dispatch(conversation_id=conversation.id, content="run")

    extended = (
        *raw, _message(MessageRole.USER, "again"),
        _message(MessageRole.ASSISTANT, "continued"),
    )
    restarted, reopened, _, _, manager = await service_factory(
        _result_with(extended, views=views)
    )
    await restarted.dispatch(conversation_id=conversation.id, content="again")
    assert manager.started[0]["tool_result_views"] == views
    assert manager.started[0]["history"] == raw
    assert (await reopened.load_messages(conversation.id))[2] == raw[2]
    assert await reopened.load_tool_result_views(conversation.id) == views


# 函数说明：test_recover_restores_and_persists_frozen_views
# 用途：回归验证回归测试与测试辅助中的 `recover_restores_and_persists_frozen_views` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_raw_tool_history` → `_frozen_views`
# → `service_factory` → `_result_with` → `store.create` → `store.save_history_state`；另
# 有 3 个调用点。
# 分支与异常：
#   验证条件：`manager.recovered[0]['tool_result_views'] == views`。
#   验证条件：`await store.load_tool_result_views(conversation.id) == views`。
#   验证条件：`await store.load_messages(conversation.id) == raw`。
# 副作用与资源：
#   更新对象字段：`manager.conversation_id`。
async def test_recover_restores_and_persists_frozen_views(service_factory) -> None:
    raw = _raw_tool_history()
    views = _frozen_views(raw)
    service, store, _, _, manager = await service_factory(
        _result_with(raw, views=views)
    )
    conversation = await store.create()
    await store.save_history_state(
        conversation.id, raw, summary_state=None, tool_result_views=views,
    )
    manager.conversation_id = conversation.id
    await service.recover("old-run")
    assert manager.recovered[0]["tool_result_views"] == views
    assert await store.load_tool_result_views(conversation.id) == views
    assert await store.load_messages(conversation.id) == raw


# 函数说明：test_synthesized_terminal_preserves_previous_context_state
# 用途：回归验证回归测试与测试辅助中的
# `synthesized_terminal_preserves_previous_context_state` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   service_factory：构造测试服务的工厂夹具。
#   status：目标状态。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_raw_tool_history` → `_frozen_views`
# → `ConversationSummaryState` → `RollingConversationSummary` → `service_factory` →
# `_result_with`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`await store.load_summary_state(conversation.id) == summary`。
#   验证条件：`await store.load_tool_result_views(conversation.id) == views`。
#   验证条件：`(await store.load_messages(conversation.id))[:len(raw)] == raw`。
@pytest.mark.parametrize("status", ["cancelled", "interrupted"])
async def test_synthesized_terminal_preserves_previous_context_state(
    service_factory, status,
) -> None:
    raw = _raw_tool_history()
    views = _frozen_views(raw)
    summary = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="continue"),
        covered_message_count=3,
    )
    service, store, _, _, manager = await service_factory(_result_with(raw))
    conversation = await store.create()
    await store.save_history_state(
        conversation.id, raw, summary_state=summary, tool_result_views=views,
    )
    setattr(manager, status, True)
    await service.dispatch(conversation_id=conversation.id, content="again")
    assert await store.load_summary_state(conversation.id) == summary
    assert await store.load_tool_result_views(conversation.id) == views
    assert (await store.load_messages(conversation.id))[: len(raw)] == raw
