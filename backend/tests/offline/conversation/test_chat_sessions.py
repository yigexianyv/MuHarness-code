from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.domain.conversation import (
    SQLiteConversationStore,
)
from app.domain.conversation.service import ConversationService
from app.domain.memory import MemoryRecord
from app.models.chat import (
    _COMMAND_OVERVIEW,
    _HELP_TEXT,
    _load_or_create_conversation,
    _mark_deferred_tools,
    _parse_args,
    _print_memories,
    _print_memory,
    _print_permission_rules,
    _remove_permission_rule,
    _send_message,
)
from app.models.types import Message, MessageRole, ModelProvider, ToolCall
from app.records.trace import SQLiteTraceStore
from app.runtime.agent.events import AgentEvent, AgentEventHandler, AgentEventType
from app.runtime.agent.result import AgentResult, AgentStopReason
from app.runtime.checkpoint import SQLiteCheckpointStore
from app.runtime.context import (
    ConversationSummaryState,
    RollingConversationSummary,
    SQLiteConversationSummaryStore,
)
from app.runtime.run import RunManager, SQLiteRunStore
from app.tools import ApprovalScope, SQLitePermissionRuleStore, build_safe_rule


class StubRuntime:
    # 函数说明：StubRuntime.__init__
    # 用途：初始化 StubRuntime；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.seen_summary_state`。
    def __init__(self) -> None:
        self.seen_summary_state: ConversationSummaryState | None = None

    # 函数说明：StubRuntime.run
    # 用途：运行StubRuntime，供回归测试与测试辅助使用。
    # 参数：
    #   user_input：本次用户输入，类型 `str`。
    #   history：原始会话历史，类型 `Sequence[Message]`；默认 `()`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   tool_result_views：工具结果的固定模型视图；默认 `()`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
    #   recovery_run_id：待恢复的运行标识，类型 `str | None`；默认 `None`。
    #   mode：Agent 执行模式或检索模式；默认 `None`。
    # 返回：类型 `AgentResult`；返回 `AgentResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `AgentResult`。
    # 副作用与资源：
    #   更新对象字段：`self.seen_summary_state`。
    async def run(
        self,
        user_input: str,
        *,
        history: Sequence[Message] = (),
        conversation_id: str | None = None,
        summary_state: ConversationSummaryState | None = None,
        tool_result_views=(),
        run_id: str | None = None,
        recovery_run_id: str | None = None,
        mode=None,
    ) -> AgentResult:
        self.seen_summary_state = summary_state
        user_message = Message(role=MessageRole.USER, content=user_input)
        final_message = Message(role=MessageRole.ASSISTANT, content="已完成")
        return AgentResult(
            run_id=run_id or "stub-run",
            final_message=final_message,
            messages=(*history, user_message, final_message),
            steps=1,
            stop_reason=AgentStopReason.FINAL_ANSWER,
            summary_state=summary_state,
            tool_result_views=tool_result_views,
        )

    # 函数说明：StubRuntime.run_stream
    # 用途：运行事件流，供回归测试与测试辅助使用。
    # 参数：
    #   user_input：本次用户输入，类型 `str`。
    #   history：原始会话历史，类型 `Sequence[Message]`；默认 `()`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   event_handler：事件回调，类型 `AgentEventHandler | None`；默认 `None`。
    #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；默认
    # `None`。
    #   tool_result_views：工具结果的固定模型视图；默认 `()`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
    #   recovery_run_id：待恢复的运行标识，类型 `str | None`；默认 `None`。
    #   mode：Agent 执行模式或检索模式；默认 `None`。
    #   tool_context_metadata：本次工具执行关联的元数据；默认 `None`。
    # 返回：异步生成器，逐项产出 `event`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.run` → `AgentEvent` →
    # `event_handler.emit`。
    # 副作用与资源：
    #   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
    async def run_stream(
        self,
        user_input: str,
        *,
        history: Sequence[Message] = (),
        conversation_id: str | None = None,
        event_handler: AgentEventHandler | None = None,
        summary_state: ConversationSummaryState | None = None,
        tool_result_views=(),
        run_id: str | None = None,
        recovery_run_id: str | None = None,
        mode=None,
        tool_context_metadata=None,
    ):
        result = await self.run(
            user_input,
            history=history,
            conversation_id=conversation_id,
            summary_state=summary_state,
            tool_result_views=tool_result_views,
            run_id=run_id,
            recovery_run_id=recovery_run_id,
            mode=mode,
        )
        events = (
            AgentEvent(
                run_id=result.run_id,
                conversation_id=conversation_id,
                type=AgentEventType.AGENT_STARTED,
            ),
            AgentEvent(
                run_id=result.run_id,
                conversation_id=conversation_id,
                sequence=1,
                type=AgentEventType.AGENT_COMPLETED,
                stop_reason=result.stop_reason,
                usage=result.usage,
                result=result,
            ),
        )
        for event in events:
            if event_handler is not None:
                await event_handler.emit(event)
            yield event


# 函数说明：_make_run_manager
# 用途：构造运行管理者，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   runtime：传给 `RunManager` 的输入，类型 `object`。
# 返回：类型 `RunManager`；返回 `RunManager(run_store, checkpoint_store, runtime)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteRunStore` →
# `SQLiteCheckpointStore` → `run_store.initialize` → `checkpoint_store.initialize` →
# `RunManager`。
async def _make_run_manager(tmp_path: Path, runtime: object) -> RunManager:

    run_store = SQLiteRunStore(tmp_path / "muharness.db")
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await run_store.initialize()
    await checkpoint_store.initialize()
    return RunManager(run_store, checkpoint_store, runtime)  


# 函数说明：_make_conversation_service
# 用途：构造会话，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   runtime：传给 `_make_run_manager` 的输入，类型 `object`。
#   conversation_store：会话历史存储，类型 `SQLiteConversationStore`。
#   trace_store：执行轨迹存储，类型 `SQLiteTraceStore`。
#   summary_store：摘要持久化存储依赖，类型 `SQLiteConversationSummaryStore | None`；默
# 认 `None`。
# 返回：类型 `ConversationService`；返回 `ConversationService(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_run_manager` →
# `ConversationService`。
async def _make_conversation_service(
    tmp_path: Path,
    runtime: object,
    *,
    conversation_store: SQLiteConversationStore,
    trace_store: SQLiteTraceStore,
    summary_store: SQLiteConversationSummaryStore | None = None,
) -> ConversationService:

    run_manager = await _make_run_manager(tmp_path, runtime)
    return ConversationService(
        conversation_store,
        run_manager,
        trace_store,
        summary_store=summary_store,
    )


# 函数说明：test_cli_restores_latest_conversation_after_restart
# 用途：回归验证回归测试与测试辅助中的 `cli_restores_latest_conversation_after_restart`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `_load_or_create_conversation` → `store.replace_messages` →
# `Message` → `reopened_store.initialize`。
# 分支与异常：
#   验证条件：`resumed is True`。
#   验证条件：`restored.id == created.id`。
#   验证条件：
# `[message.content for message in restored_history] == ['系统提示', '第一轮消息']`。
@pytest.mark.asyncio
async def test_cli_restores_latest_conversation_after_restart(tmp_path) -> None:
    database_path = tmp_path / "muharness.db"
    store = SQLiteConversationStore(database_path)
    await store.initialize()
    created, history, resumed = await _load_or_create_conversation(
        store,
        identifier=None,
        force_new=False,
        system_prompt="系统提示",
    )
    await store.replace_messages(
        created.id,
        (*history, Message(role=MessageRole.USER, content="第一轮消息")),
    )

    reopened_store = SQLiteConversationStore(database_path)
    await reopened_store.initialize()
    restored, restored_history, resumed = await _load_or_create_conversation(
        reopened_store,
        identifier=None,
        force_new=False,
        system_prompt="不会覆盖已有会话",
    )

    assert resumed is True
    assert restored.id == created.id
    assert [message.content for message in restored_history] == [
        "系统提示",
        "第一轮消息",
    ]


# 函数说明：test_cli_can_force_new_or_restore_by_short_id
# 用途：回归验证回归测试与测试辅助中的 `cli_can_force_new_or_restore_by_short_id` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `store.create` → `_load_or_create_conversation` →
# `store.load_messages`。
# 分支与异常：
#   验证条件：`resumed is True`。
#   验证条件：`selected.id == first.id`。
#   验证条件：`created_resumed is False`。
#   验证条件：`created.id != first.id`。
@pytest.mark.asyncio
async def test_cli_can_force_new_or_restore_by_short_id(tmp_path) -> None:
    store = SQLiteConversationStore(tmp_path / "muharness.db")
    await store.initialize()
    first = await store.create(title="已有会话")

    selected, _, resumed = await _load_or_create_conversation(
        store,
        identifier=first.id[:8],
        force_new=False,
        system_prompt=None,
    )
    created, _, created_resumed = await _load_or_create_conversation(
        store,
        identifier=None,
        force_new=True,
        system_prompt="新系统提示",
    )

    assert resumed is True
    assert selected.id == first.id
    assert created_resumed is False
    assert created.id != first.id
    assert (await store.load_messages(created.id))[0].content == "新系统提示"


# 函数说明：test_send_message_persists_runtime_history_and_generates_title
# 用途：回归验证回归测试与测试辅助中的
# `send_message_persists_runtime_history_and_generates_title` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   capsys：pytest 提供的标准输出捕获夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `SQLiteTraceStore` → `trace_store.initialize` → `store.create` →
# `Message`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`success is True`。
#   验证条件：`updated.title == '请读取本地项目并给出一份详细总结'`。
#   验证条件：`await store.load_messages(conversation.id) == tuple(history)`。
#   验证条件：`[message.role for message in history] == [MessageRole.SYSTEM, MessageRole
# .USER, MessageRole.ASSISTANT]`。
@pytest.mark.asyncio
async def test_send_message_persists_runtime_history_and_generates_title(
    tmp_path,
    capsys,
) -> None:
    store = SQLiteConversationStore(tmp_path / "muharness.db")
    await store.initialize()
    trace_store = SQLiteTraceStore(tmp_path / "muharness.db")
    await trace_store.initialize()
    conversation = await store.create(
        messages=(Message(role=MessageRole.SYSTEM, content="系统提示"),)
    )
    history = list(await store.load_messages(conversation.id))

    success, updated = await _send_message(
        conversation_service=await _make_conversation_service(
            tmp_path,
            StubRuntime(),
            conversation_store=store,
            trace_store=trace_store,
        ),
        conversation_store=store,
        conversation=conversation,
        provider=ModelProvider.OPENAI,
        history=history,
        content="请读取本地项目并给出一份详细总结",
        model="fake-model",
    )

    assert success is True
    assert updated.title == "请读取本地项目并给出一份详细总结"
    assert await store.load_messages(conversation.id) == tuple(history)
    assert [message.role for message in history] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    output = capsys.readouterr().out
    assert "Agent 开始执行" in output
    assert "Agent 执行完成" in output
    assert "MuHarness> 已完成" in output
    runs = await trace_store.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id
    assert runs[0].conversation_id == conversation.id
    assert runs[0].event_count == 2


# 函数说明：test_send_message_restores_and_persists_summary_state
# 用途：回归验证回归测试与测试辅助中的
# `send_message_restores_and_persists_summary_state` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `conversation_store.initialize` → `SQLiteConversationSummaryStore` →
# `summary_store.initialize` → `SQLiteTraceStore` → `trace_store.initialize`；另有 8 个
# 调用点。
# 分支与异常：
#   验证条件：`runtime.seen_summary_state == state`。
#   验证条件：`await summary_store.load(conversation.id) == state`。
@pytest.mark.asyncio
async def test_send_message_restores_and_persists_summary_state(tmp_path) -> None:
    database_path = tmp_path / "muharness.db"
    conversation_store = SQLiteConversationStore(database_path)
    await conversation_store.initialize()
    summary_store = SQLiteConversationSummaryStore(database_path)
    await summary_store.initialize()
    trace_store = SQLiteTraceStore(database_path)
    await trace_store.initialize()
    conversation = await conversation_store.create()
    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="继续测试"),
        covered_message_count=0,
    )
    await summary_store.save(conversation.id, state)
    runtime = StubRuntime()
    history: list[Message] = []

    await _send_message(
        conversation_service=await _make_conversation_service(
            tmp_path,
            runtime,
            conversation_store=conversation_store,
            trace_store=trace_store,
            summary_store=summary_store,
        ),
        conversation_store=conversation_store,
        conversation=conversation,
        provider=ModelProvider.OPENAI,
        history=history,
        content="继续",
        model="fake-model",
    )

    assert runtime.seen_summary_state == state
    assert await summary_store.load(conversation.id) == state


# 函数说明：test_cli_persists_and_restores_complete_tool_protocol_history
# 用途：回归验证回归测试与测试辅助中的
# `cli_persists_and_restores_complete_tool_protocol_history` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   capsys：pytest 提供的标准输出捕获夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `SQLiteTraceStore` → `trace_store.initialize` → `store.create` →
# `ToolCall`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`resumed is True`。
#   验证条件：`restored_history == history`。
#   验证条件：`[message.role for message in restored_history] == [MessageRole.USER,
# MessageRole.ASSISTANT, MessageRole.TOOL,…`。
#   验证条件：`restored_history[1].tool_calls == (call,)`。
@pytest.mark.asyncio
async def test_cli_persists_and_restores_complete_tool_protocol_history(
    tmp_path,
    capsys,
) -> None:
    store = SQLiteConversationStore(tmp_path / "muharness.db")
    await store.initialize()
    trace_store = SQLiteTraceStore(tmp_path / "muharness.db")
    await trace_store.initialize()
    conversation = await store.create()
    history: list[Message] = []
    call = ToolCall(id="search-1", name="web_search", arguments={"query": "AI"})

    class ToolProtocolRuntime(StubRuntime):
        # 函数说明：test_cli_persists_and_restores_complete_tool_protocol_history.
        # ToolProtocolRuntime.run
        # 用途：运行ToolProtocolRuntime，供回归测试与测试辅助使用。
        # 参数：
        #   user_input：本次用户输入，类型 `str`。
        #   history：原始会话历史，类型 `Sequence[Message]`；默认 `()`。
        #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
        #   summary_state：会话摘要及覆盖水位，类型 `ConversationSummaryState | None`；
        # 默认 `None`。
        #   tool_result_views：工具结果的固定模型视图；默认 `()`。
        #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
        #   recovery_run_id：待恢复的运行标识，类型 `str | None`；默认 `None`。
        #   mode：Agent 执行模式或检索模式；默认 `None`。
        # 返回：类型 `AgentResult`；返回 `AgentResult(…)`。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `AgentResult`。
        # 闭包依赖：从外层读取 `call`。
        async def run(
            self,
            user_input: str,
            *,
            history: Sequence[Message] = (),
            conversation_id: str | None = None,
            summary_state: ConversationSummaryState | None = None,
            tool_result_views=(),
            run_id: str | None = None,
            recovery_run_id: str | None = None,
            mode=None,
        ) -> AgentResult:
            user_message = Message(role=MessageRole.USER, content=user_input)
            tool_call_message = Message(
                role=MessageRole.ASSISTANT,
                tool_calls=(call,),
            )
            tool_message = Message(
                role=MessageRole.TOOL,
                tool_call_id=call.id,
                name=call.name,
                content="搜索原始结果",
            )
            final_message = Message(role=MessageRole.ASSISTANT, content="搜索摘要")
            return AgentResult(
                run_id="tool-run",
                final_message=final_message,
                messages=(
                    *history,
                    user_message,
                    tool_call_message,
                    tool_message,
                    final_message,
                ),
                steps=2,
                stop_reason=AgentStopReason.FINAL_ANSWER,
                summary_state=summary_state,
                tool_result_views=tool_result_views,
            )

    await _send_message(
        conversation_service=await _make_conversation_service(
            tmp_path,
            ToolProtocolRuntime(),
            conversation_store=store,
            trace_store=trace_store,
        ),
        conversation_store=store,
        conversation=conversation,
        provider=ModelProvider.OPENAI,
        history=history,
        content="搜索 AI",
        model="fake-model",
    )
    _, restored_history, resumed = await _load_or_create_conversation(
        store,
        identifier=conversation.id,
        force_new=False,
        system_prompt=None,
    )

    assert resumed is True
    assert restored_history == history
    assert [message.role for message in restored_history] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert restored_history[1].tool_calls == (call,)
    assert restored_history[2].tool_call_id == call.id
    capsys.readouterr()


# 函数说明：test_cli_lists_and_removes_conversation_permission_rules
# 用途：回归验证回归测试与测试辅助中的
# `cli_lists_and_removes_conversation_permission_rules` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   capsys：pytest 提供的标准输出捕获夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLitePermissionRuleStore` →
# `store.initialize` → `build_safe_rule` → `store.add` → `_print_permission_rules` →
# `store.list`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`rule.id[:8] in output`。
#   验证条件：`rule.description in output`。
#   验证条件：
# `await _remove_permission_rule(store, 'conversation-1', rule.id[:8]) is True`。
#   验证条件：`await store.list(scope_ids=('conversation-1',)) == ()`。
@pytest.mark.asyncio
async def test_cli_lists_and_removes_conversation_permission_rules(
    tmp_path,
    capsys,
) -> None:
    store = SQLitePermissionRuleStore(tmp_path / "muharness.db")
    await store.initialize()
    rule = build_safe_rule(
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
        scope=ApprovalScope.CONVERSATION,
        scope_id="conversation-1",
    )
    await store.add(rule)

    _print_permission_rules(await store.list(scope_ids=("conversation-1",)))
    output = capsys.readouterr().out
    assert rule.id[:8] in output
    assert rule.description in output

    assert (
        await _remove_permission_rule(
            store,
            "conversation-1",
            rule.id[:8],
        )
        is True
    )
    assert await store.list(scope_ids=("conversation-1",)) == ()


# 函数说明：test_cli_uses_provider_default_output_tokens_when_unspecified
# 用途：回归验证回归测试与测试辅助中的
# `cli_uses_provider_default_output_tokens_when_unspecified` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`monkeypatch.setattr` → `_parse_args`
# 。
# 分支与异常：
#   验证条件：`args.max_output_tokens is None`。
def test_cli_uses_provider_default_output_tokens_when_unspecified(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["muharness-chat"])

    args = _parse_args()

    assert args.max_output_tokens is None


# 函数说明：test_cli_accepts_explicit_output_tokens
# 用途：回归验证回归测试与测试辅助中的 `cli_accepts_explicit_output_tokens` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`monkeypatch.setattr` → `_parse_args`
# 。
# 分支与异常：
#   验证条件：`args.max_output_tokens == 8192`。
def test_cli_accepts_explicit_output_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["muharness-chat", "--max-output-tokens", "8192"],
    )

    args = _parse_args()

    assert args.max_output_tokens == 8192


# 函数说明：test_cli_prints_memory_list_and_details
# 用途：回归验证回归测试与测试辅助中的 `cli_prints_memory_list_and_details` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   capsys：pytest 提供的标准输出捕获夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` → `MemoryRecord` →
# `_print_memories` → `_print_memory` → `capsys.readouterr`。
# 分支与异常：
#   验证条件：`'M001' in output`。
#   验证条件：`'[active]' in output`。
#   验证条件：`'MuHarness 使用 SQLite 历史' in output`。
#   验证条件：`'访问次数: 3' in output`。
def test_cli_prints_memory_list_and_details(capsys) -> None:
    now = datetime.now(UTC)
    memory = MemoryRecord(
        id="M001",
        title="MuHarness 使用 SQLite 历史",
        summary="旧的 SQLite 记忆架构说明",
        content="MuHarness 不再使用 SQLite + Embedding 作为长期记忆。",
        created_at=now,
        updated_at=now,
        last_accessed_at=now,
        access_count=3,
    )

    _print_memories((memory,))
    _print_memory(memory)

    output = capsys.readouterr().out
    assert "M001" in output
    assert "[active]" in output
    assert "MuHarness 使用 SQLite 历史" in output
    assert "访问次数: 3" in output


# 函数说明：test_cli_help_exposes_memory_commands
# 用途：回归验证回归测试与测试辅助中的 `cli_help_exposes_memory_commands` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'/help' in _COMMAND_OVERVIEW`。
#   验证条件：`'/memories 查看活跃长期记忆及 Recall Cue' in _HELP_TEXT`。
#   验证条件：`'/memory <记忆ID> 查看一条长期记忆的完整内容' in _HELP_TEXT`。
def test_cli_help_exposes_memory_commands() -> None:
    assert "/help" in _COMMAND_OVERVIEW
    assert "/memories 查看活跃长期记忆及 Recall Cue" in _HELP_TEXT
    assert "/memory <记忆ID> 查看一条长期记忆的完整内容" in _HELP_TEXT


# 函数说明：test_cli_parser_supports_short_startup_options
# 用途：回归验证回归测试与测试辅助中的 `cli_parser_supports_short_startup_options` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具，类型 `pytest.MonkeyPatch`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`monkeypatch.setattr` → `_parse_args`
# 。
# 分支与异常：
#   验证条件：`args.provider == 'deepseek'`。
#   验证条件：`args.model == 'deepseek-chat'`。
#   验证条件：`args.new_conversation is True`。
def test_cli_parser_supports_short_startup_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["muharness", "-p", "deepseek", "-m", "deepseek-chat", "--new"],
    )

    args = _parse_args()

    assert args.provider == "deepseek"
    assert args.model == "deepseek-chat"
    assert args.new_conversation is True


# 函数说明：test_cli_parser_supports_setup
# 用途：回归验证回归测试与测试辅助中的 `cli_parser_supports_setup` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具，类型 `pytest.MonkeyPatch`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`monkeypatch.setattr` → `_parse_args`
# 。
# 分支与异常：
#   验证条件：`args.setup is True`。
def test_cli_parser_supports_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["muharness", "--setup"])

    args = _parse_args()

    assert args.setup is True


# 函数说明：test_mark_deferred_tools_hides_tools_until_activated
# 用途：回归验证回归测试与测试辅助中的 `mark_deferred_tools_hides_tools_until_activated`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_builtin_tool_registry` →
# `FileTaskStore` → `task_store.initialize` → `register_task_tools` → `MemoryManager` →
# `manager.initialize`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`'http_request' not in default_names`。
#   验证条件：`'memory_list' not in default_names`。
#   验证条件：`'core_memory_update' not in default_names`。
#   验证条件：`'memory_read' in default_names`。
@pytest.mark.asyncio
async def test_mark_deferred_tools_hides_tools_until_activated(
    tmp_path: Path,
) -> None:
    from app.domain.memory import MemoryManager, register_memory_tools
    from app.domain.task import FileTaskStore, register_task_tools
    from app.tools import build_builtin_tool_registry
    from app.tools.catalog import ToolCatalog

    registry = build_builtin_tool_registry()
    task_store = FileTaskStore(tmp_path / "tasks")
    await task_store.initialize()
    register_task_tools(registry, task_store)
    manager = MemoryManager(tmp_path / "memory")
    await manager.initialize()
    register_memory_tools(registry, manager)
    _mark_deferred_tools(
        registry,
        frozenset(
            {
                "http_request",
                "memory_list",
                "core_memory_update",
                "core_memory_remove",
            }
        ),
    )

    default_names = {
        definition.name for definition in registry.model_definitions()
    }
    assert "http_request" not in default_names
    assert "memory_list" not in default_names
    assert "core_memory_update" not in default_names
    assert "memory_read" in default_names
    assert "task_update" in default_names

    assert "http_request" in registry.deferred_names()
    catalog = ToolCatalog(registry)
    matches = catalog.search("http_request")
    assert any(match.name == "http_request" for match in matches)
    core_matches = catalog.search("core memory update")
    assert any(match.name == "core_memory_update" for match in core_matches)

    activated = {match.name for match in matches}
    activated_names = {
        definition.name
        for definition in registry.model_definitions(
            activated_names=activated
        )
    }
    assert "http_request" in activated_names
