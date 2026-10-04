"""上下文透明度（第一阶段）：运行原文记录、压缩数据、必须记住的事项。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from app.domain.conversation import (
    ConstraintsRevisionConflict,
    SQLiteConversationStore,
)
from app.models.types import Message, MessageRole, ModelUsage, ToolCall
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.loop import PINNED_CONSTRAINTS_MESSAGE_NAME
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.context import (
    ContextBudgetPolicy,
    ContextManager,
    ContextSummarizer,
    ConversationReducer,
    ConversationSummaryState,
    ModelCapabilityRegistry,
    RollingConversationSummary,
    SummaryGenerationResult,
)
from app.runtime.run import SQLiteRunMessageStore
from app.server.rpc.methods.conversations import (
    conversation_constraints_get,
    conversation_constraints_set,
)
from app.server.rpc.methods.runs import run_context_messages
from app.server.rpc.protocol import JsonRpcError
from app.tools.registry import ToolRegistry
from tests.offline.agent.test_agent_runtime import (
    CountingTool,
    DeterministicTokenEstimator,
    fake_registry,
    model_response,
)


def _offline_context() -> ContextManager:
    # 离线测试不下载 tiktoken 编码表
    return ContextManager(estimator=DeterministicTokenEstimator())


def _tool_call(call_id: str, value: int) -> ToolCall:
    return ToolCall(id=call_id, name="count", arguments={"value": value})


def _registry_with_count() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CountingTool())
    return registry


async def _stores(tmp_path) -> tuple[SQLiteConversationStore, SQLiteRunMessageStore]:
    database = tmp_path / "muharness.sqlite3"
    conversations = SQLiteConversationStore(database)
    await conversations.initialize()
    run_messages = SQLiteRunMessageStore(database)
    await run_messages.initialize()
    return conversations, run_messages


def _rpc_ctx(conversations, run_messages) -> SimpleNamespace:
    return SimpleNamespace(
        application=SimpleNamespace(
            conversation_store=conversations,
            run_message_store=run_messages,
        )
    )


@pytest.mark.asyncio
async def test_run_records_only_new_messages_and_references_history(tmp_path) -> None:
    conversations, run_messages = await _stores(tmp_path)
    history = (
        Message(role=MessageRole.USER, content="旧问题"),
        Message(role=MessageRole.ASSISTANT, content="旧回答"),
    )
    conversation = await conversations.create(title="t", messages=history)
    registry, _ = fake_registry(
        [
            model_response(tool_calls=(_tool_call("call-1", 7),)),
            model_response(content="完成"),
        ]
    )
    runtime = AgentRuntime(
        registry,
        _registry_with_count(),
        provider="fake",
        context_manager=_offline_context(),
        run_message_store=run_messages,
    )

    result = await runtime.run(
        "新问题",
        history=history,
        conversation_id=conversation.id,
        run_id="run-1",
    )

    assert result.ok is True
    ref = await run_messages.history_ref("run-1")
    assert ref is not None
    assert ref.inherited_count == 2
    stored = await run_messages.load("run-1")
    assert [index for index, _ in stored] == [0, 1, 2, 3]
    assert tuple(message for _, message in stored) == result.messages[2:]

    ctx = _rpc_ctx(conversations, run_messages)
    page = await run_context_messages({"run_id": "run-1", "limit": 3}, ctx)
    assert page["total"] == 6
    assert page["inherited_count"] == 2
    assert [item["index"] for item in page["messages"]] == [0, 1, 2]
    assert [item["inherited"] for item in page["messages"]] == [True, True, False]
    assert page["messages"][0]["message"] == history[0]
    assert page["messages"][2]["message"].content == "新问题"

    second = await run_context_messages(
        {"run_id": "run-1", "offset": 3, "limit": 10}, ctx
    )
    assert [item["index"] for item in second["messages"]] == [3, 4, 5]
    assert second["messages"][-1]["message"].content == "完成"


@pytest.mark.asyncio
async def test_interrupted_run_keeps_messages_produced_before_interrupt(
    tmp_path,
) -> None:
    conversations, run_messages = await _stores(tmp_path)
    conversation = await conversations.create(title="t")
    registry, adapter = fake_registry(
        [model_response(tool_calls=(_tool_call("call-1", 1),))]
    )
    original_complete = adapter.complete

    async def complete_then_cancel(request):
        if adapter.requests:
            raise asyncio.CancelledError()
        return await original_complete(request)

    adapter.complete = complete_then_cancel
    runtime = AgentRuntime(
        registry,
        _registry_with_count(),
        provider="fake",
        context_manager=_offline_context(),
        run_message_store=run_messages,
    )

    with pytest.raises(asyncio.CancelledError):
        await runtime.run(
            "开始",
            conversation_id=conversation.id,
            run_id="run-interrupted",
        )

    stored = [message for _, message in await run_messages.load("run-interrupted")]
    assert [message.role for message in stored] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert stored[1].tool_calls[0].id == "call-1"
    assert stored[2].tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_context_messages_reports_rewritten_history(tmp_path) -> None:
    conversations, run_messages = await _stores(tmp_path)
    history = (Message(role=MessageRole.USER, content="原始问题"),)
    conversation = await conversations.create(title="t", messages=history)
    registry, _ = fake_registry([model_response(content="好的")])
    await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        context_manager=_offline_context(),
        run_message_store=run_messages,
    ).run("继续", history=history, conversation_id=conversation.id, run_id="r")
    await conversations.replace_messages(
        conversation.id,
        (Message(role=MessageRole.USER, content="被改写的问题"),),
    )
    ctx = _rpc_ctx(conversations, run_messages)

    with pytest.raises(JsonRpcError) as error:
        await run_context_messages({"run_id": "r"}, ctx)
    assert "被改写" in error.value.message

    own = await run_context_messages({"run_id": "r", "offset": 1}, ctx)
    assert [item["message"].content for item in own["messages"]] == ["继续", "好的"]


class _DroppingSummarizer(ContextSummarizer):
    """模拟摘要模型改写时漏掉一条用户约束。"""

    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        return SummaryGenerationResult(
            summary=RollingConversationSummary(
                current_objective="修复导出分页",
                user_constraints=("只修改 backend",),
            ),
            usage=ModelUsage(input_tokens=7, output_tokens=3, total_tokens=10),
        )


@pytest.mark.asyncio
async def test_model_started_reports_summary_change_and_dropped_constraints() -> None:
    history_messages: list[Message] = []
    for index in range(8):
        history_messages.extend(
            (
                Message(role=MessageRole.USER, content=f"旧问题 {index} " + "问" * 150),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"旧回答 {index} " + "答" * 150,
                ),
            )
        )
    history = tuple(history_messages)
    previous = ConversationSummaryState(
        summary=RollingConversationSummary(
            current_objective="修复导出分页",
            user_constraints=("只修改 backend", "不要修改数据库结构"),
        ),
        covered_message_count=0,
    )
    registry, _ = fake_registry([model_response(content="最终回答")])
    capabilities = ModelCapabilityRegistry()
    capabilities.register_override(
        "fake", "fake-model", context_window=2_000, max_output_tokens=100
    )
    handler = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        max_output_tokens=100,
        context_manager=ContextManager(
            estimator=DeterministicTokenEstimator(),
            registry=capabilities,
            budget_policy=ContextBudgetPolicy(safety_margin_tokens=0),
            conversation_reducer=ConversationReducer(
                _DroppingSummarizer(),
                keep_recent_conversation_blocks=2,
                keep_recent_tool_rounds=0,
            ),
        ),
    ).run(
        "当前问题",
        history=history,
        summary_state=previous,
        event_handler=handler,
    )

    assert result.ok is True
    started = next(
        event
        for event in handler.events
        if event.type is AgentEventType.MODEL_STARTED
    )
    assert started.summary_updated is True
    assert started.summary_covered_before == 0
    assert started.summary_covered_after == result.summary_state.covered_message_count
    assert started.summary_covered_after > 0
    assert started.source_message_count == len(history) + 1
    assert started.summary_snapshot["user_constraints"] == ["只修改 backend"]
    assert started.summary_previous_snapshot["user_constraints"] == [
        "只修改 backend",
        "不要修改数据库结构",
    ]
    assert started.constraints_possibly_dropped == ("不要修改数据库结构",)


@pytest.mark.asyncio
async def test_first_step_carries_summary_baseline_without_compaction() -> None:
    history = (
        Message(role=MessageRole.USER, content="旧问题"),
        Message(role=MessageRole.ASSISTANT, content="旧回答"),
    )
    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="已有目标"),
        covered_message_count=2,
    )
    registry, _ = fake_registry(
        [
            model_response(tool_calls=(_tool_call("call-1", 1),)),
            model_response(content="完成"),
        ]
    )
    handler = InMemoryEventHandler()

    await AgentRuntime(
        registry,
        _registry_with_count(),
        provider="fake",
        context_manager=_offline_context(),
    ).run(
        "问题",
        history=history,
        summary_state=state,
        event_handler=handler,
    )

    started = [
        event
        for event in handler.events
        if event.type is AgentEventType.MODEL_STARTED
    ]
    assert len(started) == 2
    assert started[0].summary_snapshot["current_objective"] == "已有目标"
    assert started[1].summary_snapshot is None
    assert started[0].constraints_possibly_dropped == ()


@pytest.mark.asyncio
async def test_pinned_constraints_are_read_for_every_request() -> None:
    versions = iter([("只修改 backend", 1), ("只修改 backend\n不要改数据库", 2)])
    seen: list[str] = []

    async def provider(conversation_id: str) -> tuple[str, int]:
        seen.append(conversation_id)
        return next(versions)

    registry, adapter = fake_registry(
        [
            model_response(tool_calls=(_tool_call("call-1", 1),)),
            model_response(content="完成"),
        ]
    )
    handler = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        _registry_with_count(),
        provider="fake",
        context_manager=_offline_context(),
        constraints_provider=provider,
    ).run("问题", conversation_id="conv-1", event_handler=handler)

    assert result.ok is True
    assert seen == ["conv-1", "conv-1"]
    pinned = [
        [m for m in request.messages if m.name == PINNED_CONSTRAINTS_MESSAGE_NAME]
        for request in adapter.requests
    ]
    assert len(pinned[0]) == 1 and "第 1 版" in pinned[0][0].content
    assert "不要改数据库" not in pinned[0][0].content
    assert len(pinned[1]) == 1 and "不要改数据库" in pinned[1][0].content
    assert pinned[1][0].role is MessageRole.SYSTEM
    revisions = [
        event.constraints_revision
        for event in handler.events
        if event.type is AgentEventType.MODEL_STARTED
    ]
    assert revisions == [1, 2]
    # 约定只进入请求，不写进会话原文
    assert not any(
        m.name == PINNED_CONSTRAINTS_MESSAGE_NAME for m in result.messages
    )


@pytest.mark.asyncio
async def test_empty_constraints_and_provider_failure_do_not_break_run() -> None:
    calls = 0

    async def provider(conversation_id: str) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ("", 3)
        raise RuntimeError("database locked")

    registry, adapter = fake_registry(
        [
            model_response(tool_calls=(_tool_call("call-1", 1),)),
            model_response(content="完成"),
        ]
    )
    result = await AgentRuntime(
        registry,
        _registry_with_count(),
        provider="fake",
        context_manager=_offline_context(),
        constraints_provider=provider,
    ).run("问题", conversation_id="conv-1")

    assert result.ok is True
    assert not any(
        m.name == PINNED_CONSTRAINTS_MESSAGE_NAME
        for request in adapter.requests
        for m in request.messages
    )


@pytest.mark.asyncio
async def test_runtime_without_provider_never_injects_constraints() -> None:
    registry, adapter = fake_registry([model_response(content="完成")])
    await AgentRuntime(
        registry, ToolRegistry(), provider="fake", context_manager=_offline_context()
    ).run(
        "问题", conversation_id="conv-1"
    )
    assert not any(
        m.name == PINNED_CONSTRAINTS_MESSAGE_NAME for m in adapter.requests[0].messages
    )


@pytest.mark.asyncio
async def test_constraints_store_revisions_and_conflicts(tmp_path) -> None:
    conversations, _ = await _stores(tmp_path)
    conversation = await conversations.create(title="t")

    empty = await conversations.get_constraints(conversation.id)
    assert (empty.text, empty.revision) == ("", 0)

    first = await conversations.set_constraints(
        conversation.id, "  只修改 backend\r\n", expected_revision=0
    )
    assert (first.text, first.revision) == ("只修改 backend", 1)
    unchanged = await conversations.set_constraints(conversation.id, "只修改 backend")
    assert unchanged.revision == 1
    second = await conversations.set_constraints(conversation.id, "不要改数据库")
    assert second.revision == 2
    assert await conversations.constraints_for_request(conversation.id) == (
        "不要改数据库",
        2,
    )

    with pytest.raises(ConstraintsRevisionConflict):
        await conversations.set_constraints(
            conversation.id, "过期编辑", expected_revision=1
        )
    with pytest.raises(ValueError):
        await conversations.set_constraints(conversation.id, "长" * 4001)
    with pytest.raises(KeyError):
        await conversations.set_constraints("missing", "x")
    with pytest.raises(KeyError):
        await conversations.get_constraints("missing")

    await conversations.delete(conversation.id)
    other = await conversations.create(title="t2")
    assert (await conversations.get_constraints(other.id)).revision == 0


@pytest.mark.asyncio
async def test_constraints_rpc_round_trip(tmp_path) -> None:
    conversations, run_messages = await _stores(tmp_path)
    conversation = await conversations.create(title="t")
    ctx = _rpc_ctx(conversations, run_messages)

    saved = await conversation_constraints_set(
        {"conversation_id": conversation.id, "text": "只修改 backend"}, ctx
    )
    assert saved["constraints"].revision == 1
    loaded = await conversation_constraints_get(
        {"conversation_id": conversation.id}, ctx
    )
    assert loaded["constraints"].text == "只修改 backend"

    with pytest.raises(JsonRpcError):
        await conversation_constraints_set(
            {
                "conversation_id": conversation.id,
                "text": "冲突",
                "expected_revision": 0,
            },
            ctx,
        )
    with pytest.raises(JsonRpcError):
        await conversation_constraints_get({"conversation_id": "missing"}, ctx)


@pytest.mark.asyncio
async def test_long_conversation_does_not_copy_history_per_run(tmp_path) -> None:
    conversations, run_messages = await _stores(tmp_path)
    conversation = await conversations.create(title="t")
    history: tuple[Message, ...] = ()
    for index in range(3):
        registry, _ = fake_registry([model_response(content=f"回答 {index}")])
        result = await AgentRuntime(
            registry,
            ToolRegistry(),
            provider="fake",
            context_manager=_offline_context(),
            run_message_store=run_messages,
        ).run(
            f"问题 {index}",
            history=history,
            conversation_id=conversation.id,
            run_id=f"run-{index}",
        )
        await conversations.save_history_state(
            conversation.id, result.messages, summary_state=None
        )
        history = result.messages
        assert await run_messages.count(f"run-{index}") == 2
        assert (await run_messages.history_ref(f"run-{index}")).inherited_count == (
            2 * index
        )

    # 早期运行继承的前缀没有被后续运行改写，仍可完整还原
    ctx = _rpc_ctx(conversations, run_messages)
    page = await run_context_messages({"run_id": "run-1"}, ctx)
    assert [item["message"].content for item in page["messages"]] == [
        "问题 0",
        "回答 0",
        "问题 1",
        "回答 1",
    ]

    await run_messages.delete_for_conversation(conversation.id)
    assert await run_messages.history_ref("run-0") is None
    assert await run_messages.count("run-2") == 0
