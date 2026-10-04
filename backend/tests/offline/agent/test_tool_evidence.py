"""工具输出截短后：事件带截短标记，"查看完整工具原文"能分页读到末尾。"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.domain.conversation import SQLiteConversationStore
from app.models.types import MessageRole, ToolCall, ToolDefinition
from app.records.evidence import EvidenceRecorder, SQLiteEvidenceStore
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.context import ContextManager
from app.runtime.rewind.service import ConversationFork
from app.runtime.run import SQLiteRunMessageStore
from app.runtime.run.models import Run, RunStatus
from app.server.rpc.methods.runs import run_context_evidence
from app.server.rpc.protocol import JsonRpcError
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from tests.offline.agent.test_agent_runtime import (
    DeterministicTokenEstimator,
    fake_registry,
    model_response,
)

_LONG_OUTPUT = "HEAD_MARKER\n" + "x" * 133_000 + "\nEND_MARKER"


class DumpTool(BaseTool):
    definition = ToolDefinition(
        name="dump",
        description="return a long or short text",
        parameters={
            "type": "object",
            "properties": {"long": {"type": "boolean"}},
            "required": ["long"],
        },
    )

    async def execute(self, arguments: dict[str, object]) -> str:
        return _LONG_OUTPUT if arguments["long"] else "short output"


class _Forks:
    def __init__(self) -> None:
        self.links: dict[str, str] = {}

    async def fork_for(self, conversation_id: str) -> ConversationFork | None:
        source = self.links.get(conversation_id)
        if source is None:
            return None
        return ConversationFork(
            conversation_id=conversation_id,
            source_conversation_id=source,
            source_run_id="r",
            source_step=1,
            rewind_key="k",
        )


@pytest.fixture
async def env(tmp_path):
    database = tmp_path / "muharness.sqlite3"
    conversations = SQLiteConversationStore(database)
    await conversations.initialize()
    evidence = SQLiteEvidenceStore(database)
    await evidence.initialize()
    run_messages = SQLiteRunMessageStore(database)
    await run_messages.initialize()
    runs: dict[str, Run] = {}
    forks = _Forks()

    async def get_run(run_id: str) -> Run | None:
        return runs.get(run_id)

    ctx = SimpleNamespace(
        application=SimpleNamespace(
            run_manager=SimpleNamespace(get_run=get_run),
            evidence_store=evidence,
            rewind_service=forks,
        )
    )

    async def run_tool(conversation_id: str, run_id: str, *, long: bool):
        registry, _ = fake_registry(
            [
                model_response(
                    tool_calls=(
                        ToolCall(
                            id=f"call-{run_id}", name="dump", arguments={"long": long}
                        ),
                    )
                ),
                model_response(content="完成"),
            ]
        )
        tools = ToolRegistry()
        tools.register(DumpTool())
        handler = InMemoryEventHandler()
        await AgentRuntime(
            registry,
            tools,
            provider="fake",
            context_manager=ContextManager(estimator=DeterministicTokenEstimator()),
            tool_output_recorder=EvidenceRecorder(evidence),
            run_message_store=run_messages,
        ).run(
            "运行工具",
            conversation_id=conversation_id,
            run_id=run_id,
            event_handler=handler,
        )
        now = datetime.now(UTC)
        runs[run_id] = Run(
            id=run_id,
            conversation_id=conversation_id,
            status=RunStatus.COMPLETED,
            created_at=now,
            updated_at=now,
        )
        return handler

    return SimpleNamespace(
        conversations=conversations,
        run_messages=run_messages,
        ctx=ctx,
        forks=forks,
        run_tool=run_tool,
    )


async def _read_all(ctx, run_id: str, tool_call_id: str) -> str:
    parts: list[str] = []
    offset: int | None = 0
    while offset is not None:
        page = await run_context_evidence(
            {"run_id": run_id, "tool_call_id": tool_call_id, "offset": offset}, ctx
        )
        assert len(page["content"]) <= 12_000
        parts.append(page["content"])
        offset = page["next_offset"]
    return "".join(parts)


@pytest.mark.asyncio
async def test_long_output_is_truncated_for_model_but_full_text_is_readable(
    env,
) -> None:
    conversation = await env.conversations.create(title="长输出")
    handler = await env.run_tool(conversation.id, "run-long", long=True)

    completed = [
        event
        for event in handler.events
        if event.type is AgentEventType.TOOL_COMPLETED
    ]
    assert len(completed) == 1
    result = completed[0].tool_result
    assert result.output_truncated is True
    assert result.evidence_id
    assert "END_MARKER" not in (result.output or "")

    # 模型收到（也是消息记录里保存）的是截短版本
    tool_message = next(
        message
        for _, message in await env.run_messages.load("run-long")
        if message.role is MessageRole.TOOL
    )
    assert "END_MARKER" not in (tool_message.content or "")
    assert len(tool_message.content or "") < len(_LONG_OUTPUT)

    full = await _read_all(env.ctx, "run-long", "call-run-long")
    assert full == _LONG_OUTPUT
    assert full.endswith("END_MARKER")


@pytest.mark.asyncio
async def test_short_output_is_not_marked_truncated(env) -> None:
    conversation = await env.conversations.create(title="短输出")
    handler = await env.run_tool(conversation.id, "run-short", long=False)
    result = next(
        event.tool_result
        for event in handler.events
        if event.type is AgentEventType.TOOL_COMPLETED
    )
    assert not result.output_truncated
    page = await run_context_evidence(
        {"run_id": "run-short", "tool_call_id": "call-run-short"}, env.ctx
    )
    assert page["content"] == "short output"
    assert page["next_offset"] is None


@pytest.mark.asyncio
async def test_evidence_scope_follows_fork_chain_only(env) -> None:
    source = await env.conversations.create(title="来源")
    fork = await env.conversations.create(title="分支")
    unrelated = await env.conversations.create(title="无关")
    await env.run_tool(source.id, "run-source", long=True)
    await env.run_tool(fork.id, "run-fork", long=False)
    await env.run_tool(unrelated.id, "run-unrelated", long=False)

    # 分支会话还没登记来源时，读不到来源会话的证据
    with pytest.raises(JsonRpcError, match="完整原文不可用"):
        await run_context_evidence(
            {"run_id": "run-fork", "tool_call_id": "call-run-source"}, env.ctx
        )

    env.forks.links[fork.id] = source.id
    full = await _read_all(env.ctx, "run-fork", "call-run-source")
    assert full.endswith("END_MARKER")

    with pytest.raises(JsonRpcError, match="完整原文不可用"):
        await run_context_evidence(
            {"run_id": "run-unrelated", "tool_call_id": "call-run-source"}, env.ctx
        )
    with pytest.raises(JsonRpcError):
        await run_context_evidence(
            {"run_id": "missing", "tool_call_id": "call-run-source"}, env.ctx
        )


@pytest.mark.parametrize("size", [8_000, 8_001, 20_000, 20_001, 133_021])
@pytest.mark.asyncio
async def test_request_view_matches_what_adapter_received(tmp_path, size) -> None:
    from app.runtime.rewind import SQLiteRunStepStore
    from app.server.rpc.methods.runs import run_context_tool_view

    database = tmp_path / "muharness.sqlite3"
    conversations = SQLiteConversationStore(database)
    await conversations.initialize()
    evidence = SQLiteEvidenceStore(database)
    await evidence.initialize()
    steps = SQLiteRunStepStore(database)
    await steps.initialize()
    conversation = await conversations.create(title="边界")
    payload = "HEAD_MARKER" + "x" * (size - len("HEAD_MARKER") - len("END_MARKER"))
    payload += "END_MARKER"
    assert len(payload) == size

    class SizedTool(BaseTool):
        definition = ToolDefinition(
            name="dump",
            description="sized output",
            parameters={"type": "object", "properties": {}},
        )

        async def execute(self, arguments: dict[str, object]) -> str:
            return payload

    registry, adapter = fake_registry(
        [
            model_response(tool_calls=(ToolCall(id="c1", name="dump", arguments={}),)),
            model_response(content="完成"),
        ]
    )
    tools = ToolRegistry()
    tools.register(SizedTool())
    handler = InMemoryEventHandler()
    await AgentRuntime(
        registry,
        tools,
        provider="fake",
        context_manager=ContextManager(estimator=DeterministicTokenEstimator()),
        tool_output_recorder=EvidenceRecorder(evidence),
        run_step_store=steps,
    ).run("运行", conversation_id=conversation.id, run_id="r", event_handler=handler)

    # 第 2 步请求：适配器实际收到的工具消息
    sent = next(
        message
        for message in adapter.requests[1].messages
        if message.role is MessageRole.TOOL
    )
    ctx = SimpleNamespace(application=SimpleNamespace(run_step_store=steps))
    view = await run_context_tool_view(
        {"run_id": "r", "step": 2, "tool_call_id": "c1"}, ctx
    )
    assert view["included"] is True
    assert view["content"] == sent.content
    assert view["original_chars"] == size

    import json

    sent_output = json.loads(sent.content)["output"]
    assert view["request_chars"] == len(sent_output)
    assert view["stored_truncated"] is (size > 20_000)
    # 上下文层超过 8,000 字符才改写，改写后保留开头和结尾
    assert view["request_shortened"] is (size > 8_000)
    if size > 8_000:
        assert view["request_chars"] < view["stored_chars"]
        assert sent_output.startswith("HEAD_MARKER")
    else:
        assert sent_output == payload

    started = [
        event
        for event in handler.events
        if event.type is AgentEventType.MODEL_STARTED
    ]
    [meta] = started[1].request_tool_views
    assert "content" not in meta
    assert meta["request_chars"] == view["request_chars"]
    assert meta["stored_chars"] == view["stored_chars"]


def test_tool_not_in_request_is_reported_as_not_included() -> None:
    from app.models.types import Message
    from app.runtime.agent.loop import request_tool_views

    raw = (
        Message(role=MessageRole.USER, content="问题"),
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=(ToolCall(id="old", name="read_file", arguments={}),),
        ),
        Message(
            role=MessageRole.TOOL,
            name="read_file",
            tool_call_id="old",
            content='{"output":"旧内容","output_chars":3}',
        ),
    )
    # 摘要替代后，请求里只剩摘要和用户消息
    request = (Message(role=MessageRole.SYSTEM, content="摘要"), raw[0])
    [view] = request_tool_views(request, raw)
    assert view["included"] is False
    assert "content" not in view
    assert view["original_chars"] == 3
