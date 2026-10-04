from __future__ import annotations

import json

import pytest

from app.models.types import AgentMode, Message, MessageRole, ToolCall, ToolResult
from app.records.trace import SQLiteTraceStore
from app.runtime.agent.events import AgentEvent, AgentEventType
from app.runtime.mea.executor_evidence import ExecutorEvidenceProvider
from app.runtime.run import SQLiteRunStore


async def recorded_executor(tmp_path, *, count=1, finished=True, reply="内容是 v2"):
    traces = SQLiteTraceStore(tmp_path / "m.db")
    runs = SQLiteRunStore(tmp_path / "m.db")
    await traces.initialize()
    await runs.initialize()
    await runs.create(
        run_id="exec",
        conversation_id="conv",
        source="mea:executor",
        source_id="mea",
        mode=AgentMode.EXECUTE,
    )
    sequence = 0

    async def emit(kind, **kwargs):
        nonlocal sequence
        sequence += 1
        await traces.record_event(
            AgentEvent(
                run_id="exec",
                conversation_id="conv",
                sequence=sequence,
                type=kind,
                **kwargs,
            )
        )

    await emit(AgentEventType.AGENT_STARTED)
    calls = tuple(
        ToolCall(id=f"c{i}", name="read_file", arguments={"path": f"demo/{i}.txt"})
        for i in range(count)
    )
    await emit(
        AgentEventType.MODEL_COMPLETED,
        step=1,
        message=Message(role=MessageRole.ASSISTANT, tool_calls=calls),
    )
    for call in calls:
        await emit(AgentEventType.TOOL_STARTED, step=1, tool_call=call)
        if finished:
            await emit(
                AgentEventType.TOOL_COMPLETED,
                step=1,
                tool_call=call,
                tool_result=ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    success=True,
                    output="v2",
                    duration_ms=1,
                    evidence_id=f"ev-{call.id}",
                ),
            )
    if finished:
        await emit(
            AgentEventType.MODEL_COMPLETED,
            step=2,
            message=Message(role=MessageRole.ASSISTANT, content=reply),
        )
        await emit(AgentEventType.AGENT_COMPLETED)
    return ExecutorEvidenceProvider(traces, runs)


async def test_executor_evidence_has_exact_reply_and_tool_batch(tmp_path):
    provider = await recorded_executor(tmp_path, count=2)
    data = json.loads(await provider("exec", "conv", "mea"))
    assert data["call_list_complete"] is True
    assert data["final_reply"] == "内容是 v2"
    assert data["call_count"] == 2
    assert [call["model_step"] for call in data["calls"]] == [1, 1]
    assert data["calls"][0]["arguments"] == {"path": "demo/0.txt"}
    assert data["calls"][0]["output_excerpt"] == "v2"
    assert data["calls"][0]["evidence_id"] == "ev-c0"


@pytest.mark.parametrize("conversation,mea", [("other", "mea"), ("conv", "other")])
async def test_executor_evidence_cannot_cross_conversation_or_mea(
    tmp_path, conversation, mea
):
    provider = await recorded_executor(tmp_path)
    with pytest.raises(ValueError, match="scope"):
        await provider("exec", conversation, mea)


async def test_incomplete_trace_cannot_prove_absence_of_operations(tmp_path):
    provider = await recorded_executor(tmp_path, finished=False)
    data = json.loads(await provider("exec", "conv", "mea"))
    assert data["call_list_complete"] is False
    assert "missing_tool_results" in data["trace_gaps"]
    assert data["final_reply"] is None


async def test_bounded_evidence_explicitly_marks_omitted_calls(tmp_path):
    provider = await recorded_executor(tmp_path, count=65, reply="x" * 10_000)
    encoded = await provider("exec", "conv", "mea")
    data = json.loads(encoded)
    assert len(encoded) <= 12_000
    assert data["call_count"] == 65
    assert data["omitted_calls"] == 65 - len(data["calls"])
    assert data["omitted_calls"] > 0
    assert data["call_list_complete"] is False
    assert data["final_reply_truncated"] is True


async def test_escaped_reply_does_not_exceed_evidence_budget(tmp_path):
    provider = await recorded_executor(tmp_path, reply="\x00" * 4_000)
    encoded = await provider("exec", "conv", "mea")
    assert len(encoded) <= 12_000
    assert json.loads(encoded)["final_reply_truncated"] is True
