"""Build bounded, run-scoped audit records from persisted Executor traces."""

from __future__ import annotations

import json
from typing import Any

from app.models.types import AgentMode, ToolCall
from app.records.trace import SQLiteTraceStore
from app.runtime.agent.events import AgentEventType
from app.runtime.run import SQLiteRunStore

_MAX_CALLS = 60
_FIELD_CHARS = 600
_REPLY_CHARS = 4_000
_RECORD_CHARS = 12_000
_TERMINAL_EVENTS = frozenset(
    {
        AgentEventType.AGENT_COMPLETED,
        AgentEventType.AGENT_FAILED,
        AgentEventType.AGENT_CANCELLED,
    }
)


class ExecutorEvidenceProvider:
    def __init__(self, traces: SQLiteTraceStore, runs: SQLiteRunStore) -> None:
        self._traces = traces
        self._runs = runs

    async def __call__(self, run_id: str, conversation_id: str, mea_id: str) -> str:
        run = await self._runs.get(run_id)
        if (
            run is None
            or run.conversation_id != conversation_id
            or run.source != "mea:executor"
            or run.source_id != mea_id
            or run.mode is not AgentMode.EXECUTE
        ):
            raise ValueError("Executor evidence is outside the current MEA scope")
        events = await self._traces.load_events(run_id)
        calls: dict[str, dict[str, Any]] = {}
        gaps: list[str] = []
        final_reply: str | None = None
        began = False
        ended = False
        model_recorded = False
        for event in events:
            if event.run_id != run_id or event.conversation_id != conversation_id:
                raise ValueError("Executor trace has inconsistent ownership")
            if event.type is AgentEventType.AGENT_STARTED:
                began = True
            if event.type in _TERMINAL_EVENTS:
                ended = True
                if event.result is not None:
                    final_reply = event.result.final_message.content
            if (
                event.type is AgentEventType.MODEL_COMPLETED
                and event.message is not None
            ):
                model_recorded = True
                if not event.message.tool_calls:
                    final_reply = event.message.content
                for call in event.message.tool_calls:
                    if call.id in calls:
                        gaps.append(f"duplicate_tool_call_id:{call.id}")
                    calls[call.id] = _call_record(call, event.step)
            if (
                event.type is AgentEventType.TOOL_STARTED
                and event.tool_call is not None
            ):
                call = event.tool_call
                record = calls.setdefault(call.id, _call_record(call, event.step))
                record["start_event_recorded"] = True
            if (
                event.type is AgentEventType.TOOL_COMPLETED
                and event.tool_result is not None
            ):
                result = event.tool_result
                record = calls.get(result.tool_call_id)
                if record is None:
                    gaps.append(f"missing_call_intent:{result.tool_call_id}")
                    record = {
                        "tool_call_id": result.tool_call_id,
                        "tool_name": result.tool_name,
                        "model_step": event.step,
                        "arguments": None,
                    }
                    calls[result.tool_call_id] = record
                record.update(
                    {
                        "result_recorded": True,
                        "success": result.success,
                        "evidence_id": result.evidence_id,
                        "output_excerpt": (result.output or "")[:_FIELD_CHARS],
                        "output_excerpt_truncated": len(result.output or "")
                        > _FIELD_CHARS
                        or bool(result.output_truncated),
                        "error": (result.error or "")[:_FIELD_CHARS],
                    }
                )
        if not began or not ended or not model_recorded:
            gaps.append("incomplete_run_trace")
        if any(not call.get("result_recorded") for call in calls.values()):
            gaps.append("missing_tool_results")
        kept = list(calls.values())[:_MAX_CALLS]
        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": run.status.value,
            "call_count": len(calls),
            "calls": kept,
            "call_list_complete": not gaps and len(kept) == len(calls),
            "trace_gaps": [gap[:_FIELD_CHARS] for gap in gaps[:8]],
            "omitted_calls": len(calls) - len(kept),
            "final_reply": final_reply[:_REPLY_CHARS]
            if final_reply is not None
            else None,
            "final_reply_truncated": len(final_reply or "") > _REPLY_CHARS,
        }
        while True:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) <= _RECORD_CHARS:
                return encoded
            if kept:
                kept.pop()
                payload["omitted_calls"] += 1
                payload["call_list_complete"] = False
            else:
                payload["final_reply"] = (payload["final_reply"] or "")[
                    : len(payload["final_reply"] or "") // 2
                ]
                payload["final_reply_truncated"] = True


def _call_record(call: ToolCall, step: int | None) -> dict[str, Any]:
    arguments = call.arguments
    encoded = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    )
    return {
        "tool_call_id": call.id,
        "tool_name": call.name,
        "model_step": step,
        "arguments": arguments
        if len(encoded) <= _FIELD_CHARS
        else encoded[:_FIELD_CHARS],
        "arguments_truncated": len(encoded) > _FIELD_CHARS,
        "start_event_recorded": False,
        "result_recorded": False,
    }


__all__ = ["ExecutorEvidenceProvider"]
