from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from app.domain.conversation import SQLiteConversationStore
from app.domain.conversation.service import ConversationService
from app.models.types import Message, MessageRole, ToolCall, ToolDefinition
from app.records.evidence import EvidenceRecorder, SQLiteEvidenceStore
from app.records.trace import SQLiteTraceStore
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.checkpoint import SQLiteCheckpointStore
from app.runtime.context import (
    ContextBudgetPolicy,
    ContextManager,
    ContextSettings,
    ContextSummarizer,
    ConversationReducer,
    ModelCapabilityRegistry,
    RollingConversationSummary,
    SummaryGenerationResult,
    partition_messages,
)
from app.runtime.context.blocks import BlockType
from app.runtime.context.inventory import ContextInventory
from app.runtime.context.summary import SUMMARY_MESSAGE_NAME
from app.runtime.context.tool_views import ToolResultViewState
from app.runtime.run import RunManager, SQLiteRunStore
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from tests.offline.agent.test_agent_runtime import (
    DeterministicTokenEstimator,
    fake_registry,
    model_response,
)

_USER_CONSTRAINT = "Keep all source files unchanged; do not upload anything."
_CURRENT_REQUEST = f"Build a report. {_USER_CONSTRAINT}"
_CONTINUE_REQUEST = "Continue from saved work. Do not repeat earlier tool actions."
_FAILURE_CUE = "EXPECTED_FAILURE:second"


class EvidenceActionTool(BaseTool):
    definition = ToolDefinition(
        name="evidence_action",
        description="Offline action with a durable evidence record",
        parameters={
            "type": "object",
            "properties": {"stage": {"type": "string"}},
            "required": ["stage"],
        },
    )

    # 函数说明：EvidenceActionTool.__init__
    # 用途：初始化 EvidenceActionTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   side_effects：`side_effects`输入或配置值，类型 `list[str]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.side_effects`、`self.payloads`。
    def __init__(self, side_effects: list[str]) -> None:
        self.side_effects = side_effects
        self.payloads = {
            "first": "SUCCESS:first\n" + "A" * 21_000 + "\nEND:first",
            "third": "SUCCESS:third\n" + "C" * 21_000 + "\nEND:third",
            "after_compaction": "SUCCESS:after_compaction",
            "resume": "SUCCESS:resume",
        }

    # 函数说明：EvidenceActionTool.execute
    # 用途：执行EvidenceActionTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, object]`；读取键
    # `stage`。
    # 返回：类型 `str`；返回 `self.payloads[stage]`。
    # 分支与异常：
    #   当 `stage == 'failure'` 时，抛出 `RuntimeError(_FAILURE_CUE)`。
    async def execute(self, arguments: dict[str, object]) -> str:
        stage = str(arguments["stage"])
        self.side_effects.append(stage)
        if stage == "failure":
            raise RuntimeError(_FAILURE_CUE)
        return self.payloads[stage]


class EvidenceSummaryFake(ContextSummarizer):
    """Summarize only facts supplied by the actual runtime's fold input."""

    # 函数说明：EvidenceSummaryFake.__init__
    # 用途：初始化 EvidenceSummaryFake；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.inputs`。
    def __init__(self) -> None:
        self.inputs: list[tuple[Message, ...]] = []

    # 函数说明：EvidenceSummaryFake.summarize
    # 用途：生成摘要EvidenceSummaryFake，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要，类型 `RollingConversationSummary | None`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `SummaryGenerationResult`；返回 `SummaryGenerationResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` →
    # `SummaryGenerationResult` → `RollingConversationSummary` →
    # `envelope['output'].splitlines`。
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        self.inputs.append(tuple(messages))
        users = tuple(
            message.content or ""
            for message in messages
            if message.role is MessageRole.USER
        )
        envelopes = tuple(
            json.loads(message.content)
            for message in messages
            if message.role is MessageRole.TOOL
        )
        successes = tuple(envelope for envelope in envelopes if envelope["success"])
        failures = tuple(envelope for envelope in envelopes if not envelope["success"])
        constraints = tuple(user.split("Build a report. ", 1)[-1] for user in users)
        return SummaryGenerationResult(
            summary=RollingConversationSummary(
                current_objective="Build a report",
                user_constraints=constraints,
                completed_work=tuple(
                    f"Succeeded: {envelope['output'].splitlines()[0]}"
                    for envelope in successes
                ),
                current_state=tuple(
                    f"Failed, not completed: {envelope['error']}"
                    for envelope in failures
                ),
                important_facts=tuple(
                    f"Read full output with evidence_id={envelope['evidence_id']}"
                    for envelope in successes
                ),
            )
        )


# 函数说明：_context_manager
# 用途：在回归测试与测试辅助中处理 `_context_manager`，通过
# `capabilities.register_override` 完成首个内部处理步骤。
# 参数：
#   summarizer：传给 `ConversationReducer` 的输入。
#   restored：`restored`输入或配置值，类型 `bool`；默认 `False`。
# 返回：类型 `ContextManager`；返回 `ContextManager(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelCapabilityRegistry` →
# `capabilities.register_override` → `ContextSettings` → `ContextManager` →
# `DeterministicTokenEstimator` → `ContextBudgetPolicy`；另有 1 个调用点。
def _context_manager(summarizer, *, restored: bool = False) -> ContextManager:
    capabilities = ModelCapabilityRegistry()
    capabilities.register_override(
        "fake",
        "fake-model",
        context_window=30_000,
        max_output_tokens=512,
    )
    settings = ContextSettings(
        _env_file=None,
        context_keep_recent_tool_rounds=1,
        # Change the settings on restart to verify existing views are frozen.
        context_max_tool_result_chars=1000 if restored else 8000,
        context_tool_result_head_chars=300 if restored else 4000,
        context_tool_result_tail_chars=100 if restored else 2000,
    )
    return ContextManager(
        estimator=DeterministicTokenEstimator(),
        registry=capabilities,
        context_settings=settings,
        budget_policy=ContextBudgetPolicy(
            safety_margin_tokens=0,
            preferred_input_tokens=11_000,
            compact_input_tokens=11_000,
            working_target_ratio=0.30,
        ),
        conversation_reducer=ConversationReducer(
            summarizer,
            keep_recent_conversation_blocks=1,
            keep_recent_tool_rounds=1,
        ),
    )


# 函数说明：_service
# 用途：在回归测试与测试辅助中处理 `_service`，通过 `store.initialize` 完成首个内部处理
# 步骤。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `Path`。
#   registry：工具、模型或能力注册表。
#   tool：目标工具实例。
#   summarizer：传给 `_context_manager` 的输入。
#   restored：`restored`输入或配置值；默认 `False`。
# 返回：返回
# `(ConversationService(conversations, manager, traces), conversations, evidence)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `SQLiteEvidenceStore` → `SQLiteTraceStore` → `SQLiteRunStore` →
# `SQLiteCheckpointStore` → `store.initialize`；另有 8 个调用点。
async def _service(database: Path, registry, tool, summarizer, *, restored=False):
    conversations = SQLiteConversationStore(database)
    evidence = SQLiteEvidenceStore(database)
    traces = SQLiteTraceStore(database)
    runs = SQLiteRunStore(database)
    checkpoints = SQLiteCheckpointStore(database)
    for store in (conversations, evidence, traces, runs, checkpoints):
        await store.initialize()
    tools = ToolRegistry()
    tools.register(tool)
    runtime = AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_output_tokens=512,
        context_manager=_context_manager(summarizer, restored=restored),
        checkpoint_store=checkpoints,
        tool_output_recorder=EvidenceRecorder(evidence),
    )
    manager = RunManager(runs, checkpoints, runtime)
    await manager.initialize()
    return ConversationService(conversations, manager, traces), conversations, evidence


# 函数说明：_action
# 用途：返回 `model_response(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   call_id：工具调用标识，类型 `str`。
#   stage：`stage`输入或配置值，类型 `str`。
# 返回：返回 `model_response(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`model_response` → `ToolCall`。
def _action(call_id: str, stage: str):
    return model_response(
        tool_calls=(
            ToolCall(id=call_id, name="evidence_action", arguments={"stage": stage}),
        )
    )


# 函数说明：_assert_paired
# 用途：执行 `_assert_paired` 测试辅助流程并检查预期结果。
# 参数：
#   messages：本次处理的消息序列，类型 `Sequence[Message]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`partition_messages`。
# 分支与异常：
#   验证条件：`all((block.block_type is not BlockType.MALFORMED_TOOL for block in
# partition_messages(messages)))`。
def _assert_paired(messages: Sequence[Message]) -> None:
    assert all(
        block.block_type is not BlockType.MALFORMED_TOOL
        for block in partition_messages(messages)
    )


# 函数说明：test_large_tool_work_compacts_persists_and_continues_after_restart
# 用途：回归验证回归测试与测试辅助中的
# `large_tool_work_compacts_persists_and_continues_after_restart` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`EvidenceActionTool` →
# `EvidenceSummaryFake` → `fake_registry` → `_action` → `model_response` → `_service`；
# 另有 20 个调用点。
# 分支与异常：
#   当 `not record.result.success` 时，跳过当前循环项。
#   验证条件：`result.ok`。
#   验证条件：`side_effects == ['first', 'failure', 'third', 'after_compaction']`。
#   验证条件：`[event.prefix_decision for event in started] == ['rebuild', 'append', '
# append', 'compact', 'append']`。
#   验证条件：
# `[event.summary_updated for event in started] == [False, False, False, True, False]`。
async def test_large_tool_work_compacts_persists_and_continues_after_restart(
    tmp_path,
) -> None:
    side_effects: list[str] = []
    tool = EvidenceActionTool(side_effects)
    summarizer = EvidenceSummaryFake()
    registry, adapter = fake_registry(
        [
            _action("call-1", "first"),
            _action("call-2", "failure"),
            _action("call-3", "third"),
            _action("call-4", "after_compaction"),
            model_response(
                content="Report prepared; the failed action remains failed."
            ),
        ]
    )
    database = tmp_path / "end_to_end.db"
    service, conversations, evidence = await _service(
        database,
        registry,
        tool,
        summarizer,
    )
    conversation = await conversations.create()
    events = InMemoryEventHandler()

    dispatched = await service.dispatch(
        conversation_id=conversation.id,
        content=_CURRENT_REQUEST,
        event_handler=events,
    )

    result = dispatched.result
    assert result.ok
    assert side_effects == ["first", "failure", "third", "after_compaction"]
    started = [
        event for event in events.events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert [event.prefix_decision for event in started] == [
        "rebuild",
        "append",
        "append",
        "compact",
        "append",
    ]
    assert [event.summary_updated for event in started] == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert len(summarizer.inputs) == 1
    assert started[3].cache_prefix_reused is False
    assert started[4].cache_prefix_reused is True
    compact_request, following_request = adapter.requests[3:5]
    assert (
        following_request.messages[: len(compact_request.messages)]
        == compact_request.messages
    )
    for request in adapter.requests:
        _assert_paired(request.messages)
        assert any(message.content == _CURRENT_REQUEST for message in request.messages)

    raw = await conversations.load_messages(conversation.id)
    summary = await conversations.load_summary_state(conversation.id)
    views = await conversations.load_tool_result_views(conversation.id)
    assert raw == result.messages
    assert summary == result.summary_state
    assert views == result.tool_result_views
    assert summary is not None
    watermark = summary.covered_message_count
    assert 0 < watermark < len(raw)
    assert watermark == 5  # User + two complete call/result pairs, not half a pair.
    _assert_paired(raw[:watermark])
    _assert_paired(raw[watermark:])
    assert (
        ContextInventory.build(
            raw,
            history_count=len(raw),
            summary_state=summary,
        ).summary_state
        == summary
    )
    assert summary.summary.user_constraints == (_USER_CONSTRAINT,)
    assert "SUCCESS:first" in " ".join(summary.summary.completed_work)
    assert _FAILURE_CUE in " ".join(summary.summary.current_state)
    assert _FAILURE_CUE not in " ".join(summary.summary.completed_work)

    raw_tools = [message for message in raw if message.role is MessageRole.TOOL]
    for message, record in zip(raw_tools, result.tool_calls, strict=True):
        assert json.loads(message.content) == record.result.model_dump(
            mode="json",
            exclude_none=True,
        )
    assert result.tool_calls[1].result.success is False
    assert result.tool_calls[1].result.error.endswith(_FAILURE_CUE)
    assert [view.representation for view in views] == [
        "excerpt",
        "full",
        "excerpt",
        "full",
    ]
    for record, stage in zip(result.tool_calls, side_effects, strict=True):
        if not record.result.success:
            continue
        document = await evidence.resolve(
            record.result.evidence_id,
            conversation_id=conversation.id,
        )
        assert document is not None
        assert document.content == tool.payloads[stage]
    first_evidence_id = result.tool_calls[0].result.evidence_id
    assert first_evidence_id in " ".join(summary.summary.important_facts)
    # Summarizer receives frozen excerpts, while evidence and raw history retain
    # their independent original records.
    folded_first = next(
        message
        for message in summarizer.inputs[0]
        if message.role is MessageRole.TOOL and message.tool_call_id == "call-1"
    )
    assert json.loads(folded_first.content)["evidence_id"] == first_evidence_id
    assert len(json.loads(folded_first.content)["output"]) < 8000
    assert len(json.loads(raw_tools[0].content)["output"]) > 8000

    # Restart every store/service/runtime. Reuse the old call ID deliberately:
    # it belongs to a new run/sequence, not the earlier side effect or view.
    resumed_registry, resumed_adapter = fake_registry(
        [
            _action("call-1", "resume"),
            model_response(content="Continuation complete."),
        ]
    )
    resumed_summarizer = EvidenceSummaryFake()
    resumed_tool = EvidenceActionTool(side_effects)
    resumed_service, reopened, reopened_evidence = await _service(
        database,
        resumed_registry,
        resumed_tool,
        resumed_summarizer,
        restored=True,
    )
    resumed_events = InMemoryEventHandler()
    continued = await resumed_service.dispatch(
        conversation_id=conversation.id,
        content=_CONTINUE_REQUEST,
        event_handler=resumed_events,
    )
    assert continued.result.ok
    assert side_effects == ["first", "failure", "third", "after_compaction", "resume"]
    assert len(continued.result.tool_calls) == 1
    assert continued.result.tool_calls[0].tool_call.arguments == {"stage": "resume"}
    assert resumed_summarizer.inputs == []
    resumed_started = [
        event
        for event in resumed_events.events
        if event.type is AgentEventType.MODEL_STARTED
    ]
    assert [event.prefix_decision for event in resumed_started] == ["rebuild", "append"]
    assert resumed_started[0].cache_prefix_reused is False
    assert resumed_started[1].cache_prefix_reused is True
    first_resumed, next_resumed = resumed_adapter.requests
    assert (
        next_resumed.messages[: len(first_resumed.messages)] == first_resumed.messages
    )
    assert summary.summary.to_message() in first_resumed.messages
    assert raw[:watermark] != first_resumed.messages[:watermark]
    assert any(
        message.content == _CONTINUE_REQUEST for message in first_resumed.messages
    )
    assert (
        sum(message.name == SUMMARY_MESSAGE_NAME for message in first_resumed.messages)
        == 1
    )
    rendered_summary = summary.summary.to_message().content
    assert _USER_CONSTRAINT in rendered_summary
    assert _FAILURE_CUE in rendered_summary
    assert first_evidence_id in rendered_summary
    assert all(
        call.id not in {"call-1", "call-2"}
        for message in first_resumed.messages
        for call in message.tool_calls
    )
    projected = ToolResultViewState(raw, views).project(raw)
    retained_third = next(
        message
        for message in first_resumed.messages
        if message.role is MessageRole.TOOL and message.tool_call_id == "call-3"
    )
    assert retained_third == projected[6]
    for request in resumed_adapter.requests:
        _assert_paired(request.messages)
    restored_raw = await reopened.load_messages(conversation.id)
    assert restored_raw[: len(raw)] == raw
    assert restored_raw == continued.result.messages
    assert await reopened.load_summary_state(conversation.id) == summary
    restored_views = await reopened.load_tool_result_views(conversation.id)
    assert restored_views[: len(views)] == views
    assert restored_views[-1].source_sequence > views[-1].source_sequence
    assert restored_views[-1].representation == "full"
    original_evidence = await reopened_evidence.resolve(
        first_evidence_id,
        conversation_id=conversation.id,
    )
    assert original_evidence is not None
    assert original_evidence.content == tool.payloads["first"]
