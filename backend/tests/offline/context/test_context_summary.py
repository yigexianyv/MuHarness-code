from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import SecretStr

from app.domain.conversation import SQLiteConversationStore
from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from app.runtime.context import (
    ContextManager,
    ContextSettings,
    ContextSummarizer,
    ConversationReducer,
    ConversationSummaryState,
    ModelContextSummarizer,
    RollingConversationSummary,
    SQLiteConversationSummaryStore,
    SummaryGenerationResult,
    build_summary_candidate,
)


class FakeSummarizer(ContextSummarizer):
    provider_hint = "fake-summary"
    model_hint = "summary-small"

    # 函数说明：FakeSummarizer.__init__
    # 用途：初始化 FakeSummarizer；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   error：异常或错误信息，类型 `Exception | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.error`、`self.calls`。
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[
            tuple[RollingConversationSummary | None, tuple[Message, ...]]
        ] = []

    # 函数说明：FakeSummarizer.summarize
    # 用途：生成摘要FakeSummarizer，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要，类型 `RollingConversationSummary | None`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `SummaryGenerationResult`；返回 `SummaryGenerationResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SummaryGenerationResult` →
    # `RollingConversationSummary` → `ModelUsage`。
    # 分支与异常：
    #   当 `self.error is not None` 时，抛出 `self.error`。
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        self.calls.append((previous_summary, tuple(messages)))
        if self.error is not None:
            raise self.error
        return SummaryGenerationResult(
            summary=RollingConversationSummary(
                current_objective="继续完成当前任务",
                key_decisions=("使用滚动摘要",),
            ),
            usage=ModelUsage(input_tokens=30, output_tokens=10, total_tokens=40),
        )


class SummaryModelAdapter(ModelAdapter):
    # 函数说明：SummaryModelAdapter.__init__
    # 用途：初始化 SummaryModelAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   contents：`contents`输入或配置值，类型 `list[str | None] | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.requests`、`self.contents`。
    def __init__(
        self,
        config: ProviderConfig,
        *,
        contents: list[str | None] | None = None,
    ) -> None:
        super().__init__(config)
        self.requests: list[ModelRequest] = []
        self.contents = contents or [
            (
                '{"current_objective":"完成测试","user_constraints":[], '
                '"key_decisions":[],"completed_work":[],"current_state":[], '
                '"pending_work":[],"important_facts":[]}'
            )
        ]

    # 函数说明：SummaryModelAdapter.complete
    # 用途：完成SummaryModelAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.contents.pop` →
    # `ModelResponse` → `Message` → `ModelUsage`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        content = self.contents.pop(0)
        return ModelResponse(
            id="summary-response",
            provider="fake",
            model="fake-model",
            message=Message(
                role=MessageRole.ASSISTANT,
                content=content,
            ),
            usage=ModelUsage(input_tokens=11, output_tokens=4, total_tokens=15),
        )

    # 函数说明：SummaryModelAdapter.close
    # 用途：关闭SummaryModelAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class LongThenShortSummarizer(ContextSummarizer):
    # 函数说明：LongThenShortSummarizer.__init__
    # 用途：初始化 LongThenShortSummarizer；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.initial_calls`、`self.retry_calls`、`self.retry_reason`。
    def __init__(self) -> None:
        self.initial_calls = 0
        self.retry_calls = 0
        self.retry_reason: str | None = None

    # 函数说明：LongThenShortSummarizer.summarize
    # 用途：生成摘要LongThenShortSummarizer，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要，类型 `RollingConversationSummary | None`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `SummaryGenerationResult`；返回 `SummaryGenerationResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SummaryGenerationResult` →
    # `RollingConversationSummary` → `ModelUsage`。
    # 副作用与资源：
    #   更新对象字段：`self.initial_calls`。
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        self.initial_calls += 1
        return SummaryGenerationResult(
            summary=RollingConversationSummary(current_objective="长" * 5_000),
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    # 函数说明：LongThenShortSummarizer.retry_compact
    # 用途：压缩`retry`，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要，类型 `RollingConversationSummary | None`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `SummaryGenerationResult`；返回 `SummaryGenerationResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SummaryGenerationResult` →
    # `RollingConversationSummary` → `ModelUsage`。
    # 副作用与资源：
    #   更新对象字段：`self.retry_calls`、`self.retry_reason`。
    async def retry_compact(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        reason: str,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        self.retry_calls += 1
        self.retry_reason = reason
        return SummaryGenerationResult(
            summary=RollingConversationSummary(current_objective="短目标"),
            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
        )


# 函数说明：_history
# 用途：在回归测试与测试辅助中处理 `_history`，通过 `messages.extend` 完成首个内部处理步
# 骤。
# 参数：
#   rounds：已完成的模型或工具轮次，类型 `int`。
# 返回：类型 `tuple[Message, ...]`；返回 `tuple(messages)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
def _history(rounds: int) -> tuple[Message, ...]:
    messages = [Message(role=MessageRole.SYSTEM, content="主系统提示")]
    for index in range(rounds):
        messages.extend(
            (
                Message(
                    role=MessageRole.USER,
                    content=f"旧问题 {index} " + "问" * 80,
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"旧回答 {index} " + "答" * 80,
                ),
            )
        )
    return tuple(messages)


# 函数说明：_estimate
# 用途：估算回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   messages：本次处理的消息序列，类型 `tuple[Message, ...]`。
# 返回：类型 `int`；返回
# `sum((20 + len(message.content or '') for message in messages))`。
def _estimate(messages: tuple[Message, ...]) -> int:
    return sum(20 + len(message.content or "") for message in messages)


# 函数说明：_tool_round
# 用途：返回 `(Message(role=MessageRole.ASSISTANT, tool_calls=(ToolCall(id=call_id, name
# ='read_file',…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   call_id：工具调用标识，类型 `str`。
# 返回：类型 `tuple[Message, Message]`；返回 `(Message(role=MessageRole.ASSISTANT,
# tool_calls=(ToolCall(id=call_id, name='read_file',…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolCall`。
def _tool_round(call_id: str) -> tuple[Message, Message]:
    return (
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=(
                ToolCall(id=call_id, name="read_file", arguments={"path": "x"}),
            ),
        ),
        Message(
            role=MessageRole.TOOL,
            content=f"工具结果 {call_id}",
            tool_call_id=call_id,
        ),
    )


# 函数说明：test_context_below_trigger_does_not_call_summarizer
# 用途：回归验证回归测试与测试辅助中的 `context_below_trigger_does_not_call_summarizer`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeSummarizer` → `Message` →
# `ContextManager(conversation_reducer=ConversationReducer(summarizer)).…` →
# `ContextManager` → `ConversationReducer`。
# 分支与异常：
#   验证条件：`decision.messages == messages`。
#   验证条件：`decision.summary_updated is False`。
#   验证条件：`summarizer.calls == []`。
@pytest.mark.asyncio
async def test_context_below_trigger_does_not_call_summarizer() -> None:
    summarizer = FakeSummarizer()
    messages = (
        Message(role=MessageRole.USER, content="短问题"),
        Message(role=MessageRole.ASSISTANT, content="短回答"),
    )

    decision = await ContextManager(
        conversation_reducer=ConversationReducer(summarizer),
    ).prepare(messages, history_count=len(messages))

    assert decision.messages == messages
    assert decision.summary_updated is False
    assert summarizer.calls == []


# 函数说明：test_block_count_does_not_trigger_summary_below_pressure
# 用途：回归验证回归测试与测试辅助中的
# `block_count_does_not_trigger_summary_below_pressure` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `Message` →
# `FakeSummarizer` → `ContextManager` → `ContextSettings` → `ConversationReducer`；另有
# 1 个调用点。
# 分支与异常：
#   验证条件：`decision.original_estimated_input_tokens < decision.trigger_tokens`。
#   验证条件：`decision.conversation_block_triggered is False`。
#   验证条件：`decision.unsummarized_conversation_blocks == 3`。
#   验证条件：`decision.summary_updated is False`。
@pytest.mark.asyncio
async def test_block_count_does_not_trigger_summary_below_pressure() -> None:
    history = _history(3)
    current = (Message(role=MessageRole.USER, content="继续"),)
    summarizer = FakeSummarizer()
    manager = ContextManager(
        context_settings=ContextSettings(
            _env_file=None,
            context_preferred_input_tokens=100_000,
        ),
        conversation_reducer=ConversationReducer(
            summarizer,
            keep_recent_conversation_blocks=1,
            keep_recent_tool_rounds=0,
        ),
    )

    decision = await manager.prepare(
        (*history, *current),
        history_count=len(history),
    )

    assert decision.original_estimated_input_tokens < decision.trigger_tokens
    assert decision.conversation_block_triggered is False
    assert decision.unsummarized_conversation_blocks == 3
    assert decision.summary_updated is False
    assert decision.summary_state is None
    assert len(summarizer.calls) == 0


# 函数说明：test_summary_watermark_prevents_resummarizing_same_history
# 用途：回归验证回归测试与测试辅助中的
# `summary_watermark_prevents_resummarizing_same_history` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `Message` →
# `FakeSummarizer` → `ContextSettings` → `ContextManager` → `ConversationReducer`；另有
# 1 个调用点。
# 分支与异常：
#   验证条件：`first.summary_state is not None`。
#   验证条件：`second.summary_updated is False`。
#   验证条件：`second.unsummarized_conversation_blocks == 1`。
#   验证条件：`len(summarizer.calls) == 1`。
@pytest.mark.asyncio
async def test_summary_watermark_prevents_resummarizing_same_history() -> None:
    history = _history(3)
    current = (Message(role=MessageRole.USER, content="继续"),)
    summarizer = FakeSummarizer()
    settings = ContextSettings(
        _env_file=None,
        context_preferred_input_tokens=100_000,
    )
    manager = ContextManager(
        context_settings=settings,
        conversation_reducer=ConversationReducer(
            summarizer,
            keep_recent_conversation_blocks=1,
            keep_recent_tool_rounds=0,
        ),
    )
    first = await manager.prepare(
        (*history, *current),
        history_count=len(history),
        compaction_target_tokens=500,
    )

    second = await manager.prepare(
        (*history, *current),
        history_count=len(history),
        summary_state=first.summary_state,
    )

    assert first.summary_state is not None
    assert second.summary_updated is False
    assert second.unsummarized_conversation_blocks == 1
    assert len(summarizer.calls) == 1


# 函数说明：test_conversation_reducer_summarizes_old_prefix_and_keeps_recent
# 用途：回归验证回归测试与测试辅助中的
# `conversation_reducer_summarizes_old_prefix_and_keeps_recent` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `Message` →
# `FakeSummarizer` →
# `ConversationReducer(summarizer, keep_recent_conversation_blocks=2,…` →
# `ConversationReducer` → `_estimate`。
# 分支与异常：
#   验证条件：`result.error is None`。
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.summary_state.covered_message_count == 9`。
#   验证条件：`result.summarized_conversation_blocks == 4`。
@pytest.mark.asyncio
async def test_conversation_reducer_summarizes_old_prefix_and_keeps_recent() -> None:
    history = _history(6)
    current = (Message(role=MessageRole.USER, content="当前问题"),)
    prepared = (*history, *current)
    summarizer = FakeSummarizer()

    result = await ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=2,
        keep_recent_tool_rounds=0,
    ).reduce(
        raw_history=history,
        prepared_messages=prepared,
        current_messages=current,
        previous_state=None,
        initial_estimated_input_tokens=_estimate(prepared),
        target_tokens=700,
        estimate=_estimate,
    )

    assert result.error is None
    assert result.summary_state is not None
    assert result.summary_state.covered_message_count == 9
    assert result.summarized_conversation_blocks == 4
    assert result.summary_usage.total_tokens == 40
    assert result.summary_provider == "fake-summary"
    assert result.summary_model == "summary-small"
    assert result.summary_duration_ms is not None
    assert result.summary_duration_ms >= 0
    assert result.messages[0] == history[0]
    assert result.messages[-5:] == (*history[-4:], *current)
    assert result.messages[1].name == "muharness_rolling_summary"
    assert len(summarizer.calls[0][1]) == 8


# 函数说明：test_stale_tool_rounds_do_not_block_later_summary
# 用途：回归验证回归测试与测试辅助中的 `stale_tool_rounds_do_not_block_later_summary` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_round` → `_history` → `Message`
#  → `FakeSummarizer` →
# `ConversationReducer(summarizer, keep_recent_conversation_blocks=4,…` →
# `ConversationReducer`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.error is None`。
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.summarized_conversation_blocks == 3`。
#   验证条件：`result.summary_state.covered_message_count == 11`。
@pytest.mark.asyncio
async def test_stale_tool_rounds_do_not_block_later_summary() -> None:
    stale_tools = (*_tool_round("old-1"), *_tool_round("old-2"))
    conversations = _history(7)[1:]
    history = (
        Message(role=MessageRole.SYSTEM, content="主系统提示"),
        *stale_tools,
        *conversations,
    )
    current = (Message(role=MessageRole.USER, content="当前问题"),)
    prepared = (*history, *current)
    summarizer = FakeSummarizer()

    result = await ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=4,
        keep_recent_tool_rounds=2,
    ).reduce(
        raw_history=history,
        prepared_messages=prepared,
        current_messages=current,
        previous_state=None,
        initial_estimated_input_tokens=_estimate(prepared),
        target_tokens=100,
        estimate=_estimate,
    )

    assert result.error is None
    assert result.summary_state is not None
    assert result.summarized_conversation_blocks == 3
    assert result.summary_state.covered_message_count == 11
    assert all(
        message.tool_call_id not in {"old-1", "old-2"} for message in result.messages
    )
    assert all(not message.tool_calls for message in result.messages)
    assert result.messages[-9:] == (*conversations[-8:], *current)


# 函数说明：test_tool_rounds_inside_recent_conversation_region_remain_protected
# 用途：回归验证回归测试与测试辅助中的
# `tool_rounds_inside_recent_conversation_region_remain_protected` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `_tool_round` → `Message`
#  → `FakeSummarizer` →
# `ConversationReducer(summarizer, keep_recent_conversation_blocks=2,…` →
# `ConversationReducer`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.summary_state is not None`。
#   验证条件：`any((message.tool_calls and message.tool_calls[0].id == 'recent' for
# message in result.messages))`。
#   验证条件：`any((message.tool_call_id == 'recent' for message in result.messages))`。
@pytest.mark.asyncio
async def test_tool_rounds_inside_recent_conversation_region_remain_protected() -> None:
    older = _history(4)
    recent_tool = _tool_round("recent")
    recent_conversation = (
        Message(role=MessageRole.ASSISTANT, content="工具后的结论"),
        Message(role=MessageRole.USER, content="继续追问"),
        Message(role=MessageRole.ASSISTANT, content="继续回答"),
    )
    history = (*older, *recent_tool, *recent_conversation)
    current = (Message(role=MessageRole.USER, content="当前问题"),)
    prepared = (*history, *current)
    summarizer = FakeSummarizer()

    result = await ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=2,
        keep_recent_tool_rounds=1,
    ).reduce(
        raw_history=history,
        prepared_messages=prepared,
        current_messages=current,
        previous_state=None,
        initial_estimated_input_tokens=_estimate(prepared),
        target_tokens=100,
        estimate=_estimate,
    )

    assert result.summary_state is not None
    assert any(
        message.tool_calls and message.tool_calls[0].id == "recent"
        for message in result.messages
    )
    assert any(message.tool_call_id == "recent" for message in result.messages)


# 函数说明：test_summary_failure_never_removes_original_messages
# 用途：回归验证回归测试与测试辅助中的 `summary_failure_never_removes_original_messages`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `Message` →
# `FakeSummarizer` →
# `ConversationReducer(summarizer, keep_recent_conversation_blocks=1,…` →
# `ConversationReducer` → `_estimate`。
# 分支与异常：
#   验证条件：`result.messages == prepared`。
#   验证条件：`result.summary_state is None`。
#   验证条件：`result.error == 'RuntimeError: 模型不可用'`。
#   验证条件：`len(summarizer.calls) == 2`。
@pytest.mark.asyncio
async def test_summary_failure_never_removes_original_messages() -> None:
    history = _history(5)
    current = (Message(role=MessageRole.USER, content="继续"),)
    prepared = (*history, *current)
    summarizer = FakeSummarizer(error=RuntimeError("模型不可用"))

    result = await ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=1,
        keep_recent_tool_rounds=0,
    ).reduce(
        raw_history=history,
        prepared_messages=prepared,
        current_messages=current,
        previous_state=None,
        initial_estimated_input_tokens=_estimate(prepared),
        target_tokens=100,
        estimate=_estimate,
    )

    assert result.messages == prepared
    assert result.summary_state is None
    assert result.error == "RuntimeError: 模型不可用"
    assert len(summarizer.calls) == 2


# 函数说明：test_rolling_summary_uses_previous_summary_and_advances_coverage
# 用途：回归验证回归测试与测试辅助中的
# `rolling_summary_uses_previous_summary_and_advances_coverage` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` →
# `ConversationSummaryState` → `RollingConversationSummary` → `Message` →
# `build_summary_candidate` → `FakeSummarizer`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`summarizer.calls[0][0] == previous.summary`。
#   验证条件：`result.summary_state is not None`。
#   验证条件：
# `result.summary_state.covered_message_count > previous.covered_message_count`。
#   验证条件：`len(summary_messages) == 1`。
@pytest.mark.asyncio
async def test_rolling_summary_uses_previous_summary_and_advances_coverage() -> None:
    history = _history(7)
    previous = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="旧目标"),
        covered_message_count=5,
    )
    current = (Message(role=MessageRole.USER, content="新输入"),)
    candidate, _ = build_summary_candidate(history, current, previous)
    summarizer = FakeSummarizer()

    result = await ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=2,
        keep_recent_tool_rounds=0,
    ).reduce(
        raw_history=history,
        prepared_messages=candidate,
        current_messages=current,
        previous_state=previous,
        initial_estimated_input_tokens=_estimate(candidate),
        target_tokens=100,
        estimate=_estimate,
    )

    assert summarizer.calls[0][0] == previous.summary
    assert result.summary_state is not None
    assert result.summary_state.covered_message_count > previous.covered_message_count
    summary_messages = [
        message
        for message in result.messages
        if message.name == "muharness_rolling_summary"
    ]
    assert len(summary_messages) == 1


# 函数说明：test_canonical_system_named_like_legacy_summary_is_preserved
# 用途：回归验证回归测试与测试辅助中的
# `canonical_system_named_like_legacy_summary_is_preserved` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ConversationSummaryState`
#  → `RollingConversationSummary` → `build_summary_candidate`。
# 分支与异常：
#   验证条件：`len(candidate) == 2`。
#   验证条件：`candidate[0] == history[0]`。
#   验证条件：`candidate[1].name == 'muharness_rolling_summary'`。
def test_canonical_system_named_like_legacy_summary_is_preserved() -> None:
    history = (
        Message(
            role=MessageRole.SYSTEM,
            name="vesta_rolling_summary",
            content="旧版摘要",
        ),
        Message(role=MessageRole.USER, content="旧问题"),
    )
    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="当前目标"),
        covered_message_count=len(history),
    )

    candidate, _ = build_summary_candidate(history, (), state)

    assert len(candidate) == 2
    assert candidate[0] == history[0]
    assert candidate[1].name == "muharness_rolling_summary"


# 函数说明：test_sqlite_summary_store_round_trip
# 用途：回归验证回归测试与测试辅助中的 `sqlite_summary_store_round_trip` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `conversation_store.initialize` → `conversation_store.create` →
# `SQLiteConversationSummaryStore` → `store.initialize` → `ConversationSummaryState`；另
# 有 5 个调用点。
# 分支与异常：
#   验证条件：`await store.load(conversation.id) == state`。
#   验证条件：`await store.delete(conversation.id) is True`。
#   验证条件：`await store.load(conversation.id) is None`。
@pytest.mark.asyncio
async def test_sqlite_summary_store_round_trip(tmp_path) -> None:
    database_path = tmp_path / "muharness.db"
    conversation_store = SQLiteConversationStore(database_path)
    await conversation_store.initialize()
    conversation = await conversation_store.create()
    store = SQLiteConversationSummaryStore(database_path)
    await store.initialize()
    state = ConversationSummaryState(
        summary=RollingConversationSummary(
            current_objective="测试持久化",
            pending_work=("继续运行",),
        ),
        covered_message_count=12,
    )

    await store.save(conversation.id, state)

    assert await store.load(conversation.id) == state
    assert await store.delete(conversation.id) is True
    assert await store.load(conversation.id) is None

    await store.save(conversation.id, state)
    await conversation_store.delete(conversation.id)
    assert await store.load(conversation.id) is None


# 函数说明：test_model_summarizer_requests_strict_json_without_tools
# 用途：回归验证回归测试与测试辅助中的
# `model_summarizer_requests_strict_json_without_tools` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `SummaryModelAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`
# ；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result.summary.current_objective == '完成测试'`。
#   验证条件：`result.usage.total_tokens == 15`。
#   验证条件：`adapter.requests[0].tools == ()`。
#   验证条件：`adapter.requests[0].messages[0].role is MessageRole.SYSTEM`。
@pytest.mark.asyncio
async def test_model_summarizer_requests_strict_json_without_tools() -> None:
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = SummaryModelAdapter(config)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)
    summarizer = ModelContextSummarizer(registry, provider="fake")

    result = await summarizer.summarize(
        None,
        (Message(role=MessageRole.USER, content="请完成测试"),),
    )

    assert result.summary.current_objective == "完成测试"
    assert result.usage.total_tokens == 15
    assert adapter.requests[0].tools == ()
    assert adapter.requests[0].messages[0].role is MessageRole.SYSTEM


# 函数说明：test_invalid_model_summary_retries_once_and_counts_usage
# 用途：回归验证回归测试与测试辅助中的
# `invalid_model_summary_retries_once_and_counts_usage` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `SummaryModelAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`
# ；另有 6 个调用点。
# 分支与异常：
#   验证条件：`result.error is None`。
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.summary_state.summary.current_objective == '短目标'`。
#   验证条件：`result.summary_usage.total_tokens == 30`。
@pytest.mark.asyncio
async def test_invalid_model_summary_retries_once_and_counts_usage() -> None:
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    too_many_entries = (
        '{"current_objective":"目标","user_constraints":'
        '["1","2","3","4","5","6","7","8","9"],"key_decisions":[],'
        '"completed_work":[],"current_state":[],"pending_work":[],'
        '"important_facts":[]}'
    )
    valid = (
        '{"current_objective":"短目标","user_constraints":[],'
        '"key_decisions":[],"completed_work":[],"current_state":[],'
        '"pending_work":[],"important_facts":[]}'
    )
    adapter = SummaryModelAdapter(
        config,
        contents=[too_many_entries, valid],
    )
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)
    history = _history(5)
    current = (Message(role=MessageRole.USER, content="继续"),)

    result = await ConversationReducer(
        ModelContextSummarizer(registry, provider="fake"),
        keep_recent_conversation_blocks=1,
        keep_recent_tool_rounds=0,
    ).reduce(
        raw_history=history,
        prepared_messages=(*history, *current),
        current_messages=current,
        previous_state=None,
        initial_estimated_input_tokens=_estimate((*history, *current)),
        target_tokens=100,
        estimate=_estimate,
    )

    assert result.error is None
    assert result.summary_state is not None
    assert result.summary_state.summary.current_objective == "短目标"
    assert result.summary_usage.total_tokens == 30
    assert len(adapter.requests) == 2
    assert "唯一重试机会" in (adapter.requests[1].messages[0].content or "")
    schema_text = adapter.requests[0].messages[1].content or ""
    assert '"maxItems":8' in schema_text
    assert '"maxLength":80' in schema_text


# 函数说明：test_summary_that_does_not_reduce_retries_once
# 用途：回归验证回归测试与测试辅助中的 `summary_that_does_not_reduce_retries_once` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `Message` →
# `LongThenShortSummarizer` →
# `ConversationReducer(summarizer, keep_recent_conversation_blocks=1,…` →
# `ConversationReducer` → `_estimate`。
# 分支与异常：
#   验证条件：`result.error is None`。
#   验证条件：`result.summary_state is not None`。
#   验证条件：`summarizer.initial_calls == 1`。
#   验证条件：`summarizer.retry_calls == 1`。
@pytest.mark.asyncio
async def test_summary_that_does_not_reduce_retries_once() -> None:
    history = _history(5)
    current = (Message(role=MessageRole.USER, content="继续"),)
    prepared = (*history, *current)
    summarizer = LongThenShortSummarizer()

    result = await ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=1,
        keep_recent_tool_rounds=0,
    ).reduce(
        raw_history=history,
        prepared_messages=prepared,
        current_messages=current,
        previous_state=None,
        initial_estimated_input_tokens=_estimate(prepared),
        target_tokens=100,
        estimate=_estimate,
    )

    assert result.error is None
    assert result.summary_state is not None
    assert summarizer.initial_calls == 1
    assert summarizer.retry_calls == 1
    assert summarizer.retry_reason == (
        "generated summary did not reduce the request context"
    )
    assert result.summary_usage.total_tokens == 25
