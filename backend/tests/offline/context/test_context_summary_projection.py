from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.models.types import Message, MessageRole, ModelUsage, ToolCall
from app.runtime.context.inventory import ContextInventory
from app.runtime.context.reducers.conversation import (
    ConversationReducer,
    build_summary_candidate,
)
from app.runtime.context.summarizer import ContextSummarizer, SummaryGenerationError
from app.runtime.context.summary import (
    SUMMARY_MESSAGE_NAME,
    ConversationSummaryState,
    RollingConversationSummary,
    SummaryGenerationResult,
)


# 函数说明：_history
# 用途：在回归测试与测试辅助中处理 `_history`，通过 `messages.extend` 完成首个内部处理步
# 骤。
# 参数：
#   rounds：已完成的模型或工具轮次，类型 `int`；默认 `6`。
# 返回：类型 `tuple[Message, ...]`；返回 `tuple(messages)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
def _history(rounds: int = 6) -> tuple[Message, ...]:
    messages = [Message(role=MessageRole.SYSTEM, content="系统约束")]
    for index in range(rounds):
        messages.extend(
            (
                Message(role=MessageRole.USER, content=f"问题 {index}: " + "问" * 500),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"回答 {index}: " + "答" * 500,
                ),
            )
        )
    return tuple(messages)


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
            tool_calls=(ToolCall(id=call_id, name="read_file", arguments={}),),
        ),
        Message(role=MessageRole.TOOL, content="原工具输出", tool_call_id=call_id),
    )


# 函数说明：_estimate
# 用途：估算回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   messages：本次处理的消息序列，类型 `tuple[Message, ...]`。
# 返回：类型 `int`；返回
# `sum((20 + len(message.content or '') for message in messages))`。
def _estimate(messages: tuple[Message, ...]) -> int:
    return sum(20 + len(message.content or "") for message in messages)


class _ScriptedSummarizer(ContextSummarizer):
    provider_hint = "summary-test"
    model_hint = "summary-test-small"

    # 函数说明：_ScriptedSummarizer.__init__
    # 用途：初始化 _ScriptedSummarizer；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   *responses：额外位置参数，按实现向内部调用传递。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.sources`、`self.previous`、
    # `self.output_limits`。
    def __init__(self, *responses: SummaryGenerationResult | Exception) -> None:
        self.responses = list(responses)
        self.sources: list[tuple[Message, ...]] = []
        self.previous: list[RollingConversationSummary | None] = []
        self.output_limits: list[int | None] = []

    # 函数说明：_ScriptedSummarizer.summarize
    # 用途：生成摘要_ScriptedSummarizer，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要，类型 `RollingConversationSummary | None`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `SummaryGenerationResult`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.responses.pop`。
    # 分支与异常：
    #   当 `isinstance(response, Exception)` 时，抛出 `response`。
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        self.previous.append(previous_summary)
        self.sources.append(tuple(messages))
        self.output_limits.append(max_output_tokens)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# 函数说明：_short_result
# 用途：返回 `SummaryGenerationResult(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   total_tokens：Token 数量或 Token 预算，类型 `int`；默认 `10`。
# 返回：类型 `SummaryGenerationResult`；返回 `SummaryGenerationResult(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SummaryGenerationResult` →
# `RollingConversationSummary` → `ModelUsage`。
def _short_result(total_tokens: int = 10) -> SummaryGenerationResult:
    return SummaryGenerationResult(
        summary=RollingConversationSummary(current_objective="继续原任务"),
        usage=ModelUsage(total_tokens=total_tokens),
    )


# 函数说明：test_copied_history_keeps_equal_current
# 用途：回归验证回归测试与测试辅助中的 `copied_history_keeps_equal_current` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `_ScriptedSummarizer` →
# `_short_result` →
# `ConversationReducer(summarizer, keep_recent_conversation_blocks=1,…` →
# `ConversationReducer` → `_estimate`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.error is None`。
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.summary_state.covered_message_count == 11`。
#   验证条件：`result.messages == (history[0], result.summary_state.summary.to_message()
# , *history[-2:], *current)`。
@pytest.mark.asyncio
async def test_copied_history_keeps_equal_current() -> None:
    history = _history()
    current = (history[1].model_copy(deep=True),)
    prepared = tuple(message.model_copy(deep=True) for message in (*history, *current))
    summarizer = _ScriptedSummarizer(_short_result())

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
        target_tokens=500,
        estimate=_estimate,
    )

    assert result.error is None
    assert result.summary_state is not None
    assert result.summary_state.covered_message_count == 11
    assert result.messages == (
        history[0],
        result.summary_state.summary.to_message(),
        *history[-2:],
        *current,
    )
    assert summarizer.sources == [history[1:11]]
    assert history[1].content == current[0].content


# 函数说明：test_reused_tool_call_id_keeps_current_round
# 用途：回归验证回归测试与测试辅助中的 `reused_tool_call_id_keeps_current_round` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_round` → `_history` → `Message`
#  → `_ScriptedSummarizer` → `_short_result` →
# `ConversationReducer(summarizer, keep_recent_conversation_blocks=1,…`；另有 2 个调用点
# 。
# 分支与异常：
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.summary_state.covered_message_count == 9`。
#   验证条件：`result.messages[-len(current):] == current`。
#   验证条件：`sum((bool(message.tool_calls) for message in result.messages)) == 1`。
@pytest.mark.asyncio
async def test_reused_tool_call_id_keeps_current_round() -> None:
    old_round = _tool_round("reused-id")
    history = (_history()[0], *old_round, *_history(4)[1:])
    current = (
        *_tool_round("reused-id"),
        Message(role=MessageRole.USER, content="继续"),
    )
    prepared = tuple(message.model_copy(deep=True) for message in (*history, *current))
    summarizer = _ScriptedSummarizer(_short_result())

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
        target_tokens=500,
        estimate=_estimate,
    )

    assert result.summary_state is not None
    assert result.summary_state.covered_message_count == 9
    assert result.messages[-len(current) :] == current
    assert sum(bool(message.tool_calls) for message in result.messages) == 1
    assert summarizer.sources == [history[1:9]]


# 函数说明：test_repeated_messages_keep_recent_and_current_coordinates
# 用途：回归验证回归测试与测试辅助中的
# `repeated_messages_keep_recent_and_current_coordinates` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_history` →
# `ContextInventory.build` → `_ScriptedSummarizer` → `_short_result` →
# `ConversationReducer(summarizer, keep_recent_conversation_blocks=1,…`；另有 2 个调用点
# 。
# 分支与异常：
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.summary_state.covered_message_count == 11`。
#   验证条件：`result.messages[-4:] == (*repeated, *current)`。
#   验证条件：
# `[entry.source_index for entry in projected.entries] == [0, None, 11, 12, 13, 14]`。
@pytest.mark.asyncio
async def test_repeated_messages_keep_recent_and_current_coordinates() -> None:
    repeated = (
        Message(role=MessageRole.USER, content="同样的问题" + "问" * 500),
        Message(role=MessageRole.ASSISTANT, content="同样的回答" + "答" * 500),
    )
    history = (_history()[0], *(repeated * 6))
    current = tuple(message.model_copy(deep=True) for message in repeated)
    inventory = ContextInventory.build(
        (*history, *current), history_count=len(history), keep_recent_tool_rounds=0
    )
    summarizer = _ScriptedSummarizer(_short_result())

    result, projected = await ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=1,
        keep_recent_tool_rounds=0,
    ).reduce_inventory(
        inventory,
        previous_state=None,
        initial_estimated_input_tokens=_estimate(inventory.messages),
        target_tokens=500,
        estimate=_estimate,
    )

    assert result.summary_state is not None
    assert result.summary_state.covered_message_count == 11
    assert result.messages[-4:] == (*repeated, *current)
    assert [entry.source_index for entry in projected.entries] == [
        0,
        None,
        11,
        12,
        13,
        14,
    ]
    assert inventory.messages == (*history, *current)


# 函数说明：test_tool_projection_keeps_raw_summary_watermark
# 用途：回归验证回归测试与测试辅助中的 `tool_projection_keeps_raw_summary_watermark` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `_tool_round` → `Message`
#  → `ContextInventory.build` → `ContextInventory.from_projection` →
# `_ScriptedSummarizer`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.summary_state.covered_message_count == 13`。
#   验证条件：`result.messages == (history[0], result.summary_state.summary.to_message()
# , *history[-2:], *current)`。
#   验证条件：`summarizer.sources == [(history[1], projected_tool, *history[3:13])]`。
@pytest.mark.asyncio
async def test_tool_projection_keeps_raw_summary_watermark() -> None:
    history = (
        _history()[0],
        *_tool_round("same-id"),
        *_tool_round("omitted"),
        *_history(5)[1:],
    )
    current = (*_tool_round("same-id"), Message(role=MessageRole.USER, content="继续"))
    inventory = ContextInventory.build(
        (*history, *current), history_count=len(history), keep_recent_tool_rounds=0
    )
    projected_tool = history[2].model_copy(update={"content": "缩短工具输出"})
    prepared = ContextInventory.from_projection(
        history,
        current,
        (history[0], history[1], projected_tool, *history[3:], *current),
        None,
        keep_recent_tool_rounds=0,
    )
    summarizer = _ScriptedSummarizer(_short_result())

    result, projected = await ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=1,
        keep_recent_tool_rounds=0,
    ).reduce_inventory(
        prepared,
        previous_state=None,
        initial_estimated_input_tokens=_estimate(prepared.messages),
        target_tokens=500,
        estimate=_estimate,
    )

    assert result.summary_state is not None
    assert result.summary_state.covered_message_count == 13
    assert result.messages == (
        history[0],
        result.summary_state.summary.to_message(),
        *history[-2:],
        *current,
    )
    assert summarizer.sources == [(history[1], projected_tool, *history[3:13])]
    assert projected.source_messages == inventory.source_messages
    assert projected.history_count == len(history)


# 函数说明：test_failed_retry_preserves_projection_watermark_and_usage
# 用途：回归验证回归测试与测试辅助中的
# `failed_retry_preserves_projection_watermark_and_usage` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` →
# `ConversationSummaryState` → `RollingConversationSummary` → `Message` →
# `ContextInventory.build` → `_ScriptedSummarizer`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`projected is inventory`。
#   验证条件：`result.messages == inventory.messages`。
#   验证条件：`result.summary_state == previous`。
#   验证条件：`result.summary_usage.total_tokens == 18`。
@pytest.mark.asyncio
async def test_failed_retry_preserves_projection_watermark_and_usage() -> None:
    history = _history()
    previous = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="前次目标"),
        covered_message_count=3,
    )
    current = (Message(role=MessageRole.USER, content="当前输入"),)
    inventory = ContextInventory.build(
        (*history, *current),
        history_count=len(history),
        summary_state=previous,
    )
    summarizer = _ScriptedSummarizer(
        SummaryGenerationResult(
            summary=RollingConversationSummary(current_objective="过长" * 10_000),
            usage=ModelUsage(total_tokens=7),
        ),
        SummaryGenerationError("重试不可用", usage=ModelUsage(total_tokens=11)),
    )

    result, projected = await ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=1,
        keep_recent_tool_rounds=0,
        large_fold_span_tokens=1_000,
        large_fold_max_output_tokens=2_048,
    ).reduce_inventory(
        inventory,
        previous_state=previous,
        initial_estimated_input_tokens=_estimate(inventory.messages),
        target_tokens=500,
        estimate=_estimate,
    )

    assert projected is inventory
    assert result.messages == inventory.messages
    assert result.summary_state == previous
    assert result.summary_usage.total_tokens == 18
    assert result.error == "SummaryGenerationError: 重试不可用"
    assert summarizer.previous == [previous.summary, previous.summary]
    assert summarizer.sources == [history[3:11], history[3:11]]
    assert summarizer.output_limits == [2_048, 2_048]
    assert result.summary_provider == "summary-test"
    assert result.summary_model == "summary-test-small"
    assert result.summary_duration_ms is not None
    assert result.summary_duration_ms >= 0


# 函数说明：test_saved_summary_watermark_uses_raw_count_after_restart
# 用途：回归验证回归测试与测试辅助中的
# `saved_summary_watermark_uses_raw_count_after_restart` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `Message` →
# `ConversationSummaryState.model_validate_json` → `build_summary_candidate` →
# `ContextInventory.build` → `state.summary.to_message`。
# 分支与异常：
#   验证条件：
# `candidate == (history[0], state.summary.to_message(), *history[7:], *current)`。
#   验证条件：`prepared_history_count == 2 + len(history[7:])`。
#   验证条件：`inventory.history_count == len(history)`。
#   验证条件：`state.covered_message_count == 7`。
def test_saved_summary_watermark_uses_raw_count_after_restart() -> None:
    history = _history()
    current = (Message(role=MessageRole.USER, content="继续"),)
    state = ConversationSummaryState.model_validate_json(
        '{"summary":{"current_objective":"原目标"},"covered_message_count":7}'
    )

    candidate, prepared_history_count = build_summary_candidate(history, current, state)
    inventory = ContextInventory.build(
        (*history, *current), history_count=len(history), summary_state=state
    )

    assert candidate == (history[0], state.summary.to_message(), *history[7:], *current)
    assert prepared_history_count == 2 + len(history[7:])
    assert inventory.history_count == len(history)
    assert state.covered_message_count == 7
    assert sum(message.name == SUMMARY_MESSAGE_NAME for message in candidate) == 1


# 函数说明：test_stale_watermark_keeps_projection
# 用途：回归验证回归测试与测试辅助中的 `stale_watermark_keeps_projection` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `Message` →
# `ConversationSummaryState` → `RollingConversationSummary` → `_ScriptedSummarizer` →
# `ConversationReducer(summarizer).reduce`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`result.messages == prepared`。
#   验证条件：`result.summary_state == state`。
#   验证条件：
# `result.error == 'summary covered_message_count exceeds current history length'`。
#   验证条件：`summarizer.sources == []`。
@pytest.mark.asyncio
async def test_stale_watermark_keeps_projection() -> None:
    history = _history(1)
    current = (Message(role=MessageRole.USER, content="当前输入"),)
    prepared = tuple(message.model_copy(deep=True) for message in (*history, *current))
    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="旧记录"),
        covered_message_count=len(history) + 1,
    )
    summarizer = _ScriptedSummarizer()

    result = await ConversationReducer(summarizer).reduce(
        raw_history=history,
        prepared_messages=prepared,
        current_messages=current,
        previous_state=state,
        initial_estimated_input_tokens=_estimate(prepared),
        target_tokens=100,
        estimate=_estimate,
    )

    assert result.messages == prepared
    assert result.summary_state == state
    assert result.error == (
        "summary covered_message_count exceeds current history length"
    )
    assert summarizer.sources == []


# 函数说明：test_invalid_saved_watermark_never_summarizes
# 用途：回归验证回归测试与测试辅助中的 `invalid_saved_watermark_never_summarizes` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   entry_point：条目输入或配置值，类型 `str`。
#   invalid_case：`invalid_case`输入或配置值，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` → `Message` → `_tool_round`
#  → `ConversationSummaryState` → `RollingConversationSummary` → `_ScriptedSummarizer`；
# 另有 5 个调用点。
# 分支与异常：
#   验证条件：`projected is inventory`。
#   验证条件：`result.messages == prepared`。
#   验证条件：`result.summary_state == state`。
#   验证条件：`result.summary_usage.total_tokens == 0`。
@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ("public", "inventory"))
@pytest.mark.parametrize("invalid_case", ("conversation", "tool", "malformed"))
async def test_invalid_saved_watermark_never_summarizes(
    entry_point: str, invalid_case: str
) -> None:
    conversation = _history(1)[1:]
    next_user = Message(role=MessageRole.USER, content="后续问题" + "问" * 500)
    if invalid_case == "conversation":
        history = (*conversation, next_user)
        covered = 1
    elif invalid_case == "tool":
        history = (*conversation, *_tool_round("old-call"), next_user)
        covered = 3
    else:
        history = (
            *conversation,
            Message(role=MessageRole.TOOL, tool_call_id="orphan", content="未配对输出"),
            next_user,
        )
        covered = 3
    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="旧摘要"),
        covered_message_count=covered,
    )
    current = (Message(role=MessageRole.USER, content="当前输入"),)
    prepared = tuple(message.model_copy(deep=True) for message in (*history, *current))
    summarizer = _ScriptedSummarizer()
    reducer = ConversationReducer(
        summarizer, keep_recent_conversation_blocks=0, keep_recent_tool_rounds=0
    )
    if entry_point == "public":
        result = await reducer.reduce(
            raw_history=history,
            prepared_messages=prepared,
            current_messages=current,
            previous_state=state,
            initial_estimated_input_tokens=_estimate(prepared),
            target_tokens=100,
            estimate=_estimate,
        )
    else:
        inventory = ContextInventory.build(
            (*history, *current), history_count=len(history), summary_state=state
        )
        result, projected = await reducer.reduce_inventory(
            inventory,
            previous_state=state,
            initial_estimated_input_tokens=_estimate(prepared),
            target_tokens=100,
            estimate=_estimate,
        )
        assert projected is inventory

    assert result.messages == prepared
    assert result.summary_state == state
    assert result.summary_usage.total_tokens == 0
    assert result.error == (
        "summary covered_message_count is not a valid raw-history boundary"
    )
    assert summarizer.sources == []


# 函数说明：test_valid_previous_state_accepts_unprojected_inventory
# 用途：回归验证回归测试与测试辅助中的
# `valid_previous_state_accepts_unprojected_inventory` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_history` →
# `ConversationSummaryState` → `RollingConversationSummary` → `Message` →
# `ContextInventory.build` → `_ScriptedSummarizer`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`inventory.summary_state is None`。
#   验证条件：`result.error is None`。
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.summary_state.covered_message_count == 11`。
@pytest.mark.asyncio
async def test_valid_previous_state_accepts_unprojected_inventory() -> None:
    history = _history()
    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="旧目标"),
        covered_message_count=3,
    )
    current = (Message(role=MessageRole.USER, content="继续"),)
    inventory = ContextInventory.build(
        (*history, *current), history_count=len(history), keep_recent_tool_rounds=0
    )
    assert inventory.summary_state is None
    summarizer = _ScriptedSummarizer(_short_result())

    result, projected = await ConversationReducer(
        summarizer, keep_recent_conversation_blocks=1, keep_recent_tool_rounds=0
    ).reduce_inventory(
        inventory,
        previous_state=state,
        initial_estimated_input_tokens=_estimate(inventory.messages),
        target_tokens=100,
        estimate=_estimate,
    )

    assert result.error is None
    assert result.summary_state is not None
    assert result.summary_state.covered_message_count == 11
    assert summarizer.previous == [state.summary]
    assert summarizer.sources == [history[3:11]]
    assert projected.messages == (
        history[0],
        result.summary_state.summary.to_message(),
        *history[-2:],
        *current,
    )
