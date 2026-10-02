from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from app.models.types import Message, MessageRole, ModelUsage, ToolCall, ToolDefinition
from app.runtime.context import (
    ContextBudgetPolicy,
    ContextManager,
    ContextSettings,
    ContextSummarizer,
    ConversationReducer,
    ConversationSummaryState,
    ModelCapabilityRegistry,
    RollingConversationSummary,
    SummaryGenerationResult,
)
from app.runtime.context.summarizer import _history_item


class _Estimator:
    # 函数说明：_Estimator.estimate_messages
    # 用途：估算消息序列，供回归测试与测试辅助使用。
    # 参数：
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：类型 `int`；返回
    # `sum((20 + len(message.content or '') for message in messages))`。
    def estimate_messages(self, messages: Sequence[Message], **kwargs) -> int:
        return sum(20 + len(message.content or "") for message in messages)

    # 函数说明：_Estimator.estimate_tools
    # 用途：估算工具集合，供回归测试与测试辅助使用。
    # 参数：
    #   tools：可用工具定义或工具实例集合，类型 `Sequence[ToolDefinition]`。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：类型 `int`；返回
    # `sum((len(tool.name) + len(tool.description) for tool in tools))`。
    def estimate_tools(self, tools: Sequence[ToolDefinition], **kwargs) -> int:
        return sum(len(tool.name) + len(tool.description) for tool in tools)

    # 函数说明：_Estimator.estimate_request
    # 用途：估算请求，供回归测试与测试辅助使用。
    # 参数：
    #   messages：本次处理的消息序列。
    #   tools：可用工具定义或工具实例集合；默认 `()`。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：类型 `int`；返回
    # `self.estimate_messages(messages) + self.estimate_tools(tools)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.estimate_messages` →
    # `self.estimate_tools`。
    def estimate_request(self, messages, *, tools=(), **kwargs) -> int:
        return self.estimate_messages(messages) + self.estimate_tools(tools)


class _Summary(ContextSummarizer):
    # 函数说明：_Summary.__init__
    # 用途：初始化 _Summary；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   empty：`empty`输入或配置值；默认 `False`。
    #   error：异常或错误信息；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.sources`、`self.empty`、`self.error`。
    def __init__(self, *, empty=False, error=None) -> None:
        self.sources: list[tuple[Message, ...]] = []
        self.empty = empty
        self.error = error

    # 函数说明：_Summary.summarize
    # 用途：生成摘要_Summary，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要。
    #   messages：本次处理的消息序列。
    #   max_output_tokens：模型输出 Token 上限；默认 `None`。
    # 返回：返回 `SummaryGenerationResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SummaryGenerationResult` →
    # `RollingConversationSummary` → `ModelUsage`。
    # 分支与异常：
    #   当 `self.error` 时，抛出 `self.error`。
    async def summarize(self, previous_summary, messages, *, max_output_tokens=None):
        self.sources.append(tuple(messages))
        if self.error:
            raise self.error
        return SummaryGenerationResult(
            summary=RollingConversationSummary()
            if self.empty
            else RollingConversationSummary(current_objective="继续当前任务"),
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


# 函数说明：_manager
# 用途：在回归测试与测试辅助中处理 `_manager`，通过 `registry.register_override` 完成首
# 个内部处理步骤。
# 参数：
#   summarizer：传给 `ConversationReducer` 的输入；默认 `None`。
#   ceiling：`ceiling`输入或配置值；默认 `2200`。
#   keep：`keep`输入或配置值；默认 `2`。
#   window：`window`输入或配置值；默认 `30000`。
#   keep_conversations：`keep_conversations`输入或配置值；默认 `2`。
# 返回：返回 `ContextManager(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelCapabilityRegistry` →
# `registry.register_override` → `ContextManager` → `_Estimator` → `ContextBudgetPolicy`
#  → `ContextSettings`；另有 1 个调用点。
def _manager(
    summarizer=None, *, ceiling=2_200, keep=2, window=30_000, keep_conversations=2
):
    registry = ModelCapabilityRegistry()
    registry.register_override(
        "test", "test-model", context_window=window, max_output_tokens=100
    )
    return ContextManager(
        _Estimator(),
        registry=registry,
        budget_policy=ContextBudgetPolicy(
            safety_margin_tokens=0,
            preferred_input_tokens=1_200,
            compact_input_tokens=ceiling,
        ),
        context_settings=ContextSettings(_env_file=None),
        conversation_reducer=ConversationReducer(
            summarizer,
            keep_recent_tool_rounds=keep,
            keep_recent_conversation_blocks=keep_conversations,
        )
        if summarizer is not None
        else None,
    )


# 函数说明：_round
# 用途：在回归测试与测试辅助中处理 `_round`，通过 `json.dumps` 完成首个内部处理步骤。
# 参数：
#   index：当前位置或索引。
#   size：`size`输入或配置值；默认 `600`。
# 返回：返回 `(Message(role=MessageRole.ASSISTANT, tool_calls=(call,)), Message(role=
# MessageRole.TOOL,…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `json.dumps` → `Message`
# 。
def _round(index, *, size=600):
    call = ToolCall(
        id=f"call-{index}", name="read_file", arguments={"path": f"file{index}.py"}
    )
    output = json.dumps(
        {
            "tool_call_id": call.id,
            "tool_name": call.name,
            "success": index != 0,
            "error": "test failed" if index == 0 else None,
            "evidence_id": f"evidence-{index}",
            "output": "x" * size,
        }
    )
    return (
        Message(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        Message(
            role=MessageRole.TOOL, name=call.name, tool_call_id=call.id, content=output
        ),
    )


# 函数说明：_prepare
# 用途：准备回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   manager：当前业务管理器。
#   candidate：候选记录。
#   source：输入来源或原始数据；默认 `None`。
#   **kwargs：额外关键字参数，按实现处理或转交。
# 返回：返回 `await manager.prepare(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`manager.prepare`。
async def _prepare(manager, candidate, source=None, **kwargs):
    return await manager.prepare(
        candidate,
        source_messages=source,
        model="test-model",
        provider="test",
        max_output_tokens=100,
        **kwargs,
    )


# 函数说明：test_soft_and_tool_pressure_do_not_rewrite_a_prefix
# 用途：回归验证回归测试与测试辅助中的 `soft_and_tool_pressure_do_not_rewrite_a_prefix`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `_Summary` →
# `_prepare` → `_manager`。
# 分支与异常：
#   验证条件：`decision.original_estimated_input_tokens > 960`。
#   验证条件：`decision.tool_result_tokens_before > decision.tool_result_budget_tokens`
# 。
#   验证条件：`decision.messages == source`。
#   验证条件：`decision.requires_compaction is False`。
@pytest.mark.asyncio
async def test_soft_and_tool_pressure_do_not_rewrite_a_prefix() -> None:
    source = (Message(role=MessageRole.USER, content="原任务"), *_round(0, size=1_300))
    summarizer = _Summary()
    decision = await _prepare(
        _manager(summarizer), source, source, protected_source_indices=(0,)
    )
    assert decision.original_estimated_input_tokens > 960
    assert decision.tool_result_tokens_before > decision.tool_result_budget_tokens
    assert decision.messages == source
    assert decision.requires_compaction is False
    assert summarizer.sources == []
    assert decision.compacted_tool_results == decision.removed_tool_rounds == 0


# 函数说明：test_current_run_tools_fold_with_evidence_and_pinned_user
# 用途：回归验证回归测试与测试辅助中的
# `current_run_tools_fold_with_evidence_and_pinned_user` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `_Summary` →
# `_prepare` → `_manager` → `_history_item`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`decision.summary_updated`。
#   验证条件：`decision.summary_state.covered_message_count == 5`。
#   验证条件：`user in decision.messages`。
#   验证条件：`decision.messages[-4:] == source[-4:]`。
@pytest.mark.asyncio
async def test_current_run_tools_fold_with_evidence_and_pinned_user() -> None:
    user = Message(role=MessageRole.USER, content="不能改接口；验证后再报完成")
    source = (user, *sum((_round(i) for i in range(4)), ()))
    system = Message(role=MessageRole.SYSTEM, content="固定规则")
    summarizer = _Summary()
    decision = await _prepare(
        _manager(summarizer), (system, *source), source, protected_source_indices=(0,)
    )
    assert decision.summary_updated
    assert decision.summary_state.covered_message_count == 5
    assert user in decision.messages
    assert decision.messages[-4:] == source[-4:]
    assert decision.messages[0] == system
    assert summarizer.sources == [source[:5]]
    assert _history_item(summarizer.sources[0][1])["tool_calls"][0]["arguments"] == {
        "path": "file0.py"
    }
    failed = json.loads(summarizer.sources[0][2].content)
    assert failed["success"] is False
    assert failed["error"] == "test failed"
    assert failed["evidence_id"] == "evidence-0"
    assert decision.compaction_stage.value == "rolling_summary"
    assert decision.removed_tool_rounds == 0


# 函数说明：test_pending_round_and_result_pairs_remain_intact
# 用途：回归验证回归测试与测试辅助中的 `pending_round_and_result_pairs_remain_intact` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `ToolCall` →
# `_prepare` → `_manager` → `_Summary`。
# 分支与异常：
#   验证条件：`decision.summary_updated`。
#   验证条件：`decision.messages[-1] == source[-1]`。
#   验证条件：`decision.summary_state.covered_message_count == 5`。
@pytest.mark.asyncio
async def test_pending_round_and_result_pairs_remain_intact() -> None:
    source = (
        Message(role=MessageRole.USER, content="go"),
        *sum((_round(i) for i in range(4)), ()),
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=(ToolCall(id="pending", name="write_file", arguments={}),),
        ),
    )
    decision = await _prepare(
        _manager(_Summary()), source, source, protected_source_indices=(0,)
    )
    assert decision.summary_updated
    assert decision.messages[-1] == source[-1]
    assert decision.summary_state.covered_message_count == 5


# 函数说明：test_previous_summary_applies_once_and_keeps_append_order
# 用途：回归验证回归测试与测试辅助中的
# `previous_summary_applies_once_and_keeps_append_order` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `_prepare` →
# `_manager` → `_Summary` → `from_raw.messages.count`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`user in from_raw.messages`。
#   验证条件：`from_raw.messages.count(first.summary_state.summary.to_message()) == 1`。
#   验证条件：`continuation.messages == sent`。
@pytest.mark.asyncio
async def test_previous_summary_applies_once_and_keeps_append_order() -> None:
    user = Message(role=MessageRole.USER, content="current")
    source = (user, *sum((_round(i) for i in range(4)), ()))
    first = await _prepare(
        _manager(_Summary()), source, source, protected_source_indices=(0,)
    )
    addition = (Message(role=MessageRole.ASSISTANT, content="下一步"),)
    raw = (*source, *addition)
    manager = _manager(ceiling=20_000)
    from_raw = await _prepare(
        manager,
        raw,
        raw,
        summary_state=first.summary_state,
        protected_source_indices=(0,),
    )
    assert user in from_raw.messages
    assert from_raw.messages.count(first.summary_state.summary.to_message()) == 1
    sent = (*first.messages, *addition)
    continuation = await _prepare(
        manager,
        sent,
        raw,
        summary_state=first.summary_state,
        protected_source_indices=(0,),
    )
    assert continuation.messages == sent


# 函数说明：test_runtime_task_dedup_never_drops_canonical_named_user
# 用途：回归验证回归测试与测试辅助中的
# `runtime_task_dedup_never_drops_canonical_named_user` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `_Summary` →
# `_prepare` → `_manager`。
# 分支与异常：
#   验证条件：`decision.summary_state.covered_message_count == 5`。
#   验证条件：`user in decision.messages and new_task in decision.messages`。
#   验证条件：`old_task not in decision.messages`。
#   验证条件：
# `all((message not in summarizer.sources[0] for message in (old_task, new_task)))`。
@pytest.mark.asyncio
async def test_runtime_task_dedup_never_drops_canonical_named_user() -> None:
    user = Message(
        role=MessageRole.USER, name="muharness_active_task", content="原始用户数据"
    )
    source = (user, *sum((_round(i) for i in range(4)), ()))
    old_task = Message(
        role=MessageRole.USER, name="muharness_active_task", content="revision=1"
    )
    new_task = old_task.model_copy(update={"content": "revision=2"})
    summarizer = _Summary()
    decision = await _prepare(
        _manager(summarizer),
        (*source, old_task, new_task),
        source,
        protected_source_indices=(0,),
    )
    assert decision.summary_state.covered_message_count == 5
    assert user in decision.messages and new_task in decision.messages
    assert old_task not in decision.messages
    assert all(message not in summarizer.sources[0] for message in (old_task, new_task))


# 函数说明：test_repeated_user_text_and_task_injection_do_not_shift_watermark
# 用途：回归验证回归测试与测试辅助中的
# `repeated_user_text_and_task_injection_do_not_shift_watermark` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `_prepare` →
# `_manager` → `_Summary`。
# 分支与异常：
#   验证条件：`decision.summary_updated`。
#   验证条件：`decision.summary_state.covered_message_count <= len(source)`。
#   验证条件：`decision.messages[-1] == repeated`。
#   验证条件：`task in decision.messages`。
@pytest.mark.asyncio
async def test_repeated_user_text_and_task_injection_do_not_shift_watermark() -> None:
    repeated = Message(role=MessageRole.USER, content="重复目标")
    source = (repeated, *sum((_round(i) for i in range(4)), ()), repeated)
    task = Message(
        role=MessageRole.USER, name="muharness_active_task", content="进度数据"
    )
    decision = await _prepare(
        _manager(_Summary(), keep_conversations=0),
        (task, *source),
        source,
        protected_source_indices=(9,),
    )
    assert decision.summary_updated
    assert decision.summary_state.covered_message_count <= len(source)
    assert decision.messages[-1] == repeated
    assert task in decision.messages


# 函数说明：test_repeated_tool_rounds_with_injected_task_still_fold
# 用途：回归验证回归测试与测试辅助中的
# `repeated_tool_rounds_with_injected_task_still_fold` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `_Summary` →
# `_prepare` → `_manager`。
# 分支与异常：
#   验证条件：`decision.summary_updated`。
#   验证条件：`decision.summary_state.covered_message_count == 9`。
#   验证条件：`len(summarizer.sources) == 1`。
#   验证条件：
# `decision.prepared_input_tokens < decision.original_estimated_input_tokens`。
@pytest.mark.asyncio
async def test_repeated_tool_rounds_with_injected_task_still_fold() -> None:
    user = Message(role=MessageRole.USER, content="current goal")
    repeated_round = _round(0, size=3_000)
    source = (user, *sum((repeated_round for _ in range(6)), ()))
    task = Message(
        role=MessageRole.USER, name="muharness_active_task", content="progress"
    )
    summarizer = _Summary()
    decision = await _prepare(
        _manager(summarizer), (task, *source), source, protected_source_indices=(0,)
    )
    assert decision.summary_updated
    assert decision.summary_state.covered_message_count == 9
    assert len(summarizer.sources) == 1
    assert decision.prepared_input_tokens < decision.original_estimated_input_tokens
    assert decision.messages[-4:] == source[-4:]
    assert user in decision.messages and task in decision.messages


# 函数说明：test_empty_or_failed_compaction_keeps_source_and_is_reported
# 用途：回归验证回归测试与测试辅助中的
# `empty_or_failed_compaction_keeps_source_and_is_reported` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   empty：`empty`输入或配置值。
#   error：异常或错误信息。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `_Summary` →
# `_prepare` → `_manager`。
# 分支与异常：
#   验证条件：`decision.messages == source`。
#   验证条件：`decision.summary_updated is False`。
#   验证条件：`decision.summary_state is None`。
#   验证条件：`len(summarizer.sources) == 2`。
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty,error", [(True, None), (False, ValueError("summary failed"))]
)
async def test_empty_or_failed_compaction_keeps_source_and_is_reported(
    empty, error
) -> None:
    source = (
        Message(role=MessageRole.USER, content="go"),
        *sum((_round(i) for i in range(4)), ()),
    )
    summarizer = _Summary(empty=empty, error=error)
    decision = await _prepare(
        _manager(summarizer), source, source, protected_source_indices=(0,)
    )
    assert decision.messages == source
    assert decision.summary_updated is False
    assert decision.summary_state is None
    assert len(summarizer.sources) == 2
    assert decision.summary_error


# 函数说明：test_failed_source_retry_can_be_suppressed_without_a_global_cache
# 用途：回归验证回归测试与测试辅助中的
# `failed_source_retry_can_be_suppressed_without_a_global_cache` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `_Summary` →
# `_manager` → `_prepare`。
# 分支与异常：
#   验证条件：`len(summarizer.sources) == 2`。
#   验证条件：`second.messages == source`。
#   验证条件：`'suppressed' in second.summary_error`。
@pytest.mark.asyncio
async def test_failed_source_retry_can_be_suppressed_without_a_global_cache() -> None:
    source = (
        Message(role=MessageRole.USER, content="go"),
        *sum((_round(i) for i in range(4)), ()),
    )
    summarizer = _Summary(error=ValueError("fail"))
    manager = _manager(summarizer)
    first = await _prepare(manager, source, source, protected_source_indices=(0,))
    second = await _prepare(
        manager,
        first.messages,
        source,
        protected_source_indices=(0,),
        allow_compaction=False,
    )
    assert len(summarizer.sources) == 2
    assert second.messages == source
    assert "suppressed" in second.summary_error


# 函数说明：test_no_summarizer_does_not_delete_old_tools_to_fake_a_fit
# 用途：回归验证回归测试与测试辅助中的
# `no_summarizer_does_not_delete_old_tools_to_fake_a_fit` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` → `_prepare` →
# `_manager`。
# 分支与异常：
#   验证条件：`decision.requires_compaction and (not decision.reached_target)`。
#   验证条件：`decision.messages == source`。
#   验证条件：`decision.summary_error`。
@pytest.mark.asyncio
async def test_no_summarizer_does_not_delete_old_tools_to_fake_a_fit() -> None:
    source = (
        Message(role=MessageRole.USER, content="go"),
        *sum((_round(i) for i in range(4)), ()),
    )
    decision = await _prepare(_manager(), source, source, protected_source_indices=(0,))
    assert decision.requires_compaction and not decision.reached_target
    assert decision.messages == source
    assert decision.summary_error


# 函数说明：test_remeasure_includes_final_closing_payload_without_second_summary
# 用途：回归验证回归测试与测试辅助中的
# `remeasure_includes_final_closing_payload_without_second_summary` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_Summary` → `_manager` →
# `_prepare` → `ToolDefinition` → `manager.remeasure`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`final.exceeds_input_budget`。
#   验证条件：
# `final.tool_schema_tokens == sum((len(t.name) + len(t.description) for t in tools))`。
#   验证条件：`final.prepared_input_tokens == manager.estimator.estimate_request(final.
# messages, tools=tools)`。
#   验证条件：`final.summary_usage == first.summary_usage`。
@pytest.mark.asyncio
async def test_remeasure_includes_final_closing_payload_without_second_summary() -> (
    None
):
    source = (Message(role=MessageRole.USER, content="hi"),)
    summarizer = _Summary()
    manager = _manager(summarizer, window=500, ceiling=300)
    first = await _prepare(manager, source, source)
    closing = Message(role=MessageRole.SYSTEM, content="收尾" * 300)
    tools = (ToolDefinition(name="closing", description="schema" * 40),)
    final = manager.remeasure(
        first,
        messages=(*first.messages, closing),
        tools=tools,
        model="test-model",
        provider="test",
        max_output_tokens=100,
    )
    assert final.exceeds_input_budget
    assert final.tool_schema_tokens == sum(
        len(t.name) + len(t.description) for t in tools
    )
    assert final.prepared_input_tokens == manager.estimator.estimate_request(
        final.messages, tools=tools
    )
    assert final.summary_usage == first.summary_usage
    assert summarizer.sources == []


# 函数说明：test_tool_view_limits_are_configured_and_copied
# 用途：回归验证回归测试与测试辅助中的 `tool_view_limits_are_configured_and_copied` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager`。
# 分支与异常：
#   验证条件：
# `limits == {'max_output_chars': 8000, 'head_chars': 4000, 'tail_chars': 2000}`。
#   验证条件：`manager.tool_view_limits['max_output_chars'] == 8000`。
def test_tool_view_limits_are_configured_and_copied() -> None:
    manager = _manager()
    limits = manager.tool_view_limits
    assert limits == {
        "max_output_chars": 8_000,
        "head_chars": 4_000,
        "tail_chars": 2_000,
    }
    limits["max_output_chars"] = 1
    assert manager.tool_view_limits["max_output_chars"] == 8_000


# 函数说明：test_summary_tool_receipt_bounds_output_without_cutting_evidence_or_errors
# 用途：回归验证回归测试与测试辅助中的
# `summary_tool_receipt_bounds_output_without_cutting_evidence_or_errors` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `json.dumps` →
# `json.loads` → `_history_item`。
# 分支与异常：
#   验证条件：`len(payload['output']) < 2100`。
#   验证条件：`payload['success'] is False`。
#   验证条件：`payload['error'] == '失败原因不能被裁掉'`。
#   验证条件：`payload['evidence_id'] == 'complete-evidence-id'`。
def test_summary_tool_receipt_bounds_output_without_cutting_evidence_or_errors() -> (
    None
):
    message = Message(
        role=MessageRole.TOOL,
        tool_call_id="call",
        content=json.dumps(
            {
                "output": "x" * 20_000,
                "success": False,
                "error": "失败原因不能被裁掉",
                "evidence_id": "complete-evidence-id",
                "output_sha256": "a" * 64,
            }
        ),
    )
    payload = json.loads(_history_item(message)["content"])
    assert len(payload["output"]) < 2_100
    assert payload["success"] is False
    assert payload["error"] == "失败原因不能被裁掉"
    assert payload["evidence_id"] == "complete-evidence-id"
    assert payload["output_sha256"] == "a" * 64
    assert len(json.loads(message.content)["output"]) == 20_000


# 函数说明：test_invalid_watermark_does_not_split_protocol
# 用途：回归验证回归测试与测试辅助中的 `invalid_watermark_does_not_split_protocol` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `_round` →
# `ConversationSummaryState` → `RollingConversationSummary` → `_prepare` → `_manager`。
# 分支与异常：
#   验证条件：`decision.summary_state is None`。
#   验证条件：`decision.messages == source`。
@pytest.mark.asyncio
async def test_invalid_watermark_does_not_split_protocol() -> None:
    source = (Message(role=MessageRole.USER, content="go"), *_round(0))
    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="旧摘要"),
        covered_message_count=2,
    )
    decision = await _prepare(_manager(), source, source, summary_state=state)
    assert decision.summary_state is None
    assert decision.messages == source
