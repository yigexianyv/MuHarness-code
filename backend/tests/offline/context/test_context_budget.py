from __future__ import annotations

import pytest

from app.models.types import Message, MessageRole, ToolCall
from app.runtime.context import (
    CapabilitySource,
    ContextBudgetPolicy,
    ContextManager,
    ContextSettings,
    ModelCapabilities,
    build_budget_policy,
    build_model_capability_registry,
)


# 函数说明：_caps
# 用途：返回 `ModelCapabilities(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   context_window：上下文输入或配置值，类型 `int`；默认 `131072`。
#   max_output：输出输入或配置值，类型 `int`；默认 `8192`。
# 返回：类型 `ModelCapabilities`；返回 `ModelCapabilities(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelCapabilities`。
def _caps(
    context_window: int = 131_072,
    max_output: int = 8_192,
) -> ModelCapabilities:
    return ModelCapabilities(
        provider="qwen",
        model="qwen3.7-plus",
        context_window=context_window,
        max_output_tokens=max_output,
        source=CapabilitySource.BUILTIN,
    )


# 函数说明：test_budget_formula
# 用途：回归验证回归测试与测试辅助中的 `budget_formula` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudgetPolicy` →
# `policy.compute` → `_caps`。
# 分支与异常：
#   验证条件：`budget.context_window == 131072`。
#   验证条件：`budget.reserved_output_tokens == 4000`。
#   验证条件：`budget.safety_margin_tokens == 4096`。
#   验证条件：`budget.input_budget == 131072 - 4000 - 4096`。
def test_budget_formula() -> None:
    policy = ContextBudgetPolicy(safety_margin_tokens=4_096)

    budget = policy.compute(_caps(), max_output_tokens=4_000)

    assert budget.context_window == 131_072
    assert budget.reserved_output_tokens == 4_000
    assert budget.safety_margin_tokens == 4_096
    assert budget.input_budget == 131_072 - 4_000 - 4_096
    assert budget.working_input_budget == 64_000
    assert budget.hard_trigger_tokens == int(budget.input_budget * 0.8)
    assert budget.hard_target_tokens == int(budget.input_budget * 0.6)
    assert budget.trigger_tokens == int(64_000 * 0.8)
    assert budget.target_tokens == int(64_000 * 0.45)
    assert budget.tool_result_budget_tokens == int(budget.target_tokens * 0.35)


# 函数说明：test_model_default_max_output_used_when_no_override
# 用途：回归验证回归测试与测试辅助中的 `model_default_max_output_used_when_no_override`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudgetPolicy` →
# `policy.compute` → `_caps`。
# 分支与异常：
#   验证条件：`budget.reserved_output_tokens == 8192`。
#   验证条件：`budget.input_budget == 131072 - 8192`。
def test_model_default_max_output_used_when_no_override() -> None:
    policy = ContextBudgetPolicy(safety_margin_tokens=0)

    budget = policy.compute(_caps(max_output=8_192))

    assert budget.reserved_output_tokens == 8_192
    assert budget.input_budget == 131_072 - 8_192


# 函数说明：test_explicit_max_output_preferred
# 用途：回归验证回归测试与测试辅助中的 `explicit_max_output_preferred` 场景，下方断言说
# 明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudgetPolicy` →
# `policy.compute` → `_caps`。
# 分支与异常：
#   验证条件：`budget.reserved_output_tokens == 2000`。
#   验证条件：`budget.input_budget == 131072 - 2000`。
def test_explicit_max_output_preferred() -> None:
    policy = ContextBudgetPolicy(safety_margin_tokens=0)

    budget = policy.compute(_caps(max_output=8_192), max_output_tokens=2_000)

    assert budget.reserved_output_tokens == 2_000
    assert budget.input_budget == 131_072 - 2_000


# 函数说明：test_negative_budget_raises_clear_error
# 用途：回归验证回归测试与测试辅助中的 `negative_budget_raises_clear_error` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudgetPolicy` →
# `pytest.raises` → `policy.compute` → `_caps`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='input_budget')`。
def test_negative_budget_raises_clear_error() -> None:
    policy = ContextBudgetPolicy(safety_margin_tokens=0)

    with pytest.raises(ValueError, match="input_budget"):
        policy.compute(_caps(context_window=10, max_output=50))


# 函数说明：test_requested_output_cannot_exceed_model_maximum
# 用途：回归验证回归测试与测试辅助中的 `requested_output_cannot_exceed_model_maximum` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudgetPolicy` →
# `pytest.raises` → `policy.compute` → `_caps`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='model maximum')`。
def test_requested_output_cannot_exceed_model_maximum() -> None:
    policy = ContextBudgetPolicy(safety_margin_tokens=0)

    with pytest.raises(ValueError, match="model maximum"):
        policy.compute(_caps(max_output=1_000), max_output_tokens=1_001)


# 函数说明：test_invalid_ratio_config_raises
# 用途：回归验证回归测试与测试辅助中的 `invalid_ratio_config_raises` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `ContextBudgetPolicy`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='trigger_ratio')`。
#   预期异常：`pytest.raises(ValueError, match='target_ratio')`。
def test_invalid_ratio_config_raises() -> None:
    with pytest.raises(ValueError, match="trigger_ratio"):
        ContextBudgetPolicy(trigger_ratio=1.5)
    with pytest.raises(ValueError, match="target_ratio"):
        ContextBudgetPolicy(target_ratio=0.9)
    with pytest.raises(ValueError, match="target_ratio"):
        ContextBudgetPolicy(trigger_ratio=0.6, target_ratio=0.7)


# 函数说明：test_without_summary_tool_protocol_is_never_deleted
# 用途：回归验证回归测试与测试辅助中的 `without_summary_tool_protocol_is_never_deleted`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `Message` →
# `ContextManager().prepare` → `ContextManager`。
# 分支与异常：
#   验证条件：`decision.messages == messages`。
#   验证条件：`decision.summary_updated is False`。
#   验证条件：`decision.summary_error is not None`。
@pytest.mark.asyncio
async def test_without_summary_tool_protocol_is_never_deleted() -> None:
    call = ToolCall(id="search-1", name="web_search", arguments={"query": "AI"})
    messages = (
        Message(role=MessageRole.SYSTEM, content="系统提示"),
        Message(role=MessageRole.USER, content="搜索新闻"),
        Message(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        Message(
            role=MessageRole.TOOL,
            tool_call_id=call.id,
            name=call.name,
            content="很长的搜索结果",
        ),
        Message(role=MessageRole.ASSISTANT, content="新闻摘要"),
    )

    decision = await ContextManager().prepare(messages, compaction_target_tokens=1)
    assert decision.messages == messages
    assert decision.summary_updated is False
    assert decision.summary_error is not None


# 函数说明：test_context_manager_below_trigger_preserves_complete_protocol
# 用途：回归验证回归测试与测试辅助中的
# `context_manager_below_trigger_preserves_complete_protocol` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextManager` → `ToolCall` →
# `Message` → `manager.prepare`。
# 分支与异常：
#   验证条件：`decision.messages == (*history, *current_run)`。
#   验证条件：`decision.trimmed is False`。
#   验证条件：`decision.requires_compaction is False`。
#   验证条件：`decision.compacted_tool_results == 0`。
@pytest.mark.asyncio
async def test_context_manager_below_trigger_preserves_complete_protocol() -> None:
    manager = ContextManager(keep_recent_tool_rounds=0)
    old_call = ToolCall(
        id="old-search",
        name="web_search",
        arguments={"query": "旧查询"},
    )
    current_call = ToolCall(
        id="current-search",
        name="web_search",
        arguments={"query": "新查询"},
    )
    history = (
        Message(role=MessageRole.USER, content="上一轮"),
        Message(role=MessageRole.ASSISTANT, tool_calls=(old_call,)),
        Message(
            role=MessageRole.TOOL,
            tool_call_id=old_call.id,
            name=old_call.name,
            content="旧工具输出",
        ),
        Message(role=MessageRole.ASSISTANT, content="上一轮答案"),
    )
    current_run = (
        Message(role=MessageRole.USER, content="这一轮"),
        Message(role=MessageRole.ASSISTANT, tool_calls=(current_call,)),
        Message(
            role=MessageRole.TOOL,
            tool_call_id=current_call.id,
            name=current_call.name,
            content="当前工具输出",
        ),
    )

    decision = await manager.prepare(
        (*history, *current_run),
        history_count=len(history),
        model="qwen3.7-plus",
        provider="qwen",
    )

    assert decision.messages == (*history, *current_run)
    assert decision.trimmed is False
    assert decision.requires_compaction is False
    assert decision.compacted_tool_results == 0
    assert decision.removed_tool_rounds == 0
    assert history[1].tool_calls == (old_call,)
    assert history[2].content == "旧工具输出"


# 函数说明：test_short_request_stays_below_working_trigger
# 用途：回归验证回归测试与测试辅助中的 `short_request_stays_below_working_trigger` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextManager` →
# `build_model_capability_registry` → `ContextSettings` → `build_budget_policy` →
# `manager.prepare` → `Message`。
# 分支与异常：
#   验证条件：`decision.requires_compaction is False`。
#   验证条件：`decision.usage_ratio is not None`。
#   验证条件：`decision.usage_ratio < 0.8`。
@pytest.mark.asyncio
async def test_short_request_stays_below_working_trigger() -> None:
    manager = ContextManager(
        registry=build_model_capability_registry(
            context_settings=ContextSettings(_env_file=None),
        ),
        budget_policy=build_budget_policy(ContextSettings(_env_file=None)),
    )

    decision = await manager.prepare(
        (Message(role=MessageRole.USER, content="hi"),),
        model="qwen3.7-plus",
        provider="qwen",
    )

    assert decision.requires_compaction is False
    assert decision.usage_ratio is not None
    assert decision.usage_ratio < 0.8


# 函数说明：test_small_window_caps_working_budget_at_physical_input_budget
# 用途：回归验证回归测试与测试辅助中的
# `small_window_caps_working_budget_at_physical_input_budget` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_model_capability_registry` →
# `ContextSettings` → `registry.register_override` → `ContextManager` →
# `ContextBudgetPolicy` → `manager.prepare`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`decision.requires_compaction is True`。
#   验证条件：`decision.context_window == 200`。
#   验证条件：`decision.input_budget == 150`。
#   验证条件：`decision.working_input_budget == 150`。
@pytest.mark.asyncio
async def test_small_window_caps_working_budget_at_physical_input_budget() -> None:
    registry = build_model_capability_registry(
        context_settings=ContextSettings(_env_file=None),
    )
    registry.register_override(
        "qwen",
        "qwen3.7-plus",
        context_window=200,
        max_output_tokens=50,
    )
    manager = ContextManager(
        registry=registry,
        budget_policy=ContextBudgetPolicy(safety_margin_tokens=0),
    )

    decision = await manager.prepare(
        (Message(role=MessageRole.USER, content="x" * 2000),),
        model="qwen3.7-plus",
        provider="qwen",
    )

    assert decision.requires_compaction is True
    assert decision.context_window == 200
    assert decision.input_budget == 150
    assert decision.working_input_budget == 150
    assert decision.hard_trigger_tokens == 120
    assert decision.trigger_tokens == int(150 * 0.8)
    assert decision.target_tokens == int(150 * 0.45)
    assert decision.capability_source == CapabilitySource.OVERRIDE.value
