from __future__ import annotations

import pytest

from app.models.config import ModelSettings
from app.models.types import Message, MessageRole, ModelProvider
from app.runtime.context import (
    CapabilitySource,
    ContextManager,
    ContextSettings,
    build_budget_policy,
    build_model_capability_registry,
)


# 函数说明：_settings
# 用途：返回 `ContextSettings(_env_file=None, **overrides)`，提供 回归测试与测试辅助 的
# 派生值。
# 参数：
#   **overrides：额外关键字参数，按实现处理或转交。
# 返回：类型 `ContextSettings`；返回 `ContextSettings(_env_file=None, **overrides)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextSettings`。
def _settings(**overrides: object) -> ContextSettings:
    return ContextSettings(_env_file=None, **overrides)


# 函数说明：_qwen_model_settings
# 用途：返回 `ModelSettings(…)`，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `ModelSettings`；返回 `ModelSettings(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettings`。
def _qwen_model_settings() -> ModelSettings:
    return ModelSettings(
        _env_file=None,
        DASHSCOPE_API_KEY="qwen-key",
        model_default_provider=ModelProvider.QWEN,
    )


# 函数说明：test_context_settings_defaults
# 用途：回归验证回归测试与测试辅助中的 `context_settings_defaults` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_settings`。
# 分支与异常：
#   验证条件：`settings.context_window_qwen == 1000000`。
#   验证条件：`settings.context_trigger_ratio == 0.8`。
#   验证条件：`settings.context_target_ratio == 0.6`。
#   验证条件：`settings.context_preferred_input_tokens == 64000`。
def test_context_settings_defaults() -> None:
    settings = _settings()

    assert settings.context_window_qwen == 1_000_000
    assert settings.context_trigger_ratio == 0.80
    assert settings.context_target_ratio == 0.60
    assert settings.context_preferred_input_tokens == 64_000
    assert settings.context_working_trigger_ratio == 0.80
    assert settings.context_working_target_ratio == 0.45
    assert settings.context_tool_result_budget_ratio == 0.35
    assert settings.context_safety_margin_tokens == 4_096
    assert settings.context_keep_recent_tool_rounds == 2
    assert settings.context_keep_recent_conversation_blocks == 4
    assert settings.context_summary_max_output_tokens == 1_024
    assert settings.context_max_tool_result_chars == 8_000
    assert settings.context_tool_result_head_chars == 4_000
    assert settings.context_tool_result_tail_chars == 2_000


# 函数说明：test_retired_block_trigger_config_is_ignored
# 用途：回归验证回归测试与测试辅助中的 `retired_block_trigger_config_is_ignored` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_settings`。
# 分支与异常：
#   验证条件：
# `'context_max_unsummarized_conversation_blocks' not in settings.model_dump()`。
def test_retired_block_trigger_config_is_ignored() -> None:
    settings = _settings(context_max_unsummarized_conversation_blocks=1)
    assert "context_max_unsummarized_conversation_blocks" not in settings.model_dump()


# 函数说明：test_tool_result_segment_config_must_fit_maximum
# 用途：回归验证回归测试与测试辅助中的 `tool_result_segment_config_must_fit_maximum` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `_settings`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='head/tail')`。
def test_tool_result_segment_config_must_fit_maximum() -> None:
    with pytest.raises(ValueError, match="head/tail"):
        _settings(
            context_max_tool_result_chars=100,
            context_tool_result_head_chars=80,
            context_tool_result_tail_chars=30,
        )


# 函数说明：test_builtin_lookup_uses_exact_model
# 用途：回归验证回归测试与测试辅助中的 `builtin_lookup_uses_exact_model` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_model_capability_registry` →
# `_qwen_model_settings` → `_settings` → `registry.lookup`。
# 分支与异常：
#   验证条件：`cap.context_window == 1000000`。
#   验证条件：`cap.source is CapabilitySource.BUILTIN`。
def test_builtin_lookup_uses_exact_model() -> None:
    registry = build_model_capability_registry(
        model_settings=_qwen_model_settings(),
        context_settings=_settings(),
    )

    cap = registry.lookup("qwen", "qwen3.7-plus")

    assert cap.context_window == 1_000_000
    assert cap.source is CapabilitySource.BUILTIN


# 函数说明：test_provider_default_window_reads_settings
# 用途：回归验证回归测试与测试辅助中的 `provider_default_window_reads_settings` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_model_capability_registry` →
# `_qwen_model_settings` → `_settings` → `registry.lookup`。
# 分支与异常：
#   验证条件：`cap.context_window == 200000`。
#   验证条件：`cap.source is CapabilitySource.PROVIDER_DEFAULT`。
def test_provider_default_window_reads_settings() -> None:
    registry = build_model_capability_registry(
        model_settings=_qwen_model_settings(),
        context_settings=_settings(context_window_qwen=200_000),
    )

    cap = registry.lookup("qwen", "some-other-qwen-model")

    assert cap.context_window == 200_000
    assert cap.source is CapabilitySource.PROVIDER_DEFAULT


# 函数说明：test_override_from_settings_applies_to_current_model
# 用途：回归验证回归测试与测试辅助中的 `override_from_settings_applies_to_current_model`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_model_capability_registry` →
# `_qwen_model_settings` → `_settings` → `registry.lookup`。
# 分支与异常：
#   验证条件：`cap.context_window == 99999`。
#   验证条件：`cap.max_output_tokens == 5000`。
#   验证条件：`cap.source is CapabilitySource.OVERRIDE`。
def test_override_from_settings_applies_to_current_model() -> None:
    registry = build_model_capability_registry(
        model_settings=_qwen_model_settings(),
        context_settings=_settings(
            context_window_override=99_999,
            max_output_tokens_override=5_000,
        ),
    )

    cap = registry.lookup("qwen", "qwen3.7-plus")

    assert cap.context_window == 99_999
    assert cap.max_output_tokens == 5_000
    assert cap.source is CapabilitySource.OVERRIDE


# 函数说明：test_override_with_explicit_provider_model
# 用途：回归验证回归测试与测试辅助中的 `override_with_explicit_provider_model` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_model_capability_registry` →
# `_qwen_model_settings` → `_settings` → `registry.lookup`。
# 分支与异常：
#   验证条件：`cap.context_window == 66000`。
#   验证条件：`cap.source is CapabilitySource.OVERRIDE`。
def test_override_with_explicit_provider_model() -> None:
    registry = build_model_capability_registry(
        model_settings=_qwen_model_settings(),
        context_settings=_settings(
            context_override_provider="qwen",
            context_override_model="qwen3.7-plus",
            context_window_override=66_000,
        ),
    )

    cap = registry.lookup("qwen", "qwen3.7-plus")

    assert cap.context_window == 66_000
    assert cap.source is CapabilitySource.OVERRIDE


# 函数说明：test_explicit_override_does_not_require_provider_api_key
# 用途：回归验证回归测试与测试辅助中的
# `explicit_override_does_not_require_provider_api_key` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_model_capability_registry` →
# `ModelSettings` → `_settings` → `registry.lookup`。
# 分支与异常：
#   验证条件：`cap.context_window == 55555`。
#   验证条件：`cap.source is CapabilitySource.OVERRIDE`。
def test_explicit_override_does_not_require_provider_api_key() -> None:
    registry = build_model_capability_registry(
        model_settings=ModelSettings(_env_file=None),
        context_settings=_settings(
            context_override_provider="qwen",
            context_override_model="qwen3.7-plus",
            context_window_override=55_555,
        ),
    )

    cap = registry.lookup("qwen", "qwen3.7-plus")

    assert cap.context_window == 55_555
    assert cap.source is CapabilitySource.OVERRIDE


# 函数说明：test_zero_max_output_override_is_rejected
# 用途：回归验证回归测试与测试辅助中的 `zero_max_output_override_is_rejected` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `_settings`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='greater than 0')`。
def test_zero_max_output_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        _settings(max_output_tokens_override=0)


# 函数说明：test_context_manager_decision_fields
# 用途：回归验证回归测试与测试辅助中的 `context_manager_decision_fields` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_settings` →
# `build_model_capability_registry` → `_qwen_model_settings` → `ContextManager` →
# `build_budget_policy` → `Message`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`decision.provider == 'qwen'`。
#   验证条件：`decision.model == 'qwen3.7-plus'`。
#   验证条件：`decision.context_window == 1000000`。
#   验证条件：`decision.input_budget == 1000000 - 65536 - 4096`。
@pytest.mark.asyncio
async def test_context_manager_decision_fields() -> None:
    settings = _settings()
    registry = build_model_capability_registry(
        model_settings=_qwen_model_settings(),
        context_settings=settings,
    )
    manager = ContextManager(
        registry=registry,
        budget_policy=build_budget_policy(settings),
    )
    messages = (Message(role=MessageRole.USER, content="hello"),)

    decision = await manager.prepare(
        messages,
        model="qwen3.7-plus",
        provider="qwen",
    )

    assert decision.provider == "qwen"
    assert decision.model == "qwen3.7-plus"
    assert decision.context_window == 1_000_000
    assert decision.input_budget == 1_000_000 - 65_536 - 4_096
    assert decision.working_input_budget == 64_000
    assert decision.hard_trigger_tokens == int(decision.input_budget * 0.8)
    assert decision.hard_target_tokens == int(decision.input_budget * 0.6)
    assert decision.trigger_tokens == decision.compact_ceiling_tokens == 128_000
    assert decision.target_tokens == decision.forced_target_tokens == 64_000
    assert decision.tool_result_budget_tokens == int(64_000 * 0.45 * 0.35)
    assert decision.usage_ratio is not None
    assert decision.requires_compaction is False
    assert decision.capability_source == CapabilitySource.BUILTIN.value
    assert decision.trimmed is False
    assert decision.messages == messages
    assert "original_estimated=" in (decision.reason or "")
