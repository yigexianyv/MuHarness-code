
from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import SecretStr

from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.result import AgentStopReason
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.context import (
    CapabilitySource,
    ContextBudgetPolicy,
    ContextDecision,
    ContextManager,
    ContextSettings,
    ModelCapabilities,
    build_budget_policy,
    build_model_capability_registry,
)
from app.tools import ToolRegistry
from app.tools.base import BaseTool


# 函数说明：_settings
# 用途：返回 `ContextSettings(_env_file=None, **overrides)`，提供 回归测试与测试辅助 的
# 派生值。
# 参数：
#   **overrides：额外关键字参数，按实现处理或转交。
# 返回：类型 `ContextSettings`；返回 `ContextSettings(_env_file=None, **overrides)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextSettings`。
def _settings(**overrides: object) -> ContextSettings:
    return ContextSettings(_env_file=None, **overrides)


# 函数说明：_default_registry
# 用途：返回 `build_model_capability_registry(…)`，提供 回归测试与测试辅助 的派生值。
# 返回：返回 `build_model_capability_registry(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_model_capability_registry` →
# `ModelSettings` → `_settings`。
def _default_registry():
    return build_model_capability_registry(
        model_settings=ModelSettings(_env_file=None),
        context_settings=_settings(),
    )


# 函数说明：test_lookup_known_provider_and_model
# 用途：回归验证回归测试与测试辅助中的 `lookup_known_provider_and_model` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_default_registry().lookup` →
# `_default_registry`。
# 分支与异常：
#   验证条件：`cap.provider == 'qwen'`。
#   验证条件：`cap.model == 'qwen3.7-plus'`。
#   验证条件：`cap.context_window == 1000000`。
#   验证条件：`cap.source is CapabilitySource.BUILTIN`。
def test_lookup_known_provider_and_model() -> None:
    cap = _default_registry().lookup("qwen", "qwen3.7-plus")

    assert cap.provider == "qwen"
    assert cap.model == "qwen3.7-plus"
    assert cap.context_window == 1_000_000
    assert cap.source is CapabilitySource.BUILTIN


# 函数说明：test_deepseek_builtin_matches_current_official_capacity
# 用途：回归验证回归测试与测试辅助中的
# `deepseek_builtin_matches_current_official_capacity` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_default_registry().lookup` →
# `_default_registry`。
# 分支与异常：
#   验证条件：`cap.context_window == 1048576`。
#   验证条件：`cap.max_output_tokens == 393216`。
def test_deepseek_builtin_matches_current_official_capacity() -> None:
    cap = _default_registry().lookup("deepseek", "deepseek-v4-flash")

    assert cap.context_window == 1_048_576
    assert cap.max_output_tokens == 393_216


# 函数说明：test_model_capabilities_reject_invalid_limits
# 用途：回归验证回归测试与测试辅助中的 `model_capabilities_reject_invalid_limits` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `ModelCapabilities`
# 。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='context_window')`。
def test_model_capabilities_reject_invalid_limits() -> None:
    with pytest.raises(ValueError, match="context_window"):
        ModelCapabilities(
            provider="qwen",
            model="bad",
            context_window=0,
            max_output_tokens=1,
            source=CapabilitySource.OVERRIDE,
        )


# 函数说明：test_same_provider_different_models_have_different_windows
# 用途：回归验证回归测试与测试辅助中的
# `same_provider_different_models_have_different_windows` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_default_registry` →
# `registry.lookup`。
# 分支与异常：
#   验证条件：`mini.provider == mini4o.provider == 'openai'`。
#   验证条件：`mini.context_window != mini4o.context_window`。
def test_same_provider_different_models_have_different_windows() -> None:
    registry = _default_registry()

    mini = registry.lookup("openai", "gpt-5.4-mini")
    mini4o = registry.lookup("openai", "gpt-4o-mini")

    assert mini.provider == mini4o.provider == "openai"
    assert mini.context_window != mini4o.context_window


# 函数说明：test_user_override_priority
# 用途：回归验证回归测试与测试辅助中的 `user_override_priority` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_default_registry` →
# `registry.register_override` → `registry.lookup`。
# 分支与异常：
#   验证条件：`cap.context_window == 50000`。
#   验证条件：`cap.max_output_tokens == 2000`。
#   验证条件：`cap.source is CapabilitySource.OVERRIDE`。
def test_user_override_priority() -> None:
    registry = _default_registry()
    registry.register_override(
        "qwen",
        "qwen3.7-plus",
        context_window=50_000,
        max_output_tokens=2_000,
    )

    cap = registry.lookup("qwen", "qwen3.7-plus")

    assert cap.context_window == 50_000
    assert cap.max_output_tokens == 2_000
    assert cap.source is CapabilitySource.OVERRIDE


# 函数说明：test_unknown_model_uses_conservative_fallback
# 用途：回归验证回归测试与测试辅助中的 `unknown_model_uses_conservative_fallback` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   caplog：`caplog`输入或配置值。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_default_registry().lookup` →
# `_default_registry`。
# 分支与异常：
#   验证条件：`cap.context_window == 32768`。
#   验证条件：`cap.max_output_tokens == 4096`。
#   验证条件：`cap.source is CapabilitySource.FALLBACK`。
#   验证条件：`'fallback' in caplog.text`。
def test_unknown_model_uses_conservative_fallback(caplog) -> None:
    cap = _default_registry().lookup("mystery", "unknown-model")

    assert cap.context_window == 32_768
    assert cap.max_output_tokens == 4_096
    assert cap.source is CapabilitySource.FALLBACK
    assert "fallback" in caplog.text




class _ScriptedAdapter(ModelAdapter):
    # 函数说明：_ScriptedAdapter.__init__
    # 用途：初始化 _ScriptedAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`
    # 。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.requests`。
    def __init__(
        self,
        config: ProviderConfig,
        responses: Sequence[ModelResponse | Exception],
    ) -> None:
        super().__init__(config)
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    # 函数说明：_ScriptedAdapter.complete
    # 用途：完成_ScriptedAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.responses.pop`。
    # 分支与异常：
    #   当 `isinstance(response, Exception)` 时，抛出 `response`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    # 函数说明：_ScriptedAdapter.close
    # 用途：关闭_ScriptedAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class _NoopTool(BaseTool):
    definition = ToolDefinition(name="noop", description="noop")

    # 函数说明：_NoopTool.execute
    # 用途：执行_NoopTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
    # 返回：类型 `str`；返回 `'ok'`。
    async def execute(self, arguments: dict) -> str:
        return "ok"


class _FailingContextManager(ContextManager):

    # 函数说明：_FailingContextManager.prepare
    # 用途：准备_FailingContextManager，供回归测试与测试辅助使用。
    # 参数：
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   tools：可用工具定义或工具实例集合，类型 `Sequence[ToolDefinition]`；默认 `()`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    #   history_count：原始历史消息数量，类型 `int | None`；默认 `None`。
    #   summary_state：会话摘要及覆盖水位；默认 `None`。
    # 返回：类型 `ContextDecision`；不返回结果值（隐式 None）。
    async def prepare(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition] = (),
        model: str | None = None,
        provider: str | None = None,
        max_output_tokens: int | None = None,
        history_count: int | None = None,
        summary_state=None,
    ) -> ContextDecision:
        raise RuntimeError("context estimator unavailable")


# 函数说明：_qwen_adapter
# 用途：处理回归测试与测试辅助中的 `_qwen_adapter` 数据；结果及边界条件见下方说明。
# 参数：
#   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`。
# 返回：类型 `_ScriptedAdapter`；返回 `_ScriptedAdapter(config, responses)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `_ScriptedAdapter`。
def _qwen_adapter(responses: Sequence[ModelResponse | Exception]) -> _ScriptedAdapter:
    config = ProviderConfig(
        provider="qwen",
        model="qwen3.7-plus",
        api_key=SecretStr("test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    return _ScriptedAdapter(config, responses)


# 函数说明：_final_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str`；默认 `'完成'`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message`。
def _final_response(content: str = "完成") -> ModelResponse:
    return ModelResponse(
        id="r",
        provider="qwen",
        model="qwen3.7-plus",
        message=Message(role=MessageRole.ASSISTANT, content=content),
    )


# 函数说明：_make_runtime
# 用途：构造执行环境，供回归测试与测试辅助使用。
# 参数：
#   adapter：`adapter`输入或配置值，类型 `_ScriptedAdapter`。
#   model：模型名称，类型 `str | None`；默认 `None`。
#   max_tool_rounds：工具调用轮次上限，类型 `int | None`；默认 `None`。
#   context_manager：上下文预算与压缩管理器，类型 `ContextManager | None`；默认 `None`。
# 返回：类型 `tuple[AgentRuntime, _ScriptedAdapter]`；返回 `(runtime, adapter)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelAdapterRegistry` →
# `ModelSettings` → `registry.register` → `ToolRegistry` → `tools.register` →
# `_NoopTool`；另有 5 个调用点。
def _make_runtime(
    adapter: _ScriptedAdapter,
    *,
    model: str | None = None,
    max_tool_rounds: int | None = None,
    context_manager: ContextManager | None = None,
) -> tuple[AgentRuntime, _ScriptedAdapter]:
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register(
        "qwen",
        lambda _: adapter,
        config=adapter.config,
        replace=True,
    )
    tools = ToolRegistry()
    tools.register(_NoopTool())
    runtime = AgentRuntime(
        registry,
        tools,
        provider="qwen",
        model=model,
        max_tool_rounds=max_tool_rounds,
        context_manager=context_manager
        or ContextManager(
            registry=_default_registry(),
            budget_policy=build_budget_policy(_settings()),
        ),
    )
    return runtime, adapter


# 函数说明：test_runtime_resolves_model_from_adapter_when_self_model_none
# 用途：回归验证回归测试与测试辅助中的
# `runtime_resolves_model_from_adapter_when_self_model_none` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_runtime` → `_qwen_adapter` →
# `_final_response` → `InMemoryEventHandler` → `runtime.run` → `next`。
# 分支与异常：
#   验证条件：`started.model == 'qwen3.7-plus'`。
#   验证条件：`started.provider == 'qwen'`。
#   验证条件：`adapter.requests[0].model == 'qwen3.7-plus'`。
#   验证条件：`adapter.requests[0].model == started.model`。
@pytest.mark.asyncio
async def test_runtime_resolves_model_from_adapter_when_self_model_none() -> None:
    runtime, adapter = _make_runtime(_qwen_adapter([_final_response()]))
    handler = InMemoryEventHandler()

    await runtime.run("hi", event_handler=handler)

    started = next(
        event for event in handler.events if event.type is AgentEventType.MODEL_STARTED
    )
    assert started.model == "qwen3.7-plus"
    assert started.provider == "qwen"
    assert adapter.requests[0].model == "qwen3.7-plus"
    assert adapter.requests[0].model == started.model
    assert (
        adapter.requests[0].max_output_tokens
        == adapter.config.default_max_output_tokens
    )
    assert started.input_budget == 1_000_000 - 4_096 - 4_096


# 函数说明：test_context_manager_does_not_modify_messages
# 用途：回归验证回归测试与测试辅助中的 `context_manager_does_not_modify_messages` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextManager` → `_default_registry`
#  → `build_budget_policy` → `_settings` → `Message` → `manager.prepare`。
# 分支与异常：
#   验证条件：`decision.messages == original`。
#   验证条件：`decision.trimmed is False`。
#   验证条件：`decision.requires_compaction is False`。
@pytest.mark.asyncio
async def test_context_manager_does_not_modify_messages() -> None:
    manager = ContextManager(
        registry=_default_registry(),
        budget_policy=build_budget_policy(_settings()),
    )
    original = (
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.ASSISTANT, content="yo"),
    )

    decision = await manager.prepare(
        original,
        model="qwen3.7-plus",
        provider="qwen",
    )

    assert decision.messages == original
    assert decision.trimmed is False
    assert decision.requires_compaction is False


# 函数说明：test_force_final_answer_still_computes_budget
# 用途：回归验证回归测试与测试辅助中的 `force_final_answer_still_computes_budget` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ToolCall` → `_qwen_adapter` → `_final_response` → `_make_runtime`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`len(started) == 2`。
#   验证条件：`event.context_window == 1000000`。
#   验证条件：`event.input_budget is not None`。
#   验证条件：`event.input_budget > 0`。
@pytest.mark.asyncio
async def test_force_final_answer_still_computes_budget() -> None:
    tool_response = ModelResponse(
        id="r1",
        provider="qwen",
        model="qwen3.7-plus",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=(ToolCall(id="c1", name="noop", arguments={}),),
        ),
    )
    adapter = _qwen_adapter([tool_response, _final_response("done")])
    runtime, adapter = _make_runtime(
        adapter,
        max_tool_rounds=1,
    )
    handler = InMemoryEventHandler()

    await runtime.run("go", event_handler=handler)

    started = [
        event for event in handler.events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert len(started) == 2  
    for event in started:
        assert event.context_window == 1_000_000
        assert event.input_budget is not None
        assert event.input_budget > 0
        assert event.working_input_budget == 64_000
        assert event.hard_trigger_tokens is not None
        assert event.hard_target_tokens is not None
        assert event.tool_result_budget_tokens is not None
        assert event.tool_schema_tokens is not None
        assert event.message_tokens_before is not None
        assert event.message_tokens_after is not None
        assert event.unsummarized_conversation_blocks is not None
        assert event.capability_source == CapabilitySource.BUILTIN.value
        assert event.requires_compaction is False


# 函数说明：test_runtime_blocks_request_that_exceeds_input_budget
# 用途：回归验证回归测试与测试辅助中的
# `runtime_blocks_request_that_exceeds_input_budget` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_qwen_adapter` → `_final_response` →
# `_default_registry` → `registry.register_override` → `ContextManager` →
# `ContextBudgetPolicy`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`result.stop_reason is AgentStopReason.CONTEXT_ERROR`。
#   验证条件：`result.error is not None`。
#   验证条件：`result.error.type == 'ContextWindowExceededError'`。
#   验证条件：`adapter.requests == []`。
@pytest.mark.asyncio
async def test_runtime_blocks_request_that_exceeds_input_budget() -> None:
    adapter = _qwen_adapter([_final_response("不应调用")])
    registry = _default_registry()
    registry.register_override(
        "qwen",
        "qwen3.7-plus",
        context_window=5_000,
        max_output_tokens=4_096,
    )
    manager = ContextManager(
        registry=registry,
        budget_policy=ContextBudgetPolicy(safety_margin_tokens=0),
    )
    runtime, adapter = _make_runtime(adapter, context_manager=manager)
    handler = InMemoryEventHandler()

    result = await runtime.run("x" * 20_000, event_handler=handler)

    assert result.stop_reason is AgentStopReason.CONTEXT_ERROR
    assert result.error is not None
    assert result.error.type == "ContextWindowExceededError"
    assert adapter.requests == []
    started = next(
        event for event in handler.events if event.type is AgentEventType.MODEL_STARTED
    )
    assert started.exceeds_input_budget is True
    assert started.requires_compaction is True
    assert started.original_estimated_input_tokens is not None
    assert started.prepared_input_tokens == started.estimated_input_tokens
    assert started.original_usage_ratio is not None
    assert started.prepared_usage_ratio is not None
    assert started.compaction_stage == "none"
    assert started.compacted_tool_results == 0
    assert started.removed_tool_rounds == 0
    assert started.reached_target is False
    assert started.needs_next_compaction_stage is True


# 函数说明：test_context_preparation_failure_is_not_model_error
# 用途：回归验证回归测试与测试辅助中的 `context_preparation_failure_is_not_model_error`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_qwen_adapter` → `_final_response` →
# `_make_runtime` → `_FailingContextManager` → `runtime.run`。
# 分支与异常：
#   验证条件：`result.stop_reason is AgentStopReason.CONTEXT_ERROR`。
#   验证条件：`result.error is not None`。
#   验证条件：`result.error.type == 'ContextPreparationError'`。
#   验证条件：`adapter.requests == []`。
@pytest.mark.asyncio
async def test_context_preparation_failure_is_not_model_error() -> None:
    adapter = _qwen_adapter([_final_response("不应调用")])
    runtime, adapter = _make_runtime(
        adapter,
        context_manager=_FailingContextManager(),
    )

    result = await runtime.run("hello")

    assert result.stop_reason is AgentStopReason.CONTEXT_ERROR
    assert result.error is not None
    assert result.error.type == "ContextPreparationError"
    assert adapter.requests == []
