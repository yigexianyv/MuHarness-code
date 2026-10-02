from __future__ import annotations

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
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.context import ContextDecision, ContextManager, TokenEstimator
from app.tools import ToolRegistry


# 函数说明：test_estimate_text_returns_positive_count
# 用途：回归验证回归测试与测试辅助中的 `estimate_text_returns_positive_count` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` →
# `estimator.estimate_text`。
# 分支与异常：
#   验证条件：`estimator.estimate_text('hello world') > 0`。
def test_estimate_text_returns_positive_count() -> None:
    estimator = TokenEstimator()
    assert estimator.estimate_text("hello world") > 0


# 函数说明：test_estimate_empty_messages_is_zero
# 用途：回归验证回归测试与测试辅助中的 `estimate_empty_messages_is_zero` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` →
# `estimator.estimate_messages`。
# 分支与异常：
#   验证条件：`estimator.estimate_messages([]) == 0`。
def test_estimate_empty_messages_is_zero() -> None:
    estimator = TokenEstimator()
    assert estimator.estimate_messages([]) == 0


# 函数说明：test_estimate_messages_counts_content_and_role
# 用途：回归验证回归测试与测试辅助中的 `estimate_messages_counts_content_and_role` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` → `Message` →
# `estimator.estimate_messages` → `estimator.estimate_text`。
# 分支与异常：
#   验证条件：
# `estimator.estimate_messages(messages) > estimator.estimate_text('hello world')`。
def test_estimate_messages_counts_content_and_role() -> None:
    estimator = TokenEstimator()
    messages = (Message(role=MessageRole.USER, content="hello world"),)
    assert estimator.estimate_messages(messages) > estimator.estimate_text(
        "hello world"
    )


# 函数说明：test_estimate_messages_counts_tool_calls
# 用途：回归验证回归测试与测试辅助中的 `estimate_messages_counts_tool_calls` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` → `Message` →
# `ToolCall` → `estimator.estimate_messages`。
# 分支与异常：
#   验证条件：
# `estimator.estimate_messages(with_tool) > estimator.estimate_messages(plain)`。
def test_estimate_messages_counts_tool_calls() -> None:
    estimator = TokenEstimator()
    plain = (Message(role=MessageRole.USER, content="hello"),)
    with_tool = (
        Message(
            role=MessageRole.ASSISTANT,
            content="searching",
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="web_search",
                    arguments={"query": "MuHarness"},
                ),
            ),
        ),
    )
    assert estimator.estimate_messages(with_tool) > estimator.estimate_messages(plain)


# 函数说明：test_estimate_tools_counts_definition
# 用途：回归验证回归测试与测试辅助中的 `estimate_tools_counts_definition` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` → `ToolDefinition` →
# `estimator.estimate_tools`。
# 分支与异常：
#   验证条件：`estimator.estimate_tools((tool,)) > 0`。
def test_estimate_tools_counts_definition() -> None:
    estimator = TokenEstimator()
    tool = ToolDefinition(
        name="web_search",
        description="search the web",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )
    assert estimator.estimate_tools((tool,)) > 0


# 函数说明：test_estimate_request_sums_messages_and_tools
# 用途：回归验证回归测试与测试辅助中的 `estimate_request_sums_messages_and_tools` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` → `Message` →
# `ToolDefinition` → `estimator.estimate_request` → `estimator.estimate_messages` →
# `estimator.estimate_tools`。
# 分支与异常：
#   验证条件：`request_tokens >= estimator.estimate_messages(messages)`。
#   验证条件：`request_tokens >= estimator.estimate_tools((tool,))`。
def test_estimate_request_sums_messages_and_tools() -> None:
    estimator = TokenEstimator()
    messages = (Message(role=MessageRole.USER, content="hi"),)
    tool = ToolDefinition(name="foo", description="bar")
    request_tokens = estimator.estimate_request(messages, tools=(tool,))
    assert request_tokens >= estimator.estimate_messages(messages)
    assert request_tokens >= estimator.estimate_tools((tool,))


# 函数说明：test_default_token_estimator_is_shared
# 用途：回归验证回归测试与测试辅助中的 `default_token_estimator_is_shared` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`default_token_estimator`。
# 分支与异常：
#   验证条件：`default_token_estimator() is default_token_estimator()`。
def test_default_token_estimator_is_shared() -> None:
    from app.runtime.context import default_token_estimator

    assert default_token_estimator() is default_token_estimator()


# 函数说明：test_openai_model_has_no_conservative_factor
# 用途：回归验证回归测试与测试辅助中的 `openai_model_has_no_conservative_factor` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` →
# `estimator.factor_for`。
# 分支与异常：
#   验证条件：`estimator.factor_for('openai', 'gpt-4o') == 1.0`。
#   验证条件：`estimator.factor_for(None, 'gpt-5.4-mini') == 1.0`。
def test_openai_model_has_no_conservative_factor() -> None:
    estimator = TokenEstimator()
    assert estimator.factor_for("openai", "gpt-4o") == 1.0
    assert estimator.factor_for(None, "gpt-5.4-mini") == 1.0


# 函数说明：test_non_openai_models_get_conservative_factor
# 用途：回归验证回归测试与测试辅助中的 `non_openai_models_get_conservative_factor` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` →
# `estimator.factor_for`。
# 分支与异常：
#   验证条件：`estimator.factor_for('qwen', 'qwen3.7-plus') > 1.0`。
#   验证条件：`estimator.factor_for('deepseek', 'deepseek-v4-flash') > 1.0`。
#   验证条件：`estimator.factor_for('anthropic', 'claude-sonnet-4-6') > 1.0`。
#   验证条件：`estimator.factor_for(None, 'unknown-model') > 1.0`。
def test_non_openai_models_get_conservative_factor() -> None:
    estimator = TokenEstimator()
    assert estimator.factor_for("qwen", "qwen3.7-plus") > 1.0
    assert estimator.factor_for("deepseek", "deepseek-v4-flash") > 1.0
    assert estimator.factor_for("anthropic", "claude-sonnet-4-6") > 1.0
    assert estimator.factor_for(None, "unknown-model") > 1.0


# 函数说明：test_conservative_factor_inflates_non_openai_estimate
# 用途：回归验证回归测试与测试辅助中的
# `conservative_factor_inflates_non_openai_estimate` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` →
# `estimator.estimate_text` → `raw_estimator.estimate_text`。
# 分支与异常：
#   验证条件：`inflated > raw`。
def test_conservative_factor_inflates_non_openai_estimate() -> None:
    estimator = TokenEstimator()
    raw_estimator = TokenEstimator(factors={"qwen": 1.0, "other": 1.0})

    inflated = estimator.estimate_text(
        "hello world",
        model="qwen3.7-plus",
        provider="qwen",
    )
    raw = raw_estimator.estimate_text(
        "hello world",
        model="qwen3.7-plus",
        provider="qwen",
    )

    assert inflated > raw


# 函数说明：test_custom_factors_override_defaults
# 用途：回归验证回归测试与测试辅助中的 `custom_factors_override_defaults` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TokenEstimator` →
# `estimator.factor_for`。
# 分支与异常：
#   验证条件：`estimator.factor_for('qwen', 'qwen3.7-plus') == 1.5`。
def test_custom_factors_override_defaults() -> None:
    estimator = TokenEstimator(factors={"qwen": 1.5})
    assert estimator.factor_for("qwen", "qwen3.7-plus") == 1.5


class _FakeAdapter(ModelAdapter):
    # 函数说明：_FakeAdapter.__init__
    # 用途：初始化 _FakeAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   content：内容正文，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self._content`、`self.requests`。
    def __init__(self, config: ProviderConfig, content: str) -> None:
        super().__init__(config)
        self._content = content
        self.requests: list[ModelRequest] = []

    # 函数说明：_FakeAdapter.complete
    # 用途：完成_FakeAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            id="r",
            provider=self.provider,
            model=self.default_model,
            message=Message(role=MessageRole.ASSISTANT, content=self._content),
        )

    # 函数说明：_FakeAdapter.close
    # 用途：关闭_FakeAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：_fake_runtime
# 用途：在回归测试与测试辅助中处理 `_fake_runtime`，通过 `registry.register` 完成首个内
# 部处理步骤。
# 参数：
#   content：内容正文，类型 `str`；默认 `'完成'`。
#   context_manager：上下文预算与压缩管理器，类型 `ContextManager | None`；默认 `None`。
# 返回：类型 `tuple[AgentRuntime, _FakeAdapter]`；返回 `(AgentRuntime(registry,
# ToolRegistry(), provider='fake', context_manager=context_manager)…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `_FakeAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`；另有
# 2 个调用点。
def _fake_runtime(
    content: str = "完成",
    *,
    context_manager: ContextManager | None = None,
) -> tuple[AgentRuntime, _FakeAdapter]:
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = _FakeAdapter(config, content)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)
    return (
        AgentRuntime(
            registry,
            ToolRegistry(),
            provider="fake",
            context_manager=context_manager,
        ),
        adapter,
    )


# 函数说明：test_runtime_emits_estimated_input_tokens_before_model_call
# 用途：回归验证回归测试与测试辅助中的
# `runtime_emits_estimated_input_tokens_before_model_call` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_fake_runtime` → `ContextManager` →
# `InMemoryEventHandler` → `runtime.run` → `next`。
# 分支与异常：
#   验证条件：`started.estimated_input_tokens is not None`。
#   验证条件：`started.estimated_input_tokens > 0`。
#   验证条件：`started.context_window is not None`。
#   验证条件：`started.context_window > 0`。
@pytest.mark.asyncio
async def test_runtime_emits_estimated_input_tokens_before_model_call() -> None:
    runtime, adapter = _fake_runtime(context_manager=ContextManager())
    handler = InMemoryEventHandler()

    await runtime.run("hello world", event_handler=handler)

    started = next(
        event for event in handler.events if event.type is AgentEventType.MODEL_STARTED
    )
    assert started.estimated_input_tokens is not None
    assert started.estimated_input_tokens > 0
    assert started.context_window is not None
    assert started.context_window > 0
    assert started.input_budget is not None
    assert started.input_budget > 0
    assert started.step == 1


class _EmptyContextManager(ContextManager):
    # 函数说明：_EmptyContextManager.prepare
    # 用途：准备_EmptyContextManager，供回归测试与测试辅助使用。
    # 参数：
    #   messages：本次处理的消息序列。
    #   tools：可用工具定义或工具实例集合；默认 `()`。
    #   model：模型名称；默认 `None`。
    #   provider：模型或搜索服务商；默认 `None`。
    #   max_output_tokens：模型输出 Token 上限；默认 `None`。
    #   history_count：原始历史消息数量；默认 `None`。
    #   summary_state：会话摘要及覆盖水位；默认 `None`。
    #   source_messages：消息序列输入或配置值；默认 `None`。
    #   protected_source_indices：必须保留的原始消息索引；默认 `()`。
    #   allow_compaction：`allow_compaction`输入或配置值；默认 `True`。
    # 返回：返回 `ContextDecision(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextDecision`。
    async def prepare(
        self,
        messages,
        *,
        tools=(),
        model=None,
        provider=None,
        max_output_tokens=None,
        history_count=None,
        summary_state=None,
        source_messages=None,
        protected_source_indices=(),
        allow_compaction=True,
    ):
        return ContextDecision(
            messages=tuple(messages),
            tools=tuple(tools),
            estimated_input_tokens=None,
            trimmed=False,
        )

    # 函数说明：_EmptyContextManager.remeasure
    # 用途：返回 `decision`，提供 _EmptyContextManager 的派生值。
    # 参数：
    #   decision：权限、上下文或审计决策。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：返回 `decision`。
    def remeasure(self, decision, **kwargs):
        return decision


# 函数说明：test_runtime_respects_context_manager_decision
# 用途：回归验证回归测试与测试辅助中的 `runtime_respects_context_manager_decision` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_fake_runtime` →
# `_EmptyContextManager` → `InMemoryEventHandler` → `runtime.run` → `next`。
# 分支与异常：
#   验证条件：`started.estimated_input_tokens is None`。
#   验证条件：`started.context_trimmed is False`。
@pytest.mark.asyncio
async def test_runtime_respects_context_manager_decision() -> None:
    runtime, _ = _fake_runtime(context_manager=_EmptyContextManager())
    handler = InMemoryEventHandler()

    await runtime.run("hi", event_handler=handler)

    started = next(
        event for event in handler.events if event.type is AgentEventType.MODEL_STARTED
    )
    assert started.estimated_input_tokens is None
    assert started.context_trimmed is False


# 函数说明：test_context_manager_returns_messages_unchanged_and_estimates
# 用途：回归验证回归测试与测试辅助中的
# `context_manager_returns_messages_unchanged_and_estimates` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextManager` → `TokenEstimator` →
# `Message` → `manager.prepare`。
# 分支与异常：
#   验证条件：`decision.messages == messages`。
#   验证条件：`decision.tools == ()`。
#   验证条件：`decision.trimmed is False`。
#   验证条件：`decision.estimated_input_tokens is not None`。
@pytest.mark.asyncio
async def test_context_manager_returns_messages_unchanged_and_estimates() -> None:
    manager = ContextManager(TokenEstimator())
    messages = (Message(role=MessageRole.USER, content="hello"),)

    decision = await manager.prepare(
        messages,
        tools=(),
        model="qwen3.7-plus",
        provider="qwen",
    )

    assert decision.messages == messages
    assert decision.tools == ()
    assert decision.trimmed is False
    assert decision.estimated_input_tokens is not None
    assert decision.estimated_input_tokens > 0
