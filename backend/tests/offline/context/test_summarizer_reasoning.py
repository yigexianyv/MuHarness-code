
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
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from app.runtime.context import ModelContextSummarizer

_VALID_SUMMARY_JSON = (
    '{"current_objective":"完成测试","user_constraints":[],"key_decisions":[],'
    '"completed_work":[],"current_state":[],"pending_work":[],"important_facts":[]}'
)


class RecordingAdapter(ModelAdapter):

    # 函数说明：RecordingAdapter.__init__
    # 用途：初始化 RecordingAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.requests`。
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.requests: list[ModelRequest] = []

    # 函数说明：RecordingAdapter.complete
    # 用途：完成RecordingAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
    # `ModelUsage`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            id="summary-response",
            provider=self.provider,
            model=self.default_model,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=_VALID_SUMMARY_JSON,
            ),
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    # 函数说明：RecordingAdapter.close
    # 用途：关闭RecordingAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：_registry_and_adapter
# 用途：在回归测试与测试辅助中处理 `_registry_and_adapter`，通过 `registry.register` 完
# 成首个内部处理步骤。
# 参数：
#   provider：模型或搜索服务商，类型 `str`。
# 返回：类型 `tuple[ModelAdapterRegistry, RecordingAdapter]`；返回 `(registry, adapter)`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `RecordingAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`。
def _registry_and_adapter(
    provider: str,
) -> tuple[ModelAdapterRegistry, RecordingAdapter]:
    config = ProviderConfig(
        provider=provider,
        model="summary-model",
        api_key=SecretStr("offline-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = RecordingAdapter(config)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register(
        provider,
        lambda _: adapter,
        config=config,
        replace=True,
    )
    return registry, adapter


# 函数说明：_run_summary
# 用途：运行摘要，供回归测试与测试辅助使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ModelAdapterRegistry`。
#   provider：模型或搜索服务商，类型 `ModelProvider | str`。
#   disable_reasoning：`disable_reasoning`输入或配置值，类型 `bool | None`；默认 `None`
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelContextSummarizer` →
# `summarizer.summarize` → `Message`。
async def _run_summary(
    registry: ModelAdapterRegistry,
    *,
    provider: ModelProvider | str,
    disable_reasoning: bool | None = None,
) -> None:
    kwargs: dict = {"provider": provider}
    if disable_reasoning is not None:
        kwargs["disable_reasoning"] = disable_reasoning
    summarizer = ModelContextSummarizer(registry, **kwargs)
    await summarizer.summarize(
        None,
        (Message(role=MessageRole.USER, content="请压缩"),),
    )


# 函数说明：test_deepseek_disables_reasoning_by_default
# 用途：回归验证回归测试与测试辅助中的 `deepseek_disables_reasoning_by_default` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry_and_adapter` →
# `_run_summary`。
# 分支与异常：
#   验证条件：`adapter.requests[0].extra_body == {'thinking': {'type': 'disabled'}}`。
@pytest.mark.asyncio
async def test_deepseek_disables_reasoning_by_default() -> None:
    registry, adapter = _registry_and_adapter(ModelProvider.DEEPSEEK.value)

    await _run_summary(registry, provider=ModelProvider.DEEPSEEK)

    assert adapter.requests[0].extra_body == {"thinking": {"type": "disabled"}}


# 函数说明：test_qwen_keeps_reasoning_by_default
# 用途：回归验证回归测试与测试辅助中的 `qwen_keeps_reasoning_by_default` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry_and_adapter` →
# `_run_summary`。
# 分支与异常：
#   验证条件：`adapter.requests[0].extra_body == {}`。
@pytest.mark.asyncio
async def test_qwen_keeps_reasoning_by_default() -> None:
    registry, adapter = _registry_and_adapter(ModelProvider.QWEN.value)

    await _run_summary(registry, provider=ModelProvider.QWEN)

    assert adapter.requests[0].extra_body == {}


# 函数说明：test_unknown_provider_keeps_reasoning_by_default
# 用途：回归验证回归测试与测试辅助中的 `unknown_provider_keeps_reasoning_by_default` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry_and_adapter` →
# `_run_summary`。
# 分支与异常：
#   验证条件：`adapter.requests[0].extra_body == {}`。
@pytest.mark.asyncio
async def test_unknown_provider_keeps_reasoning_by_default() -> None:
    registry, adapter = _registry_and_adapter("fake")

    await _run_summary(registry, provider="fake")

    assert adapter.requests[0].extra_body == {}


# 函数说明：test_disable_flag_false_overrides_deepseek
# 用途：回归验证回归测试与测试辅助中的 `disable_flag_false_overrides_deepseek` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry_and_adapter` →
# `_run_summary`。
# 分支与异常：
#   验证条件：`adapter.requests[0].extra_body == {}`。
@pytest.mark.asyncio
async def test_disable_flag_false_overrides_deepseek() -> None:
    registry, adapter = _registry_and_adapter(ModelProvider.DEEPSEEK.value)

    await _run_summary(
        registry,
        provider=ModelProvider.DEEPSEEK,
        disable_reasoning=False,
    )

    assert adapter.requests[0].extra_body == {}


# 函数说明：test_disable_flag_true_overrides_qwen
# 用途：回归验证回归测试与测试辅助中的 `disable_flag_true_overrides_qwen` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry_and_adapter` →
# `_run_summary`。
# 分支与异常：
#   验证条件：`adapter.requests[0].extra_body == {'thinking': {'type': 'disabled'}}`。
@pytest.mark.asyncio
async def test_disable_flag_true_overrides_qwen() -> None:
    registry, adapter = _registry_and_adapter(ModelProvider.QWEN.value)

    await _run_summary(
        registry,
        provider=ModelProvider.QWEN,
        disable_reasoning=True,
    )

    assert adapter.requests[0].extra_body == {"thinking": {"type": "disabled"}}


# 函数说明：test_summary_request_keeps_schema_and_max_tokens
# 用途：回归验证回归测试与测试辅助中的 `summary_request_keeps_schema_and_max_tokens` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry_and_adapter` →
# `_run_summary`。
# 分支与异常：
#   验证条件：`request.tools == ()`。
#   验证条件：`request.max_output_tokens == 1024`。
#   验证条件：`request.messages[0].role is MessageRole.SYSTEM`。
@pytest.mark.asyncio
async def test_summary_request_keeps_schema_and_max_tokens() -> None:
    registry, adapter = _registry_and_adapter(ModelProvider.DEEPSEEK.value)

    await _run_summary(registry, provider=ModelProvider.DEEPSEEK)

    request = adapter.requests[0]
    assert request.tools == ()
    assert request.max_output_tokens == 1_024
    assert request.messages[0].role is MessageRole.SYSTEM
