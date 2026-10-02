from __future__ import annotations

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from app.application import select_provider
from app.models import (
    ApiStyle,
    Message,
    MessageRole,
    ModelAdapterError,
    ModelProvider,
    ModelRequest,
    ModelSettings,
    ProviderConfig,
    ProviderNotConfiguredError,
    ToolCall,
    ToolDefinition,
)
from app.models.providers import AnthropicAdapter, OpenAICompatibleAdapter


class AsyncRecorder:
    # 函数说明：AsyncRecorder.__init__
    # 用途：初始化 AsyncRecorder；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   response：模型、工具或服务返回的响应，类型 `Any`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.response`、`self.kwargs`。
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] | None = None

    # 函数说明：AsyncRecorder.create
    # 用途：创建AsyncRecorder，供回归测试与测试辅助使用。
    # 参数：
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：类型 `Any`；返回 `self.response`。
    # 副作用与资源：
    #   更新对象字段：`self.kwargs`。
    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


class AsyncSequenceRecorder:
    # 函数说明：AsyncSequenceRecorder.__init__
    # 用途：初始化 AsyncSequenceRecorder；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   responses：预设的模型或服务响应序列，类型 `list[Any]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`iter`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.call_count`。
    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)
        self.call_count = 0

    # 函数说明：AsyncSequenceRecorder.create
    # 用途：创建AsyncSequenceRecorder，供回归测试与测试辅助使用。
    # 参数：
    #   **_：额外关键字参数，按实现处理或转交。
    # 返回：类型 `Any`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`next`。
    # 分支与异常：
    #   当 `isinstance(response, BaseException)` 时，抛出 `response`。
    # 副作用与资源：
    #   更新对象字段：`self.call_count`。
    async def create(self, **_: Any) -> Any:
        self.call_count += 1
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


class AsyncStream:
    # 函数说明：AsyncStream.__init__
    # 用途：初始化 AsyncStream；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   items：`items`输入或配置值，类型 `list[Any]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._items`。
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    # 函数说明：AsyncStream.__aiter__
    # 用途：处理回归测试与测试辅助中的 `__aiter__` 数据；结果及边界条件见下方说明。
    # 返回：返回 `self`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`iter`。
    # 副作用与资源：
    #   更新对象字段：`self._iterator`。
    def __aiter__(self):
        self._iterator = iter(self._items)
        return self

    # 函数说明：AsyncStream.__anext__
    # 用途：处理回归测试与测试辅助中的 `__anext__` 数据；结果及边界条件见下方说明。
    # 返回：类型 `Any`；返回 `item`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`next`。
    # 分支与异常：
    #   捕获 `StopIteration` 后，转换或抛出 `StopAsyncIteration`。
    #   当 `isinstance(item, BaseException)` 时，抛出 `item`。
    async def __anext__(self) -> Any:
        try:
            item = next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        if isinstance(item, BaseException):
            raise item
        return item


class FakeAnthropicMessageStream:
    # 函数说明：FakeAnthropicMessageStream.__init__
    # 用途：初始化 FakeAnthropicMessageStream；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   deltas：传给 `AsyncStream` 的输入，类型 `list[str]`。
    #   final_message：消息输入或配置值，类型 `Any`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AsyncStream`。
    # 副作用与资源：
    #   更新对象字段：`self.text_stream`、`self.final_message`。
    def __init__(self, deltas: list[str], final_message: Any) -> None:
        self.text_stream = AsyncStream(deltas)
        self.final_message = final_message

    # 函数说明：FakeAnthropicMessageStream.__aenter__
    # 用途：进入 FakeAnthropicMessageStream 的资源管理上下文。
    # 返回：类型 `FakeAnthropicMessageStream`；返回 `self`。
    async def __aenter__(self) -> FakeAnthropicMessageStream:
        return self

    # 函数说明：FakeAnthropicMessageStream.__aexit__
    # 用途：退出 FakeAnthropicMessageStream 的资源管理上下文，执行当前实现规定的清理。
    # 参数：
    #   *args：额外位置参数，按实现向内部调用传递。
    # 返回：类型 `None`；无结果值，显式返回 None。
    async def __aexit__(self, *args: Any) -> None:
        return None

    # 函数说明：FakeAnthropicMessageStream.get_final_message
    # 用途：获取消息，供回归测试与测试辅助使用。
    # 返回：类型 `Any`；返回 `self.final_message`。
    async def get_final_message(self) -> Any:
        return self.final_message


class FakeOpenAIClient:
    # 函数说明：FakeOpenAIClient.__init__
    # 用途：初始化 FakeOpenAIClient；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   responses_result：传给 `AsyncRecorder` 的输入，类型 `Any | None`；默认 `None`。
    #   chat_result：传给 `AsyncRecorder` 的输入，类型 `Any | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AsyncRecorder` →
    # `SimpleNamespace`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.chat`、`self.closed`。
    def __init__(
        self,
        *,
        responses_result: Any | None = None,
        chat_result: Any | None = None,
    ) -> None:
        self.responses = AsyncRecorder(responses_result)
        self.chat = SimpleNamespace(completions=AsyncRecorder(chat_result))
        self.closed = False

    # 函数说明：FakeOpenAIClient.close
    # 用途：关闭FakeOpenAIClient，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.closed`。
    async def close(self) -> None:
        self.closed = True


class FakeAnthropicClient:
    # 函数说明：FakeAnthropicClient.__init__
    # 用途：初始化 FakeAnthropicClient；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   response：模型、工具或服务返回的响应，类型 `Any`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AsyncRecorder`。
    # 副作用与资源：
    #   更新对象字段：`self.messages`、`self.closed`。
    def __init__(self, response: Any) -> None:
        self.messages = AsyncRecorder(response)
        self.closed = False

    # 函数说明：FakeAnthropicClient.close
    # 用途：关闭FakeAnthropicClient，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.closed`。
    async def close(self) -> None:
        self.closed = True


class FakeStreamingAnthropicClient:
    # 函数说明：FakeStreamingAnthropicClient.__init__
    # 用途：初始化 FakeStreamingAnthropicClient；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   stream：事件流输入或配置值，类型 `FakeAnthropicMessageStream`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace`。
    # 副作用与资源：
    #   更新对象字段：`self.messages`。
    def __init__(self, stream: FakeAnthropicMessageStream) -> None:
        self.messages = SimpleNamespace(stream=lambda **_: stream)

    # 函数说明：FakeStreamingAnthropicClient.close
    # 用途：关闭FakeStreamingAnthropicClient，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：capture_deltas
# 用途：处理回归测试与测试辅助中的 `capture_deltas` 数据；结果及边界条件见下方说明。
# 参数：
#   target：`target`输入或配置值，类型 `list[str]`。
# 返回：类型 `Callable[[str], Awaitable[None]]`；返回 `capture`。
def capture_deltas(target: list[str]) -> Callable[[str], Awaitable[None]]:
    # 函数说明：capture_deltas.capture
    # 用途：在回归测试与测试辅助中处理 `capture`，通过 `target.append` 完成首个内部处理
    # 步骤。
    # 参数：
    #   delta：传给 `target.append` 的输入，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 闭包依赖：从外层读取 `target`。
    async def capture(delta: str) -> None:
        target.append(delta)

    return capture


# 函数说明：provider_config
# 用途：返回 `ProviderConfig(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   provider：模型或搜索服务商，类型 `str`。
#   api_style：`api_style`输入或配置值，类型 `ApiStyle`。
#   model：模型名称，类型 `str`。
#   max_retries：`max_retries`输入或配置值，类型 `int`；默认 `2`。
# 返回：类型 `ProviderConfig`；返回 `ProviderConfig(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr`。
def provider_config(
    provider: str,
    api_style: ApiStyle,
    model: str,
    *,
    max_retries: int = 2,
) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model=model,
        api_key=SecretStr("test-key"),
        api_style=api_style,
        max_retries=max_retries,
    )


# 函数说明：test_openai_responses_adapter_normalizes_tool_calls
# 用途：回归验证回归测试与测试辅助中的 `openai_responses_adapter_normalizes_tool_calls`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` → `FakeOpenAIClient`
#  → `OpenAICompatibleAdapter` → `provider_config` → `adapter.complete` → `ModelRequest`
# ；另有 2 个调用点。
# 分支与异常：
#   验证条件：`response.provider == 'openai'`。
#   验证条件：`response.finish_reason == 'tool_calls'`。
#   验证条件：`response.message.tool_calls[0].arguments == {'city': 'Shanghai'}`。
#   验证条件：`client.responses.kwargs['tools'][0]['name'] == 'weather'`。
@pytest.mark.asyncio
async def test_openai_responses_adapter_normalizes_tool_calls() -> None:
    result = SimpleNamespace(
        id="resp_1",
        model="gpt-test",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="weather",
                arguments='{"city":"Shanghai"}',
            )
        ],
        status="completed",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
            input_tokens_details=SimpleNamespace(
                cached_tokens=6,
                cache_creation_input_tokens=2,
            ),
        ),
    )
    client = FakeOpenAIClient(responses_result=result)
    adapter = OpenAICompatibleAdapter(
        provider_config("openai", ApiStyle.RESPONSES, "gpt-test"),
        client=client,
    )

    response = await adapter.complete(
        ModelRequest(
            messages=(Message(role=MessageRole.USER, content="Weather?"),),
            tools=(
                ToolDefinition(
                    name="weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            ),
        )
    )

    assert response.provider == "openai"
    assert response.finish_reason == "tool_calls"
    assert response.message.tool_calls[0].arguments == {"city": "Shanghai"}
    assert client.responses.kwargs["tools"][0]["name"] == "weather"
    assert response.usage.cached_input_tokens == 6
    assert response.usage.uncached_input_tokens == 4
    assert response.usage.cache_write_input_tokens == 2
    assert response.usage.model_calls == 1


# 函数说明：test_openai_responses_stream_emits_text_and_returns_final_response
# 用途：回归验证回归测试与测试辅助中的
# `openai_responses_stream_emits_text_and_returns_final_response` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` → `AsyncStream` →
# `OpenAICompatibleAdapter` → `provider_config` → `FakeOpenAIClient` →
# `adapter.complete_stream`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`deltas == ['你', '好']`。
#   验证条件：`response.message.content == '你好'`。
#   验证条件：`response.usage.total_tokens == 6`。
@pytest.mark.asyncio
async def test_openai_responses_stream_emits_text_and_returns_final_response() -> None:
    final = SimpleNamespace(
        id="resp_stream",
        model="gpt-test",
        output_text="你好",
        output=[],
        status="completed",
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
    )
    stream = AsyncStream(
        [
            SimpleNamespace(type="response.output_text.delta", delta="你"),
            SimpleNamespace(type="response.output_text.delta", delta="好"),
            SimpleNamespace(type="response.completed", response=final),
        ]
    )
    adapter = OpenAICompatibleAdapter(
        provider_config("openai", ApiStyle.RESPONSES, "gpt-test"),
        client=FakeOpenAIClient(responses_result=stream),
    )
    deltas: list[str] = []

    response = await adapter.complete_stream(
        ModelRequest(messages=(Message(role=MessageRole.USER, content="hello"),)),
        on_text_delta=capture_deltas(deltas),
    )

    assert deltas == ["你", "好"]
    assert response.message.content == "你好"
    assert response.usage.total_tokens == 6


# 函数说明：test_openai_compatible_chat_adapter_preserves_tool_history
# 用途：回归验证回归测试与测试辅助中的
# `openai_compatible_chat_adapter_preserves_tool_history` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` → `FakeOpenAIClient`
#  → `OpenAICompatibleAdapter` → `provider_config` → `adapter.complete` → `ModelRequest`
# ；另有 2 个调用点。
# 分支与异常：
#   验证条件：`sent[0]['tool_calls'][0]['function']['arguments'] == '{"id":7}'`。
#   验证条件：`sent[1]['tool_call_id'] == 'call_1'`。
#   验证条件：`response.message.content == 'done'`。
#   验证条件：`response.message.reasoning == '先分析用户意图'`。
@pytest.mark.asyncio
async def test_openai_compatible_chat_adapter_preserves_tool_history() -> None:
    result = SimpleNamespace(
        id="chat_1",
        model="deepseek-test",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="done",
                    tool_calls=None,
                    reasoning_content="先分析用户意图",
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=8,
            completion_tokens=2,
            total_tokens=10,
            prompt_cache_hit_tokens=5,
            prompt_cache_miss_tokens=3,
        ),
    )
    client = FakeOpenAIClient(chat_result=result)
    adapter = OpenAICompatibleAdapter(
        provider_config(
            "deepseek",
            ApiStyle.CHAT_COMPLETIONS,
            "deepseek-test",
        ),
        client=client,
    )

    response = await adapter.complete(
        ModelRequest(
            messages=(
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="call_1",
                            name="lookup",
                            arguments={"id": 7},
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.TOOL,
                    tool_call_id="call_1",
                    content="record",
                ),
            )
        )
    )

    sent = client.chat.completions.kwargs["messages"]
    assert sent[0]["tool_calls"][0]["function"]["arguments"] == '{"id":7}'
    assert sent[1]["tool_call_id"] == "call_1"
    assert response.message.content == "done"
    assert response.message.reasoning == "先分析用户意图"
    assert response.usage.cached_input_tokens == 5
    assert response.usage.uncached_input_tokens == 3


# 函数说明：test_openai_chat_stream_rebuilds_text_and_tool_calls
# 用途：回归验证回归测试与测试辅助中的 `openai_chat_stream_rebuilds_text_and_tool_calls`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AsyncStream` → `SimpleNamespace` →
# `OpenAICompatibleAdapter` → `provider_config` → `FakeOpenAIClient` →
# `adapter.complete_stream`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`deltas == ['先']`。
#   验证条件：`response.message.content == '先'`。
#   验证条件：`response.message.tool_calls[0].arguments == {'query': 'MuHarness'}`。
#   验证条件：`response.message.reasoning == '思考中'`。
@pytest.mark.asyncio
async def test_openai_chat_stream_rebuilds_text_and_tool_calls() -> None:
    chunks = AsyncStream(
        [
            SimpleNamespace(
                id="chat-stream",
                model="deepseek-test",
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content="先",
                            tool_calls=None,
                            reasoning_content="思考中",
                        ),
                    )
                ],
            ),
            SimpleNamespace(
                id="chat-stream",
                model="deepseek-test",
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="search",
                                        arguments='{"query":"MuHarness"}',
                                    ),
                                )
                            ],
                        ),
                    )
                ],
            ),
            SimpleNamespace(
                id="chat-stream",
                model="deepseek-test",
                usage=SimpleNamespace(
                    prompt_tokens=5,
                    completion_tokens=3,
                    total_tokens=8,
                ),
                choices=[],
            ),
        ]
    )
    adapter = OpenAICompatibleAdapter(
        provider_config("deepseek", ApiStyle.CHAT_COMPLETIONS, "deepseek-test"),
        client=FakeOpenAIClient(chat_result=chunks),
    )
    deltas: list[str] = []

    response = await adapter.complete_stream(
        ModelRequest(messages=(Message(role=MessageRole.USER, content="search"),)),
        on_text_delta=capture_deltas(deltas),
    )

    assert deltas == ["先"]
    assert response.message.content == "先"
    assert response.message.tool_calls[0].arguments == {"query": "MuHarness"}
    assert response.message.reasoning == "思考中"
    assert response.usage.total_tokens == 8


# 函数说明：test_openai_responses_stream_retries_before_visible_delta
# 用途：回归验证回归测试与测试辅助中的
# `openai_responses_stream_retries_before_visible_delta` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` →
# `AsyncSequenceRecorder` → `AsyncStream` → `FakeOpenAIClient` →
# `OpenAICompatibleAdapter` → `provider_config`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`recorder.call_count == 2`。
#   验证条件：`deltas == ['完成']`。
#   验证条件：`response.message.content == '完成'`。
# 副作用与资源：
#   更新对象字段：`client.responses`。
@pytest.mark.asyncio
async def test_openai_responses_stream_retries_before_visible_delta() -> None:
    final = SimpleNamespace(
        id="resp-retried",
        model="gpt-test",
        output_text="完成",
        output=[],
        status="completed",
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
    )
    recorder = AsyncSequenceRecorder(
        [
            AsyncStream([RuntimeError("incomplete chunked read")]),
            AsyncStream(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="完成"),
                    SimpleNamespace(type="response.completed", response=final),
                ]
            ),
        ]
    )
    client = FakeOpenAIClient()
    client.responses = recorder
    adapter = OpenAICompatibleAdapter(
        provider_config(
            "openai",
            ApiStyle.RESPONSES,
            "gpt-test",
            max_retries=1,
        ),
        client=client,
    )
    deltas: list[str] = []

    response = await adapter.complete_stream(
        ModelRequest(messages=(Message(role=MessageRole.USER, content="hello"),)),
        on_text_delta=capture_deltas(deltas),
    )

    assert recorder.call_count == 2
    assert deltas == ["完成"]
    assert response.message.content == "完成"


# 函数说明：test_openai_chat_stream_does_not_retry_after_text_delta
# 用途：回归验证回归测试与测试辅助中的
# `openai_chat_stream_does_not_retry_after_text_delta` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AsyncStream` → `SimpleNamespace` →
# `AsyncSequenceRecorder` → `FakeOpenAIClient` → `OpenAICompatibleAdapter` →
# `provider_config`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`recorder.call_count == 1`。
#   验证条件：`deltas == ['半']`。
#   预期异常：`pytest.raises(ModelAdapterError, match='incomplete chunked read')`。
# 副作用与资源：
#   更新对象字段：`client.chat.completions`。
@pytest.mark.asyncio
async def test_openai_chat_stream_does_not_retry_after_text_delta() -> None:
    first_stream = AsyncStream(
        [
            SimpleNamespace(
                id="chat-first",
                model="deepseek-test",
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content="半",
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                    )
                ],
            ),
            RuntimeError("incomplete chunked read"),
        ]
    )
    recorder = AsyncSequenceRecorder([first_stream, AsyncStream([])])
    client = FakeOpenAIClient()
    client.chat.completions = recorder
    adapter = OpenAICompatibleAdapter(
        provider_config(
            "deepseek",
            ApiStyle.CHAT_COMPLETIONS,
            "deepseek-test",
            max_retries=2,
        ),
        client=client,
    )
    deltas: list[str] = []

    with pytest.raises(ModelAdapterError, match="incomplete chunked read"):
        await adapter.complete_stream(
            ModelRequest(messages=(Message(role=MessageRole.USER, content="hello"),)),
            on_text_delta=capture_deltas(deltas),
        )

    assert recorder.call_count == 1
    assert deltas == ["半"]


# 函数说明：test_openai_chat_stream_does_not_retry_after_reasoning_delta
# 用途：回归验证回归测试与测试辅助中的
# `openai_chat_stream_does_not_retry_after_reasoning_delta` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AsyncStream` → `SimpleNamespace` →
# `AsyncSequenceRecorder` → `FakeOpenAIClient` → `OpenAICompatibleAdapter` →
# `provider_config`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`recorder.call_count == 1`。
#   验证条件：`reasoning_deltas == ['正在分析']`。
#   预期异常：`pytest.raises(ModelAdapterError, match='stream interrupted')`。
# 副作用与资源：
#   更新对象字段：`client.chat.completions`。
@pytest.mark.asyncio
async def test_openai_chat_stream_does_not_retry_after_reasoning_delta() -> None:
    first_stream = AsyncStream(
        [
            SimpleNamespace(
                id="chat-first",
                model="deepseek-test",
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content="正在分析",
                            tool_calls=None,
                        ),
                    )
                ],
            ),
            RuntimeError("stream interrupted"),
        ]
    )
    recorder = AsyncSequenceRecorder([first_stream, AsyncStream([])])
    client = FakeOpenAIClient()
    client.chat.completions = recorder
    adapter = OpenAICompatibleAdapter(
        provider_config(
            "deepseek",
            ApiStyle.CHAT_COMPLETIONS,
            "deepseek-test",
            max_retries=2,
        ),
        client=client,
    )
    reasoning_deltas: list[str] = []

    with pytest.raises(ModelAdapterError, match="stream interrupted"):
        await adapter.complete_stream(
            ModelRequest(messages=(Message(role=MessageRole.USER, content="hello"),)),
            on_text_delta=capture_deltas([]),
            on_reasoning_delta=capture_deltas(reasoning_deltas),
        )

    assert recorder.call_count == 1
    assert reasoning_deltas == ["正在分析"]


# 函数说明：test_openai_chat_stream_retries_after_only_tool_deltas
# 用途：回归验证回归测试与测试辅助中的
# `openai_chat_stream_retries_after_only_tool_deltas` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AsyncStream` → `SimpleNamespace` →
# `AsyncSequenceRecorder` → `FakeOpenAIClient` → `OpenAICompatibleAdapter` →
# `provider_config`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`recorder.call_count == 2`。
#   验证条件：`len(response.message.tool_calls) == 1`。
#   验证条件：`response.message.tool_calls[0].id == 'new-call'`。
#   验证条件：`response.message.tool_calls[0].name == 'search'`。
# 副作用与资源：
#   更新对象字段：`client.chat.completions`。
@pytest.mark.asyncio
async def test_openai_chat_stream_retries_after_only_tool_deltas() -> None:
    first_stream = AsyncStream(
        [
            SimpleNamespace(
                id="chat-first",
                model="deepseek-test",
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="old-call",
                                    function=SimpleNamespace(
                                        name="old_tool",
                                        arguments='{"old":true}',
                                    ),
                                )
                            ],
                        ),
                    )
                ],
            ),
            RuntimeError("stream interrupted"),
        ]
    )
    second_stream = AsyncStream(
        [
            SimpleNamespace(
                id="chat-second",
                model="deepseek-test",
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="new-call",
                                    function=SimpleNamespace(
                                        name="search",
                                        arguments='{"query":"MuHarness"}',
                                    ),
                                )
                            ],
                        ),
                    )
                ],
            )
        ]
    )
    recorder = AsyncSequenceRecorder([first_stream, second_stream])
    client = FakeOpenAIClient()
    client.chat.completions = recorder
    adapter = OpenAICompatibleAdapter(
        provider_config(
            "deepseek",
            ApiStyle.CHAT_COMPLETIONS,
            "deepseek-test",
            max_retries=1,
        ),
        client=client,
    )

    response = await adapter.complete_stream(
        ModelRequest(messages=(Message(role=MessageRole.USER, content="search"),)),
        on_text_delta=capture_deltas([]),
    )

    assert recorder.call_count == 2
    assert len(response.message.tool_calls) == 1
    assert response.message.tool_calls[0].id == "new-call"
    assert response.message.tool_calls[0].name == "search"


# 函数说明：test_openai_chat_stream_retry_count_uses_provider_config
# 用途：回归验证回归测试与测试辅助中的
# `openai_chat_stream_retry_count_uses_provider_config` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AsyncSequenceRecorder` →
# `AsyncStream` → `FakeOpenAIClient` → `OpenAICompatibleAdapter` → `provider_config` →
# `pytest.raises`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`recorder.call_count == 2`。
#   预期异常：`pytest.raises(ModelAdapterError, match='second interruption')`。
# 副作用与资源：
#   更新对象字段：`client.chat.completions`。
@pytest.mark.asyncio
async def test_openai_chat_stream_retry_count_uses_provider_config() -> None:
    recorder = AsyncSequenceRecorder(
        [
            AsyncStream([RuntimeError("first interruption")]),
            AsyncStream([RuntimeError("second interruption")]),
            AsyncStream([]),
        ]
    )
    client = FakeOpenAIClient()
    client.chat.completions = recorder
    adapter = OpenAICompatibleAdapter(
        provider_config(
            "deepseek",
            ApiStyle.CHAT_COMPLETIONS,
            "deepseek-test",
            max_retries=1,
        ),
        client=client,
    )

    with pytest.raises(ModelAdapterError, match="second interruption"):
        await adapter.complete_stream(
            ModelRequest(messages=(Message(role=MessageRole.USER, content="hello"),)),
            on_text_delta=capture_deltas([]),
        )

    assert recorder.call_count == 2


# 函数说明：test_anthropic_adapter_separates_system_and_tool_messages
# 用途：回归验证回归测试与测试辅助中的
# `anthropic_adapter_separates_system_and_tool_messages` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` →
# `FakeAnthropicClient` → `AnthropicAdapter` → `provider_config` → `adapter.complete` →
# `ModelRequest`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`client.messages.kwargs['system'] == 'Be concise.'`。
#   验证条件：
# `client.messages.kwargs['messages'] == [{'role': 'user', 'content': 'Search.'}]`。
#   验证条件：`response.message.content == 'Calling '`。
#   验证条件：`response.message.tool_calls[0].name == 'search'`。
@pytest.mark.asyncio
async def test_anthropic_adapter_separates_system_and_tool_messages() -> None:
    result = SimpleNamespace(
        id="msg_1",
        model="claude-test",
        content=[
            SimpleNamespace(type="text", text="Calling "),
            SimpleNamespace(
                type="tool_use",
                id="tool_1",
                name="search",
                input={"query": "MuHarness"},
            ),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=5,
            cache_read_input_tokens=7,
            cache_creation_input_tokens=3,
        ),
    )
    client = FakeAnthropicClient(result)
    adapter = AnthropicAdapter(
        provider_config(
            "anthropic",
            ApiStyle.ANTHROPIC_MESSAGES,
            "claude-test",
        ),
        client=client,
    )

    response = await adapter.complete(
        ModelRequest(
            messages=(
                Message(role=MessageRole.SYSTEM, content="Be concise."),
                Message(role=MessageRole.USER, content="Search."),
            )
        )
    )

    assert client.messages.kwargs["system"] == "Be concise."
    assert client.messages.kwargs["messages"] == [
        {"role": "user", "content": "Search."}
    ]
    assert response.message.content == "Calling "
    assert response.message.tool_calls[0].name == "search"
    assert response.usage.input_tokens == 22
    assert response.usage.total_tokens == 27
    assert response.usage.cached_input_tokens == 7
    assert response.usage.uncached_input_tokens == 15
    assert response.usage.cache_write_input_tokens == 3


# 函数说明：test_anthropic_adapter_groups_parallel_tool_results
# 用途：回归验证回归测试与测试辅助中的 `anthropic_adapter_groups_parallel_tool_results`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` →
# `FakeAnthropicClient` → `AnthropicAdapter` → `provider_config` → `ToolCall` →
# `adapter.complete`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`client.messages.kwargs['messages'] == [{'role': 'assistant', 'content': [
# {'type': 'tool_use', 'id': 'call_1', 'name': 'read_file'…`。
@pytest.mark.asyncio
async def test_anthropic_adapter_groups_parallel_tool_results() -> None:
    result = SimpleNamespace(
        id="msg_parallel",
        model="claude-test",
        content=[SimpleNamespace(type="text", text="Done")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=12, output_tokens=2),
    )
    client = FakeAnthropicClient(result)
    adapter = AnthropicAdapter(
        provider_config(
            "anthropic",
            ApiStyle.ANTHROPIC_MESSAGES,
            "claude-test",
        ),
        client=client,
    )
    first = ToolCall(id="call_1", name="read_file", arguments={"path": "a"})
    second = ToolCall(id="call_2", name="read_file", arguments={"path": "b"})

    await adapter.complete(
        ModelRequest(
            messages=(
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(first, second),
                ),
                Message(
                    role=MessageRole.TOOL,
                    tool_call_id=first.id,
                    content="first result",
                ),
                Message(
                    role=MessageRole.TOOL,
                    tool_call_id=second.id,
                    content="second result",
                ),
            )
        )
    )

    assert client.messages.kwargs["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "read_file",
                    "input": {"path": "a"},
                },
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "read_file",
                    "input": {"path": "b"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "first result",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "second result",
                },
            ],
        },
    ]


# 函数说明：test_anthropic_stream_emits_deltas_and_returns_complete_message
# 用途：回归验证回归测试与测试辅助中的
# `anthropic_stream_emits_deltas_and_returns_complete_message` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` → `AnthropicAdapter`
#  → `provider_config` → `FakeStreamingAnthropicClient` → `FakeAnthropicMessageStream` →
#  `adapter.complete_stream`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`deltas == ['完', '成']`。
#   验证条件：`response.message.content == '完成'`。
#   验证条件：`response.usage.total_tokens == 8`。
@pytest.mark.asyncio
async def test_anthropic_stream_emits_deltas_and_returns_complete_message() -> None:
    final = SimpleNamespace(
        id="msg-stream",
        model="claude-test",
        content=[SimpleNamespace(type="text", text="完成")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=6, output_tokens=2),
    )
    adapter = AnthropicAdapter(
        provider_config("anthropic", ApiStyle.ANTHROPIC_MESSAGES, "claude-test"),
        client=FakeStreamingAnthropicClient(
            FakeAnthropicMessageStream(["完", "成"], final)
        ),
    )
    deltas: list[str] = []

    response = await adapter.complete_stream(
        ModelRequest(messages=(Message(role=MessageRole.USER, content="do it"),)),
        on_text_delta=capture_deltas(deltas),
    )

    assert deltas == ["完", "成"]
    assert response.message.content == "完成"
    assert response.usage.total_tokens == 8


# 函数说明：isolated_model_api_keys
# 用途：在回归测试与测试辅助中处理 `isolated_model_api_keys`，通过 `monkeypatch.delenv`
# 完成首个内部处理步骤。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具，类型 `pytest.MonkeyPatch`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`monkeypatch.delenv`。
@pytest.fixture
def isolated_model_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (
        "OPENAI_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)


# 函数说明：test_settings_are_lazy_and_accept_dashscope_key_alias
# 用途：回归验证回归测试与测试辅助中的
# `settings_are_lazy_and_accept_dashscope_key_alias` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   isolated_model_api_keys：模型输入或配置值，类型 `None`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettings` →
# `settings.configured_providers` →
# `settings.provider_config(ModelProvider.QWEN).api_key_value` →
# `settings.provider_config` → `pytest.raises`。
# 分支与异常：
#   验证条件：`settings.configured_providers() == (ModelProvider.QWEN,)`。
#   验证条件：
# `settings.provider_config(ModelProvider.QWEN).api_key_value() == 'qwen-key'`。
#   预期异常：`pytest.raises(ProviderNotConfiguredError)`。
def test_settings_are_lazy_and_accept_dashscope_key_alias(
    isolated_model_api_keys: None,
) -> None:
    settings = ModelSettings(
        _env_file=None,
        DASHSCOPE_API_KEY="qwen-key",
    )

    assert settings.configured_providers() == (ModelProvider.QWEN,)
    assert settings.provider_config(ModelProvider.QWEN).api_key_value() == "qwen-key"
    with pytest.raises(ProviderNotConfiguredError):
        settings.provider_config(ModelProvider.OPENAI)


# 函数说明：test_chat_auto_selects_the_only_configured_provider
# 用途：回归验证回归测试与测试辅助中的 `chat_auto_selects_the_only_configured_provider`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   isolated_model_api_keys：模型输入或配置值，类型 `None`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettings` → `select_provider`。
# 分支与异常：
#   验证条件：`select_provider(settings, None) is ModelProvider.QWEN`。
def test_chat_auto_selects_the_only_configured_provider(
    isolated_model_api_keys: None,
) -> None:
    settings = ModelSettings(
        _env_file=None,
        DASHSCOPE_API_KEY="qwen-key",
    )

    assert select_provider(settings, None) is ModelProvider.QWEN
