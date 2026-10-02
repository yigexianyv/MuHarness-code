
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from ..adapter import ModelAdapter
from ..config import ProviderConfig
from ..errors import ModelAdapterError
from ..types import (
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolDefinition,
)


class OpenAICompatibleAdapter(ModelAdapter):

    # 函数说明：OpenAICompatibleAdapter.__init__
    # 用途：初始化 OpenAICompatibleAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   client：模型、HTTP 或 MCP 客户端，类型 `Any | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super` →
    # `config.api_key_value` → `AsyncOpenAI`。
    # 分支与异常：
    #   当 `config.api_style is ApiStyle.ANTHROPIC_MESSAGES` 时，抛出
    # `ValueError('Anthropic Messages requires AnthropicAdapter')`。
    # 副作用与资源：
    #   更新对象字段：`self._client`。
    def __init__(
        self,
        config: ProviderConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if config.api_style is ApiStyle.ANTHROPIC_MESSAGES:
            raise ValueError("Anthropic Messages requires AnthropicAdapter")
        super().__init__(config)

        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key_value(),
            "timeout": config.timeout_seconds,
            "max_retries": config.max_retries,
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = client or AsyncOpenAI(**client_kwargs)

    # 函数说明：OpenAICompatibleAdapter.complete
    # 用途：完成OpenAICompatibleAdapter，供模型服务商协议适配使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；按分支返回 `await self._complete_responses(request)`；
    # `await self._complete_chat(request)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._complete_responses` →
    # `self._complete_chat`。
    # 分支与异常：
    #   当 `self.config.api_style is ApiStyle.RESPONSES` 时，返回
    # `await self._complete_responses(request)`。
    #   捕获 `ModelAdapterError` 后，重新抛出原异常。
    #   捕获 `Exception` 后，转换或抛出
    # `ModelAdapterError(f'{self.provider} model request failed: {exc}')`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            if self.config.api_style is ApiStyle.RESPONSES:
                return await self._complete_responses(request)
            return await self._complete_chat(request)
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise ModelAdapterError(
                f"{self.provider} model request failed: {exc}"
            ) from exc

    # 函数说明：OpenAICompatibleAdapter.complete_stream
    # 用途：完成事件流，供模型服务商协议适配使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    #   on_text_delta：增量文本回调，类型 `Callable[[str], Awaitable[None]]`。
    #   on_reasoning_delta：增量推理文本回调，类型
    # `Callable[[str], Awaitable[None]] | None`；默认 `None`。
    # 返回：类型 `ModelResponse`；按分支返回
    # `await self._stream_responses(request, on_text_delta, attempt=attempt)`；`await
    # self._stream_chat(request, on_text_delta, on_reasoning_delta, attempt=attempt)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_StreamAttemptState` →
    # `self._stream_responses` → `self._stream_chat`。
    # 分支与异常：
    #   当 `self.config.api_style is ApiStyle.RESPONSES` 时，返回
    # `await self._stream_responses(…)`。
    #   捕获 `Exception` 后，重新抛出原异常。
    #   `can_retry` 分支在完成前置处理后跳过当前循环项。
    #   当 `isinstance(exc, ModelAdapterError)` 时，抛出 `None`。
    async def complete_stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> ModelResponse:

        retries = 0
        while True:
            attempt = _StreamAttemptState()
            try:
                if self.config.api_style is ApiStyle.RESPONSES:
                    return await self._stream_responses(
                        request,
                        on_text_delta,
                        attempt=attempt,
                    )
                return await self._stream_chat(
                    request,
                    on_text_delta,
                    on_reasoning_delta,
                    attempt=attempt,
                )
            except Exception as exc:
                can_retry = (
                    attempt.stream_opened
                    and not attempt.visible_delta_emitted
                    and retries < self.config.max_retries
                )
                if can_retry:
                    retries += 1
                    continue
                if isinstance(exc, ModelAdapterError):
                    raise
                raise ModelAdapterError(
                    f"{self.provider} model stream failed: {exc}"
                ) from exc

    # 函数说明：OpenAICompatibleAdapter.close
    # 用途：关闭OpenAICompatibleAdapter，供模型服务商协议适配使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._client.close`。
    async def close(self) -> None:
        await self._client.close()

    # 函数说明：OpenAICompatibleAdapter._complete_responses
    # 用途：完成`responses`，供模型服务商协议适配使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回
    # `_normalize_responses_response(response, self.provider)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_responses_input` →
    # `_responses_tool` → `self._client.responses.create` →
    # `_normalize_responses_response`。
    async def _complete_responses(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "input": _responses_input(request.messages),
        }
        if request.tools:
            kwargs["tools"] = [_responses_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body

        response = await self._client.responses.create(**kwargs)
        return _normalize_responses_response(response, self.provider)

    # 函数说明：OpenAICompatibleAdapter._complete_chat
    # 用途：完成`chat`，供模型服务商协议适配使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_chat_message` → `_chat_tool` →
    # `self._client.chat.completions.create` → `ToolCall` → `_parse_arguments` →
    # `ModelResponse`；另有 3 个调用点。
    async def _complete_chat(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": [_chat_message(message) for message in request.messages],
        }
        if request.tools:
            kwargs["tools"] = [_chat_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        response_message = choice.message
        tool_calls = tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
            )
            for call in (response_message.tool_calls or ())
        )

        return ModelResponse(
            id=response.id,
            provider=self.provider,
            model=response.model,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=response_message.content,
                tool_calls=tool_calls,
                reasoning=getattr(response_message, "reasoning_content", None),
            ),
            finish_reason=choice.finish_reason,
            usage=_chat_usage(getattr(response, "usage", None)),
            raw=_model_dump(response),
        )

    # 函数说明：OpenAICompatibleAdapter._stream_responses
    # 用途：在模型服务商协议适配中处理 `_stream_responses`，通过
    # `self._client.responses.create` 完成首个内部处理步骤。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    #   on_text_delta：增量文本回调，类型 `Callable[[str], Awaitable[None]]`。
    #   attempt：`attempt`输入或配置值，类型 `_StreamAttemptState`。
    # 返回：类型 `ModelResponse`；返回
    # `_normalize_responses_response(final_response, self.provider)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_responses_input` →
    # `_responses_tool` → `self._client.responses.create` → `on_text_delta` →
    # `_normalize_responses_response`。
    # 分支与异常：
    #   当 `final_response is None` 时，抛出 `ModelAdapterError(…)`。
    # 副作用与资源：
    #   更新对象字段：`attempt.stream_opened`、`attempt.visible_delta_emitted`。
    async def _stream_responses(
        self,
        request: ModelRequest,
        on_text_delta: Callable[[str], Awaitable[None]],
        *,
        attempt: _StreamAttemptState,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "input": _responses_input(request.messages),
            "stream": True,
        }
        if request.tools:
            kwargs["tools"] = [_responses_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body

        stream = await self._client.responses.create(**kwargs)
        attempt.stream_opened = True
        final_response: Any | None = None
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    attempt.visible_delta_emitted = True
                    await on_text_delta(delta)
            elif event_type == "response.completed":
                final_response = getattr(event, "response", None)

        if final_response is None:
            raise ModelAdapterError(
                f"{self.provider} response stream ended without response.completed"
            )
        return _normalize_responses_response(final_response, self.provider)

    # 函数说明：OpenAICompatibleAdapter._stream_chat
    # 用途：在模型服务商协议适配中处理 `_stream_chat`，通过
    # `self._client.chat.completions.create` 完成首个内部处理步骤。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    #   on_text_delta：增量文本回调，类型 `Callable[[str], Awaitable[None]]`。
    #   on_reasoning_delta：增量推理文本回调，类型
    # `Callable[[str], Awaitable[None]] | None`；默认 `None`。
    #   attempt：`attempt`输入或配置值，类型 `_StreamAttemptState`。
    # 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_chat_message` → `_chat_tool` →
    # `self._client.chat.completions.create` → `on_text_delta` → `on_reasoning_delta` →
    # `tool_parts.setdefault`；另有 5 个调用点。
    # 分支与异常：
    #   当 `not choices` 时，跳过当前循环项。
    #   当 `delta is None` 时，跳过当前循环项。
    # 副作用与资源：
    #   更新对象字段：`attempt.stream_opened`、`attempt.visible_delta_emitted`。
    async def _stream_chat(
        self,
        request: ModelRequest,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        *,
        attempt: _StreamAttemptState,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": [_chat_message(message) for message in request.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            kwargs["tools"] = [_chat_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body

        stream = await self._client.chat.completions.create(**kwargs)
        attempt.stream_opened = True
        response_id = ""
        response_model = request.model or self.default_model
        finish_reason: str | None = None
        usage: Any | None = None
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            response_id = getattr(chunk, "id", response_id) or response_id
            response_model = getattr(chunk, "model", response_model) or response_model
            usage = getattr(chunk, "usage", None) or usage
            choices = getattr(chunk, "choices", None) or ()
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)
                attempt.visible_delta_emitted = True
                await on_text_delta(content)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                if on_reasoning_delta is not None:
                    attempt.visible_delta_emitted = True
                    await on_reasoning_delta(reasoning)
            for call in getattr(delta, "tool_calls", None) or ():
                index = int(getattr(call, "index", 0) or 0)
                part = tool_parts.setdefault(
                    index,
                    {"id": "", "name": "", "arguments": ""},
                )
                part["id"] += getattr(call, "id", None) or ""
                function = getattr(call, "function", None)
                if function is not None:
                    part["name"] += getattr(function, "name", None) or ""
                    part["arguments"] += getattr(function, "arguments", None) or ""

        tool_calls = tuple(
            ToolCall(
                id=part["id"],
                name=part["name"],
                arguments=_parse_arguments(part["arguments"]),
            )
            for _, part in sorted(tool_parts.items())
        )
        return ModelResponse(
            id=response_id or "stream",
            provider=self.provider,
            model=response_model,
            message=Message(
                role=MessageRole.ASSISTANT,
                content="".join(text_parts) or None,
                tool_calls=tool_calls,
                reasoning="".join(reasoning_parts) or None,
            ),
            finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
            usage=_chat_usage(usage),
            raw=None,
        )


@dataclass
class _StreamAttemptState:

    stream_opened: bool = False
    visible_delta_emitted: bool = False


# 函数说明：_arguments_json
# 用途：将工具参数规范化为模型协议所需的 JSON 文本。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any] | str`。
# 返回：类型 `str`；按分支返回 `arguments`；
# `json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
# 分支与异常：
#   当 `isinstance(arguments, str)` 时，返回 `arguments`。
def _arguments_json(arguments: dict[str, Any] | str) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


# 函数说明：_normalize_responses_response
# 用途：规范化响应，供模型服务商协议适配使用。
# 参数：
#   response：模型、工具或服务返回的响应，类型 `Any`。
#   provider：模型或搜索服务商，类型 `str`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `_parse_arguments` →
# `ModelResponse` → `Message` → `_responses_reasoning` → `_responses_finish_reason`；另
# 有 2 个调用点。
def _normalize_responses_response(response: Any, provider: str) -> ModelResponse:
    tool_calls = tuple(
        ToolCall(
            id=item.call_id,
            name=item.name,
            arguments=_parse_arguments(item.arguments),
        )
        for item in response.output
        if getattr(item, "type", None) == "function_call"
    )
    return ModelResponse(
        id=response.id,
        provider=provider,
        model=response.model,
        message=Message(
            role=MessageRole.ASSISTANT,
            content=response.output_text or None,
            tool_calls=tool_calls,
            reasoning=_responses_reasoning(response),
        ),
        finish_reason=_responses_finish_reason(response, tool_calls),
        usage=_responses_usage(getattr(response, "usage", None)),
        raw=_model_dump(response),
    )


# 函数说明：_responses_reasoning
# 用途：在模型服务商协议适配中处理 `_responses_reasoning`，通过 `parts.append` 完成首个
# 内部处理步骤。
# 参数：
#   response：模型、工具或服务返回的响应，类型 `Any`。
# 返回：类型 `str | None`；返回 `''.join(parts) or None`。
def _responses_reasoning(response: Any) -> str | None:
    parts: list[str] = []
    for item in getattr(response, "reasoning", None) or ():
        summary = getattr(item, "summary", None)
        if summary:
            parts.append(summary)
    return "".join(parts) or None


# 函数说明：_parse_arguments
# 用途：解析调用参数，供模型服务商协议适配使用。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `Any`。
# 返回：类型 `dict[str, Any] | str`；按分支返回 `arguments`；`str(arguments)`；
# `parsed if isinstance(parsed, dict) else arguments`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads`。
# 分支与异常：
#   当 `isinstance(arguments, dict)` 时，返回 `arguments`。
#   当 `not isinstance(arguments, str)` 时，返回 `str(arguments)`。
#   捕获 `json.JSONDecodeError` 后，返回 `arguments`。
def _parse_arguments(arguments: Any) -> dict[str, Any] | str:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return str(arguments)
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return parsed if isinstance(parsed, dict) else arguments


# 函数说明：_responses_input
# 用途：在模型服务商协议适配中处理 `_responses_input`，通过 `items.append` 完成首个内部
# 处理步骤。
# 参数：
#   messages：本次处理的消息序列，类型 `tuple[Message, ...]`。
# 返回：类型 `list[dict[str, Any]]`；返回 `items`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_arguments_json`。
# 分支与异常：
#   `message.role is MessageRole.TOOL` 分支在完成前置处理后跳过当前循环项。
def _responses_input(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role is MessageRole.TOOL:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content or "",
                }
            )
            continue

        if message.content is not None:
            items.append(
                {
                    "role": message.role.value,
                    "content": message.content,
                }
            )
        for call in message.tool_calls:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": _arguments_json(call.arguments),
                }
            )
    return items


# 函数说明：_chat_message
# 用途：处理模型服务商协议适配中的 `_chat_message` 数据；结果及边界条件见下方说明。
# 参数：
#   message：单条消息或通知，类型 `Message`。
# 返回：类型 `dict[str, Any]`；返回 `result`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_arguments_json`。
def _chat_message(message: Message) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.name is not None:
        result["name"] = message.name
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": _arguments_json(call.arguments),
                },
            }
            for call in message.tool_calls
        ]
    return result


# 函数说明：_responses_tool
# 用途：处理模型服务商协议适配中的 `_responses_tool` 数据；结果及边界条件见下方说明。
# 参数：
#   tool：目标工具实例，类型 `ToolDefinition`。
# 返回：类型 `dict[str, Any]`；返回 `result`。
def _responses_tool(tool: ToolDefinition) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        result["strict"] = tool.strict
    return result


# 函数说明：_chat_tool
# 用途：处理模型服务商协议适配中的 `_chat_tool` 数据；结果及边界条件见下方说明。
# 参数：
#   tool：目标工具实例，类型 `ToolDefinition`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `type`、`function`。
def _chat_tool(tool: ToolDefinition) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


# 函数说明：_responses_finish_reason
# 用途：结束原因，供模型服务商协议适配使用。
# 参数：
#   response：模型、工具或服务返回的响应，类型 `Any`。
#   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`。
# 返回：类型 `str | None`；按分支返回 `'tool_calls'`；
# `getattr(details, 'reason', None) or 'incomplete'`；
# `getattr(response, 'status', None) or 'stop'`。
# 分支与异常：
#   当 `tool_calls` 时，返回 `'tool_calls'`。
#   `getattr(response, 'status', None) == 'incomplete'` 分支在完成前置处理后返回
# `getattr(details, 'reason', None) or 'incomplete'`。
def _responses_finish_reason(
    response: Any,
    tool_calls: tuple[ToolCall, ...],
) -> str | None:
    if tool_calls:
        return "tool_calls"
    if getattr(response, "status", None) == "incomplete":
        details = getattr(response, "incomplete_details", None)
        return getattr(details, "reason", None) or "incomplete"
    return getattr(response, "status", None) or "stop"


# 函数说明：_responses_usage
# 用途：处理模型服务商协议适配中的 `_responses_usage` 数据；结果及边界条件见下方说明。
# 参数：
#   usage：模型调用用量统计，类型 `Any | None`。
# 返回：类型 `ModelUsage`；按分支返回 `ModelUsage()`；`ModelUsage(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelUsage` → `_openai_cache_usage`。
# 分支与异常：
#   当 `usage is None` 时，返回 `ModelUsage()`。
def _responses_usage(usage: Any | None) -> ModelUsage:
    if usage is None:
        return ModelUsage()
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cached, cache_write, cache_reported = _openai_cache_usage(
        usage,
        details_name="input_tokens_details",
    )
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(
            getattr(usage, "total_tokens", input_tokens + output_tokens)
            or input_tokens + output_tokens
        ),
        cached_input_tokens=cached if cache_reported else None,
        uncached_input_tokens=(
            max(0, input_tokens - cached) if cache_reported else None
        ),
        cache_read_input_tokens=cached if cache_reported else None,
        cache_write_input_tokens=cache_write,
        model_calls=1,
    )


# 函数说明：_chat_usage
# 用途：处理模型服务商协议适配中的 `_chat_usage` 数据；结果及边界条件见下方说明。
# 参数：
#   usage：模型调用用量统计，类型 `Any | None`；读取键 `prompt_cache_miss_tokens`。
# 返回：类型 `ModelUsage`；按分支返回 `ModelUsage()`；`ModelUsage(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelUsage` → `_openai_cache_usage` →
#  `_optional_int`。
# 分支与异常：
#   当 `usage is None` 时，返回 `ModelUsage()`。
def _chat_usage(usage: Any | None) -> ModelUsage:
    if usage is None:
        return ModelUsage()
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cached, cache_write, cache_reported = _openai_cache_usage(
        usage,
        details_name="prompt_tokens_details",
    )
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(
            getattr(usage, "total_tokens", input_tokens + output_tokens)
            or input_tokens + output_tokens
        ),
        cached_input_tokens=cached if cache_reported else None,
        uncached_input_tokens=(
            _optional_int(usage, "prompt_cache_miss_tokens")
            if _optional_int(usage, "prompt_cache_miss_tokens") is not None
            else (max(0, input_tokens - cached) if cache_reported else None)
        ),
        cache_read_input_tokens=cached if cache_reported else None,
        cache_write_input_tokens=cache_write,
        model_calls=1,
    )


# 函数说明：_openai_cache_usage
# 用途：处理模型服务商协议适配中的 `_openai_cache_usage` 数据；结果及边界条件见下方说明
# 。
# 参数：
#   usage：模型调用用量统计，类型 `Any`；读取键 `prompt_cache_hit_tokens`、
# `cached_tokens`。
#   details_name：传给 `getattr` 的输入，类型 `str`。
# 返回：类型 `tuple[int, int | None, bool]`；返回
# `(cached_value or 0, cache_write, cached_value is not None)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_optional_int` → `next`。
def _openai_cache_usage(
    usage: Any,
    *,
    details_name: str,
) -> tuple[int, int | None, bool]:

    deepseek_hit = _optional_int(usage, "prompt_cache_hit_tokens")
    details = getattr(usage, details_name, None)
    nested_cached = _optional_int(details, "cached_tokens")
    direct_cached = _optional_int(usage, "cached_tokens")
    cached_value = next(
        (
            value
            for value in (deepseek_hit, nested_cached, direct_cached)
            if value is not None
        ),
        None,
    )
    cache_write = _optional_int(details, "cache_creation_input_tokens")
    return cached_value or 0, cache_write, cached_value is not None


# 函数说明：_optional_int
# 用途：解析可选整数；无有效值时按实现返回空值。
# 参数：
#   value：待校验、规范化或转换的值，类型 `Any`。
#   field：待校验的字段名，类型 `str`。
# 返回：类型 `int | None`；按分支返回 `None`；`max(0, int(raw))`。
# 分支与异常：
#   当 `value is None` 时，返回 `None`。
#   当 `raw is None` 时，返回 `None`。
def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    raw = value.get(field) if isinstance(value, dict) else getattr(value, field, None)
    if raw is None:
        return None
    return max(0, int(raw))


# 函数说明：_model_dump
# 用途：将模型响应转换为可供协议适配处理的字典数据。
# 参数：
#   value：待校验、规范化或转换的值，类型 `Any`。
# 返回：类型 `dict[str, Any] | None`；按分支返回 `value.model_dump(mode='json')`；
# `value if isinstance(value, dict) else None`。
# 分支与异常：
#   当 `hasattr(value, 'model_dump')` 时，返回 `value.model_dump(mode='json')`。
def _model_dump(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else None
