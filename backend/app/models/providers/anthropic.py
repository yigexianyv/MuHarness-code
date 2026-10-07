
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from anthropic import AsyncAnthropic

from ..adapter import ModelAdapter
from ..config import ProviderConfig
from ..errors import ModelAdapterError, UnsupportedMessageError
from ..types import (
    RUNTIME_NOTICE_NAME,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolDefinition,
)


class AnthropicAdapter(ModelAdapter):
    # 函数说明：AnthropicAdapter.__init__
    # 用途：初始化 AnthropicAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   client：模型、HTTP 或 MCP 客户端，类型 `Any | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super` →
    # `config.api_key_value` → `AsyncAnthropic`。
    # 副作用与资源：
    #   更新对象字段：`self._client`。
    def __init__(
        self,
        config: ProviderConfig,
        *,
        client: Any | None = None,
    ) -> None:
        super().__init__(config)
        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key_value(),
            "timeout": config.timeout_seconds,
            "max_retries": config.max_retries,
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = client or AsyncAnthropic(**client_kwargs)

    # 函数说明：AnthropicAdapter.complete
    # 用途：完成AnthropicAdapter，供模型服务商协议适配使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回
    # `_normalize_anthropic_response(response, self.provider)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._request_kwargs` →
    # `self._client.messages.create` → `_normalize_anthropic_response`。
    # 分支与异常：
    #   捕获 `Exception` 后，转换或抛出
    # `ModelAdapterError(f'{self.provider} model request failed: {exc}')`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs = self._request_kwargs(request)

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ModelAdapterError(
                f"{self.provider} model request failed: {exc}"
            ) from exc

        return _normalize_anthropic_response(response, self.provider)

    # 函数说明：AnthropicAdapter.complete_stream
    # 用途：完成事件流，供模型服务商协议适配使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    #   on_text_delta：增量文本回调，类型 `Callable[[str], Awaitable[None]]`。
    #   on_reasoning_delta：增量推理文本回调，类型
    # `Callable[[str], Awaitable[None]] | None`；默认 `None`。
    # 返回：类型 `ModelResponse`；返回
    # `_normalize_anthropic_response(response, self.provider)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._request_kwargs` →
    # `self._client.messages.stream` → `on_text_delta` → `stream.get_final_message` →
    # `_normalize_anthropic_response`。
    # 分支与异常：
    #   捕获 `Exception` 后，转换或抛出
    # `ModelAdapterError(f'{self.provider} model stream failed: {exc}')`。
    async def complete_stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> ModelResponse:
        kwargs = self._request_kwargs(request)
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if text:
                        await on_text_delta(text)
                response = await stream.get_final_message()
        except Exception as exc:
            raise ModelAdapterError(
                f"{self.provider} model stream failed: {exc}"
            ) from exc
        return _normalize_anthropic_response(response, self.provider)

    # 函数说明：AnthropicAdapter._request_kwargs
    # 用途：处理模型服务商协议适配中的 `_request_kwargs` 数据；结果及边界条件见下方说明
    # 。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `dict[str, Any]`；返回 `kwargs`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_anthropic_messages` →
    # `_anthropic_tool` → `_anthropic_tool_choice`。
    def _request_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        system, messages = _anthropic_messages(request.messages)
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "max_tokens": (
                request.max_output_tokens or self.config.default_max_output_tokens
            ),
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if request.tools:
            kwargs["tools"] = [_anthropic_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = _anthropic_tool_choice(request.tool_choice)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body
        return kwargs

    # 函数说明：AnthropicAdapter.close
    # 用途：关闭AnthropicAdapter，供模型服务商协议适配使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._client.close`。
    async def close(self) -> None:
        await self._client.close()


# 函数说明：_anthropic_messages
# 用途：在模型服务商协议适配中处理 `_anthropic_messages`，通过 `system_parts.append` 完
# 成首个内部处理步骤。
# 参数：
#   messages：本次处理的消息序列，类型 `tuple[Message, ...]`。
# 返回：类型 `tuple[str | None, list[dict[str, Any]]]`；返回
# `('\n\n'.join(system_parts) or None, result)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`flush_tool_results` →
# `_arguments_dict`。
# 分支与异常：
#   `message.role is MessageRole.SYSTEM` 分支在完成前置处理后跳过当前循环项。
#   `message.role is MessageRole.TOOL` 分支在完成前置处理后跳过当前循环项。
#   当 `not message.tool_call_id` 时，抛出 `UnsupportedMessageError(…)`。
def _anthropic_messages(
    messages: tuple[Message, ...],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    result: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    # 函数说明：_anthropic_messages.flush_tool_results
    # 用途：在模型服务商协议适配中处理 `flush_tool_results`，通过 `result.append` 完成首
    # 个内部处理步骤。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`pending_tool_results.clear`。
    # 分支与异常：
    #   当 `not pending_tool_results` 时，返回 `None`。
    # 闭包依赖：从外层读取 `pending_tool_results`、`result`。
    def flush_tool_results() -> None:
        if not pending_tool_results:
            return
        result.append({"role": "user", "content": list(pending_tool_results)})
        pending_tool_results.clear()

    for message in messages:
        if message.role is MessageRole.SYSTEM:
            if message.name == RUNTIME_NOTICE_NAME:
                # 运行时提醒原位发送（并入当前 user 轮），不改顶部 system，保住缓存前缀。
                if message.content:
                    _append_runtime_notice(
                        result, pending_tool_results, message.content
                    )
                continue
            if message.content:
                system_parts.append(message.content)
            continue

        if message.role is MessageRole.TOOL:
            if not message.tool_call_id:
                raise UnsupportedMessageError(
                    "Anthropic tool results require tool_call_id."
                )
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content or "",
                }
            )
            continue

        flush_tool_results()
        content: str | list[dict[str, Any]]
        if message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": _arguments_dict(call.arguments),
                }
                for call in message.tool_calls
            )
            content = blocks
        else:
            content = message.content or ""

        result.append({"role": message.role.value, "content": content})

    flush_tool_results()
    return "\n\n".join(system_parts) or None, result


# 函数说明：_normalize_anthropic_response
# 用途：规范化响应，供模型服务商协议适配使用。
# 参数：
#   response：模型、工具或服务返回的响应，类型 `Any`。
#   provider：模型或搜索服务商，类型 `str`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `_optional_usage_int` →
# `ModelResponse` → `Message` → `ModelUsage` → `_model_dump`。
def _normalize_anthropic_response(response: Any, provider: str) -> ModelResponse:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "thinking":
            thinking = getattr(block, "thinking", None)
            if thinking:
                reasoning_parts.append(thinking)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                )
            )

    base_input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
    cache_read = _optional_usage_int(response.usage, "cache_read_input_tokens")
    cache_write = _optional_usage_int(
        response.usage,
        "cache_creation_input_tokens",
    )
    cache_reported = cache_read is not None or cache_write is not None
    input_tokens = (
        base_input_tokens + (cache_read or 0) + (cache_write or 0)
        if cache_reported
        else base_input_tokens
    )
    return ModelResponse(
        id=response.id,
        provider=provider,
        model=response.model,
        message=Message(
            role=MessageRole.ASSISTANT,
            content="".join(text_parts) or None,
            tool_calls=tuple(tool_calls),
            reasoning="".join(reasoning_parts) or None,
        ),
        finish_reason=response.stop_reason,
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_input_tokens=(cache_read or 0) if cache_reported else None,
            uncached_input_tokens=(
                base_input_tokens + (cache_write or 0)
                if cache_reported
                else None
            ),
            cache_read_input_tokens=(cache_read or 0) if cache_reported else None,
            cache_write_input_tokens=cache_write,
            model_calls=1,
        ),
        raw=_model_dump(response),
    )


# 函数说明：_optional_usage_int
# 用途：解析可选模型用量字段，保留未知用量的空值语义。
# 参数：
#   usage：模型调用用量统计，类型 `Any`。
#   field：待校验的字段名，类型 `str`。
# 返回：类型 `int | None`；按分支返回 `None`；`max(0, int(value))`。
# 分支与异常：
#   当 `value is None` 时，返回 `None`。
def _optional_usage_int(usage: Any, field: str) -> int | None:
    value = getattr(usage, field, None)
    if value is None:
        return None
    return max(0, int(value))


# 函数说明：_arguments_dict
# 用途：将工具参数规范化为模型协议所需的字典对象。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any] | str`。
# 返回：类型 `dict[str, Any]`；按分支返回 `arguments`；`parsed`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads`。
# 分支与异常：
#   当 `isinstance(arguments, dict)` 时，返回 `arguments`。
#   捕获 `json.JSONDecodeError` 后，转换或抛出
# `UnsupportedMessageError('Anthropic tool arguments must be a JSON object.')`。
#   当 `not isinstance(parsed, dict)` 时，抛出 `UnsupportedMessageError(…)`。
def _arguments_dict(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise UnsupportedMessageError(
            "Anthropic tool arguments must be a JSON object."
        ) from exc
    if not isinstance(parsed, dict):
        raise UnsupportedMessageError("Anthropic tool arguments must be a JSON object.")
    return parsed


# 函数说明：_anthropic_tool
# 用途：返回 `{'name': tool.name, 'description': tool.description, 'input_schema': tool.
# parameters}`，提供 模型服务商协议适配 的派生值。
# 参数：
#   tool：目标工具实例，类型 `ToolDefinition`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `name`、`description`、`input_schema`。
def _anthropic_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


# 函数说明：_anthropic_tool_choice
# 用途：处理模型服务商协议适配中的 `_anthropic_tool_choice` 数据；结果及边界条件见下方说
# 明。
# 参数：
#   tool_choice：工具输入或配置值，类型 `str`。
# 返回：类型 `dict[str, Any]`；按分支返回 `{'type': normalized}`；`{'type': 'any'}`；
# `{'type': 'tool', 'name': tool_choice}`。
# 分支与异常：
#   `tool_choice in {'auto', 'none', 'any'}` 分支在完成前置处理后返回
# `{'type': normalized}`。
#   当 `tool_choice == 'required'` 时，返回 `{'type': 'any'}`。
def _anthropic_tool_choice(tool_choice: str) -> dict[str, Any]:
    if tool_choice in {"auto", "none", "any"}:
        normalized = "any" if tool_choice == "required" else tool_choice
        return {"type": normalized}
    if tool_choice == "required":
        return {"type": "any"}
    return {"type": "tool", "name": tool_choice}


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


# 标明来源，避免模型把运行时提醒当成用户本人说的话。
RUNTIME_NOTICE_SOURCE = "[来源：MuHarness 应用的运行时提醒，非用户输入]"


def _append_runtime_notice(
    result: list[dict[str, Any]],
    pending_tool_results: list[dict[str, Any]],
    content: str,
) -> None:
    text = (
        f"<system-reminder>\n{RUNTIME_NOTICE_SOURCE}\n"
        f"{content}\n</system-reminder>"
    )
    # 工具循环中：并进最后一个 tool_result 的内容末尾，不新增 text 块。
    # DeepSeek 等供应商把含 text 块的 user 轮视为“新用户消息”，会丢弃此前拼接的
    # 思考内容，导致前文重新渲染、缓存整体失效（实测命中率跌到 17%）。
    if pending_tool_results:
        last = pending_tool_results[-1]
        previous = last.get("content") or ""
        if isinstance(previous, str):
            last["content"] = f"{previous}\n\n{text}" if previous else text
        else:
            last["content"] = [*previous, {"type": "text", "text": text}]
        return
    block = {"type": "text", "text": text}
    if result and result[-1]["role"] == "user":
        previous = result[-1]["content"]
        if isinstance(previous, str):
            previous = [{"type": "text", "text": previous}] if previous else []
        result[-1]["content"] = [*previous, block]
        return
    result.append({"role": "user", "content": [block]})
