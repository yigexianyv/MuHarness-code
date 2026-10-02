
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import tiktoken
from tiktoken import Encoding

from app.models.types import Message, ToolDefinition

DEFAULT_ENCODING = "cl100k_base"

DEFAULT_FAMILY_FACTORS: dict[str, float] = {
    "openai": 1.0,
    "qwen": 1.2,
    "deepseek": 1.2,
    "anthropic": 1.15,
    "other": 1.25,
}


class TokenEstimator:

    # 函数说明：TokenEstimator.__init__
    # 用途：初始化 TokenEstimator；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   default_encoding：`default_encoding`输入或配置值，类型 `str`；默认
    # `DEFAULT_ENCODING`。
    #   factors：`factors`输入或配置值，类型 `dict[str, float] | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._default_encoding`、`self._factors`、`self._cache`。
    def __init__(
        self,
        default_encoding: str = DEFAULT_ENCODING,
        factors: dict[str, float] | None = None,
    ) -> None:
        self._default_encoding = default_encoding
        self._factors = {**DEFAULT_FAMILY_FACTORS, **(factors or {})}
        self._cache: dict[str, Encoding] = {}

    # 函数说明：TokenEstimator.estimate_text
    # 用途：估算文本，供模型上下文与输入预算使用。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    # 返回：类型 `int`；按分支返回 `0`；`base`；`math.ceil(base * factor)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._encoding(model).encode` →
    # `self._encoding` → `self.factor_for` → `math.ceil`。
    # 分支与异常：
    #   当 `not text` 时，返回 `0`。
    #   当 `factor <= 1.0` 时，返回 `base`。
    def estimate_text(
        self,
        text: str,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        if not text:
            return 0
        base = len(self._encoding(model).encode(text, disallowed_special=()))
        factor = self.factor_for(provider, model)
        if factor <= 1.0:
            return base
        return math.ceil(base * factor)

    # 函数说明：TokenEstimator.estimate_messages
    # 用途：估算消息序列，供模型上下文与输入预算使用。
    # 参数：
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    # 返回：类型 `int`；返回 `total`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.estimate_text` →
    # `json.dumps`。
    def estimate_messages(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        total = 0
        for message in messages:
            total += 3
            if message.role is not None:
                total += self.estimate_text(
                    message.role.value,
                    model=model,
                    provider=provider,
                )
            if message.content:
                total += self.estimate_text(
                    message.content,
                    model=model,
                    provider=provider,
                )
            if message.name:
                total += 1 + self.estimate_text(
                    message.name,
                    model=model,
                    provider=provider,
                )
            if message.tool_call_id:
                total += 1 + self.estimate_text(
                    message.tool_call_id,
                    model=model,
                    provider=provider,
                )
            for call in message.tool_calls:
                total += 4
                total += self.estimate_text(
                    call.name,
                    model=model,
                    provider=provider,
                )
                arguments: Any = call.arguments
                if not isinstance(arguments, str):
                    arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                total += self.estimate_text(
                    arguments,
                    model=model,
                    provider=provider,
                )
        return total

    # 函数说明：TokenEstimator.estimate_tools
    # 用途：估算工具集合，供模型上下文与输入预算使用。
    # 参数：
    #   tools：可用工具定义或工具实例集合，类型 `Sequence[ToolDefinition]`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    # 返回：类型 `int`；返回 `total`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.estimate_text` →
    # `json.dumps`。
    def estimate_tools(
        self,
        tools: Sequence[ToolDefinition],
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        total = 0
        for tool in tools:
            total += 5
            total += self.estimate_text(
                tool.name,
                model=model,
                provider=provider,
            )
            total += self.estimate_text(
                tool.description,
                model=model,
                provider=provider,
            )
            total += self.estimate_text(
                json.dumps(
                    tool.parameters,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                model=model,
                provider=provider,
            )
        return total

    # 函数说明：TokenEstimator.estimate_request
    # 用途：估算请求，供模型上下文与输入预算使用。
    # 参数：
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   tools：可用工具定义或工具实例集合，类型 `Sequence[ToolDefinition]`；默认 `()`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    # 返回：类型 `int`；返回 `self.estimate_messages(messages, model=model, provider=
    # provider) + self.estimate_tools(…`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.estimate_messages` →
    # `self.estimate_tools`。
    def estimate_request(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition] = (),
        model: str | None = None,
        provider: str | None = None,
    ) -> int:

        return self.estimate_messages(
            messages,
            model=model,
            provider=provider,
        ) + self.estimate_tools(tools, model=model, provider=provider)

    # 函数说明：TokenEstimator.factor_for
    # 用途：在模型上下文与输入预算中处理 `factor_for`，通过 `self._factors.get` 完成首个
    # 内部处理步骤。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str | None`。
    #   model：模型名称，类型 `str | None`。
    # 返回：类型 `float`；返回 `self._factors.get(family, self._factors['other'])`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`model_family`。
    def factor_for(self, provider: str | None, model: str | None) -> float:

        family = model_family(provider, model)
        return self._factors.get(family, self._factors["other"])

    # 函数说明：TokenEstimator._encoding
    # 用途：在模型上下文与输入预算中处理 `_encoding`，通过 `self._encoding_for` 完成首个
    # 内部处理步骤。
    # 参数：
    #   model：模型名称，类型 `str | None`。
    # 返回：类型 `Encoding`；返回 `self._cache[cache_key]`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._encoding_for`。
    def _encoding(self, model: str | None) -> Encoding:
        cache_key = model or self._default_encoding
        if cache_key not in self._cache:
            self._cache[cache_key] = self._encoding_for(model)
        return self._cache[cache_key]

    # 函数说明：TokenEstimator._encoding_for
    # 用途：在模型上下文与输入预算中处理 `_encoding_for`，通过
    # `tiktoken.encoding_for_model` 完成首个内部处理步骤。
    # 参数：
    #   model：模型名称，类型 `str | None`。
    # 返回：类型 `Encoding`；按分支返回 `tiktoken.encoding_for_model(model)`；
    # `tiktoken.get_encoding(self._default_encoding)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`tiktoken.encoding_for_model` →
    # `tiktoken.get_encoding`。
    # 分支与异常：
    #   捕获 `KeyError` 后，忽略该异常并继续当前流程。
    def _encoding_for(self, model: str | None) -> Encoding:
        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                pass
        return tiktoken.get_encoding(self._default_encoding)


# 函数说明：model_family
# 用途：在模型上下文与输入预算中处理 `model_family`，通过 `provider.strip().lower` 完成
# 首个内部处理步骤。
# 参数：
#   provider：模型或搜索服务商，类型 `str | None`。
#   model：模型名称，类型 `str | None`。
# 返回：类型 `str`；按分支返回 `'openai'`；`'qwen'`；`'deepseek'`；`'anthropic'` 等 5 种
# 表达式。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`provider.strip().lower` →
# `(model or '').lower` → `lowered.startswith`。
# 分支与异常：
#   当 `name == 'openai'` 时，返回 `'openai'`。
#   当 `name == 'qwen'` 时，返回 `'qwen'`。
#   当 `name == 'deepseek'` 时，返回 `'deepseek'`。
#   当 `name == 'anthropic'` 时，返回 `'anthropic'`。
def model_family(provider: str | None, model: str | None) -> str:

    if provider:
        name = provider.strip().lower()
        if name == "openai":
            return "openai"
        if name == "qwen":
            return "qwen"
        if name == "deepseek":
            return "deepseek"
        if name == "anthropic":
            return "anthropic"
    lowered = (model or "").lower()
    if lowered.startswith(("gpt-", "o1", "o3", "o4")) or "openai" in lowered:
        return "openai"
    if "qwen" in lowered:
        return "qwen"
    if "deepseek" in lowered:
        return "deepseek"
    if "claude" in lowered or "anthropic" in lowered:
        return "anthropic"
    return "other"


_DEFAULT = TokenEstimator()


# 函数说明：default_token_estimator
# 用途：返回 `_DEFAULT`，提供 模型上下文与输入预算 的派生值。
# 返回：类型 `TokenEstimator`；返回 `_DEFAULT`。
def default_token_estimator() -> TokenEstimator:

    return _DEFAULT


__all__ = [
    "DEFAULT_ENCODING",
    "DEFAULT_FAMILY_FACTORS",
    "TokenEstimator",
    "default_token_estimator",
    "model_family",
]
