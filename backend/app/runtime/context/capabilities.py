
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import StrEnum

from app.models.config import ModelSettings
from app.models.types import ModelProvider

from .config import ContextSettings

logger = logging.getLogger("muharness.context.capabilities")


class CapabilitySource(StrEnum):

    OVERRIDE = "override"
    BUILTIN = "builtin"
    PROVIDER_DEFAULT = "provider_default"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ModelCapabilities:

    provider: str
    model: str
    context_window: int
    max_output_tokens: int
    source: CapabilitySource

    # 函数说明：ModelCapabilities.__post_init__
    # 用途：在模型上下文与输入预算中处理 `__post_init__`，通过 `self.provider.strip` 完
    # 成首个内部处理步骤。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `not self.provider.strip()` 时，抛出 `ValueError('provider cannot be empty')`
    # 。
    #   当 `not self.model.strip()` 时，抛出 `ValueError('model cannot be empty')`。
    #   当 `self.context_window <= 0` 时，抛出
    # `ValueError('context_window must be greater than zero')`。
    #   当 `self.max_output_tokens <= 0` 时，抛出
    # `ValueError('max_output_tokens must be greater than zero')`。
    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider cannot be empty")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.context_window <= 0:
            raise ValueError("context_window must be greater than zero")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")


_BUILTIN_MODELS: dict[tuple[str, str], tuple[int, int]] = {
    ("openai", "gpt-5.4-mini"): (200_000, 16_384),
    ("openai", "gpt-4o-mini"): (128_000, 16_384),
    ("qwen", "qwen3.7-plus"): (1_000_000, 65_536),
    ("deepseek", "deepseek-v4-flash"): (1_048_576, 393_216),
    ("anthropic", "claude-sonnet-4-6"): (200_000, 16_384),
}

_PROVIDER_DEFAULT_MAX_OUTPUT: dict[str, int] = {
    "openai": 16_384,
    "qwen": 65_536,
    "deepseek": 393_216,
    "anthropic": 16_384,
}

FALLBACK_CONTEXT_WINDOW = 32_768
FALLBACK_MAX_OUTPUT_TOKENS = 4_096


class ModelCapabilityRegistry:

    # 函数说明：ModelCapabilityRegistry.__init__
    # 用途：初始化 ModelCapabilityRegistry；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   builtin：`builtin`输入或配置值，类型
    # `dict[tuple[str, str], ModelCapabilities] | None`；默认 `None`。
    #   provider_defaults：服务商输入或配置值，类型
    # `dict[str, ModelCapabilities] | None`；默认 `None`。
    #   fallback：`fallback`输入或配置值，类型 `ModelCapabilities | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelCapabilities`。
    # 副作用与资源：
    #   更新对象字段：`self._overrides`、`self._builtin`、`self._provider_defaults`、
    # `self._fallback`。
    def __init__(
        self,
        *,
        builtin: dict[tuple[str, str], ModelCapabilities] | None = None,
        provider_defaults: dict[str, ModelCapabilities] | None = None,
        fallback: ModelCapabilities | None = None,
    ) -> None:
        self._overrides: dict[tuple[str, str], ModelCapabilities] = {}
        self._builtin = {**(builtin or {})}
        self._provider_defaults = {**(provider_defaults or {})}
        self._fallback = fallback or ModelCapabilities(
            provider="*",
            model="*",
            context_window=FALLBACK_CONTEXT_WINDOW,
            max_output_tokens=FALLBACK_MAX_OUTPUT_TOKENS,
            source=CapabilitySource.FALLBACK,
        )

    # 函数说明：ModelCapabilityRegistry.register_override
    # 用途：注册`override`，供模型上下文与输入预算使用。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    #   model：模型名称，类型 `str`。
    #   context_window：上下文输入或配置值，类型 `int | None`；默认 `None`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `ModelCapabilities`；返回 `capabilities`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.lookup` →
    # `ModelCapabilities`。
    def register_override(
        self,
        provider: str,
        model: str,
        *,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelCapabilities:

        base = self.lookup(provider, model)
        capabilities = ModelCapabilities(
            provider=provider,
            model=model,
            context_window=(
                context_window
                if context_window is not None
                else base.context_window
            ),
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else base.max_output_tokens
            ),
            source=CapabilitySource.OVERRIDE,
        )
        self._overrides[(provider, model)] = capabilities
        return capabilities

    # 函数说明：ModelCapabilityRegistry.lookup
    # 用途：查找ModelCapabilityRegistry，供模型上下文与输入预算使用。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str | None`。
    #   model：模型名称，类型 `str | None`。
    # 返回：类型 `ModelCapabilities`；按分支返回 `self._overrides[key]`；
    # `self._builtin[key]`；
    # `replace(default, provider=provider or '*', model=model or '*')`；
    # `replace(self._fallback, provider=provider or '*', model=model or '*')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`replace` → `logger.warning`。
    # 分支与异常：
    #   当 `key in self._overrides` 时，返回 `self._overrides[key]`。
    #   当 `key in self._builtin` 时，返回 `self._builtin[key]`。
    #   当 `default is not None` 时，返回 `replace(…)`。
    def lookup(self, provider: str | None, model: str | None) -> ModelCapabilities:

        key = (provider, model)
        if key in self._overrides:
            return self._overrides[key]
        if key in self._builtin:
            return self._builtin[key]
        default = self._provider_defaults.get(provider or "")
        if default is not None:
            return replace(
                default,
                provider=provider or "*",
                model=model or "*",
            )
        logger.warning(
            "Unknown model capability; using conservative fallback "
            "provider=%s model=%s",
            provider,
            model,
        )
        return replace(
            self._fallback,
            provider=provider or "*",
            model=model or "*",
        )


# 函数说明：build_model_capability_registry
# 用途：构建模型，供模型上下文与输入预算使用。
# 参数：
#   model_settings：传给 `_resolve_override_target` 的输入，类型 `ModelSettings | None`
# ；默认 `None`。
#   context_settings：上下文设置输入或配置值，类型 `ContextSettings | None`；默认 `None`
# 。
# 返回：类型 `ModelCapabilityRegistry`；返回 `registry`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextSettings` →
# `ModelCapabilities` → `_window_for` → `ModelCapabilityRegistry` →
# `_resolve_override_target` → `registry.register_override`。
def build_model_capability_registry(
    model_settings: ModelSettings | None = None,
    context_settings: ContextSettings | None = None,
) -> ModelCapabilityRegistry:

    settings = context_settings or ContextSettings()
    builtin: dict[tuple[str, str], ModelCapabilities] = {}
    for (provider, model), (window, max_output) in _BUILTIN_MODELS.items():
        builtin[(provider, model)] = ModelCapabilities(
            provider=provider,
            model=model,
            context_window=window,
            max_output_tokens=max_output,
            source=CapabilitySource.BUILTIN,
        )
    provider_defaults = {
        name: ModelCapabilities(
            provider=name,
            model="*",
            context_window=_window_for(name, settings),
            max_output_tokens=_PROVIDER_DEFAULT_MAX_OUTPUT.get(name, 4_096),
            source=CapabilitySource.PROVIDER_DEFAULT,
        )
        for name in _PROVIDER_DEFAULT_MAX_OUTPUT
    }
    registry = ModelCapabilityRegistry(
        builtin=builtin,
        provider_defaults=provider_defaults,
    )

    target = _resolve_override_target(model_settings, settings)
    if target is not None and (
        settings.context_window_override is not None
        or settings.max_output_tokens_override is not None
    ):
        provider, model = target
        registry.register_override(
            provider,
            model,
            context_window=settings.context_window_override,
            max_output_tokens=settings.max_output_tokens_override,
        )
    return registry


# 函数说明：_window_for
# 用途：处理模型上下文与输入预算中的 `_window_for` 数据；结果及边界条件见下方说明。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   settings：业务或模型设置，类型 `ContextSettings`。
# 返回：类型 `int`；按分支返回 `settings.context_window_openai`；
# `settings.context_window_qwen`；`settings.context_window_deepseek`；
# `settings.context_window_anthropic` 等 5 种表达式。
# 分支与异常：
#   当 `name == 'openai'` 时，返回 `settings.context_window_openai`。
#   当 `name == 'qwen'` 时，返回 `settings.context_window_qwen`。
#   当 `name == 'deepseek'` 时，返回 `settings.context_window_deepseek`。
#   当 `name == 'anthropic'` 时，返回 `settings.context_window_anthropic`。
def _window_for(name: str, settings: ContextSettings) -> int:
    if name == "openai":
        return settings.context_window_openai
    if name == "qwen":
        return settings.context_window_qwen
    if name == "deepseek":
        return settings.context_window_deepseek
    if name == "anthropic":
        return settings.context_window_anthropic
    return settings.context_window_default


# 函数说明：_resolve_override_target
# 用途：解析或定位`override_target`，供模型上下文与输入预算使用。
# 参数：
#   model_settings：模型设置输入或配置值，类型 `ModelSettings | None`。
#   settings：业务或模型设置，类型 `ContextSettings`。
# 返回：类型 `tuple[str, str] | None`；按分支返回 `None`；`(provider.value, model)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettings` → `ModelProvider`。
# 分支与异常：
#   捕获 `ValueError` 后，返回 `None`。
def _resolve_override_target(
    model_settings: ModelSettings | None,
    settings: ContextSettings,
) -> tuple[str, str] | None:

    resolved_settings = model_settings or ModelSettings()
    try:
        provider = (
            ModelProvider(settings.context_override_provider)
            if settings.context_override_provider
            else resolved_settings.model_default_provider
        )
    except ValueError:
        return None
    model = settings.context_override_model or str(
        getattr(resolved_settings, f"{provider.value}_model")
    )
    return provider.value, model


__all__ = [
    "CapabilitySource",
    "FALLBACK_CONTEXT_WINDOW",
    "ModelCapabilities",
    "ModelCapabilityRegistry",
    "build_model_capability_registry",
]
