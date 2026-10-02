
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from pydantic import SecretStr

from app.domain.memory import MemoryMaintenanceConfig, MemoryReflectionConfig
from app.models.config import ModelSettings, ProviderConfig
from app.models.providers import AnthropicAdapter, OpenAICompatibleAdapter
from app.models.types import Message, MessageRole, ModelProvider, ModelRequest
from app.runtime.context import ContextSummaryModelConfig

from .models import (
    ModelRoleSettings,
    ModelSettingsUpdate,
    ProviderSettings,
    ProviderSettingsUpdate,
    StoredModelSettings,
)
from .secrets import EnvironmentSecretStore, ModelSecretStore
from .store import ModelSettingsStore

_PROVIDER_LABELS = {
    ModelProvider.OPENAI: "OpenAI",
    ModelProvider.QWEN: "Qwen",
    ModelProvider.DEEPSEEK: "DeepSeek",
    ModelProvider.ANTHROPIC: "Claude",
}
_KEY_FIELDS = {
    ModelProvider.OPENAI: "openai_api_key",
    ModelProvider.QWEN: "qwen_api_key",
    ModelProvider.DEEPSEEK: "deepseek_api_key",
    ModelProvider.ANTHROPIC: "anthropic_api_key",
}
_OFFICIAL_TEST_HOSTS = {
    ModelProvider.OPENAI: "api.openai.com",
    ModelProvider.QWEN: "dashscope.aliyuncs.com",
    ModelProvider.DEEPSEEK: "api.deepseek.com",
    ModelProvider.ANTHROPIC: "api.anthropic.com",
}


@dataclass(frozen=True)
class EffectiveModelConfiguration:
    settings: ModelSettings
    reflection: MemoryReflectionConfig | None
    maintenance: MemoryMaintenanceConfig | None
    summary: ContextSummaryModelConfig | None


class ModelSettingsService:
    # 函数说明：ModelSettingsService.__init__
    # 用途：初始化 ModelSettingsService；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `ModelSettingsStore | None`；默认 `None`。
    #   secrets：`secrets`输入或配置值，类型 `ModelSecretStore | None`；默认配置见签名。
    #   base_settings：设置输入或配置值，类型 `ModelSettings | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettingsStore` →
    # `EnvironmentSecretStore` → `ModelSettings`。
    # 副作用与资源：
    #   更新对象字段：`self.store`、`self.secrets`、`self.base_settings`。
    def __init__(
        self,
        *,
        store: ModelSettingsStore | None = None,
        secrets: ModelSecretStore | None = None,
        base_settings: ModelSettings | None = None,
    ) -> None:
        self.store = store or ModelSettingsStore()
        self.secrets = secrets or EnvironmentSecretStore()
        self.base_settings = base_settings or ModelSettings()

    # 函数说明：ModelSettingsService.view
    # 用途：在模型配置与密钥管理中处理 `view`，通过 `self.store.load` 完成首个内部处理步
    # 骤。
    # 参数：
    #   active_provider：活跃项服务商输入或配置值，类型 `str`。
    #   active_model：活跃项模型输入或配置值，类型 `str`。
    #   active_roles：活跃项输入或配置值，类型 `dict[str, dict[str, Any]] | None`；默认
    # `None`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `default_provider`、`providers`、
    # `reflection`、`maintenance`、`summary`、`active_provider`、`active_model`、
    # `active_roles`、`restart_required`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.store.load` → `_defaults` →
    # `_secret_value` → `_resolved_saved_roles`。
    def view(
        self,
        *,
        active_provider: str,
        active_model: str,
        active_roles: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        base = self.base_settings
        stored = self.store.load() or _defaults(base)
        providers = []
        for provider in ModelProvider:
            item = stored.providers[provider.value]
            stored_key = self.secrets.get(provider.value)
            env_key = _secret_value(getattr(base, _KEY_FIELDS[provider]))
            source = "environment" if stored_key or env_key else "none"
            providers.append(
                {
                    **item.model_dump(mode="json"),
                    "label": _PROVIDER_LABELS[provider],
                    "configured": source != "none",
                    "key_source": source,
                }
            )
        current_roles = active_roles or {
            "main": {
                "enabled": True,
                "provider": active_provider,
                "model": active_model,
            }
        }
        saved_roles = _resolved_saved_roles(stored)
        return {
            "default_provider": stored.default_provider.value,
            "providers": providers,
            "reflection": stored.reflection.model_dump(mode="json"),
            "maintenance": stored.maintenance.model_dump(mode="json"),
            "summary": stored.summary.model_dump(mode="json"),
            "active_provider": active_provider,
            "active_model": active_model,
            "active_roles": current_roles,
            "restart_required": saved_roles != current_roles,
        }

    # 函数说明：ModelSettingsService.save
    # 用途：保存ModelSettingsService，供模型配置与密钥管理使用。
    # 参数：
    #   update：`update`输入或配置值，类型 `ModelSettingsUpdate`。
    # 返回：类型 `StoredModelSettings`；返回 `stored`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_secret_value` →
    # `StoredModelSettings` → `ProviderSettings.model_validate` → `self.secrets.set` →
    # `self.store.save`。
    # 分支与异常：
    #   当 `not default_item.api_key and (not existing_keys[…` 时，抛出
    # `ValueError('default provider requires an API key')`。
    #   当 `role.enabled and (not role.inherit_main) and (role.provider…` 时，抛出
    # `ValueError(…)`。
    def save(self, update: ModelSettingsUpdate) -> StoredModelSettings:
        base = self.base_settings
        existing_keys = {
            provider: bool(self.secrets.get(provider.value))
            or bool(_secret_value(getattr(base, _KEY_FIELDS[provider])))
            for provider in ModelProvider
        }
        submitted = {item.provider: item for item in update.providers}
        default_item = submitted[update.default_provider]
        if not default_item.api_key and not existing_keys[update.default_provider]:
            raise ValueError("default provider requires an API key")
        for role in (update.reflection, update.maintenance, update.summary):
            if (
                role.enabled
                and not role.inherit_main
                and role.provider is not None
                and not submitted[role.provider].api_key
                and not existing_keys[role.provider]
            ):
                raise ValueError(
                    f"model role provider '{role.provider.value}' requires an API key"
                )

        stored = StoredModelSettings(
            default_provider=update.default_provider,
            providers={
                item.provider.value: ProviderSettings.model_validate(
                    item.model_dump(exclude={"api_key"})
                )
                for item in update.providers
            },
            reflection=update.reflection,
            maintenance=update.maintenance,
            summary=update.summary,
        )
        for item in update.providers:
            if item.api_key:
                self.secrets.set(item.provider.value, item.api_key)
        self.store.save(stored)
        return stored

    # 函数说明：ModelSettingsService.test
    # 用途：在模型配置与密钥管理中处理 `test`，通过 `self.secrets.get` 完成首个内部处理
    # 步骤。
    # 参数：
    #   item：当前集合元素，类型 `ProviderSettingsUpdate`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `success`、`provider`、`model`、
    # `duration_ms`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `_validate_connection_test_endpoint` → `base_key.get_secret_value` →
    # `ProviderConfig` → `SecretStr` → `AnthropicAdapter` → `OpenAICompatibleAdapter`；
    # 另有 6 个调用点。
    # 分支与异常：
    #   当 `not key` 时，抛出 `ValueError('API key is required before testing')`。
    async def test(self, item: ProviderSettingsUpdate) -> dict[str, Any]:
        _validate_connection_test_endpoint(item)
        key = item.api_key or self.secrets.get(item.provider.value)
        if not key:
            base_key = getattr(self.base_settings, _KEY_FIELDS[item.provider])
            key = base_key.get_secret_value() if base_key else None
        if not key:
            raise ValueError("API key is required before testing")
        config = ProviderConfig(
            provider=item.provider.value,
            model=item.model,
            api_key=SecretStr(key),
            api_style=item.api_style,
            base_url=item.base_url,
            timeout_seconds=30,
            max_retries=0,
            default_max_output_tokens=64,
        )
        adapter = (
            AnthropicAdapter(config)
            if item.provider is ModelProvider.ANTHROPIC
            else OpenAICompatibleAdapter(config)
        )
        started = time.perf_counter()
        try:
            response = await adapter.complete(
                ModelRequest(
                    messages=(
                        Message(role=MessageRole.USER, content="Reply with OK."),
                    ),
                    model=item.model,
                    max_output_tokens=64,
                    temperature=0,
                )
            )
        finally:
            await adapter.close()
        return {
            "success": True,
            "provider": response.provider,
            "model": response.model,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }


# 函数说明：load_effective_model_configuration
# 用途：加载模型，供模型配置与密钥管理使用。
# 参数：
#   store：持久化存储依赖，类型 `ModelSettingsStore | None`；默认 `None`。
#   secrets：`secrets`输入或配置值，类型 `ModelSecretStore | None`；默认配置见签名。
#   base_settings：设置输入或配置值，类型 `ModelSettings | None`；默认 `None`。
# 返回：类型 `EffectiveModelConfiguration`；按分支返回
# `EffectiveModelConfiguration(base, None, None, None)`；
# `EffectiveModelConfiguration(settings, reflection, maintenance, summary)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettingsStore` →
# `resolved_store.load` → `ModelSettings` → `EffectiveModelConfiguration` →
# `EnvironmentSecretStore` → `SecretStr`；另有 4 个调用点。
# 分支与异常：
#   当 `stored is None` 时，返回 `EffectiveModelConfiguration(base, None, None, None)`。
def load_effective_model_configuration(
    *,
    store: ModelSettingsStore | None = None,
    secrets: ModelSecretStore | None = None,
    base_settings: ModelSettings | None = None,
) -> EffectiveModelConfiguration:

    resolved_store = store or ModelSettingsStore()
    stored = resolved_store.load()
    base = base_settings or ModelSettings()
    if stored is None:
        return EffectiveModelConfiguration(base, None, None, None)
    resolved_secrets = secrets or EnvironmentSecretStore()
    settings_values = base.model_dump(mode="python")
    settings_values["model_default_provider"] = stored.default_provider
    for provider in ModelProvider:
        item = stored.providers[provider.value]
        prefix = provider.value
        settings_values[f"{prefix}_model"] = item.model
        settings_values[f"{prefix}_base_url"] = item.base_url
        if provider is not ModelProvider.ANTHROPIC:
            settings_values[f"{prefix}_api_style"] = item.api_style
        secret = resolved_secrets.get(provider.value)
        if secret:
            settings_values[_KEY_FIELDS[provider]] = SecretStr(secret)
    settings = ModelSettings.model_validate(settings_values)
    reflection = _reflection_config(stored.reflection)
    maintenance = _maintenance_config(stored.maintenance)
    summary = _summary_config(stored.summary)
    return EffectiveModelConfiguration(settings, reflection, maintenance, summary)


# 函数说明：_defaults
# 用途：处理模型配置与密钥管理中的 `_defaults` 数据；结果及边界条件见下方说明。
# 参数：
#   settings：业务或模型设置，类型 `ModelSettings`。
# 返回：类型 `StoredModelSettings`；返回 `StoredModelSettings(default_provider=settings.
# model_default_provider, providers=providers)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderSettings` →
# `StoredModelSettings`。
def _defaults(settings: ModelSettings) -> StoredModelSettings:
    providers = {}
    for provider in ModelProvider:
        prefix = provider.value
        model = getattr(settings, f"{prefix}_model")
        base_url = getattr(settings, f"{prefix}_base_url")
        api_style = (
            getattr(settings, f"{prefix}_api_style")
            if provider is not ModelProvider.ANTHROPIC
            else "anthropic_messages"
        )
        providers[provider.value] = ProviderSettings(
            provider=provider,
            model=model,
            base_url=base_url,
            api_style=api_style,
        )
    return StoredModelSettings(
        default_provider=settings.model_default_provider,
        providers=providers,
    )


# 函数说明：_reflection_config
# 用途：返回 `MemoryReflectionConfig(…)`，提供 模型配置与密钥管理 的派生值。
# 参数：
#   role：规划、执行、审计等运行角色，类型 `ModelRoleSettings`。
# 返回：类型 `MemoryReflectionConfig`；返回 `MemoryReflectionConfig(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryReflectionConfig`。
def _reflection_config(role: ModelRoleSettings) -> MemoryReflectionConfig:
    return MemoryReflectionConfig(
        enabled=role.enabled,
        provider=None if role.inherit_main else role.provider.value,
        model=None if role.inherit_main else role.model,
    )


# 函数说明：_maintenance_config
# 用途：返回 `MemoryMaintenanceConfig(…)`，提供 模型配置与密钥管理 的派生值。
# 参数：
#   role：规划、执行、审计等运行角色，类型 `ModelRoleSettings`。
# 返回：类型 `MemoryMaintenanceConfig`；返回 `MemoryMaintenanceConfig(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryMaintenanceConfig`。
def _maintenance_config(role: ModelRoleSettings) -> MemoryMaintenanceConfig:
    return MemoryMaintenanceConfig(
        enabled=role.enabled,
        provider=None if role.inherit_main else role.provider.value,
        model=None if role.inherit_main else role.model,
    )


# 函数说明：_summary_config
# 用途：返回 `ContextSummaryModelConfig(…)`，提供 模型配置与密钥管理 的派生值。
# 参数：
#   role：规划、执行、审计等运行角色，类型 `ModelRoleSettings`。
# 返回：类型 `ContextSummaryModelConfig`；返回 `ContextSummaryModelConfig(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextSummaryModelConfig`。
def _summary_config(role: ModelRoleSettings) -> ContextSummaryModelConfig:
    return ContextSummaryModelConfig(
        enabled=role.enabled,
        provider=None if role.inherit_main else role.provider.value,
        model=None if role.inherit_main else role.model,
    )


# 函数说明：_resolved_saved_roles
# 用途：处理模型配置与密钥管理中的 `_resolved_saved_roles` 数据；结果及边界条件见下方说
# 明。
# 参数：
#   stored：`stored`输入或配置值，类型 `StoredModelSettings`。
# 返回：类型 `dict[str, dict[str, Any]]`；字典，包含字段 `main`、`summary`、`reflection`
# 、`maintenance`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`resolve`。
def _resolved_saved_roles(
    stored: StoredModelSettings,
) -> dict[str, dict[str, Any]]:
    main_provider = stored.default_provider
    main_model = stored.providers[main_provider.value].model

    # 函数说明：_resolved_saved_roles.resolve
    # 用途：解析或定位模型配置与密钥管理，供模型配置与密钥管理使用。
    # 参数：
    #   role：规划、执行、审计等运行角色，类型 `ModelRoleSettings`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `enabled`、`provider`、`model`。
    # 闭包依赖：从外层读取 `main_model`、`main_provider`。
    def resolve(role: ModelRoleSettings) -> dict[str, Any]:
        return {
            "enabled": role.enabled,
            "provider": (
                main_provider.value if role.inherit_main else role.provider.value
            ),
            "model": main_model if role.inherit_main else role.model,
        }

    return {
        "main": {
            "enabled": True,
            "provider": main_provider.value,
            "model": main_model,
        },
        "summary": resolve(stored.summary),
        "reflection": resolve(stored.reflection),
        "maintenance": resolve(stored.maintenance),
    }


# 函数说明：_secret_value
# 用途：在模型配置与密钥管理中处理 `_secret_value`，通过
# `secret.get_secret_value().strip` 完成首个内部处理步骤。
# 参数：
#   secret：`secret`输入或配置值，类型 `SecretStr | None`。
# 返回：类型 `str | None`；按分支返回 `None`；
# `secret.get_secret_value().strip() or None`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`secret.get_secret_value`。
# 分支与异常：
#   当 `secret is None` 时，返回 `None`。
def _secret_value(secret: SecretStr | None) -> str | None:
    if secret is None:
        return None
    return secret.get_secret_value().strip() or None


# 函数说明：_validate_connection_test_endpoint
# 用途：校验连接，供模型配置与密钥管理使用。
# 参数：
#   item：当前集合元素，类型 `ProviderSettingsUpdate`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`urlparse`。
# 分支与异常：
#   `item.base_url is None` 分支在完成前置处理后抛出 `ValueError(…)`。
#   当 `item.provider in {ModelProvider.OPENAI,…` 时，返回 `None`。
#   当 `parsed.scheme != 'https' or parsed.hostname != expected_host` 时，抛出
# `ValueError(…)`。
def _validate_connection_test_endpoint(item: ProviderSettingsUpdate) -> None:

    expected_host = _OFFICIAL_TEST_HOSTS[item.provider]
    if item.base_url is None:
        if item.provider in {ModelProvider.OPENAI, ModelProvider.ANTHROPIC}:
            return
        raise ValueError("connection test requires the official provider endpoint")
    parsed = urlparse(item.base_url)
    if parsed.scheme != "https" or parsed.hostname != expected_host:
        raise ValueError(
            "connection test only supports the provider's official HTTPS endpoint"
        )


__all__ = [
    "EffectiveModelConfiguration",
    "ModelSettingsService",
    "load_effective_model_configuration",
]
