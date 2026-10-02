
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.types import ApiStyle, ModelProvider


class ProviderSettings(BaseModel):

    model_config = ConfigDict(extra="forbid")

    provider: ModelProvider
    model: str
    base_url: str | None = None
    api_style: ApiStyle

    # 函数说明：ProviderSettings.normalize_model
    # 用途：校验并规范化模型字段 'model'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `value.strip()`。
    # 分支与异常：
    #   当 `not isinstance(value, str) or not value.strip()` 时，抛出
    # `ValueError('model cannot be empty')`。
    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model cannot be empty")
        return value.strip()

    # 函数说明：ProviderSettings.normalize_base_url
    # 用途：校验并规范化模型字段 'base_url'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().rstrip` →
    # `normalized.startswith`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('base_url must be a string')`。
    #   当 `not normalized` 时，返回 `None`。
    #   当 `not normalized.startswith(('http://', 'https://'))` 时，抛出
    # `ValueError('base_url must use http or https')`。
    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("base_url must be a string")
        normalized = value.strip().rstrip("/")
        if not normalized:
            return None
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        return normalized

    # 函数说明：ProviderSettings.validate_provider_style
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `ProviderSettings`；返回 `self`。
    # 分支与异常：
    #   当 `self.provider is ModelProvider.ANTHROPIC and self.api_style…` 时，抛出
    # `ValueError(…)`。
    #   当 `self.provider is not ModelProvider.ANTHROPIC and…` 时，抛出
    # `ValueError('anthropic_messages is only valid for anthropic')`。
    @model_validator(mode="after")
    def validate_provider_style(self) -> ProviderSettings:
        if (
            self.provider is ModelProvider.ANTHROPIC
            and self.api_style is not ApiStyle.ANTHROPIC_MESSAGES
        ):
            raise ValueError("anthropic requires anthropic_messages api style")
        if (
            self.provider is not ModelProvider.ANTHROPIC
            and self.api_style is ApiStyle.ANTHROPIC_MESSAGES
        ):
            raise ValueError("anthropic_messages is only valid for anthropic")
        return self


class ProviderSettingsUpdate(ProviderSettings):

    api_key: str | None = Field(default=None, max_length=20_000)

    # 函数说明：ProviderSettingsUpdate.normalize_api_key
    # 用途：校验并规范化模型字段 'api_key'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`value.strip() or None`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出 `TypeError('api_key must be a string')`
    # 。
    @field_validator("api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("api_key must be a string")
        return value.strip() or None


class ModelRoleSettings(BaseModel):

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    inherit_main: bool = True
    provider: ModelProvider | None = None
    model: str | None = None

    # 函数说明：ModelRoleSettings.normalize_model
    # 用途：校验并规范化模型字段 'model'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`value.strip() or None`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('role model must be a string')`。
    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("role model must be a string")
        return value.strip() or None

    # 函数说明：ModelRoleSettings.validate_override
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `ModelRoleSettings`；返回 `self`。
    # 分支与异常：
    #   当 `self.inherit_main` 时，返回 `self`。
    #   当 `self.provider is None or self.model is None` 时，抛出
    # `ValueError('custom model role requires provider and model')`。
    @model_validator(mode="after")
    def validate_override(self) -> ModelRoleSettings:
        if self.inherit_main:
            return self
        if self.provider is None or self.model is None:
            raise ValueError("custom model role requires provider and model")
        return self


class StoredModelSettings(BaseModel):

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    default_provider: ModelProvider
    providers: dict[str, ProviderSettings]
    reflection: ModelRoleSettings = Field(default_factory=ModelRoleSettings)
    maintenance: ModelRoleSettings = Field(default_factory=ModelRoleSettings)
    summary: ModelRoleSettings = Field(default_factory=ModelRoleSettings)


class ModelSettingsUpdate(BaseModel):

    model_config = ConfigDict(extra="forbid")

    default_provider: ModelProvider
    providers: tuple[ProviderSettingsUpdate, ...]
    reflection: ModelRoleSettings = Field(default_factory=ModelRoleSettings)
    maintenance: ModelRoleSettings = Field(default_factory=ModelRoleSettings)
    summary: ModelRoleSettings = Field(default_factory=ModelRoleSettings)

    # 函数说明：ModelSettingsUpdate.unique_providers
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `ModelSettingsUpdate`；返回 `self`。
    # 分支与异常：
    #   当 `len(names) != len(set(names))` 时，抛出
    # `ValueError('provider settings contain duplicate providers')`。
    #   当 `set(names) != set(ModelProvider)` 时，抛出 `ValueError(…)`。
    @model_validator(mode="after")
    def unique_providers(self) -> ModelSettingsUpdate:
        names = [item.provider for item in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("provider settings contain duplicate providers")
        if set(names) != set(ModelProvider):
            raise ValueError("provider settings must contain every built-in provider")
        return self


__all__ = [
    "ModelRoleSettings",
    "ModelSettingsUpdate",
    "ProviderSettings",
    "ProviderSettingsUpdate",
    "StoredModelSettings",
]
