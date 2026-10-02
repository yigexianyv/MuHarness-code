


from __future__ import annotations

from collections.abc import Callable

from .adapter import ModelAdapter
from .config import ModelSettings, ProviderConfig
from .errors import UnsupportedProviderError
from .providers import AnthropicAdapter, OpenAICompatibleAdapter
from .types import ModelProvider

AdapterFactory = Callable[[ProviderConfig], ModelAdapter]


class ModelAdapterRegistry:

    # 函数说明：ModelAdapterRegistry.__init__
    # 用途：初始化 ModelAdapterRegistry；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   settings：业务或模型设置，类型 `ModelSettings | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettings`。
    # 副作用与资源：
    #   更新对象字段：`self.settings`、`self._factories`、`self._configs`、
    # `self._instances`。
    def __init__(self, settings: ModelSettings | None = None) -> None:
        self.settings = settings or ModelSettings()
        self._factories: dict[str, AdapterFactory] = {
            ModelProvider.OPENAI.value: OpenAICompatibleAdapter,
            ModelProvider.QWEN.value: OpenAICompatibleAdapter,
            ModelProvider.DEEPSEEK.value: OpenAICompatibleAdapter,
            ModelProvider.ANTHROPIC.value: AnthropicAdapter,
        }
        self._configs: dict[str, ProviderConfig] = {}
        self._instances: dict[str, ModelAdapter] = {}

    # 函数说明：ModelAdapterRegistry.register
    # 用途：登记模型适配器配置，供运行时按提供商和模型选择。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    #   factory：构造目标依赖的工厂，类型 `AdapterFactory`。
    #   config：运行配置，类型 `ProviderConfig | None`；默认 `None`。
    #   replace：`replace`输入或配置值，类型 `bool`；默认 `False`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`provider.strip().lower` →
    # `self._instances.pop`。
    # 分支与异常：
    #   当 `not provider_name` 时，抛出 `ValueError('provider cannot be empty')`。
    #   当 `not replace and provider_name in self._factories` 时，抛出 `ValueError(…)`。
    def register(
        self,
        provider: str,
        factory: AdapterFactory,
        *,
        config: ProviderConfig | None = None,
        replace: bool = False,
    ) -> None:

        """登记模型适配器配置，供运行时按提供商和模型选择。"""
        provider_name = provider.strip().lower()
        if not provider_name:
            raise ValueError("provider cannot be empty")
        if not replace and provider_name in self._factories:
            raise ValueError(f"Provider '{provider_name}' is already registered.")
        self._factories[provider_name] = factory
        if config is not None:
            self._configs[provider_name] = config
        self._instances.pop(provider_name, None)

    # 函数说明：ModelAdapterRegistry.get
    # 用途：解析模型配置并返回对应适配器。
    # 参数：
    #   provider：模型或搜索服务商，类型 `ModelProvider | str | None`；默认 `None`。
    # 返回：类型 `ModelAdapter`；按分支返回 `self._instances[provider_name]`；`adapter`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`provider_name.strip().lower` →
    # `self.settings.provider_config` → `factory`。
    # 分支与异常：
    #   当 `provider_name in self._instances` 时，返回 `self._instances[provider_name]`
    # 。
    #   当 `factory is None` 时，抛出 `UnsupportedProviderError(provider_name)`。
    #   捕获 `ValueError` 后，转换或抛出 `UnsupportedProviderError(provider_name)`。
    def get(
        self,
        provider: ModelProvider | str | None = None,
    ) -> ModelAdapter:
        """解析模型配置并返回对应适配器。"""
        provider_name = (
            provider.value
            if isinstance(provider, ModelProvider)
            else (provider or self.settings.model_default_provider.value)
        )
        provider_name = provider_name.strip().lower()

        if provider_name in self._instances:
            return self._instances[provider_name]
        factory = self._factories.get(provider_name)
        if factory is None:
            raise UnsupportedProviderError(provider_name)

        config = self._configs.get(provider_name)
        if config is None:
            try:
                config = self.settings.provider_config(provider_name)
            except ValueError as exc:
                raise UnsupportedProviderError(provider_name) from exc

        adapter = factory(config)
        self._instances[provider_name] = adapter
        return adapter

    # 函数说明：ModelAdapterRegistry.close
    # 用途：关闭ModelAdapterRegistry，供模型请求与响应处理使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`adapter.close` →
    # `self._instances.clear`。
    async def close(self) -> None:
        for adapter in tuple(self._instances.values()):
            await adapter.close()
        self._instances.clear()
