
from __future__ import annotations

from typing import Protocol


class ModelSecretStore(Protocol):

    # 函数说明：ModelSecretStore.get
    # 用途：获取ModelSecretStore，供模型配置与密钥管理使用。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    # 返回：类型 `str | None`；不返回结果值（隐式 None）。
    def get(self, provider: str) -> str | None: ...

    # 函数说明：ModelSecretStore.set
    # 用途：设置ModelSecretStore，供模型配置与密钥管理使用。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def set(self, provider: str, value: str) -> None: ...


class EnvironmentSecretStore:

    # 函数说明：EnvironmentSecretStore.get
    # 用途：获取EnvironmentSecretStore，供模型配置与密钥管理使用。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    # 返回：类型 `str | None`；无结果值，显式返回 None。
    def get(self, provider: str) -> str | None:
        del provider
        return None

    # 函数说明：EnvironmentSecretStore.set
    # 用途：设置EnvironmentSecretStore，供模型配置与密钥管理使用。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def set(self, provider: str, value: str) -> None:
        del provider, value
        raise RuntimeError(
            "Web deployment does not persist API keys; "
            "configure server environment variables"
        )


__all__ = ["EnvironmentSecretStore", "ModelSecretStore"]
