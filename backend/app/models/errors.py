

class ModelAdapterError(RuntimeError):
    pass


class ProviderNotConfiguredError(ModelAdapterError):
    # 函数说明：ProviderNotConfiguredError.__init__
    # 用途：初始化 ProviderNotConfiguredError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    #   environment_variable：`environment_variable`输入或配置值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    def __init__(self, provider: str, environment_variable: str) -> None:
        super().__init__(
            f"Provider '{provider}' is not configured. "
            f"Set {environment_variable} in the environment."
        )


class UnsupportedProviderError(ModelAdapterError):
    # 函数说明：UnsupportedProviderError.__init__
    # 用途：初始化 UnsupportedProviderError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    def __init__(self, provider: str) -> None:
        super().__init__(f"No model adapter is registered for provider '{provider}'.")


class UnsupportedMessageError(ModelAdapterError):
    pass
