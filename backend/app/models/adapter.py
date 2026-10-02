

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from .config import ProviderConfig
from .types import ModelRequest, ModelResponse


class ModelAdapter(ABC):
    # 函数说明：ModelAdapter.__init__
    # 用途：初始化 ModelAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.config`。
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    # 函数说明：ModelAdapter.provider
    # 用途：返回 `self.config.provider`，提供 ModelAdapter 的派生值。
    # 返回：类型 `str`；返回 `self.config.provider`。
    @property
    def provider(self) -> str:
        return self.config.provider

    # 函数说明：ModelAdapter.default_model
    # 用途：返回 `self.config.model`，提供 ModelAdapter 的派生值。
    # 返回：类型 `str`；返回 `self.config.model`。
    @property
    def default_model(self) -> str:
        return self.config.model

    # 函数说明：ModelAdapter.complete
    # 用途：完成ModelAdapter，供模型请求与响应处理使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；不返回结果值（隐式 None）。
    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        pass

    # 函数说明：ModelAdapter.complete_stream
    # 用途：完成事件流，供模型请求与响应处理使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    #   on_text_delta：增量文本回调，类型 `Callable[[str], Awaitable[None]]`。
    #   on_reasoning_delta：增量推理文本回调，类型
    # `Callable[[str], Awaitable[None]] | None`；默认 `None`。
    # 返回：类型 `ModelResponse`；返回 `await self.complete(request)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.complete`。
    async def complete_stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> ModelResponse:

        return await self.complete(request)

    # 函数说明：ModelAdapter.close
    # 用途：关闭ModelAdapter，供模型请求与响应处理使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    @abstractmethod
    async def close(self) -> None:
        pass
