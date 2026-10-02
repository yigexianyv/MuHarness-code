
from __future__ import annotations

from abc import ABC, abstractmethod

from .types import SearchRequest, SearchResponse


class SearchProvider(ABC):

    # 函数说明：SearchProvider.name
    # 用途：处理网页搜索服务商与降级中的 `name` 数据；结果及边界条件见下方说明。
    # 返回：类型 `str`；不返回结果值（隐式 None）。
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    # 函数说明：SearchProvider.search
    # 用途：检索SearchProvider，供网页搜索服务商与降级使用。
    # 参数：
    #   request：待处理的请求对象，类型 `SearchRequest`。
    # 返回：类型 `SearchResponse`；不返回结果值（隐式 None）。
    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResponse:
        pass


__all__ = ["SearchProvider"]
