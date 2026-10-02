
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx

from .base import SearchProvider
from .errors import (
    SearchAuthenticationError,
    SearchError,
    SearchNetworkError,
    SearchNoResultsError,
    SearchRateLimitError,
)
from .types import SearchRequest, SearchResponse, SearchResult

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilySearchProvider(SearchProvider):

    # 函数说明：TavilySearchProvider.__init__
    # 用途：初始化 TavilySearchProvider；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   api_key：键输入或配置值，类型 `str`。
    #   timeout_seconds：等待或执行超时，单位为秒，类型 `float`；默认 `15.0`。
    #   client：模型、HTTP 或 MCP 客户端，类型 `httpx.AsyncClient | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `not normalized_key` 时，抛出 `ValueError('Tavily API key cannot be empty')`
    # 。
    # 副作用与资源：
    #   更新对象字段：`self._api_key`、`self._timeout_seconds`、`self._client`。
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("Tavily API key cannot be empty")
        self._api_key = normalized_key
        self._timeout_seconds = timeout_seconds
        self._client = client

    # 函数说明：TavilySearchProvider.name
    # 用途：返回 `'tavily'`，提供 TavilySearchProvider 的派生值。
    # 返回：类型 `str`；返回 `'tavily'`。
    @property
    def name(self) -> str:
        return "tavily"

    # 函数说明：TavilySearchProvider.search
    # 用途：检索TavilySearchProvider，供网页搜索服务商与降级使用。
    # 参数：
    #   request：待处理的请求对象，类型 `SearchRequest`。
    # 返回：类型 `SearchResponse`；返回 `SearchResponse(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._post` →
    # `self._raise_for_status` → `response.json` → `urlsplit` → `SearchResult` →
    # `SearchResponse`。
    # 分支与异常：
    #   捕获 `ValueError` 后，转换或抛出
    # `SearchNetworkError('Tavily returned invalid JSON')`。
    #   当 `not isinstance(body, dict)` 时，抛出 `SearchNetworkError(…)`。
    #   当 `not isinstance(raw_results, list)` 时，抛出
    # `SearchNetworkError('Tavily response is missing results')`。
    #   当 `not isinstance(item, dict)` 时，跳过当前循环项。
    async def search(self, request: SearchRequest) -> SearchResponse:
        payload: dict[str, Any] = {
            "query": request.query,
            "search_depth": "basic",
            "topic": request.topic.value,
            "max_results": request.max_results,
            "include_answer": False,
            "include_raw_content": False,
            "auto_parameters": False,
        }
        if request.time_range is not None:
            payload["time_range"] = request.time_range.value
        if request.include_domains:
            payload["include_domains"] = list(request.include_domains)
        if request.exclude_domains:
            payload["exclude_domains"] = list(request.exclude_domains)

        response = await self._post(payload)
        self._raise_for_status(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise SearchNetworkError("Tavily returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise SearchNetworkError("Tavily returned an invalid response body")

        raw_results = body.get("results")
        if not isinstance(raw_results, list):
            raise SearchNetworkError("Tavily response is missing results")

        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if urlsplit(url).scheme not in {"http", "https"}:
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            raw_score = item.get("score")
            score = float(raw_score) if isinstance(raw_score, (int, float)) else None
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=str(item.get("content") or "").strip(),
                    score=score,
                    published_at=(
                        str(item["published_date"])
                        if item.get("published_date")
                        else None
                    ),
                )
            )

        if not results:
            raise SearchNoResultsError("Tavily returned no usable results")
        return SearchResponse(
            query=request.query,
            provider=self.name,
            results=tuple(results[: request.max_results]),
        )

    # 函数说明：TavilySearchProvider._post
    # 用途：在网页搜索服务商与降级中处理 `_post`，通过 `self._client.post` 完成首个内部
    # 处理步骤。
    # 参数：
    #   payload：传输或持久化载荷，类型 `dict[str, Any]`。
    # 返回：类型 `httpx.Response`；按分支返回 `await self._client.post(…)`；
    # `await client.post(TAVILY_SEARCH_URL, json=payload, headers=headers)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._client.post` →
    # `httpx.AsyncClient` → `client.post`。
    # 资源/并发边界：`httpx.AsyncClient(timeout=self._timeout_seconds)`，上下文退出时执
    # 行相应清理。
    # 分支与异常：
    #   当 `self._client is not None` 时，返回 `await self._client.post(…)`。
    #   捕获 `httpx.HTTPError` 后，转换或抛出
    # `SearchNetworkError(f'Tavily request failed: {exc}')`。
    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            if self._client is not None:
                return await self._client.post(
                    TAVILY_SEARCH_URL,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                return await client.post(
                    TAVILY_SEARCH_URL,
                    json=payload,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise SearchNetworkError(f"Tavily request failed: {exc}") from exc

    # 函数说明：TavilySearchProvider._raise_for_status
    # 用途：处理网页搜索服务商与降级中的 `_raise_for_status` 数据；结果及边界条件见下方
    # 说明。
    # 参数：
    #   response：模型、工具或服务返回的响应，类型 `httpx.Response`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `response.status_code in {401, 403}` 时，抛出 `SearchAuthenticationError(…)`
    # 。
    #   当 `response.status_code in {429, 432, 433}` 时，抛出 `SearchRateLimitError(…)`
    # 。
    #   当 `response.status_code >= 500` 时，抛出 `SearchNetworkError(…)`。
    #   当 `response.status_code >= 400` 时，抛出 `SearchError(…)`。
    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise SearchAuthenticationError(
                "Tavily API key is invalid or lacks permission"
            )
        if response.status_code in {429, 432, 433}:
            raise SearchRateLimitError(
                f"Tavily quota or rate limit reached ({response.status_code})"
            )
        if response.status_code >= 500:
            raise SearchNetworkError(
                f"Tavily service error ({response.status_code})"
            )
        if response.status_code >= 400:
            raise SearchError(f"Tavily request rejected ({response.status_code})")


__all__ = ["TAVILY_SEARCH_URL", "TavilySearchProvider"]
