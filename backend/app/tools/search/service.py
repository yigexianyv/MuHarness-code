
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from .base import SearchProvider
from .config import SearchProviderName, SearchSettings
from .duckduckgo import DuckDuckGoSearchProvider
from .errors import (
    SearchAuthenticationError,
    SearchError,
    SearchNetworkError,
    SearchNoResultsError,
    SearchRateLimitError,
    SearchUnavailableError,
)
from .tavily import TavilySearchProvider
from .types import SearchRequest, SearchResponse, SearchResult

MAX_TITLE_CHARS = 300
MAX_SNIPPET_CHARS = 500

_FALLBACK_ERRORS = (
    SearchNetworkError,
    SearchNoResultsError,
    SearchRateLimitError,
)


class SearchService:

    # 函数说明：SearchService.__init__
    # 用途：初始化 SearchService；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   primary：`primary`输入或配置值，类型 `SearchProvider`。
    #   fallback：`fallback`输入或配置值，类型 `SearchProvider | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._primary`、`self._fallback`。
    def __init__(
        self,
        primary: SearchProvider,
        fallback: SearchProvider | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    # 函数说明：SearchService.primary_provider
    # 用途：返回 `self._primary.name`，提供 SearchService 的派生值。
    # 返回：类型 `str`；返回 `self._primary.name`。
    @property
    def primary_provider(self) -> str:
        return self._primary.name

    # 函数说明：SearchService.search
    # 用途：检索SearchService，供网页搜索服务商与降级使用。
    # 参数：
    #   request：待处理的请求对象，类型 `SearchRequest`。
    # 返回：类型 `SearchResponse`；按分支返回
    # `_normalize_response(await self._primary.search(request), request)`；
    # `normalized.model_copy(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_response` →
    # `self._primary.search` → `self._fallback.search`。
    # 分支与异常：
    #   捕获 `SearchAuthenticationError` 后，重新抛出原异常。
    #   捕获 `_FALLBACK_ERRORS` 后，转换或抛出 `SearchUnavailableError(…)`。
    #   当 `self._fallback is None` 时，抛出 `SearchUnavailableError(…)`。
    #   捕获 `SearchError` 后，转换或抛出 `SearchUnavailableError(…)`。
    async def search(self, request: SearchRequest) -> SearchResponse:
        try:
            return _normalize_response(await self._primary.search(request), request)
        except SearchAuthenticationError:
            raise
        except _FALLBACK_ERRORS as primary_error:
            if self._fallback is None:
                raise SearchUnavailableError(
                    f"Search provider {self._primary.name!r} unavailable: "
                    f"{primary_error}"
                ) from primary_error
            try:
                response = await self._fallback.search(request)
                normalized = _normalize_response(response, request)
            except SearchError as fallback_error:
                raise SearchUnavailableError(
                    f"Search providers {self._primary.name!r} and "
                    f"{self._fallback.name!r} unavailable: "
                    f"{primary_error}; {fallback_error}"
                ) from fallback_error
            return normalized.model_copy(
                update={
                    "fallback_used": True,
                    "fallback_reason": (
                        f"{type(primary_error).__name__}: {primary_error}"
                    ),
                }
            )


# 函数说明：build_search_service
# 用途：构建检索，供网页搜索服务商与降级使用。
# 参数：
#   settings：业务或模型设置，类型 `SearchSettings | None`；默认 `None`。
# 返回：类型 `SearchService`；按分支返回 `SearchService(duckduckgo)`；
# `SearchService(tavily, duckduckgo)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SearchSettings` →
# `DuckDuckGoSearchProvider` → `resolved.tavily_api_key_value` → `SearchService` →
# `TavilySearchProvider`。
# 分支与异常：
#   当 `resolved.search_provider is SearchProviderName.DUCKDUCKGO` 时，返回
# `SearchService(duckduckgo)`。
#   当 `resolved.search_provider is SearchProviderName.TAVILY and…` 时，抛出
# `SearchAuthenticationError(…)`。
#   当 `api_key is None` 时，返回 `SearchService(duckduckgo)`。
def build_search_service(settings: SearchSettings | None = None) -> SearchService:

    resolved = settings or SearchSettings()
    duckduckgo = DuckDuckGoSearchProvider(
        timeout_seconds=resolved.search_timeout_seconds
    )
    api_key = resolved.tavily_api_key_value()

    if resolved.search_provider is SearchProviderName.DUCKDUCKGO:
        return SearchService(duckduckgo)
    if resolved.search_provider is SearchProviderName.TAVILY and api_key is None:
        raise SearchAuthenticationError(
            "SEARCH_PROVIDER=tavily requires TAVILY_API_KEY"
        )
    if api_key is None:
        return SearchService(duckduckgo)

    tavily = TavilySearchProvider(
        api_key,
        timeout_seconds=resolved.search_timeout_seconds,
    )
    return SearchService(tavily, duckduckgo)


# 函数说明：_normalize_response
# 用途：规范化响应，供网页搜索服务商与降级使用。
# 参数：
#   response：模型、工具或服务返回的响应，类型 `SearchResponse`。
#   request：待处理的请求对象，类型 `SearchRequest`。
# 返回：类型 `SearchResponse`；返回
# `response.model_copy(update={'query': request.query, 'results': tuple(results)})`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize_url` → `seen_urls.add`。
# 分支与异常：
#   当 `not url or url in seen_urls` 时，跳过当前循环项。
#   当 `not title` 时，跳过当前循环项。
#   当 `len(results) >= request.max_results` 时，结束当前循环。
#   当 `not results` 时，抛出 `SearchNoResultsError(…)`。
def _normalize_response(
    response: SearchResponse,
    request: SearchRequest,
) -> SearchResponse:
    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in response.results:
        url = _normalize_url(result.url)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = " ".join(result.title.split())[:MAX_TITLE_CHARS]
        if not title:
            continue
        results.append(
            result.model_copy(
                update={
                    "title": title,
                    "url": url,
                    "snippet": " ".join(result.snippet.split())[
                        :MAX_SNIPPET_CHARS
                    ],
                }
            )
        )
        if len(results) >= request.max_results:
            break

    if not results:
        raise SearchNoResultsError(
            f"Search provider {response.provider!r} returned no usable results"
        )
    return response.model_copy(
        update={
            "query": request.query,
            "results": tuple(results),
        }
    )


# 函数说明：_normalize_url
# 用途：规范化`url`，供网页搜索服务商与降级使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `str`；按分支返回 `''`；
# `urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ''))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`urlsplit` → `urlunsplit`。
# 分支与异常：
#   当 `parsed.scheme not in {'http', 'https'} or not parsed.netloc` 时，返回 `''`。
def _normalize_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


__all__ = ["SearchService", "build_search_service"]
