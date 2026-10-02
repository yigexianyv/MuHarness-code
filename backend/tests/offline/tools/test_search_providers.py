
from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from app.models.types import ToolCall
from app.tools import ToolExecutor, ToolRegistry, WebSearchTool
from app.tools.search import (
    DuckDuckGoSearchProvider,
    SearchAuthenticationError,
    SearchNetworkError,
    SearchNoResultsError,
    SearchProvider,
    SearchProviderName,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchService,
    SearchSettings,
    SearchUnavailableError,
    TavilySearchProvider,
    build_search_service,
)


class StubSearchProvider(SearchProvider):
    # 函数说明：StubSearchProvider.__init__
    # 用途：初始化 StubSearchProvider；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   outcome：执行或验收结果，类型 `SearchResponse | Exception`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._name`、`self._outcome`、`self.requests`。
    def __init__(
        self,
        name: str,
        outcome: SearchResponse | Exception,
    ) -> None:
        self._name = name
        self._outcome = outcome
        self.requests: list[SearchRequest] = []

    # 函数说明：StubSearchProvider.name
    # 用途：返回 `self._name`，提供 StubSearchProvider 的派生值。
    # 返回：类型 `str`；返回 `self._name`。
    @property
    def name(self) -> str:
        return self._name

    # 函数说明：StubSearchProvider.search
    # 用途：检索StubSearchProvider，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `SearchRequest`。
    # 返回：类型 `SearchResponse`；返回 `self._outcome`。
    # 分支与异常：
    #   当 `isinstance(self._outcome, Exception)` 时，抛出 `self._outcome`。
    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


# 函数说明：response
# 用途：返回 `SearchResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   provider：模型或搜索服务商，类型 `str`。
#   title：面向用户的标题，类型 `str`；默认 `'Result'`。
#   url：目标 HTTP 地址，类型 `str`；默认 `'https://example.com/article'`。
#   snippet：`snippet`输入或配置值，类型 `str`；默认 `'useful content'`。
# 返回：类型 `SearchResponse`；返回 `SearchResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SearchResponse` → `SearchResult`。
def response(
    provider: str,
    *,
    title: str = "Result",
    url: str = "https://example.com/article",
    snippet: str = "useful content",
) -> SearchResponse:
    return SearchResponse(
        query="MuHarness",
        provider=provider,
        results=(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                score=0.9,
            ),
        ),
    )


# 函数说明：test_tavily_search_maps_request_and_response
# 用途：回归验证回归测试与测试辅助中的 `tavily_search_maps_request_and_response` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`httpx.AsyncClient` →
# `httpx.MockTransport` → `TavilySearchProvider` → `provider.search` → `SearchRequest`。
# 分支与异常：
#   验证条件：`captured['authorization'] == 'Bearer tvly-test'`。
#   验证条件：`isinstance(payload, dict)`。
#   验证条件：`payload['search_depth'] == 'basic'`。
#   验证条件：`payload['include_raw_content'] is False`。
@pytest.mark.asyncio
async def test_tavily_search_maps_request_and_response() -> None:
    captured: dict[str, object] = {}

    # 函数说明：test_tavily_search_maps_request_and_response.handler
    # 用途：在回归测试与测试辅助中处理 `handler`，通过 `request.headers.get` 完成首个内
    # 部处理步骤。
    # 参数：
    #   request：待处理的请求对象，类型 `httpx.Request`。
    # 返回：类型 `httpx.Response`；返回 `httpx.Response(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` → `httpx.Response`。
    # 闭包依赖：从外层读取 `captured`。
    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "河北天气预报",
                        "url": "https://weather.example/hebei",
                        "content": "未来三天气温",
                        "score": 0.87,
                        "published_date": "2026-08-04",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = TavilySearchProvider("tvly-test", client=client)

    result = await provider.search(
        SearchRequest(
            query="河北天气",
            topic="news",
            time_range="week",
            max_results=3,
            include_domains=("weather.com.cn",),
        )
    )

    assert captured["authorization"] == "Bearer tvly-test"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["search_depth"] == "basic"
    assert payload["include_raw_content"] is False
    assert payload["auto_parameters"] is False
    assert payload["time_range"] == "week"
    assert payload["include_domains"] == ["weather.com.cn"]
    assert result.provider == "tavily"
    assert result.results[0].score == 0.87
    assert result.results[0].published_at == "2026-08-04"


# 函数说明：test_tavily_authentication_error_is_explicit
# 用途：回归验证回归测试与测试辅助中的 `tavily_authentication_error_is_explicit` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`httpx.AsyncClient` →
# `httpx.MockTransport` → `pytest.raises` →
# `TavilySearchProvider('bad-key', client=client).search` → `TavilySearchProvider` →
# `SearchRequest`。
# 分支与异常：
#   预期异常：`pytest.raises(SearchAuthenticationError, match='API key')`。
@pytest.mark.asyncio
async def test_tavily_authentication_error_is_explicit() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"detail": "invalid key"})
        )
    )

    with pytest.raises(SearchAuthenticationError, match="API key"):
        await TavilySearchProvider("bad-key", client=client).search(
            SearchRequest(query="test")
        )


# 函数说明：test_duckduckgo_parses_redirect_url_and_domain_filters
# 用途：回归验证回归测试与测试辅助中的
# `duckduckgo_parses_redirect_url_and_domain_filters` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`DuckDuckGoSearchProvider` →
# `provider.search` → `SearchRequest`。
# 分支与异常：
#   验证条件：`'site:a.example' in captured[0]`。
#   验证条件：`'-site:spam.example' in captured[0]`。
#   验证条件：`result.results[0].url == 'https://a.example/news'`。
#   验证条件：`result.results[0].snippet == 'first snippet'`。
@pytest.mark.asyncio
async def test_duckduckgo_parses_redirect_url_and_domain_filters() -> None:
    captured: list[str] = []
    html = """
    <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2Fnews">
      Alpha News
    </a>
    <td class="result-snippet">first snippet</td>
    """

    # 函数说明：test_duckduckgo_parses_redirect_url_and_domain_filters.fetcher
    # 用途：在回归测试与测试辅助中处理 `fetcher`，通过 `captured.append` 完成首个内部处
    # 理步骤。
    # 参数：
    #   query：检索查询文本，类型 `str`。
    # 返回：类型 `str`；返回 `html`。
    # 闭包依赖：从外层读取 `captured`、`html`。
    async def fetcher(query: str) -> str:
        captured.append(query)
        return html

    provider = DuckDuckGoSearchProvider(fetcher=fetcher)
    result = await provider.search(
        SearchRequest(
            query="AI news",
            include_domains=("a.example",),
            exclude_domains=("spam.example",),
        )
    )

    assert "site:a.example" in captured[0]
    assert "-site:spam.example" in captured[0]
    assert result.results[0].url == "https://a.example/news"
    assert result.results[0].snippet == "first snippet"


# 函数说明：test_duckduckgo_empty_page_is_not_reported_as_success
# 用途：回归验证回归测试与测试辅助中的
# `duckduckgo_empty_page_is_not_reported_as_success` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `DuckDuckGoSearchProvider(fetcher=fetcher).search` → `DuckDuckGoSearchProvider` →
# `SearchRequest`。
# 分支与异常：
#   预期异常：`pytest.raises(SearchNoResultsError, match='no usable results')`。
@pytest.mark.asyncio
async def test_duckduckgo_empty_page_is_not_reported_as_success() -> None:
    # 函数说明：test_duckduckgo_empty_page_is_not_reported_as_success.fetcher
    # 用途：返回 `'<html><title>DuckDuckGo</title></html>'`，提供 回归测试与测试辅助 的
    # 派生值。
    # 参数：
    #   query：检索查询文本，类型 `str`。
    # 返回：类型 `str`；返回 `'<html><title>DuckDuckGo</title></html>'`。
    async def fetcher(query: str) -> str:
        return "<html><title>DuckDuckGo</title></html>"

    with pytest.raises(SearchNoResultsError, match="no usable results"):
        await DuckDuckGoSearchProvider(fetcher=fetcher).search(
            SearchRequest(query="missing")
        )


# 函数说明：test_search_service_falls_back_and_reports_reason
# 用途：回归验证回归测试与测试辅助中的 `search_service_falls_back_and_reports_reason` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StubSearchProvider` → `response` →
# `SearchService(primary, fallback).search` → `SearchService` → `SearchRequest`。
# 分支与异常：
#   验证条件：`result.provider == 'duckduckgo'`。
#   验证条件：`result.fallback_used is True`。
#   验证条件：`'temporary failure' in (result.fallback_reason or '')`。
#   验证条件：`len(primary.requests) == 1`。
@pytest.mark.asyncio
async def test_search_service_falls_back_and_reports_reason() -> None:
    primary = StubSearchProvider(
        "tavily",
        SearchNetworkError("temporary failure"),
    )
    fallback = StubSearchProvider("duckduckgo", response("duckduckgo"))

    result = await SearchService(primary, fallback).search(
        SearchRequest(query="MuHarness")
    )

    assert result.provider == "duckduckgo"
    assert result.fallback_used is True
    assert "temporary failure" in (result.fallback_reason or "")
    assert len(primary.requests) == 1
    assert len(fallback.requests) == 1


# 函数说明：test_search_authentication_error_does_not_hide_configuration_bug
# 用途：回归验证回归测试与测试辅助中的
# `search_authentication_error_does_not_hide_configuration_bug` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StubSearchProvider` → `response` →
# `pytest.raises` → `SearchService(primary, fallback).search` → `SearchService` →
# `SearchRequest`。
# 分支与异常：
#   验证条件：`fallback.requests == []`。
#   预期异常：`pytest.raises(SearchAuthenticationError, match='invalid key')`。
@pytest.mark.asyncio
async def test_search_authentication_error_does_not_hide_configuration_bug() -> None:
    primary = StubSearchProvider(
        "tavily",
        SearchAuthenticationError("invalid key"),
    )
    fallback = StubSearchProvider("duckduckgo", response("duckduckgo"))

    with pytest.raises(SearchAuthenticationError, match="invalid key"):
        await SearchService(primary, fallback).search(SearchRequest(query="test"))

    assert fallback.requests == []


# 函数说明：test_search_service_reports_when_both_providers_fail
# 用途：回归验证回归测试与测试辅助中的 `search_service_reports_when_both_providers_fail`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StubSearchProvider` → `pytest.raises`
#  → `SearchService(primary, fallback).search` → `SearchService` → `SearchRequest`。
# 分支与异常：
#   预期异常：`pytest.raises(SearchUnavailableError, match='tavily.*duckduckgo')`。
@pytest.mark.asyncio
async def test_search_service_reports_when_both_providers_fail() -> None:
    primary = StubSearchProvider("tavily", SearchNetworkError("offline"))
    fallback = StubSearchProvider(
        "duckduckgo",
        SearchNoResultsError("blocked"),
    )

    with pytest.raises(SearchUnavailableError, match="tavily.*duckduckgo"):
        await SearchService(primary, fallback).search(SearchRequest(query="test"))


# 函数说明：test_auto_search_settings_select_provider_by_api_key
# 用途：回归验证回归测试与测试辅助中的 `auto_search_settings_select_provider_by_api_key`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SearchSettings` → `SecretStr` →
# `build_search_service`。
# 分支与异常：
#   验证条件：`build_search_service(no_key).primary_provider == 'duckduckgo'`。
#   验证条件：`build_search_service(with_key).primary_provider == 'tavily'`。
def test_auto_search_settings_select_provider_by_api_key() -> None:
    no_key = SearchSettings(_env_file=None, search_provider="auto")
    with_key = SearchSettings(
        _env_file=None,
        search_provider="auto",
        tavily_api_key=SecretStr("tvly-test"),
    )

    assert build_search_service(no_key).primary_provider == "duckduckgo"
    assert build_search_service(with_key).primary_provider == "tavily"


# 函数说明：test_explicit_tavily_requires_api_key
# 用途：回归验证回归测试与测试辅助中的 `explicit_tavily_requires_api_key` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SearchSettings` → `pytest.raises` →
# `build_search_service`。
# 分支与异常：
#   预期异常：`pytest.raises(SearchAuthenticationError, match='TAVILY_API_KEY')`。
def test_explicit_tavily_requires_api_key() -> None:
    settings = SearchSettings(
        _env_file=None,
        search_provider=SearchProviderName.TAVILY,
    )

    with pytest.raises(SearchAuthenticationError, match="TAVILY_API_KEY"):
        build_search_service(settings)


# 函数说明：test_web_search_tool_returns_unified_result_without_approval
# 用途：回归验证回归测试与测试辅助中的
# `web_search_tool_returns_unified_result_without_approval` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StubSearchProvider` → `response` →
# `SearchSettings` → `ToolRegistry` → `registry.register` → `WebSearchTool`；另有 5 个调
# 用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`output['provider'] == 'tavily'`。
#   验证条件：`output['count'] == 1`。
#   验证条件：`len(output['results'][0]['snippet']) == 500`。
@pytest.mark.asyncio
async def test_web_search_tool_returns_unified_result_without_approval() -> None:
    provider = StubSearchProvider(
        "tavily",
        response("tavily", snippet="x" * 800),
    )
    settings = SearchSettings(
        _env_file=None,
        search_max_results=3,
    )
    registry = ToolRegistry()
    registry.register(
        WebSearchTool(
            service=SearchService(provider),
            settings=settings,
        )
    )

    result = await ToolExecutor(registry).execute(
        ToolCall(
            id="search-1",
            name="web_search",
            arguments={
                "query": "MuHarness",
                "max_results": 10,
                "topic": "general",
            },
        )
    )

    assert result.success is True
    output = json.loads(result.output or "{}")
    assert output["provider"] == "tavily"
    assert output["count"] == 1
    assert len(output["results"][0]["snippet"]) == 500
    assert provider.requests[0].max_results == 3
