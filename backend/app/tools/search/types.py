
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchTopic(StrEnum):
    GENERAL = "general"
    NEWS = "news"
    FINANCE = "finance"


class SearchTimeRange(StrEnum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class SearchRequest(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)
    topic: SearchTopic = SearchTopic.GENERAL
    time_range: SearchTimeRange | None = None
    include_domains: tuple[str, ...] = Field(default=(), max_length=10)
    exclude_domains: tuple[str, ...] = Field(default=(), max_length=10)

    # 函数说明：SearchRequest.normalize_query
    # 用途：校验并规范化模型字段 'query'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `normalized`。
    # 分支与异常：
    #   当 `not normalized` 时，抛出 `ValueError('query cannot be empty')`。
    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query cannot be empty")
        return normalized

    # 函数说明：SearchRequest.normalize_domains
    # 用途：校验并规范化模型字段 'include_domains'、'exclude_domains'。
    # 参数：
    #   values：待处理的值集合，类型 `tuple[str, ...]`。
    # 返回：类型 `tuple[str, ...]`；返回 `tuple(normalized)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `value.strip().lower().removeprefix('https://').removeprefix` →
    # `value.strip().lower().removeprefix` → `value.strip().lower`。
    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def normalize_domains(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            domain = value.strip().lower().removeprefix("https://").removeprefix(
                "http://"
            )
            domain = domain.split("/", 1)[0]
            if domain and domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)


class SearchResult(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    url: str
    snippet: str = ""
    score: float | None = None
    published_at: str | None = None


class SearchResponse(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    provider: str
    results: tuple[SearchResult, ...]
    fallback_used: bool = False
    fallback_reason: str | None = None


__all__ = [
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchTimeRange",
    "SearchTopic",
]
