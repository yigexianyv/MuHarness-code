
from __future__ import annotations

from typing import Any

from app.models.types import ToolDefinition, ToolPermission

from ..base import BaseTool
from ..search import (
    SearchRequest,
    SearchService,
    SearchSettings,
    build_search_service,
)

MAX_QUERY_CHARS = 500


class WebSearchTool(BaseTool):

    # 函数说明：WebSearchTool.__init__
    # 用途：初始化 WebSearchTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   service：业务服务依赖，类型 `SearchService | None`；默认 `None`。
    #   settings：业务或模型设置，类型 `SearchSettings | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SearchSettings` →
    # `build_search_service`。
    # 副作用与资源：
    #   更新对象字段：`self._service`、`self._max_results`。
    def __init__(
        self,
        *,
        service: SearchService | None = None,
        settings: SearchSettings | None = None,
    ) -> None:
        resolved_settings = settings or SearchSettings()
        self._service = service or build_search_service(resolved_settings)
        self._max_results = resolved_settings.search_max_results

    # 函数说明：WebSearchTool.definition
    # 用途：提供 WebSearchTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description=(
                "搜索网页来源，返回标题、URL、简短摘要，以及可用的相关性分数和发布日期。"
                "需要外部事实、最新信息或来源依据时使用；普通稳定知识或已有充分来源时"
                "不要额外搜索。每次围绕缺失的信息做聚焦查询，避免重复宽泛改词。"
                "涉及相对日期先用 get_current_time 确定基准。搜索命中是来源线索，"
                "不代表已读取完整网页或结论已验证；已知 URL 需要正文时用 http_request，"
                "发现工具能力用 tool_search。引用实际支持结论的返回来源。它只读且无需审批，"
                "成功返回不代表资料搜集或用户任务已经完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "围绕待核实事实的聚焦查询，包含必要主题、时间或来源条件；不完整复述用户任务。",
                        "maxLength": MAX_QUERY_CHARS,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "本次最多返回的来源数量，不是已验证来源数量；按需求取值，受当前配置上限限制。",
                        "default": self._max_results,
                        "minimum": 1,
                        "maximum": self._max_results,
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news", "finance"],
                        "default": "general",
                        "description": "搜索主题，默认 general；需要新闻或金融来源时选 news/finance，分类支持取决于实际 provider。",
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "仅需近期信息时指定；过滤支持取决于实际 provider，仍须核对来源发布日期，不当作完整历史。",
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                        "description": "可选优先限定的来源域名列表，例如 example.com；只在有明确来源要求时填写。",
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                        "description": "可选排除的来源域名列表；用于避开明确不适用的来源，不填写搜索词或页面正文。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
            permission=ToolPermission.ALLOWED,
        )

    # 函数说明：WebSearchTool.provider_name
    # 用途：返回 `self._service.primary_provider`，提供 WebSearchTool 的派生值。
    # 返回：类型 `str`；返回 `self._service.primary_provider`。
    @property
    def provider_name(self) -> str:

        return self._service.primary_provider

    # 函数说明：WebSearchTool.execute
    # 用途：执行WebSearchTool，供内置工作区工具使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `max_results`。
    # 返回：类型 `dict[str, Any]`；返回 `output`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SearchRequest.model_validate` →
    # `self._service.search`。
    # 分支与异常：
    #   当 `isinstance(raw_max_results, bool) or not isinstance(…` 时，抛出
    # `ValueError("'max_results' must be an integer")`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_max_results = arguments.get("max_results", self._max_results)
        if isinstance(raw_max_results, bool) or not isinstance(raw_max_results, int):
            raise ValueError("'max_results' must be an integer")

        request = SearchRequest.model_validate(
            {
                **arguments,
                "max_results": min(raw_max_results, self._max_results),
            }
        )
        response = await self._service.search(request)
        output = response.model_dump(mode="json")
        output["count"] = len(response.results)
        return output


__all__ = ["MAX_QUERY_CHARS", "WebSearchTool"]
