
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.models.types import ToolDefinition

from .base import BaseTool
from .registry import ToolRegistry

TOOL_SEARCH_NAME = "tool_search"
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")


@dataclass(frozen=True, slots=True)
class ToolCatalogMatch:

    name: str
    description: str
    parameter_names: tuple[str, ...]
    score: int

    # 函数说明：ToolCatalogMatch.as_dict
    # 用途：将当前记录转为字典载荷，具体公开字段及转换规则由返回表达式确定。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `name`、`description`、`parameters`。
    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": list(self.parameter_names),
        }


class ToolCatalog:

    # 函数说明：ToolCatalog.__init__
    # 用途：初始化 ToolCatalog；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._registry`。
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # 函数说明：ToolCatalog.search
    # 用途：按查询词查找延迟暴露工具，返回可供模型选择的目录项。
    # 参数：
    #   query：检索查询文本，类型 `str`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `5`。
    # 返回：类型 `tuple[ToolCatalogMatch, ...]`；返回 `tuple(matches[:limit])`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`query.casefold` →
    # `self._registry.deferred_names` → `definition.permission.model_visible` →
    # `_relevance_score` → `ToolCatalogMatch` → `_compact_text`；另有 1 个调用点。
    # 分支与异常：
    #   当 `not normalized_query` 时，抛出 `ValueError('query 不能为空')`。
    #   当 `not 1 <= limit <= 5` 时，抛出 `ValueError('limit 必须在 1 到 5 之间')`。
    #   当 `not definition.permission.model_visible()` 时，跳过当前循环项。
    #   当 `score <= 0` 时，跳过当前循环项。
    def search(self, query: str, *, limit: int = 5) -> tuple[ToolCatalogMatch, ...]:
        """按查询词查找延迟暴露工具，返回可供模型选择的目录项。"""
        normalized_query = " ".join(query.casefold().split())
        if not normalized_query:
            raise ValueError("query 不能为空")
        if not 1 <= limit <= 5:
            raise ValueError("limit 必须在 1 到 5 之间")

        matches: list[ToolCatalogMatch] = []
        for name in self._registry.deferred_names():
            definition = self._registry.get(name).definition
            if not definition.permission.model_visible():
                continue
            score = _relevance_score(normalized_query, definition)
            if score <= 0:
                continue
            properties = definition.parameters.get("properties", {})
            parameter_names = (
                tuple(str(key) for key in properties)
                if isinstance(properties, dict)
                else ()
            )
            matches.append(
                ToolCatalogMatch(
                    name=name,
                    description=_compact_text(definition.description, max_chars=500),
                    parameter_names=parameter_names,
                    score=score,
                )
            )
        matches.sort(key=lambda item: (-item.score, item.name))
        return tuple(matches[:limit])


class ToolSearchTool(BaseTool):

    definition = ToolDefinition(
        name=TOOL_SEARCH_NAME,
        record_output=False,
        description=(
            "按需发现尚未向模型展示的能力（capability discovery）。仅当当前可见"
            "工具无法满足必要操作、确实缺少额外能力时搜索；已知可用工具就直接调用，"
            "不要把搜索当作每轮必经步骤。query 用简短能力关键词，可补英文同义词"
            "以匹配第三方说明，不复述完整任务。命中只表示工具定义会在下一轮加载、"
            "届时可按参数调用，不表示能力已执行或任务完成。已有命中不重复搜；"
            "无命中时只在能明确改进关键词时再试，不做无意义试探。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "所缺能力的短关键词，如 weather forecast 或 core memory update；"
                        "可含英文同义词，不填完整任务、长背景或敏感原文。"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "本次最多发现的工具数，默认 5，范围 1–5；按所缺能力控制候选量。",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    # 函数说明：ToolSearchTool.__init__
    # 用途：初始化 ToolSearchTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCatalog`。
    # 副作用与资源：
    #   更新对象字段：`self._catalog`。
    def __init__(self, registry: ToolRegistry) -> None:
        self._catalog = ToolCatalog(registry)

    # 函数说明：ToolSearchTool.execute
    # 用途：执行ToolSearchTool，供工具注册、执行与权限钩子使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `query`
    # 、`limit`。
    # 返回：类型 `str`；返回 `json.dumps(payload, ensure_ascii=False)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._catalog.search` →
    # `match.as_dict` → `json.dumps`。
    # 分支与异常：
    #   当 `not isinstance(query, str)` 时，抛出 `TypeError('query 必须是字符串')`。
    #   当 `isinstance(limit, bool) or not isinstance(limit, int)` 时，抛出
    # `TypeError('limit 必须是整数')`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str):
            raise TypeError("query 必须是字符串")
        limit = arguments.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit 必须是整数")
        matches = self._catalog.search(query, limit=limit)
        payload = {
            "query": query,
            "count": len(matches),
            "tools": [match.as_dict() for match in matches],
            "hint": (
                "这些工具已激活，可在下一步直接调用。"
                if matches
                else "没有匹配工具，请换用更接近工具名称或英文描述的关键词。"
            ),
        }
        return json.dumps(payload, ensure_ascii=False)


# 函数说明：ensure_tool_search_registered
# 用途：当存在延迟暴露工具时注册工具搜索入口。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.deferred_names` →
# `registry.register` → `ToolSearchTool`。
# 分支与异常：
#   当 `not registry.deferred_names()` 时，返回 `None`。
#   捕获 `KeyError` 后，返回 `None`。
#   当 `not isinstance(existing, ToolSearchTool)` 时，抛出
# `ValueError(f"Tool name '{TOOL_SEARCH_NAME}' is reserved.")`。
def ensure_tool_search_registered(registry: ToolRegistry) -> None:

    """当存在延迟暴露工具时注册工具搜索入口。"""
    if not registry.deferred_names():
        return
    try:
        existing = registry.get(TOOL_SEARCH_NAME)
    except KeyError:
        registry.register(ToolSearchTool(registry))
        return
    if not isinstance(existing, ToolSearchTool):
        raise ValueError(f"Tool name '{TOOL_SEARCH_NAME}' is reserved.")


# 函数说明：activated_tool_names
# 用途：在工具注册、执行与权限钩子中处理 `activated_tool_names`，通过 `json.loads` 完成
# 首个内部处理步骤。
# 参数：
#   output：工具、模型或转换步骤的输出，类型 `str | None`。
# 返回：类型 `tuple[str, ...]`；按分支返回 `()`；`tuple(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads`。
# 分支与异常：
#   当 `not output` 时，返回 `()`。
#   捕获 `(AttributeError, TypeError, ValueError, json.JSONDecodeError)` 后，返回 `()`。
def activated_tool_names(output: str | None) -> tuple[str, ...]:

    if not output:
        return ()
    try:
        payload = json.loads(output)
        tools = payload.get("tools", [])
        return tuple(
            item["name"]
            for item in tools
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        )
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return ()


# 函数说明：_relevance_score
# 用途：在工具注册、执行与权限钩子中处理 `_relevance_score`，通过
# `definition.parameters.get` 完成首个内部处理步骤。
# 参数：
#   query：检索查询文本，类型 `str`。
#   definition：工具定义输入或配置值，类型 `ToolDefinition`。
# 返回：类型 `int`；返回 `score`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`definition.name.casefold` →
# `definition.description.casefold` → `parameter_text.casefold` → `_query_tokens`。
def _relevance_score(query: str, definition: ToolDefinition) -> int:
    properties = definition.parameters.get("properties", {})
    parameter_text = ""
    if isinstance(properties, dict):
        parameter_text = " ".join(
            f"{name} {value.get('description', '') if isinstance(value, dict) else ''}"
            for name, value in properties.items()
        )
    name = definition.name.casefold()
    description = definition.description.casefold()
    searchable = f"{name} {description} {parameter_text.casefold()}"
    score = 0
    if query in searchable:
        score += 30
    for token in _query_tokens(query):
        if token in name:
            score += 12
        elif token in description:
            score += 6
        elif token in searchable:
            score += 3
    return score


# 函数说明：_query_tokens
# 用途：在工具注册、执行与权限钩子中处理 `_query_tokens`，通过 `_ASCII_TOKEN_RE.findall`
#  完成首个内部处理步骤。
# 参数：
#   query：检索查询文本，类型 `str`。
# 返回：类型 `tuple[str, ...]`；返回
# `tuple(dict.fromkeys((token for token in tokens if len(token) >= 2)))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_ASCII_TOKEN_RE.findall` →
# `_CJK_RE.findall` → `dict.fromkeys`。
def _query_tokens(query: str) -> tuple[str, ...]:
    tokens = list(_ASCII_TOKEN_RE.findall(query))
    for sequence in _CJK_RE.findall(query):
        if len(sequence) <= 2:
            tokens.append(sequence)
        else:
            tokens.extend(
                sequence[index : index + 2]
                for index in range(len(sequence) - 1)
            )
    return tuple(dict.fromkeys(token for token in tokens if len(token) >= 2))


# 函数说明：_compact_text
# 用途：压缩文本，供工具注册、执行与权限钩子使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
#   max_chars：保留的字符数上限，类型 `int`。
# 返回：类型 `str`；按分支返回 `compacted`；`f'{compacted[:max_chars]}…'`。
# 分支与异常：
#   当 `len(compacted) <= max_chars` 时，返回 `compacted`。
def _compact_text(value: str, *, max_chars: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= max_chars:
        return compacted
    return f"{compacted[:max_chars]}…"


__all__ = [
    "TOOL_SEARCH_NAME",
    "ToolCatalog",
    "ToolCatalogMatch",
    "ToolSearchTool",
    "activated_tool_names",
    "ensure_tool_search_registered",
]
