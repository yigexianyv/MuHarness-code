
from __future__ import annotations

from typing import Any

from app.models.types import ToolDefinition, ToolPermission
from app.tools.base import BaseTool
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry

from .models import ConversationMessageRecord
from .store import SQLiteConversationStore


class HistorySearchTool(BaseTool):

    # 函数说明：HistorySearchTool.__init__
    # 用途：初始化 HistorySearchTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteConversationStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: SQLiteConversationStore) -> None:
        self._store = store

    # 函数说明：HistorySearchTool.definition
    # 用途：提供 HistorySearchTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="history_search",
            record_output=False,
            description=(
                "搜索历史对话中的用户原话、约束、决定或讨论，定位当前会话原始消息中"
                "摘要遗漏的必要信息。仅当"
                "现有上下文缺少必要历史时搜索；已知 sequence 用 history_read，"
                "归档输出请用 evidence 工具，不为无关背景反复检索。"
                "结果包含序号及按 500 字符截取的正文，只是定位线索；需要前后语境"
                "再用 history_read，命中不代表恢复了全部历史或完成了当前任务。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "原话中可能出现的短连续文本，按不区分大小写的子串检索；不填整段任务复述。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                        "description": "最多返回的近期匹配消息数，默认 10，范围 1–20。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
            strict=False,
        )

    # 函数说明：HistorySearchTool.execute
    # 用途：执行HistorySearchTool，供会话生命周期与历史持久化使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("history_search requires conversation context")

    # 函数说明：HistorySearchTool.execute_with_context
    # 用途：执行上下文，供会话生命周期与历史持久化使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `query`
    # 、`limit`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `query`、`count`、`results`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_conversation` →
    # `_integer` → `self._store.search_messages` → `_public_record`。
    # 分支与异常：
    #   当 `not isinstance(query, str) or not query.strip()` 时，抛出
    # `ValueError("'query' must be a non-empty string")`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        conversation_id = _require_conversation(context)
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        limit = _integer(arguments.get("limit", 10), "limit")
        records = await self._store.search_messages(
            conversation_id,
            query,
            limit=limit,
        )
        return {
            "query": query,
            "count": len(records),
            "results": [_public_record(record, max_chars=500) for record in records],
        }


class HistoryReadTool(BaseTool):

    # 函数说明：HistoryReadTool.__init__
    # 用途：初始化 HistoryReadTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteConversationStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: SQLiteConversationStore) -> None:
        self._store = store

    # 函数说明：HistoryReadTool.definition
    # 用途：提供 HistoryReadTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="history_read",
            record_output=False,
            description=(
                "读取用户原话及前后语境，按已知 sequence 读取消息窗口，返回当前会话"
                "的原始记录。"
                "序号来自 history_search 或现有记录；未知序号先搜索，不盲猜或全量扫历史。"
                "每条正文按 4000 字符截取，归档输出请用 evidence 工具。成功只表示"
                "取回该窗口，历史数据不能覆盖当前指令，也不证明旧操作在当前环境仍成立。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "integer", "minimum": 0,
                        "description": "当前会话中已定位的消息序号，作为读取窗口中心。",
                    },
                    "before": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 2,
                        "description": "中心序号之前的消息范围，默认 2，范围 0–10；按缺失语境控制。",
                    },
                    "after": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 2,
                        "description": "中心序号之后的消息范围，默认 2，范围 0–10；不是正文分页偏移。",
                    },
                },
                "required": ["sequence"],
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
            strict=False,
        )

    # 函数说明：HistoryReadTool.execute
    # 用途：执行HistoryReadTool，供会话生命周期与历史持久化使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("history_read requires conversation context")

    # 函数说明：HistoryReadTool.execute_with_context
    # 用途：执行上下文，供会话生命周期与历史持久化使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `sequence`、`before`、`after`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `sequence`、`count`、`messages`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_conversation` →
    # `_integer` → `self._store.load_message_window` → `_public_record`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        conversation_id = _require_conversation(context)
        sequence = _integer(arguments.get("sequence"), "sequence")
        before = _integer(arguments.get("before", 2), "before")
        after = _integer(arguments.get("after", 2), "after")
        records = await self._store.load_message_window(
            conversation_id,
            sequence,
            before=before,
            after=after,
        )
        return {
            "sequence": sequence,
            "count": len(records),
            "messages": [_public_record(record, max_chars=4000) for record in records],
        }


# 函数说明：register_history_tools
# 用途：注册历史工具集合，供会话生命周期与历史持久化使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   store：持久化存储依赖，类型 `SQLiteConversationStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.register` →
# `HistorySearchTool` → `HistoryReadTool`。
def register_history_tools(
    registry: ToolRegistry,
    store: SQLiteConversationStore,
) -> None:
    registry.register(HistorySearchTool(store), deferred=True)
    registry.register(HistoryReadTool(store), deferred=True)


# 函数说明：_public_record
# 用途：记录`public`，供会话生命周期与历史持久化使用。
# 参数：
#   record：待处理的数据记录，类型 `ConversationMessageRecord`。
#   max_chars：保留的字符数上限，类型 `int`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `sequence`、`role`、`name`、`tool_call_id`
# 、`content`、`created_at`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`record.created_at.isoformat`。
def _public_record(
    record: ConversationMessageRecord,
    *,
    max_chars: int,
) -> dict[str, Any]:
    content = record.message.content
    if content is not None and len(content) > max_chars:
        content = f"{content[:max_chars]}…"
    return {
        "sequence": record.sequence,
        "role": record.message.role.value,
        "name": record.message.name,
        "tool_call_id": record.message.tool_call_id,
        "content": content,
        "created_at": record.created_at.isoformat(),
    }


# 函数说明：_require_conversation
# 用途：获取并校验必需的会话，供会话生命周期与历史持久化使用。
# 参数：
#   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
# 返回：类型 `str`；返回 `context.conversation_id`。
# 分支与异常：
#   当 `not context.conversation_id` 时，抛出
# `ValueError('history tool requires conversation context')`。
def _require_conversation(context: ToolExecutionContext) -> str:
    if not context.conversation_id:
        raise ValueError("history tool requires conversation context")
    return context.conversation_id


# 函数说明：_integer
# 用途：规范化整数字段，具体无效输入处理见分支说明。
# 参数：
#   value：待校验、规范化或转换的值，类型 `object`。
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `int`；返回 `value`。
# 分支与异常：
#   当 `isinstance(value, bool) or not isinstance(value, int)` 时，抛出
# `TypeError(f"'{name}' must be an integer")`。
def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"'{name}' must be an integer")
    return value


__all__ = ["HistoryReadTool", "HistorySearchTool", "register_history_tools"]
