
from __future__ import annotations

from typing import Any

from app.models.types import ToolDefinition, ToolPermission
from app.tools.base import BaseTool
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry

from .store import SQLiteEvidenceStore


class EvidenceSearchTool(BaseTool):

    # 函数说明：EvidenceSearchTool.__init__
    # 用途：初始化 EvidenceSearchTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteEvidenceStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: SQLiteEvidenceStore) -> None:
        self._store = store

    # 函数说明：EvidenceSearchTool.definition
    # 用途：提供 EvidenceSearchTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="evidence_search",
            record_output=False,
            description=(
                "搜索工具原文，定位当前会话已归档的输出，"
                "返回 Evidence ID、元数据和匹配片段。"
                "必要结果被裁剪、清理或摘要且没有已知 ID 时使用；已知 ID 直接"
                " evidence_read，会话讨论请用 history 工具，不重复搜索已有证据。"
                "命中仅表示发现候选，依赖具体正文前按需读取；无命中不证明原操作未执行，"
                "搜索成功也不表示任务完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "缺失输出中的短关键词、标识或错误文本，不复述完整用户任务。",
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "可选，限定产生归档输出的真实工具名，不是当前检索工具名。",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "可选，限定证据关联的真实任务 ID；不解析 current 别名。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                        "description": "最多返回的匹配记录数，默认 10，范围 1–20；不是正文读取长度。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
            strict=False,
        )

    # 函数说明：EvidenceSearchTool.execute
    # 用途：执行EvidenceSearchTool，供原始工具证据持久化使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("evidence_search requires conversation context")

    # 函数说明：EvidenceSearchTool.execute_with_context
    # 用途：执行上下文，供原始工具证据持久化使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `query`
    # 、`tool_name`、`task_id`、`limit`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `query`、`count`、`results`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_conversation` →
    # `self._store.search` → `_optional_text` → `_integer` →
    # `hit.record.created_at.isoformat`。
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
        hits = await self._store.search(
            conversation_id=conversation_id,
            query=query,
            tool_name=_optional_text(arguments.get("tool_name")),
            task_id=_optional_text(arguments.get("task_id")),
            limit=_integer(arguments.get("limit", 10), "limit"),
        )
        return {
            "query": query,
            "count": len(hits),
            "results": [
                {
                    "evidence_id": hit.record.id,
                    "tool_name": hit.record.tool_name,
                    "run_id": hit.record.run_id,
                    "content_chars": hit.record.content_chars,
                    "task_id": hit.record.task_id,
                    "task_step_id": hit.record.task_step_id,
                    "created_at": hit.record.created_at.isoformat(),
                    "snippet": hit.snippet,
                }
                for hit in hits
            ],
        }


class EvidenceReadTool(BaseTool):

    # 函数说明：EvidenceReadTool.__init__
    # 用途：初始化 EvidenceReadTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteEvidenceStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: SQLiteEvidenceStore) -> None:
        self._store = store

    # 函数说明：EvidenceReadTool.definition
    # 用途：提供 EvidenceReadTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="evidence_read",
            record_output=False,
            description=(
                "读取证据正文，按已知 Evidence ID 取回当前会话归档的原始工具输出，"
                "用于核实被裁剪"
                "结果中的必要事实。未知 ID 先 evidence_search；只读所需部分，不遍历"
                "无关证据。offset/limit 按字符分页，next_offset 非空表示尚有后文。"
                "found=true 只确认该页读取成功，不能代替判断原工具是否成功或当前任务"
                "是否完成；跨会话证据表现为不存在，found=false 不能证明操作从未发生。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "description": "现有记录中的完整 Evidence ID，或至少 4 位的唯一前缀；不猜测 ID。",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "正文起始字符偏移，默认 0；续读使用返回的 next_offset，不是字节或行号。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12000,
                        "default": 6000,
                        "description": "本页最多读取的字符数，默认 6000，范围 1–12000；按核实所需控制。",
                    },
                },
                "required": ["evidence_id"],
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
            strict=False,
        )

    # 函数说明：EvidenceReadTool.execute
    # 用途：执行EvidenceReadTool，供原始工具证据持久化使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("evidence_read requires conversation context")

    # 函数说明：EvidenceReadTool.execute_with_context
    # 用途：执行上下文，供原始工具证据持久化使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `evidence_id`、`offset`、`limit`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；按分支返回
    # `{'found': False, 'evidence_id': identifier}`；`{'found': True, 'evidence_id':
    # document.record.id, 'tool_name': document.record.tool_name…`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_conversation` →
    # `_integer` → `self._store.resolve`。
    # 分支与异常：
    #   当 `not isinstance(identifier, str) or not identifier.strip()` 时，抛出
    # `ValueError("'evidence_id' must be a non-empty string")`。
    #   当 `offset < 0` 时，抛出 `ValueError('offset cannot be negative')`。
    #   当 `limit < 1 or limit > 12000` 时，抛出
    # `ValueError('limit must be between 1 and 12000')`。
    #   当 `document is None` 时，返回 `{'found': False, 'evidence_id': identifier}`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        conversation_id = _require_conversation(context)
        identifier = arguments.get("evidence_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("'evidence_id' must be a non-empty string")
        offset = _integer(arguments.get("offset", 0), "offset")
        limit = _integer(arguments.get("limit", 6000), "limit")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if limit < 1 or limit > 12000:
            raise ValueError("limit must be between 1 and 12000")
        document = await self._store.resolve(
            identifier,
            conversation_id=conversation_id,
        )
        if document is None:
            return {"found": False, "evidence_id": identifier}
        content = document.content[offset : offset + limit]
        next_offset = offset + len(content)
        return {
            "found": True,
            "evidence_id": document.record.id,
            "tool_name": document.record.tool_name,
            "run_id": document.record.run_id,
            "task_id": document.record.task_id,
            "task_step_id": document.record.task_step_id,
            "sha256": document.record.sha256,
            "content_chars": document.record.content_chars,
            "offset": offset,
            "content": content,
            "next_offset": (
                next_offset
                if next_offset < document.record.content_chars
                else None
            ),
        }


# 函数说明：register_evidence_tools
# 用途：注册原始证据工具集合，供原始工具证据持久化使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   store：持久化存储依赖，类型 `SQLiteEvidenceStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.register` →
# `EvidenceSearchTool` → `EvidenceReadTool`。
def register_evidence_tools(
    registry: ToolRegistry,
    store: SQLiteEvidenceStore,
) -> None:
    registry.register(EvidenceSearchTool(store), deferred=True)
    registry.register(EvidenceReadTool(store), deferred=True)


# 函数说明：_require_conversation
# 用途：获取并校验必需的会话，供原始工具证据持久化使用。
# 参数：
#   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
# 返回：类型 `str`；返回 `context.conversation_id`。
# 分支与异常：
#   当 `not context.conversation_id` 时，抛出
# `ValueError('evidence tool requires conversation context')`。
def _require_conversation(context: ToolExecutionContext) -> str:
    if not context.conversation_id:
        raise ValueError("evidence tool requires conversation context")
    return context.conversation_id


# 函数说明：_optional_text
# 用途：在原始工具证据持久化中处理 `_optional_text`，通过 `value.strip` 完成首个内部处理
# 步骤。
# 参数：
#   value：待校验、规范化或转换的值，类型 `object`。
# 返回：类型 `str | None`；按分支返回 `None`；`value.strip() or None`。
# 分支与异常：
#   当 `value is None` 时，返回 `None`。
#   当 `not isinstance(value, str)` 时，抛出
# `TypeError('optional filter must be a string')`。
def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional filter must be a string")
    return value.strip() or None


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


__all__ = ["EvidenceReadTool", "EvidenceSearchTool", "register_evidence_tools"]
