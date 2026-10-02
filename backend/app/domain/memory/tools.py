
from __future__ import annotations

from typing import Any

from app.models.types import ToolDefinition
from app.tools.base import BaseTool
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry

from .manager import MemoryManager
from .prompts import MEMORY_WRITE_POLICY

DEFAULT_DEFERRED_MEMORY_TOOL_NAMES = frozenset(
    {"memory_list", "core_memory_update", "core_memory_remove"}
)

MEMORY_SEARCH_TOOL_NAME = "memory_search"


class MemoryReadTool(BaseTool):

    # 函数说明：MemoryReadTool.__init__
    # 用途：初始化 MemoryReadTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`。
    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # 函数说明：MemoryReadTool.definition
    # 用途：提供 MemoryReadTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_read",
            record_output=False,
            description=(
                "按 ID 读取一条 active 普通记忆的完整正文与 revision。仅在当前任务"
                "确实需要依赖该历史信息回答、决策或执行时使用；索引、召回候选、摘要"
                "和 memory_search 片段都不能替代正式读取。不为熟悉背景批量读取，"
                "已有且仍适用的完整读取结果可复用；发现相关条目用 memory_search，"
                "浏览目录用 memory_list。found=true 表示本次已取得完整记忆并记录"
                "一次访问，不证明历史内容仍符合当前环境或任务已经完成；found=false"
                "表示未取得可读条目，不能继续将候选内容当作已读取事实。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "与当前任务确实相关的记忆 ID，例如 M001；从索引或候选中选取，不编造 ID。",
                    },
                },
                "required": ["memory_id"],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：MemoryReadTool.execute
    # 用途：执行MemoryReadTool，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `memory_id`。
    # 返回：类型 `dict[str, Any]`；按分支返回 `{'found': False, 'memory_id': memory_id}`
    # ；`{'found': True, 'id': record.id, 'title': record.title, 'revision': record.
    # revision, '…`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._manager.read` →
    # `record.render_full`。
    # 分支与异常：
    #   当 `not isinstance(memory_id, str) or not memory_id.strip()` 时，抛出
    # `ValueError("'memory_id' must be a non-empty string")`。
    #   当 `record is None` 时，返回 `{'found': False, 'memory_id': memory_id}`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = arguments.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("'memory_id' must be a non-empty string")
        record = await self._manager.read(memory_id)
        if record is None:
            return {"found": False, "memory_id": memory_id}
        return {
            "found": True,
            "id": record.id,
            "title": record.title,
            "revision": record.revision,
            "content": record.render_full(),
        }


class MemorySearchTool(BaseTool):

    # 函数说明：MemorySearchTool.__init__
    # 用途：初始化 MemorySearchTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`。
    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # 函数说明：MemorySearchTool.definition
    # 用途：提供 MemorySearchTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_search",
            record_output=False,
            description=(
                "检索 active 普通记忆，发现与当前任务相关的历史候选；可使用关键词"
                "与语义检索，具体模式以返回值为准。仅在自动召回或现有索引线索不足、"
                "任务换题且确需历史信息时使用；已有明确相关 ID 就用 memory_read，"
                "不要重复搜索同一线索或为无关背景搜索。返回的标题、摘要、revision"
                "和片段仅供选择，不是可信完整记忆；依赖内容前必须 memory_read。"
                "available=true 只表示搜索可用，可能没有命中，也不计为读取；"
                "available=false 表示搜索不可用，应按已有索引线索选择必要条目，"
                "不能断言记忆不存在。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要找回的历史知识主题、关键词或简短问题；聚焦缺失信息，不机械复述整个任务。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "返回候选数量上限，默认 5；按选择相关记忆所需的数量设置，不表示要读取全部候选。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：MemorySearchTool.execute
    # 用途：执行MemorySearchTool，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `query`
    # 、`limit`。
    # 返回：类型 `dict[str, Any]`；按分支返回 `{'available': False, 'query': query, '
    # reason': result.degrade_reason or 'search index…`；`{'available': True, 'query':
    # query, 'mode': result.mode.value, 'results': [{'memory_id':…`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._manager.search`。
    # 分支与异常：
    #   当 `not isinstance(query, str) or not query.strip()` 时，抛出
    # `ValueError("'query' must be a non-empty string")`。
    #   当 `limit is not None and (not isinstance(limit, int) or limit…` 时，抛出
    # `ValueError("'limit' must be a positive integer")`。
    #   当 `result.mode.value == 'unavailable'` 时，返回
    # `{'available': False, 'query': query, 'reason':…`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        limit = arguments.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("'limit' must be a positive integer")
        result = await self._manager.search(query, limit=limit)
        if result.mode.value == "unavailable":
            return {
                "available": False,
                "query": query,
                "reason": result.degrade_reason or "search index unavailable",
                "hint": "fall back to Memory Index cues and memory_read",
            }
        return {
            "available": True,
            "query": query,
            "mode": result.mode.value,
            "results": [
                {
                    "memory_id": candidate.memory_id,
                    "title": candidate.title,
                    "summary": candidate.summary,
                    "revision": candidate.revision,
                    "snippet": candidate.snippet,
                    "matched_by": [
                        source
                        for source, hit in (
                            ("vector", candidate.matched_by_vector),
                            ("fts", candidate.matched_by_fts),
                        )
                        if hit
                    ],
                }
                for candidate in result.candidates
            ],
        }


class MemoryListTool(BaseTool):

    # 函数说明：MemoryListTool.__init__
    # 用途：初始化 MemoryListTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`。
    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # 函数说明：MemoryListTool.definition
    # 用途：提供 MemoryListTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_list",
            record_output=False,
            description=(
                "列出 active 普通记忆的 ID、标题与摘要，供浏览当前记忆目录。仅在确需"
                "了解有哪些条目且现有索引不够时使用；按主题找候选用 memory_search，"
                "有明确相关 ID 就用 memory_read，不重复列目录或逐条无脑读取。返回"
                "目录只表示发现条目，不包含完整正文、不算正式读取，也不证明任务完成。"
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：MemoryListTool.execute
    # 用途：执行MemoryListTool，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `memories`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._manager.list`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        records = await self._manager.list()
        return {
            "memories": [
                {
                    "id": record.id,
                    "title": record.title,
                    "summary": record.summary,
                }
                for record in records
            ]
        }


class MemoryCreateTool(BaseTool):

    # 函数说明：MemoryCreateTool.__init__
    # 用途：初始化 MemoryCreateTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`。
    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # 函数说明：MemoryCreateTool.definition
    # 用途：提供 MemoryCreateTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_create",
            record_output=False,
            description=(
                "创建一条普通长期记忆，用于运行结束后的记忆整理，主任务循环不承担"
                "普通记忆写入决策。仅保存有事实支持、未来跨会话仍有明显价值且尚未"
                "覆盖的独立知识；同主题已有条目优先 memory_read 后 memory_update。"
                "当前进度属于 Task，可复用流程属于 Skills；全局身份、长期偏好和安全/"
                "隐私约束属于 Core，不能用普通记忆兜底。成功返回 ID 与 revision 只"
                "表示新记忆已保存，不表示记录中的工作已执行或当前任务完成；容量不足"
                "时会失败，不为腾容量随意归档。"
                f"写入条件：{MEMORY_WRITE_POLICY}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "精简且可区分主题的完整标题；不要写成当前任务进度。",
                    },
                    "summary": {
                        "type": "string",
                        "description": "索引中的召回线索，概括主题及何时需要读取；摘要本身不能替代完整正文。",
                    },
                    "content": {
                        "type": "string",
                        "description": "有依据的完整跨会话知识，保留适用范围与关键约束；不写临时进度、原始工具输出或未经确认的推断。",
                    },
                },
                "required": ["title", "summary", "content"],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：MemoryCreateTool.execute
    # 用途：执行MemoryCreateTool，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `title`
    # 、`summary`、`content`。
    # 返回：类型 `dict[str, Any]`；返回 `result`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._manager.create_if_capacity`
    # 。
    # 分支与异常：
    #   当 `not isinstance(title, str) or not title.strip()` 时，抛出
    # `ValueError("'title' must be a non-empty string")`。
    #   当 `not isinstance(summary, str) or not summary.strip()` 时，抛出
    # `ValueError("'summary' must be a non-empty string")`。
    #   当 `not isinstance(content, str) or not content.strip()` 时，抛出
    # `ValueError("'content' must be a non-empty string")`。
    #   当 `record is None` 时，抛出 `ValueError('active memory capacity is full')`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        title = arguments.get("title")
        summary = arguments.get("summary")
        content = arguments.get("content")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("'title' must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("'summary' must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("'content' must be a non-empty string")
        record = await self._manager.create_if_capacity(
            title=title,
            summary=summary,
            content=content,
        )
        if record is None:
            raise ValueError("active memory capacity is full")
        result: dict[str, Any] = {
            "id": record.id,
            "title": record.title,
            "summary": record.summary,
            "revision": record.revision,
        }
        return result


class MemoryUpdateTool(BaseTool):

    # 函数说明：MemoryUpdateTool.__init__
    # 用途：初始化 MemoryUpdateTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`。
    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # 函数说明：MemoryUpdateTool.definition
    # 用途：提供 MemoryUpdateTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_update",
            record_output=False,
            description=(
                "按 expected_revision 全量替换一条 active 普通记忆的标题、摘要与正文，"
                "用于运行结束后的记忆整理。仅在同主题已有记忆且出现有依据的新增、"
                "纠正或替代信息时使用；先 memory_read 获取完整内容和 revision，保留"
                "仍有效的事实、否定条件与关键约束，不把增量片段当完整正文，不无变化"
                "地改写。独立新主题用 memory_create；Task、Skills 和 Core 信息不写入"
                "普通记忆。版本冲突时先重读再判断，不能盲目覆盖；updated=true 与新"
                "revision 只表示记忆替换成功，不证明实际工作完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "已通过 memory_read 读取、需要同主题补充或纠正的记忆 ID。",
                    },
                    "content": {
                        "type": "string",
                        "description": "替换后的完整正文；合入本次变化并保留所有仍有效的旧事实和约束，不只提交新增段落。",
                    },
                    "title": {
                        "type": "string",
                        "description": "替换后的完整标题；主题未变时保留恰当的原标题。",
                    },
                    "summary": {
                        "type": "string",
                        "description": "替换后的完整召回摘要，准确提示该记忆的主题与适用场景。",
                    },
                    "expected_revision": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "最近一次 memory_read 返回的 revision，用于防止覆盖并发修改；"
                            "不要猜测或自行递增。"
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "具体说明本次新增、纠正或替代的事实及依据，作为更新记录。",
                    },
                },
                "required": [
                    "memory_id",
                    "title",
                    "summary",
                    "content",
                    "reason",
                    "expected_revision",
                ],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：MemoryUpdateTool.execute
    # 用途：执行MemoryUpdateTool，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `memory_id`、`title`、`summary`、`content`、`reason`、`expected_revision`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `id`、`title`、`summary`、`revision`、
    # `updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._manager.update_if_revision`
    # 。
    # 分支与异常：
    #   当 `not isinstance(memory_id, str) or not memory_id.strip()` 时，抛出
    # `ValueError("'memory_id' must be a non-empty string")`。
    #   当 `not isinstance(title, str) or not title.strip()` 时，抛出
    # `ValueError("'title' must be a non-empty string")`。
    #   当 `not isinstance(summary, str) or not summary.strip()` 时，抛出
    # `ValueError("'summary' must be a non-empty string")`。
    #   当 `not isinstance(content, str) or not content.strip()` 时，抛出
    # `ValueError("'content' must be a non-empty string")`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = arguments.get("memory_id")
        title = arguments.get("title")
        summary = arguments.get("summary")
        content = arguments.get("content")
        reason = arguments.get("reason")
        expected_revision = arguments.get("expected_revision")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("'memory_id' must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("'title' must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("'summary' must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("'content' must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("'reason' must be a non-empty string")
        if not isinstance(expected_revision, int) or expected_revision < 1:
            raise ValueError("'expected_revision' must be a positive integer")
        record = await self._manager.update_if_revision(
            memory_id,
            expected_revision=expected_revision,
            title=title,
            summary=summary,
            content=content,
            reason=reason,
        )
        return {
            "id": record.id,
            "title": record.title,
            "summary": record.summary,
            "revision": record.revision,
            "updated": True,
        }


class MemoryArchiveTool(BaseTool):

    # 函数说明：MemoryArchiveTool.__init__
    # 用途：初始化 MemoryArchiveTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`。
    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # 函数说明：MemoryArchiveTool.definition
    # 用途：提供 MemoryArchiveTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_archive",
            record_output=False,
            description=(
                "将一条普通记忆移出 active 集合并保留归档内容，用于运行结束后的记忆"
                "整理。仅在有明确依据表明该条目过时、已被替代或不再需要时使用；"
                "不知道正文时先 memory_read，不为容量压力、低相似度或当前暂时无用"
                "而归档。仍有效但需纠正的条目用 memory_update，撤销 Core 条目用"
                "core_memory_remove。返回 status=archived 表示条目已在归档中，"
                "不再出现在 active 索引和召回中；这是保留历史的归档，不是删除，也"
                "不代表当前任务完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "有明确过时、替代或不再需要依据的普通记忆 ID。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "说明条目为何不再需要及相关依据，作为归档记录；容量不足本身不是理由。",
                    },
                },
                "required": ["memory_id", "reason"],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：MemoryArchiveTool.execute
    # 用途：执行MemoryArchiveTool，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `memory_id`、`reason`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `id`、`status`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._manager.archive`。
    # 分支与异常：
    #   当 `not isinstance(memory_id, str) or not memory_id.strip()` 时，抛出
    # `ValueError("'memory_id' must be a non-empty string")`。
    #   当 `not isinstance(reason, str) or not reason.strip()` 时，抛出
    # `ValueError("'reason' must be a non-empty string")`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = arguments.get("memory_id")
        reason = arguments.get("reason")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("'memory_id' must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("'reason' must be a non-empty string")
        record = await self._manager.archive(memory_id, reason=reason)
        return {"id": record.id, "status": record.status.value}


class CoreMemoryUpdateTool(BaseTool):

    # 函数说明：CoreMemoryUpdateTool.__init__
    # 用途：初始化 CoreMemoryUpdateTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`。
    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # 函数说明：CoreMemoryUpdateTool.definition
    # 用途：提供 CoreMemoryUpdateTool 的模型可见定义，包含名称、说明、参数结构及权限声明
    # 。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="core_memory_update",
            record_output=False,
            description=(
                "按稳定 key 创建或替换一条每次 Run 常驻的 Core Memory。仅用于当前"
                "用户明确表达的稳定身份、全局长期偏好或全局安全/隐私约束；换到完全"
                "无关的项目仍须适用才属于 Core。不从旧消息、助理回复、工具输出或"
                "推断保存；必须将当前用户明确原话逐字复制到 explicit_user_statement，"
                "工具会核对当前消息。项目架构、选型、路径、实现限制和历史决定属于"
                "普通记忆，由运行结束后的整理处理；进度属于 Task，流程属于 Skills。"
                "撤销已有 Core 条目用 core_memory_remove。只有成功返回 created 或"
                "updated 回执才能确认该 key 已保存；发现工具、尝试调用或拟好文本"
                "都不算保存，失败时不能承诺已经记住，也不表示当前任务完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "稳定的小写点分 key，例如 communication.language；同一全局"
                            "事实更新时复用已有 key，避免重复条目。"
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "从当前用户明确陈述提炼的精简全局长期事实或约束；不要加入项目信息或推断。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "说明该信息为何在无关项目中仍应常驻，以及本次创建或替换原因。",
                    },
                    "explicit_user_statement": {
                        "type": "string",
                        "description": "逐字复制当前用户消息中支持本次保存的明确长期陈述；不是转述、旧消息或模型生成的文字。",
                    },
                },
                "required": [
                    "key",
                    "value",
                    "reason",
                    "explicit_user_statement",
                ],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：CoreMemoryUpdateTool.execute
    # 用途：执行CoreMemoryUpdateTool，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("core_memory_update requires the current user message")

    # 函数说明：CoreMemoryUpdateTool.execute_with_context
    # 用途：执行上下文，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `key`、
    # `value`、`reason`、`explicit_user_statement`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `key`、`value`、`created`、`updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_string` →
    # `self._manager.upsert_core`。
    # 分支与异常：
    #   当 `not user_input or statement not in user_input` 时，抛出 `ValueError(…)`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        key = _required_string(arguments, "key")
        value = _required_string(arguments, "value")
        reason = _required_string(arguments, "reason")
        statement = _required_string(arguments, "explicit_user_statement")
        user_input = context.user_input
        if not user_input or statement not in user_input:
            raise ValueError(
                "'explicit_user_statement' must be copied exactly from the "
                "current user message"
            )
        entry, created = await self._manager.upsert_core(
            key=key,
            value=value,
            reason=reason,
            source_statement=statement,
        )
        return {
            "key": entry.key,
            "value": entry.value,
            "created": created,
            "updated": not created,
        }


class CoreMemoryRemoveTool(BaseTool):

    # 函数说明：CoreMemoryRemoveTool.__init__
    # 用途：初始化 CoreMemoryRemoveTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`。
    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # 函数说明：CoreMemoryRemoveTool.definition
    # 用途：提供 CoreMemoryRemoveTool 的模型可见定义，包含名称、说明、参数结构及权限声明
    # 。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="core_memory_remove",
            record_output=False,
            description=(
                "移除指定 key 的 Core Memory，使该全局条目不再常驻。仅在当前用户"
                "明确撤销已有稳定身份、全局长期偏好或安全/隐私约束时使用；必须把"
                "当前消息中的撤销原话逐字复制到 explicit_user_statement，工具会核对"
                "当前消息。不依据推断、旧消息或项目临时例外移除；修改仍成立的全局"
                "值用 core_memory_update，普通记忆归档用 memory_archive。removed=true"
                "只确认该 key 已移除，不表示其他记忆或历史记录被清除，也不表示任务"
                "完成；没有成功回执就不能声称撤销已保存。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "当前用户明确撤销的已有 Core 条目的稳定小写点分 key；不要猜测或用普通记忆 ID。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "说明本次明确撤销的内容和依据，会保留在 Tool/Trace 记录中。",
                    },
                    "explicit_user_statement": {
                        "type": "string",
                        "description": "逐字复制当前用户消息中的明确撤销原话；不是转述、旧消息或推断。",
                    },
                },
                "required": ["key", "reason", "explicit_user_statement"],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：CoreMemoryRemoveTool.execute
    # 用途：执行CoreMemoryRemoveTool，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("core_memory_remove requires the current user message")

    # 函数说明：CoreMemoryRemoveTool.execute_with_context
    # 用途：执行上下文，供长期记忆管理与检索使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `key`、
    # `reason`、`explicit_user_statement`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `key`、`removed`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_required_string` →
    # `self._manager.remove_core`。
    # 分支与异常：
    #   当 `not user_input or statement not in user_input` 时，抛出 `ValueError(…)`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        key = _required_string(arguments, "key")
        _required_string(arguments, "reason")
        statement = _required_string(arguments, "explicit_user_statement")
        user_input = context.user_input
        if not user_input or statement not in user_input:
            raise ValueError(
                "'explicit_user_statement' must be copied exactly from the "
                "current user message"
            )
        removed = await self._manager.remove_core(key)
        return {"key": removed.key, "removed": True}


# 函数说明：register_memory_tools
# 用途：注册记忆工具集合，供长期记忆管理与检索使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   manager：当前业务管理器，类型 `MemoryManager`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.register` → `MemoryReadTool`
#  → `MemorySearchTool` → `MemoryListTool` → `CoreMemoryUpdateTool` →
# `CoreMemoryRemoveTool`。
def register_memory_tools(
    registry: ToolRegistry,
    manager: MemoryManager,
) -> None:

    registry.register(MemoryReadTool(manager))
    registry.register(MemorySearchTool(manager))
    registry.register(MemoryListTool(manager))
    registry.register(CoreMemoryUpdateTool(manager))
    registry.register(CoreMemoryRemoveTool(manager))


# 函数说明：register_memory_write_tools
# 用途：注册记忆工具集合，供长期记忆管理与检索使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   manager：当前业务管理器，类型 `MemoryManager`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.register` →
# `MemoryCreateTool` → `MemoryUpdateTool` → `MemoryArchiveTool`。
def register_memory_write_tools(
    registry: ToolRegistry,
    manager: MemoryManager,
) -> None:

    registry.register(MemoryCreateTool(manager))
    registry.register(MemoryUpdateTool(manager))
    registry.register(MemoryArchiveTool(manager))


# 函数说明：_required_string
# 用途：提取并校验必需的字符串字段。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `str`；返回 `value.strip()`。
# 分支与异常：
#   当 `not isinstance(value, str) or not value.strip()` 时，抛出
# `ValueError(f"'{name}' must be a non-empty string")`。
def _required_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{name}' must be a non-empty string")
    return value.strip()


__all__ = [
    "DEFAULT_DEFERRED_MEMORY_TOOL_NAMES",
    "MEMORY_SEARCH_TOOL_NAME",
    "CoreMemoryRemoveTool",
    "CoreMemoryUpdateTool",
    "MemoryArchiveTool",
    "MemoryCreateTool",
    "MemoryListTool",
    "MemoryReadTool",
    "MemorySearchTool",
    "MemoryUpdateTool",
    "register_memory_tools",
    "register_memory_write_tools",
]
