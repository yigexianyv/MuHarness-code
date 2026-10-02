from __future__ import annotations

import logging
from typing import Any

from app.models.types import ToolDefinition, ToolPermission
from app.tools.base import BaseTool
from app.tools.hooks import ToolExecutionContext
from app.tools.output import (
    TASK_ATTRIBUTION_RESOLVED_METADATA_KEY,
    ToolOutputAttribution,
    ToolOutputAttributionResolver,
    explicit_tool_attribution,
)
from app.tools.registry import ToolRegistry

from .service import ArtifactService

logger = logging.getLogger("muharness.artifact.tools")


class ArtifactListTool(BaseTool):
    """Read committed publication records without publishing or mutating files."""

    # 函数说明：ArtifactListTool.__init__
    # 用途：初始化 ArtifactListTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   service：业务服务依赖，类型 `ArtifactService`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._service`。
    def __init__(self, service: ArtifactService) -> None:
        self._service = service

    # 函数说明：ArtifactListTool.definition
    # 用途：提供 ArtifactListTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="artifact_list",
            record_output=False,
            description=(
                "查询当前会话已提交的交付物发布记录，可按任务或运行筛选。仅在需要"
                "找回交付位置、核对发布状态或处理回执归档中断时使用；不枚举工作区"
                "文件，也不创建新发布，发布新交付物用 artifact_publish。文件记录含"
                " size_bytes/sha256，链接记录没有文件哈希。记录证明发布已提交，"
                "不证明原调用已返回、交付内容已验证或整个任务已完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "可选，按发布记录中的真实任务 ID 筛选；不是 current 别名。",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "可选，按产生发布记录的真实 Run ID 筛选。",
                    },
                    "limit": {
                        "type": "integer", "minimum": 1, "maximum": 200,
                        "default": 50,
                        "description": "最多返回的记录数，默认 50，范围 1–200；不代表全部历史。",
                    },
                },
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
            strict=False,
        )

    # 函数说明：ArtifactListTool.execute
    # 用途：执行ArtifactListTool，供交付物发布与存储使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("artifact_list requires conversation context")

    # 函数说明：ArtifactListTool.execute_with_context
    # 用途：执行上下文，供交付物发布与存储使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `limit`
    # 。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `record_source`、`count`、`artifacts`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._service.store.list` →
    # `artifact.public_dict`。
    # 分支与异常：
    #   当 `not context.conversation_id` 时，抛出
    # `ValueError('artifact_list requires conversation context')`。
    #   当 `set(arguments) - {'task_id', 'run_id', 'limit'}` 时，抛出
    # `ValueError('unsupported artifact_list arguments')`。
    #   当 `not isinstance(value, str) or not value.strip()` 时，抛出
    # `ValueError(f'{name} must be a non-empty string')`。
    #   当 `isinstance(limit, bool) or not isinstance(limit, int) or (…` 时，抛出
    # `ValueError('limit must be an integer between 1 and 200')`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if not context.conversation_id:
            raise ValueError("artifact_list requires conversation context")
        if set(arguments) - {"task_id", "run_id", "limit"}:
            raise ValueError("unsupported artifact_list arguments")
        filters: dict[str, str] = {}
        for name in ("task_id", "run_id"):
            value = arguments.get(name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a non-empty string")
                filters[name] = value.strip()
        limit = arguments.get("limit", 50)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("limit must be an integer between 1 and 200")
        artifacts = await self._service.store.list(
            conversation_id=context.conversation_id,
            limit=limit,
            **filters,
        )
        return {
            "record_source": "committed_artifact_store",
            "count": len(artifacts),
            "artifacts": [artifact.public_dict() for artifact in artifacts],
        }


class ArtifactPublishTool(BaseTool):
    # 函数说明：ArtifactPublishTool.__init__
    # 用途：初始化 ArtifactPublishTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   service：业务服务依赖，类型 `ArtifactService`。
    #   attribution_resolver：`attribution_resolver`输入或配置值，类型
    # `ToolOutputAttributionResolver | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._service`、`self._attribution_resolver`。
    def __init__(
        self,
        service: ArtifactService,
        *,
        attribution_resolver: ToolOutputAttributionResolver | None = None,
    ) -> None:
        self._service = service
        self._attribution_resolver = attribution_resolver

    # 函数说明：ArtifactPublishTool.definition
    # 用途：提供 ArtifactPublishTool 的模型可见定义，包含名称、说明、参数结构及权限声明
    # 。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="artifact_publish",
            record_output=True,
            description=(
                "将用户需要保留、下载或查看的最终文件或 http(s) 结果链接登记为交付物。"
                "已有实际交付物时在最终答复前使用；不发布中间文件、临时文件、Trace"
                " 或运行日志，也不以发布代替创建或验证内容。path 与 url 二选一："
                "文件会复制保存，成功回执含 size_bytes/sha256；链接只登记元数据，"
                "不下载、不验证可达性或生成文件哈希。成功表示该次发布已提交，"
                "不等于内容正确或任务通过；核对已提交记录用 artifact_list，勿重复发布试探。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "已存在的 workspace 内相对文件路径，不能是目录；与 url 二选一。",
                    },
                    "url": {
                        "type": "string",
                        "description": "最终交付的 http(s) 链接；只存元数据，不下载或检验内容，与 path 二选一。",
                    },
                    "title": {
                        "type": "string",
                        "description": "可选，便于用户识别的交付标题；省略时使用文件名或链接。",
                    },
                    "description": {
                        "type": "string",
                        "description": "可选，交付内容及用途的简短说明；不填未经验证的完成声明。",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
            strict=False,
            closing_allowed=True,
        )

    # 函数说明：ArtifactPublishTool.execute
    # 用途：执行ArtifactPublishTool，供交付物发布与存储使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("artifact_publish requires run context")

    # 函数说明：ArtifactPublishTool.execute_with_context
    # 用途：执行上下文，供交付物发布与存储使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `path`
    # 、`url`、`title`、`description`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；返回 `artifact.public_dict()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._resolve_attribution` →
    # `context.metadata.pop` → `self._service.publish_file` →
    # `self._service.publish_url` → `artifact.public_dict`。
    # 分支与异常：
    #   当 `not context.run_id` 时，抛出
    # `ValueError('artifact_publish requires run context')`。
    #   当 `unexpected` 时，抛出 `ValueError(…)`。
    #   当 `(path is None) == (url is None)` 时，抛出
    # `ValueError("provide exactly one of 'path' or 'url'")`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if not context.run_id:
            raise ValueError("artifact_publish requires run context")

        allowed_arguments = {"path", "url", "title", "description"}
        unexpected = sorted(set(arguments) - allowed_arguments)
        if unexpected:
            raise ValueError(
                "unsupported artifact_publish arguments: " + ", ".join(unexpected)
            )

        path = arguments.get("path")
        url = arguments.get("url")
        if (path is None) == (url is None):
            raise ValueError("provide exactly one of 'path' or 'url'")

        title = str(arguments.get("title") or "")
        description = arguments.get("description")
        description = str(description) if description is not None else None

        attribution = await self._resolve_attribution(context)
        task_id = attribution.task_id
        context.metadata[TASK_ATTRIBUTION_RESOLVED_METADATA_KEY] = True
        if task_id is not None:
            # The recorder receives this same per-call context, so artifact and
            # evidence keep one attribution decision even if task state changes.
            context.metadata["task_id"] = task_id
            if attribution.task_step_id is not None:
                context.metadata["task_step_id"] = attribution.task_step_id
            else:
                context.metadata.pop("task_step_id", None)
        else:
            context.metadata.pop("task_id", None)
            context.metadata.pop("task_step_id", None)

        if path is not None:
            artifact = await self._service.publish_file(
                path=str(path),
                title=title,
                description=description,
                run_id=context.run_id,
                conversation_id=context.conversation_id,
                task_id=task_id,
            )
        else:
            artifact = await self._service.publish_url(
                url=str(url),
                title=title,
                description=description,
                run_id=context.run_id,
                conversation_id=context.conversation_id,
                task_id=task_id,
            )
        return artifact.public_dict()

    # 函数说明：ArtifactPublishTool._resolve_attribution
    # 用途：解析或定位`attribution`，供交付物发布与存储使用。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `ToolOutputAttribution`；按分支返回 `explicit`；
    # `ToolOutputAttribution()`；`attribution`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`explicit_tool_attribution` →
    # `ToolOutputAttribution` → `self._attribution_resolver.resolve` → `logger.warning`
    # 。
    # 分支与异常：
    #   当 `explicit is not None` 时，返回 `explicit`。
    #   当 `self._attribution_resolver is None or not…` 时，返回
    # `ToolOutputAttribution()`。
    #   捕获 `Exception` 后，返回 `ToolOutputAttribution()`。
    async def _resolve_attribution(
        self,
        context: ToolExecutionContext,
    ) -> ToolOutputAttribution:
        explicit = explicit_tool_attribution(context)
        if explicit is not None:
            return explicit
        if self._attribution_resolver is None or not context.conversation_id:
            return ToolOutputAttribution()
        try:
            attribution = await self._attribution_resolver.resolve(
                context.conversation_id
            )
        except Exception as exc:
            # Task attribution enriches the artifact but must not make publishing
            # unavailable when the task store has a transient read failure.
            logger.warning(
                "Artifact attribution failed conversation_id=%s error=%s: %s",
                context.conversation_id,
                type(exc).__name__,
                exc,
            )
            return ToolOutputAttribution()
        return attribution


# 函数说明：register_artifact_tools
# 用途：注册交付物工具集合，供交付物发布与存储使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   service：业务服务依赖，类型 `ArtifactService`。
#   attribution_resolver：`attribution_resolver`输入或配置值，类型
# `ToolOutputAttributionResolver | None`；默认 `None`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.register` →
# `ArtifactPublishTool` → `ArtifactListTool`。
def register_artifact_tools(
    registry: ToolRegistry,
    service: ArtifactService,
    *,
    attribution_resolver: ToolOutputAttributionResolver | None = None,
) -> None:
    registry.register(
        ArtifactPublishTool(service, attribution_resolver=attribution_resolver)
    )
    registry.register(ArtifactListTool(service), deferred=True)


__all__ = ["ArtifactListTool", "ArtifactPublishTool", "register_artifact_tools"]
