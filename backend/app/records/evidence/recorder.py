
from __future__ import annotations

import logging
from hashlib import sha256

from app.tools.hooks import ToolExecutionContext
from app.tools.output import (
    TASK_ATTRIBUTION_RESOLVED_METADATA_KEY,
    RecordedToolOutput,
    ToolOutputAttribution,
    ToolOutputAttributionResolver,
    explicit_tool_attribution,
)

from .store import SQLiteEvidenceStore

logger = logging.getLogger("muharness.evidence.recorder")


class EvidenceRecorder:

    # 函数说明：EvidenceRecorder.__init__
    # 用途：初始化 EvidenceRecorder；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteEvidenceStore`。
    #   attribution_resolver：`attribution_resolver`输入或配置值，类型
    # `ToolOutputAttributionResolver | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`、`self._attribution_resolver`。
    def __init__(
        self,
        store: SQLiteEvidenceStore,
        *,
        attribution_resolver: ToolOutputAttributionResolver | None = None,
    ) -> None:
        self._store = store
        self._attribution_resolver = attribution_resolver

    # 函数说明：EvidenceRecorder.record
    # 用途：记录EvidenceRecorder，供原始工具证据持久化使用。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   content：内容正文，类型 `str`。
    # 返回：类型 `RecordedToolOutput | None`；按分支返回 `None`；`RecordedToolOutput(id=
    # record.id, content_chars=record.content_chars, sha256=record.sha256)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`explicit_tool_attribution` →
    # `ToolOutputAttribution` → `self._attribution_resolver.resolve` → `logger.warning`
    # → `sha256(content.encode('utf-8')).hexdigest` → `sha256`；另有 3 个调用点。
    # 分支与异常：
    #   当 `not context.run_id or not context.conversation_id` 时，返回 `None`。
    #   当 `definition is not None and (not definition.record_output)` 时，返回 `None`。
    #   捕获 `Exception` 后，执行异常处理调用 `logger.warning`、`type`。
    async def record(
        self,
        context: ToolExecutionContext,
        content: str,
    ) -> RecordedToolOutput | None:
        if not context.run_id or not context.conversation_id:
            return None
        tool_name = context.tool_call.name
        definition = context.tool_definition
        if definition is not None and not definition.record_output:
            return None
        explicit_attribution = explicit_tool_attribution(context)
        attribution = explicit_attribution or ToolOutputAttribution()
        if (
            explicit_attribution is None
            and not context.metadata.get(TASK_ATTRIBUTION_RESOLVED_METADATA_KEY)
            and self._attribution_resolver is not None
        ):
            try:
                attribution = await self._attribution_resolver.resolve(
                    context.conversation_id
                )
            except Exception as exc:
                logger.warning(
                    "Evidence attribution failed conversation_id=%s error=%s: %s",
                    context.conversation_id,
                    type(exc).__name__,
                    exc,
                )
        digest = sha256(content.encode("utf-8")).hexdigest()
        record = await self._store.create(
            conversation_id=context.conversation_id,
            run_id=context.run_id,
            tool_call_id=context.tool_call.id,
            tool_name=tool_name,
            content=content,
            sha256=digest,
            task_id=attribution.task_id,
            task_step_id=attribution.task_step_id,
        )
        return RecordedToolOutput(
            id=record.id,
            content_chars=record.content_chars,
            sha256=record.sha256,
        )


__all__ = [
    "EvidenceRecorder",
]
