
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .hooks import ToolExecutionContext

TASK_ATTRIBUTION_RESOLVED_METADATA_KEY = "_task_attribution_resolved"


@dataclass(frozen=True, slots=True)
class RecordedToolOutput:

    id: str
    content_chars: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ToolOutputAttribution:

    task_id: str | None = None
    task_step_id: str | None = None


class ToolOutputAttributionResolver(Protocol):

    # 函数说明：ToolOutputAttributionResolver.resolve
    # 用途：解析或定位ToolOutputAttributionResolver，供工具注册、执行与权限钩子使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `ToolOutputAttribution`；不返回结果值（隐式 None）。
    async def resolve(
        self,
        conversation_id: str,
    ) -> ToolOutputAttribution:
        pass


# 函数说明：explicit_tool_attribution
# 用途：在工具注册、执行与权限钩子中处理 `explicit_tool_attribution`，通过
# `context.metadata.get` 完成首个内部处理步骤。
# 参数：
#   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
# 返回：类型 `ToolOutputAttribution | None`；按分支返回 `None`；
# `ToolOutputAttribution(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolOutputAttribution`。
# 分支与异常：
#   当 `not isinstance(task_id, str) or not task_id.strip()` 时，返回 `None`。
def explicit_tool_attribution(
    context: ToolExecutionContext,
) -> ToolOutputAttribution | None:
    """Read task identity supplied by the trusted runtime, never model arguments."""
    task_id = context.metadata.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        return None
    task_step_id = context.metadata.get("task_step_id")
    return ToolOutputAttribution(
        task_id=task_id.strip(),
        task_step_id=(
            task_step_id.strip()
            if isinstance(task_step_id, str) and task_step_id.strip()
            else None
        ),
    )


class ToolOutputRecorder(Protocol):

    # 函数说明：ToolOutputRecorder.record
    # 用途：记录ToolOutputRecorder，供工具注册、执行与权限钩子使用。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   content：内容正文，类型 `str`。
    # 返回：类型 `RecordedToolOutput | None`；不返回结果值（隐式 None）。
    async def record(
        self,
        context: ToolExecutionContext,
        content: str,
    ) -> RecordedToolOutput | None:
        pass


__all__ = [
    "RecordedToolOutput",
    "ToolOutputAttribution",
    "ToolOutputAttributionResolver",
    "ToolOutputRecorder",
    "TASK_ATTRIBUTION_RESOLVED_METADATA_KEY",
    "explicit_tool_attribution",
]
