
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog

from app.models.types import ToolPermission, ToolResult

from .hooks import ToolExecutionContext, ToolHook


# 函数说明：_now_iso
# 用途：获取当前时间并转换为 ISO 格式文本。
# 返回：类型 `str`；返回 `datetime.now(UTC).isoformat()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now(UTC).isoformat` →
# `datetime.now`。
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ToolExecutionRecord:

    __slots__ = (
        "id",
        "tool_call_id",
        "tool_name",
        "permission",
        "started_at",
        "duration_ms",
        "success",
        "output",
        "error",
    )

    # 函数说明：ToolExecutionRecord.__init__
    # 用途：初始化 ToolExecutionRecord；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   tool_call_id：工具调用标识，类型 `str`。
    #   tool_name：工具名称，类型 `str`。
    #   permission：所需权限等级，类型 `ToolPermission | str`。
    #   started_at：开始时间，用于计算时长，类型 `str`。
    #   duration_ms：传给 `round` 的输入，类型 `float`。
    #   success：执行是否成功，类型 `bool`。
    #   output：工具、模型或转换步骤的输出，类型 `str | None`；默认 `None`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`uuid4` → `round`。
    # 副作用与资源：
    #   更新对象字段：`self.id`、`self.tool_call_id`、`self.tool_name`、
    # `self.permission`、`self.started_at`、`self.duration_ms`、`self.success`、
    # `self.output` 等 9 个字段。
    def __init__(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        permission: ToolPermission | str,
        started_at: str,
        duration_ms: float,
        success: bool,
        output: str | None = None,
        error: str | None = None,
    ) -> None:
        self.id = uuid4().hex
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.permission = (
            permission.value if isinstance(permission, ToolPermission) else permission
        )
        self.started_at = started_at
        self.duration_ms = round(duration_ms, 3)
        self.success = success
        self.output = output
        self.error = error

    # 函数说明：ToolExecutionRecord.to_dict
    # 用途：将当前记录转为字典载荷，具体公开字段及转换规则由返回表达式确定。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `id`、`tool_call_id`、`tool_name`、
    # `permission`、`started_at`、`duration_ms`、`success`、`output`、`error`。
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "permission": self.permission,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


class ToolExecutionLogger(ABC):

    # 函数说明：ToolExecutionLogger.record
    # 用途：记录ToolExecutionLogger，供工具注册、执行与权限钩子使用。
    # 参数：
    #   record：待处理的数据记录，类型 `ToolExecutionRecord`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    @abstractmethod
    def record(self, record: ToolExecutionRecord) -> None:
        pass


class InMemoryExecutionLogger(ToolExecutionLogger):

    # 函数说明：InMemoryExecutionLogger.__init__
    # 用途：初始化 InMemoryExecutionLogger；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   maxlen：`maxlen`输入或配置值，类型 `int`；默认 `200`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `maxlen < 1` 时，抛出 `ValueError('maxlen must be at least 1')`。
    # 副作用与资源：
    #   更新对象字段：`self._maxlen`、`self._records`。
    def __init__(self, maxlen: int = 200) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be at least 1")
        self._maxlen = maxlen
        self._records: list[ToolExecutionRecord] = []

    # 函数说明：InMemoryExecutionLogger.record
    # 用途：记录InMemoryExecutionLogger，供工具注册、执行与权限钩子使用。
    # 参数：
    #   record：待处理的数据记录，类型 `ToolExecutionRecord`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def record(self, record: ToolExecutionRecord) -> None:
        self._records.append(record)
        if len(self._records) > self._maxlen:
            del self._records[: len(self._records) - self._maxlen]

    # 函数说明：InMemoryExecutionLogger.records
    # 用途：返回 `tuple(self._records)`，提供 InMemoryExecutionLogger 的派生值。
    # 返回：类型 `tuple[ToolExecutionRecord, ...]`；返回 `tuple(self._records)`。
    @property
    def records(self) -> tuple[ToolExecutionRecord, ...]:
        return tuple(self._records)

    # 函数说明：InMemoryExecutionLogger.recent
    # 用途：返回 `tuple(self._records[-limit:])`，提供 InMemoryExecutionLogger 的派生值
    # 。
    # 参数：
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `10`。
    # 返回：类型 `tuple[ToolExecutionRecord, ...]`；返回 `tuple(self._records[-limit:])`
    # 。
    def recent(self, limit: int = 10) -> tuple[ToolExecutionRecord, ...]:
        return tuple(self._records[-limit:])

    # 函数说明：InMemoryExecutionLogger.count
    # 用途：统计InMemoryExecutionLogger，供工具注册、执行与权限钩子使用。
    # 返回：类型 `int`；返回 `len(self._records)`。
    @property
    def count(self) -> int:
        return len(self._records)

    # 函数说明：InMemoryExecutionLogger.clear
    # 用途：清理InMemoryExecutionLogger，供工具注册、执行与权限钩子使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._records.clear`。
    def clear(self) -> None:
        self._records.clear()


class StructLogExecutionLogger(ToolExecutionLogger):

    # 函数说明：StructLogExecutionLogger.__init__
    # 用途：初始化 StructLogExecutionLogger；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   logger_name：传给 `structlog.get_logger` 的输入，类型 `str`；默认
    # `'muharness.tools'`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`structlog.get_logger`。
    # 副作用与资源：
    #   更新对象字段：`self._log`。
    def __init__(self, logger_name: str = "muharness.tools") -> None:
        self._log = structlog.get_logger(logger_name)

    # 函数说明：StructLogExecutionLogger.record
    # 用途：记录StructLogExecutionLogger，供工具注册、执行与权限钩子使用。
    # 参数：
    #   record：待处理的数据记录，类型 `ToolExecutionRecord`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`record.to_dict` →
    # `self._log.info` → `self._log.warning`。
    def record(self, record: ToolExecutionRecord) -> None:
        event = record.to_dict()
        if record.success:
            self._log.info("tool.executed", **event)
        else:
            self._log.warning(
                "tool.failed",
                **event,
                error_reason=record.error,
            )


class ObservabilityHook(ToolHook):

    # 函数说明：ObservabilityHook.__init__
    # 用途：初始化 ObservabilityHook；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   logger：日志或工具执行记录器，类型 `ToolExecutionLogger`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._logger`。
    def __init__(self, logger: ToolExecutionLogger) -> None:
        self._logger = logger

    # 函数说明：ObservabilityHook.after_execute
    # 用途：在 `execute` 前后执行 ObservabilityHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolExecutionRecord` →
    # `self._logger.record`。
    async def after_execute(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        permission = (
            context.tool_definition.permission
            if context.tool_definition is not None
            else ToolPermission.ALLOWED
        )
        record = ToolExecutionRecord(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            permission=permission,
            started_at=str(context.metadata["started_at"]),
            duration_ms=result.duration_ms,
            success=result.success,
            output=result.output,
            error=result.error,
        )
        self._logger.record(record)


__all__ = [
    "InMemoryExecutionLogger",
    "ObservabilityHook",
    "StructLogExecutionLogger",
    "ToolExecutionLogger",
    "ToolExecutionRecord",
]
