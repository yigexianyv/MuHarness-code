
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.models.types import ToolDefinition, ToolPermission

from ..base import BaseTool


class CurrentTimeTool(BaseTool):

    definition = ToolDefinition(
        name="get_current_time",
        record_output=False,
        description=(
            "读取调用时的实际日期、时间、时区和时间戳。需要解释今天、明天、昨天、"
            "现在、近期或截止时间时，用它确定时间基准；已有足够新鲜的基准且时区未变时，"
            "不要重复查询，也不为与时间无关的任务调用。省略时区使用 MuHarness 进程本地时区。"
            "它只读且无需审批，不搜索新闻、不创建提醒。成功返回仅证明该时刻的时间基准，"
            "不表示任何定时任务已创建或完成。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "可选 IANA 时区名，例如 Asia/Shanghai、America/New_York；"
                        "省略或留空使用进程本地时区，不填写城市简称或猜测的偏移值。"
                    ),
                }
            },
            "additionalProperties": False,
        },
        strict=False,
        permission=ToolPermission.ALLOWED,
    )

    # 函数说明：CurrentTimeTool.execute
    # 用途：执行CurrentTimeTool，供内置工作区工具使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `timezone`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `datetime`、`date`、`time`、`timezone`
    # 、`utc_offset`、`unix_timestamp`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now().astimezone` →
    # `datetime.now` → `_local_timezone_name` → `ZoneInfo` → `current.isoformat` →
    # `current.date().isoformat`；另有 5 个调用点。
    # 分支与异常：
    #   当 `timezone is not None and (not isinstance(timezone, str))` 时，抛出
    # `TypeError('timezone 必须是字符串')`。
    #   捕获 `ZoneInfoNotFoundError` 后，转换或抛出
    # `ValueError(f'未知 IANA 时区: {normalized}')`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        timezone = arguments.get("timezone")
        if timezone is not None and not isinstance(timezone, str):
            raise TypeError("timezone 必须是字符串")

        if timezone is None or not timezone.strip():
            current = datetime.now().astimezone()
            timezone_name = _local_timezone_name(current)
        else:
            normalized = timezone.strip()
            try:
                zone = ZoneInfo(normalized)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"未知 IANA 时区: {normalized}") from exc
            current = datetime.now(zone)
            timezone_name = normalized

        return {
            "datetime": current.isoformat(timespec="seconds"),
            "date": current.date().isoformat(),
            "time": current.time().isoformat(timespec="seconds"),
            "timezone": timezone_name,
            "utc_offset": current.strftime("%z")[:3] + ":" + current.strftime("%z")[3:],
            "unix_timestamp": int(current.timestamp()),
        }


# 函数说明：_local_timezone_name
# 用途：在内置工作区工具中处理 `_local_timezone_name`，通过 `current.tzname` 完成首个内
# 部处理步骤。
# 参数：
#   current：当前值或状态，类型 `datetime`。
# 返回：类型 `str`；按分支返回 `key`；`current.tzname() or str(current.tzinfo)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`current.tzname`。
# 分支与异常：
#   当 `isinstance(key, str) and key` 时，返回 `key`。
def _local_timezone_name(current: datetime) -> str:
    key = getattr(current.tzinfo, "key", None)
    if isinstance(key, str) and key:
        return key
    return current.tzname() or str(current.tzinfo)


__all__ = ["CurrentTimeTool"]
