"""Map persisted Run rows to the existing public Trace model."""

from __future__ import annotations

from datetime import datetime

import aiosqlite

from .models import AgentRunTrace


# 函数说明：trace_from_row
# 用途：将数据库行解析为执行轨迹记录。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `run_id`、
# `conversation_id`、`status`、`started_at`、`completed_at`、`provider`、`model`、
# `steps`、`stop_reason`、`input_tokens`、`output_tokens`、`total_tokens`、`event_count`
# 。
# 返回：类型 `AgentRunTrace`；返回 `AgentRunTrace(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentRunTrace` →
# `datetime.fromisoformat`。
def trace_from_row(row: aiosqlite.Row) -> AgentRunTrace:
    return AgentRunTrace(
        run_id=row["run_id"],
        conversation_id=row["conversation_id"],
        status=row["status"],
        started_at=datetime.fromisoformat(row["started_at"]),
        completed_at=(
            datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None
        ),
        provider=row["provider"],
        model=row["model"],
        steps=row["steps"],
        stop_reason=row["stop_reason"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        event_count=row["event_count"],
    )
