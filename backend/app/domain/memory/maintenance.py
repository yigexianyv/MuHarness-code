
from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime

from .models import MemoryRecord


class MemoryMaintenance:

    # 函数说明：MemoryMaintenance.__init__
    # 用途：初始化 MemoryMaintenance；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   max_active：允许同时活跃的数量上限，类型 `int`；默认 `25`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `max_active <= 0` 时，抛出
    # `ValueError('max_active must be greater than zero')`。
    # 副作用与资源：
    #   更新对象字段：`self.max_active`。
    def __init__(self, *, max_active: int = 25) -> None:
        if max_active <= 0:
            raise ValueError("max_active must be greater than zero")
        self.max_active = max_active

    # 函数说明：MemoryMaintenance.exceeds_capacity
    # 用途：返回 `active_count > self.max_active`，提供 MemoryMaintenance 的派生值。
    # 参数：
    #   active_count：活跃项输入或配置值，类型 `int`。
    # 返回：类型 `bool`；返回 `active_count > self.max_active`。
    def exceeds_capacity(self, active_count: int) -> bool:

        return active_count > self.max_active

    # 函数说明：MemoryMaintenance.select_candidates
    # 用途：选取候选集合，供长期记忆管理与检索使用。
    # 参数：
    #   memories：传给 `sorted` 的输入，类型 `Sequence[MemoryRecord]`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `5`。
    # 返回：类型 `tuple[MemoryRecord, ...]`；按分支返回 `()`；`tuple(scored[:limit])`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now`。
    # 分支与异常：
    #   当 `limit <= 0` 时，返回 `()`。
    def select_candidates(
        self,
        memories: Sequence[MemoryRecord],
        *,
        limit: int = 5,
    ) -> tuple[MemoryRecord, ...]:

        if limit <= 0:
            return ()
        now = datetime.now(UTC)
        scored = sorted(
            memories,
            key=lambda record: _retention_score(record, now),
        )
        return tuple(scored[:limit])


# 函数说明：_retention_score
# 用途：在长期记忆管理与检索中处理 `_retention_score`，通过
# `(now - record.last_accessed_at).total_seconds` 完成首个内部处理步骤。
# 参数：
#   record：待处理的数据记录，类型 `MemoryRecord`。
#   now：`now`输入或配置值，类型 `datetime`。
# 返回：类型 `float`；返回 `-hours_since_accessed + access_score + -hours_since_updated`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `(now - record.last_accessed_at).total_seconds` →
# `(now - record.updated_at).total_seconds` → `math.log1p`。
def _retention_score(record: MemoryRecord, now: datetime) -> float:
    hours_since_accessed = max(
        0.0, (now - record.last_accessed_at).total_seconds() / 3600
    )
    hours_since_updated = max(
        0.0, (now - record.updated_at).total_seconds() / 3600
    )
    access_score = math.log1p(record.access_count)
    return -hours_since_accessed + access_score + -hours_since_updated


__all__ = ["MemoryMaintenance"]
