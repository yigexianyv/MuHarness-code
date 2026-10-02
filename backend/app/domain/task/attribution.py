
from __future__ import annotations

from app.tools.output import ToolOutputAttribution

from .models import TaskStepStatus
from .store import FileTaskStore


class TaskToolOutputAttributionResolver:

    # 函数说明：TaskToolOutputAttributionResolver.__init__
    # 用途：初始化 TaskToolOutputAttributionResolver；参数及实际保存的实例字段见下方说明
    # 。
    # 参数：
    #   store：持久化存储依赖，类型 `FileTaskStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: FileTaskStore) -> None:
        self._store = store

    # 函数说明：TaskToolOutputAttributionResolver.resolve
    # 用途：解析或定位TaskToolOutputAttributionResolver，供任务状态与步骤管理使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `ToolOutputAttribution`；按分支返回 `ToolOutputAttribution()`；`
    # ToolOutputAttribution(task_id=task.id, task_step_id=step.id if step is not None
    # else None)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._store.active_for_conversation` → `ToolOutputAttribution` → `next`。
    # 分支与异常：
    #   当 `task is None` 时，返回 `ToolOutputAttribution()`。
    async def resolve(self, conversation_id: str) -> ToolOutputAttribution:
        task = await self._store.active_for_conversation(conversation_id)
        if task is None:
            return ToolOutputAttribution()
        step = next(
            (
                item
                for item in task.steps
                if item.status is TaskStepStatus.IN_PROGRESS
            ),
            None,
        )
        return ToolOutputAttribution(
            task_id=task.id,
            task_step_id=step.id if step is not None else None,
        )


__all__ = ["TaskToolOutputAttributionResolver"]
