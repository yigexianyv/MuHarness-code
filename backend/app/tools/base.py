
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from app.models.types import ToolDefinition

if TYPE_CHECKING:
    from .hooks import ToolExecutionContext


class BaseTool(ABC):

    # 函数说明：BaseTool.definition
    # 用途：提供 BaseTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；不返回结果值（隐式 None）。
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        pass

    # 函数说明：BaseTool.execute
    # 用途：执行BaseTool，供工具注册、执行与权限钩子使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        pass

    # 函数说明：BaseTool.execute_with_context
    # 用途：执行上下文，供工具注册、执行与权限钩子使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `Any`；返回 `await self.execute(arguments)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.execute`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any:

        return await self.execute(arguments)

    def execution_timeout(self, arguments: dict[str, Any]) -> float | None:
        """本次调用需要的执行时限（秒）；None 表示使用执行器的统一时限。

        自带超时参数的工具（如 Shell）在这里声明实际需要的时间，执行器据此放宽外层时限，
        避免外层先于工具自己的超时处理把调用取消。
        """
        del arguments
        return None
