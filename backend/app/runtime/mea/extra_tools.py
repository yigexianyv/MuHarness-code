"""长任务启动时可以额外授权给 Executor 的工具。

默认白名单之外，只有网络类工具和 MCP 工具可以由用户逐个勾选加入 EXECUTE；
artifact_publish 属于默认执行工具；记忆、任务写操作、自动化、历史检索和
tool_search 不在可选范围内。
Manager 和 Auditor 的白名单不能扩展。
"""

from __future__ import annotations

from collections.abc import Iterable

from app.tools.registry import ToolRegistry
from app.tools.role_boundary import EXECUTE_ALLOWED_TOOLS

EXECUTE_OPTIONAL_TOOLS = frozenset({"http_request", "web_search"})
MCP_TOOL_PREFIX = "mcp__"


class ExtraToolsError(ValueError):
    pass


# 函数说明：is_optional_executor_tool
# 用途：判断执行者工具是否满足当前实现的条件。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `bool`；返回
# `name in EXECUTE_OPTIONAL_TOOLS or name.startswith(MCP_TOOL_PREFIX)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`name.startswith`。
def is_optional_executor_tool(name: str) -> bool:
    return name in EXECUTE_OPTIONAL_TOOLS or name.startswith(MCP_TOOL_PREFIX)


# 函数说明：optional_executor_tools
# 用途：当前注册表里可以勾选的额外工具，供前端展示。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
# 返回：类型 `tuple[str, ...]`；返回 `tuple(sorted((name for name in registry.names() if
#  is_optional_executor_tool(name))))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.names` →
# `is_optional_executor_tool`。
def optional_executor_tools(registry: ToolRegistry) -> tuple[str, ...]:
    """当前注册表里可以勾选的额外工具，供前端展示。"""

    return tuple(sorted(name for name in registry.names() if is_optional_executor_tool(name)))


# 函数说明：validate_extra_tools
# 用途：去重排序后返回；包含未注册或不可选的工具时整体拒绝，不静默丢弃。
# 参数：
#   names：`names`输入或配置值，类型 `Iterable[str]`。
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
# 返回：类型 `tuple[str, ...]`；返回
# `tuple((name for name in normalized if name not in EXECUTE_ALLOWED_TOOLS))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.names` →
# `is_optional_executor_tool`。
# 分支与异常：
#   当 `rejected` 时，抛出 `ExtraToolsError(…)`。
def validate_extra_tools(names: Iterable[str], registry: ToolRegistry) -> tuple[str, ...]:
    """去重排序后返回；包含未注册或不可选的工具时整体拒绝，不静默丢弃。"""

    normalized = sorted({name.strip() for name in names if name and name.strip()})
    registered = set(registry.names())
    rejected = [
        name
        for name in normalized
        if name not in EXECUTE_ALLOWED_TOOLS
        and (not is_optional_executor_tool(name) or name not in registered)
    ]
    if rejected:
        raise ExtraToolsError("这些工具不能授权给长任务 Executor：" + "、".join(rejected))
    return tuple(name for name in normalized if name not in EXECUTE_ALLOWED_TOOLS)


__all__ = [
    "EXECUTE_OPTIONAL_TOOLS",
    "MCP_TOOL_PREFIX",
    "ExtraToolsError",
    "is_optional_executor_tool",
    "optional_executor_tools",
    "validate_extra_tools",
]
