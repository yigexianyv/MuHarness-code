
"""MEA 三种角色模式的工具边界。

白名单是唯一依据：不在表里的工具（记忆、任务写操作、自动化、网络、MCP、tool_search 等）
在角色模式下一律不可见、不可调用。拒绝发生在两处：
- ``ToolRoundExecutor._rejection_reason``：模型发起调用时；
- ``RoleBoundaryHook``：挂在每个 ``ToolExecutor`` 上，覆盖绕过 ToolRoundExecutor 的调用路径。
两处的错误文本都以 ``ROLE_REJECTION_PREFIX`` 开头，runner 据此统计每个子 Run 的越权次数。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping

from app.models.types import AgentMode, ToolResult

from .hooks import ToolExecutionContext, ToolHook, ToolHookDecision

MANAGE_ALLOWED_TOOLS: frozenset[str] = frozenset()

EXECUTE_ALLOWED_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "write_file",
        "edit_file",
        "artifact_publish",
        "artifact_list",
        "run_shell_command",
        "evidence_search",
        "evidence_read",
        "skill_read",
        "skill_resource_read",
        "get_current_time",
    }
)

AUDIT_ALLOWED_TOOLS = frozenset(
    {
        "artifact_list",
        "read_file",
        "list_files",
        "run_shell_command",  # 在 AUDIT 模式下由 shell 工具自己改为只读挂载
        "evidence_search",
        "evidence_read",
        "task_get",
        "get_current_time",
    }
)

ROLE_ALLOWED_TOOLS: Mapping[AgentMode, frozenset[str]] = {
    AgentMode.MANAGE: MANAGE_ALLOWED_TOOLS,
    AgentMode.EXECUTE: EXECUTE_ALLOWED_TOOLS,
    AgentMode.AUDIT: AUDIT_ALLOWED_TOOLS,
}

ROLE_MODES = frozenset(ROLE_ALLOWED_TOOLS)

# 这些模式下工具不能改动 workspace：shell 用只读挂载，write_file 直接报错。
READ_ONLY_MODES = frozenset({AgentMode.MANAGE, AgentMode.AUDIT})

ROLE_REJECTION_PREFIX = "Role boundary:"


# 函数说明：is_role_mode
# 用途：判断角色模式是否满足当前实现的条件。
# 参数：
#   mode：Agent 执行模式或检索模式，类型 `AgentMode | None`。
# 返回：类型 `bool`；返回 `mode in ROLE_MODES`。
def is_role_mode(mode: AgentMode | None) -> bool:
    return mode in ROLE_MODES


# 函数说明：role_allows
# 用途：非角色模式（NORMAL、PLAN、未知）不由这里决定，返回 True。
# 参数：
#   mode：Agent 执行模式或检索模式，类型 `AgentMode | None`。
#   tool_name：工具名称，类型 `str`。
# 返回：类型 `bool`；返回 `allowed is None or tool_name in allowed`。
def role_allows(mode: AgentMode | None, tool_name: str) -> bool:
    """非角色模式（NORMAL、PLAN、未知）不由这里决定，返回 True。"""

    allowed = ROLE_ALLOWED_TOOLS.get(mode) if mode is not None else None
    return allowed is None or tool_name in allowed


# 函数说明：role_rejection_message
# 用途：返回 `f"{ROLE_REJECTION_PREFIX} tool '{tool_name}' is not allowed in {mode.value
# } mode (role…`，提供 工具注册、执行与权限钩子 的派生值。
# 参数：
#   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
#   tool_name：工具名称，类型 `str`。
# 返回：类型 `str`；返回 `f"{ROLE_REJECTION_PREFIX} tool '{tool_name}' is not allowed in
#  {mode.value} mode (role…`。
def role_rejection_message(mode: AgentMode, tool_name: str) -> str:
    return (
        f"{ROLE_REJECTION_PREFIX} tool '{tool_name}' is not allowed in "
        f"{mode.value} mode (role whitelist only)."
    )


# 函数说明：is_role_rejection
# 用途：判断角色是否满足当前实现的条件。
# 参数：
#   result：上一步计算或执行得到的结果，类型 `ToolResult`。
# 返回：类型 `bool`；返回
# `not result.success and (result.error or '').startswith(ROLE_REJECTION_PREFIX)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(result.error or '').startswith`。
def is_role_rejection(result: ToolResult) -> bool:
    return not result.success and (result.error or "").startswith(ROLE_REJECTION_PREFIX)


# 函数说明：count_role_rejections
# 用途：按工具名统计被角色边界拒绝的调用次数。
# 参数：
#   results：结果集合输入或配置值，类型 `Iterable[ToolResult]`。
# 返回：类型 `Counter[str]`；返回
# `Counter((result.tool_name for result in results if is_role_rejection(result)))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Counter` → `is_role_rejection`。
def count_role_rejections(results: Iterable[ToolResult]) -> Counter[str]:
    """按工具名统计被角色边界拒绝的调用次数。"""

    return Counter(result.tool_name for result in results if is_role_rejection(result))


class RoleBoundaryHook(ToolHook):
    """按 ``context.mode`` 再检查一次白名单；拒绝发生在权限审批和实际执行之前。

    ``allows`` 默认用固定白名单；``ToolExecutor`` 传入自己 registry 的判断，
    这样长任务单独授权给 Executor 的额外工具在两处检查里保持一致。
    """

    critical = True

    # 函数说明：RoleBoundaryHook.__init__
    # 用途：初始化 RoleBoundaryHook；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   allows：`allows`输入或配置值，类型 `Callable[[AgentMode, str], bool]`；默认
    # `role_allows`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._allows`。
    def __init__(
        self,
        allows: Callable[[AgentMode, str], bool] = role_allows,
    ) -> None:
        self._allows = allows

    # 函数说明：RoleBoundaryHook.before_execute
    # 用途：在 `execute` 前后执行 RoleBoundaryHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `ToolHookDecision | None`；按分支返回 `None`；
    # `ToolHookDecision(denied_reason=role_rejection_message(mode, name))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._allows` →
    # `ToolHookDecision` → `role_rejection_message`。
    # 分支与异常：
    #   当 `mode is None or self._allows(mode, name)` 时，返回 `None`。
    async def before_execute(
        self,
        context: ToolExecutionContext,
    ) -> ToolHookDecision | None:
        mode = context.mode
        name = context.tool_call.name
        if mode is None or self._allows(mode, name):
            return None
        return ToolHookDecision(denied_reason=role_rejection_message(mode, name))


__all__ = [
    "AUDIT_ALLOWED_TOOLS",
    "EXECUTE_ALLOWED_TOOLS",
    "MANAGE_ALLOWED_TOOLS",
    "READ_ONLY_MODES",
    "ROLE_ALLOWED_TOOLS",
    "ROLE_MODES",
    "ROLE_REJECTION_PREFIX",
    "RoleBoundaryHook",
    "count_role_rejections",
    "is_role_mode",
    "is_role_rejection",
    "role_allows",
    "role_rejection_message",
]
