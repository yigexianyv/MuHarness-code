
from __future__ import annotations

from collections.abc import Collection, Mapping

from app.models.types import AgentMode

from .role_boundary import ROLE_ALLOWED_TOOLS

PLAN_MODE_ALLOWED_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "web_search",
        "get_current_time",
        "current_time",
        "memory_read",
        "memory_search",
        "history_search",
        "history_read",
        "evidence_search",
        "evidence_read",
        "task_create",
        "task_update",
        "task_get",
        "task_list",
    }
)


class ToolAvailabilityPolicy:

    # 函数说明：ToolAvailabilityPolicy.__init__
    # 用途：初始化 ToolAvailabilityPolicy；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   plan_allowed_tools：传给 `frozenset` 的输入，类型 `Collection[str]`；默认
    # `PLAN_MODE_ALLOWED_TOOLS`。
    #   role_allowed_tools：角色工具集合输入或配置值，类型
    # `Mapping[AgentMode, Collection[str]]`；默认 `ROLE_ALLOWED_TOOLS`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`frozenset`。
    # 副作用与资源：
    #   更新对象字段：`self._plan_allowed_tools`、`self._role_allowed_tools`。
    def __init__(
        self,
        *,
        plan_allowed_tools: Collection[str] = PLAN_MODE_ALLOWED_TOOLS,
        role_allowed_tools: Mapping[AgentMode, Collection[str]] = ROLE_ALLOWED_TOOLS,
    ) -> None:
        self._plan_allowed_tools = frozenset(plan_allowed_tools)
        self._role_allowed_tools = {
            mode: frozenset(names) for mode, names in role_allowed_tools.items()
        }

    # 函数说明：ToolAvailabilityPolicy.with_role_extras
    # 用途：返回一个新策略：指定角色模式的白名单追加 ``extras``，其余不变。
    # 参数：
    #   extras：`extras`输入或配置值，类型 `Mapping[AgentMode, Collection[str]]`。
    # 返回：类型 `ToolAvailabilityPolicy`；返回 `ToolAvailabilityPolicy(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`frozenset` →
    # `ToolAvailabilityPolicy`。
    # 分支与异常：
    #   当 `mode not in role_allowed` 时，抛出 `ValueError(f'not a role mode: {mode}')`
    # 。
    def with_role_extras(
        self,
        extras: Mapping[AgentMode, Collection[str]],
    ) -> ToolAvailabilityPolicy:
        """返回一个新策略：指定角色模式的白名单追加 ``extras``，其余不变。"""

        role_allowed = dict(self._role_allowed_tools)
        for mode, names in extras.items():
            if mode not in role_allowed:
                raise ValueError(f"not a role mode: {mode}")
            role_allowed[mode] = role_allowed[mode] | frozenset(names)
        return ToolAvailabilityPolicy(
            plan_allowed_tools=self._plan_allowed_tools,
            role_allowed_tools=role_allowed,
        )

    # 函数说明：ToolAvailabilityPolicy.is_role_mode
    # 用途：判断角色模式是否满足当前实现的条件。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode | None`。
    # 返回：类型 `bool`；返回 `mode in self._role_allowed_tools`。
    def is_role_mode(self, mode: AgentMode | None) -> bool:
        return mode in self._role_allowed_tools

    # 函数说明：ToolAvailabilityPolicy.allowed_names
    # 用途：在工具注册、执行与权限钩子中处理 `allowed_names`，通过
    # `self._role_allowed_tools.get` 完成首个内部处理步骤。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   registered_names：传给 `frozenset` 的输入，类型 `Collection[str]`。
    # 返回：类型 `frozenset[str]`；按分支返回 `self._plan_allowed_tools`；`role_allowed`
    # ；`frozenset(registered_names)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`frozenset`。
    # 分支与异常：
    #   当 `mode is AgentMode.PLAN` 时，返回 `self._plan_allowed_tools`。
    #   当 `role_allowed is not None` 时，返回 `role_allowed`。
    def allowed_names(
        self,
        mode: AgentMode,
        *,
        registered_names: Collection[str],
    ) -> frozenset[str]:
        if mode is AgentMode.PLAN:
            return self._plan_allowed_tools
        role_allowed = self._role_allowed_tools.get(mode)
        if role_allowed is not None:
            return role_allowed
        return frozenset(registered_names)

    # 函数说明：ToolAvailabilityPolicy.is_available
    # 用途：判断`available`是否满足当前实现的条件。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   deferred_names：`deferred_names`输入或配置值，类型 `Collection[str]`。
    #   activated_names：`activated_names`输入或配置值，类型 `Collection[str]`。
    # 返回：类型 `bool`；按分支返回 `name in role_allowed`；`True`；
    # `mode is AgentMode.PLAN and name in self._plan_allowed_tools`。
    # 分支与异常：
    #   当 `role_allowed is not None` 时，返回 `name in role_allowed`。
    #   当 `name not in deferred_names or name in activated_names` 时，返回 `True`。
    def is_available(
        self,
        name: str,
        mode: AgentMode,
        *,
        deferred_names: Collection[str],
        activated_names: Collection[str],
    ) -> bool:

        role_allowed = self._role_allowed_tools.get(mode)
        if role_allowed is not None:
            # 角色模式只看白名单：白名单内的延迟工具无需激活，白名单外的即使激活也不可用
            return name in role_allowed
        if name not in deferred_names or name in activated_names:
            return True
        return mode is AgentMode.PLAN and name in self._plan_allowed_tools


__all__ = ["PLAN_MODE_ALLOWED_TOOLS", "ToolAvailabilityPolicy"]
