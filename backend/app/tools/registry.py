
from __future__ import annotations

import re
from collections.abc import Collection, Mapping

from app.models.types import AgentMode, ToolDefinition

from .availability import ToolAvailabilityPolicy
from .base import BaseTool

_VALID_NAME = re.compile(r"^[a-zA-Z0-9_]+$")

class ToolRegistry:
    # 函数说明：ToolRegistry.__init__
    # 用途：初始化 ToolRegistry；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   availability_policy：`availability_policy`输入或配置值，类型
    # `ToolAvailabilityPolicy | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolAvailabilityPolicy`。
    # 副作用与资源：
    #   更新对象字段：`self._tools`、`self._deferred_names`、`self._availability_policy`
    # 。
    def __init__(
        self,
        *,
        availability_policy: ToolAvailabilityPolicy | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._deferred_names: set[str] = set()
        self._availability_policy = (
            availability_policy or ToolAvailabilityPolicy()
        )

    # 函数说明：ToolRegistry.with_role_extras
    # 用途：返回一个共享同一批工具的视图，只是角色白名单多了 ``extras``。
    # 参数：
    #   extras：传给 `self._availability_policy.with_role_extras` 的输入，类型
    # `Mapping[AgentMode, Collection[str]]`。
    # 返回：类型 `ToolRegistry`；返回 `view`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` →
    # `self._availability_policy.with_role_extras`。
    # 副作用与资源：
    #   更新对象字段：`view._tools`、`view._deferred_names`。
    def with_role_extras(
        self,
        extras: Mapping[AgentMode, Collection[str]],
    ) -> ToolRegistry:
        """返回一个共享同一批工具的视图，只是角色白名单多了 ``extras``。

        工具字典和延迟集合是同一个对象，之后注册的工具（例如 MCP）在视图里同样可见；
        视图本身只用来构造长任务 Executor 的 Runtime。
        """

        view = ToolRegistry(
            availability_policy=self._availability_policy.with_role_extras(extras)
        )
        view._tools = self._tools
        view._deferred_names = self._deferred_names
        return view

    # 函数说明：ToolRegistry.is_role_mode
    # 用途：判断角色模式是否满足当前实现的条件。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode | None`。
    # 返回：类型 `bool`；返回 `self._availability_policy.is_role_mode(mode)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._availability_policy.is_role_mode`。
    def is_role_mode(self, mode: AgentMode | None) -> bool:
        return self._availability_policy.is_role_mode(mode)

    # 函数说明：ToolRegistry.register
    # 用途：注册工具定义，并维护名称、可用模式与延迟暴露信息。
    # 参数：
    #   tool：目标工具实例，类型 `BaseTool`。
    #   deferred：`deferred`输入或配置值，类型 `bool`；默认 `False`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_VALID_NAME.fullmatch` →
    # `self._deferred_names.add`。
    # 分支与异常：
    #   当 `not name` 时，抛出 `ValueError('Tool name cannot be empty.')`。
    #   当 `not _VALID_NAME.fullmatch(name)` 时，抛出 `ValueError(…)`。
    #   当 `name in self._tools` 时，抛出
    # `ValueError(f"Tool '{name}' is already registered.")`。
    def register(self, tool: BaseTool, *, deferred: bool = False) -> None:

        """注册工具定义，并维护名称、可用模式与延迟暴露信息。"""
        name = tool.definition.name
        if not name:
            raise ValueError("Tool name cannot be empty.")
        if not _VALID_NAME.fullmatch(name):
            raise ValueError(
                "Tool name must use dot-separated letters, digits, or underscores: "
                f"{name!r}"
            )
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = tool
        if deferred:
            self._deferred_names.add(name)

    # 函数说明：ToolRegistry.unregister
    # 用途：注销ToolRegistry，供工具注册、执行与权限钩子使用。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `BaseTool`；返回 `tool`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._tools.pop` →
    # `self._deferred_names.discard`。
    # 分支与异常：
    #   捕获 `KeyError` 后，转换或抛出 `KeyError(f"Tool '{name}' is not registered.")`。
    def unregister(self, name: str) -> BaseTool:
        try:
            tool = self._tools.pop(name)
        except KeyError:
            raise KeyError(f"Tool '{name}' is not registered.") from None
        self._deferred_names.discard(name)
        return tool

    # 函数说明：ToolRegistry.get
    # 用途：获取ToolRegistry，供工具注册、执行与权限钩子使用。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `BaseTool`；返回 `self._tools[name]`。
    # 分支与异常：
    #   捕获 `KeyError` 后，转换或抛出 `KeyError(f"Tool '{name}' is not registered.")`。
    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Tool '{name}' is not registered.") from None

    # 函数说明：ToolRegistry.names
    # 用途：返回 `tuple(self._tools)`，提供 ToolRegistry 的派生值。
    # 返回：类型 `tuple[str, ...]`；返回 `tuple(self._tools)`。
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    # 函数说明：ToolRegistry.deferred_names
    # 用途：返回 `tuple(sorted(self._deferred_names))`，提供 ToolRegistry 的派生值。
    # 返回：类型 `tuple[str, ...]`；返回 `tuple(sorted(self._deferred_names))`。
    def deferred_names(self) -> tuple[str, ...]:

        return tuple(sorted(self._deferred_names))

    # 函数说明：ToolRegistry.is_deferred
    # 用途：判断`deferred`是否满足当前实现的条件。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `bool`；返回 `name in self._deferred_names`。
    def is_deferred(self, name: str) -> bool:
        return name in self._deferred_names

    # 函数说明：ToolRegistry.is_available_for_mode
    # 用途：判断模式是否满足当前实现的条件。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   activated_names：`activated_names`输入或配置值，类型 `Collection[str]`；默认
    # `()`。
    # 返回：类型 `bool`；返回 `self._availability_policy.is_available(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._availability_policy.is_available`。
    def is_available_for_mode(
        self,
        name: str,
        mode: AgentMode,
        *,
        activated_names: Collection[str] = (),
    ) -> bool:

        return self._availability_policy.is_available(
            name,
            mode,
            deferred_names=self._deferred_names,
            activated_names=activated_names,
        )

    # 函数说明：ToolRegistry.model_definitions
    # 用途：在工具注册、执行与权限钩子中处理 `model_definitions`，通过
    # `self._tools[name].definition.permission.model_visible` 完成首个内部处理步骤。
    # 参数：
    #   activated_names：传给 `set` 的输入，类型 `Collection[str]`；默认 `()`。
    # 返回：类型 `tuple[ToolDefinition, ...]`；返回 `tuple(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._tools[name].definition.permission.model_visible`。
    def model_definitions(
        self,
        *,
        activated_names: Collection[str] = (),
    ) -> tuple[ToolDefinition, ...]:

        activated = set(activated_names)
        return tuple(
            self._tools[name].definition
            for name in sorted(self._tools)
            if self._tools[name].definition.permission.model_visible()
            and (name not in self._deferred_names or name in activated)
        )

    # 函数说明：ToolRegistry.allowed_names_for_mode
    # 用途：返回
    # `self._availability_policy.allowed_names(mode, registered_names=self._tools)`，提
    # 供 ToolRegistry 的派生值。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    # 返回：类型 `frozenset[str]`；返回
    # `self._availability_policy.allowed_names(mode, registered_names=self._tools)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._availability_policy.allowed_names`。
    def allowed_names_for_mode(self, mode: AgentMode) -> frozenset[str]:

        return self._availability_policy.allowed_names(
            mode,
            registered_names=self._tools,
        )

    # 函数说明：ToolRegistry.is_allowed_for_mode
    # 用途：判断模式是否满足当前实现的条件。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    # 返回：类型 `bool`；返回 `name in self.allowed_names_for_mode(mode)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.allowed_names_for_mode`。
    def is_allowed_for_mode(self, name: str, mode: AgentMode) -> bool:

        return name in self.allowed_names_for_mode(mode)

    # 函数说明：ToolRegistry.is_allowed_during_closing
    # 用途：判断`allowed_during_closing`是否满足当前实现的条件。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    # 返回：类型 `bool`；按分支返回 `False`；
    # `tool is not None and tool.definition.closing_allowed`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.is_allowed_for_mode`。
    # 分支与异常：
    #   当 `not self.is_allowed_for_mode(name, mode)` 时，返回 `False`。
    def is_allowed_during_closing(self, name: str, mode: AgentMode) -> bool:

        if not self.is_allowed_for_mode(name, mode):
            return False
        tool = self._tools.get(name)
        return tool is not None and tool.definition.closing_allowed

    # 函数说明：ToolRegistry.model_definitions_for_mode
    # 用途：筛选当前模式下可发送给模型的工具定义。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   activated_names：传给 `set` 的输入，类型 `Collection[str]`；默认 `()`。
    # 返回：类型 `tuple[ToolDefinition, ...]`；返回 `tuple(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.allowed_names_for_mode` →
    # `self._tools[name].definition.permission.model_visible` →
    # `self.is_available_for_mode`。
    def model_definitions_for_mode(
        self,
        mode: AgentMode,
        *,
        activated_names: Collection[str] = (),
    ) -> tuple[ToolDefinition, ...]:

        """筛选当前模式下可发送给模型的工具定义。"""
        allowed = self.allowed_names_for_mode(mode)
        activated = set(activated_names)
        return tuple(
            self._tools[name].definition
            for name in sorted(self._tools)
            if name in allowed
            and self._tools[name].definition.permission.model_visible()
            and self.is_available_for_mode(
                name,
                mode,
                activated_names=activated,
            )
        )

    # 函数说明：ToolRegistry.closing_definitions_for_mode
    # 用途：筛选收尾阶段仍允许模型调用的工具定义。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   activated_names：`activated_names`输入或配置值，类型 `Collection[str]`；默认
    # `()`。
    # 返回：类型 `tuple[ToolDefinition, ...]`；返回 `tuple(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.model_definitions_for_mode`
    # 。
    def closing_definitions_for_mode(
        self,
        mode: AgentMode,
        *,
        activated_names: Collection[str] = (),
    ) -> tuple[ToolDefinition, ...]:

        """筛选收尾阶段仍允许模型调用的工具定义。"""
        return tuple(
            definition
            for definition in self.model_definitions_for_mode(
                mode,
                activated_names=activated_names,
            )
            if definition.closing_allowed
        )

    # 函数说明：ToolRegistry.definitions
    # 用途：在工具注册、执行与权限钩子中处理 `definitions`，通过 `self._tools.values` 完
    # 成首个内部处理步骤。
    # 参数：
    #   for_model：模型输入或配置值，类型 `bool`；默认 `True`。
    # 返回：类型 `tuple[ToolDefinition, ...]`；按分支返回
    # `tuple((tool.definition for tool in self._tools.values()))`；`tuple(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `tool.definition.permission.model_visible`。
    # 分支与异常：
    #   当 `not for_model` 时，返回
    # `tuple((tool.definition for tool in self._tools.values()))`。
    def definitions(
        self,
        *,
        for_model: bool = True,
    ) -> tuple[ToolDefinition, ...]:
        if not for_model:
            return tuple(tool.definition for tool in self._tools.values())
        return tuple(
            tool.definition
            for tool in self._tools.values()
            if tool.definition.permission.model_visible()
        )
