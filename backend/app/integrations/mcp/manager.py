
from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from app.safety.sandbox import SandboxSupervisor
from app.tools.registry import ToolRegistry

from .client import MCPClientProtocol, StdioMCPClient
from .errors import MCPConfigurationError
from .models import (
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
)
from .tool import MCPToolAdapter

MCPClientFactory = Callable[[MCPServerConfig], MCPClientProtocol]
_INVALID_TOOL_NAME = re.compile(r"[^a-zA-Z0-9_]+")


class MCPClientManager:

    # 函数说明：MCPClientManager.__init__
    # 用途：初始化 MCPClientManager；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   configs：传给 `tuple` 的输入，类型 `Sequence[MCPServerConfig]`。
    #   client_factory：客户端构造工厂，类型 `MCPClientFactory | None`；默认 `None`。
    #   sandbox_supervisor：`sandbox_supervisor`输入或配置值，类型
    # `SandboxSupervisor | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerStatus`。
    # 分支与异常：
    #   当 `len(names) != len(set(names))` 时，抛出
    # `MCPConfigurationError('MCP Server 名称不能重复')`。
    # 副作用与资源：
    #   更新对象字段：`self._configs`、`self._client_factory`、`self._clients`、
    # `self._registered_names`、`self._states`。
    def __init__(
        self,
        configs: Sequence[MCPServerConfig],
        *,
        client_factory: MCPClientFactory | None = None,
        sandbox_supervisor: SandboxSupervisor | None = None,
    ) -> None:
        names = [config.name for config in configs]
        if len(names) != len(set(names)):
            raise MCPConfigurationError("MCP Server 名称不能重复")
        self._configs = tuple(configs)
        self._client_factory = client_factory or (
            lambda config: StdioMCPClient(
                config,
                sandbox_supervisor=sandbox_supervisor,
            )
        )
        self._clients: dict[str, MCPClientProtocol] = {}
        self._registered_names: dict[str, tuple[str, ...]] = {}
        self._states = {
            config.name: MCPServerStatus(
                name=config.name,
                state=MCPServerState.STOPPED,
            )
            for config in self._configs
        }

    # 函数说明：MCPClientManager.start
    # 用途：启动已配置的 MCP 服务，将远端工具注册到本地工具表。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
    # 返回：类型 `tuple[MCPServerStatus, ...]`；返回 `self.statuses()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._start_server` →
    # `self.statuses`。
    # 分支与异常：
    #   当 `not config.enabled` 时，跳过当前循环项。
    async def start(self, registry: ToolRegistry) -> tuple[MCPServerStatus, ...]:

        """启动已配置的 MCP 服务，将远端工具注册到本地工具表。"""
        for config in self._configs:
            if not config.enabled:
                continue
            await self._start_server(config, registry)
        return self.statuses()

    # 函数说明：MCPClientManager._start_server
    # 用途：启动服务，供MCP 连接与外部工具适配使用。
    # 参数：
    #   config：运行配置，类型 `MCPServerConfig`。
    #   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._set_status` →
    # `self._client_factory` → `client.start` → `client.list_tools` → `_build_adapters`
    # → `registry.register`；另有 2 个调用点。
    # 分支与异常：
    #   当 `config.name in self._clients` 时，返回 `None`。
    #   捕获 `Exception` 后，返回 `None`。
    #   捕获 `Exception` 后，忽略该异常并继续当前流程。
    async def _start_server(
        self,
        config: MCPServerConfig,
        registry: ToolRegistry,
    ) -> None:
        if config.name in self._clients:
            return
        self._set_status(config.name, MCPServerState.STARTING)
        client = self._client_factory(config)
        registered: list[str] = []
        try:
            await client.start()
            remote_tools = await client.list_tools()
            adapters = _build_adapters(config, client, remote_tools)
            for adapter in adapters:
                registry.register(adapter, deferred=True)
                registered.append(adapter.definition.name)
        except Exception as exc:
            for name in reversed(registered):
                registry.unregister(name)
            try:
                await client.close()
            except Exception:
                pass
            self._set_status(
                config.name,
                MCPServerState.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        self._clients[config.name] = client
        self._registered_names[config.name] = tuple(registered)
        self._set_status(
            config.name,
            MCPServerState.RUNNING,
            tool_names=tuple(registered),
            sandboxed=getattr(getattr(client, "launch_spec", None), "sandboxed", None),
            sandbox_backend=getattr(
                getattr(client, "launch_spec", None),
                "backend",
                None,
            ),
        )

    # 函数说明：MCPClientManager.close
    # 用途：注销远端工具并关闭所有已启动的 MCP 客户端。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ToolRegistry | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.unregister` →
    # `client.close` → `self._set_status` → `self._clients.clear` →
    # `self._registered_names.clear`。
    # 分支与异常：
    #   捕获 `KeyError` 后，忽略该异常并继续当前流程。
    #   捕获 `Exception` 后，跳过当前循环项，继续处理后续项。
    async def close(self, registry: ToolRegistry | None = None) -> None:

        """注销远端工具并关闭所有已启动的 MCP 客户端。"""
        for server_name, client in reversed(tuple(self._clients.items())):
            if registry is not None:
                for name in self._registered_names.get(server_name, ()):
                    try:
                        registry.unregister(name)
                    except KeyError:
                        pass
            try:
                await client.close()
            except Exception as exc:
                self._set_status(
                    server_name,
                    MCPServerState.FAILED,
                    error=f"关闭失败: {type(exc).__name__}: {exc}",
                )
                continue
            self._set_status(server_name, MCPServerState.STOPPED)
        self._clients.clear()
        self._registered_names.clear()

    # 函数说明：MCPClientManager.statuses
    # 用途：返回 `tuple((self._states[config.name] for config in self._configs))`，提供
    # MCPClientManager 的派生值。
    # 返回：类型 `tuple[MCPServerStatus, ...]`；返回
    # `tuple((self._states[config.name] for config in self._configs))`。
    def statuses(self) -> tuple[MCPServerStatus, ...]:

        return tuple(self._states[config.name] for config in self._configs)

    # 函数说明：MCPClientManager._set_status
    # 用途：设置状态，供MCP 连接与外部工具适配使用。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   state：当前状态快照，类型 `MCPServerState`。
    #   tool_names：工具输入或配置值，类型 `tuple[str, ...]`；默认 `()`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    #   sandboxed：`sandboxed`输入或配置值，类型 `bool | None`；默认 `None`。
    #   sandbox_backend：`sandbox_backend`输入或配置值，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerStatus`。
    def _set_status(
        self,
        name: str,
        state: MCPServerState,
        *,
        tool_names: tuple[str, ...] = (),
        error: str | None = None,
        sandboxed: bool | None = None,
        sandbox_backend: str | None = None,
    ) -> None:
        self._states[name] = MCPServerStatus(
            name=name,
            state=state,
            tool_names=tool_names,
            error=error,
            sandboxed=sandboxed,
            sandbox_backend=sandbox_backend,
        )


# 函数说明：mcp_tool_name
# 用途：在MCP 连接与外部工具适配中处理 `mcp_tool_name`，通过
# `_INVALID_TOOL_NAME.sub('_', remote_name).strip` 完成首个内部处理步骤。
# 参数：
#   server_name：服务名称输入或配置值，类型 `str`。
#   remote_name：传给 `_INVALID_TOOL_NAME.sub` 的输入，类型 `str`。
# 返回：类型 `str`；返回 `f'mcp__{server_name}__{normalized}'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_INVALID_TOOL_NAME.sub`。
# 分支与异常：
#   当 `not normalized` 时，抛出 `MCPConfigurationError(…)`。
def mcp_tool_name(server_name: str, remote_name: str) -> str:

    normalized = _INVALID_TOOL_NAME.sub("_", remote_name).strip("_")
    if not normalized:
        raise MCPConfigurationError(
            f"MCP Server '{server_name}' 返回了无法注册的工具名 {remote_name!r}"
        )
    return f"mcp__{server_name}__{normalized}"


# 函数说明：_build_adapters
# 用途：构建`adapters`，供MCP 连接与外部工具适配使用。
# 参数：
#   config：运行配置，类型 `MCPServerConfig`。
#   client：模型、HTTP 或 MCP 客户端，类型 `MCPClientProtocol`。
#   remote_tools：工具集合输入或配置值，类型 `Sequence`。
# 返回：类型 `tuple[MCPToolAdapter, ...]`；返回 `tuple(adapters)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`mcp_tool_name` → `seen.add` →
# `MCPToolAdapter`。
# 分支与异常：
#   当 `registered_name in seen` 时，抛出 `MCPConfigurationError(…)`。
def _build_adapters(
    config: MCPServerConfig,
    client: MCPClientProtocol,
    remote_tools: Sequence,
) -> tuple[MCPToolAdapter, ...]:
    adapters: list[MCPToolAdapter] = []
    seen: set[str] = set()
    for remote_tool in remote_tools:
        registered_name = mcp_tool_name(config.name, remote_tool.name)
        if registered_name in seen:
            raise MCPConfigurationError(
                f"MCP Server '{config.name}' 的工具名规范化后发生冲突: "
                f"{registered_name}"
            )
        seen.add(registered_name)
        adapters.append(
            MCPToolAdapter(
                server_name=config.name,
                registered_name=registered_name,
                remote_tool=remote_tool,
                client=client,
                permission=config.permission,
            )
        )
    return tuple(adapters)


__all__ = ["MCPClientFactory", "MCPClientManager", "mcp_tool_name"]
