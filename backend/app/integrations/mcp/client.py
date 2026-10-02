
from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ListToolsResult

from app.safety.sandbox import SandboxLaunchSpec, SandboxSupervisor

from .errors import MCPConnectionError, MCPToolCallError, MCPToolDiscoveryError
from .models import MCPRemoteTool, MCPServerConfig

_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_SAFE_INHERITED_ENVIRONMENT = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
)


class MCPClientProtocol(Protocol):

    # 函数说明：MCPClientProtocol.start
    # 用途：启动MCPClientProtocol，供MCP 连接与外部工具适配使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def start(self) -> None: ...

    # 函数说明：MCPClientProtocol.list_tools
    # 用途：列出工具集合，供MCP 连接与外部工具适配使用。
    # 返回：类型 `tuple[MCPRemoteTool, ...]`；不返回结果值（隐式 None）。
    async def list_tools(self) -> tuple[MCPRemoteTool, ...]: ...

    # 函数说明：MCPClientProtocol.call_tool
    # 用途：处理MCP 连接与外部工具适配中的 `call_tool` 数据；结果及边界条件见下方说明。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `str`；不返回结果值（隐式 None）。
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...

    # 函数说明：MCPClientProtocol.close
    # 用途：关闭MCPClientProtocol，供MCP 连接与外部工具适配使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None: ...


class StdioMCPClient:

    # 函数说明：StdioMCPClient.__init__
    # 用途：初始化 StdioMCPClient；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `MCPServerConfig`。
    #   sandbox_supervisor：`sandbox_supervisor`输入或配置值，类型
    # `SandboxSupervisor | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.config`、`self.sandbox_supervisor`、`self._stack`、
    # `self._session`、`self.launch_spec`。
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        sandbox_supervisor: SandboxSupervisor | None = None,
    ) -> None:
        self.config = config
        self.sandbox_supervisor = sandbox_supervisor
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self.launch_spec: SandboxLaunchSpec | None = None

    # 函数说明：StdioMCPClient.start
    # 用途：在沙箱约束下启动 stdio MCP 进程并建立会话。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_StdioSessionLifecycle.open`。
    # 分支与异常：
    #   当 `self._session is not None` 时，返回 `None`。
    # 副作用与资源：
    #   更新对象字段：`self._stack`、`self._session`、`self.launch_spec`。
    #   文件或资源访问：`_StdioSessionLifecycle.open`。
    async def start(self) -> None:

        """在沙箱约束下启动 stdio MCP 进程并建立会话。"""
        if self._session is not None:
            return
        stack, session, launch = await _StdioSessionLifecycle.open(
            lambda: self.config, self._prepare_launch,
        )
        self._stack = stack
        self._session = session
        self.launch_spec = launch

    # 函数说明：StdioMCPClient.list_tools
    # 用途：读取远端工具定义并转换为本地描述。
    # 返回：类型 `tuple[MCPRemoteTool, ...]`；返回 `tuple(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._require_session` →
    # `_StdioSessionLifecycle.discover` → `MCPRemoteTool`。
    async def list_tools(self) -> tuple[MCPRemoteTool, ...]:

        """读取远端工具定义并转换为本地描述。"""
        session = self._require_session()
        result = await _StdioSessionLifecycle.discover(lambda: self.config, session)
        return tuple(
            MCPRemoteTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in result.tools
        )

    # 函数说明：StdioMCPClient.call_tool
    # 用途：调用远端工具并序列化 MCP 返回内容。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `str`；返回 `output`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._require_session` →
    # `_StdioSessionLifecycle.invoke` → `serialize_mcp_result`。
    # 分支与异常：
    #   当 `result.isError` 时，抛出 `MCPToolCallError(…)`。
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:

        """调用远端工具并序列化 MCP 返回内容。"""
        session = self._require_session()
        result = await _StdioSessionLifecycle.invoke(
            lambda: self.config, session, name, arguments,
        )
        output = serialize_mcp_result(result)
        if result.isError:
            raise MCPToolCallError(
                f"MCP 工具 '{self.config.name}/{name}' 返回错误: {output}"
            )
        return output

    # 函数说明：StdioMCPClient.close
    # 用途：关闭StdioMCPClient，供MCP 连接与外部工具适配使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_StdioSessionLifecycle.close`。
    # 副作用与资源：
    #   更新对象字段：`self._stack`、`self._session`。
    async def close(self) -> None:

        stack = self._stack
        self._stack = None
        self._session = None
        await _StdioSessionLifecycle.close(stack)

    # 函数说明：StdioMCPClient._prepare_launch
    # 用途：准备`launch`，供MCP 连接与外部工具适配使用。
    # 参数：
    #   environment：传给 `_LaunchPolicy.prepare` 的输入，类型 `dict[str, str]`。
    # 返回：类型 `SandboxLaunchSpec`；返回
    # `_LaunchPolicy.prepare(self.config, self.sandbox_supervisor, environment)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_LaunchPolicy.prepare`。
    def _prepare_launch(self, environment: dict[str, str]) -> SandboxLaunchSpec:
        return _LaunchPolicy.prepare(
            self.config, self.sandbox_supervisor, environment,
        )

    # 函数说明：StdioMCPClient._require_session
    # 用途：获取并校验必需的`session`，供MCP 连接与外部工具适配使用。
    # 返回：类型 `ClientSession`；返回 `self._session`。
    # 分支与异常：
    #   当 `self._session is None` 时，抛出
    # `MCPConnectionError(f"MCP Server '{self.config.name}' 尚未连接")`。
    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPConnectionError(
                f"MCP Server '{self.config.name}' 尚未连接"
            )
        return self._session


# 函数说明：serialize_mcp_result
# 用途：序列化结果，供MCP 连接与外部工具适配使用。
# 参数：
#   result：上一步计算或执行得到的结果，类型 `CallToolResult`。
# 返回：类型 `str`；返回 `_ResultSerializer.serialize(result)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_ResultSerializer.serialize`。
def serialize_mcp_result(result: CallToolResult) -> str:
    return _ResultSerializer.serialize(result)


class _ResultSerializer:
    """Translate remote content without coupling encoding to session errors."""

    # 函数说明：_ResultSerializer.serialize
    # 用途：序列化_ResultSerializer，供MCP 连接与外部工具适配使用。
    # 参数：
    #   result：上一步计算或执行得到的结果，类型 `CallToolResult`。
    # 返回：类型 `str`；按分支返回 `text`；
    # `json.dumps(payload, ensure_ascii=False, separators=(',', ':'))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
    # 分支与异常：
    #   当 `result.structuredContent is None` 时，返回 `text`。
    @staticmethod
    def serialize(result: CallToolResult) -> str:
        if len(result.content) == 1 and result.content[0].type == "text":
            text = result.content[0].text
            if result.structuredContent is None:
                return text
        payload: dict[str, Any] = {
            "content": [
                item.model_dump(mode="json", by_alias=True, exclude_none=True)
                for item in result.content
            ]
        }
        if result.structuredContent is not None:
            payload["structured_content"] = result.structuredContent
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class _StdioSessionLifecycle:
    """Own acquisition, timed protocol I/O and disposal for one stdio session.

    A session is handed to the facade only after initialization succeeds. The
    exit stack stays in the caller's async task throughout acquisition/disposal.
    """

    # 函数说明：_StdioSessionLifecycle.open
    # 用途：打开_StdioSessionLifecycle，供MCP 连接与外部工具适配使用。
    # 参数：
    #   config：运行配置，类型 `Callable[[], MCPServerConfig]`。
    #   prepare_launch：`prepare_launch`输入或配置值，类型
    # `Callable[[dict[str, str]], SandboxLaunchSpec]`。
    # 返回：类型 `tuple[AsyncExitStack, ClientSession, SandboxLaunchSpec]`；返回
    # `(stack, session, launch)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AsyncExitStack` →
    # `_resolve_environment` → `config` → `prepare_launch` → `asyncio.timeout` →
    # `stack.enter_async_context`；另有 5 个调用点。
    # 资源/并发边界：`asyncio.timeout(config().startup_timeout_seconds)`，上下文退出时执
    # 行相应清理。
    # 分支与异常：
    #   捕获 `Exception` 后，转换或抛出 `MCPConnectionError(f"MCP Server '{config().name
    # }' 启动失败: {type(exc).__name__}: {exc}")`。
    @staticmethod
    async def open(
        config: Callable[[], MCPServerConfig],
        prepare_launch: Callable[[dict[str, str]], SandboxLaunchSpec],
    ) -> tuple[AsyncExitStack, ClientSession, SandboxLaunchSpec]:
        stack = AsyncExitStack()
        try:
            environment = _resolve_environment(config())
            launch = prepare_launch(environment)
            async with asyncio.timeout(config().startup_timeout_seconds):
                streams = await stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=launch.command,
                            args=list(launch.args),
                            env=launch.env,
                            cwd=launch.cwd,
                        )
                    )
                )
                session = await stack.enter_async_context(ClientSession(*streams))
                await session.initialize()
        except Exception as exc:
            await _close_quietly(stack)
            raise MCPConnectionError(
                f"MCP Server '{config().name}' 启动失败: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return stack, session, launch

    # 函数说明：_StdioSessionLifecycle.discover
    # 用途：发现_StdioSessionLifecycle，供MCP 连接与外部工具适配使用。
    # 参数：
    #   config：运行配置，类型 `Callable[[], MCPServerConfig]`。
    #   session：`session`输入或配置值，类型 `ClientSession`。
    # 返回：类型 `ListToolsResult`；返回 `await session.list_tools()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.timeout` → `config` →
    # `session.list_tools`。
    # 资源/并发边界：`asyncio.timeout(config().call_timeout_seconds)`，上下文退出时执行
    # 相应清理。
    # 分支与异常：
    #   捕获 `Exception` 后，转换或抛出 `MCPToolDiscoveryError(f"MCP Server '{config().
    # name}' 工具发现失败: {type(exc).__name__}: {exc}")`。
    @staticmethod
    async def discover(
        config: Callable[[], MCPServerConfig], session: ClientSession,
    ) -> ListToolsResult:
        try:
            async with asyncio.timeout(config().call_timeout_seconds):
                return await session.list_tools()
        except Exception as exc:
            raise MCPToolDiscoveryError(
                f"MCP Server '{config().name}' 工具发现失败: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # 函数说明：_StdioSessionLifecycle.invoke
    # 用途：调用_StdioSessionLifecycle，供MCP 连接与外部工具适配使用。
    # 参数：
    #   config：运行配置，类型 `Callable[[], MCPServerConfig]`。
    #   session：`session`输入或配置值，类型 `ClientSession`。
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `CallToolResult`；返回 `await session.call_tool(name, arguments)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.timeout` → `config` →
    # `session.call_tool`。
    # 资源/并发边界：`asyncio.timeout(config().call_timeout_seconds)`，上下文退出时执行
    # 相应清理。
    # 分支与异常：
    #   捕获 `Exception` 后，转换或抛出 `MCPToolCallError(f"MCP 工具 '{config().name}/{
    # name}' 调用失败: {type(exc).__name__}: {exc}")`。
    @staticmethod
    async def invoke(
        config: Callable[[], MCPServerConfig],
        session: ClientSession,
        name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        try:
            async with asyncio.timeout(config().call_timeout_seconds):
                return await session.call_tool(name, arguments)
        except Exception as exc:
            raise MCPToolCallError(
                f"MCP 工具 '{config().name}/{name}' 调用失败: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # 函数说明：_StdioSessionLifecycle.close
    # 用途：关闭_StdioSessionLifecycle，供MCP 连接与外部工具适配使用。
    # 参数：
    #   stack：`stack`输入或配置值，类型 `AsyncExitStack | None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`stack.aclose`。
    @staticmethod
    async def close(stack: AsyncExitStack | None) -> None:
        if stack is not None:
            await stack.aclose()

# 函数说明：_close_quietly
# 用途：关闭`quietly`，供MCP 连接与外部工具适配使用。
# 参数：
#   stack：`stack`输入或配置值，类型 `AsyncExitStack`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`stack.aclose`。
# 分支与异常：
#   捕获 `BaseException` 后，忽略该异常并继续当前流程。
async def _close_quietly(stack: AsyncExitStack) -> None:
    try:
        await stack.aclose()
    except BaseException:
        pass


# 函数说明：_resolve_environment
# 用途：解析或定位`environment`，供MCP 连接与外部工具适配使用。
# 参数：
#   config：运行配置，类型 `MCPServerConfig`。
# 返回：类型 `dict[str, str]`；返回 `_LaunchPolicy.resolve_environment(config)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_LaunchPolicy.resolve_environment`。
def _resolve_environment(config: MCPServerConfig) -> dict[str, str]:
    return _LaunchPolicy.resolve_environment(config)


class _LaunchPolicy:
    """Resolve explicit environment bindings and delegate sandbox launch policy."""

    # 函数说明：_LaunchPolicy.prepare
    # 用途：准备_LaunchPolicy，供MCP 连接与外部工具适配使用。
    # 参数：
    #   config：运行配置，类型 `MCPServerConfig`。
    #   supervisor：`supervisor`输入或配置值，类型 `SandboxSupervisor | None`。
    #   environment：`environment`输入或配置值，类型 `dict[str, str]`。
    # 返回：类型 `SandboxLaunchSpec`；按分支返回 `SandboxLaunchSpec(…)`；
    # `supervisor.prepare_launch(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SandboxLaunchSpec` → `os.getcwd`
    # → `supervisor.prepare_launch`。
    # 分支与异常：
    #   当 `supervisor is None` 时，返回 `SandboxLaunchSpec(…)`。
    @staticmethod
    def prepare(
        config: MCPServerConfig,
        supervisor: SandboxSupervisor | None,
        environment: dict[str, str],
    ) -> SandboxLaunchSpec:
        if supervisor is None:
            return SandboxLaunchSpec(
                command=config.command,
                args=config.args,
                cwd=config.cwd or os.getcwd(),
                env=environment,
                backend="host_unmanaged",
                sandboxed=False,
            )
        return supervisor.prepare_launch(
            command=config.command,
            args=config.args,
            env=environment,
            cwd=config.cwd,
            config=config.sandbox,
            persistent=True,
        )

    # 函数说明：_LaunchPolicy.resolve_environment
    # 用途：解析或定位`environment`，供MCP 连接与外部工具适配使用。
    # 参数：
    #   config：运行配置，类型 `MCPServerConfig`。
    # 返回：类型 `dict[str, str]`；返回 `resolved`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_ENV_REFERENCE.fullmatch` →
    # `match.group`。
    # 分支与异常：
    #   `match is None` 分支在完成前置处理后跳过当前循环项。
    #   当 `variable not in os.environ` 时，抛出 `MCPConnectionError(…)`。
    @staticmethod
    def resolve_environment(config: MCPServerConfig) -> dict[str, str]:
        resolved = {
            key: os.environ[key]
            for key in _SAFE_INHERITED_ENVIRONMENT
            if os.environ.get(key)
        }
        for key, value in config.env.items():
            match = _ENV_REFERENCE.fullmatch(value)
            if match is None:
                resolved[key] = value
                continue
            variable = match.group(1)
            if variable not in os.environ:
                raise MCPConnectionError(
                    f"MCP Server '{config.name}' 缺少环境变量 {variable}"
                )
            resolved[key] = os.environ[variable]
        return resolved

__all__ = ["MCPClientProtocol", "StdioMCPClient", "serialize_mcp_result"]
