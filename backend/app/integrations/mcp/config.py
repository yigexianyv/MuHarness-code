
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from app.paths import runtime_data_path

from .errors import MCPConfigurationError
from .models import MCPServerConfig, MCPSettings

DEFAULT_MCP_CONFIG_PATH = runtime_data_path("mcp.json")


# 函数说明：load_mcp_settings
# 用途：加载设置，供MCP 连接与外部工具适配使用。
# 参数：
#   path：目标文件或目录路径，类型 `str | Path`；默认 `DEFAULT_MCP_CONFIG_PATH`。
# 返回：类型 `MCPSettings`；按分支返回 `MCPSettings()`；
# `MCPSettings.model_validate(payload)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(path).expanduser().resolve` →
# `Path(path).expanduser` → `Path` → `config_path.exists` → `MCPSettings` →
# `asyncio.to_thread`；另有 2 个调用点。
# 分支与异常：
#   当 `not config_path.exists()` 时，返回 `MCPSettings()`。
#   捕获 `(OSError, json.JSONDecodeError, ValidationError)` 后，转换或抛出 `
# MCPConfigurationError(f'无法加载 MCP 配置 {config_path}: {type(exc).__name__}: {exc}')
# `。
# 副作用与资源：
#   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
async def load_mcp_settings(
    path: str | Path = DEFAULT_MCP_CONFIG_PATH,
) -> MCPSettings:

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        return MCPSettings()
    try:
        raw = await asyncio.to_thread(config_path.read_text, encoding="utf-8")
        payload = json.loads(raw)
        return MCPSettings.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise MCPConfigurationError(
            f"无法加载 MCP 配置 {config_path}: {type(exc).__name__}: {exc}"
        ) from exc


class MCPConfigurationStore:

    # 函数说明：MCPConfigurationStore.__init__
    # 用途：初始化 MCPConfigurationStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   path：目标文件或目录路径，类型 `str | Path`；默认 `DEFAULT_MCP_CONFIG_PATH`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(path).expanduser().resolve`
    # → `Path(path).expanduser` → `Path` → `asyncio.Lock`。
    # 副作用与资源：
    #   更新对象字段：`self.path`、`self._lock`、`self._restart_required`。
    def __init__(self, path: str | Path = DEFAULT_MCP_CONFIG_PATH) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = asyncio.Lock()
        self._restart_required: set[str] = set()

    # 函数说明：MCPConfigurationStore.load
    # 用途：加载MCPConfigurationStore，供MCP 连接与外部工具适配使用。
    # 返回：类型 `MCPSettings`；返回 `await load_mcp_settings(self.path)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`load_mcp_settings`。
    async def load(self) -> MCPSettings:

        return await load_mcp_settings(self.path)

    # 函数说明：MCPConfigurationStore.add
    # 用途：添加MCPConfigurationStore，供MCP 连接与外部工具适配使用。
    # 参数：
    #   server：服务输入或配置值，类型 `MCPServerConfig`。
    # 返回：类型 `MCPSettings`；返回 `await self.add_many((server,))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.add_many`。
    async def add(self, server: MCPServerConfig) -> MCPSettings:

        return await self.add_many((server,))

    # 函数说明：MCPConfigurationStore.add_many
    # 用途：添加`many`，供MCP 连接与外部工具适配使用。
    # 参数：
    #   servers：`servers`输入或配置值，类型 `tuple[MCPServerConfig, ...]`。
    # 返回：类型 `MCPSettings`；返回 `updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.load` → `next` →
    # `MCPSettings` → `asyncio.to_thread` → `self._restart_required.update`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `len(incoming_names) != len(set(incoming_names))` 时，抛出
    # `ValueError('duplicate MCP Server names in import')`。
    #   当 `duplicate is not None` 时，抛出
    # `ValueError(f"MCP Server '{duplicate}' already exists")`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def add_many(
        self,
        servers: tuple[MCPServerConfig, ...],
    ) -> MCPSettings:

        async with self._lock:
            current = await self.load()
            incoming_names = [server.name for server in servers]
            if len(incoming_names) != len(set(incoming_names)):
                raise ValueError("duplicate MCP Server names in import")
            existing_names = {item.name for item in current.servers}
            duplicate = next(
                (name for name in incoming_names if name in existing_names),
                None,
            )
            if duplicate is not None:
                raise ValueError(f"MCP Server '{duplicate}' already exists")
            updated = MCPSettings(servers=(*current.servers, *servers))
            await asyncio.to_thread(self._write, updated)
            self._restart_required.update(incoming_names)
            return updated

    # 函数说明：MCPConfigurationStore.set_enabled
    # 用途：设置`enabled`，供MCP 连接与外部工具适配使用。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   enabled：`enabled`输入或配置值，类型 `bool`。
    # 返回：类型 `MCPServerConfig`；返回 `updated_server`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.load` → `next` →
    # `MCPSettings` → `asyncio.to_thread` → `self._restart_required.add`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `found is None` 时，抛出 `KeyError(f"MCP Server '{name}' not found")`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def set_enabled(self, name: str, *, enabled: bool) -> MCPServerConfig:

        async with self._lock:
            current = await self.load()
            found = next((item for item in current.servers if item.name == name), None)
            if found is None:
                raise KeyError(f"MCP Server '{name}' not found")
            updated_server = found.model_copy(update={"enabled": enabled})
            updated = MCPSettings(
                servers=tuple(
                    updated_server if item.name == name else item
                    for item in current.servers
                )
            )
            await asyncio.to_thread(self._write, updated)
            self._restart_required.add(name)
            return updated_server

    # 函数说明：MCPConfigurationStore.delete
    # 用途：删除MCPConfigurationStore，供MCP 连接与外部工具适配使用。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.load` → `MCPSettings` →
    # `asyncio.to_thread` → `self._restart_required.add`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `not any((item.name == name for item in current.servers))` 时，抛出
    # `KeyError(f"MCP Server '{name}' not found")`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def delete(self, name: str) -> None:

        async with self._lock:
            current = await self.load()
            if not any(item.name == name for item in current.servers):
                raise KeyError(f"MCP Server '{name}' not found")
            updated = MCPSettings(
                servers=tuple(item for item in current.servers if item.name != name)
            )
            await asyncio.to_thread(self._write, updated)
            self._restart_required.add(name)

    # 函数说明：MCPConfigurationStore.restart_required
    # 用途：返回 `name in self._restart_required`，提供 MCPConfigurationStore 的派生值。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `bool`；返回 `name in self._restart_required`。
    def restart_required(self, name: str) -> bool:

        return name in self._restart_required

    # 函数说明：MCPConfigurationStore.has_pending_changes
    # 用途：判断待处理项是否满足当前实现的条件。
    # 返回：类型 `bool`；返回 `bool(self._restart_required)`。
    @property
    def has_pending_changes(self) -> bool:

        return bool(self._restart_required)

    # 函数说明：MCPConfigurationStore._write
    # 用途：写入MCPConfigurationStore，供MCP 连接与外部工具适配使用。
    # 参数：
    #   settings：业务或模型设置，类型 `MCPSettings`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.path.parent.mkdir` →
    # `json.dumps` → `self.path.with_name` → `uuid4` → `temporary.open` → `handle.write`
    # ；另有 6 个调用点。
    # 副作用与资源：
    #   文件或资源访问：`self.path.parent.mkdir`、`temporary.open`、`os.replace`、
    # `temporary.unlink`。
    def _write(self, settings: MCPSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            settings.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        temporary = self.path.with_name(
            f".{self.path.name}.{uuid4().hex}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


__all__ = [
    "DEFAULT_MCP_CONFIG_PATH",
    "MCPConfigurationStore",
    "load_mcp_settings",
]
