
from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.types import ToolPermission
from app.safety.sandbox import SandboxConfig

_SERVER_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


class MCPTransport(StrEnum):

    STDIO = "stdio"


class MCPServerState(StrEnum):

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"


class MCPServerConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    name: str
    transport: MCPTransport = MCPTransport.STDIO
    command: str = Field(min_length=1)
    args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True
    startup_timeout_seconds: float = Field(default=15.0, gt=0)
    call_timeout_seconds: float = Field(default=30.0, gt=0)
    permission: ToolPermission = ToolPermission.HUMAN_APPROVAL
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    # 函数说明：MCPServerConfig.validate_name
    # 用途：校验并规范化模型字段 'name'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `value`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_SERVER_NAME_RE.fullmatch`。
    # 分支与异常：
    #   当 `not _SERVER_NAME_RE.fullmatch(value)` 时，抛出
    # `ValueError('name 只能包含字母、数字和下划线')`。
    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _SERVER_NAME_RE.fullmatch(value):
            raise ValueError("name 只能包含字母、数字和下划线")
        return value


class MCPSettings(BaseModel):

    model_config = ConfigDict(extra="forbid")

    servers: tuple[MCPServerConfig, ...] = ()

    # 函数说明：MCPSettings.reject_duplicate_names
    # 用途：校验并规范化模型字段 'servers'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `tuple[MCPServerConfig, ...]`。
    # 返回：类型 `tuple[MCPServerConfig, ...]`；返回 `value`。
    # 分支与异常：
    #   当 `len(names) != len(set(names))` 时，抛出
    # `ValueError('MCP Server 名称不能重复')`。
    @field_validator("servers")
    @classmethod
    def reject_duplicate_names(
        cls,
        value: tuple[MCPServerConfig, ...],
    ) -> tuple[MCPServerConfig, ...]:
        names = [server.name for server in value]
        if len(names) != len(set(names)):
            raise ValueError("MCP Server 名称不能重复")
        return value


class MCPRemoteTool(BaseModel):

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class MCPServerStatus(BaseModel):

    model_config = ConfigDict(extra="forbid")

    name: str
    state: MCPServerState
    tool_names: tuple[str, ...] = ()
    error: str | None = None
    sandboxed: bool | None = None
    sandbox_backend: str | None = None


__all__ = [
    "MCPRemoteTool",
    "MCPServerConfig",
    "MCPServerState",
    "MCPServerStatus",
    "MCPSettings",
    "MCPTransport",
]
