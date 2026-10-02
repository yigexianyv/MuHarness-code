
from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SandboxFilesystemMode(StrEnum):

    NONE = "none"
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"


class SandboxNetworkMode(StrEnum):

    DENIED = "denied"
    UNRESTRICTED = "unrestricted"


class SandboxConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    filesystem: SandboxFilesystemMode = SandboxFilesystemMode.WORKSPACE_WRITE
    network: SandboxNetworkMode = SandboxNetworkMode.UNRESTRICTED
    readable_roots: tuple[str, ...] = ()
    writable_roots: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()

    # 函数说明：SandboxConfig.reject_empty_values
    # 用途：校验并规范化模型字段 'readable_roots'、'writable_roots'、'allowed_domains'。
    # 参数：
    #   values：待处理的值集合，类型 `tuple[str, ...]`。
    # 返回：类型 `tuple[str, ...]`；返回 `normalized`。
    # 分支与异常：
    #   当 `any((not value for value in normalized))` 时，抛出
    # `ValueError('sandbox list values cannot be empty')`。
    #   当 `len(normalized) != len(set(normalized))` 时，抛出
    # `ValueError('sandbox list values cannot contain duplicates')`。
    @field_validator("readable_roots", "writable_roots", "allowed_domains")
    @classmethod
    def reject_empty_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("sandbox list values cannot be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("sandbox list values cannot contain duplicates")
        return normalized


class SandboxPolicy(BaseModel):

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    filesystem: SandboxFilesystemMode
    network: SandboxNetworkMode
    working_directory: Path
    readable_roots: tuple[Path, ...]
    writable_roots: tuple[Path, ...]
    denied_read_paths: tuple[Path, ...] = ()
    denied_write_paths: tuple[Path, ...] = ()
    allowed_domains: tuple[str, ...] = ()


class SandboxLaunchSpec(BaseModel):

    model_config = ConfigDict(frozen=True)

    command: str
    args: tuple[str, ...]
    cwd: str
    env: dict[str, str] = Field(default_factory=dict)
    backend: str
    sandboxed: bool
    cleanup_command: tuple[str, ...] = ()
    cleanup_paths: tuple[str, ...] = ()


__all__ = [
    "SandboxConfig",
    "SandboxFilesystemMode",
    "SandboxLaunchSpec",
    "SandboxNetworkMode",
    "SandboxPolicy",
]
