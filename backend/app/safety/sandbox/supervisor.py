
from __future__ import annotations

from pathlib import Path

from app.paths import preferred_env

from .backends import (
    DockerSandboxBackend,
    SandboxBackend,
    UnsupportedSandboxBackend,
    resolve_executable,
)
from .errors import SandboxPolicyError
from .models import (
    SandboxConfig,
    SandboxFilesystemMode,
    SandboxLaunchSpec,
    SandboxPolicy,
)

_PROTECTED_RELATIVE_PATHS = (
    ".git",
    ".muharness",
    "backend/.muharness",
    ".env",
    "backend/.env",
)


class SandboxSupervisor:

    # 函数说明：SandboxSupervisor.__init__
    # 用途：初始化 SandboxSupervisor；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path`。
    #   native_backend：`native_backend`输入或配置值，类型 `SandboxBackend | None`；默认
    #  `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(workspace_root).expanduser().resolve` → `Path(workspace_root).expanduser` →
    # `Path` → `_platform_backend`。
    # 副作用与资源：
    #   更新对象字段：`self.workspace_root`、`self._native_backend`。
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        native_backend: SandboxBackend | None = None,
        instance_id: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self._native_backend = native_backend or _platform_backend(
            self.workspace_root,
            instance_id=instance_id,
        )

    async def remove_orphans(self) -> int:
        """清理上一次进程被强杀时遗留的沙箱，返回清理数量。"""
        return await self._native_backend.remove_orphans()

    # 函数说明：SandboxSupervisor.prepare_launch
    # 用途：根据工作目录和额外挂载生成沙箱启动参数。
    # 参数：
    #   command：待执行的 Shell 命令，类型 `str`。
    #   args：`args`输入或配置值，类型 `tuple[str, ...]`。
    #   env：环境变量映射，类型 `dict[str, str]`。
    #   cwd：传给 `self._resolve_working_directory` 的输入，类型 `str | None`。
    #   config：运行配置，类型 `SandboxConfig`。
    #   persistent：`persistent`输入或配置值，类型 `bool`；默认 `False`。
    # 返回：类型 `SandboxLaunchSpec`；返回
    # `backend.prepare(command=resolved_command, args=args, env=env, policy=policy)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`resolve_executable` →
    # `self._resolve_working_directory` → `self._build_policy` → `backend.prepare`。
    # 分支与异常：
    #   当 `persistent and (not backend.supports_persistent_processes)` 时，抛出
    # `SandboxPolicyError('当前 Docker V1 仅隔离一次性 Shell，不支持长驻 MCP 进程')`。
    def prepare_launch(
        self,
        *,
        command: str,
        args: tuple[str, ...],
        env: dict[str, str],
        cwd: str | None,
        config: SandboxConfig,
        persistent: bool = False,
    ) -> SandboxLaunchSpec:
        """根据工作目录和额外挂载生成沙箱启动参数。"""
        backend = self._native_backend
        if persistent and not backend.supports_persistent_processes:
            raise SandboxPolicyError(
                "当前 Docker V1 仅隔离一次性 Shell，不支持长驻 MCP 进程"
            )
        resolved_command: str | Path
        command_path: Path | None
        if backend.resolve_executable_on_host:
            command_path = resolve_executable(command, env=env)
            resolved_command = command_path
        else:
            command_path = None
            resolved_command = command
        working_directory = self._resolve_working_directory(cwd)
        policy = self._build_policy(
            config,
            command=command_path,
            working_directory=working_directory,
            env=env,
        )
        return backend.prepare(
            command=resolved_command,
            args=args,
            env=env,
            policy=policy,
        )

    # 函数说明：SandboxSupervisor._build_policy
    # 用途：将请求的目录与权限转换为可执行的沙箱策略。
    # 参数：
    #   config：运行配置，类型 `SandboxConfig`。
    #   command：待执行的 Shell 命令，类型 `Path | None`。
    #   working_directory：目录输入或配置值，类型 `Path`。
    #   env：环境变量映射，类型 `dict[str, str]`。
    # 返回：类型 `SandboxPolicy`；返回 `SandboxPolicy(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`command.resolve` →
    # `self._resolve_extra_root` → `_runtime_support_roots` →
    # `(path := (self.workspace_root / relative)).exists` → `SandboxPolicy` →
    # `_deduplicate`。
    def _build_policy(
        self,
        config: SandboxConfig,
        *,
        command: Path | None,
        working_directory: Path,
        env: dict[str, str],
    ) -> SandboxPolicy:
        """将请求的目录与权限转换为可执行的沙箱策略。"""
        readable: list[Path] = []
        if command is not None:
            readable.extend((command.parent, command.resolve().parent))
        writable: list[Path] = []
        if config.filesystem in {
            SandboxFilesystemMode.READ_ONLY,
            SandboxFilesystemMode.WORKSPACE_WRITE,
        }:
            readable.append(self.workspace_root)
        if config.filesystem is SandboxFilesystemMode.WORKSPACE_WRITE:
            writable.append(self.workspace_root)
        readable.extend(
            self._resolve_extra_root(value) for value in config.readable_roots
        )
        for value in config.writable_roots:
            root = self._resolve_extra_root(value)
            readable.append(root)
            writable.append(root)

        if command is not None:
            runtime_read, runtime_write = _runtime_support_roots(command, env)
            readable.extend(runtime_read)
            writable.extend(runtime_write)
        denied = tuple(
            path
            for relative in _PROTECTED_RELATIVE_PATHS
            if (path := self.workspace_root / relative).exists()
        )
        return SandboxPolicy(
            filesystem=config.filesystem,
            network=config.network,
            working_directory=working_directory,
            readable_roots=_deduplicate(readable),
            writable_roots=_deduplicate(writable),
            denied_read_paths=denied,
            denied_write_paths=denied,
            allowed_domains=config.allowed_domains,
        )

    # 函数说明：SandboxSupervisor._resolve_working_directory
    # 用途：解析或定位目录，供隔离执行与沙箱生命周期使用。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str | None`。
    # 返回：类型 `Path`；按分支返回 `self.workspace_root`；
    # `self._resolve_extra_root(value)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._resolve_extra_root`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `self.workspace_root`。
    def _resolve_working_directory(self, value: str | None) -> Path:
        if value is None:
            return self.workspace_root
        return self._resolve_extra_root(value)

    # 函数说明：SandboxSupervisor._resolve_extra_root
    # 用途：解析或定位`extra_root`，供隔离执行与沙箱生命周期使用。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `Path`；返回 `resolved`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(value).expanduser` → `Path`
    # → `candidate.is_absolute` → `candidate.resolve` → `resolved.is_relative_to` →
    # `resolved.exists`。
    # 分支与异常：
    #   当 `is_relative and (not resolved.is_relative_to(…` 时，抛出
    # `SandboxPolicyError(f'沙箱相对路径不能越过 workspace：{value}')`。
    #   当 `not resolved.exists()` 时，抛出
    # `SandboxPolicyError(f'沙箱路径不存在：{resolved}')`。
    def _resolve_extra_root(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        is_relative = not candidate.is_absolute()
        if is_relative:
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        if is_relative and not resolved.is_relative_to(self.workspace_root):
            raise SandboxPolicyError(
                f"沙箱相对路径不能越过 workspace：{value}"
            )
        if not resolved.exists():
            raise SandboxPolicyError(f"沙箱路径不存在：{resolved}")
        return resolved


# 函数说明：_platform_backend
# 用途：在隔离执行与沙箱生命周期中处理 `_platform_backend`，通过 `preferred_env('
# MUHARNESS_SANDBOX_BACKEND', 'auto').strip().…` 完成首个内部处
# 理步骤。
# 参数：
#   workspace_root：文件工具允许访问的工作区根目录，类型 `Path`。
# 返回：类型 `SandboxBackend`；按分支返回
# `UnsupportedSandboxBackend(f'未知 MUHARNESS_SANDBOX_BACKEND={requested!r}')`；
# `DockerSandboxBackend(workspace_root)`；`UnsupportedSandboxBackend('docker disabled')`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `preferred_env('MUHARNESS_SANDBOX_BACKEND', 'auto'…` →
# `preferred_env` → `UnsupportedSandboxBackend` → `DockerSandboxBackend`。
# 分支与异常：
#   当 `requested not in {'auto', 'docker', 'unsupported'}` 时，返回
# `UnsupportedSandboxBackend(…)`。
#   当 `requested in {'auto', 'docker'}` 时，返回 `DockerSandboxBackend(workspace_root)`
# 。
def _platform_backend(
    workspace_root: Path,
    *,
    instance_id: str | None = None,
) -> SandboxBackend:
    requested = preferred_env(
        "MUHARNESS_SANDBOX_BACKEND", "auto"
    ).strip().lower()
    if requested not in {"auto", "docker", "unsupported"}:
        return UnsupportedSandboxBackend(
            f"未知 MUHARNESS_SANDBOX_BACKEND={requested!r}"
        )
    if requested in {"auto", "docker"}:
        return DockerSandboxBackend(workspace_root, instance_id=instance_id)
    return UnsupportedSandboxBackend("docker disabled")


# 函数说明：_runtime_support_roots
# 用途：在隔离执行与沙箱生命周期中处理 `_runtime_support_roots`，通过 `env.get` 完成首个
# 内部处理步骤。
# 参数：
#   command：待执行的 Shell 命令，类型 `Path`。
#   env：环境变量映射，类型 `dict[str, str]`；读取键 `HOME`。
# 返回：类型 `tuple[tuple[Path, ...], tuple[Path, ...]]`；按分支返回 `((), ())`；`(tuple
# ((path.resolve() for path in readable if path.exists())), tuple((path.resolve() for…`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `Path(home_value).expanduser().resolve` → `Path(home_value).expanduser` → `Path` →
# `command.name.lower` → `command.absolute` → `command_parts.index`；另有 2 个调用点。
# 分支与异常：
#   当 `not home_value` 时，返回 `((), ())`。
def _runtime_support_roots(
    command: Path,
    env: dict[str, str],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:

    home_value = env.get("HOME")
    if not home_value:
        return (), ()
    home = Path(home_value).expanduser().resolve()
    readable: list[Path] = []
    writable: list[Path] = []
    name = command.name.lower()
    if name in {"uv", "uvx"}:
        readable.extend((home / ".cache" / "uv", home / ".local" / "share" / "uv"))
        writable.extend((home / ".cache" / "uv", home / ".local" / "share" / "uv"))
    if name in {"node", "npm", "npx", "pnpm", "yarn", "bun", "bunx"}:
        readable.extend((home / ".npm", home / ".cache" / "node"))
        writable.extend((home / ".npm", home / ".cache" / "node"))
    command_parts = command.absolute().parts
    if "node_modules" in command_parts:
        index = command_parts.index("node_modules")
        readable.append(Path(*command_parts[: index + 1]))
    return (
        tuple(path.resolve() for path in readable if path.exists()),
        tuple(path.resolve() for path in writable if path.exists()),
    )


# 函数说明：_deduplicate
# 用途：返回 `tuple(dict.fromkeys((path.resolve() for path in paths if path.exists())))`
# ，提供 隔离执行与沙箱生命周期 的派生值。
# 参数：
#   paths：待处理的路径集合，类型 `list[Path]`。
# 返回：类型 `tuple[Path, ...]`；返回
# `tuple(dict.fromkeys((path.resolve() for path in paths if path.exists())))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dict.fromkeys` → `path.resolve` →
# `path.exists`。
def _deduplicate(paths: list[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(path.resolve() for path in paths if path.exists()))


__all__ = ["SandboxSupervisor"]
