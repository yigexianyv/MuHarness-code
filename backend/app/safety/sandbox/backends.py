
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from app.paths import preferred_env

from .errors import SandboxUnavailableError
from .models import (
    SandboxFilesystemMode,
    SandboxLaunchSpec,
    SandboxNetworkMode,
    SandboxPolicy,
)

logger = logging.getLogger("muharness.sandbox")

SANDBOX_LABEL = "com.muharness.sandbox"
INSTANCE_LABEL = "com.muharness.instance"


class SandboxBackend(ABC):

    resolve_executable_on_host = True
    supports_persistent_processes = True

    # 函数说明：SandboxBackend.prepare
    # 用途：准备SandboxBackend，供隔离执行与沙箱生命周期使用。
    # 参数：
    #   command：待执行的 Shell 命令，类型 `str | Path`。
    #   args：`args`输入或配置值，类型 `tuple[str, ...]`。
    #   env：环境变量映射，类型 `dict[str, str]`。
    #   policy：权限决策策略，类型 `SandboxPolicy`。
    # 返回：类型 `SandboxLaunchSpec`；不返回结果值（隐式 None）。
    @abstractmethod
    def prepare(
        self,
        *,
        command: str | Path,
        args: tuple[str, ...],
        env: dict[str, str],
        policy: SandboxPolicy,
    ) -> SandboxLaunchSpec:
        pass

    async def remove_orphans(self) -> int:
        """清理上一次进程遗留、仍在运行的沙箱；没有可清理的对象时返回 0。"""
        return 0


class UnsupportedSandboxBackend(SandboxBackend):

    # 函数说明：UnsupportedSandboxBackend.__init__
    # 用途：初始化 UnsupportedSandboxBackend；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   platform：`platform`输入或配置值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.platform`。
    def __init__(self, platform: str) -> None:
        self.platform = platform

    # 函数说明：UnsupportedSandboxBackend.prepare
    # 用途：准备UnsupportedSandboxBackend，供隔离执行与沙箱生命周期使用。
    # 参数：
    #   command：待执行的 Shell 命令，类型 `str | Path`。
    #   args：`args`输入或配置值，类型 `tuple[str, ...]`。
    #   env：环境变量映射，类型 `dict[str, str]`。
    #   policy：权限决策策略，类型 `SandboxPolicy`。
    # 返回：类型 `SandboxLaunchSpec`；不返回结果值（隐式 None）。
    def prepare(
        self,
        *,
        command: str | Path,
        args: tuple[str, ...],
        env: dict[str, str],
        policy: SandboxPolicy,
    ) -> SandboxLaunchSpec:
        del command, args, env, policy
        raise SandboxUnavailableError(
            f"平台 {self.platform!r} 尚无可用的 MuHarness 沙箱后端，拒绝降级执行"
        )


class DockerSandboxBackend(SandboxBackend):

    resolve_executable_on_host = False
    supports_persistent_processes = False

    # 函数说明：DockerSandboxBackend.__init__
    # 用途：初始化 DockerSandboxBackend；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path`。
    #   docker_command：传给 `str` 的输入，类型 `str | Path | None`；默认 `None`。
    #   image：`image`输入或配置值，类型 `str | None`；默认 `None`。
    #   memory：记忆输入或配置值，类型 `str`；默认 `'2g'`。
    #   cpus：`cpus`输入或配置值，类型 `str`；默认 `'2'`。
    #   pids_limit：上限输入或配置值，类型 `int`；默认 `128`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(workspace_root).expanduser().resolve` → `Path(workspace_root).expanduser` →
    # `Path` → `shutil.which` → `preferred_env`。
    # 副作用与资源：
    #   更新对象字段：`self.workspace_root`、`self.docker_command`、`self.image`、
    # `self.memory`、`self.cpus`、`self.pids_limit`。
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        docker_command: str | Path | None = None,
        image: str | None = None,
        memory: str = "2g",
        cpus: str = "2",
        pids_limit: int = 128,
        instance_id: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        # 区分同一台 Docker 上的多个后端：启动清理只删除本实例创建的容器
        self.instance_id = instance_id
        self.docker_command = (
            str(docker_command)
            if docker_command is not None
            else shutil.which("docker")
        )
        self.image = image or preferred_env(
            "MUHARNESS_SANDBOX_IMAGE",
            "muharness-sandbox:latest",
        )
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit

    # 函数说明：DockerSandboxBackend.prepare
    # 用途：将沙箱策略映射为 Docker 命令和挂载配置。
    # 参数：
    #   command：待执行的 Shell 命令，类型 `str | Path`。
    #   args：`args`输入或配置值，类型 `tuple[str, ...]`。
    #   env：环境变量映射，类型 `dict[str, str]`。
    #   policy：权限决策策略，类型 `SandboxPolicy`。
    # 返回：类型 `SandboxLaunchSpec`；返回 `SandboxLaunchSpec(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._reject_external_roots` →
    # `uuid.uuid4` → `_container_user` → `self._mounts` → `SandboxLaunchSpec` →
    # `shutil.rmtree`。
    # 分支与异常：
    #   当 `not self.docker_command` 时，抛出 `SandboxUnavailableError(…)`。
    #   当 `policy.allowed_domains` 时，抛出
    # `SandboxUnavailableError('Docker 后端尚不能强制域名白名单，拒绝弱化网络策略')`。
    #   捕获 `BaseException` 后，重新抛出原异常。
    def prepare(
        self,
        *,
        command: str | Path,
        args: tuple[str, ...],
        env: dict[str, str],
        policy: SandboxPolicy,
    ) -> SandboxLaunchSpec:
        """将沙箱策略映射为 Docker 命令和挂载配置。"""
        if not self.docker_command:
            raise SandboxUnavailableError(
                "Docker CLI 不可用；请安装 Docker Engine，并确保 docker 命令可用"
            )
        if policy.allowed_domains:
            raise SandboxUnavailableError(
                "Docker 后端尚不能强制域名白名单，拒绝弱化网络策略"
            )
        self._reject_external_roots(policy)

        container_name = f"muharness-sandbox-{uuid.uuid4().hex[:16]}"
        docker_args = [
            "run",
            "--rm",
            "--name",
            container_name,
            "--label",
            f"{SANDBOX_LABEL}=true",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--user",
            _container_user(),
        ]
        if self.instance_id:
            docker_args.extend(("--label", f"{INSTANCE_LABEL}={self.instance_id}"))
        if policy.network is SandboxNetworkMode.DENIED:
            docker_args.extend(("--network", "none"))

        cleanup_paths: tuple[str, ...] = ()
        try:
            container_cwd, mounts, cleanup_paths = self._mounts(policy)
            for mount in mounts:
                docker_args.extend(("--mount", mount))
            docker_args.extend(("--workdir", container_cwd, self.image))
            docker_args.extend((str(command), *args))
            return SandboxLaunchSpec(
                command=self.docker_command,
                args=tuple(docker_args),
                cwd=str(self.workspace_root),
                env=env,
                backend="docker",
                sandboxed=True,
                cleanup_command=(
                    self.docker_command,
                    "rm",
                    "--force",
                    "--volumes",
                    container_name,
                ),
                cleanup_paths=cleanup_paths,
            )
        except BaseException:
            for path in cleanup_paths:
                shutil.rmtree(path, ignore_errors=True)
            raise

    # 函数说明：DockerSandboxBackend._reject_external_roots
    # 用途：拒绝`external_roots`，供隔离执行与沙箱生命周期使用。
    # 参数：
    #   policy：权限决策策略，类型 `SandboxPolicy`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`root.is_relative_to`。
    # 分支与异常：
    #   当 `external` 时，抛出 `SandboxUnavailableError(…)`。
    async def remove_orphans(self) -> int:
        """删除本实例上次运行遗留的容器。

        正常结束时 Shell 会在 finally 里删除容器；后端进程被强杀时执行不到，
        容器会继续运行并写工作区。启动时、在恢复任何运行之前调用：此时本进程
        还没有启动沙箱，带本实例标签的容器一定是遗留的。没有 instance_id 时
        不清理，避免误删其他后端正在使用的容器。
        """
        if not self.docker_command or not self.instance_id:
            return 0
        listed = await _run_docker(
            self.docker_command,
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label={SANDBOX_LABEL}=true",
            "--filter",
            f"label={INSTANCE_LABEL}={self.instance_id}",
        )
        if listed is None:
            return 0
        container_ids = [line.strip() for line in listed.splitlines() if line.strip()]
        if not container_ids:
            return 0
        removed = await _run_docker(
            self.docker_command, "rm", "--force", "--volumes", *container_ids
        )
        if removed is None:
            return 0
        logger.warning(
            "removed %d orphaned sandbox container(s) left by a previous process",
            len(container_ids),
        )
        return len(container_ids)

    def _reject_external_roots(self, policy: SandboxPolicy) -> None:
        roots = (*policy.readable_roots, *policy.writable_roots)
        external = tuple(
            root for root in roots if not root.is_relative_to(self.workspace_root)
        )
        if external:
            raise SandboxUnavailableError(
                "Docker V1 不映射 workspace 外部路径，拒绝启动："
                + ", ".join(str(path) for path in external)
            )

    # 函数说明：DockerSandboxBackend._mounts
    # 用途：在隔离执行与沙箱生命周期中处理 `_mounts`，通过
    # `policy.working_directory.relative_to` 完成首个内部处理步骤。
    # 参数：
    #   policy：权限决策策略，类型 `SandboxPolicy`。
    # 返回：类型 `tuple[str, tuple[str, ...], tuple[str, ...]]`；按分支返回
    # `('/workspace', ('type=tmpfs,destination=/workspace,tmpfs-mode=0777',), ())`；
    # `(container_cwd, tuple(mounts), ())`；
    # `(container_cwd, tuple(mounts), (str(mask_root),))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `policy.working_directory.relative_to` → `relative_cwd.as_posix` →
    # `path.is_relative_to` → `Path` → `tempfile.mkdtemp` → `empty_directory.mkdir`；另
    # 有 5 个调用点。
    # 分支与异常：
    #   当 `policy.filesystem is SandboxFilesystemMode.NONE` 时，返回
    # `('/workspace', ('type=tmpfs,destination=/workspace,tmpfs-…`。
    #   捕获 `ValueError` 后，转换或抛出
    # `SandboxUnavailableError('Docker 工作目录必须位于 workspace 内')`。
    #   当 `not denied` 时，返回 `(container_cwd, tuple(mounts), ())`。
    # 副作用与资源：
    #   文件或资源访问：`empty_directory.mkdir`。
    def _mounts(
        self,
        policy: SandboxPolicy,
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        if policy.filesystem is SandboxFilesystemMode.NONE:
            return "/workspace", (
                "type=tmpfs,destination=/workspace,tmpfs-mode=0777",
            ), ()

        try:
            relative_cwd = policy.working_directory.relative_to(
                self.workspace_root
            )
        except ValueError as exc:
            raise SandboxUnavailableError(
                "Docker 工作目录必须位于 workspace 内"
            ) from exc
        container_cwd = "/workspace"
        if relative_cwd.parts:
            container_cwd += "/" + relative_cwd.as_posix()

        workspace_mount = (
            f"type=bind,source={self.workspace_root},target=/workspace"
        )
        if policy.filesystem is SandboxFilesystemMode.READ_ONLY:
            workspace_mount += ",readonly"
        mounts = [workspace_mount]

        denied = tuple(
            path
            for path in (*policy.denied_read_paths, *policy.denied_write_paths)
            if path.is_relative_to(self.workspace_root)
        )
        if not denied:
            return container_cwd, tuple(mounts), ()

        mask_root = Path(tempfile.mkdtemp(prefix="muharness-docker-mask-"))
        empty_directory = mask_root / "empty-directory"
        empty_file = mask_root / "empty-file"
        empty_directory.mkdir()
        empty_file.touch()
        for protected_path in dict.fromkeys(denied):
            relative = protected_path.relative_to(self.workspace_root)
            target = "/workspace/" + relative.as_posix()
            source = empty_directory if protected_path.is_dir() else empty_file
            mounts.append(
                f"type=bind,source={source},target={target},readonly"
            )
        return container_cwd, tuple(mounts), (str(mask_root),)


# 函数说明：_container_user
# 用途：处理隔离执行与沙箱生命周期中的 `_container_user` 数据；结果及边界条件见下方说明
# 。
# 返回：类型 `str`；按分支返回 `f'{uid}:{gid}'`；`'65532:65532'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`callable` → `getuid` → `getgid`。
# 分支与异常：
#   当 `uid != 0` 时，返回 `f'{uid}:{gid}'`。
def _container_user() -> str:

    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if callable(getuid) and callable(getgid):
        uid = getuid()
        gid = getgid()
        if uid != 0:
            return f"{uid}:{gid}"
    return "65532:65532"


# 函数说明：resolve_executable
# 用途：解析或定位`executable`，供隔离执行与沙箱生命周期使用。
# 参数：
#   command：待执行的 Shell 命令，类型 `str`。
#   env：环境变量映射，类型 `dict[str, str]`；读取键 `PATH`。
# 返回：类型 `Path`；返回 `executable`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(command).expanduser` → `Path` →
# `candidate.is_absolute` → `candidate.absolute` → `shutil.which` →
# `Path(found).absolute`；另有 3 个调用点。
# 分支与异常：
#   当 `found is None` 时，抛出
# `SandboxUnavailableError(f'找不到可执行文件：{command}')`。
#   捕获 `OSError` 后，转换或抛出
# `SandboxUnavailableError(f'可执行文件无效：{executable}')`。
#   当 `not resolved.is_file() or not executable.is_file()` 时，抛出
# `SandboxUnavailableError(f'可执行文件无效：{executable}')`。
def resolve_executable(command: str, *, env: dict[str, str]) -> Path:

    candidate = Path(command).expanduser()
    if candidate.is_absolute() or "/" in command:
        executable = candidate.absolute()
    else:
        found = shutil.which(command, path=env.get("PATH"))
        if found is None:
            raise SandboxUnavailableError(f"找不到可执行文件：{command}")
        executable = Path(found).absolute()
    try:
        resolved = executable.resolve(strict=True)
    except OSError as exc:
        raise SandboxUnavailableError(f"可执行文件无效：{executable}") from exc
    if not resolved.is_file() or not executable.is_file():
        raise SandboxUnavailableError(f"可执行文件无效：{executable}")
    return executable


async def _run_docker(
    docker_command: str, *args: str, timeout: float = 30
) -> str | None:
    """执行一条 docker 命令，返回标准输出；Docker 不可用或失败时返回 None。"""
    try:
        process = await asyncio.create_subprocess_exec(
            docker_command,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async with asyncio.timeout(timeout):
            stdout, stderr = await process.communicate()
    except (OSError, TimeoutError) as exc:
        logger.warning("docker %s failed: %s", args[0], exc)
        return None
    if process.returncode != 0:
        logger.warning(
            "docker %s exited with %s: %s",
            args[0],
            process.returncode,
            stderr.decode("utf-8", errors="replace").strip(),
        )
        return None
    return stdout.decode("utf-8", errors="replace")


__all__ = [
    "DockerSandboxBackend",
    "SandboxBackend",
    "UnsupportedSandboxBackend",
    "resolve_executable",
]
