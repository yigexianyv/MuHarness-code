
from __future__ import annotations

import asyncio
import os
import shutil
import signal
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.models.types import ToolDefinition, ToolPermission
from app.safety.sandbox import (
    SandboxConfig,
    SandboxFilesystemMode,
    SandboxLaunchSpec,
    SandboxNetworkMode,
    SandboxSupervisor,
)

from ..base import BaseTool
from ..role_boundary import READ_ONLY_MODES
from ._workspace import resolve_workspace_path, workspace_root_path

if TYPE_CHECKING:
    from ..hooks import ToolExecutionContext

MAX_SHELL_TIMEOUT_SECONDS = 120.0


class ShellCommandTool(BaseTool):
    # 函数说明：ShellCommandTool.__init__
    # 用途：初始化 ShellCommandTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path | None`；默认
    # `None`。
    #   sandbox_supervisor：`sandbox_supervisor`输入或配置值，类型
    # `SandboxSupervisor | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace_root_path` →
    # `SandboxSupervisor`。
    # 副作用与资源：
    #   更新对象字段：`self._workspace_root`、`self._sandbox_supervisor`。
    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        sandbox_supervisor: SandboxSupervisor | None = None,
    ) -> None:
        self._workspace_root = workspace_root_path(workspace_root)
        self._sandbox_supervisor = sandbox_supervisor or SandboxSupervisor(
            self._workspace_root
        )

    # 函数说明：ShellCommandTool.definition
    # 用途：提供 ShellCommandTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_shell_command",
            description=(
                "在工作区沙箱内用 /bin/sh 执行命令，返回退出码、标准输出、标准错误和超时状态。"
                "需要运行测试、构建或其他本地程序时使用；简单列目录、读写文本优先用对应文件"
                "工具。模型工具名不等于沙箱中的可执行程序，不通过 shell 猜测工具入口或绕过"
                "审批、只读角色和沙箱限制。需要人工审批，沙箱网络禁用。默认从工作区根目录"
                "启动，子目录用 working_directory 指定，不猜工作区目录名。依赖命令用 &&，"
                "不要用结尾 echo 掩盖失败。调用成功不代表命令成功；检查 exit_code、timed_out"
                "及输出，退出码为零也只支持实际执行的检查，不能证明整个任务完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "交给 /bin/sh 执行的非空命令；确保适用于沙箱环境，依赖步骤用 && 并保留失败退出码。",
                    },
                    "working_directory": {
                        "type": "string",
                        "description": (
                            "可选的工作区相对子目录，省略从工作区根目录启动；"
                            "不接受绝对路径或越界路径。"
                        ),
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": (
                            "正数超时秒数，默认 30；"
                            f"最多 {MAX_SHELL_TIMEOUT_SECONDS:g} 秒，超时后终止进程，"
                            "已产生的文件或其他局部效果不因此回滚。"
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            strict=True,
            permission=ToolPermission.HUMAN_APPROVAL,
        )

    # 函数说明：ShellCommandTool.execute
    # 用途：执行ShellCommandTool，供内置工作区工具使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；返回
    # `await self._run(arguments, filesystem=SandboxFilesystemMode.WORKSPACE_WRITE)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._run(
            arguments,
            filesystem=SandboxFilesystemMode.WORKSPACE_WRITE,
        )

    # 函数说明：ShellCommandTool.execute_with_context
    # 用途：执行上下文，供内置工作区工具使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；返回
    # `await self._run(arguments, filesystem=filesystem)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._run`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        # Auditor 等只读角色：workspace 以只读方式挂进沙箱，写文件会在容器里失败
        filesystem = (
            SandboxFilesystemMode.READ_ONLY
            if context.mode in READ_ONLY_MODES
            else SandboxFilesystemMode.WORKSPACE_WRITE
        )
        return await self._run(arguments, filesystem=filesystem)

    # 函数说明：ShellCommandTool._run
    # 用途：运行ShellCommandTool，供内置工作区工具使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `command`、`working_directory`、`timeout_seconds`。
    #   filesystem：`filesystem`输入或配置值，类型 `SandboxFilesystemMode`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `command`、`working_directory`、
    # `exit_code`、`timed_out`、`stdout`、`stderr`、`duration_ms`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`resolve_workspace_path` →
    # `perf_counter` → `_shell_environment` → `self._sandbox_supervisor.prepare_launch`
    # → `SandboxConfig` → `asyncio.create_subprocess_exec`；另有 9 个调用点。
    # 资源/并发边界：`asyncio.timeout(timeout)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `not isinstance(command, str) or not command.strip()` 时，抛出
    # `ValueError("'command' must be a non-empty string")`。
    #   当 `not isinstance(raw_directory, str)` 时，抛出
    # `ValueError("'working_directory' must be a string")`。
    #   当 `not isinstance(timeout, (int, float)) or timeout <= 0` 时，抛出
    # `ValueError("'timeout_seconds' must be a positive number")`。
    #   捕获 `TimeoutError` 后，执行异常处理调用 `_terminate_process`、
    # `process.communicate`。
    #   捕获 `asyncio.CancelledError` 后，重新抛出原异常。
    async def _run(
        self,
        arguments: dict[str, Any],
        *,
        filesystem: SandboxFilesystemMode,
    ) -> dict[str, Any]:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("'command' must be a non-empty string")

        cwd = self._workspace_root
        raw_directory = arguments.get("working_directory")
        if raw_directory is not None:
            if not isinstance(raw_directory, str):
                raise ValueError("'working_directory' must be a string")
            cwd = resolve_workspace_path(
                self._workspace_root,
                raw_directory,
                allow_root=True,
            )

        timeout = arguments.get("timeout_seconds", 30.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("'timeout_seconds' must be a positive number")
        timeout = min(float(timeout), MAX_SHELL_TIMEOUT_SECONDS)

        started_at = perf_counter()
        timed_out = False
        launch: SandboxLaunchSpec | None = None
        process: asyncio.subprocess.Process | None = None
        try:
            environment = _shell_environment()
            launch = self._sandbox_supervisor.prepare_launch(
                command="/bin/sh",
                args=("-c", command),
                env=environment,
                cwd=str(cwd),
                config=SandboxConfig(
                    filesystem=filesystem,
                    network=SandboxNetworkMode.DENIED,
                ),
            )
            process = await asyncio.create_subprocess_exec(
                launch.command,
                *launch.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=launch.cwd,
                env=launch.env,
                start_new_session=True,
            )
            async with asyncio.timeout(timeout):
                stdout_bytes, stderr_bytes = await process.communicate()
        except TimeoutError:
            timed_out = True
            if process is not None:
                _terminate_process(process)
                stdout_bytes, stderr_bytes = await process.communicate()
            else:
                stdout_bytes, stderr_bytes = b"", b""
        except asyncio.CancelledError:
            if process is not None:
                _terminate_process(process)
                with suppress(Exception):
                    await process.communicate()
            raise
        finally:
            if launch is not None:
                await _cleanup_sandbox_launch(launch)

        if process is None:
            raise RuntimeError("shell process was not created")

        duration_ms = (perf_counter() - started_at) * 1000
        return {
            "command": command,
            "working_directory": cwd.as_posix(),
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "duration_ms": round(duration_ms, 3),
        }


# 函数说明：_terminate_process
# 用途：处理`terminate`，供内置工作区工具使用。
# 参数：
#   process：`process`输入或配置值，类型 `asyncio.subprocess.Process`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`os.killpg` → `os.getpgid` →
# `process.kill`。
# 分支与异常：
#   捕获 `(ProcessLookupError, PermissionError)` 后，忽略该异常并继续当前流程。
def _terminate_process(process: asyncio.subprocess.Process) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass


# 函数说明：_shell_environment
# 用途：在内置工作区工具中处理 `_shell_environment`，通过 `os.environ.get` 完成首个内部
# 处理步骤。
# 返回：类型 `dict[str, str]`；返回
# `{key: os.environ[key] for key in safe_keys if os.environ.get(key)}`。
def _shell_environment() -> dict[str, str]:

    safe_keys = (
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "TMPDIR",
        "COMSPEC",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
    )
    return {key: os.environ[key] for key in safe_keys if os.environ.get(key)}


# 函数说明：_cleanup_sandbox_launch
# 用途：清理`sandbox_launch`，供内置工作区工具使用。
# 参数：
#   launch：`launch`输入或配置值，类型 `SandboxLaunchSpec`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`suppress` →
# `asyncio.create_subprocess_exec` → `asyncio.timeout` → `process.communicate` → `Path`
# → `asyncio.to_thread`。
# 资源/并发边界：`asyncio.timeout(10)`，上下文退出时执行相应清理。
# 副作用与资源：
#   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
async def _cleanup_sandbox_launch(launch: SandboxLaunchSpec) -> None:

    if launch.cleanup_command:
        with suppress(Exception):
            process = await asyncio.create_subprocess_exec(
                *launch.cleanup_command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=launch.cwd,
                env=launch.env,
            )
            async with asyncio.timeout(10):
                await process.communicate()
    for raw_path in launch.cleanup_paths:
        path = Path(raw_path)
        with suppress(OSError):
            await asyncio.to_thread(shutil.rmtree, path)
