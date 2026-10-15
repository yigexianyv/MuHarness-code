"""工作区环境准备：用户触发、人工审批后，在独立的联网容器里执行准备脚本。

Agent 的 Shell 始终在断网的一次性容器中执行，这里是唯一允许联网的入口：

- 只执行用户放在工作区里的脚本；入口不是工具，模型无法触发。
- 审批展示脚本全文和 SHA-256；执行的是审批时读到的那份内容，之后改文件不影响本次执行。
- 审批直接走基础审批入口，不经过 ToolExecutor 的规则，也不经过长任务的沙箱自动批准。
- 准备容器结束并确认清理后，另起断网容器验证网络隔离和环境可用，全部通过才标记为已准备。
- 准备期间禁止启动新的 Run，避免 Agent 用到装了一半的依赖。

准备脚本联网时能读写工作区（受保护路径除外），这是用户在审批时接受的单次风险。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from app.models.types import Message, MessageRole
from app.safety.sandbox import (
    SandboxConfig,
    SandboxFilesystemMode,
    SandboxLaunchSpec,
    SandboxNetworkMode,
    SandboxSupervisor,
)
from app.tools.approval import (
    WORKSPACE_SETUP_TOOL_NAME,
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
)
from app.tools.builtin._workspace import resolve_workspace_path
# 复用 Shell 工具的进程清理和 docker 命令环境，保证两条路径行为一致
from app.tools.builtin.shell import (
    _cleanup_sandbox_launch,
    _shell_environment,
    _terminate_process,
)

logger = logging.getLogger("muharness.workspace_setup")

DEFAULT_SETUP_SCRIPT = "setup.sh"
DEFAULT_SETUP_TIMEOUT_SECONDS = 900.0
CHECK_TIMEOUT_SECONDS = 120.0
MAX_SCRIPT_BYTES = 64 * 1024
_OUTPUT_EXCERPT_BYTES = 4_000
_VENV_PYTHON = Path(".venv") / "bin" / "python"
# 容器内是 Linux：ENETUNREACH / EHOSTUNREACH。宿主机可能是 Windows，不能用本机 errno 常量
_UNREACHABLE_ERRNOS = {"101", "113"}
_NETWORK_PROBE = (
    "import socket, sys, time\n"
    "try:\n"
    "    socket.create_connection(('1.1.1.1', 443), timeout=3)\n"
    "    print('PROBE CONNECTED')\n"
    "except OSError as exc:\n"
    "    print(f'PROBE BLOCKED errno={exc.errno}')\n"
    "sys.stdout.flush()\n"
    # 留出时间让宿主机读取容器的网络配置
    "time.sleep(2)\n"
)


class SetupStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"


class SetupCheck(BaseModel):
    """passed 为 None 表示跳过（例如没有可验证的环境）。"""

    passed: bool | None
    detail: str


class WorkspaceSetupRecord(BaseModel):
    id: str
    status: SetupStatus
    script_path: str
    script_sha256: str
    verify_command: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    exit_code: int | None = None
    timed_out: bool = False
    log_path: str | None = None
    output_head: str = ""
    output_tail: str = ""
    network_check: SetupCheck | None = None
    environment_check: SetupCheck | None = None
    error: str | None = None


class WorkspaceSetupError(ValueError):
    """请求本身不合法：脚本不存在、过大或不在工作区内。"""


class WorkspaceSetupBusy(RuntimeError):
    """已有准备任务，或工作区里还有运行中的 Agent。"""


class WorkspaceSetupService:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        supervisor: SandboxSupervisor,
        approval_gate: ApprovalGate,
        state_dir: str | Path,
        approval_store: object | None = None,
        timeout_seconds: float = DEFAULT_SETUP_TIMEOUT_SECONDS,
        shell_allowed_domains: tuple[str, ...] = (),
    ) -> None:
        self._workspace_root = Path(workspace_root).resolve()
        # 只影响告诉 Agent 的话；准备流程本身和网络验证不变
        self._shell_allowed_domains = shell_allowed_domains
        self._supervisor = supervisor
        self._approval_gate = approval_gate
        self._approval_store = approval_store
        self._state_dir = Path(state_dir)
        self._timeout_seconds = timeout_seconds
        self._task: asyncio.Task[WorkspaceSetupRecord] | None = None
        self._current: WorkspaceSetupRecord | None = None
        self._last: WorkspaceSetupRecord | None = self._load_last()

    # ---------------------------------------------------------------- 状态

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def current(self) -> WorkspaceSetupRecord | None:
        return self._current if self.running else None

    @property
    def last(self) -> WorkspaceSetupRecord | None:
        return self._last

    def run_block_reason(self) -> str | None:
        """RunManager 启动新 Run 前调用：准备期间拒绝启动。"""
        if self.running:
            return "工作区正在准备环境，完成前不能启动新的运行"
        return None

    def context_message(self) -> Message | None:
        """最近一次准备成功时，告诉 Agent 依赖在哪、Shell 仍然断网。"""
        record = self._last
        if record is None or record.status is not SetupStatus.SUCCEEDED:
            return None
        lines = [
            "运行环境：用户已在联网的准备容器中执行工作区准备脚本 "
            f"{record.script_path}（sha256 {record.script_sha256[:12]}），"
            "并验证了 Shell 断网容器的网络隔离。",
        ]
        if self._shell_allowed_domains:
            lines.append(
                "Shell 只能经代理以 HTTPS 访问这些域名："
                + "、".join(self._shell_allowed_domains)
                + "，其他网络不可用；优先使用工作区中已准备好的依赖。"
            )
        else:
            lines.append(
                "Shell 命令仍然无法联网，不要尝试 pip、npm 等联网安装；需要的依赖已在工作区中准备。"
            )
        if os.path.lexists(self._workspace_root / _VENV_PYTHON):
            lines.append(
                "Python 依赖位于 .venv，运行 Python 请使用 .venv/bin/python，"
                "例如 .venv/bin/python -m pytest。"
            )
        lines.append(
            "白名单之外的依赖，请告诉用户把它加入准备脚本后重新准备环境。"
            if self._shell_allowed_domains
            else "缺少依赖时，请告诉用户把它加入准备脚本后重新准备环境，不要自行联网。"
        )
        return Message(role=MessageRole.SYSTEM, content="\n".join(lines))

    # ---------------------------------------------------------------- 入口

    def start(
        self,
        script_path: str = DEFAULT_SETUP_SCRIPT,
        *,
        verify_command: str | None = None,
        busy_reason: str | None = None,
    ) -> WorkspaceSetupRecord:
        """读取脚本并在后台开始准备；审批、执行和验证都在后台任务里完成。

        ``busy_reason`` 由调用方根据当前运行情况给出（有 Run 或长任务在跑时非空）。
        检查和占用之间没有 await，所以不会和新 Run 的启动交错。
        """
        if self.running:
            raise WorkspaceSetupBusy("已有准备任务在进行")
        if busy_reason:
            raise WorkspaceSetupBusy(busy_reason)
        script = self._read_script(script_path)
        verify = verify_command.strip() if verify_command else None
        record = WorkspaceSetupRecord(
            id=f"setup-{uuid4().hex[:12]}",
            status=SetupStatus.RUNNING,
            script_path=script_path,
            script_sha256=hashlib.sha256(script.encode("utf-8")).hexdigest(),
            verify_command=verify or None,
            started_at=_now(),
        )
        self._current = record
        self._task = asyncio.create_task(self._run(record, script))
        return record

    async def wait(self) -> WorkspaceSetupRecord | None:
        task = self._task
        if task is None:
            return self._last
        return await asyncio.shield(task)

    async def cancel(self) -> bool:
        """取消进行中的准备；等待审批时撤销审批，执行中确认容器清理后才返回。"""
        task = self._task
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    # ---------------------------------------------------------------- 流程

    async def _run(
        self, record: WorkspaceSetupRecord, script: str
    ) -> WorkspaceSetupRecord:
        try:
            if not await self._approve(record, script):
                record.status = SetupStatus.DENIED
                return record
            await self._execute(record, script)
            if record.timed_out or record.exit_code != 0:
                record.status = SetupStatus.FAILED
                record.error = record.error or (
                    "准备脚本超时" if record.timed_out else f"准备脚本退出码 {record.exit_code}"
                )
                return record
            record.network_check = await self._check_network()
            record.environment_check = await self._check_environment(
                record.verify_command
            )
            passed = (
                record.network_check.passed is True
                and record.environment_check.passed is not False
            )
            record.status = SetupStatus.SUCCEEDED if passed else SetupStatus.FAILED
            if not passed:
                record.error = "准备后验证未通过"
            return record
        except asyncio.CancelledError:
            record.status = SetupStatus.CANCELLED
            await self._cancel_pending_approval(record)
            return record
        except Exception as exc:
            logger.exception("workspace setup %s failed", record.id)
            record.status = SetupStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            return record
        finally:
            record.finished_at = _now()
            self._last = record
            self._save_last(record)

    async def _approve(self, record: WorkspaceSetupRecord, script: str) -> bool:
        # 直接调用基础审批入口：不经过权限规则，也不经过长任务的沙箱自动批准；
        # 用户在命令行选择"记住"也不会生成规则，每次准备单独审批
        request = ApprovalRequest(
            tool_call_id=record.id,
            tool_name=WORKSPACE_SETUP_TOOL_NAME,
            arguments={
                "script_path": record.script_path,
                "script": script,
                "sha256": record.script_sha256,
                "network": "enabled",
                "verify_command": record.verify_command,
            },
            description=(
                "在独立的联网容器中执行工作区准备脚本。脚本可以访问外网，并能读写工作区"
                "（.env、.git 等受保护路径除外）。执行结束后，Agent 仍只使用断网容器。"
            ),
            run_id=record.id,
        )
        response = await self._approval_gate.request_approval(request)
        return response.decision is ApprovalDecision.APPROVED

    async def _cancel_pending_approval(self, record: WorkspaceSetupRecord) -> None:
        store = self._approval_store
        cancel = getattr(store, "cancel_pending_for_run", None)
        if cancel is None:
            return
        try:
            await cancel(record.id)
        except Exception:
            logger.exception("failed to cancel pending setup approval %s", record.id)

    async def _execute(self, record: WorkspaceSetupRecord, script: str) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._state_dir / f"{record.id}.log"
        record.log_path = str(log_path)
        launch = self._supervisor.prepare_launch(
            command="/bin/sh",
            args=("-c", script),
            env=_docker_cli_environment(),
            cwd=None,
            config=SandboxConfig(
                filesystem=SandboxFilesystemMode.WORKSPACE_WRITE,
                network=SandboxNetworkMode.UNRESTRICTED,
            ),
        )
        if not launch.sandboxed:
            raise RuntimeError("准备脚本必须在沙箱容器中执行")
        head = bytearray()
        tail: deque[bytes] = deque()
        tail_size = 0
        process: asyncio.subprocess.Process | None = None
        try:
            process = await _spawn(launch, merge_stderr=True)
            assert process.stdout is not None
            with log_path.open("wb") as log:
                try:
                    async with asyncio.timeout(self._timeout_seconds):
                        while chunk := await process.stdout.read(8192):
                            # 边执行边落盘，内存里只保留开头和结尾
                            log.write(chunk)
                            log.flush()
                            if len(head) < _OUTPUT_EXCERPT_BYTES:
                                head.extend(chunk[: _OUTPUT_EXCERPT_BYTES - len(head)])
                            tail.append(chunk)
                            tail_size += len(chunk)
                            while tail_size - len(tail[0]) >= _OUTPUT_EXCERPT_BYTES:
                                tail_size -= len(tail.popleft())
                        await process.wait()
                except TimeoutError:
                    record.timed_out = True
                    _terminate_process(process)
                    await process.wait()
            record.exit_code = process.returncode
        except BaseException:
            if process is not None and process.returncode is None:
                _terminate_process(process)
                await process.wait()
            raise
        finally:
            # 取消或超时时也要清理容器，并确认确实没有残留才返回
            removed = await asyncio.shield(self._remove_container(launch))
            if not removed:
                # 清理没能确认就不能当成成功：联网容器可能还在运行
                record.error = "联网准备容器的清理未能确认"
                if not record.exit_code:
                    record.exit_code = -1
            record.output_head = head.decode("utf-8", errors="replace")
            record.output_tail = b"".join(tail)[-_OUTPUT_EXCERPT_BYTES:].decode(
                "utf-8", errors="replace"
            )

    async def _check_network(self) -> SetupCheck:
        """新起一个和 Agent 相同配置的断网容器：先看实际网络配置，再做直连探测。"""
        launch = self._supervisor.prepare_launch(
            command="python3",
            args=("-c", _NETWORK_PROBE),
            env=_docker_cli_environment(),
            cwd=None,
            config=SandboxConfig(
                filesystem=SandboxFilesystemMode.READ_ONLY,
                network=SandboxNetworkMode.DENIED,
            ),
        )
        process: asyncio.subprocess.Process | None = None
        communicate: asyncio.Task[tuple[bytes, bytes]] | None = None
        network_mode: str | None = None
        try:
            process = await _spawn(launch, merge_stderr=True)
            communicate = asyncio.create_task(process.communicate())
            async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
                while not communicate.done() and network_mode is None:
                    network_mode = await self._inspect_network(launch)
                    if network_mode is None:
                        await asyncio.sleep(0.2)
                stdout, _ = await communicate
        except TimeoutError:
            if process is not None:
                _terminate_process(process)
            if communicate is not None:
                communicate.cancel()
            return SetupCheck(passed=False, detail="网络验证超时")
        finally:
            await self._remove_container(launch)
        output = stdout.decode("utf-8", errors="replace").strip()
        if network_mode is None:
            return SetupCheck(passed=False, detail=f"未能读取验证容器的网络配置；输出：{output}")
        if network_mode != "none":
            return SetupCheck(passed=False, detail=f"验证容器的网络配置不是断网：{network_mode}")
        if "PROBE CONNECTED" in output:
            return SetupCheck(passed=False, detail="断网容器仍能连通外网")
        marker = "PROBE BLOCKED errno="
        if marker in output:
            errno_text = output.split(marker, 1)[1].split()[0]
            if errno_text in _UNREACHABLE_ERRNOS:
                return SetupCheck(
                    passed=True,
                    detail=f"网络模式 none，直连外网返回 errno {errno_text}",
                )
            return SetupCheck(
                passed=False,
                detail=f"直连失败但不是网络不可达（errno {errno_text}），不能确认隔离",
            )
        # 探测程序本身没有正常运行，例如 Python 启动失败：不能当成隔离通过
        return SetupCheck(passed=False, detail=f"探测程序未正常运行；输出：{output or '(空)'}")

    async def _check_environment(self, verify_command: str | None) -> SetupCheck:
        """在断网容器里确认准备出来的环境能用；没有可验证的东西就跳过。"""
        if verify_command:
            command, args = "/bin/sh", ("-c", verify_command)
            label = verify_command
        elif os.path.lexists(self._workspace_root / _VENV_PYTHON):
            command = _VENV_PYTHON.as_posix()
            args = ("-c", "import sys; print(sys.prefix)")
            label = ".venv/bin/python"
        else:
            return SetupCheck(passed=None, detail="未提供验证命令，工作区也没有 .venv，跳过环境验证")
        launch = self._supervisor.prepare_launch(
            command=command,
            args=args,
            env=_docker_cli_environment(),
            cwd=None,
            config=SandboxConfig(
                filesystem=SandboxFilesystemMode.WORKSPACE_WRITE,
                network=SandboxNetworkMode.DENIED,
            ),
        )
        process: asyncio.subprocess.Process | None = None
        try:
            process = await _spawn(launch, merge_stderr=True)
            async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
                stdout, _ = await process.communicate()
        except TimeoutError:
            if process is not None:
                _terminate_process(process)
            return SetupCheck(passed=False, detail=f"环境验证超时：{label}")
        finally:
            await self._remove_container(launch)
        output = stdout.decode("utf-8", errors="replace").strip()[-1_000:]
        if process.returncode == 0:
            return SetupCheck(passed=True, detail=f"{label} 在断网容器中运行成功")
        return SetupCheck(
            passed=False,
            detail=f"{label} 在断网容器中失败（退出码 {process.returncode}）：{output}",
        )

    # ---------------------------------------------------------------- 容器

    async def _inspect_network(self, launch: SandboxLaunchSpec) -> str | None:
        """读取正在运行的验证容器的网络模式和所连网络；容器尚未出现时返回 None。"""
        name = _container_name(launch)
        if name is None:
            return None
        process = await asyncio.create_subprocess_exec(
            launch.command,
            "container",
            "inspect",
            "--format",
            "{{.HostConfig.NetworkMode}}|"
            "{{range $k, $v := .NetworkSettings.Networks}}{{$k}},{{end}}",
            name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=launch.env,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            return None
        mode, _, networks = stdout.decode().strip().partition("|")
        attached = {item for item in networks.split(",") if item}
        # 网络模式是 none，且没有接入任何其他网络
        if mode == "none" and attached <= {"none"}:
            return "none"
        return f"{mode} (networks: {', '.join(sorted(attached)) or '-'})"

    async def _remove_container(self, launch: SandboxLaunchSpec) -> bool:
        await _cleanup_sandbox_launch(launch)
        name = _container_name(launch)
        if name is None:
            return True
        process = await asyncio.create_subprocess_exec(
            launch.command,
            "container",
            "inspect",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=launch.env,
        )
        await process.wait()
        # inspect 失败表示容器已经不存在
        return process.returncode != 0

    # ---------------------------------------------------------------- 存储

    def _read_script(self, script_path: str) -> str:
        try:
            path = resolve_workspace_path(self._workspace_root, script_path)
        except ValueError as exc:
            raise WorkspaceSetupError(str(exc)) from exc
        if not path.is_file():
            raise WorkspaceSetupError(f"准备脚本不存在：{script_path}")
        data = path.read_bytes()
        if len(data) > MAX_SCRIPT_BYTES:
            raise WorkspaceSetupError(f"准备脚本超过 {MAX_SCRIPT_BYTES // 1024} KB")
        try:
            script = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceSetupError("准备脚本必须是 UTF-8 文本") from exc
        if not script.strip():
            raise WorkspaceSetupError("准备脚本为空")
        # Windows 编辑器可能写入 CRLF，容器里的 sh 会把 \r 当成命令的一部分
        return script.replace("\r\n", "\n")

    @property
    def _state_file(self) -> Path:
        return self._state_dir / "last.json"

    def _load_last(self) -> WorkspaceSetupRecord | None:
        try:
            text = self._state_file.read_text("utf-8")
            record = WorkspaceSetupRecord.model_validate_json(text)
        except (OSError, ValueError):
            return None
        if record.status is SetupStatus.RUNNING:
            # 上次进程在准备途中退出：结果未知，按失败处理（残留容器由启动时的孤儿清理负责）
            record.status = SetupStatus.FAILED
            record.error = "上次准备过程中进程退出，结果未知"
        return record

    def _save_last(self, record: WorkspaceSetupRecord) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            temporary = self._state_file.with_suffix(".tmp")
            temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, self._state_file)
        except OSError:
            logger.exception("failed to save workspace setup state")


async def _spawn(
    launch: SandboxLaunchSpec,
    *,
    merge_stderr: bool,
) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        launch.command,
        *launch.args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT if merge_stderr else asyncio.subprocess.PIPE,
        cwd=launch.cwd,
        env=launch.env,
        start_new_session=True,
    )


def _container_name(launch: SandboxLaunchSpec) -> str | None:
    args = launch.args
    for index, value in enumerate(args[:-1]):
        if value == "--name":
            return args[index + 1]
    return None


def _docker_cli_environment() -> dict[str, str]:
    """只传给宿主机上的 docker 命令，不会进入容器。"""
    return _shell_environment()


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "DEFAULT_SETUP_SCRIPT",
    "SetupCheck",
    "SetupStatus",
    "WorkspaceSetupBusy",
    "WorkspaceSetupError",
    "WorkspaceSetupRecord",
    "WorkspaceSetupService",
]
