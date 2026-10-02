from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from app.safety.sandbox import (
    DockerSandboxBackend,
    SandboxBackend,
    SandboxConfig,
    SandboxFilesystemMode,
    SandboxLaunchSpec,
    SandboxNetworkMode,
    SandboxPolicy,
    SandboxPolicyError,
    SandboxSupervisor,
    SandboxUnavailableError,
    UnsupportedSandboxBackend,
)
from app.tools.builtin.shell import ShellCommandTool


class RecordingBackend(SandboxBackend):
    # 函数说明：RecordingBackend.__init__
    # 用途：初始化 RecordingBackend；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.policy`。
    def __init__(self) -> None:
        self.policy: SandboxPolicy | None = None

    # 函数说明：RecordingBackend.prepare
    # 用途：准备RecordingBackend，供回归测试与测试辅助使用。
    # 参数：
    #   command：待执行的 Shell 命令，类型 `str | Path`。
    #   args：`args`输入或配置值，类型 `tuple[str, ...]`。
    #   env：环境变量映射，类型 `dict[str, str]`。
    #   policy：权限决策策略，类型 `SandboxPolicy`。
    # 返回：类型 `SandboxLaunchSpec`；返回 `SandboxLaunchSpec(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SandboxLaunchSpec`。
    # 副作用与资源：
    #   更新对象字段：`self.policy`。
    def prepare(
        self,
        *,
        command: str | Path,
        args: tuple[str, ...],
        env: dict[str, str],
        policy: SandboxPolicy,
    ) -> SandboxLaunchSpec:
        self.policy = policy
        return SandboxLaunchSpec(
            command=str(command),
            args=args,
            cwd=str(policy.working_directory),
            env=env,
            backend="recording",
            sandboxed=True,
        )


# 函数说明：test_supervisor_compiles_workspace_policy_and_protects_metadata
# 用途：回归验证回归测试与测试辅助中的
# `supervisor_compiles_workspace_policy_and_protects_metadata` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `(workspace / '.git').mkdir` → `(workspace / '.env').write_text` → `extra.mkdir` →
# `RecordingBackend` → `SandboxSupervisor`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`launch.sandboxed is True`。
#   验证条件：`launch.backend == 'recording'`。
#   验证条件：`backend.policy is not None`。
#   验证条件：`workspace in backend.policy.readable_roots`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / '.git').mkdir`、
# `(workspace / '.env').write_text`、`extra.mkdir`。
def test_supervisor_compiles_workspace_policy_and_protects_metadata(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / ".env").write_text("SECRET=value", encoding="utf-8")
    extra = workspace / "input"
    extra.mkdir()
    backend = RecordingBackend()
    supervisor = SandboxSupervisor(workspace, native_backend=backend)

    launch = supervisor.prepare_launch(
        command=sys.executable,
        args=("-V",),
        env={"PATH": os.environ.get("PATH", "")},
        cwd=None,
        config=SandboxConfig(
            filesystem=SandboxFilesystemMode.WORKSPACE_WRITE,
            network=SandboxNetworkMode.DENIED,
            readable_roots=("input",),
        ),
    )

    assert launch.sandboxed is True
    assert launch.backend == "recording"
    assert backend.policy is not None
    assert workspace in backend.policy.readable_roots
    assert workspace in backend.policy.writable_roots
    assert workspace / ".git" in backend.policy.denied_write_paths
    assert workspace / ".env" in backend.policy.denied_read_paths
    assert backend.policy.network is SandboxNetworkMode.DENIED


# 函数说明：test_supervisor_rejects_relative_path_traversal
# 用途：回归验证回归测试与测试辅助中的 `supervisor_rejects_relative_path_traversal` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` → `outside.mkdir` →
# `SandboxSupervisor` → `RecordingBackend` → `pytest.raises` →
# `supervisor.prepare_launch`；另有 1 个调用点。
# 分支与异常：
#   预期异常：`pytest.raises(SandboxPolicyError, match='不能越过 workspace')`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`outside.mkdir`。
def test_supervisor_rejects_relative_path_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    supervisor = SandboxSupervisor(workspace, native_backend=RecordingBackend())

    with pytest.raises(SandboxPolicyError, match="不能越过 workspace"):
        supervisor.prepare_launch(
            command=sys.executable,
            args=(),
            env={"PATH": os.environ.get("PATH", "")},
            cwd=None,
            config=SandboxConfig(readable_roots=("../outside",)),
        )


# 函数说明：test_unsupported_platform_fails_closed
# 用途：回归验证回归测试与测试辅助中的 `unsupported_platform_fails_closed` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SandboxSupervisor` →
# `UnsupportedSandboxBackend` → `pytest.raises` → `supervisor.prepare_launch` →
# `SandboxConfig`。
# 分支与异常：
#   预期异常：`pytest.raises(SandboxUnavailableError, match='拒绝降级执行')`。
def test_unsupported_platform_fails_closed(tmp_path: Path) -> None:
    supervisor = SandboxSupervisor(
        tmp_path,
        native_backend=UnsupportedSandboxBackend("unsupported"),
    )

    with pytest.raises(SandboxUnavailableError, match="拒绝降级执行"):
        supervisor.prepare_launch(
            command=sys.executable,
            args=(),
            env={"PATH": os.environ.get("PATH", "")},
            cwd=None,
            config=SandboxConfig(),
        )


# 函数说明：test_docker_backend_compiles_hardened_ephemeral_container
# 用途：回归验证回归测试与测试辅助中的
# `docker_backend_compiles_hardened_ephemeral_container` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `(workspace / '.git').mkdir` → `(workspace / '.env').write_text` →
# `DockerSandboxBackend` → `SandboxSupervisor` → `supervisor.prepare_launch`；另有 3 个
# 调用点。
# 分支与异常：
#   验证条件：`launch.command == sys.executable`。
#   验证条件：`launch.backend == 'docker'`。
#   验证条件：`launch.sandboxed is True`。
#   验证条件：`'--network' in launch.args`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / '.git').mkdir`、
# `(workspace / '.env').write_text`。
def test_docker_backend_compiles_hardened_ephemeral_container(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / ".env").write_text("SECRET=value", encoding="utf-8")
    backend = DockerSandboxBackend(
        workspace,
        docker_command=sys.executable,
        image="test-muharness-sandbox:latest",
    )
    supervisor = SandboxSupervisor(workspace, native_backend=backend)

    launch = supervisor.prepare_launch(
        command="/bin/sh",
        args=("-c", "printf safe"),
        env={"PATH": os.environ.get("PATH", "")},
        cwd=str(workspace),
        config=SandboxConfig(network=SandboxNetworkMode.DENIED),
    )

    assert launch.command == sys.executable
    assert launch.backend == "docker"
    assert launch.sandboxed is True
    assert "--network" in launch.args
    assert launch.args[launch.args.index("--network") + 1] == "none"
    assert "--read-only" in launch.args
    assert "--cap-drop" in launch.args
    assert "--security-opt" in launch.args
    assert "--pids-limit" in launch.args
    assert "--memory" in launch.args
    assert "--cpus" in launch.args
    assert "--user" in launch.args
    assert "test-muharness-sandbox:latest" in launch.args
    assert launch.args[-3:] == ("/bin/sh", "-c", "printf safe")
    mount_values = tuple(
        launch.args[index + 1]
        for index, value in enumerate(launch.args)
        if value == "--mount"
    )
    assert any("target=/workspace" in value for value in mount_values)
    assert any("target=/workspace/.git" in value for value in mount_values)
    assert any("target=/workspace/.env" in value for value in mount_values)
    assert launch.cleanup_command[:3] == (
        sys.executable,
        "rm",
        "--force",
    )
    assert len(launch.cleanup_paths) == 1
    for path in launch.cleanup_paths:
        shutil.rmtree(path)


# 函数说明：test_docker_backend_rejects_domain_allowlist
# 用途：回归验证回归测试与测试辅助中的 `docker_backend_rejects_domain_allowlist` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SandboxSupervisor` →
# `DockerSandboxBackend` → `pytest.raises` → `supervisor.prepare_launch` →
# `SandboxConfig`。
# 分支与异常：
#   预期异常：`pytest.raises(SandboxUnavailableError, match='域名白名单')`。
def test_docker_backend_rejects_domain_allowlist(tmp_path: Path) -> None:
    supervisor = SandboxSupervisor(
        tmp_path,
        native_backend=DockerSandboxBackend(
            tmp_path,
            docker_command=sys.executable,
        ),
    )

    with pytest.raises(SandboxUnavailableError, match="域名白名单"):
        supervisor.prepare_launch(
            command="/bin/sh",
            args=("-c", "true"),
            env={"PATH": os.environ.get("PATH", "")},
            cwd=None,
            config=SandboxConfig(allowed_domains=("example.com",)),
        )


class CleanupBackend(SandboxBackend):
    resolve_executable_on_host = False

    # 函数说明：CleanupBackend.__init__
    # 用途：初始化 CleanupBackend；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   marker：`marker`输入或配置值，类型 `Path`。
    #   cleanup_path：路径输入或配置值，类型 `Path`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.marker`、`self.cleanup_path`。
    def __init__(self, marker: Path, cleanup_path: Path) -> None:
        self.marker = marker
        self.cleanup_path = cleanup_path

    # 函数说明：CleanupBackend.prepare
    # 用途：准备CleanupBackend，供回归测试与测试辅助使用。
    # 参数：
    #   command：待执行的 Shell 命令，类型 `str | Path`。
    #   args：`args`输入或配置值，类型 `tuple[str, ...]`。
    #   env：环境变量映射，类型 `dict[str, str]`。
    #   policy：权限决策策略，类型 `SandboxPolicy`。
    # 返回：类型 `SandboxLaunchSpec`；返回 `SandboxLaunchSpec(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SandboxLaunchSpec`。
    def prepare(
        self,
        *,
        command: str | Path,
        args: tuple[str, ...],
        env: dict[str, str],
        policy: SandboxPolicy,
    ) -> SandboxLaunchSpec:
        del command, args
        cleanup_script = (
            "from pathlib import Path; "
            f"Path({str(self.marker)!r}).write_text('cleaned', encoding='utf-8')"
        )
        return SandboxLaunchSpec(
            command=sys.executable,
            args=("-c", "import time; time.sleep(30)"),
            cwd=str(policy.working_directory),
            env=env,
            backend="cleanup-test",
            sandboxed=True,
            cleanup_command=(sys.executable, "-c", cleanup_script),
            cleanup_paths=(str(self.cleanup_path),),
        )


# 函数说明：test_docker_backend_rejects_persistent_mcp_process
# 用途：回归验证回归测试与测试辅助中的 `docker_backend_rejects_persistent_mcp_process`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SandboxSupervisor` →
# `DockerSandboxBackend` → `pytest.raises` → `supervisor.prepare_launch` →
# `SandboxConfig`。
# 分支与异常：
#   预期异常：`pytest.raises(SandboxPolicyError, match='不支持长驻 MCP')`。
def test_docker_backend_rejects_persistent_mcp_process(tmp_path: Path) -> None:
    supervisor = SandboxSupervisor(
        tmp_path,
        native_backend=DockerSandboxBackend(
            tmp_path,
            docker_command=sys.executable,
        ),
    )

    with pytest.raises(SandboxPolicyError, match="不支持长驻 MCP"):
        supervisor.prepare_launch(
            command="third-party-mcp",
            args=(),
            env={"PATH": os.environ.get("PATH", "")},
            cwd=None,
            config=SandboxConfig(),
            persistent=True,
        )


# 函数说明：test_shell_timeout_runs_backend_cleanup
# 用途：回归验证回归测试与测试辅助中的 `shell_timeout_runs_backend_cleanup` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`cleanup_path.mkdir` →
# `CleanupBackend` → `ShellCommandTool` → `SandboxSupervisor` → `tool.execute` →
# `marker.read_text`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result['timed_out'] is True`。
#   验证条件：`marker.read_text(encoding='utf-8') == 'cleaned'`。
#   验证条件：`not cleanup_path.exists()`。
# 副作用与资源：
#   文件或资源访问：`cleanup_path.mkdir`、`marker.read_text`。
@pytest.mark.asyncio
async def test_shell_timeout_runs_backend_cleanup(tmp_path: Path) -> None:
    marker = tmp_path / "cleanup-complete"
    cleanup_path = tmp_path / "temporary-mask"
    cleanup_path.mkdir()
    backend = CleanupBackend(marker, cleanup_path)
    tool = ShellCommandTool(
        tmp_path,
        sandbox_supervisor=SandboxSupervisor(
            tmp_path,
            native_backend=backend,
        ),
    )

    result = await tool.execute(
        {"command": "ignored", "timeout_seconds": 0.05}
    )

    assert result["timed_out"] is True
    assert marker.read_text(encoding="utf-8") == "cleaned"
    assert not cleanup_path.exists()


# 函数说明：test_real_docker_shell_isolated_workspace
# 用途：回归验证回归测试与测试辅助中的 `real_docker_shell_isolated_workspace` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `(workspace / '.env').write_text` → `ShellCommandTool` → `SandboxSupervisor` →
# `tool.execute` → `(workspace / 'result.txt').read_text`。
# 分支与异常：
#   验证条件：`result['exit_code'] == 0`。
#   验证条件：`result['stdout'] == 'not-present'`。
#   验证条件：`(workspace / 'result.txt').read_text(encoding='utf-8') == 'safe'`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / '.env').write_text`、
# `(workspace / 'result.txt').read_text`。
@pytest.mark.skipif(
    os.environ.get(
        "MUHARNESS_RUN_DOCKER_E2E", os.environ.get("VESTA_RUN_DOCKER_E2E")
    ) != "1",
    reason="需要显式启用真实 Docker 集成测试",
)
@pytest.mark.asyncio
async def test_real_docker_shell_isolated_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("HOST_SECRET=hidden", encoding="utf-8")
    tool = ShellCommandTool(
        workspace,
        sandbox_supervisor=SandboxSupervisor(workspace),
    )

    result = await tool.execute(
        {
            "command": (
                "printf safe > result.txt; "
                "printf '%s' \"${OPENAI_API_KEY:-not-present}\""
            )
        }
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == "not-present"
    assert (workspace / "result.txt").read_text(encoding="utf-8") == "safe"
