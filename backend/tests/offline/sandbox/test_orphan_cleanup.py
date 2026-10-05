"""后端被强杀后遗留的沙箱容器：启动时只清理本实例的容器。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from app.safety.sandbox import (
    SandboxConfig,
    SandboxNetworkMode,
    SandboxSupervisor,
    backends,
)
from app.safety.sandbox.backends import (
    INSTANCE_LABEL,
    SANDBOX_LABEL,
    DockerSandboxBackend,
)


class _FakeDocker:
    def __init__(self, listed: str | None = "", removed: str | None = "") -> None:
        self.calls: list[tuple[str, ...]] = []
        self._listed = listed
        self._removed = removed

    async def __call__(self, docker_command: str, *args: str, timeout: float = 30):
        self.calls.append(args)
        return self._listed if args[0] == "ps" else self._removed


def _backend(tmp_path: Path, instance_id: str | None) -> DockerSandboxBackend:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return DockerSandboxBackend(
        workspace,
        docker_command=sys.executable,
        image="test-muharness-sandbox:latest",
        instance_id=instance_id,
    )


def test_containers_carry_instance_label(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "abc123")
    launch = SandboxSupervisor(
        backend.workspace_root, native_backend=backend
    ).prepare_launch(
        command="/bin/sh",
        args=("-c", "true"),
        env={"PATH": os.environ.get("PATH", "")},
        cwd=str(backend.workspace_root),
        config=SandboxConfig(network=SandboxNetworkMode.DENIED),
    )
    labels = [
        launch.args[index + 1]
        for index, value in enumerate(launch.args)
        if value == "--label"
    ]
    assert f"{SANDBOX_LABEL}=true" in labels
    assert f"{INSTANCE_LABEL}=abc123" in labels


@pytest.mark.asyncio
async def test_remove_orphans_only_targets_this_instance(tmp_path, monkeypatch) -> None:
    fake = _FakeDocker(listed="c1\nc2\n")
    monkeypatch.setattr(backends, "_run_docker", fake)
    backend = _backend(tmp_path, "abc123")

    assert (
        await SandboxSupervisor(
            backend.workspace_root, native_backend=backend
        ).remove_orphans()
        == 2
    )

    ps, rm = fake.calls
    assert ps[0] == "ps"
    assert f"label={SANDBOX_LABEL}=true" in ps
    assert f"label={INSTANCE_LABEL}=abc123" in ps
    assert rm == ("rm", "--force", "--volumes", "c1", "c2")


@pytest.mark.asyncio
async def test_remove_orphans_skips_when_nothing_left(tmp_path, monkeypatch) -> None:
    fake = _FakeDocker(listed="")
    monkeypatch.setattr(backends, "_run_docker", fake)
    assert await _backend(tmp_path, "abc123").remove_orphans() == 0
    assert [call[0] for call in fake.calls] == ["ps"]


@pytest.mark.asyncio
async def test_remove_orphans_without_instance_id_never_touches_docker(
    tmp_path, monkeypatch
) -> None:
    fake = _FakeDocker(listed="c1\n")
    monkeypatch.setattr(backends, "_run_docker", fake)
    # 没有实例标识时无法区分其他后端的容器，宁可不清理
    assert await _backend(tmp_path, None).remove_orphans() == 0
    assert fake.calls == []


@pytest.mark.asyncio
async def test_remove_orphans_tolerates_docker_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backends, "_run_docker", _FakeDocker(listed=None))
    assert await _backend(tmp_path, "abc123").remove_orphans() == 0
    monkeypatch.setattr(backends, "_run_docker", _FakeDocker(listed="c1", removed=None))
    assert await _backend(tmp_path, "abc123").remove_orphans() == 0


@pytest.mark.asyncio
async def test_run_docker_returns_none_when_cli_missing(tmp_path) -> None:
    missing = str(tmp_path / "no-such-docker")
    assert await backends._run_docker(missing, "ps") is None


def test_instance_id_is_stable_per_database(tmp_path) -> None:
    from app.application import _sandbox_instance_id

    first = _sandbox_instance_id(tmp_path / "a.db")
    assert first == _sandbox_instance_id(tmp_path / "a.db")
    assert first != _sandbox_instance_id(tmp_path / "b.db")
