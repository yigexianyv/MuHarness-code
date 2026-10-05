from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.application import Application
from app.domain.conversation import SQLiteConversationStore
from app.domain.memory import MemoryMaintenanceConfig, MemoryReflectionConfig
from app.runtime.instance_lock import DatabaseInstanceLock
from app.safety.sandbox import SandboxSupervisor
from tests.offline.agent.test_agent_runtime import fake_registry


def test_database_ownership_is_exclusive_and_reusable(tmp_path):
    owner = DatabaseInstanceLock(tmp_path / "test.db")
    contender = DatabaseInstanceLock(tmp_path / "test.db")
    other = DatabaseInstanceLock(tmp_path / "other.db")
    owner.acquire()
    try:
        with pytest.raises(RuntimeError, match="其他后端占用"):
            contender.acquire()
        other.acquire()
        other.release()
        with pytest.raises(RuntimeError, match="其他后端占用"):
            contender.acquire()
    finally:
        owner.release()
    contender.acquire()
    contender.release()
    assert owner.path.exists()  # A stale filename does not mean the lock is held.


def test_process_kill_releases_database_ownership(tmp_path):
    database = tmp_path / "test.db"
    backend = Path(__file__).resolve().parents[3]
    code = (
        f"import sys; sys.path.insert(0, {str(backend)!r}); "
        "from pathlib import Path; "
        "from app.runtime.instance_lock import DatabaseInstanceLock; "
        f"lock = DatabaseInstanceLock(Path({str(database)!r})); "
        "lock.acquire(); print('locked', flush=True); sys.stdin.read()"
    )
    # Windows venv launchers spawn a child; kill the actual lock owner instead.
    executable = getattr(sys, "_base_executable", sys.executable)
    process = subprocess.Popen(
        [executable, "-c", code], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    contender = DatabaseInstanceLock(database)
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="其他后端占用"):
            contender.acquire()
        process.kill()
        process.wait(timeout=10)
        contender.acquire()
    finally:
        contender.release()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def _application(tmp_path):
    registry, _ = fake_registry([])
    return Application(
        registry=registry, database=tmp_path / "test.db",
        workspace_root=tmp_path / "workspace", tasks_dir=tmp_path / "tasks",
        memory_dir=tmp_path / "memory", mcp_config=tmp_path / "mcp.json",
        skills_user_dir=tmp_path / "skills-user",
        skills_project_dir=tmp_path / "skills-project",
        memory_reflection_config=MemoryReflectionConfig(enabled=False),
        memory_maintenance_config=MemoryMaintenanceConfig(enabled=False),
    )


@pytest.mark.asyncio
async def test_duplicate_application_stops_before_cleanup(tmp_path, monkeypatch):
    cleanup_calls = []

    async def remove_orphans(supervisor):
        cleanup_calls.append(supervisor.workspace_root)
        return 0

    monkeypatch.setattr(SandboxSupervisor, "remove_orphans", remove_orphans)
    owner = _application(tmp_path)
    contender = _application(tmp_path)
    try:
        await asyncio.gather(owner.start(), owner.start())
        with pytest.raises(RuntimeError, match="其他后端占用"):
            await contender.start()
        assert len(cleanup_calls) == 1
        assert contender.conversation_store is None
        await owner.close()
        await contender.start()
        assert len(cleanup_calls) == 2
    finally:
        await contender.close()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_failed_or_cancelled_start_releases_ownership(
    tmp_path, monkeypatch, error_type,
):
    application = _application(tmp_path)

    async def fail_initialize(store):
        raise error_type("startup failed")

    monkeypatch.setattr(SQLiteConversationStore, "initialize", fail_initialize)
    with pytest.raises(error_type, match="startup failed"):
        await application.start()
    contender = DatabaseInstanceLock(application.database)
    contender.acquire()
    contender.release()
    await application.close()


@pytest.mark.asyncio
async def test_close_keeps_ownership_until_running_work_exits(tmp_path, monkeypatch):
    owner = _application(tmp_path)
    contender = DatabaseInstanceLock(owner.database)
    cleanup_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    waited = []

    async def cancel(run_id):
        cleanup_started.set()
        await cleanup_finished.wait()

    async def wait(run_id):
        waited.append(run_id)

    await owner.start()
    monkeypatch.setattr(owner, "run_manager", SimpleNamespace(
        active_run_ids=("active-run",), cancel=cancel, wait=wait,
    ))
    closing = asyncio.create_task(owner.close())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="其他后端占用"):
            contender.acquire()
        assert not closing.done()
    finally:
        cleanup_finished.set()
        await closing
    assert waited == ["active-run"]
    contender.acquire()
    contender.release()
