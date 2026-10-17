from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from .common import (
    VARIANTS,
    atomic_write,
    digest,
    load_experiment,
    now,
    read_json,
    write_json,
)
from .container import TaskContainer, docker
from .runner import completed_result, experiment_lock, live_attempts, worker_status


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # access denied is treated as alive
        try:
            code = wintypes.DWORD()
            return (
                not kernel.GetExitCodeProcess(handle, ctypes.byref(code))
                or code.value == 259
            )
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def same_mount(source: str, workspace: Path) -> bool:
    if os.name != "nt":
        return Path(source).resolve() == workspace.resolve()
    expected = str(workspace.resolve()).replace("\\", "/").lower()
    drive, suffix = expected.split(":", 1)
    accepted = {
        expected,
        f"/run/desktop/mnt/host/{drive}{suffix}",
        f"/host_mnt/{drive}{suffix}",
    }
    return source.replace("\\", "/").lower() in accepted


def recover(root: Path) -> list[dict]:
    """Seal interrupted controller attempts without ever making another model call."""
    manifest = load_experiment(root, verify_source=False)
    recovered = []
    with experiment_lock(root):
        for variant in VARIANTS:
            for task in read_json(root / "tasks.json"):
                instance = task["instance_id"]
                if completed_result(root, variant, instance, manifest["fingerprint"]):
                    continue
                directory = root / "runs" / variant / instance
                attempts = live_attempts(directory) if directory.exists() else []
                if not attempts:
                    continue
                if len(attempts) != 1:
                    raise ValueError(f"Ambiguous interrupted attempts: {directory}")
                output = attempts[0]
                process_file = output / "worker-process.json"
                if process_file.exists() and process_alive(
                    read_json(process_file)["pid"]
                ):
                    raise RuntimeError(
                        f"Worker still alive; leave it running: {output}"
                    )
                result = read_json(output / "status.json")
                if result["fingerprint"] != manifest["fingerprint"]:
                    raise ValueError(
                        "Interrupted attempt belongs to another experiment"
                    )
                worker_file = output / "worker-result.json"
                worker = (
                    read_json(worker_file)
                    if worker_file.exists()
                    else {"stop_reason": "controller_interrupted"}
                )
                result.update(worker)
                result["status"] = (
                    worker_status(worker) if worker_file.exists() else "infra_error"
                )
                container_file = output / "container.json"
                if container_file.exists():
                    container_info = read_json(container_file)
                    name = container_info["name"]
                    if not re.fullmatch(r"mh-swe-[0-9a-f]{20}", name):
                        raise ValueError(
                            "Refusing to recover a container without an evaluation name"
                        )
                    item = json.loads(docker("inspect", name).stdout)[0]
                    mounts = [
                        m for m in item["Mounts"] if m["Destination"] == "/testbed"
                    ]
                    if (
                        item["Config"].get("Labels", {}).get("mh.swebench") != "1"
                        or len(mounts) != 1
                        or not same_mount(mounts[0]["Source"], output / "workspace")
                    ):
                        raise ValueError(
                            "Container does not belong to this interrupted attempt"
                        )
                    container = TaskContainer(name, output / "workspace")
                    try:
                        container.quiesce()
                        raw, patch = container.export_patch(
                            container_info.get("patch_base") or task["base_commit"]
                        )
                        atomic_write(output / "raw.patch", raw)
                        atomic_write(output / "model.patch", patch)
                        result.update(
                            patch_sha256=digest(patch), patch_bytes=len(patch)
                        )
                    finally:
                        container.close()
                usage_file = output / "usage.json"
                result["usage"] = read_json(usage_file) if usage_file.exists() else {}
                result["duration_seconds"] = max(
                    0,
                    (
                        datetime.now(UTC) - datetime.fromisoformat(result["started_at"])
                    ).total_seconds(),
                )
                result["finished_at"] = now()
                result["recovered_controller"] = True
                write_json(output / "status.json", result)
                write_json(directory / "result.json", result)
                recovered.append(result)
    return recovered
