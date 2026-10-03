"""在第二个只读 Docker 工作区运行独立验收；不执行宿主机上的模型产物。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.safety.sandbox import (
    SandboxConfig,
    SandboxFilesystemMode,
    SandboxNetworkMode,
    SandboxSupervisor,
)
from app.tools.builtin.shell import _cleanup_sandbox_launch, _shell_environment

from .v1_cases import Verification


async def _docker_query(*args: str) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
    except TimeoutError:
        process.kill()
        await process.communicate()
        return -1, "Docker 检查超时"
    return process.returncode or 0, (stdout + stderr).decode(
        "utf-8", errors="replace"
    ).strip()


async def docker_preflight() -> dict[str, Any]:
    image = os.environ.get(
        "MUHARNESS_SANDBOX_IMAGE",
        os.environ.get("VESTA_SANDBOX_IMAGE", "muharness-sandbox:latest"),
    )
    result: dict[str, Any] = {"available": False, "image": image}
    if not shutil.which("docker"):
        return {**result, "reason": "没有 Docker CLI"}
    if os.environ.get("MUHARNESS_SANDBOX_BACKEND", "auto") not in {"auto", "docker"}:
        return {**result, "reason": "当前配置禁用了 Docker 沙箱"}
    code, output = await _docker_query("info", "--format", "{{.ServerVersion}}")
    if code:
        return {**result, "reason": "Docker Engine 不可用", "diagnostic": output}
    result["server_version"] = output
    code, output = await _docker_query("image", "inspect", image, "--format", "{{.Id}}")
    if code:
        return {**result, "reason": "缺少 MuHarness 沙箱镜像", "diagnostic": output}
    return {**result, "available": True, "image_id": output}


async def verify(
    workspace: Path, spec: Verification, *, timeout: float = 45
) -> dict[str, Any]:
    """仅复制允许的最终代码；可信测试在模型运行结束后才加入验收目录。"""
    launch = None
    process = None
    with tempfile.TemporaryDirectory(
        prefix="muharness-independent-grade-"
    ) as directory:
        root = Path(directory)
        copied: dict[str, str] = {}
        for relative in spec.files:
            source = workspace / relative
            if source.is_symlink() or not source.resolve().is_relative_to(
                workspace.resolve()
            ):
                return {"status": "FAIL", "reason": f"验收产物路径越界：{relative}"}
            if not source.is_file():
                return {"status": "FAIL", "reason": f"缺少产物：{relative}"}
            data = source.read_bytes()
            if len(data) > 256_000:
                return {"status": "FAIL", "reason": f"验收产物过大：{relative}"}
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            copied[relative] = hashlib.sha256(data).hexdigest()
        (root / "_trusted_grade.py").write_text(spec.script, encoding="utf-8")
        try:
            launch = SandboxSupervisor(root).prepare_launch(
                command="python",
                args=(
                    "-B",
                    "-I",
                    "-c",
                    "import sys; sys.path.insert(0,'/workspace'); "
                    "sys.argv=['_trusted_grade.py']; "
                    "exec(compile(open('/workspace/_trusted_grade.py').read(),'/workspace/_trusted_grade.py','exec'))",
                ),
                env=_shell_environment(),
                cwd=str(root),
                config=SandboxConfig(
                    filesystem=SandboxFilesystemMode.READ_ONLY,
                    network=SandboxNetworkMode.DENIED,
                ),
            )
            if not launch.sandboxed or launch.backend != "docker":
                return {"status": "BLOCKED", "reason": "独立验收要求 Docker 隔离"}
            process = await asyncio.create_subprocess_exec(
                launch.command,
                *launch.args,
                cwd=launch.cwd,
                env=launch.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            code = process.returncode
            # Docker 自身的启动错误不属于 Agent 代码失败。
            status = (
                "BLOCKED"
                if code in {125, 126, 127}
                else ("PASS" if code == 0 else "FAIL")
            )
            log = stderr.decode("utf-8", errors="replace")
            if status == "PASS" and not (
                re.search(r"Ran [1-9]\d* tests?", log) and re.search(r"\bOK\b", log)
            ):
                status = "FAIL"
            return {
                "status": status,
                "exit_code": code,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "code_hashes": copied,
                "grader_sha256": hashlib.sha256(spec.script.encode()).hexdigest(),
                "sandbox": {
                    "backend": "docker",
                    "filesystem": "read_only",
                    "network": "denied",
                },
            }
        except TimeoutError:
            if process is not None:
                process.kill()
                await process.communicate()
            return {"status": "FAIL", "reason": "独立验收超时", "code_hashes": copied}
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.communicate()
            raise
        except Exception as exc:
            return {
                "status": "BLOCKED",
                "reason": f"独立验收环境错误：{type(exc).__name__}: {exc}",
            }
        finally:
            if launch is not None:
                await _cleanup_sandbox_launch(launch)
