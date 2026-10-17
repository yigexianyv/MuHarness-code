from __future__ import annotations

import asyncio

from app.models.types import ToolPermission
from app.tools.builtin.shell import ShellCommandTool

from .container import TaskContainer
from .meter import Meter


class ContainerShellTool(ShellCommandTool):
    """Retain MuHarness's tool interface, execute inside the persistent task image."""

    def __init__(
        self,
        container: TaskContainer,
        meter: Meter,
        read_only: TaskContainer | None = None,
    ):
        # Do not initialize the generic one-command sandbox supervisor.
        self.container = container
        self.meter = meter
        # MEA Manager/Auditor: same image, workspace mounted read-only.
        self.read_only = read_only
        self._allowed_domains = ()

    @property
    def definition(self):
        original = super().definition
        return original.model_copy(
            update={
                "permission": ToolPermission.ALLOWED,
                "description": (
                    "在 SWE-bench 持久容器中用 bash 执行命令；仓库位于 /testbed。"
                    "网络禁用，当前评测自动批准。working_directory 可用仓库相对路径。"
                    f"默认及最长命令超时均为 {self.container.timeout:g} 秒。"
                    "检查 exit_code 和 timed_out。"
                ),
            }
        )

    def execution_timeout(self, arguments):
        return self.container.timeout + 20

    async def execute_with_context(self, arguments, context):
        from app.tools.role_boundary import READ_ONLY_MODES

        if self.read_only is not None and context.mode in READ_ONLY_MODES:
            return await self.execute(arguments, target=self.read_only)
        return await self.execute(arguments)

    async def execute(self, arguments, target: TaskContainer | None = None):
        target = target or self.container
        output = await asyncio.to_thread(
            target.execute,
            arguments["command"],
            cwd=arguments.get("working_directory") or "/testbed",
            timeout=arguments.get("timeout_seconds"),
        )
        command = " ".join(str(arguments["command"]).split())
        self.meter.progress(
            f"  {'[只读] ' if target is not self.container else ''}$ {command[:100]}"
            f" → exit {output['returncode']} ({output['duration_seconds']:.0f}s)"
        )
        self.meter.event(
            "shell",
            {
                "arguments": arguments,
                "result": output,
                "read_only": target is not self.container,
            },
        )
        return {
            "exit_code": output["returncode"],
            "stdout": output["output"],
            "stderr": "",
            "timed_out": bool(output["exception_info"]),
            "duration_ms": output["duration_seconds"] * 1000,
        }
