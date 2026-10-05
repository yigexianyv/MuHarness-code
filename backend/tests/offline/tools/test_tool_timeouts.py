"""执行器外层时限不得抢在工具自己的超时之前取消调用。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.models.types import ToolCall, ToolDefinition
from app.tools.base import BaseTool
from app.tools.builtin.shell import (
    MAX_SHELL_TIMEOUT_SECONDS,
    SHELL_CLEANUP_GRACE_SECONDS,
    ShellCommandTool,
)
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class _SleepTool(BaseTool):
    definition = ToolDefinition(name="sleep")

    def __init__(self, declared: float | None) -> None:
        self._declared = declared

    async def execute(self, arguments: dict[str, Any]) -> str:
        await asyncio.sleep(0.2)
        return "done"

    def execution_timeout(self, arguments: dict[str, Any]) -> float | None:
        return self._declared


async def _run(tool: BaseTool) -> Any:
    registry = ToolRegistry()
    registry.register(tool)
    return await ToolExecutor(registry, timeout_seconds=0.05).execute(
        ToolCall(id="t1", name="sleep", arguments={})
    )


@pytest.mark.asyncio
async def test_declared_timeout_extends_outer_limit() -> None:
    result = await _run(_SleepTool(declared=1.0))
    assert result.success is True
    assert result.output == "done"


@pytest.mark.asyncio
async def test_tools_without_declaration_keep_default_limit() -> None:
    result = await _run(_SleepTool(declared=None))
    assert result.success is False
    assert result.error == "Tool timed out after 0.05 seconds."


@pytest.mark.asyncio
async def test_declared_timeout_never_shortens_default() -> None:
    # 工具自报的时限比统一时限短时，仍按统一时限
    result = await _run(_SleepTool(declared=0.01))
    assert result.success is False
    assert result.error == "Tool timed out after 0.05 seconds."


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"command": "true"}, 30.0 + SHELL_CLEANUP_GRACE_SECONDS),
        (
            {"command": "true", "timeout_seconds": 110},
            110 + SHELL_CLEANUP_GRACE_SECONDS,
        ),
        (
            {"command": "true", "timeout_seconds": 999},
            MAX_SHELL_TIMEOUT_SECONDS + SHELL_CLEANUP_GRACE_SECONDS,
        ),
        ({"command": "true", "timeout_seconds": -1}, None),
        ({"command": "true", "timeout_seconds": "60"}, None),
    ],
)
def test_shell_declares_its_own_timeout_plus_cleanup(
    tmp_path, arguments, expected
) -> None:
    tool = ShellCommandTool(tmp_path)
    assert tool.execution_timeout(arguments) == expected
