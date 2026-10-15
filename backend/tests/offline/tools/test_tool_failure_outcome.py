from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.models.types import ToolCall, ToolDefinition, ToolPermission, ToolResult
from app.runtime.agent.tool_round_executor import ToolRoundExecutor
from app.runtime.mea.recovery import _completed_line
from app.tools import BaseTool, ToolExecutor, ToolRegistry


class SideEffectThenFailureTool(BaseTool):
    """模拟操作已落地，但返回结果之前超时或抛错。"""

    def __init__(self, path: Path, failure: str) -> None:
        self.path = path
        self.failure = failure
        self.calls = 0

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="side_effect")

    async def execute(self, arguments: dict[str, Any]) -> Any:
        self.calls += 1
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write("executed\n")
        if self.failure == "timeout":
            await asyncio.sleep(60)
        if self.failure == "value_error":
            raise ValueError("error after writing")
        raise RuntimeError("connection lost after writing")


@pytest.mark.parametrize("failure", ["timeout", "runtime_error", "value_error"])
async def test_completed_side_effect_is_unknown_and_not_automatically_repeated(
    tmp_path: Path, failure: str,
) -> None:
    """副作用落地后的错误必须报告未知，执行器不能暗中再执行一次。"""
    path = tmp_path / "effects.txt"
    tool = SideEffectThenFailureTool(path, failure)
    registry = ToolRegistry()
    registry.register(tool)
    result = await ToolExecutor(registry, timeout_seconds=0.01).execute(
        ToolCall(id="call-1", name="side_effect"),
    )
    assert not result.success
    assert result.execution_outcome == "unknown"
    assert "Verify its state first" in (result.retry_advice or "")
    assert tool.calls == 1
    assert path.read_text(encoding="utf-8") == "executed\n"
    model_message = ToolRoundExecutor._result_message(result)
    assert json.loads(model_message.content or "{}")["execution_outcome"] == "unknown"
    assert "执行结果未知" in _completed_line(result, None)


@pytest.mark.parametrize("arguments", ["not JSON", "[]"])
async def test_invalid_input_never_enters_tool_body(
    tmp_path: Path, arguments: str,
) -> None:
    """JSON 解码和对象类型校验失败时，可以确定工具主体未执行。"""
    path = tmp_path / "effects.txt"
    tool = SideEffectThenFailureTool(path, "runtime_error")
    registry = ToolRegistry()
    registry.register(tool)
    result = await ToolExecutor(registry).execute(
        ToolCall(id="call-1", name="side_effect", arguments=arguments),
    )
    assert result.execution_outcome == "not_started"
    assert tool.calls == 0
    assert not path.exists()


async def test_missing_or_denied_tool_is_not_started(tmp_path: Path) -> None:
    """不存在或未获授权的调用不能被描述为已执行。"""
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    missing = await executor.execute(ToolCall(id="missing", name="missing"))
    assert missing.execution_outcome == "not_started"

    class ApprovalTool(SideEffectThenFailureTool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="side_effect", permission=ToolPermission.HUMAN_APPROVAL,
            )

    tool = ApprovalTool(tmp_path / "effects.txt", "runtime_error")
    registry.register(tool)
    denied = await executor.execute(ToolCall(id="denied", name="side_effect"))
    assert denied.execution_outcome == "not_started"
    assert tool.calls == 0


def test_old_tool_results_remain_readable() -> None:
    """旧检查点不含新字段，仍按未知元数据而非未执行处理。"""
    result = ToolResult.model_validate({
        "tool_call_id": "old", "tool_name": "write_file", "success": False,
        "error": "timeout", "duration_ms": 1,
    })
    assert result.execution_outcome is None
    assert result.retry_advice is None
