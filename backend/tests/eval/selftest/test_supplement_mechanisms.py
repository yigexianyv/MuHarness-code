"""Controlled B1/B3/B7 mechanisms; remaining mechanisms reuse existing regressions."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.models.types import ToolCall, ToolDefinition
from app.runtime.agent.result import AgentStopReason
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.checkpoint import CheckpointStatus, SQLiteCheckpointStore
from app.runtime.checkpoint.context import render_checkpoint_context
from app.safety.sandbox import SandboxSupervisor
from app.safety.sandbox.backends import DockerSandboxBackend
from app.tools.approval import AutoApproveGate
from app.tools.base import BaseTool
from app.tools.builtin.read_file import ReadFileTool
from app.tools.builtin.shell import ShellCommandTool
from app.tools.registry import ToolRegistry
from tests.offline.agent.test_agent_runtime import fake_registry, model_response


async def test_missing_docker_refuses_shell_without_starting_host_process(tmp_path, monkeypatch):
    backend = DockerSandboxBackend(tmp_path)
    backend.docker_command = None
    spawn = AsyncMock()
    monkeypatch.setattr("app.tools.builtin.shell.asyncio.create_subprocess_exec", spawn)
    tools = ToolRegistry()
    tools.register(ShellCommandTool(tmp_path, sandbox_supervisor=SandboxSupervisor(tmp_path, native_backend=backend)))
    registry, _ = fake_registry([
        model_response(tool_calls=(ToolCall(id="shell", name="run_shell_command", arguments={
            "command": "python -c \"from pathlib import Path; Path('host-marker').write_text('unsafe')\"",
        }),)),
        model_response(content="Docker unavailable; not executed"),
    ])
    result = await AgentRuntime(registry, tools, provider="fake", approval_gate=AutoApproveGate()).run("shell")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].result.success is False
    assert "Docker CLI" in result.tool_calls[0].result.error
    spawn.assert_not_awaited()
    assert not (tmp_path / "host-marker").exists()


async def test_repeated_call_ignores_key_order_and_never_executes_third(tmp_path):
    (tmp_path / "a.txt").write_text("value", encoding="utf-8")

    class CountingRead(ReadFileTool):
        count = 0

        async def execute(self, arguments):
            self.count += 1
            return await super().execute(arguments)

    tool = CountingRead(tmp_path)
    tools = ToolRegistry()
    tools.register(tool)
    arguments = {"path": "a.txt", "start_line": 1, "max_lines": 1}
    registry, _ = fake_registry([
        model_response(tool_calls=(ToolCall(id=str(i), name="read_file",
                                           arguments=dict(reversed(list(arguments.items()))) if i == 2 else arguments),))
        for i in range(1, 4)
    ])
    result = await AgentRuntime(registry, tools, provider="fake").run("read")
    assert result.stop_reason is AgentStopReason.REPEATED_TOOL_CALL
    assert tool.count == 2


class MarkTool(BaseTool):
    def __init__(self, name, path, started, *, fail=False, block=False):
        self._definition = ToolDefinition(name=name, description="controlled file effect", parameters={"type": "object"})
        self.path, self.started, self.fail, self.block = path, started, fail, block

    @property
    def definition(self):
        return self._definition

    async def execute(self, arguments):
        if self.fail:
            raise RuntimeError("controlled failure")
        self.path.write_bytes(b"executed")
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        return "written"


@pytest.mark.parametrize("boundary", ["error", "before_start", "after_effect"])
async def test_two_tool_checkpoint_preserves_actual_boundary(tmp_path, boundary):
    store = SQLiteCheckpointStore(tmp_path / "test.db")
    await store.initialize()
    reached = asyncio.Event()
    second_started = asyncio.Event()
    tools = ToolRegistry()
    tools.register(MarkTool("first", tmp_path / "first.txt", asyncio.Event()))
    tools.register(MarkTool("second", tmp_path / "second.txt", second_started,
                            fail=boundary == "error", block=boundary == "after_effect"))
    registry, _ = fake_registry([
        model_response(tool_calls=(ToolCall(id="a", name="first", arguments={}),
                                   ToolCall(id="b", name="second", arguments={}))),
        model_response(content="finished"),
    ])
    if boundary == "before_start":
        original_complete = store.complete_tool

        async def complete_and_wait(run_id, result):
            await original_complete(run_id, result)
            if result.tool_call_id == "a":
                reached.set()
                await asyncio.Event().wait()

        store.complete_tool = complete_and_wait
    runtime = AgentRuntime(registry, tools, provider="fake", checkpoint_store=store)
    running = asyncio.create_task(runtime.run("two operations", conversation_id="test"))
    try:
        if boundary == "error":
            result = await asyncio.wait_for(running, 10)
            checkpoint = await store.get(result.run_id)
            assert checkpoint.pending_tool_calls == ()
            assert [item.success for item in checkpoint.completed_tool_results] == [True, False]
            assert "controlled failure" in checkpoint.completed_tool_results[-1].error
        else:
            await asyncio.wait_for((reached if boundary == "before_start" else second_started).wait(), 10)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            restarted = SQLiteCheckpointStore(store.database_path)
            await restarted.initialize()
            await restarted.recover_running()
            checkpoint = (await restarted.list(conversation_id="test"))[0]
            assert checkpoint.status is CheckpointStatus.INTERRUPTED
            assert [item.id for item in checkpoint.pending_tool_calls] == ["b"]
            assert [item.tool_call_id for item in checkpoint.completed_tool_results] == ["a"]
            assert second_started.is_set() is (boundary == "after_effect")
            explanation = render_checkpoint_context(checkpoint).content
            payload = json.loads(explanation.split("<interrupted_run>")[1].split("</interrupted_run>")[0])
            assert payload["pending_tool_calls"][0]["recovery_semantics"] == (
                "execution outcome is uncertain; verify before retry"
            )
        assert (tmp_path / "first.txt").read_bytes() == b"executed"
        assert (tmp_path / "second.txt").exists() is (boundary == "after_effect")
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
