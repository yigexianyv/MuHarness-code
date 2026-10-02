
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.checkpoint import SQLiteCheckpointStore
from app.runtime.run import RunManager, RunStatus, SQLiteRunStore
from app.tools.approval import AutoApproveGate
from app.tools.builtin.shell import ShellCommandTool
from app.tools.registry import ToolRegistry


class _FakeModelAdapter(ModelAdapter):
    # 函数说明：_FakeModelAdapter.__init__
    # 用途：初始化 _FakeModelAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`
    # 。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.requests`。
    def __init__(
        self,
        config: ProviderConfig,
        responses: Sequence[ModelResponse | Exception],
    ) -> None:
        super().__init__(config)
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    # 函数说明：_FakeModelAdapter.complete
    # 用途：完成_FakeModelAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.responses.pop`。
    # 分支与异常：
    #   当 `isinstance(response, Exception)` 时，抛出 `response`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    # 函数说明：_FakeModelAdapter.close
    # 用途：关闭_FakeModelAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：model_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str | None`；默认 `None`。
#   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`；默认 `()`。
#   usage：模型调用用量统计，类型 `ModelUsage | None`；默认 `None`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def model_response(
    *,
    content: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
    usage: ModelUsage | None = None,
) -> ModelResponse:
    return ModelResponse(
        id="fake-response",
        provider="fake",
        model="fake-model",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        ),
        usage=usage or ModelUsage(),
    )


# 函数说明：fake_registry
# 用途：在回归测试与测试辅助中处理 `fake_registry`，通过 `registry.register` 完成首个内
# 部处理步骤。
# 参数：
#   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`。
# 返回：类型 `tuple[ModelAdapterRegistry, _FakeModelAdapter]`；返回
# `(registry, adapter)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `_FakeModelAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`。
def fake_registry(
    responses: Sequence[ModelResponse | Exception],
) -> tuple[ModelAdapterRegistry, _FakeModelAdapter]:
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = _FakeModelAdapter(config, responses)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)
    return registry, adapter


# 函数说明：_sandbox_container_ids
# 用途：在回归测试与测试辅助中处理 `_sandbox_container_ids`，通过
# `asyncio.create_subprocess_exec` 完成首个内部处理步骤。
# 返回：类型 `set[str]`；返回 `set(stdout_bytes.decode().split())`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.create_subprocess_exec` →
# `process.communicate` → `stdout_bytes.decode`。
async def _sandbox_container_ids() -> set[str]:
    process = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "-q",
        "--filter",
        "label=com.muharness.sandbox=true",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, _ = await process.communicate()
    return set(stdout_bytes.decode().split())




# 函数说明：test_shell_tool_cancel_terminates_process_group
# 用途：回归验证回归测试与测试辅助中的 `shell_tool_cancel_terminates_process_group` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   docker_engine：可用 Docker 引擎的测试前置夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_sandbox_container_ids` →
# `ShellCommandTool` → `asyncio.create_task` → `tool.execute` → `marker.exists` →
# `asyncio.sleep`；另有 2 个调用点。
# 分支与异常：
#   当 `marker.exists()` 时，结束当前循环。
#   当 `await _sandbox_container_ids() <= containers_before` 时，结束当前循环。
#   验证条件：`marker.exists()`。
#   验证条件：`await _sandbox_container_ids() <= containers_before`。
#   预期异常：`pytest.raises(asyncio.CancelledError)`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
@pytest.mark.asyncio
async def test_shell_tool_cancel_terminates_process_group(tmp_path: Path, docker_engine) -> None:
    marker = tmp_path / "shell-started"
    containers_before = await _sandbox_container_ids()
    tool = ShellCommandTool(workspace_root=tmp_path)
    task = asyncio.create_task(
        tool.execute(
            {
                "command": "touch shell-started; sleep 300",
                "timeout_seconds": 300,
            }
        )
    )

    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.05)
    assert marker.exists(), "shell 子进程未启动"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(40):
        if await _sandbox_container_ids() <= containers_before:
            break
        await asyncio.sleep(0.1)
    assert await _sandbox_container_ids() <= containers_before




# 函数说明：test_runmanager_cancel_shell_run_leaves_no_process
# 用途：回归验证回归测试与测试辅助中的 `runmanager_cancel_shell_run_leaves_no_process`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   docker_engine：可用 Docker 引擎的测试前置夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_sandbox_container_ids` → `ToolCall`
# → `fake_registry` → `model_response` → `ToolRegistry` → `tools.register`；另有 12 个调
# 用点。
# 分支与异常：
#   当 `marker.exists()` 时，结束当前循环。
#   当 `await _sandbox_container_ids() <= containers_before` 时，结束当前循环。
#   验证条件：`marker.exists()`。
#   验证条件：`cancelled.status is RunStatus.CANCELLED`。
#   验证条件：`checkpoint is not None`。
#   验证条件：`checkpoint.status.value == 'interrupted'`。
@pytest.mark.asyncio
async def test_runmanager_cancel_shell_run_leaves_no_process(
    tmp_path: Path,
    docker_engine,
) -> None:
    marker = tmp_path / "run-shell-started"
    containers_before = await _sandbox_container_ids()
    command = "touch run-shell-started; sleep 300"
    call = ToolCall(
        id="shell-1",
        name="run_shell_command",
        arguments={"command": command, "timeout_seconds": 300},
    )
    registry, _ = fake_registry(
        [
            model_response(tool_calls=(call,)),
            model_response(content="完成"),
        ]
    )
    tools = ToolRegistry()
    tools.register(ShellCommandTool(workspace_root=tmp_path))

    database = tmp_path / "muharness.db"
    run_store = SQLiteRunStore(database)
    checkpoint_store = SQLiteCheckpointStore(database)
    await run_store.initialize()
    await checkpoint_store.initialize()
    runtime = AgentRuntime(
        registry,
        tools,
        provider="fake",
        checkpoint_store=checkpoint_store,
        approval_gate=AutoApproveGate(),
    )
    manager = RunManager(run_store, checkpoint_store, runtime)

    run_id, _ = await manager.start("跑一个长命令", conversation_id="conv-1")
    for _ in range(200):
        if marker.exists():
            break
        await asyncio.sleep(0.05)
    assert marker.exists(), "shell 子进程未启动"

    cancelled = await manager.cancel(run_id)
    assert cancelled.status is RunStatus.CANCELLED
    checkpoint = await checkpoint_store.get(run_id)
    assert checkpoint is not None
    assert checkpoint.status.value == "interrupted"

    for _ in range(40):
        if await _sandbox_container_ids() <= containers_before:
            break
        await asyncio.sleep(0.1)
    assert await _sandbox_container_ids() <= containers_before
