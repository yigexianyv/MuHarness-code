
from __future__ import annotations

import asyncio
from collections.abc import Sequence

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
    ToolDefinition,
)
from app.runtime.agent.events import AgentEventType
from app.runtime.agent.result import AgentStopReason
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.checkpoint import (
    CHECKPOINT_CONTEXT_MESSAGE_NAME,
    CheckpointPhase,
    CheckpointStatus,
    SQLiteCheckpointStore,
)
from app.runtime.run import RunManager, RunStatus, SQLiteRunStore
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry


class FakeModelAdapter(ModelAdapter):
    # 函数说明：FakeModelAdapter.__init__
    # 用途：初始化 FakeModelAdapter；参数及实际保存的实例字段见下方说明。
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

    # 函数说明：FakeModelAdapter.complete
    # 用途：完成FakeModelAdapter，供回归测试与测试辅助使用。
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

    # 函数说明：FakeModelAdapter.close
    # 用途：关闭FakeModelAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class BlockingModelAdapter(ModelAdapter):

    # 函数说明：BlockingModelAdapter.__init__
    # 用途：初始化 BlockingModelAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super` →
    # `asyncio.Event`。
    # 副作用与资源：
    #   更新对象字段：`self.started`、`self.cancelled`、`self.requests`。
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.started = asyncio.Event()
        self.cancelled = False
        self.requests: list[ModelRequest] = []

    # 函数说明：BlockingModelAdapter.complete
    # 用途：完成BlockingModelAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.started.set` →
    # `asyncio.Event().wait` → `asyncio.Event`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，重新抛出原异常。
    # 副作用与资源：
    #   更新对象字段：`self.cancelled`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("阻塞模型不应正常完成")

    # 函数说明：BlockingModelAdapter.close
    # 用途：关闭BlockingModelAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class CountingTool(BaseTool):
    definition = ToolDefinition(
        name="count",
        description="Count executions",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    )

    # 函数说明：CountingTool.__init__
    # 用途：初始化 CountingTool；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    def __init__(self) -> None:
        self.executions = 0

    # 函数说明：CountingTool.execute
    # 用途：执行CountingTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, object]`；读取键
    # `value`。
    # 返回：类型 `str`；返回 `str(arguments['value'])`。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    async def execute(self, arguments: dict[str, object]) -> str:
        self.executions += 1
        return str(arguments["value"])


class BlockingTool(BaseTool):
    definition = ToolDefinition(
        name="blocking_tool",
        description="Wait until cancelled",
        parameters={"type": "object", "properties": {}},
    )

    # 函数说明：BlockingTool.__init__
    # 用途：初始化 BlockingTool；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Event`。
    # 副作用与资源：
    #   更新对象字段：`self.started`、`self.cancelled`。
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    # 函数说明：BlockingTool.execute
    # 用途：执行BlockingTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, object]`。
    # 返回：类型 `str`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.started.set` →
    # `asyncio.Event().wait` → `asyncio.Event`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，重新抛出原异常。
    # 副作用与资源：
    #   更新对象字段：`self.cancelled`。
    async def execute(self, arguments: dict[str, object]) -> str:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("阻塞工具不应正常完成")


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
# 返回：类型 `tuple[ModelAdapterRegistry, FakeModelAdapter]`；返回 `(registry, adapter)`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `FakeModelAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`。
def fake_registry(
    responses: Sequence[ModelResponse | Exception],
) -> tuple[ModelAdapterRegistry, FakeModelAdapter]:
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = FakeModelAdapter(config, responses)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)
    return registry, adapter




# 函数说明：manager_factory
# 用途：处理回归测试与测试辅助中的 `manager_factory` 数据；结果及边界条件见下方说明。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：返回 `build_manager`。
@pytest.fixture
async def manager_factory(tmp_path):

    # 函数说明：manager_factory.build_manager
    # 用途：构建管理者，供回归测试与测试辅助使用。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ModelAdapterRegistry`。
    #   tools：可用工具定义或工具实例集合，类型 `ToolRegistry | None`；默认 `None`。
    #   checkpoint_store：运行检查点存储，类型 `SQLiteCheckpointStore | None`；默认
    # `None`。
    #   run_store：运行持久化存储依赖，类型 `SQLiteRunStore | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str`；默认 `'fake'`。
    #   run_finalizers：运行输入或配置值；默认 `()`。
    # 返回：类型 `tuple[RunManager, SQLiteRunStore, SQLiteCheckpointStore]`；返回
    # `(manager, run_store, checkpoint_store)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteRunStore` →
    # `SQLiteCheckpointStore` → `run_store.initialize` → `checkpoint_store.initialize` →
    #  `AgentRuntime` → `ToolRegistry`；另有 1 个调用点。
    # 闭包依赖：从外层读取 `tmp_path`。
    async def build_manager(
        registry: ModelAdapterRegistry,
        tools: ToolRegistry | None = None,
        *,
        checkpoint_store: SQLiteCheckpointStore | None = None,
        run_store: SQLiteRunStore | None = None,
        provider: str = "fake",
        run_finalizers=(),
    ) -> tuple[RunManager, SQLiteRunStore, SQLiteCheckpointStore]:
        database = tmp_path / "muharness.db"
        run_store = run_store or SQLiteRunStore(database)
        checkpoint_store = checkpoint_store or SQLiteCheckpointStore(database)
        await run_store.initialize()
        await checkpoint_store.initialize()
        runtime = AgentRuntime(
            registry,
            tools or ToolRegistry(),
            provider=provider,
            checkpoint_store=checkpoint_store,
        )
        manager = RunManager(
            run_store, checkpoint_store, runtime, run_finalizers=run_finalizers
        )
        return manager, run_store, checkpoint_store

    return build_manager




# 函数说明：test_start_running_to_completed
# 用途：回归验证回归测试与测试辅助中的 `start_running_to_completed` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `build_manager` → `manager.start` → `manager.get_run` → `manager.wait`；另有 1 个调用
# 点。
# 分支与异常：
#   验证条件：`running is not None and running.status is RunStatus.RUNNING`。
#   验证条件：`run.status is RunStatus.COMPLETED`。
#   验证条件：`run.stop_reason == AgentStopReason.FINAL_ANSWER.value`。
#   验证条件：`run.completed_at is not None`。
async def test_start_running_to_completed(manager_factory) -> None:
    build_manager = manager_factory
    registry, _ = fake_registry([model_response(content="完成")])
    manager, run_store, _ = await build_manager(registry)

    run_id, task = await manager.start("你好", conversation_id="conv-1")
    running = await manager.get_run(run_id)
    assert running is not None and running.status is RunStatus.RUNNING

    run = await manager.wait(run_id)
    assert run.status is RunStatus.COMPLETED
    assert run.stop_reason == AgentStopReason.FINAL_ANSWER.value
    assert run.completed_at is not None
    assert run.started_at is not None
    assert run.conversation_id == "conv-1"

    result = manager.result(run_id)
    assert result is not None and result.ok is True




# 函数说明：test_runtime_exception_marks_failed
# 用途：回归验证回归测试与测试辅助中的 `runtime_exception_marks_failed` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `build_manager` →
# `manager.start` → `manager.wait`。
# 分支与异常：
#   验证条件：`run.status is RunStatus.FAILED`。
#   验证条件：`run.error is not None and 'offline' in run.error`。
#   验证条件：`run.stop_reason == AgentStopReason.MODEL_ERROR.value`。
async def test_runtime_exception_marks_failed(manager_factory) -> None:
    build_manager = manager_factory
    registry, _ = fake_registry([RuntimeError("offline")])
    manager, _, _ = await build_manager(registry)

    run_id, _ = await manager.start("你好", conversation_id="conv-1")
    run = await manager.wait(run_id)

    assert run.status is RunStatus.FAILED
    assert run.error is not None and "offline" in run.error
    assert run.stop_reason == AgentStopReason.MODEL_ERROR.value


# 函数说明：test_run_finalizer_runs_for_completed_and_failed
# 用途：回归验证回归测试与测试辅助中的 `run_finalizer_runs_for_completed_and_failed` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `manager_factory` → `manager.start` → `manager.wait` → `failed_manager.start`；另有 1
# 个调用点。
# 分支与异常：
#   验证条件：`finalized == [completed_id, failed_id]`。
async def test_run_finalizer_runs_for_completed_and_failed(manager_factory) -> None:
    finalized: list[str] = []
    registry, _ = fake_registry([model_response(content="完成")])
    manager, _, _ = await manager_factory(registry, run_finalizers=(finalized.append,))
    completed_id, _ = await manager.start("完成")
    await manager.wait(completed_id)

    failed_registry, _ = fake_registry([RuntimeError("offline")])
    failed_manager, _, _ = await manager_factory(
        failed_registry, run_finalizers=(finalized.append,)
    )
    failed_id, _ = await failed_manager.start("失败")
    await failed_manager.wait(failed_id)
    assert finalized == [completed_id, failed_id]


# 函数说明：test_run_finalizer_failure_does_not_change_terminal_state
# 用途：回归验证回归测试与测试辅助中的
# `run_finalizer_failure_does_not_change_terminal_state` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `manager_factory` → `manager.start` → `manager.wait`。
# 分支与异常：
#   验证条件：`(await manager.wait(run_id)).status is RunStatus.COMPLETED`。
async def test_run_finalizer_failure_does_not_change_terminal_state(
    manager_factory,
) -> None:
    # 函数说明：test_run_finalizer_failure_does_not_change_terminal_state.broken
    # 用途：处理回归测试与测试辅助中的 `broken` 数据；结果及边界条件见下方说明。
    # 参数：
    #   _run_id：运行标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def broken(_run_id: str) -> None:
        raise RuntimeError("cleanup failed")

    registry, _ = fake_registry([model_response(content="完成")])
    manager, _, _ = await manager_factory(registry, run_finalizers=(broken,))
    run_id, _ = await manager.start("完成")
    assert (await manager.wait(run_id)).status is RunStatus.COMPLETED




# 函数说明：test_cancel_running_run_during_model_request
# 用途：回归验证回归测试与测试辅助中的 `cancel_running_run_during_model_request` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `BlockingModelAdapter` → `ModelAdapterRegistry` → `ModelSettings` →
# `registry.register`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`cancelled_run.status is RunStatus.CANCELLED`。
#   验证条件：`cancelled_run.completed_at is not None`。
#   验证条件：`cancelled_run.error is not None`。
#   验证条件：`len(adapter.requests) == 1`。
async def test_cancel_running_run_during_model_request(
    manager_factory,
    tmp_path,
) -> None:
    build_manager = manager_factory
    config = ProviderConfig(
        provider="blocking",
        model="blocking-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = BlockingModelAdapter(config)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("blocking", lambda _: adapter, config=config)

    finalized: list[str] = []
    manager, _, checkpoint_store = await build_manager(
        registry, provider="blocking", run_finalizers=(finalized.append,)
    )

    run_id, task = await manager.start("你好", conversation_id="conv-1")
    await adapter.started.wait()

    cancelled_run = await manager.cancel(run_id)

    assert cancelled_run.status is RunStatus.CANCELLED
    assert cancelled_run.completed_at is not None
    assert cancelled_run.error is not None
    assert len(adapter.requests) == 1
    assert adapter.cancelled is True

    checkpoint = await checkpoint_store.get(run_id)
    assert checkpoint is not None
    assert checkpoint.status is CheckpointStatus.INTERRUPTED
    assert checkpoint.phase is CheckpointPhase.MODEL_REQUEST
    assert checkpoint.step == 1
    assert finalized == [run_id]


# 函数说明：test_cancel_running_run_during_tool_execution
# 用途：回归验证回归测试与测试辅助中的 `cancel_running_run_during_tool_execution` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `fake_registry` →
# `model_response` → `BlockingTool` → `ToolRegistry` → `tools.register`；另有 4 个调用点
# 。
# 分支与异常：
#   验证条件：`cancelled_run.status is RunStatus.CANCELLED`。
#   验证条件：`tool.cancelled is True`。
#   验证条件：`checkpoint is not None`。
#   验证条件：`checkpoint.status is CheckpointStatus.INTERRUPTED`。
async def test_cancel_running_run_during_tool_execution(manager_factory) -> None:
    build_manager = manager_factory
    call = ToolCall(id="uncertain-tool", name="blocking_tool", arguments={})
    registry, _ = fake_registry([model_response(tool_calls=(call,))])
    tool = BlockingTool()
    tools = ToolRegistry()
    tools.register(tool)

    manager, _, checkpoint_store = await build_manager(registry, tools)

    run_id, _ = await manager.start("执行阻塞工具", conversation_id="conv-1")
    await tool.started.wait()

    cancelled_run = await manager.cancel(run_id)

    assert cancelled_run.status is RunStatus.CANCELLED
    assert tool.cancelled is True
    checkpoint = await checkpoint_store.get(run_id)
    assert checkpoint is not None
    assert checkpoint.status is CheckpointStatus.INTERRUPTED
    assert checkpoint.phase is CheckpointPhase.TOOL_EXECUTION
    assert checkpoint.pending_tool_calls == (call,)
    assert checkpoint.completed_tool_results == ()


# 函数说明：test_interrupt_running_run_preserves_checkpoint
# 用途：回归验证回归测试与测试辅助中的 `interrupt_running_run_preserves_checkpoint` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `BlockingModelAdapter` → `ModelAdapterRegistry` → `ModelSettings` →
# `registry.register`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`interrupted.status is RunStatus.INTERRUPTED`。
#   验证条件：`interrupted.completed_at is not None`。
#   验证条件：`adapter.cancelled is True`。
#   验证条件：`checkpoint is not None`。
async def test_interrupt_running_run_preserves_checkpoint(manager_factory) -> None:
    build_manager = manager_factory
    config = ProviderConfig(
        provider="blocking",
        model="blocking-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = BlockingModelAdapter(config)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("blocking", lambda _: adapter, config=config)

    manager, _, checkpoint_store = await build_manager(
        registry, provider="blocking"
    )

    run_id, _ = await manager.start("你好", conversation_id="conv-1")
    await adapter.started.wait()

    interrupted = await manager.interrupt(run_id)

    assert interrupted.status is RunStatus.INTERRUPTED
    assert interrupted.completed_at is not None
    assert adapter.cancelled is True
    checkpoint = await checkpoint_store.get(run_id)
    assert checkpoint is not None
    assert checkpoint.status is CheckpointStatus.INTERRUPTED
    again = await manager.interrupt(run_id)
    assert again.status is RunStatus.INTERRUPTED


# 函数说明：test_terminal_run_cancel_is_idempotent
# 用途：回归验证回归测试与测试辅助中的 `terminal_run_cancel_is_idempotent` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `build_manager` → `manager.start` → `manager.wait` → `manager.cancel`。
# 分支与异常：
#   验证条件：`run.status is RunStatus.COMPLETED`。
#   验证条件：`again.status is RunStatus.COMPLETED`。
async def test_terminal_run_cancel_is_idempotent(manager_factory) -> None:
    build_manager = manager_factory
    registry, _ = fake_registry([model_response(content="完成")])
    manager, _, _ = await build_manager(registry)

    run_id, _ = await manager.start("你好", conversation_id="conv-1")
    run = await manager.wait(run_id)
    assert run.status is RunStatus.COMPLETED

    again = await manager.cancel(run_id)
    assert again.status is RunStatus.COMPLETED




# 函数说明：_make_stale_running_run
# 用途：构造运行，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   with_checkpoint：检查点输入或配置值，类型 `bool`。
#   checkpoint_status：检查点状态输入或配置值，类型 `CheckpointStatus`；默认
# `CheckpointStatus.RUNNING`。
#   user_message：当前用户消息，类型 `str`；默认 `'写入 output.md'`。
# 返回：类型 `tuple[str, SQLiteRunStore, SQLiteCheckpointStore, str]`；返回
# `(run.id, run_store, checkpoint_store, str(database))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteRunStore` →
# `SQLiteCheckpointStore` → `run_store.initialize` → `checkpoint_store.initialize` →
# `run_store.create` → `run_store.mark_started`；另有 4 个调用点。
async def _make_stale_running_run(
    tmp_path,
    *,
    with_checkpoint: bool,
    checkpoint_status: CheckpointStatus = CheckpointStatus.RUNNING,
    user_message: str = "写入 output.md",
) -> tuple[str, SQLiteRunStore, SQLiteCheckpointStore, str]:

    database = tmp_path / "muharness.db"
    run_store = SQLiteRunStore(database)
    checkpoint_store = SQLiteCheckpointStore(database)
    await run_store.initialize()
    await checkpoint_store.initialize()

    run = await run_store.create(
        conversation_id="conv-1",
        user_message=user_message,
    )
    await run_store.mark_started(run.id)
    if with_checkpoint:
        await checkpoint_store.start(
            run.id,
            conversation_id="conv-1",
            user_message=Message(role=MessageRole.USER, content=user_message),
        )
        await checkpoint_store.before_model(run.id, step=2)
        if checkpoint_status is CheckpointStatus.INTERRUPTED:
            await checkpoint_store.interrupt(run.id, error="process stopped")
    return run.id, run_store, checkpoint_store, str(database)


# 函数说明：test_reconcile_running_with_recoverable_checkpoint_becomes_interrupted
# 用途：回归验证回归测试与测试辅助中的
# `reconcile_running_with_recoverable_checkpoint_becomes_interrupted` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_stale_running_run` →
# `fake_registry` → `model_response` → `SQLiteRunStore` → `SQLiteCheckpointStore` →
# `build_manager`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`[item.id for item in reconciled] == [run_id]`。
#   验证条件：`reconciled[0].status is RunStatus.INTERRUPTED`。
#   验证条件：`checkpoint is not None`。
#   验证条件：`checkpoint.status is CheckpointStatus.INTERRUPTED`。
async def test_reconcile_running_with_recoverable_checkpoint_becomes_interrupted(
    tmp_path,
    manager_factory,
) -> None:
    build_manager = manager_factory
    run_id, _, _, database = await _make_stale_running_run(
        tmp_path,
        with_checkpoint=True,
        checkpoint_status=CheckpointStatus.RUNNING,
    )

    registry, _ = fake_registry([model_response(content="完成")])
    run_store = SQLiteRunStore(database)
    checkpoint_store = SQLiteCheckpointStore(database)
    manager, _, _ = await build_manager(
        registry,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
    )
    reconciled = await manager.initialize()

    assert [item.id for item in reconciled] == [run_id]
    assert reconciled[0].status is RunStatus.INTERRUPTED
    checkpoint = await checkpoint_store.get(run_id)
    assert checkpoint is not None
    assert checkpoint.status is CheckpointStatus.INTERRUPTED
    still_running = await run_store.list_runs(status=RunStatus.RUNNING)
    assert still_running == ()


# 函数说明：test_reconcile_running_without_checkpoint_becomes_failed
# 用途：回归验证回归测试与测试辅助中的
# `reconcile_running_without_checkpoint_becomes_failed` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_stale_running_run` →
# `fake_registry` → `model_response` → `SQLiteRunStore` → `SQLiteCheckpointStore` →
# `build_manager`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`[item.id for item in reconciled] == [run_id]`。
#   验证条件：`reconciled[0].status is RunStatus.FAILED`。
#   验证条件：`'no recoverable checkpoint' in (reconciled[0].error or '')`。
async def test_reconcile_running_without_checkpoint_becomes_failed(
    tmp_path,
    manager_factory,
) -> None:
    build_manager = manager_factory
    run_id, _, _, database = await _make_stale_running_run(
        tmp_path,
        with_checkpoint=False,
    )

    registry, _ = fake_registry([model_response(content="完成")])
    run_store = SQLiteRunStore(database)
    checkpoint_store = SQLiteCheckpointStore(database)
    manager, _, _ = await build_manager(
        registry,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
    )
    reconciled = await manager.initialize()

    assert [item.id for item in reconciled] == [run_id]
    assert reconciled[0].status is RunStatus.FAILED
    assert "no recoverable checkpoint" in (reconciled[0].error or "")


# 函数说明：test_reconcile_stale_pending_becomes_failed
# 用途：回归验证回归测试与测试辅助中的 `reconcile_stale_pending_becomes_failed` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteRunStore` →
# `run_store.initialize` → `run_store.create` → `fake_registry` → `model_response` →
# `SQLiteCheckpointStore`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`pending.status is RunStatus.PENDING`。
#   验证条件：`[item.id for item in reconciled] == [pending.id]`。
#   验证条件：`reconciled[0].status is RunStatus.FAILED`。
#   验证条件：`'never started' in (reconciled[0].error or '')`。
#   预期异常：`pytest.raises(ValueError)`。
async def test_reconcile_stale_pending_becomes_failed(
    tmp_path,
    manager_factory,
) -> None:

    build_manager = manager_factory
    database = tmp_path / "muharness.db"
    run_store = SQLiteRunStore(database)
    await run_store.initialize()
    pending = await run_store.create(
        conversation_id="conv-1",
        user_message="从未开始的运行",
    )
    assert pending.status is RunStatus.PENDING

    registry, _ = fake_registry([model_response(content="完成")])
    checkpoint_store = SQLiteCheckpointStore(database)
    await checkpoint_store.initialize()
    manager, _, _ = await build_manager(
        registry,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
    )

    reconciled = await manager.initialize()

    assert [item.id for item in reconciled] == [pending.id]
    assert reconciled[0].status is RunStatus.FAILED
    assert "never started" in (reconciled[0].error or "")
    still_pending = await run_store.list_runs(status=RunStatus.PENDING)
    assert still_pending == ()
    with pytest.raises(ValueError):
        await manager.recover(pending.id)




# 函数说明：test_recover_interrupted_run_uses_checkpoint_and_completes
# 用途：回归验证回归测试与测试辅助中的
# `recover_interrupted_run_uses_checkpoint_and_completes` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteRunStore` →
# `SQLiteCheckpointStore` → `run_store.initialize` → `checkpoint_store.initialize` →
# `ToolCall` → `run_store.create`；另有 17 个调用点。
# 分支与异常：
#   验证条件：`old is not None and old.status is RunStatus.INTERRUPTED`。
#   验证条件：`new_run.status is RunStatus.COMPLETED`。
#   验证条件：`new_run.recovered_from_run_id == old_run.id`。
#   验证条件：`new_run.conversation_id == 'conv-1'`。
async def test_recover_interrupted_run_uses_checkpoint_and_completes(
    tmp_path,
    manager_factory,
) -> None:
    build_manager = manager_factory
    database = tmp_path / "muharness.db"
    run_store = SQLiteRunStore(database)
    checkpoint_store = SQLiteCheckpointStore(database)
    await run_store.initialize()
    await checkpoint_store.initialize()

    done = ToolCall(id="done-1", name="count", arguments={"value": 1})
    pending = ToolCall(id="pending-1", name="count", arguments={"value": 2})
    old_run = await run_store.create(
        conversation_id="conv-1",
        user_message="执行任务",
    )
    await run_store.mark_started(old_run.id)
    await checkpoint_store.start(
        old_run.id,
        conversation_id="conv-1",
        user_message=Message(role=MessageRole.USER, content="执行任务"),
    )
    await checkpoint_store.before_model(old_run.id, step=1)
    await checkpoint_store.before_tools(
        old_run.id,
        step=1,
        tool_calls=(done, pending),
    )
    await checkpoint_store.complete_tool(
        old_run.id,
        result=_tool_result(done),
    )
    await checkpoint_store.interrupt(old_run.id, error="process stopped")

    registry, adapter = fake_registry([model_response(content="已核对并继续")])
    manager, _, _ = await build_manager(
        registry,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
    )
    await manager.initialize()
    old = await manager.get_run(old_run.id)
    assert old is not None and old.status is RunStatus.INTERRUPTED

    new_run_id, _ = await manager.recover(
        old_run.id,
        history=(),
    )
    new_run = await manager.wait(new_run_id)

    assert new_run.status is RunStatus.COMPLETED
    assert new_run.recovered_from_run_id == old_run.id
    assert new_run.conversation_id == "conv-1"

    injected = next(
        message
        for message in adapter.requests[0].messages
        if message.name == CHECKPOINT_CONTEXT_MESSAGE_NAME
    )
    assert "done-1" in (injected.content or "")
    assert "pending-1" in (injected.content or "")

    assert await checkpoint_store.get_unrecovered(old_run.id) is None


# 函数说明：_tool_result
# 用途：处理回归测试与测试辅助中的 `_tool_result` 数据；结果及边界条件见下方说明。
# 参数：
#   call：调用输入或配置值，类型 `ToolCall`。
# 返回：返回 `ToolResult(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolResult`。
def _tool_result(call: ToolCall):
    from app.models.types import ToolResult

    return ToolResult(
        tool_call_id=call.id,
        tool_name=call.name,
        success=True,
        output="ok",
        duration_ms=1.0,
    )




# 函数说明：test_completed_tool_results_are_not_reexecuted
# 用途：回归验证回归测试与测试辅助中的 `completed_tool_results_are_not_reexecuted` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteRunStore` →
# `SQLiteCheckpointStore` → `run_store.initialize` → `checkpoint_store.initialize` →
# `ToolCall` → `run_store.create`；另有 18 个调用点。
# 分支与异常：
#   验证条件：`new_run.status is RunStatus.COMPLETED`。
#   验证条件：`tool.executions == 0`。
#   验证条件：`checkpoint is not None and checkpoint.recovered_by_run_id == new_run_id`
# 。
async def test_completed_tool_results_are_not_reexecuted(tmp_path) -> None:
    database = tmp_path / "muharness.db"
    run_store = SQLiteRunStore(database)
    checkpoint_store = SQLiteCheckpointStore(database)
    await run_store.initialize()
    await checkpoint_store.initialize()

    done = ToolCall(id="done-1", name="count", arguments={"value": 1})
    old_run = await run_store.create(
        conversation_id="conv-1",
        user_message="统计",
    )
    await run_store.mark_started(old_run.id)
    await checkpoint_store.start(
        old_run.id,
        conversation_id="conv-1",
        user_message=Message(role=MessageRole.USER, content="统计"),
    )
    await checkpoint_store.before_model(old_run.id, step=1)
    await checkpoint_store.before_tools(old_run.id, step=1, tool_calls=(done,))
    await checkpoint_store.complete_tool(old_run.id, result=_tool_result(done))
    await checkpoint_store.interrupt(old_run.id, error="process stopped")

    tool = CountingTool()
    tools = ToolRegistry()
    tools.register(tool)
    registry, _ = fake_registry([model_response(content="统计完成")])
    runtime = AgentRuntime(
        registry,
        tools,
        provider="fake",
        checkpoint_store=checkpoint_store,
    )
    manager = RunManager(run_store, checkpoint_store, runtime)
    await manager.initialize()

    new_run_id, _ = await manager.recover(old_run.id, history=())
    new_run = await manager.wait(new_run_id)

    assert new_run.status is RunStatus.COMPLETED
    assert tool.executions == 0  
    checkpoint = await checkpoint_store.get(old_run.id)
    assert checkpoint is not None and checkpoint.recovered_by_run_id == new_run_id




# 函数说明：test_invalid_state_transitions_rejected
# 用途：回归验证回归测试与测试辅助中的 `invalid_state_transitions_rejected` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteRunStore` → `store.initialize`
# → `store.create` → `pytest.raises` → `store.mark_completed` → `store.mark_cancelled`；
# 另有 10 个调用点。
# 分支与异常：
#   验证条件：`run2 is not None`。
#   验证条件：`again.status is RunStatus.COMPLETED`。
#   预期异常：`pytest.raises(ValueError, match='invalid run transition')`。
async def test_invalid_state_transitions_rejected(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path / "muharness.db")
    await store.initialize()

    run = await store.create(conversation_id="conv-1", user_message="x")
    with pytest.raises(ValueError, match="invalid run transition"):
        await store.mark_completed(run.id)
    with pytest.raises(ValueError, match="invalid run transition"):
        await store.mark_cancelled(run.id)

    await store.mark_started(run.id)
    await store.mark_completed(run.id)
    with pytest.raises(ValueError, match="invalid run transition"):
        await store.mark_started(run.id)
    with pytest.raises(ValueError, match="invalid run transition"):
        await store.mark_cancelled(run.id)
    with pytest.raises(ValueError, match="invalid run transition"):
        await store.mark_interrupted(run.id)

    run2 = await store.create(conversation_id="conv-1", user_message="x")
    await store.mark_started(run2.id)
    await store.mark_interrupted(run2.id)
    assert run2 is not None
    with pytest.raises(ValueError, match="invalid run transition"):
        await store.mark_started(run2.id)
    with pytest.raises(ValueError, match="invalid run transition"):
        await store.mark_completed(run2.id)
    with pytest.raises(ValueError, match="invalid run transition"):
        await store.mark_cancelled(run2.id)

    manager_runtime_registry, _ = fake_registry([model_response(content="完成")])
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await checkpoint_store.initialize()
    manager = RunManager(
        store,
        checkpoint_store,
        AgentRuntime(
            manager_runtime_registry,
            ToolRegistry(),
            provider="fake",
            checkpoint_store=checkpoint_store,
        ),
    )
    again = await manager.cancel(run.id)
    assert again.status is RunStatus.COMPLETED




# 函数说明：test_multiple_runs_in_one_conversation
# 用途：回归验证回归测试与测试辅助中的 `multiple_runs_in_one_conversation` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `build_manager` → `manager.start` → `manager.wait` → `manager.list_runs`。
# 分支与异常：
#   验证条件：`len(runs) == 2`。
#   验证条件：`{item.id for item in runs} == {first_id, second_id}`。
#   验证条件：`all((item.status is RunStatus.COMPLETED for item in runs))`。
#   验证条件：`await manager.list_runs(conversation_id='conv-other') == ()`。
async def test_multiple_runs_in_one_conversation(manager_factory) -> None:
    build_manager = manager_factory
    registry, _ = fake_registry(
        [
            model_response(content="第一次"),
            model_response(content="第二次"),
        ]
    )
    manager, run_store, _ = await build_manager(registry)

    first_id, _ = await manager.start("问题一", conversation_id="conv-1")
    second_id, _ = await manager.start("问题二", conversation_id="conv-1")
    await manager.wait(first_id)
    await manager.wait(second_id)

    runs = await manager.list_runs(conversation_id="conv-1")
    assert len(runs) == 2
    assert {item.id for item in runs} == {first_id, second_id}
    assert all(item.status is RunStatus.COMPLETED for item in runs)
    assert await manager.list_runs(conversation_id="conv-other") == ()




# 函数说明：test_run_manager_keeps_trace_and_checkpoint_behavior
# 用途：回归验证回归测试与测试辅助中的 `run_manager_keeps_trace_and_checkpoint_behavior`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteTraceStore` →
# `trace_store.initialize` → `SQLiteTraceEventHandler` → `fake_registry` →
# `model_response` → `ToolCall`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`run.status is RunStatus.COMPLETED`。
#   验证条件：`trace is not None`。
#   验证条件：`trace.status.value == 'completed'`。
#   验证条件：`AgentEventType.AGENT_STARTED in types`。
async def test_run_manager_keeps_trace_and_checkpoint_behavior(
    manager_factory,
    tmp_path,
) -> None:
    from app.records.trace import SQLiteTraceEventHandler, SQLiteTraceStore

    build_manager = manager_factory
    trace_store = SQLiteTraceStore(tmp_path / "trace.db")
    await trace_store.initialize()
    trace_handler = SQLiteTraceEventHandler(trace_store)

    registry, _ = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(id="count-1", name="count", arguments={"value": 1}),
                )
            ),
            model_response(content="完成"),
        ]
    )
    tool = CountingTool()
    tools = ToolRegistry()
    tools.register(tool)
    manager, _, checkpoint_store = await build_manager(registry, tools)

    run_id, _ = await manager.start(
        "执行工具",
        conversation_id="conv-1",
        event_handler=trace_handler,
    )
    run = await manager.wait(run_id)

    assert run.status is RunStatus.COMPLETED
    trace = await trace_store.get(run_id)
    assert trace is not None
    assert trace.status.value == "completed"
    events = await trace_store.load_events(run_id)
    types = {event.type for event in events}
    assert AgentEventType.AGENT_STARTED in types
    assert AgentEventType.TOOL_STARTED in types
    assert AgentEventType.AGENT_COMPLETED in types
    checkpoint = await checkpoint_store.get(run_id)
    assert checkpoint is not None
    assert checkpoint.status is CheckpointStatus.COMPLETED
    assert [c.tool_call_id for c in checkpoint.completed_tool_results] == ["count-1"]
    assert tool.executions == 1




# 函数说明：test_list_runs_status_filter
# 用途：回归验证回归测试与测试辅助中的 `list_runs_status_filter` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `build_manager` → `manager.start` → `manager.wait` → `manager.list_runs`。
# 分支与异常：
#   验证条件：`{item.id for item in completed} == {first, second}`。
#   验证条件：`await manager.list_runs(status=RunStatus.RUNNING) == ()`。
async def test_list_runs_status_filter(manager_factory) -> None:
    build_manager = manager_factory
    registry, _ = fake_registry(
        [
            model_response(content="完成"),
            model_response(content="完成"),
        ]
    )
    manager, _, _ = await build_manager(registry)

    first, _ = await manager.start("a", conversation_id="conv-1")
    second, _ = await manager.start("b", conversation_id="conv-1")
    await manager.wait(first)
    await manager.wait(second)

    completed = await manager.list_runs(status=RunStatus.COMPLETED)
    assert {item.id for item in completed} == {first, second}
    assert await manager.list_runs(status=RunStatus.RUNNING) == ()




# 函数说明：test_plain_start_does_not_auto_recover_interrupted_checkpoint
# 用途：回归验证回归测试与测试辅助中的
# `plain_start_does_not_auto_recover_interrupted_checkpoint` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `_make_interrupted_run_with_checkpoint` → `fake_registry` → `model_response` →
# `build_manager` → `manager.start` → `manager.wait`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`new_run.status is RunStatus.COMPLETED`。
#   验证条件：`not any((message.name == CHECKPOINT_CONTEXT_MESSAGE_NAME for message in
# adapter.requests[0].messages))`。
#   验证条件：`await old_checkpoint_store.get_unrecovered(old_run_id) is not None`。
#   验证条件：`new_run.recovered_from_run_id is None`。
async def test_plain_start_does_not_auto_recover_interrupted_checkpoint(
    manager_factory,
    tmp_path,
) -> None:
    build_manager = manager_factory
    old_run = await _make_interrupted_run_with_checkpoint(tmp_path)
    old_run_id = old_run["run_id"]
    old_checkpoint_store = old_run["checkpoint_store"]

    registry, adapter = fake_registry([model_response(content="完成")])
    manager, _, checkpoint_store = await build_manager(registry)
    new_run_id, _ = await manager.start("继续", conversation_id="conv-1")
    new_run = await manager.wait(new_run_id)

    assert new_run.status is RunStatus.COMPLETED
    assert not any(
        message.name == CHECKPOINT_CONTEXT_MESSAGE_NAME
        for message in adapter.requests[0].messages
    )
    assert await old_checkpoint_store.get_unrecovered(old_run_id) is not None
    assert new_run.recovered_from_run_id is None




# 函数说明：test_reconcile_also_marks_stale_checkpoints
# 用途：回归验证回归测试与测试辅助中的 `reconcile_also_marks_stale_checkpoints` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteCheckpointStore` →
# `checkpoint_store.initialize` → `checkpoint_store.start` → `Message` → `fake_registry`
#  → `model_response`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`checkpoint is not None`。
#   验证条件：`checkpoint.status is CheckpointStatus.INTERRUPTED`。
async def test_reconcile_also_marks_stale_checkpoints(
    manager_factory,
    tmp_path,
) -> None:
    build_manager = manager_factory
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await checkpoint_store.initialize()
    await checkpoint_store.start(
        "orphan-cp",
        conversation_id="conv-9",
        user_message=Message(role=MessageRole.USER, content="遗留任务"),
    )

    registry, _ = fake_registry([model_response(content="完成")])
    manager, _, checkpoint_store2 = await build_manager(
        registry,
        checkpoint_store=checkpoint_store,
    )
    await manager.initialize()

    checkpoint = await checkpoint_store2.get("orphan-cp")
    assert checkpoint is not None
    assert checkpoint.status is CheckpointStatus.INTERRUPTED




# 函数说明：_make_interrupted_run_with_checkpoint
# 用途：构造运行检查点，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `dict[str, object]`；字典，包含字段 `run_id`、`run_store`、
# `checkpoint_store`、`database`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteRunStore` →
# `SQLiteCheckpointStore` → `run_store.initialize` → `checkpoint_store.initialize` →
# `ToolCall` → `run_store.create`；另有 8 个调用点。
async def _make_interrupted_run_with_checkpoint(
    tmp_path,
) -> dict[str, object]:
    database = tmp_path / "muharness.db"
    run_store = SQLiteRunStore(database)
    checkpoint_store = SQLiteCheckpointStore(database)
    await run_store.initialize()
    await checkpoint_store.initialize()

    done = ToolCall(id="done-x", name="count", arguments={"value": 1})
    run = await run_store.create(conversation_id="conv-1", user_message="统计")
    await run_store.mark_started(run.id)
    await checkpoint_store.start(
        run.id,
        conversation_id="conv-1",
        user_message=Message(role=MessageRole.USER, content="统计"),
    )
    await checkpoint_store.before_model(run.id, step=1)
    await checkpoint_store.before_tools(run.id, step=1, tool_calls=(done,))
    await checkpoint_store.complete_tool(run.id, result=_tool_result(done))
    await checkpoint_store.interrupt(run.id, error="process stopped")
    return {
        "run_id": run.id,
        "run_store": run_store,
        "checkpoint_store": checkpoint_store,
        "database": str(database),
    }


# 函数说明：test_preassigned_run_id_starts_once
# 用途：回归验证回归测试与测试辅助中的 `preassigned_run_id_starts_once` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `manager_factory` → `manager.start` → `manager.wait` → `pytest.raises`。
# 分支与异常：
#   验证条件：`run_id == 'mea-r001-exec'`。
#   验证条件：`(await manager.wait(run_id)).status is RunStatus.COMPLETED`。
#   验证条件：`len(adapter.requests) == 1`。
#   验证条件：`stored is not None and stored.user_message == '你好'`。
#   预期异常：`pytest.raises(RunAlreadyExists)`。
async def test_preassigned_run_id_starts_once(manager_factory) -> None:
    from app.runtime.run import RunAlreadyExists

    registry, adapter = fake_registry(
        [model_response(content="完成"), model_response(content="不应出现")]
    )
    manager, run_store, _ = await manager_factory(registry)

    run_id, _ = await manager.start("你好", conversation_id="conv-1", run_id="mea-r001-exec")
    assert run_id == "mea-r001-exec"
    assert (await manager.wait(run_id)).status is RunStatus.COMPLETED

    with pytest.raises(RunAlreadyExists):
        await manager.start("再来一次", conversation_id="conv-1", run_id="mea-r001-exec")
    assert len(adapter.requests) == 1
    stored = await run_store.get(run_id)
    assert stored is not None and stored.user_message == "你好"


# 函数说明：test_start_uses_runtime_override
# 用途：回归验证回归测试与测试辅助中的 `start_uses_runtime_override` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   manager_factory：构造测试管理器的工厂夹具。
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `manager_factory` →
# `model_response` → `AgentRuntime` → `ToolRegistry` → `manager.start`；另有 2 个调用点
# 。
# 分支与异常：
#   验证条件：`(await manager.wait(run_id)).status is RunStatus.COMPLETED`。
#   验证条件：`default_adapter.requests == []`。
#   验证条件：`len(role_adapter.requests) == 1`。
#   验证条件：`result is not None and result.content == '角色回复'`。
async def test_start_uses_runtime_override(manager_factory, tmp_path) -> None:
    default_registry, default_adapter = fake_registry([])
    manager, _, checkpoint_store = await manager_factory(default_registry)
    role_registry, role_adapter = fake_registry([model_response(content="角色回复")])
    role_runtime = AgentRuntime(
        role_registry,
        ToolRegistry(),
        provider="fake",
        checkpoint_store=checkpoint_store,
    )

    run_id, _ = await manager.start("你好", runtime=role_runtime)
    assert (await manager.wait(run_id)).status is RunStatus.COMPLETED
    assert default_adapter.requests == []
    assert len(role_adapter.requests) == 1
    result = manager.result(run_id)
    assert result is not None and result.content == "角色回复"
