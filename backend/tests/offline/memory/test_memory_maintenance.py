
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.domain.memory import (
    MemoryMaintenanceConfig,
    MemoryMaintenanceReflector,
    MemoryManager,
    MemoryReflectionConfig,
    PostRunMemoryReflector,
)
from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
)
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.runtime import AgentRuntime
from app.tools.registry import ToolRegistry


class FakeAdapter(ModelAdapter):

    # 函数说明：FakeAdapter.__init__
    # 用途：初始化 FakeAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`
    # 。
    #   delay_seconds：时间间隔或限额，单位为秒，类型 `float`；默认 `0.0`。
    #   before_return：`before_return`输入或配置值，类型
    # `Callable[[], Awaitable[None]] | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.delay_seconds`、`self.before_return`、
    # `self.requests`。
    def __init__(
        self,
        config: ProviderConfig,
        responses: Sequence[ModelResponse | Exception],
        *,
        delay_seconds: float = 0.0,
        before_return: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(config)
        self.responses = list(responses)
        self.delay_seconds = delay_seconds
        self.before_return = before_return
        self.requests: list[ModelRequest] = []

    # 函数说明：FakeAdapter.complete
    # 用途：完成FakeAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.sleep` →
    # `self.before_return` → `self.responses.pop`。
    # 分支与异常：
    #   当 `isinstance(response, Exception)` 时，抛出 `response`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.before_return is not None:
            await self.before_return()
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    # 函数说明：FakeAdapter.close
    # 用途：关闭FakeAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：_provider_config
# 用途：返回 `ProviderConfig(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   provider：模型或搜索服务商，类型 `str`。
#   model：模型名称，类型 `str`。
# 返回：类型 `ProviderConfig`；返回 `ProviderConfig(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr`。
def _provider_config(provider: str, model: str) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model=model,
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )


# 函数说明：_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str`。
#   provider：模型或搜索服务商，类型 `str`。
#   model：模型名称，类型 `str`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message`。
def _response(
    content: str,
    *,
    provider: str,
    model: str,
) -> ModelResponse:
    return ModelResponse(
        id=f"{provider}-response",
        provider=provider,
        model=model,
        message=Message(role=MessageRole.ASSISTANT, content=content),
    )


# 函数说明：_registry
# 用途：在回归测试与测试辅助中处理 `_registry`，通过 `adapters.items` 完成首个内部处理步
# 骤。
# 参数：
#   **adapters：额外关键字参数，按实现处理或转交。
# 返回：类型 `ModelAdapterRegistry`；返回 `registry`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelAdapterRegistry` →
# `ModelSettings` → `registry.register`。
def _registry(**adapters: FakeAdapter) -> ModelAdapterRegistry:
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    for provider, adapter in adapters.items():
        registry.register(
            provider,
            lambda _, current=adapter: current,
            config=adapter.config,
        )
    return registry


# 函数说明：_reflection_config
# 用途：返回
# `MemoryReflectionConfig(_env_file=None, provider='reflect', model='reflection-model')`
# ，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `MemoryReflectionConfig`；返回
# `MemoryReflectionConfig(_env_file=None, provider='reflect', model='reflection-model')`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryReflectionConfig`。
def _reflection_config() -> MemoryReflectionConfig:
    return MemoryReflectionConfig(
        _env_file=None,
        provider="reflect",
        model="reflection-model",
    )


# 函数说明：_maintenance_config
# 用途：在回归测试与测试辅助中处理 `_maintenance_config`，通过 `values.update` 完成首个
# 内部处理步骤。
# 参数：
#   **overrides：额外关键字参数，按实现处理或转交。
# 返回：类型 `MemoryMaintenanceConfig`；返回
# `MemoryMaintenanceConfig(_env_file=None, **values)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`values.update` →
# `MemoryMaintenanceConfig`。
def _maintenance_config(**overrides: object) -> MemoryMaintenanceConfig:
    values: dict[str, object] = {
        "provider": "maintain",
        "model": "maintenance-model",
    }
    values.update(overrides)
    return MemoryMaintenanceConfig(_env_file=None, **values)


# 函数说明：_manager
# 用途：在回归测试与测试辅助中处理 `_manager`，通过 `manager.initialize` 完成首个内部处
# 理步骤。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
# 返回：类型 `MemoryManager`；返回 `manager`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` → `manager.initialize`
# 。
async def _manager(path: Path) -> MemoryManager:
    manager = MemoryManager(path)
    await manager.initialize()
    return manager


# 函数说明：_fill
# 用途：在回归测试与测试辅助中处理 `_fill`，通过 `manager.create` 完成首个内部处理步骤。
# 参数：
#   manager：当前业务管理器，类型 `MemoryManager`。
#   count：`count`输入或配置值，类型 `int`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`manager.create` →
# `manager.store.create`。
async def _fill(manager: MemoryManager, count: int) -> None:
    for index in range(1, count + 1):
        values = {
            "title": f"记忆 {index}",
            "summary": f"cue {index}",
            "content": f"完整正文 {index}",
        }
        if index <= manager.max_active:
            await manager.create(**values)
        else:
            await manager.store.create(**values)


# 函数说明：_runtime
# 用途：处理回归测试与测试辅助中的 `_runtime` 数据；结果及边界条件见下方说明。
# 参数：
#   manager：当前业务管理器，类型 `MemoryManager`。
#   reflection_content：传给 `_response` 的输入，类型 `str`。
#   maintenance_responses：传给 `FakeAdapter` 的输入，类型
# `Sequence[ModelResponse | Exception]`。
#   maintenance_config：维护配置输入或配置值，类型 `MemoryMaintenanceConfig | None`；默
# 认 `None`。
#   maintenance_delay：维护输入或配置值，类型 `float`；默认 `0.0`。
#   maintenance_before_return：维护输入或配置值，类型
# `Callable[[], Awaitable[None]] | None`；默认 `None`。
# 返回：类型 `tuple[AgentRuntime, InMemoryEventHandler, FakeAdapter, FakeAdapter]`；返回
#  `(runtime, events, reflect, maintain)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeAdapter` → `_provider_config` →
# `_response` → `_registry` → `InMemoryEventHandler` → `AgentRuntime`；另有 5 个调用点。
def _runtime(
    manager: MemoryManager,
    *,
    reflection_content: str,
    maintenance_responses: Sequence[ModelResponse | Exception],
    maintenance_config: MemoryMaintenanceConfig | None = None,
    maintenance_delay: float = 0.0,
    maintenance_before_return: Callable[[], Awaitable[None]] | None = None,
) -> tuple[AgentRuntime, InMemoryEventHandler, FakeAdapter, FakeAdapter]:
    main = FakeAdapter(
        _provider_config("main", "main-model"),
        [_response("主任务完成", provider="main", model="main-model")],
    )
    reflect = FakeAdapter(
        _provider_config("reflect", "reflection-model"),
        [
            _response(
                reflection_content,
                provider="reflect",
                model="reflection-model",
            )
        ],
    )
    maintain = FakeAdapter(
        _provider_config("maintain", "maintenance-model"),
        maintenance_responses,
        delay_seconds=maintenance_delay,
        before_return=maintenance_before_return,
    )
    registry = _registry(main=main, reflect=reflect, maintain=maintain)
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="main",
        memory_manager=manager,
        memory_reflector=PostRunMemoryReflector(
            registry,
            config=_reflection_config(),
        ),
        memory_maintenance_reflector=MemoryMaintenanceReflector(
            registry,
            config=maintenance_config or _maintenance_config(),
            default_provider="reflect",
            default_model="reflection-model",
        ),
    )
    return runtime, events, reflect, maintain


# 函数说明：_create_decision
# 用途：创建`decision`，供回归测试与测试辅助使用。
# 返回：类型 `str`；返回 `'{"action":"create","title":"第 26 条","summary":"新的长期架构
# 决定","content":"新的长期正文","reason":"这是…`。
def _create_decision() -> str:
    return (
        '{"action":"create","title":"第 26 条",'
        '"summary":"新的长期架构决定","content":"新的长期正文",'
        '"reason":"这是耐久的跨会话信息"}'
    )


# 函数说明：_archive_decision
# 用途：归档`decision`，供回归测试与测试辅助使用。
# 参数：
#   memory_id：目标记忆标识，类型 `str`；默认 `'M001'`。
# 返回：类型 `ModelResponse`；返回 `_response(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_response`。
def _archive_decision(memory_id: str = "M001") -> ModelResponse:
    return _response(
        '{"action":"archive","memory_id":"'
        f'{memory_id}","reason":"该记忆已经过时"}}',
        provider="maintain",
        model="maintenance-model",
    )


# 函数说明：_defer_decision
# 用途：返回 `_response(…)`，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `ModelResponse`；返回 `_response(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_response`。
def _defer_decision() -> ModelResponse:
    return _response(
        '{"action":"defer","memory_id":null,'
        '"reason":"当前候选仍然具有独立价值"}',
        provider="maintain",
        model="maintenance-model",
    )


# 函数说明：test_full_capacity_archives_then_creates_without_exceeding_limit
# 用途：回归验证回归测试与测试辅助中的
# `full_capacity_archives_then_creates_without_exceeding_limit` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` → `_runtime` →
# `_create_decision` → `_archive_decision` → `_maintenance_config`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`await manager.active_count() == 25`。
#   验证条件：`(manager.memory_dir / 'archive' / 'M001.md').is_file()`。
#   验证条件：`(manager.memory_dir / 'active' / 'M026.md').is_file()`。
@pytest.mark.asyncio
async def test_full_capacity_archives_then_creates_without_exceeding_limit(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 25)
    runtime, events, _, maintain = _runtime(
        manager,
        reflection_content=_create_decision(),
        maintenance_responses=[_archive_decision()],
        maintenance_config=_maintenance_config(
            max_output_tokens=456,
            temperature=0.1,
        ),
    )

    result = await runtime.run("完成新的长期决定", event_handler=events)

    assert result.ok is True
    assert await manager.active_count() == 25
    assert (manager.memory_dir / "archive" / "M001.md").is_file()
    assert (manager.memory_dir / "active" / "M026.md").is_file()
    assert len(maintain.requests) == 1
    assert maintain.requests[0].max_output_tokens == 456
    assert maintain.requests[0].temperature == 0.1
    assert "完整正文 1" in (maintain.requests[0].messages[-1].content or "")
    event_types = [event.type for event in events.events]
    assert event_types.index(AgentEventType.AGENT_COMPLETED) < event_types.index(
        AgentEventType.MEMORY_REFLECTION_STARTED
    )
    assert AgentEventType.MEMORY_REFLECTION_COMPLETED in event_types
    assert AgentEventType.MEMORY_MAINTENANCE_COMPLETED in event_types
    maintenance_event = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_MAINTENANCE_COMPLETED
    )
    assert maintenance_event.maintenance_memory_id == "M001"
    assert maintenance_event.maintenance_remaining_overflow == 0
    reflection_event = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_REFLECTION_COMPLETED
    )
    assert reflection_event.reflection_memory_id == "M026"
    assert reflection_event.reflection_mutation_applied is True


# 函数说明：test_defer_keeps_old_memories_and_skips_create
# 用途：回归验证回归测试与测试辅助中的 `defer_keeps_old_memories_and_skips_create` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` → `_runtime` →
# `_create_decision` → `_defer_decision` → `runtime.run`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`await manager.active_count() == 25`。
#   验证条件：`not (manager.memory_dir / 'active' / 'M026.md').exists()`。
#   验证条件：`not any((manager.memory_dir / 'archive').iterdir())`。
@pytest.mark.asyncio
async def test_defer_keeps_old_memories_and_skips_create(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 25)
    runtime, events, _, _ = _runtime(
        manager,
        reflection_content=_create_decision(),
        maintenance_responses=[_defer_decision()],
    )

    result = await runtime.run("完成新的长期决定", event_handler=events)

    assert result.ok is True
    assert await manager.active_count() == 25
    assert not (manager.memory_dir / "active" / "M026.md").exists()
    assert not any((manager.memory_dir / "archive").iterdir())
    maintenance_event = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_MAINTENANCE_COMPLETED
    )
    assert maintenance_event.maintenance_action == "defer"
    reflection_event = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_REFLECTION_FAILED
    )
    assert reflection_event.reflection_mutation_applied is False


# 函数说明：test_maintenance_failure_never_archives_or_fails_agent
# 用途：回归验证回归测试与测试辅助中的
# `maintenance_failure_never_archives_or_fails_agent` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`。
#   delay：`delay`输入或配置值，类型 `float`。
#   timeout：等待超时配置，类型 `float`。
#   error_text：错误文本输入或配置值，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` → `_runtime` →
# `_create_decision` → `_maintenance_config` → `runtime.run`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`await manager.active_count() == 25`。
#   验证条件：`not any((manager.memory_dir / 'archive').iterdir())`。
#   验证条件：`failed.maintenance_error is not None`。
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "delay", "timeout", "error_text"),
    [
        ([RuntimeError("provider unavailable")], 0.0, 1.0, "provider unavailable"),
        (
            [_response("invalid-json", provider="maintain", model="maintenance-model")],
            0.0,
            1.0,
            "ValidationError",
        ),
        ([_defer_decision()], 0.05, 0.01, "TimeoutError"),
    ],
)
async def test_maintenance_failure_never_archives_or_fails_agent(
    tmp_path: Path,
    responses: Sequence[ModelResponse | Exception],
    delay: float,
    timeout: float,
    error_text: str,
) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 25)
    runtime, events, _, _ = _runtime(
        manager,
        reflection_content=_create_decision(),
        maintenance_responses=responses,
        maintenance_config=_maintenance_config(timeout_seconds=timeout),
        maintenance_delay=delay,
    )

    result = await runtime.run("正常完成", event_handler=events)

    assert result.ok is True
    assert await manager.active_count() == 25
    assert not any((manager.memory_dir / "archive").iterdir())
    failed = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_MAINTENANCE_FAILED
    )
    assert failed.maintenance_error is not None
    assert error_text in failed.maintenance_error
    event_types = [event.type for event in events.events]
    assert event_types.index(AgentEventType.AGENT_COMPLETED) < event_types.index(
        AgentEventType.MEMORY_MAINTENANCE_FAILED
    )


# 函数说明：test_maintenance_rejects_id_outside_candidate_set
# 用途：回归验证回归测试与测试辅助中的 `maintenance_rejects_id_outside_candidate_set` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` → `_runtime` →
# `_create_decision` → `_archive_decision` → `runtime.run`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`await manager.active_count() == 25`。
#   验证条件：`not any((manager.memory_dir / 'archive').iterdir())`。
#   验证条件：`failed.maintenance_memory_id == 'M999'`。
@pytest.mark.asyncio
async def test_maintenance_rejects_id_outside_candidate_set(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 25)
    runtime, events, _, _ = _runtime(
        manager,
        reflection_content=_create_decision(),
        maintenance_responses=[_archive_decision("M999")],
    )

    result = await runtime.run("正常完成", event_handler=events)

    assert result.ok is True
    assert await manager.active_count() == 25
    assert not any((manager.memory_dir / "archive").iterdir())
    failed = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_MAINTENANCE_FAILED
    )
    assert failed.maintenance_memory_id == "M999"
    assert failed.maintenance_error is not None
    assert "outside the candidate set" in failed.maintenance_error


# 函数说明：test_maintenance_rejects_candidate_changed_during_model_call
# 用途：回归验证回归测试与测试辅助中的
# `maintenance_rejects_candidate_changed_during_model_call` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` → `_runtime` →
# `_create_decision` → `_archive_decision` → `runtime.run`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`await manager.active_count() == 25`。
#   验证条件：`not (manager.memory_dir / 'archive' / 'M001.md').exists()`。
#   验证条件：`current is not None and current.content == '并发更新后的正文'`。
@pytest.mark.asyncio
async def test_maintenance_rejects_candidate_changed_during_model_call(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 25)

    # 函数说明：
    # test_maintenance_rejects_candidate_changed_during_model_call.mutate_candidate
    # 用途：在回归测试与测试辅助中处理 `mutate_candidate`，通过 `manager.update` 完成首
    # 个内部处理步骤。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`manager.update`。
    # 闭包依赖：从外层读取 `manager`。
    async def mutate_candidate() -> None:
        await manager.update(
            "M001",
            content="并发更新后的正文",
            reason="另一个 Run 更新了候选",
        )

    runtime, events, _, _ = _runtime(
        manager,
        reflection_content=_create_decision(),
        maintenance_responses=[_archive_decision()],
        maintenance_before_return=mutate_candidate,
    )

    result = await runtime.run("正常完成", event_handler=events)

    assert result.ok is True
    assert await manager.active_count() == 25
    assert not (manager.memory_dir / "archive" / "M001.md").exists()
    current = await manager.store.load("M001")
    assert current is not None and current.content == "并发更新后的正文"
    failed = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_MAINTENANCE_FAILED
    )
    assert failed.maintenance_error is not None
    assert "changed since maintenance snapshot" in failed.maintenance_error


# 函数说明：test_invalid_create_is_rejected_before_maintenance
# 用途：回归验证回归测试与测试辅助中的 `invalid_create_is_rejected_before_maintenance`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` → `_runtime` →
# `_archive_decision` → `runtime.run` → `manager.active_count`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`maintain.requests == []`。
#   验证条件：`await manager.active_count() == 25`。
#   验证条件：`not any((manager.memory_dir / 'archive').iterdir())`。
@pytest.mark.asyncio
async def test_invalid_create_is_rejected_before_maintenance(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 25)
    invalid_create = (
        '{"action":"create","title":"'
        + ("过长" * 101)
        + '","summary":"cue","content":"正文","reason":"耐久信息"}'
    )
    runtime, events, _, maintain = _runtime(
        manager,
        reflection_content=invalid_create,
        maintenance_responses=[_archive_decision()],
    )

    result = await runtime.run("正常完成", event_handler=events)

    assert result.ok is True
    assert maintain.requests == []
    assert await manager.active_count() == 25
    assert not any((manager.memory_dir / "archive").iterdir())
    assert any(
        event.type is AgentEventType.MEMORY_REFLECTION_FAILED
        for event in events.events
    )


# 函数说明：test_preexisting_overflow_converges_with_bounded_actions
# 用途：回归验证回归测试与测试辅助中的
# `preexisting_overflow_converges_with_bounded_actions` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` → `_runtime` →
# `_archive_decision` → `runtime.run` → `manager.active_count`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`await manager.active_count() == 25`。
#   验证条件：`len(maintain.requests) == 2`。
#   验证条件：`(manager.memory_dir / 'archive' / 'M001.md').is_file()`。
@pytest.mark.asyncio
async def test_preexisting_overflow_converges_with_bounded_actions(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 27)
    runtime, events, _, maintain = _runtime(
        manager,
        reflection_content='{"action":"none","reason":"没有新记忆"}',
        maintenance_responses=[
            _archive_decision("M001"),
            _archive_decision("M002"),
        ],
    )

    result = await runtime.run("普通请求", event_handler=events)

    assert result.ok is True
    assert await manager.active_count() == 25
    assert len(maintain.requests) == 2
    assert (manager.memory_dir / "archive" / "M001.md").is_file()
    assert (manager.memory_dir / "archive" / "M002.md").is_file()
    event_types = [event.type for event in events.events]
    assert event_types.index(AgentEventType.AGENT_COMPLETED) < event_types.index(
        AgentEventType.MEMORY_MAINTENANCE_STARTED
    )


# 函数说明：test_maintenance_can_recover_overflow_without_reflection
# 用途：回归验证回归测试与测试辅助中的
# `maintenance_can_recover_overflow_without_reflection` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` → `FakeAdapter` →
#  `_provider_config` → `_response` → `_archive_decision`；另有 9 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`await manager.active_count() == 25`。
#   验证条件：`len(maintain.requests) == 1`。
#   验证条件：`event_types.index(AgentEventType.AGENT_COMPLETED) < event_types.index(
# AgentEventType.MEMORY_MAINTENANCE_STARTED)`。
@pytest.mark.asyncio
async def test_maintenance_can_recover_overflow_without_reflection(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 26)
    main = FakeAdapter(
        _provider_config("main", "main-model"),
        [_response("主任务完成", provider="main", model="main-model")],
    )
    maintain = FakeAdapter(
        _provider_config("maintain", "maintenance-model"),
        [_archive_decision()],
    )
    registry = _registry(main=main, maintain=maintain)
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="main",
        memory_manager=manager,
        memory_maintenance_reflector=MemoryMaintenanceReflector(
            registry,
            config=_maintenance_config(),
        ),
    )

    result = await runtime.run("普通请求", event_handler=events)

    assert result.ok is True
    assert await manager.active_count() == 25
    assert len(maintain.requests) == 1
    event_types = [event.type for event in events.events]
    assert event_types.index(AgentEventType.AGENT_COMPLETED) < event_types.index(
        AgentEventType.MEMORY_MAINTENANCE_STARTED
    )


# 函数说明：test_max_actions_leaves_explicit_overflow_signal
# 用途：回归验证回归测试与测试辅助中的 `max_actions_leaves_explicit_overflow_signal` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` → `_runtime` →
# `_archive_decision` → `_maintenance_config` → `runtime.run`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`await manager.active_count() == 26`。
#   验证条件：`len(maintain.requests) == 2`。
#   验证条件：`skipped.maintenance_skip_reason == 'max_actions_reached'`。
@pytest.mark.asyncio
async def test_max_actions_leaves_explicit_overflow_signal(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 28)
    runtime, events, _, maintain = _runtime(
        manager,
        reflection_content='{"action":"none","reason":"没有新记忆"}',
        maintenance_responses=[
            _archive_decision("M001"),
            _archive_decision("M002"),
        ],
        maintenance_config=_maintenance_config(max_actions=2),
    )

    result = await runtime.run("普通请求", event_handler=events)

    assert result.ok is True
    assert await manager.active_count() == 26
    assert len(maintain.requests) == 2
    skipped = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_MAINTENANCE_SKIPPED
    )
    assert skipped.maintenance_skip_reason == "max_actions_reached"
    assert skipped.maintenance_remaining_overflow == 1


# 函数说明：test_concurrent_capacity_checked_create_never_exceeds_limit
# 用途：回归验证回归测试与测试辅助中的
# `concurrent_capacity_checked_create_never_exceeds_limit` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `_fill` →
# `asyncio.gather` → `manager.create_if_capacity` → `manager.active_count`。
# 分支与异常：
#   验证条件：`await manager.active_count() == 25`。
#   验证条件：`sum((record is not None for record in (first, second))) == 1`。
@pytest.mark.asyncio
async def test_concurrent_capacity_checked_create_never_exceeds_limit(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    await _fill(manager, 24)

    first, second = await asyncio.gather(
        manager.create_if_capacity(title="并发 A", summary="A", content="A"),
        manager.create_if_capacity(title="并发 B", summary="B", content="B"),
    )

    assert await manager.active_count() == 25
    assert sum(record is not None for record in (first, second)) == 1
