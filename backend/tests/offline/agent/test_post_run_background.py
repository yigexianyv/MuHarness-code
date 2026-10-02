
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.application import Application
from app.domain.memory import (
    MemoryMaintenanceConfig,
    MemoryMaintenanceReflector,
    MemoryManager,
    MemoryReflectionConfig,
    PostRunMemoryReflector,
    register_memory_tools,
)
from app.domain.skill_learning import SkillLearningSettings
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
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.post_run_processor import PostRunProcessor
from app.runtime.agent.runtime import AgentRuntime
from app.tools.registry import ToolRegistry


class FakeAdapter(ModelAdapter):

    # 函数说明：FakeAdapter.__init__
    # 用途：初始化 FakeAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   responses：预设的模型或服务响应序列，类型 `list[ModelResponse | Exception]`。
    #   gate：`gate`输入或配置值，类型 `asyncio.Event | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.gate`、`self.requests`。
    def __init__(
        self,
        config: ProviderConfig,
        responses: list[ModelResponse | Exception],
        *,
        gate: asyncio.Event | None = None,
    ) -> None:
        super().__init__(config)
        self.responses = list(responses)
        self.gate = gate
        self.requests: list[ModelRequest] = []

    # 函数说明：FakeAdapter.complete
    # 用途：完成FakeAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.gate.wait` →
    # `self.responses.pop`。
    # 分支与异常：
    #   当 `isinstance(response, Exception)` 时，抛出 `response`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    # 函数说明：FakeAdapter.close
    # 用途：关闭FakeAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：_config
# 用途：返回 `ProviderConfig(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   provider：模型或搜索服务商，类型 `str`。
#   model：模型名称，类型 `str`。
# 返回：类型 `ProviderConfig`；返回 `ProviderConfig(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr`。
def _config(provider: str, model: str) -> ProviderConfig:
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
#   provider：模型或搜索服务商，类型 `str`；默认 `'reflect'`。
#   model：模型名称，类型 `str`；默认 `'reflection-model'`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def _response(
    content: str,
    *,
    provider: str = "reflect",
    model: str = "reflection-model",
) -> ModelResponse:
    return ModelResponse(
        id="offline-response",
        provider=provider,
        model=model,
        message=Message(role=MessageRole.ASSISTANT, content=content),
        usage=ModelUsage(),
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


# 函数说明：_reflection_config
# 用途：返回 `MemoryReflectionConfig(_env_file=None, **overrides)`，提供 回归测试与测试
# 辅助 的派生值。
# 参数：
#   **overrides：额外关键字参数，按实现处理或转交。
# 返回：类型 `MemoryReflectionConfig`；返回
# `MemoryReflectionConfig(_env_file=None, **overrides)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryReflectionConfig`。
def _reflection_config(**overrides: object) -> MemoryReflectionConfig:
    return MemoryReflectionConfig(_env_file=None, **overrides)


# 函数说明：_create_json
# 用途：创建JSON 数据，供回归测试与测试辅助使用。
# 参数：
#   title：面向用户的标题，类型 `str`；默认 `'新项目决定'`。
#   content：内容正文，类型 `str`；默认 `'决定正文'`。
# 返回：类型 `str`；返回 `f'{{"action":"create","title":"{title}","summary":"决定摘要","
# content":"{content}","reason":"…`。
def _create_json(title: str = "新项目决定", content: str = "决定正文") -> str:
    return (
        f'{{"action":"create","title":"{title}","summary":"决定摘要",'
        f'"content":"{content}","reason":"值得长期记忆"}}'
    )


# 函数说明：_runtime
# 用途：处理回归测试与测试辅助中的 `_runtime` 数据；结果及边界条件见下方说明。
# 参数：
#   manager：当前业务管理器，类型 `MemoryManager`。
#   main_responses：传给 `FakeAdapter` 的输入，类型 `list[ModelResponse | Exception]`。
#   reflect_responses：传给 `FakeAdapter` 的输入，类型 `list[ModelResponse | Exception]`
# 。
#   processor：`processor`输入或配置值，类型 `PostRunProcessor`。
#   gate：`gate`输入或配置值，类型 `asyncio.Event | None`；默认 `None`。
#   tools：可用工具定义或工具实例集合，类型 `ToolRegistry | None`；默认 `None`。
# 返回：类型 `tuple[AgentRuntime, InMemoryEventHandler, FakeAdapter]`；返回
# `(runtime, events, reflect)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeAdapter` → `_config` →
# `_registry` → `InMemoryEventHandler` → `AgentRuntime` → `ToolRegistry`；另有 2 个调用
# 点。
def _runtime(
    manager: MemoryManager,
    *,
    main_responses: list[ModelResponse | Exception],
    reflect_responses: list[ModelResponse | Exception],
    processor: PostRunProcessor,
    gate: asyncio.Event | None = None,
    tools: ToolRegistry | None = None,
) -> tuple[AgentRuntime, InMemoryEventHandler, FakeAdapter]:
    main = FakeAdapter(
        _config("main", "main-model"),
        main_responses,
    )
    reflect = FakeAdapter(
        _config("reflect", "reflection-model"),
        reflect_responses,
        gate=gate,
    )
    registry = _registry(main=main, reflect=reflect)
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        registry,
        tools or ToolRegistry(),
        provider="main",
        memory_manager=manager,
        memory_reflector=PostRunMemoryReflector(
            registry,
            config=_reflection_config(),
            default_provider="reflect",
            default_model="reflection-model",
        ),
        post_run_submit=processor.submit,
    )
    return runtime, events, reflect


# 函数说明：_wait_active_zero
# 用途：等待活跃项，供回归测试与测试辅助使用。
# 参数：
#   processor：`processor`输入或配置值，类型 `PostRunProcessor`。
#   timeout：等待超时配置，类型 `float`；默认 `4.0`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.get_running_loop` →
# `loop.time` → `asyncio.sleep`。
# 分支与异常：
#   当 `loop.time() > deadline` 时，抛出
# `AssertionError('background job did not finish in time')`。
async def _wait_active_zero(processor: PostRunProcessor, timeout: float = 4.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while processor.active_count > 0:
        if loop.time() > deadline:
            raise AssertionError("background job did not finish in time")
        await asyncio.sleep(0.01)




# 函数说明：test_run_completed_before_slow_background_reflection
# 用途：回归验证回归测试与测试辅助中的 `run_completed_before_slow_background_reflection`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `asyncio.Event` →
# `PostRunProcessor` → `_runtime` → `_response` → `_create_json`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`processor.active_count == 1`。
#   验证条件：`AgentEventType.AGENT_COMPLETED in types`。
#   验证条件：`AgentEventType.MEMORY_REFLECTION_STARTED not in types`。
@pytest.mark.asyncio
async def test_run_completed_before_slow_background_reflection(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    gate = asyncio.Event()
    processor = PostRunProcessor()
    runtime, events, _ = _runtime(
        manager,
        main_responses=[_response("最终答案", provider="main", model="main-model")],
        reflect_responses=[_response(_create_json())],
        processor=processor,
        gate=gate,
    )

    result = await runtime.run("完成当前任务", event_handler=events)

    assert result.ok is True
    assert processor.active_count == 1
    types = [event.type for event in events.events]
    assert AgentEventType.AGENT_COMPLETED in types
    assert AgentEventType.MEMORY_REFLECTION_STARTED not in types

    gate.set()
    await _wait_active_zero(processor)
    await processor.close()

    assert await manager.active_count() == 1
    types = [event.type for event in events.events]
    assert types.index(AgentEventType.AGENT_COMPLETED) < types.index(
        AgentEventType.MEMORY_REFLECTION_STARTED
    )
    assert AgentEventType.MEMORY_REFLECTION_COMPLETED in types




# 函数说明：test_background_reflection_update_preserves_revision_semantics
# 用途：回归验证回归测试与测试辅助中的
# `background_reflection_update_preserves_revision_semantics` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `ToolRegistry` → `register_memory_tools` → `PostRunProcessor` → `_update_runtime`；另
# 有 6 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`updated is not None`。
#   验证条件：`updated.title == '项目长期记忆方向'`。
#   验证条件：`'运行后反思模型沉淀' in updated.content`。
@pytest.mark.asyncio
async def test_background_reflection_update_preserves_revision_semantics(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    record = await manager.create(
        title="项目方向",
        summary="旧方向",
        content="初始正文",
    )
    manager_ref = manager
    tools = ToolRegistry()
    register_memory_tools(tools, manager)
    processor = PostRunProcessor()

    # 函数说明：
    # test_background_reflection_update_preserves_revision_semantics._update_runtime
    # 用途：更新执行环境，供回归测试与测试辅助使用。
    # 返回：类型 `AgentRuntime`；返回 `AgentRuntime(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeAdapter` → `_config` →
    # `ModelResponse` → `Message` → `ToolCall` → `_response`；另有 4 个调用点。
    # 闭包依赖：从外层读取 `manager_ref`、`processor`、`record`、`tools`。
    def _update_runtime() -> AgentRuntime:
        main = FakeAdapter(
            _config("main", "main-model"),
            [
                ModelResponse(
                    id="read-response",
                    provider="main",
                    model="main-model",
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        tool_calls=(
                            ToolCall(
                                id="read-memory-1",
                                name="memory_read",
                                arguments={"memory_id": record.id},
                            ),
                        ),
                    ),
                ),
                _response("已完成架构调整", provider="main", model="main-model"),
            ],
        )
        reflect = FakeAdapter(
            _config("reflect", "reflection-model"),
            [
                _response(
                    '{"action":"update","memory_id":"M001",'
                    '"title":"项目长期记忆方向",'
                    '"summary":"普通记忆由运行后反思模型沉淀",'
                    '"content":"新方向：运行后反思模型沉淀。",'
                    '"reason":"本轮更新了已有架构决定"}'
                )
            ],
        )
        registry = _registry(main=main, reflect=reflect)
        return AgentRuntime(
            registry,
            tools,
            provider="main",
            memory_manager=manager_ref,
            memory_reflector=PostRunMemoryReflector(
                registry,
                config=_reflection_config(),
                default_provider="reflect",
                default_model="reflection-model",
            ),
            post_run_submit=processor.submit,
        )

    runtime = _update_runtime()
    events = InMemoryEventHandler()
    result = await runtime.run("读取旧决定并完成调整", event_handler=events)

    assert result.ok is True
    await _wait_active_zero(processor)
    await processor.close()

    updated = await manager.store.load(record.id)
    assert updated is not None
    assert updated.title == "项目长期记忆方向"
    assert "运行后反思模型沉淀" in updated.content
    assert updated.revision > record.revision
    assert "运行后反思模型沉淀" in (await manager.index.load() or "")




# 函数说明：test_background_reflection_exception_keeps_run_completed
# 用途：回归验证回归测试与测试辅助中的
# `background_reflection_exception_keeps_run_completed` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `PostRunProcessor` →
# `_runtime` → `_response` → `runtime.run` → `_wait_active_zero`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`AgentEventType.AGENT_COMPLETED in types`。
#   验证条件：`AgentEventType.MEMORY_REFLECTION_FAILED in types`。
#   验证条件：`types.index(AgentEventType.AGENT_COMPLETED) < types.index(AgentEventType.
# MEMORY_REFLECTION_FAILED)`。
@pytest.mark.asyncio
async def test_background_reflection_exception_keeps_run_completed(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    processor = PostRunProcessor()
    runtime, events, _ = _runtime(
        manager,
        main_responses=[_response("最终答案", provider="main", model="main-model")],
        reflect_responses=[RuntimeError("reflector boom")],
        processor=processor,
    )

    result = await runtime.run("正常任务", event_handler=events)

    assert result.ok is True
    await _wait_active_zero(processor)
    await processor.close()
    types = [event.type for event in events.events]
    assert AgentEventType.AGENT_COMPLETED in types
    assert AgentEventType.MEMORY_REFLECTION_FAILED in types
    assert types.index(AgentEventType.AGENT_COMPLETED) < types.index(
        AgentEventType.MEMORY_REFLECTION_FAILED
    )




# 函数说明：test_maintenance_runs_after_background_reflection
# 用途：回归验证回归测试与测试辅助中的 `maintenance_runs_after_background_reflection` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.store.create` →
# `FakeAdapter` → `_config` → `_response` → `_create_json`；另有 14 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`await manager.active_count() <= manager.max_active`。
#   验证条件：`types.index(AgentEventType.AGENT_COMPLETED) < types.index(AgentEventType.
# MEMORY_MAINTENANCE_STARTED)`。
#   验证条件：`AgentEventType.MEMORY_MAINTENANCE_COMPLETED in types`。
@pytest.mark.asyncio
async def test_maintenance_runs_after_background_reflection(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    for index in range(1, manager.max_active + 2):
        await manager.store.create(
            title=f"记忆 {index}",
            summary=f"cue {index}",
            content=f"完整正文 {index}",
        )
    main = FakeAdapter(
        _config("main", "main-model"),
        [_response("主任务完成", provider="main", model="main-model")],
    )
    reflect = FakeAdapter(
        _config("reflect", "reflection-model"),
        [_response(_create_json("第 26 条", "新的长期正文"))],
    )
    maintain = FakeAdapter(
        _config("maintain", "maintenance-model"),
        [
            _response(
                '{"action":"archive","memory_id":"M001","reason":"已经过时"}',
                provider="maintain",
                model="maintenance-model",
            )
        ],
    )
    registry = _registry(main=main, reflect=reflect, maintain=maintain)
    events = InMemoryEventHandler()
    processor = PostRunProcessor()
    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="main",
        memory_manager=manager,
        memory_reflector=PostRunMemoryReflector(
            registry,
            config=_reflection_config(),
            default_provider="reflect",
            default_model="reflection-model",
        ),
        memory_maintenance_reflector=MemoryMaintenanceReflector(
            registry,
            config=MemoryMaintenanceConfig(
                _env_file=None,
                provider="maintain",
                model="maintenance-model",
            ),
            default_provider="reflect",
            default_model="reflection-model",
        ),
        post_run_submit=processor.submit,
    )

    result = await runtime.run("普通请求", event_handler=events)

    assert result.ok is True
    await _wait_active_zero(processor)
    await processor.close()
    assert await manager.active_count() <= manager.max_active
    types = [event.type for event in events.events]
    assert types.index(AgentEventType.AGENT_COMPLETED) < types.index(
        AgentEventType.MEMORY_MAINTENANCE_STARTED
    )
    assert AgentEventType.MEMORY_MAINTENANCE_COMPLETED in types




# 函数说明：test_post_run_processor_close_drains_and_cancels
# 用途：回归验证回归测试与测试辅助中的 `post_run_processor_close_drains_and_cancels` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`PostRunProcessor` → `asyncio.Event` →
#  `processor.submit` → `processor.close` → `done.is_set` → `processor2.submit`；另有 1
# 个调用点。
# 分支与异常：
#   验证条件：`processor.submit(fast) is True`。
#   验证条件：`done.is_set()`。
#   验证条件：`processor.active_count == 0`。
#   验证条件：`processor2.active_count == 0`。
@pytest.mark.asyncio
async def test_post_run_processor_close_drains_and_cancels() -> None:
    processor = PostRunProcessor(drain_timeout=0.2)
    done = asyncio.Event()

    # 函数说明：test_post_run_processor_close_drains_and_cancels.fast
    # 用途：在回归测试与测试辅助中处理 `fast`，通过 `asyncio.sleep` 完成首个内部处理步骤
    # 。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.sleep` → `done.set`。
    # 闭包依赖：从外层读取 `done`。
    async def fast() -> None:
        await asyncio.sleep(0.01)
        done.set()

    assert processor.submit(fast) is True
    await processor.close()
    assert done.is_set()
    assert processor.active_count == 0

    processor2 = PostRunProcessor(drain_timeout=0.05)
    gate = asyncio.Event()

    # 函数说明：test_post_run_processor_close_drains_and_cancels.stuck
    # 用途：在回归测试与测试辅助中处理 `stuck`，通过 `gate.wait` 完成首个内部处理步骤。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`gate.wait`。
    # 闭包依赖：从外层读取 `gate`。
    async def stuck() -> None:
        await gate.wait()

    processor2.submit(stuck)
    await processor2.close()
    assert processor2.active_count == 0
    assert processor2.submit(stuck) is False  


# 函数说明：test_post_run_processor_cancels_only_target_conversation
# 用途：回归验证回归测试与测试辅助中的
# `post_run_processor_cancels_only_target_conversation` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`PostRunProcessor` → `asyncio.Event` →
#  `processor.submit` → `started_a.wait` → `started_b.wait` →
# `processor.cancel_for_conversation`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`processor.submit(wait_a, conversation_id='conversation-a') is True`。
#   验证条件：`processor.submit(wait_b, conversation_id='conversation-b') is True`。
#   验证条件：`await processor.cancel_for_conversation('conversation-a') == 1`。
#   验证条件：`processor.active_count == 1`。
@pytest.mark.asyncio
async def test_post_run_processor_cancels_only_target_conversation() -> None:
    processor = PostRunProcessor()
    started_a = asyncio.Event()
    started_b = asyncio.Event()
    gate = asyncio.Event()

    # 函数说明：test_post_run_processor_cancels_only_target_conversation.wait_a
    # 用途：等待`a`，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`started_a.set` → `gate.wait`。
    # 闭包依赖：从外层读取 `gate`、`started_a`。
    async def wait_a() -> None:
        started_a.set()
        await gate.wait()

    # 函数说明：test_post_run_processor_cancels_only_target_conversation.wait_b
    # 用途：等待`b`，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`started_b.set` → `gate.wait`。
    # 闭包依赖：从外层读取 `gate`、`started_b`。
    async def wait_b() -> None:
        started_b.set()
        await gate.wait()

    assert processor.submit(wait_a, conversation_id="conversation-a") is True
    assert processor.submit(wait_b, conversation_id="conversation-b") is True
    await started_a.wait()
    await started_b.wait()

    assert await processor.cancel_for_conversation("conversation-a") == 1
    assert processor.active_count == 1

    gate.set()
    await processor.close()
    assert processor.active_count == 0


# 函数说明：test_application_close_drains_post_run
# 用途：回归验证回归测试与测试辅助中的 `application_close_drains_post_run` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` → `FakeAdapter` →
# `_config` → `Application` → `MemoryReflectionConfig` → `MemoryMaintenanceConfig`；另有
#  6 个调用点。
# 分支与异常：
#   验证条件：`app.post_run_processor.submit(slow) is True`。
#   验证条件：`app.post_run_processor.active_count == 1`。
#   验证条件：`app.post_run_processor.active_count == 0`。
@pytest.mark.asyncio
async def test_application_close_drains_post_run(tmp_path: Path) -> None:
    registry = _registry(
        main=FakeAdapter(_config("main", "main-model"), []),
    )
    app = Application(
        provider="main",
        model="main-model",
        database=tmp_path / "muharness.db",
        tasks_dir=tmp_path / "tasks",
        mcp_config=tmp_path / "mcp.json",
        memory_dir=tmp_path / "memory",
        skills_user_dir=tmp_path / "skills-user",
        skills_project_dir=tmp_path / "skills-project",
        registry=registry,
        memory_reflection_config=MemoryReflectionConfig(
            _env_file=None, enabled=False
        ),
        memory_maintenance_config=MemoryMaintenanceConfig(
            _env_file=None, enabled=False
        ),
        skill_learning_settings=SkillLearningSettings(
            _env_file=None,
            skill_learning_enabled=False,
            skill_learning_data_dir=tmp_path / "skill-learning",
        ),
    )
    await app.start()
    gate = asyncio.Event()

    # 函数说明：test_application_close_drains_post_run.slow
    # 用途：在回归测试与测试辅助中处理 `slow`，通过 `gate.wait` 完成首个内部处理步骤。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`gate.wait`。
    # 闭包依赖：从外层读取 `gate`。
    async def slow() -> None:
        await gate.wait()

    assert app.post_run_processor.submit(slow) is True
    assert app.post_run_processor.active_count == 1
    gate.set()
    await app.close()
    assert app.post_run_processor.active_count == 0




# 函数说明：test_next_run_not_blocked_by_previous_slow_reflection
# 用途：回归验证回归测试与测试辅助中的
# `next_run_not_blocked_by_previous_slow_reflection` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `asyncio.Event` →
# `PostRunProcessor` → `_runtime` → `_response` → `_create_json`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`first.ok is True`。
#   验证条件：`processor.active_count == 1`。
#   验证条件：`second.ok is True`。
#   验证条件：`processor.active_count == 2`。
@pytest.mark.asyncio
async def test_next_run_not_blocked_by_previous_slow_reflection(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    gate = asyncio.Event()
    processor = PostRunProcessor()
    runtime, events, _ = _runtime(
        manager,
        main_responses=[
            _response("第一轮回答", provider="main", model="main-model"),
            _response("第二轮回答", provider="main", model="main-model"),
        ],
        reflect_responses=[
            _response(_create_json("第一轮", "内容一")),
            _response(_create_json("第二轮", "内容二")),
        ],
        processor=processor,
        gate=gate,
    )

    first = await runtime.run("第一个任务", event_handler=events)
    assert first.ok is True
    assert processor.active_count == 1

    second = await runtime.run("第二个任务", event_handler=events)
    assert second.ok is True
    assert processor.active_count == 2

    gate.set()
    await _wait_active_zero(processor)
    await processor.close()
    assert await manager.active_count() == 2
