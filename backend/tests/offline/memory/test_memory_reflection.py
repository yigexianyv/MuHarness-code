
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.domain.memory import (
    MemoryManager,
    MemoryReflectionConfig,
    MemoryReflectionInput,
    PostRunMemoryReflector,
    ReflectionAction,
    register_memory_tools,
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
    ModelUsage,
    ToolCall,
)
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.result import AgentStopReason
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
#   usage：模型调用用量统计，类型 `ModelUsage | None`；默认 `None`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def _response(
    content: str,
    *,
    provider: str = "reflect",
    model: str = "reflection-model",
    usage: ModelUsage | None = None,
) -> ModelResponse:
    return ModelResponse(
        id="offline-response",
        provider=provider,
        model=model,
        message=Message(role=MessageRole.ASSISTANT, content=content),
        usage=usage or ModelUsage(),
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


# 函数说明：_input
# 用途：返回 `MemoryReflectionInput(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   recalled_memory_ids：记忆输入或配置值，类型 `tuple[str, ...]`；默认 `()`。
# 返回：类型 `MemoryReflectionInput`；返回 `MemoryReflectionInput(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryReflectionInput`。
def _input(
    *,
    recalled_memory_ids: tuple[str, ...] = (),
) -> MemoryReflectionInput:
    return MemoryReflectionInput(
        run_id="run-1",
        conversation_id="conversation-1",
        user_input="我们决定继续使用 Markdown 长期记忆。",
        final_answer="架构调整已经完成。",
        recalled_memory_ids=recalled_memory_ids,
    )


# 函数说明：_reflection_config
# 用途：在回归测试与测试辅助中处理 `_reflection_config`，通过 `values.update` 完成首个内
# 部处理步骤。
# 参数：
#   **overrides：额外关键字参数，按实现处理或转交。
# 返回：类型 `MemoryReflectionConfig`；返回
# `MemoryReflectionConfig(_env_file=None, **values)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`values.update` →
# `MemoryReflectionConfig`。
def _reflection_config(**overrides: object) -> MemoryReflectionConfig:
    values: dict[str, object] = {
        "provider": "reflect",
        "model": "reflection-model",
    }
    values.update(overrides)
    return MemoryReflectionConfig(
        _env_file=None,
        **values,
    )


# 函数说明：_manager
# 用途：在回归测试与测试辅助中处理 `_manager`，通过 `manager.initialize` 完成首个内部处
# 理步骤。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
#   max_active：允许同时活跃的数量上限，类型 `int`；默认 `25`。
# 返回：类型 `MemoryManager`；返回 `manager`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` → `manager.initialize`
# 。
async def _manager(path: Path, *, max_active: int = 25) -> MemoryManager:
    manager = MemoryManager(path, max_active=max_active)
    await manager.initialize()
    return manager


# 函数说明：test_reflection_noop_does_not_mutate_store
# 用途：回归验证回归测试与测试辅助中的 `reflection_noop_does_not_mutate_store` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `FakeAdapter` → `_config`
#  → `_response` → `PostRunMemoryReflector` → `_registry`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`proposal.decision is not None`。
#   验证条件：`proposal.decision.action is ReflectionAction.NONE`。
#   验证条件：`proposal.error is None`。
#   验证条件：`await manager.store.count_active() == 0`。
@pytest.mark.asyncio
async def test_reflection_noop_does_not_mutate_store(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    adapter = FakeAdapter(
        _config("reflect", "reflection-model"),
        [_response('{"action":"none","reason":"只有临时任务信息"}')],
    )
    reflector = PostRunMemoryReflector(
        _registry(reflect=adapter),
        config=_reflection_config(),
    )

    proposal = await reflector.decide(_input())

    assert proposal.decision is not None
    assert proposal.decision.action is ReflectionAction.NONE
    assert proposal.error is None
    assert await manager.store.count_active() == 0


# 函数说明：test_reflection_prompt_treats_confirmed_same_topic_delta_as_update
# 用途：回归验证回归测试与测试辅助中的
# `reflection_prompt_treats_confirmed_same_topic_delta_as_update` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeAdapter` → `_config` →
# `_response` → `PostRunMemoryReflector` → `_registry` → `_reflection_config`；另有 2 个
# 调用点。
# 分支与异常：
#   验证条件：`proposal.decision is not None`。
#   验证条件：`proposal.decision.action is ReflectionAction.UPDATE`。
#   验证条件：`proposal.input_json is not None`。
#   验证条件：`'刚完成的新规则' in proposal.input_json`。
@pytest.mark.asyncio
async def test_reflection_prompt_treats_confirmed_same_topic_delta_as_update(
) -> None:
    adapter = FakeAdapter(
        _config("reflect", "reflection-model"),
        [
            _response(
                '{"action":"update","memory_id":"M001",'
                '"title":"普通记忆容量","summary":"容量维护与并发规则",'
                '"content":"上限 25 条；归档前校验 revision。",'
                '"reason":"用户明确确认了同主题的新增规则"}'
            )
        ],
    )
    reflector = PostRunMemoryReflector(
        _registry(reflect=adapter),
        config=_reflection_config(capture_raw_io=True),
    )
    reflection_input = MemoryReflectionInput(
        run_id="run-update",
        conversation_id="conversation-1",
        user_input=(
            "我们刚完成的新规则是：归档前必须校验候选 revision，并且更新 "
            "Recall Cue。"
        ),
        final_answer="当前完整规则已经说明。",
        recalled_memory_ids=("M001",),
    )

    proposal = await reflector.decide(reflection_input)

    assert proposal.decision is not None
    assert proposal.decision.action is ReflectionAction.UPDATE
    assert proposal.input_json is not None
    assert "刚完成的新规则" in proposal.input_json
    assert proposal.raw_output is not None
    prompt = adapter.requests[0].messages[0].content or ""
    assert "同一主题的新耐久规则优先 update" in prompt
    assert "不因它只是补充、未否定旧规则就选择 create 或 none" in prompt
    assert "不要求本轮修改代码或文件" in prompt


# 函数说明：test_reflection_prompt_never_uses_ordinary_memory_as_core_fallback
# 用途：回归验证回归测试与测试辅助中的
# `reflection_prompt_never_uses_ordinary_memory_as_core_fallback` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeAdapter` → `_config` →
# `_response` → `PostRunMemoryReflector` → `_registry` → `_reflection_config`；另有 2 个
# 调用点。
# 分支与异常：
#   验证条件：`proposal.decision is not None`。
#   验证条件：`proposal.decision.action is ReflectionAction.NONE`。
#   验证条件：`'属于 Core，此处返回 none' in prompt`。
#   验证条件：`'主 Agent 未调用 core_memory_update' in prompt`。
@pytest.mark.asyncio
async def test_reflection_prompt_never_uses_ordinary_memory_as_core_fallback(
) -> None:
    adapter = FakeAdapter(
        _config("reflect", "reflection-model"),
        [_response('{"action":"none","reason":"这是 Core 级全局偏好"}')],
    )
    reflector = PostRunMemoryReflector(
        _registry(reflect=adapter),
        config=_reflection_config(),
    )
    reflection_input = MemoryReflectionInput(
        run_id="run-core-preference",
        conversation_id="conversation-1",
        user_input="以后所有回答都先给结论，再解释原因。",
        final_answer="好的，我记住了。",
    )

    proposal = await reflector.decide(reflection_input)

    assert proposal.decision is not None
    assert proposal.decision.action is ReflectionAction.NONE
    prompt = adapter.requests[0].messages[0].content or ""
    assert "属于 Core，此处返回 none" in prompt
    assert "主 Agent 未调用 core_memory_update" in prompt
    assert "未找到按需工具或保存失败" in prompt
    assert "不能用普通记忆 create/update 兜底或备份 Core 信息" in prompt


# 函数说明：test_reflection_raw_io_is_disabled_by_default
# 用途：回归验证回归测试与测试辅助中的 `reflection_raw_io_is_disabled_by_default` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeAdapter` → `_config` →
# `_response` → `PostRunMemoryReflector` → `_registry` → `_reflection_config`；另有 2 个
# 调用点。
# 分支与异常：
#   验证条件：`proposal.input_json is None`。
#   验证条件：`proposal.raw_output is None`。
@pytest.mark.asyncio
async def test_reflection_raw_io_is_disabled_by_default() -> None:
    adapter = FakeAdapter(
        _config("reflect", "reflection-model"),
        [_response('{"action":"none","reason":"没有耐久变化"}')],
    )
    reflector = PostRunMemoryReflector(
        _registry(reflect=adapter),
        config=_reflection_config(),
    )

    proposal = await reflector.decide(_input())

    assert proposal.input_json is None
    assert proposal.raw_output is None


# 函数说明：test_reflection_create_uses_manager_and_rebuilds_index
# 用途：回归验证回归测试与测试辅助中的
# `reflection_create_uses_manager_and_rebuilds_index` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `FakeAdapter` → `_config`
#  → `_response` → `_registry` → `PostRunMemoryReflector`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`record is not None`。
#   验证条件：`'Post-Run Reflector' in record.content`。
#   验证条件：`'MuHarness 使用 Markdown Memory' in (await manager.index.load() or '')`。
@pytest.mark.asyncio
async def test_reflection_create_uses_manager_and_rebuilds_index(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    decision = (
        '{"action":"create","title":"Memory 架构决定",'
        '"summary":"MuHarness 使用 Markdown Memory 与运行后反思",'
        '"content":"普通长期记忆由 Post-Run Reflector 沉淀。",'
        '"reason":"这是跨会话仍有价值的架构决定"}'
    )
    adapter = FakeAdapter(
        _config("reflect", "reflection-model"),
        [_response(decision)],
    )
    main = FakeAdapter(
        _config("main", "main-model"),
        [_response("主任务完成", provider="main", model="main-model")],
    )
    registry = _registry(main=main, reflect=adapter)
    reflector = PostRunMemoryReflector(
        registry,
        config=_reflection_config(),
    )
    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="main",
        memory_manager=manager,
        memory_reflector=reflector,
    )

    result = await runtime.run("完成记忆架构调整")

    assert result.ok is True
    record = await manager.store.load("M001")
    assert record is not None
    assert "Post-Run Reflector" in record.content
    assert "MuHarness 使用 Markdown Memory" in (
        await manager.index.load() or ""
    )


# 函数说明：test_reflection_update_decision_does_not_write_store
# 用途：回归验证回归测试与测试辅助中的 `reflection_update_decision_does_not_write_store`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `(manager.memory_dir / 'INDEX.md').write_text` → `FakeAdapter` → `_config` →
# `_response`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`proposal.decision is not None`。
#   验证条件：`proposal.decision.action is ReflectionAction.UPDATE`。
#   验证条件：`proposal.decision.memory_id == record.id`。
#   验证条件：`updated is not None`。
# 副作用与资源：
#   文件或资源访问：`(manager.memory_dir / 'INDEX.md').write_text`。
@pytest.mark.asyncio
async def test_reflection_update_decision_does_not_write_store(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    record = await manager.create(
        title="Memory 架构决定",
        summary="Markdown Memory 架构",
        content="旧决定",
    )
    (manager.memory_dir / "INDEX.md").write_text("stale", encoding="utf-8")
    decision = (
        '{"action":"update","memory_id":"m001",'
        '"title":"Memory 架构决定",'
        '"summary":"普通记忆由运行后反思统一沉淀",'
        '"content":"新决定：普通记忆改由 Post-Run Reflector 写入。",'
        '"reason":"本轮改变了已有架构决定"}'
    )
    adapter = FakeAdapter(
        _config("reflect", "reflection-model"),
        [_response(decision)],
    )
    reflector = PostRunMemoryReflector(
        _registry(reflect=adapter),
        config=_reflection_config(),
    )

    proposal = await reflector.decide(
        _input(recalled_memory_ids=(record.id,))
    )

    assert proposal.decision is not None
    assert proposal.decision.action is ReflectionAction.UPDATE
    assert proposal.decision.memory_id == record.id
    updated = await manager.store.load(record.id)
    assert updated is not None
    assert updated.content == "旧决定"
    assert await manager.index.load() == "stale"


# 函数说明：test_reflection_update_requires_current_run_memory_read
# 用途：回归验证回归测试与测试辅助中的
# `reflection_update_requires_current_run_memory_read` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `(manager.memory_dir / 'active' / f'{record.id}.md').read_text` → `FakeAdapter` →
# `_config` → `_response`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`failed.reflection_memory_id == record.id`。
#   验证条件：`failed.reflection_error is not None`。
#   验证条件：`'memory_read success' in failed.reflection_error`。
# 副作用与资源：
#   文件或资源访问：`(manager.memory_dir / 'active' / f'{record.id}.md').read_text`。
@pytest.mark.asyncio
async def test_reflection_update_requires_current_run_memory_read(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    record = await manager.create(
        title="已有决定",
        summary="只能从 Index 看到的 cue",
        content="不能被猜测覆盖的完整正文",
    )
    before = (manager.memory_dir / "active" / f"{record.id}.md").read_text(
        encoding="utf-8"
    )
    adapter = FakeAdapter(
        _config("reflect", "reflection-model"),
        [
            _response(
                '{"action":"update","memory_id":"M001",'
                '"title":"已有决定",'
                '"summary":"模型根据 cue 猜测的错误更新",'
                '"content":"模型根据 cue 猜测的替代正文",'
                '"reason":"错误地尝试更新"}'
            )
        ],
    )
    main = FakeAdapter(
        _config("main", "main-model"),
        [_response("任务完成", provider="main", model="main-model")],
    )
    registry = _registry(main=main, reflect=adapter)
    reflector = PostRunMemoryReflector(
        registry,
        config=_reflection_config(),
    )
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="main",
        memory_manager=manager,
        memory_reflector=reflector,
    )

    result = await runtime.run("没有读取旧记忆", event_handler=events)

    assert result.ok is True
    failed = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_REFLECTION_FAILED
    )
    assert failed.reflection_memory_id == record.id
    assert failed.reflection_error is not None
    assert "memory_read success" in failed.reflection_error
    assert (
        manager.memory_dir / "active" / f"{record.id}.md"
    ).read_text(encoding="utf-8") == before


# 函数说明：test_reflection_failures_are_isolated
# 用途：回归验证回归测试与测试辅助中的 `reflection_failures_are_isolated` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`。
#   delay_seconds：时间间隔或限额，单位为秒，类型 `float`。
#   timeout_seconds：等待或执行超时，单位为秒，类型 `float`。
#   error_text：错误文本输入或配置值，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `FakeAdapter` → `_config`
#  → `PostRunMemoryReflector` → `_registry` → `_reflection_config`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`proposal.error is not None and error_text in proposal.error`。
#   验证条件：`await manager.store.count_active() == 0`。
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "delay_seconds", "timeout_seconds", "error_text"),
    [
        ([RuntimeError("provider unavailable")], 0.0, 1.0, "provider unavailable"),
        ([_response("not-json")], 0.0, 1.0, "ValidationError"),
        ([_response('{"action":"none","reason":"noop"}')], 0.05, 0.01, "TimeoutError"),
    ],
)
async def test_reflection_failures_are_isolated(
    tmp_path: Path,
    responses: Sequence[ModelResponse | Exception],
    delay_seconds: float,
    timeout_seconds: float,
    error_text: str,
) -> None:
    manager = await _manager(tmp_path / "memory")
    adapter = FakeAdapter(
        _config("reflect", "reflection-model"),
        responses,
        delay_seconds=delay_seconds,
    )
    reflector = PostRunMemoryReflector(
        _registry(reflect=adapter),
        config=_reflection_config(
            timeout_seconds=timeout_seconds,
            max_attempts=1,
        ),
    )

    proposal = await reflector.decide(_input())

    assert proposal.error is not None and error_text in proposal.error
    assert await manager.store.count_active() == 0


# 函数说明：test_reflection_retries_truncated_json_within_one_timeout
# 用途：回归验证回归测试与测试辅助中的
# `reflection_retries_truncated_json_within_one_timeout` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelUsage` → `FakeAdapter` →
# `_config` → `_response` → `PostRunMemoryReflector` → `_registry`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`proposal.error is None`。
#   验证条件：`proposal.decision is not None`。
#   验证条件：`proposal.decision.action is ReflectionAction.NONE`。
#   验证条件：`proposal.attempts == 2`。
@pytest.mark.asyncio
async def test_reflection_retries_truncated_json_within_one_timeout() -> None:

    first_usage = ModelUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    second_usage = ModelUsage(input_tokens=11, output_tokens=7, total_tokens=18)
    adapter = FakeAdapter(
        _config("reflect", "reflection-model"),
        [
            _response(
                '{"action":"create","title":"未结束',
                usage=first_usage,
            ),
            _response(
                '{"action":"none","reason":"没有耐久变化"}',
                usage=second_usage,
            ),
        ],
    )
    reflector = PostRunMemoryReflector(
        _registry(reflect=adapter),
        config=_reflection_config(capture_raw_io=True),
    )

    proposal = await reflector.decide(_input())

    assert proposal.error is None
    assert proposal.decision is not None
    assert proposal.decision.action is ReflectionAction.NONE
    assert proposal.attempts == 2
    assert len(adapter.requests) == 2
    assert proposal.usage.input_tokens == 21
    assert proposal.usage.output_tokens == 27
    assert proposal.usage.total_tokens == 48
    retry_content = adapter.requests[1].messages[1].content or ""
    assert "previous response was empty, invalid, or truncated" in retry_content


# 函数说明：test_runtime_final_answer_triggers_independent_reflection_model
# 用途：回归验证回归测试与测试辅助中的
# `runtime_final_answer_triggers_independent_reflection_model` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `FakeAdapter` → `_config`
#  → `_response` → `ModelUsage` → `_registry`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`len(main.requests) == 1`。
#   验证条件：`len(reflect.requests) == 1`。
#   验证条件：`request.model == 'cheap-memory-model'`。
@pytest.mark.asyncio
async def test_runtime_final_answer_triggers_independent_reflection_model(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    main = FakeAdapter(
        _config("main", "main-model"),
        [_response("主任务完成", provider="main", model="main-model")],
    )
    reflection_usage = ModelUsage(input_tokens=9, output_tokens=4, total_tokens=13)
    reflect = FakeAdapter(
        _config("reflect", "adapter-default"),
        [
            _response(
                '{"action":"none","reason":"没有耐久的新信息"}',
                usage=reflection_usage,
            )
        ],
    )
    registry = _registry(main=main, reflect=reflect)
    reflector = PostRunMemoryReflector(
        registry,
        config=_reflection_config(
            model="cheap-memory-model",
            max_output_tokens=321,
            temperature=0.2,
        ),
        default_provider="main",
        default_model="main-model",
    )
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="main",
        model="main-model",
        memory_manager=manager,
        memory_reflector=reflector,
    )

    result = await runtime.run("完成当前任务", event_handler=events)

    assert result.ok is True
    assert len(main.requests) == 1
    assert len(reflect.requests) == 1
    request = reflect.requests[0]
    assert request.model == "cheap-memory-model"
    assert request.max_output_tokens == 321
    assert request.temperature == 0.2
    assert request.tools == ()
    assert [event.type for event in events.events][-3:] == [
        AgentEventType.AGENT_COMPLETED,
        AgentEventType.MEMORY_REFLECTION_STARTED,
        AgentEventType.MEMORY_REFLECTION_COMPLETED,
    ]
    completed = events.events[-1]
    assert completed.provider == "reflect"
    assert completed.model == "cheap-memory-model"
    assert completed.reflection_action == "none"
    assert completed.usage == reflection_usage


# 函数说明：test_runtime_gate_skips_obvious_smalltalk_without_calling_reflector
# 用途：回归验证回归测试与测试辅助中的
# `runtime_gate_skips_obvious_smalltalk_without_calling_reflector` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `FakeAdapter` → `_config`
#  → `_response` → `_registry` → `InMemoryEventHandler`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`reflect.requests == []`。
#   验证条件：`skipped.type is AgentEventType.MEMORY_REFLECTION_SKIPPED`。
#   验证条件：`skipped.reflection_skip_reason == 'gate:smalltalk'`。
@pytest.mark.asyncio
async def test_runtime_gate_skips_obvious_smalltalk_without_calling_reflector(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    main = FakeAdapter(
        _config("main", "main-model"),
        [_response("你好", provider="main", model="main-model")],
    )
    reflect = FakeAdapter(
        _config("reflect", "reflection-model"),
        [_response('{"action":"none","reason":"unused"}')],
    )
    registry = _registry(main=main, reflect=reflect)
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
    )

    result = await runtime.run("你好！", event_handler=events)

    assert result.ok is True
    assert reflect.requests == []
    skipped = events.events[-1]
    assert skipped.type is AgentEventType.MEMORY_REFLECTION_SKIPPED
    assert skipped.reflection_skip_reason == "gate:smalltalk"


# 函数说明：test_runtime_successful_memory_read_authorizes_reflection_update
# 用途：回归验证回归测试与测试辅助中的
# `runtime_successful_memory_read_authorizes_reflection_update` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `(manager.memory_dir / 'INDEX.md').write_text` → `FakeAdapter` → `_config` →
# `ModelResponse`；另有 12 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`updated is not None`。
#   验证条件：`updated.title == '项目长期记忆方向'`。
#   验证条件：`updated.summary == '普通记忆由运行后反思模型沉淀'`。
# 副作用与资源：
#   文件或资源访问：`(manager.memory_dir / 'INDEX.md').write_text`。
@pytest.mark.asyncio
async def test_runtime_successful_memory_read_authorizes_reflection_update(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    record = await manager.create(
        title="项目方向",
        summary="MuHarness 的长期记忆方向",
        content="旧方向",
    )
    (manager.memory_dir / "INDEX.md").write_text("stale", encoding="utf-8")
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
                '"content":"新方向：普通记忆由运行后反思模型沉淀。",'
                '"reason":"本轮更新了已有架构决定"}'
            )
        ],
    )
    registry = _registry(main=main, reflect=reflect)
    tools = ToolRegistry()
    register_memory_tools(tools, manager)
    runtime = AgentRuntime(
        registry,
        tools,
        provider="main",
        memory_manager=manager,
        memory_reflector=PostRunMemoryReflector(
            registry,
            config=_reflection_config(),
        ),
    )

    result = await runtime.run("读取旧决定并完成调整")

    assert result.ok is True
    updated = await manager.store.load(record.id)
    assert updated is not None
    assert updated.title == "项目长期记忆方向"
    assert updated.summary == "普通记忆由运行后反思模型沉淀"
    assert "运行后反思模型" in updated.content
    assert "普通记忆由运行后反思模型沉淀" in (
        await manager.index.load() or ""
    )


# 函数说明：test_runtime_rejects_reflection_update_after_concurrent_change
# 用途：回归验证回归测试与测试辅助中的
# `runtime_rejects_reflection_update_after_concurrent_change` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `FakeAdapter` → `_config` → `ModelResponse` → `Message`；另有 12 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`current is not None`。
#   验证条件：`current.content == '另一个 Run 已经更新'`。
#   验证条件：`failed.reflection_error is not None`。
@pytest.mark.asyncio
async def test_runtime_rejects_reflection_update_after_concurrent_change(
    tmp_path: Path,
) -> None:
    manager = await _manager(tmp_path / "memory")
    record = await manager.create(
        title="项目方向",
        summary="旧方向",
        content="初始正文",
    )

    # 函数说明：
    # test_runtime_rejects_reflection_update_after_concurrent_change.concurrent_update
    # 用途：更新`concurrent`，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`manager.update`。
    # 闭包依赖：从外层读取 `manager`、`record`。
    async def concurrent_update() -> None:
        await manager.update(
            record.id,
            title="并发更新标题",
            summary="并发更新 cue",
            content="另一个 Run 已经更新",
            reason="并发测试",
        )

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
            _response("主任务完成", provider="main", model="main-model"),
        ],
    )
    reflect = FakeAdapter(
        _config("reflect", "reflection-model"),
        [
            _response(
                '{"action":"update","memory_id":"M001",'
                '"title":"过期标题","summary":"过期 cue",'
                '"content":"基于旧 revision 的正文",'
                '"reason":"模拟过期更新"}'
            )
        ],
        before_return=concurrent_update,
    )
    registry = _registry(main=main, reflect=reflect)
    tools = ToolRegistry()
    register_memory_tools(tools, manager)
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        registry,
        tools,
        provider="main",
        memory_manager=manager,
        memory_reflector=PostRunMemoryReflector(
            registry,
            config=_reflection_config(),
        ),
    )

    result = await runtime.run("读取并更新已有方向", event_handler=events)

    assert result.ok is True
    current = await manager.store.load(record.id)
    assert current is not None
    assert current.content == "另一个 Run 已经更新"
    failed = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_REFLECTION_FAILED
    )
    assert failed.reflection_error is not None
    assert "revision conflict" in failed.reflection_error


# 函数说明：test_reflection_provider_only_uses_that_adapters_default_model
# 用途：回归验证回归测试与测试辅助中的
# `reflection_provider_only_uses_that_adapters_default_model` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeAdapter` → `_config` →
# `_response` → `_registry` → `PostRunMemoryReflector` → `MemoryReflectionConfig`；另有
# 2 个调用点。
# 分支与异常：
#   验证条件：`proposal.error is None`。
#   验证条件：`proposal.model == 'provider-default-model'`。
#   验证条件：`reflect.requests[0].model == 'provider-default-model'`。
@pytest.mark.asyncio
async def test_reflection_provider_only_uses_that_adapters_default_model(
) -> None:
    reflect = FakeAdapter(
        _config("reflect", "provider-default-model"),
        [_response('{"action":"none","reason":"没有新记忆"}')],
    )
    registry = _registry(reflect=reflect)
    reflector = PostRunMemoryReflector(
        registry,
        config=MemoryReflectionConfig(
            _env_file=None,
            provider="reflect",
            model=None,
        ),
        default_provider="main",
        default_model="main-model-must-not-leak",
    )

    proposal = await reflector.decide(_input())

    assert proposal.error is None
    assert proposal.model == "provider-default-model"
    assert reflect.requests[0].model == "provider-default-model"


# 函数说明：test_runtime_reflection_failure_does_not_change_success_result
# 用途：回归验证回归测试与测试辅助中的
# `runtime_reflection_failure_does_not_change_success_result` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   failure：传给 `isinstance` 的输入，类型 `Exception`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `FakeAdapter` → `_config`
#  → `_response` → `_registry` → `InMemoryEventHandler`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.content == '最终答案'`。
#   验证条件：`event_types.index(AgentEventType.AGENT_COMPLETED) < event_types.index(
# AgentEventType.MEMORY_REFLECTION_FAILED)`。
#   验证条件：`failed.reflection_error is not None`。
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("provider failed"), ValueError("bad")],
)
async def test_runtime_reflection_failure_does_not_change_success_result(
    tmp_path: Path,
    failure: Exception,
) -> None:
    manager = await _manager(tmp_path / "memory")
    main = FakeAdapter(
        _config("main", "main-model"),
        [_response("最终答案", provider="main", model="main-model")],
    )
    reflection_response: ModelResponse | Exception
    if isinstance(failure, ValueError):
        reflection_response = _response("invalid-json")
    else:
        reflection_response = failure
    reflect = FakeAdapter(
        _config("reflect", "reflection-model"),
        [reflection_response],
    )
    registry = _registry(main=main, reflect=reflect)
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
    )

    result = await runtime.run("正常任务", event_handler=events)

    assert result.ok is True
    assert result.content == "最终答案"
    event_types = [event.type for event in events.events]
    assert event_types.index(AgentEventType.AGENT_COMPLETED) < event_types.index(
        AgentEventType.MEMORY_REFLECTION_FAILED
    )
    failed = next(
        event
        for event in events.events
        if event.type is AgentEventType.MEMORY_REFLECTION_FAILED
    )
    assert failed.reflection_error is not None


# 函数说明：test_runtime_model_error_skips_reflection
# 用途：回归验证回归测试与测试辅助中的 `runtime_model_error_skips_reflection` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `FakeAdapter` → `_config`
#  → `_response` → `_registry` → `InMemoryEventHandler`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.stop_reason is AgentStopReason.MODEL_ERROR`。
#   验证条件：`reflect.requests == []`。
#   验证条件：`events.events[-2].type is AgentEventType.AGENT_FAILED`。
#   验证条件：`events.events[-1].type is AgentEventType.MEMORY_REFLECTION_SKIPPED`。
@pytest.mark.asyncio
async def test_runtime_model_error_skips_reflection(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    main = FakeAdapter(
        _config("main", "main-model"),
        [RuntimeError("main failed")],
    )
    reflect = FakeAdapter(
        _config("reflect", "reflection-model"),
        [_response('{"action":"none","reason":"unused"}')],
    )
    registry = _registry(main=main, reflect=reflect)
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
    )

    result = await runtime.run("会失败", event_handler=events)

    assert result.stop_reason is AgentStopReason.MODEL_ERROR
    assert reflect.requests == []
    assert events.events[-2].type is AgentEventType.AGENT_FAILED
    assert events.events[-1].type is AgentEventType.MEMORY_REFLECTION_SKIPPED


# 函数说明：test_runtime_max_steps_skips_reflection
# 用途：回归验证回归测试与测试辅助中的 `runtime_max_steps_skips_reflection` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `FakeAdapter` → `_config`
#  → `ModelResponse` → `Message` → `ToolCall`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`result.stop_reason is AgentStopReason.MAX_STEPS`。
#   验证条件：`reflect.requests == []`。
#   验证条件：`events.events[-2].type is AgentEventType.AGENT_FAILED`。
#   验证条件：`events.events[-1].type is AgentEventType.MEMORY_REFLECTION_SKIPPED`。
@pytest.mark.asyncio
async def test_runtime_max_steps_skips_reflection(tmp_path: Path) -> None:
    manager = await _manager(tmp_path / "memory")
    main = FakeAdapter(
        _config("main", "main-model"),
        [
            ModelResponse(
                id="tool-response",
                provider="main",
                model="main-model",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(ToolCall(id="missing-1", name="missing"),),
                ),
            )
        ],
    )
    reflect = FakeAdapter(
        _config("reflect", "reflection-model"),
        [_response('{"action":"none","reason":"unused"}')],
    )
    registry = _registry(main=main, reflect=reflect)
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="main",
        max_steps=1,
        memory_manager=manager,
        memory_reflector=PostRunMemoryReflector(
            registry,
            config=_reflection_config(),
        ),
    )

    result = await runtime.run("需要更多步骤", event_handler=events)

    assert result.stop_reason is AgentStopReason.MAX_STEPS
    assert reflect.requests == []
    assert events.events[-2].type is AgentEventType.AGENT_FAILED
    assert events.events[-1].type is AgentEventType.MEMORY_REFLECTION_SKIPPED
