from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence

import pytest
from pydantic import SecretStr

from app.domain.memory import (
    CORE_MEMORY_MESSAGE_NAME,
    MEMORY_INDEX_MESSAGE_NAME,
    MEMORY_POLICY_MESSAGE_NAME,
    MemoryManager,
    register_memory_tools,
)
from app.domain.skills import (
    ACTIVE_SKILL_MESSAGE_NAME,
    SKILL_READ_TOOL_NAME,
    SkillContextProvider,
    SkillStore,
    register_skill_tools,
)
from app.domain.task import (
    TASK_CONTEXT_MESSAGE_NAME,
    FileTaskStore,
    TaskContextProvider,
    TaskStep,
    register_task_tools,
)
from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    RUNTIME_NOTICE_NAME,
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolDefinition,
    ToolPermission,
)
from app.runtime.agent.budget import RunBudgetConfig
from app.runtime.agent.events import (
    AgentEvent,
    AgentEventHandler,
    AgentEventType,
    InMemoryEventHandler,
)
from app.runtime.agent.result import AgentStopReason
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.checkpoint import (
    CHECKPOINT_CONTEXT_MESSAGE_NAME,
    CheckpointPhase,
    CheckpointStatus,
    SQLiteCheckpointStore,
)
from app.runtime.context import (
    ContextBudgetPolicy,
    ContextManager,
    ContextSettings,
    ContextSummarizer,
    ConversationReducer,
    ModelCapabilityRegistry,
    RollingConversationSummary,
    SummaryGenerationResult,
    TokenEstimator,
)
from app.tools.approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalScope,
    AutoApproveGate,
    DenyAllGate,
)
from app.tools.base import BaseTool
from app.tools.builtin.read_file import ReadFileTool
from app.tools.builtin.write_file import WriteFileTool
from app.tools.permissions.store import InMemoryPermissionRuleStore
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


class StreamingFakeModelAdapter(FakeModelAdapter):
    # 函数说明：StreamingFakeModelAdapter.complete_stream
    # 用途：完成事件流，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    #   on_text_delta：增量文本回调，类型 `Callable[[str], Awaitable[None]]`。
    #   on_reasoning_delta：增量推理文本回调，类型
    # `Callable[[str], Awaitable[None]] | None`；默认 `None`。
    # 返回：类型 `ModelResponse`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`on_text_delta` →
    # `on_reasoning_delta` → `self.responses.pop`。
    # 分支与异常：
    #   验证条件：`isinstance(response, ModelResponse)`。
    async def complete_stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> ModelResponse:
        self.requests.append(request)
        await on_text_delta("正在")
        await on_text_delta("完成")
        if on_reasoning_delta is not None:
            await on_reasoning_delta("先分析需求")
        response = self.responses.pop(0)
        assert isinstance(response, ModelResponse)
        return response


class FixedContextSummarizer(ContextSummarizer):
    # 函数说明：FixedContextSummarizer.summarize
    # 用途：生成摘要FixedContextSummarizer，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要，类型 `RollingConversationSummary | None`。
    #   messages：本次处理的消息序列，类型 `Sequence[Message]`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `SummaryGenerationResult`；返回 `SummaryGenerationResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SummaryGenerationResult` →
    # `RollingConversationSummary` → `ModelUsage`。
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        return SummaryGenerationResult(
            summary=RollingConversationSummary(current_objective="保留当前目标"),
            usage=ModelUsage(input_tokens=7, output_tokens=3, total_tokens=10),
        )


# 把请求按 Anthropic 适配器转换后渲染成线性文本，用于断言缓存前缀未被改动：
# 顶部 system 与工具 schema 不变，且上一次请求的渲染文本是这一次的前缀。
# 模拟 DeepSeek 思考模式的规则：含 text 块的 user 轮算“新用户消息”，只有最后一条
# 新用户消息之后的助手轮（工具循环内）才保留思考；所以末尾新增 text 块会改变前面
# 助手轮的渲染，前缀断言失败。tool_result 不写结尾标记，在其末尾追加内容不算改动。
def _anthropic_prefix_kept(previous: ModelRequest, current: ModelRequest) -> bool:
    from app.models.providers.anthropic import _anthropic_messages

    def render(request: ModelRequest) -> tuple[str | None, str]:
        system, turns = _anthropic_messages(request.messages)
        normalized = []
        for turn in turns:
            content = turn["content"]
            items = (
                content if isinstance(content, list)
                else [{"type": "text", "text": content}]
            )
            normalized.append((turn["role"], items))
        last_message = max(
            (
                index
                for index, (role, items) in enumerate(normalized)
                if role == "user" and any(i["type"] == "text" for i in items)
            ),
            default=-1,
        )
        parts: list[str] = []
        for index, (role, items) in enumerate(normalized):
            kind = role
            if role == "assistant":
                kind = "assistant:loop" if index > last_message else "assistant:done"
            parts.append(f"<turn {kind}>")
            for item in items:
                if item["type"] == "tool_result":
                    parts.append(f"<tool_result {item['tool_use_id']}>")
                    parts.append(str(item["content"]))
                elif item["type"] == "text":
                    parts.append("<text>")
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, sort_keys=True))
        return system, "".join(parts)

    system_before, text_before = render(previous)
    system_after, text_after = render(current)
    return (
        system_before == system_after
        and previous.tools == current.tools
        and text_after.startswith(text_before)
    )


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


class DeliveryTool(CountingTool):
    definition = ToolDefinition(
        name="deliver",
        description="Deliver the final result",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        closing_allowed=True,
    )


class ApprovalCountingTool(CountingTool):
    definition = ToolDefinition(
        name="approval_count",
        description="Count after approval",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        permission=ToolPermission.HUMAN_APPROVAL,
    )


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


class RememberRunGate(ApprovalGate):
    # 函数说明：RememberRunGate.request_approval
    # 用途：返回
    # `ApprovalResponse(decision=ApprovalDecision.APPROVED, scope=ApprovalScope.RUN)`，
    # 提供 RememberRunGate 的派生值。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；返回
    # `ApprovalResponse(decision=ApprovalDecision.APPROVED, scope=ApprovalScope.RUN)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalResponse`。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            decision=ApprovalDecision.APPROVED,
            scope=ApprovalScope.RUN,
        )


class FailingEventHandler(AgentEventHandler):
    # 函数说明：FailingEventHandler.emit
    # 用途：发出FailingEventHandler，供回归测试与测试辅助使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def emit(self, event: AgentEvent) -> None:
        raise RuntimeError("event sink unavailable")


class FakeMemoryManager:
    # 函数说明：FakeMemoryManager.__init__
    # 用途：初始化 FakeMemoryManager；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
    # 副作用与资源：
    #   更新对象字段：`self.messages`。
    def __init__(self) -> None:
        self.messages = (
            Message(
                role=MessageRole.SYSTEM,
                name=CORE_MEMORY_MESSAGE_NAME,
                content="# Core Memory\n\n用户偏好中文",
            ),
            Message(
                role=MessageRole.SYSTEM,
                name=MEMORY_INDEX_MESSAGE_NAME,
                content="# Long-term Memory Index\n\n[M001] demo\nCue: demo",
            ),
            Message(
                role=MessageRole.SYSTEM,
                name=MEMORY_POLICY_MESSAGE_NAME,
                content="Long-term memory is intentionally sparse.",
            ),
        )

    # 函数说明：FakeMemoryManager.context_messages
    # 用途：返回 `self.messages`，提供 FakeMemoryManager 的派生值。
    # 返回：类型 `tuple[Message, ...]`；返回 `self.messages`。
    async def context_messages(self) -> tuple[Message, ...]:
        return self.messages


class BlockingModelAdapter(ModelAdapter):
    # 函数说明：BlockingModelAdapter.__init__
    # 用途：初始化 BlockingModelAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super` →
    # `asyncio.Event`。
    # 副作用与资源：
    #   更新对象字段：`self.started`、`self.cancelled`。
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.started = asyncio.Event()
        self.cancelled = False

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


# 函数说明：model_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str | None`；默认 `None`。
#   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`；默认 `()`。
#   usage：模型调用用量统计，类型 `ModelUsage | None`；默认 `None`。
#   reasoning：`reasoning`输入或配置值，类型 `str | None`；默认 `None`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def model_response(
    *,
    content: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
    usage: ModelUsage | None = None,
    reasoning: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        id="fake-response",
        provider="fake",
        model="fake-model",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            reasoning=reasoning,
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


# 函数说明：test_runtime_emits_model_text_deltas_before_completed
# 用途：回归验证回归测试与测试辅助中的
# `runtime_emits_model_text_deltas_before_completed` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `StreamingFakeModelAdapter` → `model_response` → `ModelAdapterRegistry` →
# `ModelSettings`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`result.content == '正在完成'`。
#   验证条件：`deltas == ['正在', '完成']`。
#   验证条件：`event_types.index(AgentEventType.MODEL_STARTED) < event_types.index(
# AgentEventType.MODEL_OUTPUT_DELTA) < event_types.index(…`。
@pytest.mark.asyncio
async def test_runtime_emits_model_text_deltas_before_completed() -> None:
    config = ProviderConfig(
        provider="streaming-fake",
        model="streaming-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = StreamingFakeModelAdapter(config, [model_response(content="正在完成")])
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("streaming-fake", lambda _: adapter, config=config)
    handler = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="streaming-fake",
    ).run("开始", event_handler=handler)

    assert result.content == "正在完成"
    deltas = [
        event.delta
        for event in handler.events
        if event.type is AgentEventType.MODEL_OUTPUT_DELTA
    ]
    assert deltas == ["正在", "完成"]
    event_types = [event.type for event in handler.events]
    assert (
        event_types.index(AgentEventType.MODEL_STARTED)
        < event_types.index(AgentEventType.MODEL_OUTPUT_DELTA)
        < event_types.index(AgentEventType.MODEL_COMPLETED)
    )


# 函数说明：test_runtime_does_not_expose_provider_reasoning
# 用途：回归验证回归测试与测试辅助中的 `runtime_does_not_expose_provider_reasoning` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `StreamingFakeModelAdapter` → `model_response` → `ModelAdapterRegistry` →
# `ModelSettings`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`reasoning_deltas == []`。
#   验证条件：`completed.message is not None`。
#   验证条件：`completed.message.reasoning is None`。
#   验证条件：`all((message.reasoning is None for message in result.messages))`。
@pytest.mark.asyncio
async def test_runtime_does_not_expose_provider_reasoning() -> None:
    config = ProviderConfig(
        provider="streaming-fake",
        model="streaming-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = StreamingFakeModelAdapter(
        config,
        [model_response(content="完成", reasoning="内部完整推理")],
    )
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("streaming-fake", lambda _: adapter, config=config)
    handler = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="streaming-fake",
    ).run("开始", event_handler=handler)

    reasoning_deltas = [
        event.reasoning_delta
        for event in handler.events
        if event.type is AgentEventType.MODEL_REASONING_DELTA
    ]
    assert reasoning_deltas == []
    completed = next(
        event
        for event in handler.events
        if event.type is AgentEventType.MODEL_COMPLETED
    )
    assert completed.message is not None
    assert completed.message.reasoning is None
    assert all(message.reasoning is None for message in result.messages)


# 函数说明：test_runtime_removes_legacy_date_without_injecting_current_time
# 用途：回归验证回归测试与测试辅助中的
# `runtime_removes_legacy_date_without_injecting_current_time` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `Message` → `AgentRuntime(registry, ToolRegistry(), provider='fake').run` →
# `AgentRuntime` → `ToolRegistry`。
# 分支与异常：
#   验证条件：`'2026-08-04' not in (persisted_system.content or '')`。
#   验证条件：`not any((message.name == 'muharness_runtime_environment' or '当前本地日期
# 时间：' in (message.content or '') for message in…`。
#   验证条件：`result.messages[0] == old_system`。
@pytest.mark.asyncio
async def test_runtime_removes_legacy_date_without_injecting_current_time() -> None:
    registry, adapter = fake_registry([model_response(content="完成")])
    old_system = Message(
        role=MessageRole.SYSTEM,
        content="你是助理。当前日期是 2026-08-04。请准确回答。",
    )

    result = await AgentRuntime(registry, ToolRegistry(), provider="fake").run(
        "明天是几号？",
        history=(old_system,),
    )

    request = adapter.requests[0]
    persisted_system = request.messages[0]
    assert "2026-08-04" not in (persisted_system.content or "")
    assert not any(
        message.name == "muharness_runtime_environment"
        or "当前本地日期时间：" in (message.content or "")
        for message in request.messages
    )
    assert result.messages[0] == old_system


# 函数说明：test_runtime_system_prompt_is_request_only_and_deduplicated
# 用途：回归验证回归测试与测试辅助中的
# `runtime_system_prompt_is_request_only_and_deduplicated` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `Message` → `AgentRuntime(registry, ToolRegistry(), provider='fake', system_prompt=…`
# → `AgentRuntime` → `ToolRegistry`。
# 分支与异常：
#   验证条件：`sum((message.role is MessageRole.SYSTEM and message.content == prompt for
#  message in adapter.requests[0].messages)) == 1`。
#   验证条件：`result.messages[:len(history)] == history`。
#   验证条件：`sum((message.role is MessageRole.SYSTEM and message.content == prompt for
#  message in result.messages)) == 1`。
@pytest.mark.asyncio
async def test_runtime_system_prompt_is_request_only_and_deduplicated() -> None:

    prompt = "你是 MuHarness，请优先给出可靠结论。"
    registry, adapter = fake_registry([model_response(content="完成")])
    history = (
        Message(role=MessageRole.SYSTEM, content=prompt),
        Message(role=MessageRole.USER, content="之前的问题"),
        Message(role=MessageRole.ASSISTANT, content="之前的回答"),
    )

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        system_prompt=prompt,
    ).run("继续", history=history)

    assert (
        sum(
            message.role is MessageRole.SYSTEM and message.content == prompt
            for message in adapter.requests[0].messages
        )
        == 1
    )
    assert result.messages[: len(history)] == history
    assert (
        sum(
            message.role is MessageRole.SYSTEM and message.content == prompt
            for message in result.messages
        )
        == 1
    )


# 函数说明：test_runtime_system_prompt_survives_summary_and_prefix_reuse
# 用途：回归验证回归测试与测试辅助中的
# `runtime_system_prompt_survives_summary_and_prefix_reuse` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolCall` →
# `fake_registry` → `model_response` → `ModelCapabilityRegistry` →
# `capability_registry.register_override`；另有 10 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.summary_state.covered_message_count <= len(history_messages)`。
#   验证条件：`all((message.content != prompt for message in result.messages))`。
@pytest.mark.asyncio
async def test_runtime_system_prompt_survives_summary_and_prefix_reuse() -> None:

    prompt = "稳定系统提示：按证据回答。"
    history_messages: list[Message] = []
    for index in range(8):
        history_messages.extend(
            (
                Message(
                    role=MessageRole.USER,
                    content=f"旧问题 {index} " + "问" * 150,
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"旧回答 {index} " + "答" * 150,
                ),
            )
        )
    tool_call = ToolCall(
        id="count-with-system-prompt",
        name="count",
        arguments={"value": 1},
    )
    registry, adapter = fake_registry(
        [
            model_response(tool_calls=(tool_call,)),
            model_response(content="最终回答"),
        ]
    )
    capability_registry = ModelCapabilityRegistry()
    capability_registry.register_override(
        "fake",
        "fake-model",
        # Keep the compacted first request below the soft-compaction ceiling with
        # enough headroom for small changes to the serialized tool envelope.
        context_window=2_200,
        max_output_tokens=100,
    )
    context_manager = ContextManager(
        registry=capability_registry,
        budget_policy=ContextBudgetPolicy(safety_margin_tokens=0),
        conversation_reducer=ConversationReducer(
            FixedContextSummarizer(),
            keep_recent_conversation_blocks=2,
            keep_recent_tool_rounds=0,
        ),
    )
    tools = ToolRegistry()
    tools.register(CountingTool())
    events = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        system_prompt=prompt,
        max_output_tokens=100,
        context_manager=context_manager,
    ).run(
        "当前问题",
        history=tuple(history_messages),
        event_handler=events,
    )

    assert result.ok is True
    assert result.summary_state is not None
    assert result.summary_state.covered_message_count <= len(history_messages)
    assert all(message.content != prompt for message in result.messages)
    first, second = adapter.requests
    assert first.messages[0].role is MessageRole.SYSTEM
    assert first.messages[0].content == prompt
    assert sum(message.content == prompt for message in first.messages) == 1
    assert second.messages[: len(first.messages)] == first.messages
    started = [
        event for event in events.events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert [event.cache_prefix_reused for event in started] == [False, True]


# 函数说明：test_runtime_retries_one_empty_final_response
# 用途：回归验证回归测试与测试辅助中的 `runtime_retries_one_empty_final_response` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `AgentRuntime(registry, ToolRegistry(), provider='fake').run` → `AgentRuntime` →
# `ToolRegistry`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.steps == 2`。
#   验证条件：`result.content == '这是可展示的最终回答。'`。
#   验证条件：`len(adapter.requests) == 2`。
@pytest.mark.asyncio
async def test_runtime_retries_one_empty_final_response() -> None:

    registry, adapter = fake_registry(
        [
            model_response(content=None, reasoning="仅有内部推理"),
            model_response(content="这是可展示的最终回答。"),
        ]
    )

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
    ).run("请回答")

    assert result.ok is True
    assert result.steps == 2
    assert result.content == "这是可展示的最终回答。"
    assert len(adapter.requests) == 2
    assert "无可展示文本，也无工具调用" in (
        adapter.requests[1].messages[-1].content or ""
    )
    assert not any(
        message.role is MessageRole.ASSISTANT
        and not (message.content or "").strip()
        and not message.tool_calls
        for message in adapter.requests[1].messages
    )
    assert all(
        "响应修正：上一轮响应" not in (message.content or "")
        for message in result.messages
    )


# 函数说明：test_runtime_rejects_two_empty_final_responses
# 用途：回归验证回归测试与测试辅助中的 `runtime_rejects_two_empty_final_responses` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `AgentRuntime(registry, ToolRegistry(), provider='fake').run` → `AgentRuntime` →
# `ToolRegistry`。
# 分支与异常：
#   验证条件：`result.ok is False`。
#   验证条件：`result.steps == 2`。
#   验证条件：`result.stop_reason is AgentStopReason.MODEL_ERROR`。
#   验证条件：`result.error is not None`。
@pytest.mark.asyncio
async def test_runtime_rejects_two_empty_final_responses() -> None:

    registry, adapter = fake_registry(
        [model_response(content=None), model_response(content="   ")]
    )

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
    ).run("请回答")

    assert result.ok is False
    assert result.steps == 2
    assert result.stop_reason is AgentStopReason.MODEL_ERROR
    assert result.error is not None
    assert "empty content twice" in result.error.message
    assert len(adapter.requests) == 2
    assert all(
        "没有可展示文本" not in (message.content or "") for message in result.messages
    )


# 函数说明：test_runtime_retries_textual_tool_protocol_as_structured_response
# 用途：回归验证回归测试与测试辅助中的
# `runtime_retries_textual_tool_protocol_as_structured_response` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `AgentRuntime(registry, ToolRegistry(), provider='fake').run` → `AgentRuntime` →
# `ToolRegistry`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.steps == 2`。
#   验证条件：`result.content == '已直接完成回答。'`。
#   验证条件：`len(adapter.requests) == 2`。
@pytest.mark.asyncio
async def test_runtime_retries_textual_tool_protocol_as_structured_response() -> None:

    textual_protocol = '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="count">2'
    registry, adapter = fake_registry(
        [
            model_response(content=textual_protocol),
            model_response(content="已直接完成回答。"),
        ]
    )

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
    ).run("请回答")

    assert result.ok is True
    assert result.steps == 2
    assert result.content == "已直接完成回答。"
    assert len(adapter.requests) == 2
    assert "结构化 tool_calls" in (adapter.requests[1].messages[-1].content or "")
    assert all(
        textual_protocol not in (message.content or "")
        for message in adapter.requests[1].messages
    )
    assert all(
        textual_protocol not in (message.content or "") for message in result.messages
    )
    assert all(
        "结构化 tool_calls" not in (message.content or "")
        for message in result.messages
    )


# 函数说明：test_runtime_rejects_repeated_textual_tool_protocol
# 用途：回归验证回归测试与测试辅助中的 `runtime_rejects_repeated_textual_tool_protocol`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `AgentRuntime(registry, ToolRegistry(), provider='fake').run` → `AgentRuntime` →
# `ToolRegistry` → `(message.content or '').lower`。
# 分支与异常：
#   验证条件：`result.ok is False`。
#   验证条件：`result.steps == 2`。
#   验证条件：`result.stop_reason is AgentStopReason.MODEL_ERROR`。
#   验证条件：`result.error is not None`。
@pytest.mark.asyncio
async def test_runtime_rejects_repeated_textual_tool_protocol() -> None:

    registry, adapter = fake_registry(
        [
            model_response(content="<tool_calls>count(1)</tool_calls>"),
            model_response(
                content=('<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="count">2')
            ),
        ]
    )

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
    ).run("请回答")

    assert result.ok is False
    assert result.steps == 2
    assert result.stop_reason is AgentStopReason.MODEL_ERROR
    assert result.error is not None
    assert "textual tool call twice" in result.error.message
    assert len(adapter.requests) == 2
    assert all(
        not (
            message.role is MessageRole.ASSISTANT
            and (
                "<tool_calls" in (message.content or "").lower()
                or "<｜｜dsml｜｜" in (message.content or "").lower()
            )
        )
        for message in result.messages
    )
    assert all(
        "结构化 tool_calls" not in (message.content or "")
        for message in result.messages
    )


# 函数说明：test_runtime_reads_then_writes_and_returns_final_text
# 用途：回归验证回归测试与测试辅助中的
# `runtime_reads_then_writes_and_returns_final_text` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'input.txt').write_text`
# → `fake_registry` → `model_response` → `ToolCall` → `ModelUsage` → `ToolRegistry`；另
# 有 9 个调用点。
# 分支与异常：
#   验证条件：`result.content == '摘要已写入 output.md'`。
#   验证条件：`len(result.run_id) == 32`。
#   验证条件：`result.ok is True`。
#   验证条件：`result.steps == 3`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'input.txt').write_text`、
# `(tmp_path / 'output.md').read_text`。
@pytest.mark.asyncio
async def test_runtime_reads_then_writes_and_returns_final_text(tmp_path) -> None:
    (tmp_path / "input.txt").write_text(
        "MuHarness 可以调用本地工具完成文件任务。",
        encoding="utf-8",
    )

    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="read-1",
                        name="read_file",
                        arguments={"path": "input.txt"},
                    ),
                ),
                usage=ModelUsage(
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                ),
            ),
            model_response(
                tool_calls=(
                    ToolCall(
                        id="write-1",
                        name="write_file",
                        arguments={
                            "path": "output.md",
                            "content": "# 摘要\nMuHarness 能调用本地文件工具。",
                        },
                    ),
                ),
                usage=ModelUsage(
                    input_tokens=20,
                    output_tokens=3,
                    total_tokens=23,
                ),
            ),
            model_response(
                content="摘要已写入 output.md",
                usage=ModelUsage(
                    input_tokens=30,
                    output_tokens=4,
                    total_tokens=34,
                ),
            ),
        ]
    )
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    tools.register(WriteFileTool(tmp_path))
    event_handler = InMemoryEventHandler()
    initial_history = (Message(role=MessageRole.SYSTEM, content="你是本地文件助理。"),)

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_output_tokens=256,
    ).run(
        "读取 input.txt，生成摘要并写入 output.md",
        history=initial_history,
        conversation_id="conversation-1",
        event_handler=event_handler,
    )

    assert result.content == "摘要已写入 output.md"
    assert len(result.run_id) == 32
    assert result.ok is True
    assert result.steps == 3
    assert result.stop_reason is AgentStopReason.FINAL_ANSWER
    assert result.error is None
    assert result.messages[0] == initial_history[0]
    assert result.messages[-1] == result.final_message
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert result.usage == ModelUsage(
        input_tokens=60,
        output_tokens=9,
        total_tokens=69,
    )
    assert len(result.tool_rounds) == 2
    assert len(result.tool_calls) == 2
    assert [record.tool_call.name for record in result.tool_calls] == [
        "read_file",
        "write_file",
    ]
    assert result.tool_rounds[0].round_index == 0
    assert result.tool_rounds[0].records == (result.tool_calls[0],)
    assert result.tool_rounds[1].round_index == 1
    assert result.tool_rounds[1].records == (result.tool_calls[1],)
    events = event_handler.events
    assert [event.type for event in events] == [
        AgentEventType.AGENT_STARTED,
        AgentEventType.MODEL_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.TOOL_STARTED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.MODEL_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.TOOL_STARTED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.MODEL_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.AGENT_COMPLETED,
    ]
    assert [event.sequence for event in events] == list(range(len(events)))
    assert {event.run_id for event in events} == {result.run_id}
    assert {event.conversation_id for event in events} == {"conversation-1"}
    assert events[-1].message == result.final_message
    assert events[-1].usage == result.usage
    completed_tools = [
        event for event in events if event.type is AgentEventType.TOOL_COMPLETED
    ]
    assert [event.tool_result for event in completed_tools] == [
        result.tool_calls[0].result,
        result.tool_calls[1].result,
    ]
    assert (tmp_path / "output.md").read_text(encoding="utf-8") == (
        "# 摘要\nMuHarness 能调用本地文件工具。"
    )
    assert len(adapter.requests) == 3
    assert all(request.max_output_tokens == 256 for request in adapter.requests)
    assert {tool.name for tool in adapter.requests[0].tools} == {
        "read_file",
        "write_file",
    }
    for previous, current in zip(adapter.requests, adapter.requests[1:]):
        assert current.tools == previous.tools
        assert current.messages[: len(previous.messages)] == previous.messages

    model_started = [
        event for event in events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert [event.cache_prefix_reused for event in model_started] == [
        False,
        True,
        True,
    ]
    assert model_started[1].cache_prefix_message_count == len(
        adapter.requests[0].messages
    )
    assert model_started[2].cache_prefix_message_count == len(
        adapter.requests[1].messages
    )

    read_result_message = adapter.requests[1].messages[-1]
    assert read_result_message.role == MessageRole.TOOL
    assert read_result_message.tool_call_id == "read-1"
    read_result = json.loads(read_result_message.content or "{}")
    assert read_result["success"] is True
    assert "MuHarness 可以调用本地工具" in read_result["output"]

    write_result_message = adapter.requests[2].messages[-1]
    assert write_result_message.tool_call_id == "write-1"
    assert json.loads(write_result_message.content or "{}")["success"] is True


# 函数说明：test_runtime_below_trigger_sends_complete_history_to_model
# 用途：回归验证回归测试与测试辅助中的
# `runtime_below_trigger_sends_complete_history_to_model` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `Message` →
# `fake_registry` → `model_response` → `ToolRegistry` → `tools.register`；另有 3 个调用
# 点。
# 分支与异常：
#   验证条件：`result.messages[:len(history)] == history`。
#   验证条件：`result.messages[1].tool_calls == (older_call,)`。
#   验证条件：`result.messages[2].role is MessageRole.TOOL`。
#   验证条件：
# `first_request == (*history, Message(role=MessageRole.USER, content='这一轮'))`。
@pytest.mark.asyncio
async def test_runtime_below_trigger_sends_complete_history_to_model() -> None:
    older_call = ToolCall(
        id="older-count",
        name="count",
        arguments={"value": 1},
    )
    recent_call = ToolCall(
        id="recent-count",
        name="count",
        arguments={"value": 2},
    )
    history = (
        Message(role=MessageRole.USER, content="较旧一轮"),
        Message(role=MessageRole.ASSISTANT, tool_calls=(older_call,)),
        Message(
            role=MessageRole.TOOL,
            tool_call_id=older_call.id,
            name=older_call.name,
            content="1",
        ),
        Message(role=MessageRole.ASSISTANT, content="较旧一轮完成"),
        Message(role=MessageRole.USER, content="最近一轮"),
        Message(role=MessageRole.ASSISTANT, tool_calls=(recent_call,)),
        Message(
            role=MessageRole.TOOL,
            tool_call_id=recent_call.id,
            name=recent_call.name,
            content="2",
        ),
        Message(role=MessageRole.ASSISTANT, content="最近一轮完成"),
    )
    current_call = ToolCall(
        id="current-count",
        name="count",
        arguments={"value": 2},
    )
    registry, adapter = fake_registry(
        [
            model_response(tool_calls=(current_call,)),
            model_response(content="这一轮完成"),
        ]
    )
    tools = ToolRegistry()
    tools.register(CountingTool())

    result = await AgentRuntime(registry, tools, provider="fake").run(
        "这一轮",
        history=history,
    )

    assert result.messages[: len(history)] == history
    assert result.messages[1].tool_calls == (older_call,)
    assert result.messages[2].role is MessageRole.TOOL
    first_request = adapter.requests[0].messages
    assert first_request == (
        *history,
        Message(role=MessageRole.USER, content="这一轮"),
    )
    second_request = adapter.requests[1].messages
    assert any(older_call in message.tool_calls for message in second_request)
    assert any(recent_call in message.tool_calls for message in second_request)
    assert second_request[-2].tool_calls == (current_call,)
    assert second_request[-1].role is MessageRole.TOOL
    assert second_request[-1].tool_call_id == current_call.id


# 函数说明：test_runtime_does_not_retrim_oversized_legacy_history
# 用途：回归验证回归测试与测试辅助中的
# `runtime_does_not_retrim_oversized_legacy_history` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `Message` →
# `fake_registry` → `model_response` → `ModelCapabilityRegistry` →
# `capability_registry.register_override`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`result.ok is False`。
#   验证条件：`result.stop_reason is AgentStopReason.CONTEXT_ERROR`。
#   验证条件：`result.messages[:len(history)] == history`。
#   验证条件：`result.messages[2].content == long_result`。
@pytest.mark.asyncio
async def test_runtime_does_not_retrim_oversized_legacy_history() -> None:
    older_call = ToolCall(
        id="older-search",
        name="count",
        arguments={"value": 1},
    )
    recent_calls = (
        ToolCall(id="recent-1", name="count", arguments={"value": 2}),
        ToolCall(id="recent-2", name="count", arguments={"value": 3}),
    )
    long_result = "x" * 4_000
    history = (
        Message(role=MessageRole.USER, content="旧问题"),
        Message(role=MessageRole.ASSISTANT, tool_calls=(older_call,)),
        Message(
            role=MessageRole.TOOL,
            name=older_call.name,
            tool_call_id=older_call.id,
            content=long_result,
        ),
        Message(role=MessageRole.ASSISTANT, content="旧回答"),
        Message(role=MessageRole.ASSISTANT, tool_calls=(recent_calls[0],)),
        Message(
            role=MessageRole.TOOL,
            name=recent_calls[0].name,
            tool_call_id=recent_calls[0].id,
            content="最近结果一",
        ),
        Message(role=MessageRole.ASSISTANT, tool_calls=(recent_calls[1],)),
        Message(
            role=MessageRole.TOOL,
            name=recent_calls[1].name,
            tool_call_id=recent_calls[1].id,
            content="最近结果二",
        ),
    )
    registry, adapter = fake_registry([model_response(content="压缩后回答")])
    capability_registry = ModelCapabilityRegistry()
    capability_registry.register_override(
        "fake",
        "fake-model",
        context_window=600,
        max_output_tokens=100,
    )
    context_manager = ContextManager(
        registry=capability_registry,
        budget_policy=ContextBudgetPolicy(
            safety_margin_tokens=0,
            working_trigger_ratio=0.90,
            working_target_ratio=0.70,
        ),
        context_settings=ContextSettings(
            _env_file=None,
            context_keep_recent_tool_rounds=2,
            context_max_tool_result_chars=100,
            context_tool_result_head_chars=20,
            context_tool_result_tail_chars=20,
        ),
    )

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        max_output_tokens=100,
        context_manager=context_manager,
    ).run("当前问题", history=history)

    assert result.ok is False
    assert result.stop_reason is AgentStopReason.CONTEXT_ERROR
    assert result.messages[: len(history)] == history
    assert result.messages[2].content == long_result
    assert adapter.requests == []
    assert all(view.representation == "full" for view in result.tool_result_views)


# 函数说明：test_runtime_uses_rolling_summary_but_returns_complete_history
# 用途：回归验证回归测试与测试辅助中的
# `runtime_uses_rolling_summary_but_returns_complete_history` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `fake_registry` →
# `model_response` → `ModelUsage` → `ModelCapabilityRegistry` →
# `capability_registry.register_override`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.messages[:len(history)] == history`。
#   验证条件：`result.summary_state is not None`。
#   验证条件：`result.usage.total_tokens == 17`。
@pytest.mark.asyncio
async def test_runtime_uses_rolling_summary_but_returns_complete_history() -> None:
    history_messages = [Message(role=MessageRole.SYSTEM, content="系统提示")]
    for index in range(8):
        history_messages.extend(
            (
                Message(
                    role=MessageRole.USER,
                    content=f"旧问题 {index} " + "问" * 150,
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"旧回答 {index} " + "答" * 150,
                ),
            )
        )
    history = tuple(history_messages)
    registry, adapter = fake_registry(
        [
            model_response(
                content="最终回答",
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
            )
        ]
    )
    capability_registry = ModelCapabilityRegistry()
    capability_registry.register_override(
        "fake",
        "fake-model",
        context_window=2_000,
        max_output_tokens=100,
    )
    context_manager = ContextManager(
        registry=capability_registry,
        budget_policy=ContextBudgetPolicy(safety_margin_tokens=0),
        conversation_reducer=ConversationReducer(
            FixedContextSummarizer(),
            keep_recent_conversation_blocks=2,
            keep_recent_tool_rounds=0,
        ),
    )

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        max_output_tokens=100,
        context_manager=context_manager,
    ).run("当前问题", history=history)

    assert result.ok is True
    assert result.messages[: len(history)] == history
    assert result.summary_state is not None
    assert result.usage.total_tokens == 17
    request = adapter.requests[0]
    assert any(
        message.name == "muharness_rolling_summary" for message in request.messages
    )
    assert not any(
        message.content and "旧问题 0" in message.content
        for message in request.messages
    )


# 函数说明：test_runtime_extends_compacted_prefix_without_rebuilding_next_step
# 用途：回归验证回归测试与测试辅助中的
# `runtime_extends_compacted_prefix_without_rebuilding_next_step` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolCall` →
# `fake_registry` → `model_response` → `ModelCapabilityRegistry` →
# `capability_registry.register_override`；另有 10 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`second.messages[:len(first.messages)] == first.messages`。
#   验证条件：`second.messages[-2].tool_calls == (tool_call,)`。
#   验证条件：`second.messages[-1].tool_call_id == tool_call.id`。
@pytest.mark.asyncio
async def test_runtime_extends_compacted_prefix_without_rebuilding_next_step() -> None:
    history_messages = [Message(role=MessageRole.SYSTEM, content="系统提示")]
    for index in range(8):
        history_messages.extend(
            (
                Message(
                    role=MessageRole.USER,
                    content=f"旧问题 {index} " + "问" * 150,
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"旧回答 {index} " + "答" * 150,
                ),
            )
        )
    tool_call = ToolCall(
        id="count-after-summary",
        name="count",
        arguments={"value": 1},
    )
    registry, adapter = fake_registry(
        [
            model_response(tool_calls=(tool_call,)),
            model_response(content="最终回答"),
        ]
    )
    capability_registry = ModelCapabilityRegistry()
    capability_registry.register_override(
        "fake",
        "fake-model",
        # Keep the compacted first request below the soft-compaction ceiling with
        # enough headroom for small changes to the serialized tool envelope.
        context_window=2_200,
        max_output_tokens=100,
    )
    context_manager = ContextManager(
        registry=capability_registry,
        budget_policy=ContextBudgetPolicy(safety_margin_tokens=0),
        conversation_reducer=ConversationReducer(
            FixedContextSummarizer(),
            keep_recent_conversation_blocks=2,
            keep_recent_tool_rounds=0,
        ),
    )
    tools = ToolRegistry()
    tools.register(CountingTool())
    events = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_output_tokens=100,
        context_manager=context_manager,
    ).run(
        "当前问题",
        history=tuple(history_messages),
        event_handler=events,
    )

    assert result.ok is True
    first, second = adapter.requests
    assert second.messages[: len(first.messages)] == first.messages
    assert second.messages[-2].tool_calls == (tool_call,)
    assert second.messages[-1].tool_call_id == tool_call.id
    started = [
        event for event in events.events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert [event.summary_updated for event in started] == [True, False]
    assert [event.cache_prefix_reused for event in started] == [False, True]
    assert result.summary_state is not None


# 函数说明：test_runtime_rebuilds_prefix_for_late_compaction_with_active_skill
# 用途：回归验证回归测试与测试辅助中的
# `runtime_rebuilds_prefix_for_late_compaction_with_active_skill` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_dir.mkdir` →
# `(skill_dir / 'SKILL.md').write_text` → `SkillStore` → `store.initialize` →
# `(tmp_path / 'latest_error.txt').write_text` → `Message`；另有 18 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`[event.requires_compaction for event in started[:2]] == [False, True]`。
#   验证条件：`started[2].requires_compaction is False`。
#   验证条件：`started[1].cache_prefix_reused is False`。
# 副作用与资源：
#   文件或资源访问：`skill_dir.mkdir`、`(skill_dir / 'SKILL.md').write_text`、
# `(tmp_path / 'latest_error.txt').write_text`。
@pytest.mark.asyncio
async def test_runtime_rebuilds_prefix_for_late_compaction_with_active_skill(
    tmp_path,
) -> None:

    skill_dir = tmp_path / "project-skills" / "debug-python"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: debug-python\n"
        "description: 排查 Python 报错或异常的标准流程\n"
        "---\n\n"
        "# Debug Python\n"
        "遇到报错时：1. 复现 2. 读 traceback 3. 查代码 4. 修复并验证。",
        encoding="utf-8",
    )
    store = SkillStore(tmp_path / "user-skills", tmp_path / "project-skills")
    await store.initialize()
    (tmp_path / "latest_error.txt").write_text(
        'Traceback\nvalue = int("abc")\nValueError: invalid literal',
        encoding="utf-8",
    )

    history = (
        Message(role=MessageRole.USER, content="开始排查几个历史报错。"),
        Message(
            role=MessageRole.ASSISTANT,
            content=(
                "好的，我会逐一记录历史报错及其处理过程，并说明每一步的排查"
                "依据。为了保持排查的完整上下文，我会保留之前的报错症状、尝试"
                "过的修复和验证结果，避免遗漏关键信息。"
            ),
        ),
        Message(role=MessageRole.USER, content="继续记录更多报错细节。"),
        Message(
            role=MessageRole.ASSISTANT,
            content=(
                "继续补充历史报错：KeyError 的根因是字典访问缺键，处理方式是"
                "先用 get 提供默认值；IndexError 的根因是列表越界，处理方式是"
                "先判断长度。每一个报错都记录复现步骤、根因分析和验证结果。"
            ),
        ),
        Message(role=MessageRole.USER, content="再补充一批报错记录。"),
        Message(
            role=MessageRole.ASSISTANT,
            content=(
                "AttributeError 的根因是 None 对象访问属性，处理方式是判空；"
                "TypeError 的根因是类型不匹配，处理方式是显式转换。全部记录按"
                "时间顺序整理，包含症状、定位方法、修复与验证结果。"
            ),
        ),
        Message(role=MessageRole.USER, content="把前面的报错记录再汇总一遍。"),
        Message(
            role=MessageRole.ASSISTANT,
            content=(
                "汇总：KeyError 用 get 默认值；IndexError 先判断长度；"
                "AttributeError 先判空；TypeError 显式转换。流程始终是先复现、"
                "再读 traceback、定位根因、修复并验证。"
            ),
        ),
    )
    model_registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="activate-debug-skill",
                        name=SKILL_READ_TOOL_NAME,
                        arguments={"name": "debug-python"},
                    ),
                )
            ),
            model_response(
                content="Skill 已激活。现在读取最新报错文件：",
                tool_calls=(
                    ToolCall(
                        id="read-latest-error",
                        name="read_file",
                        arguments={"path": "latest_error.txt"},
                    ),
                ),
            ),
            model_response(content="根因是 int 转换失败导致 ValueError。"),
        ]
    )
    capability_registry = ModelCapabilityRegistry()
    capability_registry.register_override(
        "fake",
        "fake-model",
        context_window=2_200,
        max_output_tokens=512,
    )
    context_manager = ContextManager(
        registry=capability_registry,
        budget_policy=ContextBudgetPolicy(
            safety_margin_tokens=100,
            working_trigger_ratio=0.70,
            # Activation is a real instruction change and crosses the one
            # pressure gate, creating a new append-only epoch.
            compact_input_tokens=1_100,
        ),
        conversation_reducer=ConversationReducer(
            FixedContextSummarizer(),
            keep_recent_conversation_blocks=1,
            keep_recent_tool_rounds=0,
        ),
    )
    tools = ToolRegistry()
    register_skill_tools(tools, store)
    tools.unregister("skill_resource_read")
    tools.register(ReadFileTool(tmp_path))
    events = InMemoryEventHandler()

    result = await AgentRuntime(
        model_registry,
        tools,
        provider="fake",
        max_output_tokens=512,
        context_manager=context_manager,
        skill_store=store,
        skill_context_provider=SkillContextProvider(
            max_tokens=4_096,
            max_active=4,
        ),
    ).run(
        "先激活匹配的 Skill，再读取 latest_error.txt，按标准流程排查。",
        history=history,
        event_handler=events,
    )

    assert result.ok is True
    started = [
        event for event in events.events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert [event.requires_compaction for event in started[:2]] == [False, True], [
        (event.original_estimated_input_tokens, event.trigger_tokens)
        for event in started
    ]
    assert started[2].requires_compaction is False, [
        (
            event.original_estimated_input_tokens,
            event.trigger_tokens,
            event.cache_prefix_reused,
        )
        for event in started
    ]
    assert started[1].cache_prefix_reused is False
    assert started[1].summary_updated is True
    assert started[1].compaction_stage == "rolling_summary"
    assert started[2].cache_prefix_reused is True
    assert (
        adapter.requests[2].messages[: len(adapter.requests[1].messages)]
        == adapter.requests[1].messages
    )
    assert started[2].active_skill_message_names == ("debug-python",)
    assert result.summary_state is not None
    final_request = adapter.requests[2]
    assert any(
        message.name == ACTIVE_SKILL_MESSAGE_NAME
        and "Skill: debug-python" in (message.content or "")
        for message in final_request.messages
    )
    assert any(
        message.name == "muharness_rolling_summary"
        for message in final_request.messages
    )


# 函数说明：test_runtime_does_not_inject_pending_task_created_in_current_run
# 用途：回归验证回归测试与测试辅助中的
# `runtime_does_not_inject_pending_task_created_in_current_run` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` →
# `task_store.initialize` → `fake_registry` → `model_response` → `ToolCall` →
# `ToolRegistry`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`len(tasks) == 1`。
#   验证条件：`tasks[0].owner_conversation_id == 'conversation-1'`。
#   验证条件：`result.run_id in tasks[0].run_ids`。
@pytest.mark.asyncio
async def test_runtime_does_not_inject_pending_task_created_in_current_run(
    tmp_path,
) -> None:

    task_store = FileTaskStore(tmp_path / "tasks")
    await task_store.initialize()
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="create-task",
                        name="task_create",
                        arguments={
                            "title": "实现长任务",
                            "goal": "完成所有步骤",
                            "steps": [{"title": "第一步"}],
                        },
                    ),
                )
            ),
            model_response(content="任务已创建并开始执行"),
        ]
    )
    tools = ToolRegistry()
    register_task_tools(tools, task_store)
    runtime = AgentRuntime(
        registry,
        tools,
        provider="fake",
        task_context_provider=TaskContextProvider(task_store),
    )

    result = await runtime.run(
        "请完成这个长任务",
        conversation_id="conversation-1",
    )

    assert result.ok is True
    tasks = await task_store.list()
    assert len(tasks) == 1
    assert tasks[0].owner_conversation_id == "conversation-1"
    assert result.run_id in tasks[0].run_ids
    assert tasks[0].status.value == "pending"
    for request in adapter.requests:
        assert not any(
            message.name == TASK_CONTEXT_MESSAGE_NAME for message in request.messages
        )
    assert not any(
        message.name == TASK_CONTEXT_MESSAGE_NAME for message in result.messages
    )

    await task_store.plan_accept(tasks[0].id)
    injected = await TaskContextProvider(task_store).message_for("conversation-1")
    assert injected is not None and tasks[0].id in (injected.content or "")


# 函数说明：test_runtime_refreshes_task_context_after_step_update
# 用途：回归验证回归测试与测试辅助中的
# `runtime_refreshes_task_context_after_step_update` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` →
# `task_store.initialize` → `task_store.create` → `TaskStep` → `task_store.plan_accept`
# → `fake_registry`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`'"revision":2' in (first_context.content or '')`。
#   验证条件：`'"status":"todo"' in (first_context.content or '')`。
#   验证条件：`'"revision":3' in (second_context.content or '')`。
@pytest.mark.asyncio
async def test_runtime_refreshes_task_context_after_step_update(tmp_path) -> None:
    task_store = FileTaskStore(tmp_path / "tasks")
    await task_store.initialize()
    task = await task_store.create(
        title="持续任务",
        steps=(TaskStep(id="step-1", title="完成实现"),),
        owner_conversation_id="conversation-1",
    )
    task = await task_store.plan_accept(task.id)
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="update-task",
                        name="task_update",
                        arguments={
                            "task_id": task.id,
                            "expected_revision": task.revision,
                            "step_id": "step-1",
                            "step_status": "done",
                            "step_note": "实现已完成",
                        },
                    ),
                )
            ),
            model_response(content="步骤已经完成"),
        ]
    )
    tools = ToolRegistry()
    register_task_tools(tools, task_store)

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        task_context_provider=TaskContextProvider(task_store),
    ).run("继续执行", conversation_id="conversation-1")

    assert result.ok is True
    first_context = next(
        message
        for message in adapter.requests[0].messages
        if message.name == TASK_CONTEXT_MESSAGE_NAME
    )
    second_context = next(
        message
        for message in reversed(adapter.requests[1].messages)
        if message.name == TASK_CONTEXT_MESSAGE_NAME
    )
    assert '"revision":2' in (first_context.content or "")
    assert '"status":"todo"' in (first_context.content or "")
    assert '"revision":3' in (second_context.content or "")
    assert '"status":"done"' in (second_context.content or "")
    first, second = adapter.requests
    assert second.messages[: len(first.messages)] == first.messages
    assert first_context.role is MessageRole.USER
    assert second_context.role is MessageRole.USER
    assert second.messages[-1] == second_context
    assert not any(
        message.name == TASK_CONTEXT_MESSAGE_NAME for message in result.messages
    )


# 函数说明：test_frozen_tool_views_append_and_resume_without_changing_raw_results
# 用途：回归验证回归测试与测试辅助中的
# `frozen_tool_views_append_and_resume_without_changing_raw_results` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ToolRegistry` → `tools.register` → `SizedEchoTool`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`result.ok`。
#   验证条件：`json.loads(raw_tool.content or '{}')['output'] == payload`。
#   验证条件：`'characters omitted' in view_payload['output']`。
#   验证条件：`view_payload['success'] is True`。
@pytest.mark.asyncio
async def test_frozen_tool_views_append_and_resume_without_changing_raw_results() -> (
    None
):
    payload = "HEAD\n" + "middle evidence\n" * 900 + "TAIL"
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(ToolCall(id="large", name="large", arguments={}),)
            ),
            model_response(
                tool_calls=(ToolCall(id="missing", name="missing", arguments={}),)
            ),
            model_response(content="已检查结果"),
            model_response(content="继续完成"),
        ]
    )
    tools = ToolRegistry()
    tools.register(SizedEchoTool("large", payload))
    runtime = AgentRuntime(registry, tools, provider="fake")
    result = await runtime.run("读取并检查")
    assert result.ok
    raw_tool = next(
        message for message in result.messages if message.tool_call_id == "large"
    )
    assert json.loads(raw_tool.content or "{}")["output"] == payload
    view = next(
        message
        for message in adapter.requests[1].messages
        if message.tool_call_id == "large"
    )
    view_payload = json.loads(view.content or "{}")
    assert "characters omitted" in view_payload["output"]
    assert view_payload["success"] is True
    assert view_payload.get("output_sha256") == json.loads(
        raw_tool.content or "{}"
    ).get("output_sha256")
    assert result.tool_calls[0].result.output == payload
    for before, after in zip(adapter.requests[:3], adapter.requests[1:3]):
        assert after.messages[: len(before.messages)] == before.messages
    failure = next(
        message
        for message in adapter.requests[2].messages
        if message.tool_call_id == "missing"
    )
    assert json.loads(failure.content or "{}")["success"] is False
    restored = await runtime.run(
        "继续",
        history=result.messages,
        tool_result_views=result.tool_result_views,
    )
    assert restored.ok
    resumed_view = next(
        message
        for message in adapter.requests[-1].messages
        if message.tool_call_id == "large"
    )
    assert resumed_view == view


# 函数说明：test_task_updates_preserve_anthropic_system_and_protocol_prefix
# 用途：回归验证回归测试与测试辅助中的
# `task_updates_preserve_anthropic_system_and_protocol_prefix` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolCall` →
# `RequestPrefixState` → `_anthropic_messages`。
# 分支与异常：
#   验证条件：`appended is not None`。
#   验证条件：`appended[:len(first)] == first`。
#   验证条件：`old_system == new_system == system.content`。
#   验证条件：`new_wire[:len(old_wire)] == old_wire`。
def test_task_updates_preserve_anthropic_system_and_protocol_prefix() -> None:
    from app.models.providers.anthropic import _anthropic_messages
    from app.runtime.agent.runtime_helpers import RequestPrefixState

    user = Message(role=MessageRole.USER, content="任务")
    task1 = Message(
        role=MessageRole.USER, name=TASK_CONTEXT_MESSAGE_NAME, content="revision=1"
    )
    task2 = task1.model_copy(update={"content": "revision=2"})
    system = Message(role=MessageRole.SYSTEM, content="固定安全与模式规则")
    call = Message(
        role=MessageRole.ASSISTANT,
        tool_calls=(ToolCall(id="c", name="count", arguments={}),),
    )
    result = Message(
        role=MessageRole.TOOL, tool_call_id="c", content='{"success":true}'
    )
    first = (system, task1, user)
    state = RequestPrefixState((user,), (task1,), (), first)
    appended = state.extend(
        source_messages=(user, call, result), context_messages=(task2,), tools=()
    )
    assert appended is not None
    assert appended[: len(first)] == first
    old_system, old_wire = _anthropic_messages(first)
    new_system, new_wire = _anthropic_messages(appended)
    assert old_system == new_system == system.content
    assert new_wire[: len(old_wire)] == old_wire
    assert new_wire[-2]["content"][0]["tool_use_id"] == "c"
    assert new_wire[-1]["content"] == "revision=2"


# 函数说明：test_model_error_returns_assistant_message
# 用途：回归验证回归测试与测试辅助中的 `model_error_returns_assistant_message` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` →
# `InMemoryEventHandler` → `AgentRuntime(registry, ToolRegistry(), provider='fake').run`
#  → `AgentRuntime` → `ToolRegistry` → `ModelUsage`。
# 分支与异常：
#   验证条件：`result.role == MessageRole.ASSISTANT`。
#   验证条件：`result.ok is False`。
#   验证条件：`result.steps == 1`。
#   验证条件：`result.stop_reason is AgentStopReason.MODEL_ERROR`。
@pytest.mark.asyncio
async def test_model_error_returns_assistant_message() -> None:
    registry, _ = fake_registry([RuntimeError("model unavailable")])
    event_handler = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
    ).run("hello", event_handler=event_handler)

    assert result.role == MessageRole.ASSISTANT
    assert result.ok is False
    assert result.steps == 1
    assert result.stop_reason is AgentStopReason.MODEL_ERROR
    assert result.error is not None
    assert result.error.type == "ModelInvocationError"
    assert result.messages[-1] == result.final_message
    assert result.usage == ModelUsage()
    assert [event.type for event in event_handler.events] == [
        AgentEventType.AGENT_STARTED,
        AgentEventType.MODEL_STARTED,
        AgentEventType.AGENT_FAILED,
    ]
    assert event_handler.events[-1].error == result.error
    assert event_handler.events[-1].stop_reason == result.stop_reason
    assert "model invocation failed" in (result.content or "")
    assert "model unavailable" in (result.content or "")


# 函数说明：test_tool_error_is_returned_to_model
# 用途：回归验证回归测试与测试辅助中的 `tool_error_is_returned_to_model` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `AgentRuntime(registry, ToolRegistry(), provider='fake').run` →
# `AgentRuntime` → `ToolRegistry`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`result.content == '工具不可用，已停止该操作'`。
#   验证条件：`result.stop_reason is AgentStopReason.FINAL_ANSWER`。
#   验证条件：`result.error is None`。
#   验证条件：`result.messages[-1] == result.final_message`。
@pytest.mark.asyncio
async def test_tool_error_is_returned_to_model() -> None:
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="missing-1",
                        name="missing_tool",
                        arguments={},
                    ),
                )
            ),
            model_response(content="工具不可用，已停止该操作"),
        ]
    )

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
    ).run("调用不存在的工具")

    assert result.content == "工具不可用，已停止该操作"
    assert result.stop_reason is AgentStopReason.FINAL_ANSWER
    assert result.error is None
    assert result.messages[-1] == result.final_message
    assert len(result.tool_rounds) == 1
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].result.success is False
    assert result.tool_calls[0].result.tool_name == "missing_tool"
    tool_result = json.loads(adapter.requests[1].messages[-1].content or "{}")
    assert tool_result["success"] is False
    assert "not found" in tool_result["error"].lower()


# 函数说明：test_three_identical_tool_calls_stop_before_third_execution
# 用途：回归验证回归测试与测试辅助中的
# `three_identical_tool_calls_stop_before_third_execution` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`model_response` → `ToolCall` →
# `fake_registry` → `CountingTool` → `ToolRegistry` → `tools.register`；另有 2 个调用点
# 。
# 分支与异常：
#   验证条件：`counting_tool.executions == 2`。
#   验证条件：`result.ok is False`。
#   验证条件：`result.steps == 3`。
#   验证条件：`result.stop_reason is AgentStopReason.REPEATED_TOOL_CALL`。
@pytest.mark.asyncio
async def test_three_identical_tool_calls_stop_before_third_execution() -> None:
    repeated_calls = [
        model_response(
            tool_calls=(
                ToolCall(
                    id=f"count-{index}",
                    name="count",
                    arguments={"value": 1},
                ),
            )
        )
        for index in range(3)
    ]
    registry, _ = fake_registry(repeated_calls)
    counting_tool = CountingTool()
    tools = ToolRegistry()
    tools.register(counting_tool)

    result = await AgentRuntime(registry, tools, provider="fake").run("count")

    assert counting_tool.executions == 2
    assert result.ok is False
    assert result.steps == 3
    assert result.stop_reason is AgentStopReason.REPEATED_TOOL_CALL
    assert result.error is not None
    assert result.error.type == "RepeatedToolCallError"
    assert result.messages[-1] == result.final_message
    assert len(result.tool_rounds) == 2
    assert len(result.tool_calls) == 2
    assert "3 consecutive times" in (result.content or "")


# 函数说明：test_max_steps_stops_the_loop
# 用途：回归验证回归测试与测试辅助中的 `max_steps_stops_the_loop` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ToolRegistry` → `tools.register` → `CountingTool`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`len(adapter.requests) == 2`。
#   验证条件：`result.ok is False`。
#   验证条件：`result.steps == 2`。
#   验证条件：`result.stop_reason is AgentStopReason.MAX_STEPS`。
@pytest.mark.asyncio
async def test_max_steps_stops_the_loop() -> None:
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id=f"count-{index}",
                        name="count",
                        arguments={"value": index},
                    ),
                )
            )
            for index in range(2)
        ]
    )
    tools = ToolRegistry()
    tools.register(CountingTool())

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_steps=2,
    ).run("keep counting")

    assert len(adapter.requests) == 2
    assert result.ok is False
    assert result.steps == 2
    assert result.stop_reason is AgentStopReason.MAX_STEPS
    assert result.error is not None
    assert result.error.type == "MaxStepsExceededError"
    assert result.messages[-1] == result.final_message
    assert len(result.tool_rounds) == 2
    assert len(result.tool_calls) == 2
    assert "maximum step limit (2) reached" in (result.content or "")


# 函数说明：test_tool_round_budget_forces_final_answer_without_more_tools
# 用途：回归验证回归测试与测试辅助中的
# `tool_round_budget_forces_final_answer_without_more_tools` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ToolRegistry` → `tools.register` → `CountingTool`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.steps == 4`。
#   验证条件：`result.content == '根据已有结果完成回答'`。
#   验证条件：`len(result.tool_rounds) == 3`。
@pytest.mark.asyncio
async def test_tool_round_budget_forces_final_answer_without_more_tools() -> None:
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id=f"count-{index}",
                        name="count",
                        arguments={"value": index},
                    ),
                )
            )
            for index in range(3)
        ]
        + [model_response(content="根据已有结果完成回答")]
    )
    tools = ToolRegistry()
    tools.register(CountingTool())

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_steps=10,
        max_tool_rounds=3,
    ).run("连续收集信息")

    assert result.ok is True
    assert result.steps == 4
    assert result.content == "根据已有结果完成回答"
    assert len(result.tool_rounds) == 3
    final_request = adapter.requests[-1]
    assert final_request.tools == ()
    assert final_request.tool_choice is None
    assert final_request.messages[-1].role is MessageRole.SYSTEM
    assert "禁止继续调用工具" in (final_request.messages[-1].content or "")
    assert result.messages[-1] == result.final_message
    assert all(
        "运行阶段：工具轮次用尽" not in (message.content or "")
        for message in result.messages
    )


# 函数说明：test_textual_tool_call_after_tool_round_limit_gets_safe_fallback
# 用途：回归验证回归测试与测试辅助中的
# `textual_tool_call_after_tool_round_limit_gets_safe_fallback` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `CountingTool` → `ToolRegistry` → `tools.register`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`counting_tool.executions == 1`。
#   验证条件：`len(adapter.requests) == 2`。
#   验证条件：`adapter.requests[-1].tools == ()`。
#   验证条件：`result.ok is True`。
@pytest.mark.asyncio
async def test_textual_tool_call_after_tool_round_limit_gets_safe_fallback() -> None:
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="count-1",
                        name="count",
                        arguments={"value": 1},
                    ),
                )
            ),
            model_response(
                content=('<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="count">2')
            ),
        ]
    )
    counting_tool = CountingTool()
    tools = ToolRegistry()
    tools.register(counting_tool)

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_steps=10,
        max_tool_rounds=1,
    ).run("持续计数")

    assert counting_tool.executions == 1
    assert len(adapter.requests) == 2
    assert adapter.requests[-1].tools == ()
    assert result.ok is True
    assert result.steps == 2
    assert result.stop_reason is AgentStopReason.FINAL_ANSWER
    assert result.error is None
    assert "达到工具轮次上限" in (result.content or "")
    assert "不能据此认定任务完成" in (result.content or "")
    assert "maximum step limit" not in (result.content or "")


# 函数说明：test_run_budget_uses_one_dedicated_finalization_call
# 用途：回归验证回归测试与测试辅助中的 `run_budget_uses_one_dedicated_finalization_call`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ModelUsage` → `ToolRegistry` → `tools.register`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.steps == 2`。
#   验证条件：`len(adapter.requests) == 2`。
#   验证条件：`adapter.requests[-1].tools == ()`。
@pytest.mark.asyncio
async def test_run_budget_uses_one_dedicated_finalization_call() -> None:
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="count-1",
                        name="count",
                        arguments={"value": 1},
                    ),
                ),
                usage=ModelUsage(
                    input_tokens=100,
                    output_tokens=1,
                    total_tokens=101,
                    cached_input_tokens=90,
                    uncached_input_tokens=10,
                ),
            ),
            model_response(content="已根据现有结果提前收口"),
        ]
    )
    tools = ToolRegistry()
    tools.register(CountingTool())
    events = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        run_budget_config=RunBudgetConfig(
            _env_file=None,
            warning_tokens=5,
            finalization_tokens=10,
            hard_tokens=20,
            warning_model_calls=100,
            finalization_model_calls=101,
            hard_model_calls=102,
        ),
    ).run("执行计数任务", event_handler=events)

    assert result.ok is True
    assert result.steps == 2
    assert len(adapter.requests) == 2
    assert adapter.requests[-1].tools == ()
    assert adapter.requests[-1].max_output_tokens == 1_200
    assert "用量收口线" in (adapter.requests[-1].messages[-1].content or "")
    assert any(
        event.type is AgentEventType.RUN_BUDGET_FINALIZING
        and event.run_budget_reason == "tokens"
        and event.run_budget_chargeable_tokens == 11
        for event in events.events
    )


# 函数说明：test_run_budget_closing_keeps_one_delivery_tool_round
# 用途：回归验证回归测试与测试辅助中的
# `run_budget_closing_keeps_one_delivery_tool_round` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ModelUsage` → `CountingTool` → `DeliveryTool`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.content == '交付完成'`。
#   验证条件：`count_tool.executions == 1`。
#   验证条件：`delivery_tool.executions == 1`。
@pytest.mark.asyncio
async def test_run_budget_closing_keeps_one_delivery_tool_round() -> None:
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(id="count-1", name="count", arguments={"value": 1}),
                ),
                usage=ModelUsage(
                    input_tokens=100,
                    output_tokens=1,
                    total_tokens=101,
                    cached_input_tokens=90,
                    uncached_input_tokens=10,
                ),
            ),
            model_response(
                tool_calls=(
                    ToolCall(
                        id="deliver-1",
                        name="deliver",
                        arguments={"value": 2},
                    ),
                )
            ),
            model_response(content="交付完成"),
        ]
    )
    count_tool = CountingTool()
    delivery_tool = DeliveryTool()
    tools = ToolRegistry()
    tools.register(count_tool)
    tools.register(delivery_tool)

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        run_budget_config=RunBudgetConfig(
            _env_file=None,
            warning_tokens=5,
            finalization_tokens=10,
            hard_tokens=20,
            warning_model_calls=100,
            finalization_model_calls=101,
            hard_model_calls=102,
        ),
    ).run("执行并交付")

    assert result.ok is True
    assert result.content == "交付完成"
    assert count_tool.executions == 1
    assert delivery_tool.executions == 1
    assert len(adapter.requests) == 3
    assert [tool.name for tool in adapter.requests[1].tools] == ["deliver"]
    assert adapter.requests[1].max_output_tokens != 1_200
    assert adapter.requests[2].tools == ()
    assert adapter.requests[2].max_output_tokens == 1_200
    assert "Closing" in (adapter.requests[1].messages[-1].content or "")
    assert "本轮禁止工具调用" in (adapter.requests[2].messages[-1].content or "")


# 函数说明：test_run_budget_closing_rejects_forged_exploration_tool_call
# 用途：回归验证回归测试与测试辅助中的
# `run_budget_closing_rejects_forged_exploration_tool_call` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ModelUsage` → `CountingTool` → `ToolRegistry`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`count_tool.executions == 1`。
#   验证条件：`result.tool_calls[-1].result.success is False`。
#   验证条件：`'delivery tools only' in (result.tool_calls[-1].result.error or '')`。
@pytest.mark.asyncio
async def test_run_budget_closing_rejects_forged_exploration_tool_call() -> None:
    registry, _ = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(id="count-1", name="count", arguments={"value": 1}),
                ),
                usage=ModelUsage(
                    input_tokens=100,
                    output_tokens=1,
                    total_tokens=101,
                    cached_input_tokens=90,
                    uncached_input_tokens=10,
                ),
            ),
            model_response(
                tool_calls=(
                    ToolCall(id="count-2", name="count", arguments={"value": 2}),
                )
            ),
            model_response(content="未执行额外调查，已如实收口"),
        ]
    )
    count_tool = CountingTool()
    tools = ToolRegistry()
    tools.register(count_tool)
    tools.register(DeliveryTool())

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        run_budget_config=RunBudgetConfig(
            _env_file=None,
            warning_tokens=5,
            finalization_tokens=10,
            hard_tokens=20,
            warning_model_calls=100,
            finalization_model_calls=101,
            hard_model_calls=102,
        ),
    ).run("执行后收口")

    assert result.ok is True
    assert count_tool.executions == 1
    assert result.tool_calls[-1].result.success is False
    assert "delivery tools only" in (result.tool_calls[-1].result.error or "")


# 函数说明：test_run_budget_hard_limit_stops_without_another_model_call
# 用途：回归验证回归测试与测试辅助中的
# `run_budget_hard_limit_stops_without_another_model_call` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ModelUsage` → `ToolRegistry` → `tools.register`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`len(adapter.requests) == 1`。
#   验证条件：`result.stop_reason is AgentStopReason.RUN_BUDGET`。
#   验证条件：`result.error is not None`。
#   验证条件：`result.error.type == 'RunBudgetExceededError'`。
@pytest.mark.asyncio
async def test_run_budget_hard_limit_stops_without_another_model_call() -> None:
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="count-1",
                        name="count",
                        arguments={"value": 1},
                    ),
                ),
                usage=ModelUsage(
                    input_tokens=25,
                    output_tokens=1,
                    total_tokens=26,
                ),
            ),
        ]
    )
    tools = ToolRegistry()
    tools.register(CountingTool())

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        run_budget_config=RunBudgetConfig(
            _env_file=None,
            warning_tokens=5,
            finalization_tokens=10,
            hard_tokens=20,
            warning_model_calls=100,
            finalization_model_calls=101,
            hard_model_calls=102,
        ),
    ).run("执行计数任务")

    assert len(adapter.requests) == 1
    assert result.stop_reason is AgentStopReason.RUN_BUDGET
    assert result.error is not None
    assert result.error.type == "RunBudgetExceededError"


# 函数说明：test_event_handler_failure_does_not_stop_runtime
# 用途：回归验证回归测试与测试辅助中的 `event_handler_failure_does_not_stop_runtime` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `AgentRuntime(registry, ToolRegistry(), provider='fake').run` → `AgentRuntime` →
# `ToolRegistry` → `FailingEventHandler`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.content == '正常完成'`。
@pytest.mark.asyncio
async def test_event_handler_failure_does_not_stop_runtime() -> None:
    registry, _ = fake_registry([model_response(content="正常完成")])

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
    ).run("hello", event_handler=FailingEventHandler())

    assert result.ok is True
    assert result.content == "正常完成"


# 函数说明：test_run_stream_returns_events_and_final_result
# 用途：回归验证回归测试与测试辅助中的 `run_stream_returns_events_and_final_result` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `AgentRuntime` → `ToolRegistry` → `InMemoryEventHandler` → `runtime.run_stream`。
# 分支与异常：
#   验证条件：`[event.type for event in events] == [AgentEventType.AGENT_STARTED,
# AgentEventType.MODEL_STARTED, AgentEventType.MODEL_COMPLETED,…`。
#   验证条件：`final_result is not None`。
#   验证条件：`final_result.content == '流式完成'`。
#   验证条件：`final_result.run_id == events[-1].run_id`。
@pytest.mark.asyncio
async def test_run_stream_returns_events_and_final_result() -> None:
    registry, _ = fake_registry([model_response(content="流式完成")])
    runtime = AgentRuntime(registry, ToolRegistry(), provider="fake")
    observer = InMemoryEventHandler()

    events = [
        event
        async for event in runtime.run_stream(
            "hello",
            conversation_id="conversation-1",
            event_handler=observer,
        )
    ]

    assert [event.type for event in events] == [
        AgentEventType.AGENT_STARTED,
        AgentEventType.MODEL_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.AGENT_COMPLETED,
    ]
    final_result = events[-1].result
    assert final_result is not None
    assert final_result.content == "流式完成"
    assert final_result.run_id == events[-1].run_id
    assert {event.conversation_id for event in events} == {"conversation-1"}
    assert observer.events == tuple(events)


# 函数说明：test_runtime_emits_approval_required_and_completed_events
# 用途：回归验证回归测试与测试辅助中的
# `runtime_emits_approval_required_and_completed_events` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ApprovalCountingTool` → `ToolRegistry` → `tools.register`；另有 4 个调用
# 点。
# 分支与异常：
#   验证条件：`[event.type for event in approval_events] == [AgentEventType.
# TOOL_APPROVAL_REQUIRED, AgentEventType.TOOL_APPROVAL_COMPLETED]`。
#   验证条件：`approval_events[0].approval_decision is None`。
#   验证条件：`approval_events[1].approval_decision is ApprovalDecision.APPROVED`。
#   验证条件：`approval_events[0].tool_call == result.tool_calls[0].tool_call`。
@pytest.mark.asyncio
async def test_runtime_emits_approval_required_and_completed_events() -> None:
    registry, _ = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="approval-1",
                        name="approval_count",
                        arguments={"value": 1},
                    ),
                )
            ),
            model_response(content="审批工具执行完成"),
        ]
    )
    tool = ApprovalCountingTool()
    tools = ToolRegistry()
    tools.register(tool)
    handler = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        approval_gate=AutoApproveGate(),
    ).run("执行审批工具", event_handler=handler)

    approval_events = [
        event
        for event in handler.events
        if event.type
        in {
            AgentEventType.TOOL_APPROVAL_REQUIRED,
            AgentEventType.TOOL_APPROVAL_COMPLETED,
        }
    ]
    assert [event.type for event in approval_events] == [
        AgentEventType.TOOL_APPROVAL_REQUIRED,
        AgentEventType.TOOL_APPROVAL_COMPLETED,
    ]
    assert approval_events[0].approval_decision is None
    assert approval_events[1].approval_decision is ApprovalDecision.APPROVED
    assert approval_events[0].tool_call == result.tool_calls[0].tool_call
    assert tool.executions == 1


# 函数说明：test_runtime_records_denied_approval_without_executing_tool
# 用途：回归验证回归测试与测试辅助中的
# `runtime_records_denied_approval_without_executing_tool` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ApprovalCountingTool` → `ToolRegistry` → `tools.register`；另有 5 个调用
# 点。
# 分支与异常：
#   验证条件：`completed.approval_decision is ApprovalDecision.DENIED`。
#   验证条件：`result.tool_calls[0].result.success is False`。
#   验证条件：`tool.executions == 0`。
@pytest.mark.asyncio
async def test_runtime_records_denied_approval_without_executing_tool() -> None:
    registry, _ = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="approval-1",
                        name="approval_count",
                        arguments={"value": 1},
                    ),
                )
            ),
            model_response(content="审批被拒绝"),
        ]
    )
    tool = ApprovalCountingTool()
    tools = ToolRegistry()
    tools.register(tool)
    handler = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        approval_gate=DenyAllGate(),
    ).run("执行审批工具", event_handler=handler)

    completed = next(
        event
        for event in handler.events
        if event.type is AgentEventType.TOOL_APPROVAL_COMPLETED
    )
    assert completed.approval_decision is ApprovalDecision.DENIED
    assert result.tool_calls[0].result.success is False
    assert tool.executions == 0


# 函数说明：test_runtime_cleans_run_scoped_permission_rules
# 用途：回归验证回归测试与测试辅助中的 `runtime_cleans_run_scoped_permission_rules` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ToolRegistry` → `tools.register` → `ApprovalCountingTool`；另有 5 个调用
# 点。
# 分支与异常：
#   验证条件：`await store.list() == ()`。
@pytest.mark.asyncio
async def test_runtime_cleans_run_scoped_permission_rules() -> None:
    registry, _ = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="approval-1",
                        name="approval_count",
                        arguments={"value": 1},
                    ),
                )
            ),
            model_response(content="完成"),
        ]
    )
    tools = ToolRegistry()
    tools.register(ApprovalCountingTool())
    store = InMemoryPermissionRuleStore()

    await AgentRuntime(
        registry,
        tools,
        provider="fake",
        approval_gate=RememberRunGate(),
        rule_store=store,
    ).run("执行审批工具", conversation_id="conversation-1")

    assert await store.list() == ()


# 函数说明：test_closing_run_stream_cancels_background_model_request
# 用途：回归验证回归测试与测试辅助中的
# `closing_run_stream_cancels_background_model_request` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `BlockingModelAdapter` → `ModelAdapterRegistry` → `ModelSettings` →
# `registry.register`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`first_event.type is AgentEventType.AGENT_STARTED`。
#   验证条件：`adapter.cancelled is True`。
@pytest.mark.asyncio
async def test_closing_run_stream_cancels_background_model_request() -> None:
    config = ProviderConfig(
        provider="blocking",
        model="blocking-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = BlockingModelAdapter(config)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("blocking", lambda _: adapter, config=config)
    runtime = AgentRuntime(registry, ToolRegistry(), provider="blocking")
    stream = runtime.run_stream("hello")

    first_event = await anext(stream)
    await adapter.started.wait()
    await stream.aclose()

    assert first_event.type is AgentEventType.AGENT_STARTED
    assert adapter.cancelled is True


# 函数说明：test_runtime_checkpoint_records_completed_tool_run
# 用途：回归验证回归测试与测试辅助中的 `runtime_checkpoint_records_completed_tool_run`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteCheckpointStore` →
# `checkpoint_store.initialize` → `fake_registry` → `model_response` → `ToolCall` →
# `ToolRegistry`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`checkpoint is not None`。
#   验证条件：`checkpoint.status is CheckpointStatus.COMPLETED`。
#   验证条件：`checkpoint.phase is CheckpointPhase.FINISHED`。
@pytest.mark.asyncio
async def test_runtime_checkpoint_records_completed_tool_run(tmp_path) -> None:
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await checkpoint_store.initialize()
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
    tools = ToolRegistry()
    tools.register(CountingTool())

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        checkpoint_store=checkpoint_store,
    ).run("执行工具", conversation_id="conv-1")
    checkpoint = await checkpoint_store.get(result.run_id)

    assert result.ok is True
    assert checkpoint is not None
    assert checkpoint.status is CheckpointStatus.COMPLETED
    assert checkpoint.phase is CheckpointPhase.FINISHED
    assert checkpoint.pending_tool_calls == ()
    assert [item.tool_call_id for item in checkpoint.completed_tool_results] == [
        "count-1"
    ]


# 函数说明：test_runtime_checkpoint_records_structured_failure
# 用途：回归验证回归测试与测试辅助中的 `runtime_checkpoint_records_structured_failure`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteCheckpointStore` →
# `checkpoint_store.initialize` → `fake_registry` →
# `AgentRuntime(registry, ToolRegistry(), provider='fake', checkpoint_store=…` →
# `AgentRuntime` → `ToolRegistry`。
# 分支与异常：
#   验证条件：`result.stop_reason is AgentStopReason.MODEL_ERROR`。
#   验证条件：`checkpoint is not None`。
#   验证条件：`checkpoint.status is CheckpointStatus.FAILED`。
#   验证条件：`checkpoint.stop_reason is AgentStopReason.MODEL_ERROR`。
@pytest.mark.asyncio
async def test_runtime_checkpoint_records_structured_failure(tmp_path) -> None:
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await checkpoint_store.initialize()
    registry, _ = fake_registry([RuntimeError("offline")])

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        checkpoint_store=checkpoint_store,
    ).run("hello", conversation_id="conv-1")
    checkpoint = await checkpoint_store.get(result.run_id)

    assert result.stop_reason is AgentStopReason.MODEL_ERROR
    assert checkpoint is not None
    assert checkpoint.status is CheckpointStatus.FAILED
    assert checkpoint.stop_reason is AgentStopReason.MODEL_ERROR
    assert "offline" in (checkpoint.error or "")


# 函数说明：test_runtime_cancellation_preserves_model_request_checkpoint
# 用途：回归验证回归测试与测试辅助中的
# `runtime_cancellation_preserves_model_request_checkpoint` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteCheckpointStore` →
# `checkpoint_store.initialize` → `ProviderConfig` → `SecretStr` →
# `BlockingModelAdapter` → `ModelAdapterRegistry`；另有 10 个调用点。
# 分支与异常：
#   验证条件：`len(checkpoints) == 1`。
#   验证条件：`checkpoints[0].status is CheckpointStatus.INTERRUPTED`。
#   验证条件：`checkpoints[0].phase is CheckpointPhase.MODEL_REQUEST`。
#   验证条件：`checkpoints[0].step == 1`。
#   预期异常：`pytest.raises(asyncio.CancelledError)`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
@pytest.mark.asyncio
async def test_runtime_cancellation_preserves_model_request_checkpoint(
    tmp_path,
) -> None:
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await checkpoint_store.initialize()
    config = ProviderConfig(
        provider="blocking",
        model="blocking-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = BlockingModelAdapter(config)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("blocking", lambda _: adapter, config=config)
    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="blocking",
        checkpoint_store=checkpoint_store,
    )

    running = asyncio.create_task(runtime.run("hello", conversation_id="conv-1"))
    await adapter.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    checkpoints = await checkpoint_store.list(conversation_id="conv-1")
    assert len(checkpoints) == 1
    assert checkpoints[0].status is CheckpointStatus.INTERRUPTED
    assert checkpoints[0].phase is CheckpointPhase.MODEL_REQUEST
    assert checkpoints[0].step == 1


# 函数说明：test_runtime_cancellation_preserves_uncertain_tool_call
# 用途：回归验证回归测试与测试辅助中的
# `runtime_cancellation_preserves_uncertain_tool_call` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteCheckpointStore` →
# `checkpoint_store.initialize` → `ToolCall` → `fake_registry` → `model_response` →
# `BlockingTool`；另有 9 个调用点。
# 分支与异常：
#   验证条件：`len(checkpoints) == 1`。
#   验证条件：`checkpoints[0].status is CheckpointStatus.INTERRUPTED`。
#   验证条件：`checkpoints[0].phase is CheckpointPhase.TOOL_EXECUTION`。
#   验证条件：`checkpoints[0].pending_tool_calls == (call,)`。
#   预期异常：`pytest.raises(asyncio.CancelledError)`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
@pytest.mark.asyncio
async def test_runtime_cancellation_preserves_uncertain_tool_call(tmp_path) -> None:
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await checkpoint_store.initialize()
    call = ToolCall(id="uncertain-tool", name="blocking_tool", arguments={})
    registry, _ = fake_registry([model_response(tool_calls=(call,))])
    tool = BlockingTool()
    tools = ToolRegistry()
    tools.register(tool)
    runtime = AgentRuntime(
        registry,
        tools,
        provider="fake",
        checkpoint_store=checkpoint_store,
    )

    running = asyncio.create_task(runtime.run("执行阻塞工具", conversation_id="conv-1"))
    await tool.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    checkpoints = await checkpoint_store.list(conversation_id="conv-1")
    assert len(checkpoints) == 1
    assert checkpoints[0].status is CheckpointStatus.INTERRUPTED
    assert checkpoints[0].phase is CheckpointPhase.TOOL_EXECUTION
    assert checkpoints[0].pending_tool_calls == (call,)
    assert checkpoints[0].completed_tool_results == ()
    assert tool.cancelled is True


# 函数说明：test_runtime_injects_interrupted_checkpoint_without_persisting_it
# 用途：回归验证回归测试与测试辅助中的
# `runtime_injects_interrupted_checkpoint_without_persisting_it` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteCheckpointStore` →
# `checkpoint_store.initialize` → `ToolCall` → `checkpoint_store.start` → `Message` →
# `checkpoint_store.before_model`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`'禁止直接重试' in (injected.content or '')`。
#   验证条件：`'uncertain-1' in (injected.content or '')`。
#   验证条件：`not any((message.name == CHECKPOINT_CONTEXT_MESSAGE_NAME for message in
# result.messages))`。
#   验证条件：`old is not None and old.recovered_by_run_id == result.run_id`。
@pytest.mark.asyncio
async def test_runtime_injects_interrupted_checkpoint_without_persisting_it(
    tmp_path,
) -> None:

    checkpoint_store = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await checkpoint_store.initialize()
    uncertain = ToolCall(
        id="uncertain-1",
        name="write_file",
        arguments={"path": "output.md"},
    )
    await checkpoint_store.start(
        "old-run",
        conversation_id="conv-1",
        user_message=Message(role=MessageRole.USER, content="写入 output.md"),
    )
    await checkpoint_store.before_model("old-run", step=2)
    await checkpoint_store.before_tools(
        "old-run",
        step=2,
        tool_calls=(uncertain,),
    )
    await checkpoint_store.interrupt("old-run", error="process stopped")
    registry, adapter = fake_registry([model_response(content="已核对中断状态")])

    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        checkpoint_store=checkpoint_store,
    ).run(
        "继续",
        conversation_id="conv-1",
        recovery_run_id="old-run",
    )

    injected = next(
        message
        for message in adapter.requests[0].messages
        if message.name == CHECKPOINT_CONTEXT_MESSAGE_NAME
    )
    assert "禁止直接重试" in (injected.content or "")
    assert "uncertain-1" in (injected.content or "")
    assert not any(
        message.name == CHECKPOINT_CONTEXT_MESSAGE_NAME for message in result.messages
    )
    old = await checkpoint_store.get("old-run")
    assert old is not None and old.recovered_by_run_id == result.run_id


# 函数说明：test_runtime_plain_start_does_not_inject_interrupted_checkpoint
# 用途：回归验证回归测试与测试辅助中的
# `runtime_plain_start_does_not_inject_interrupted_checkpoint` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteCheckpointStore` →
# `checkpoint_store.initialize` → `ToolCall` → `checkpoint_store.start` → `Message` →
# `checkpoint_store.before_model`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`not any((message.name == CHECKPOINT_CONTEXT_MESSAGE_NAME for message in
# adapter.requests[0].messages))`。
#   验证条件：`old is not None and old.recovered_by_run_id is None`。
@pytest.mark.asyncio
async def test_runtime_plain_start_does_not_inject_interrupted_checkpoint(
    tmp_path,
) -> None:

    checkpoint_store = SQLiteCheckpointStore(tmp_path / "muharness.db")
    await checkpoint_store.initialize()
    uncertain = ToolCall(
        id="uncertain-1",
        name="write_file",
        arguments={"path": "output.md"},
    )
    await checkpoint_store.start(
        "old-run",
        conversation_id="conv-1",
        user_message=Message(role=MessageRole.USER, content="写入 output.md"),
    )
    await checkpoint_store.before_model("old-run", step=2)
    await checkpoint_store.before_tools(
        "old-run",
        step=2,
        tool_calls=(uncertain,),
    )
    await checkpoint_store.interrupt("old-run", error="process stopped")
    registry, adapter = fake_registry([model_response(content="已核对中断状态")])

    await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        checkpoint_store=checkpoint_store,
    ).run("继续", conversation_id="conv-1")

    assert not any(
        message.name == CHECKPOINT_CONTEXT_MESSAGE_NAME
        for message in adapter.requests[0].messages
    )
    old = await checkpoint_store.get("old-run")
    assert old is not None and old.recovered_by_run_id is None


# 函数说明：test_runtime_injects_memory_context_without_persisting
# 用途：回归验证回归测试与测试辅助中的
# `runtime_injects_memory_context_without_persisting` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `FakeMemoryManager` → `AgentRuntime` → `ToolRegistry` → `runtime.run`。
# 分支与异常：
#   验证条件：`CORE_MEMORY_MESSAGE_NAME in request_names`。
#   验证条件：`MEMORY_INDEX_MESSAGE_NAME in request_names`。
#   验证条件：`MEMORY_POLICY_MESSAGE_NAME in request_names`。
#   验证条件：`not any((message.name in {CORE_MEMORY_MESSAGE_NAME,
# MEMORY_INDEX_MESSAGE_NAME, MEMORY_POLICY_MESSAGE_NAME} for message in…`。
@pytest.mark.asyncio
async def test_runtime_injects_memory_context_without_persisting() -> None:
    registry, adapter = fake_registry([model_response(content="我会使用中文回答")])
    memory = FakeMemoryManager()
    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        memory_manager=memory,
    )
    result = await runtime.run("继续回答", conversation_id="conv-1")

    request_names = [message.name for message in adapter.requests[0].messages]
    assert CORE_MEMORY_MESSAGE_NAME in request_names
    assert MEMORY_INDEX_MESSAGE_NAME in request_names
    assert MEMORY_POLICY_MESSAGE_NAME in request_names
    assert not any(
        message.name
        in {
            CORE_MEMORY_MESSAGE_NAME,
            MEMORY_INDEX_MESSAGE_NAME,
            MEMORY_POLICY_MESSAGE_NAME,
        }
        for message in result.messages
    )


# 函数说明：test_runtime_core_update_uses_current_user_and_next_run_loads_it
# 用途：回归验证回归测试与测试辅助中的
# `runtime_core_update_uses_current_user_and_next_run_loads_it` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `fake_registry` →
# `model_response` → `MemoryManager` → `memory.initialize` → `ToolRegistry`；另有 4 个调
# 用点。
# 分支与异常：
#   验证条件：`first.tool_calls[0].result.success is True`。
#   验证条件：`second.content == '你好，我会继续使用中文。'`。
#   验证条件：`'始终使用中文交流' in (injected_core.content or '')`。
@pytest.mark.asyncio
async def test_runtime_core_update_uses_current_user_and_next_run_loads_it(
    tmp_path,
) -> None:
    statement = "以后都使用中文和我交流"
    core_call = ToolCall(
        id="core-update-1",
        name="core_memory_update",
        arguments={
            "key": "communication.language",
            "value": "始终使用中文交流。",
            "reason": "用户明确表达全局长期偏好",
            "explicit_user_statement": statement,
        },
    )
    registry, adapter = fake_registry(
        [
            model_response(tool_calls=(core_call,)),
            model_response(content="已经记住你的长期偏好。"),
            model_response(content="你好，我会继续使用中文。"),
        ]
    )
    memory = MemoryManager(tmp_path / "memory")
    await memory.initialize()
    tools = ToolRegistry()
    register_memory_tools(tools, memory)
    runtime = AgentRuntime(
        registry,
        tools,
        provider="fake",
        memory_manager=memory,
    )

    first = await runtime.run(f"请记住，{statement}。")
    second = await runtime.run("你好")

    assert first.tool_calls[0].result.success is True
    assert second.content == "你好，我会继续使用中文。"
    injected_core = next(
        message
        for message in adapter.requests[2].messages
        if message.name == CORE_MEMORY_MESSAGE_NAME
    )
    assert "始终使用中文交流" in (injected_core.content or "")


# 函数说明：test_memory_context_failure_does_not_block_agent_result
# 用途：回归验证回归测试与测试辅助中的
# `memory_context_failure_does_not_block_agent_result` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `AgentRuntime` → `ToolRegistry` → `FailingMemoryManager` → `asyncio.wait_for`；另有 1
# 个调用点。
# 分支与异常：
#   验证条件：`result.content == '最终回答'`。
@pytest.mark.asyncio
async def test_memory_context_failure_does_not_block_agent_result() -> None:
    registry, _ = fake_registry([model_response(content="最终回答")])

    class FailingMemoryManager:
        # 函数说明：test_memory_context_failure_does_not_block_agent_result.
        # FailingMemoryManager.context_messages
        # 用途：处理回归测试与测试辅助中的 `context_messages` 数据；结果及边界条件见下方
        # 说明。
        # 返回：类型 `tuple[Message, ...]`；不返回结果值（隐式 None）。
        async def context_messages(self) -> tuple[Message, ...]:
            raise RuntimeError("memory unavailable")

    runtime = AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        memory_manager=FailingMemoryManager(),
    )

    result = await asyncio.wait_for(runtime.run("继续"), timeout=5)
    assert result.content == "最终回答"


class DeterministicTokenEstimator(TokenEstimator):
    # 函数说明：DeterministicTokenEstimator.estimate_text
    # 用途：估算文本，供回归测试与测试辅助使用。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    # 返回：类型 `int`；返回 `len(text)`。
    def estimate_text(
        self,
        text: str,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        return len(text)

    # 函数说明：DeterministicTokenEstimator.estimate_messages
    # 用途：估算消息序列，供回归测试与测试辅助使用。
    # 参数：
    #   messages：本次处理的消息序列。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    # 返回：类型 `int`；返回 `total`。
    def estimate_messages(
        self,
        messages,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        total = 0
        for message in messages:
            total += 4 + len(message.content or "")
            if message.tool_calls:
                total += sum(8 + len(call.name) for call in message.tool_calls)
        return total

    # 函数说明：DeterministicTokenEstimator.estimate_tools
    # 用途：估算工具集合，供回归测试与测试辅助使用。
    # 参数：
    #   tools：可用工具定义或工具实例集合。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    # 返回：类型 `int`；返回 `10 * len(tools)`。
    def estimate_tools(
        self,
        tools,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        return 10 * len(tools)

    # 函数说明：DeterministicTokenEstimator.estimate_request
    # 用途：估算请求，供回归测试与测试辅助使用。
    # 参数：
    #   messages：本次处理的消息序列。
    #   tools：可用工具定义或工具实例集合；默认 `()`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    # 返回：类型 `int`；返回
    # `self.estimate_messages(messages) + self.estimate_tools(tools)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.estimate_messages` →
    # `self.estimate_tools`。
    def estimate_request(
        self,
        messages,
        *,
        tools=(),
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        return self.estimate_messages(messages) + self.estimate_tools(tools)


class SizedEchoTool(BaseTool):
    # 函数说明：SizedEchoTool.__init__
    # 用途：初始化 SizedEchoTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   payload：传输或持久化载荷，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    # 副作用与资源：
    #   更新对象字段：`self._definition`、`self._payload`。
    def __init__(self, name: str, payload: str) -> None:
        self._definition = ToolDefinition(
            name=name,
            description="Return fixed size text",
            parameters={"type": "object", "properties": {}},
        )
        self._payload = payload

    # 函数说明：SizedEchoTool.definition
    # 用途：提供 SizedEchoTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `self._definition`。
    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    # 函数说明：SizedEchoTool.execute
    # 用途：执行SizedEchoTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, object]`。
    # 返回：类型 `str`；返回 `self._payload`。
    async def execute(self, arguments: dict[str, object]) -> str:
        return self._payload


class CountingFixedSummarizer(FixedContextSummarizer):
    # 函数说明：CountingFixedSummarizer.__init__
    # 用途：初始化 CountingFixedSummarizer；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.calls`。
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    # 函数说明：CountingFixedSummarizer.summarize
    # 用途：生成摘要CountingFixedSummarizer，供回归测试与测试辅助使用。
    # 参数：
    #   previous_summary：已有会话摘要。
    #   messages：本次处理的消息序列。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `SummaryGenerationResult`；返回 `await super().summarize(
    # previous_summary, messages, max_output_tokens=max_output_tokens)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().summarize` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.calls`。
    async def summarize(
        self,
        previous_summary,
        messages,
        *,
        max_output_tokens: int | None = None,
    ) -> SummaryGenerationResult:
        self.calls += 1
        return await super().summarize(
            previous_summary, messages, max_output_tokens=max_output_tokens
        )


# 函数说明：_central_budget_policy
# 用途：返回 `ContextBudgetPolicy(…)`，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `ContextBudgetPolicy`；返回 `ContextBudgetPolicy(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudgetPolicy`。
def _central_budget_policy() -> ContextBudgetPolicy:
    return ContextBudgetPolicy(
        safety_margin_tokens=100,
        working_trigger_ratio=0.70,
        compact_input_tokens=1_222,
    )


# 函数说明：test_final_budget_counts_instructions_added_after_summary
# 用途：回归验证回归测试与测试辅助中的
# `final_budget_counts_instructions_added_after_summary` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   finalization_tokens：Token 数量或 Token 预算。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ModelCapabilityRegistry` → `capabilities.register_override` →
# `DeterministicTokenEstimator` → `CountingFixedSummarizer`；另有 13 个调用点。
# 分支与异常：
#   验证条件：`result.ok`。
#   验证条件：`summarizer.calls == 1`。
#   验证条件：`started.estimated_input_tokens == estimator.estimate_request(request.
# messages, tools=request.tools)`。
#   验证条件：`started.input_budget == 4000 - request.max_output_tokens - 100`。
@pytest.mark.asyncio
@pytest.mark.parametrize("finalization_tokens", [8, 20])
async def test_final_budget_counts_instructions_added_after_summary(
    finalization_tokens,
) -> None:
    registry, adapter = fake_registry([model_response(content="完成")])
    capabilities = ModelCapabilityRegistry()
    capabilities.register_override(
        "fake", "fake-model", context_window=4000, max_output_tokens=512
    )
    estimator = DeterministicTokenEstimator()
    summarizer = CountingFixedSummarizer()
    manager = ContextManager(
        estimator=estimator,
        registry=capabilities,
        budget_policy=ContextBudgetPolicy(
            safety_margin_tokens=100, compact_input_tokens=1000
        ),
        conversation_reducer=ConversationReducer(
            summarizer, keep_recent_conversation_blocks=1
        ),
    )
    tools = ToolRegistry()
    tools.register(CountingTool())
    events = InMemoryEventHandler()
    history = tuple(
        message
        for i in range(3)
        for message in (
            Message(role=MessageRole.USER, content=f"u{i}" + "x" * 300),
            Message(role=MessageRole.ASSISTANT, content="y" * 300),
        )
    )
    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_output_tokens=256,
        context_manager=manager,
        run_budget_config=RunBudgetConfig(
            _env_file=None,
            warning_tokens=3,
            finalization_tokens=finalization_tokens,
            hard_tokens=100,
            finalization_max_output_tokens=64,
        ),
    ).run("当前要求", history=history, event_handler=events)
    assert result.ok
    assert summarizer.calls == 1
    request = adapter.requests[0]
    started = next(
        event for event in events.events if event.type is AgentEventType.MODEL_STARTED
    )
    assert started.estimated_input_tokens == estimator.estimate_request(
        request.messages, tools=request.tools
    )
    assert started.input_budget == 4000 - request.max_output_tokens - 100
    assert "预算" in (request.messages[-1].content or "")
    if finalization_tokens == 8:
        assert request.tools == ()
        assert request.max_output_tokens == 64
    else:
        assert request.tools
        assert request.max_output_tokens == 256


# 函数说明：test_failed_compaction_is_not_retried_for_same_raw_response_repair
# 用途：回归验证回归测试与测试辅助中的
# `failed_compaction_is_not_retried_for_same_raw_response_repair` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FailingSummarizer` → `ContextManager`
#  → `DeterministicTokenEstimator` → `ContextBudgetPolicy` → `ConversationReducer` →
# `fake_registry`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`result.ok`。
#   验证条件：`summarizer.calls == 2`。
#   验证条件：`len(adapter.requests) == 2`。
#   验证条件：`result.summary_state is None`。
@pytest.mark.asyncio
async def test_failed_compaction_is_not_retried_for_same_raw_response_repair() -> None:
    class FailingSummarizer(CountingFixedSummarizer):
        # 函数说明：test_failed_compaction_is_not_retried_for_same_raw_response_repair.
        # FailingSummarizer.summarize
        # 用途：生成摘要FailingSummarizer，供回归测试与测试辅助使用。
        # 参数：
        #   previous_summary：已有会话摘要。
        #   messages：本次处理的消息序列。
        #   max_output_tokens：模型输出 Token 上限；默认 `None`。
        # 返回：不返回结果值（隐式 None）。
        # 副作用与资源：
        #   更新对象字段：`self.calls`。
        async def summarize(
            self, previous_summary, messages, *, max_output_tokens=None
        ):
            self.calls += 1
            raise RuntimeError("offline summary unavailable")

    summarizer = FailingSummarizer()
    manager = ContextManager(
        estimator=DeterministicTokenEstimator(),
        budget_policy=ContextBudgetPolicy(
            safety_margin_tokens=100, compact_input_tokens=1000
        ),
        conversation_reducer=ConversationReducer(
            summarizer, keep_recent_conversation_blocks=1
        ),
    )
    registry, adapter = fake_registry(
        [model_response(content=""), model_response(content="已恢复")]
    )
    events = InMemoryEventHandler()
    history = (
        Message(role=MessageRole.USER, content="旧要求" + "x" * 800),
        Message(role=MessageRole.ASSISTANT, content="y" * 800),
    )
    result = await AgentRuntime(
        registry, ToolRegistry(), provider="fake", context_manager=manager
    ).run(
        "最新要求",
        history=history,
        event_handler=events,
    )
    assert result.ok
    assert summarizer.calls == 2  # One attempt and one bounded repair only.
    assert len(adapter.requests) == 2
    assert result.summary_state is None
    assert result.messages[: len(history)] == history
    started = [
        event for event in events.events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert "suppressed" in (started[1].summary_error or "")


# 函数说明：test_runtime_appends_until_central_pressure_then_compacts_once
# 用途：回归验证回归测试与测试辅助中的
# `runtime_appends_until_central_pressure_then_compacts_once` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SizedEchoTool` → `fake_registry` →
# `model_response` → `ToolCall` → `ModelCapabilityRegistry` →
# `capability_registry.register_override`；另有 11 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`len(started) == 3`。
#   验证条件：`started[0].prefix_decision == 'rebuild'`。
#   验证条件：`started[0].requires_compaction is False`。
@pytest.mark.asyncio
async def test_runtime_appends_until_central_pressure_then_compacts_once() -> None:

    # Leave deliberate room on both sides of the 1_222-token compact ceiling.
    # The result envelope contains runtime metadata whose serialized length can
    # vary by a few characters without changing the behavior under test.
    tool_a = SizedEchoTool("echo_a", "X" * 950)
    tool_b = SizedEchoTool("echo_b", "Y" * 100)
    model_registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(ToolCall(id="call-1", name="echo_a", arguments={}),)
            ),
            model_response(
                tool_calls=(ToolCall(id="call-2", name="echo_b", arguments={}),)
            ),
            model_response(content="完成"),
        ]
    )
    capability_registry = ModelCapabilityRegistry()
    capability_registry.register_override(
        "fake",
        "fake-model",
        context_window=2_140,
        max_output_tokens=512,
    )
    summarizer = CountingFixedSummarizer()
    context_manager = ContextManager(
        estimator=DeterministicTokenEstimator(),
        registry=capability_registry,
        budget_policy=_central_budget_policy(),
        conversation_reducer=ConversationReducer(
            summarizer,
            keep_recent_conversation_blocks=1,
            keep_recent_tool_rounds=1,
        ),
    )
    tools = ToolRegistry()
    tools.register(tool_a)
    tools.register(tool_b)
    events = InMemoryEventHandler()

    result = await AgentRuntime(
        model_registry,
        tools,
        provider="fake",
        max_output_tokens=512,
        context_manager=context_manager,
    ).run(
        "go",
        history=(
            Message(role=MessageRole.USER, content="u1"),
            Message(role=MessageRole.ASSISTANT, content="a1"),
        ),
        event_handler=events,
    )

    assert result.ok is True
    started = [
        event for event in events.events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert len(started) == 3
    assert started[0].prefix_decision == "rebuild"
    assert started[0].requires_compaction is False
    assert started[0].cache_prefix_reused is False
    assert started[1].prefix_decision == "append"
    assert started[1].requires_compaction is False
    assert started[1].cache_prefix_reused is True
    assert started[1].summary_updated is False
    assert started[2].prefix_decision == "compact"
    assert started[2].cache_prefix_reused is False
    assert started[2].summary_updated is True
    assert summarizer.calls == 1
    assert (
        adapter.requests[1].messages[: len(adapter.requests[0].messages)]
        == adapter.requests[0].messages
    )
    assert result.messages[0].content == "u1"
    assert started[0].original_estimated_input_tokens == 38
    assert 1_069 <= started[1].original_estimated_input_tokens < 1_222
    assert 1_222 <= started[2].original_estimated_input_tokens < 1_528


# 函数说明：test_runtime_block_count_does_not_rewrite_reusable_prefix
# 用途：回归验证回归测试与测试辅助中的
# `runtime_block_count_does_not_rewrite_reusable_prefix` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`fake_registry` → `model_response` →
# `ToolCall` → `ModelCapabilityRegistry` → `capability_registry.register_override` →
# `Message`；另有 12 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`len(started) == 2`。
#   验证条件：`started[1].original_estimated_input_tokens < started[1].trigger_tokens`。
#   验证条件：`started[1].conversation_block_triggered is False`。
@pytest.mark.asyncio
async def test_runtime_block_count_does_not_rewrite_reusable_prefix() -> None:

    model_registry, _ = fake_registry(
        [
            model_response(
                tool_calls=(ToolCall(id="call-1", name="echo_a", arguments={}),)
            ),
            model_response(content="完成"),
        ]
    )
    capability_registry = ModelCapabilityRegistry()
    capability_registry.register_override(
        "fake",
        "fake-model",
        context_window=200_000,
        max_output_tokens=8_192,
    )
    history = tuple(
        message
        for index in range(3)
        for message in (
            Message(role=MessageRole.USER, content=f"问题 {index}"),
            Message(role=MessageRole.ASSISTANT, content=f"回答 {index}"),
        )
    )
    context_manager = ContextManager(
        estimator=DeterministicTokenEstimator(),
        registry=capability_registry,
        budget_policy=ContextBudgetPolicy(safety_margin_tokens=100),
        context_settings=ContextSettings(
            _env_file=None,
            context_max_unsummarized_conversation_blocks=2,
        ),
        conversation_reducer=ConversationReducer(
            FixedContextSummarizer(),
            keep_recent_conversation_blocks=3,
        ),
    )
    tools = ToolRegistry()
    tools.register(SizedEchoTool("echo", "ok"))
    events = InMemoryEventHandler()

    result = await AgentRuntime(
        model_registry,
        tools,
        provider="fake",
        max_output_tokens=512,
        context_manager=context_manager,
    ).run(
        "go",
        history=history,
        event_handler=events,
    )

    assert result.ok is True
    started = [
        event for event in events.events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert len(started) == 2
    assert started[1].original_estimated_input_tokens < started[1].trigger_tokens
    assert started[1].conversation_block_triggered is False
    assert started[1].requires_compaction is False
    assert started[1].prefix_decision == "append"
    assert started[1].cache_prefix_reused is True


# 函数说明：test_empty_retry_budget_and_diagnostics
# 用途：回归验证回归测试与测试辅助中的 `empty_retry_budget_and_diagnostics` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   finish_reason：原因输入或配置值。
#   explicit_limit：上限输入或配置值。
#   expected：`expected`输入或配置值。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`model_response` → `fake_registry` →
# `ModelCapabilityRegistry` → `capabilities.register_override` → `InMemoryEventHandler`
# → `AgentRuntime(registry, ToolRegistry(), provider='fake', max_output_tokens=…`；另有
# 3 个调用点。
# 分支与异常：
#   验证条件：`result.ok`。
#   验证条件：`adapter.requests[1].max_output_tokens == expected`。
#   验证条件：`completed[0].model_finish_reason == finish_reason`。
#   验证条件：`completed[0].reasoning_chars == len('internal')`。
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason,explicit_limit,expected",
    [
        ("max_tokens", None, 8192),
        ("length", None, 8192),
        ("end_turn", None, 4096),
        ("max_tokens", 2048, 2048),
    ],
)
async def test_empty_retry_budget_and_diagnostics(
    finish_reason,
    explicit_limit,
    expected,
) -> None:
    truncated = model_response(content=None, reasoning="internal").model_copy(
        update={"finish_reason": finish_reason}
    )
    registry, adapter = fake_registry([truncated, model_response(content="done")])
    capabilities = ModelCapabilityRegistry()
    capabilities.register_override("fake", "fake-model", max_output_tokens=16384)
    events = InMemoryEventHandler()
    result = await AgentRuntime(
        registry,
        ToolRegistry(),
        provider="fake",
        max_output_tokens=explicit_limit,
        context_manager=ContextManager(registry=capabilities),
    ).run("finish the task", event_handler=events)
    assert result.ok
    assert adapter.requests[1].max_output_tokens == expected
    completed = [e for e in events.events if e.type is AgentEventType.MODEL_COMPLETED]
    assert completed[0].model_finish_reason == finish_reason
    assert completed[0].reasoning_chars == len("internal")
    assert completed[0].message.reasoning is None
    assert completed[0].model_duration_ms >= 0
    assert completed[1].requested_max_output_tokens == expected


# 函数说明：test_truncated_empty_retry_stays_bounded
# 用途：回归验证回归测试与测试辅助中的 `truncated_empty_retry_stays_bounded` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`model_response` → `fake_registry` →
# `AgentRuntime(registry, ToolRegistry(), provider='fake').run` → `AgentRuntime` →
# `ToolRegistry`。
# 分支与异常：
#   验证条件：`not result.ok`。
#   验证条件：`len(adapter.requests) == 2`。
#   验证条件：`[r.max_output_tokens for r in adapter.requests] == [4096, 4096]`。
@pytest.mark.asyncio
async def test_truncated_empty_retry_stays_bounded() -> None:
    empty = model_response(content=None).model_copy(
        update={"finish_reason": "max_tokens"}
    )
    registry, adapter = fake_registry([empty, empty, empty])
    result = await AgentRuntime(registry, ToolRegistry(), provider="fake").run("work")
    assert not result.ok
    # 截断空回复最多恢复 2 次（共 3 次请求），之后报错，不会无限重试
    assert len(adapter.requests) == 3
    assert [r.max_output_tokens for r in adapter.requests] == [4096, 4096, 4096]
    assert result.error is not None
    assert "truncated by the output limit 3 times" in result.error.message
    for request in adapter.requests[1:]:
        assert request.messages[-1].name == RUNTIME_NOTICE_NAME
        assert "输出上限截断" in (request.messages[-1].content or "")


# 贴近真实执行者：角色显式指定了输出上限（不扩容），思考连续两次写满上限，
# 第三次在“缩小动作、立即调用工具”的提示下成功。
@pytest.mark.asyncio
async def test_truncated_empty_recovers_with_explicit_limit() -> None:
    empty = model_response(content=None).model_copy(
        update={"finish_reason": "max_tokens"}
    )
    registry, adapter = fake_registry(
        [
            empty,
            empty,
            model_response(
                tool_calls=(ToolCall(id="c1", name="count", arguments={"value": 1}),)
            ),
            model_response(content="done"),
        ]
    )
    tools = ToolRegistry()
    tools.register(CountingTool())
    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_output_tokens=2048,
    ).run("work")
    assert result.ok
    assert result.content == "done"
    assert [r.max_output_tokens for r in adapter.requests] == [2048] * 4
    notices = [
        m for m in adapter.requests[2].messages if m.name == RUNTIME_NOTICE_NAME
    ]
    # 两次截断的提示都保留在已发送前缀里，提示内容要求立即调用工具
    assert len(notices) == 2
    assert all("立即执行" in (m.content or "") for m in notices)
    assert _anthropic_prefix_kept(adapter.requests[0], adapter.requests[1])
    assert _anthropic_prefix_kept(adapter.requests[1], adapter.requests[2])


# 函数说明：test_truncated_retry_budget_resets_after_tool_response
# 用途：回归验证回归测试与测试辅助中的
# `truncated_retry_budget_resets_after_tool_response` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`model_response` → `fake_registry` →
# `ToolCall` → `ModelCapabilityRegistry` → `capabilities.register_override` →
# `ToolRegistry`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.ok`。
#   验证条件：`[r.max_output_tokens for r in adapter.requests] == [4096, 8192, 4096]`。
@pytest.mark.asyncio
async def test_truncated_retry_budget_resets_after_tool_response() -> None:
    empty = model_response(content=None).model_copy(
        update={"finish_reason": "max_tokens"}
    )
    registry, adapter = fake_registry(
        [
            empty,
            model_response(
                tool_calls=(ToolCall(id="count", name="count", arguments={"value": 1}),)
            ),
            model_response(content="done"),
        ]
    )
    capabilities = ModelCapabilityRegistry()
    capabilities.register_override("fake", "fake-model", max_output_tokens=16384)
    tools = ToolRegistry()
    tools.register(CountingTool())
    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        context_manager=ContextManager(registry=capabilities),
    ).run("work")
    assert result.ok
    assert [r.max_output_tokens for r in adapter.requests] == [4096, 8192, 4096]


# 函数说明：test_partial_final_preserves_model_truncation_metadata
# 用途：回归验证回归测试与测试辅助中的
# `partial_final_preserves_model_truncation_metadata` 场景，下方断言说明列出实际通过条件
# 。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`model_response` → `fake_registry` →
# `AgentRuntime(registry, ToolRegistry(), provider='fake').run` → `AgentRuntime` →
# `ToolRegistry` → `result.content.endswith`。
# 分支与异常：
#   验证条件：`result.model_finish_reason == 'max_tokens'`。
#   验证条件：`result.content.endswith('partial report')`。
@pytest.mark.asyncio
async def test_partial_final_preserves_model_truncation_metadata():
    response = model_response(content="状态: complete\npartial report").model_copy(
        update={"finish_reason": "max_tokens"}
    )
    registry, _ = fake_registry([response])
    result = await AgentRuntime(registry, ToolRegistry(), provider="fake").run("audit")
    assert result.model_finish_reason == "max_tokens"
    assert result.content.endswith("partial report")


# 函数说明：test_audit_report_recovery_keeps_evidence_without_tools
# 用途：回归验证回归测试与测试辅助中的
# `audit_report_recovery_keeps_evidence_without_tools` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   retry_finish：`retry_finish`输入或配置值。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'proof.txt').write_text`
# → `model_response` → `fake_registry` → `ToolCall` → `ModelCapabilityRegistry` →
# `capabilities.register_override`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`len(result.tool_calls) == 1`。
#   验证条件：`len(adapter.requests) == 3`。
#   验证条件：`recovery.tools == ()`。
#   验证条件：`recovery.max_output_tokens == 8192`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'proof.txt').write_text`。
@pytest.mark.asyncio
@pytest.mark.parametrize("retry_finish", ["end_turn", "max_tokens"])
async def test_audit_report_recovery_keeps_evidence_without_tools(
    tmp_path, retry_finish
):
    from app.models.types import AgentMode

    (tmp_path / "proof.txt").write_text("verified evidence", encoding="utf-8")
    truncated = model_response(content="状态: complete\npartial").model_copy(
        update={"finish_reason": "max_tokens"}
    )
    final = model_response(content="full report").model_copy(
        update={"finish_reason": retry_finish}
    )
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="read",
                        name="read_file",
                        arguments={"path": "proof.txt"},
                    ),
                )
            ),
            truncated,
            final,
        ]
    )
    capabilities = ModelCapabilityRegistry()
    capabilities.register_override("fake", "fake-model", max_output_tokens=16384)
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_steps=3,
        context_manager=ContextManager(registry=capabilities),
    ).run("audit", mode=AgentMode.AUDIT)
    assert len(result.tool_calls) == 1
    assert len(adapter.requests) == 3
    recovery = adapter.requests[-1]
    assert recovery.tools == ()
    assert recovery.max_output_tokens == 8192
    assert any("verified evidence" in (m.content or "") for m in recovery.messages)
    assert not any("partial" in (m.content or "") for m in recovery.messages)
    assert result.model_finish_reason == retry_finish


# 回归：截断空回复 → 正常工具调用 → 再次截断空回复，不应被累计为“两次空回复”。
# 对应 run dab8eb0e…：第 4 步与第 6 步之间隔着成功的第 5 步，却被判定失败。
@pytest.mark.asyncio
async def test_empty_retry_counter_resets_after_tool_round() -> None:
    empty = model_response(content=None).model_copy(
        update={"finish_reason": "max_tokens"}
    )
    registry, adapter = fake_registry(
        [
            empty,
            model_response(
                tool_calls=(ToolCall(id="count", name="count", arguments={"value": 1}),)
            ),
            empty,
            model_response(content="done"),
        ]
    )
    capabilities = ModelCapabilityRegistry()
    capabilities.register_override("fake", "fake-model", max_output_tokens=16384)
    tools = ToolRegistry()
    tools.register(CountingTool())
    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        context_manager=ContextManager(registry=capabilities),
    ).run("work")
    assert result.ok
    assert result.content == "done"
    assert [r.max_output_tokens for r in adapter.requests] == [4096, 8192, 4096, 8192]
    # 响应修正提醒追加在末尾，不改动已发送前缀
    assert _anthropic_prefix_kept(adapter.requests[0], adapter.requests[1])
    assert _anthropic_prefix_kept(adapter.requests[2], adapter.requests[3])


# 回归：预算预警只追加在请求末尾，不改动已发送的前缀（否则整段缓存失效，
# 一次重算即可冲破硬上限，见 run 4aa87bff / c51a4e87）。
@pytest.mark.asyncio
async def test_budget_warning_appends_at_tail_and_keeps_prefix() -> None:
    def usage(uncached: int) -> ModelUsage:
        return ModelUsage(
            input_tokens=100,
            output_tokens=1,
            total_tokens=101,
            cached_input_tokens=100 - uncached,
            uncached_input_tokens=uncached,
        )

    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(ToolCall(id="c1", name="count", arguments={"value": 1}),),
                usage=usage(1),
            ),
            model_response(
                tool_calls=(ToolCall(id="c2", name="count", arguments={"value": 2}),),
                usage=usage(10),
            ),
            model_response(
                tool_calls=(ToolCall(id="c3", name="count", arguments={"value": 3}),)
            ),
            model_response(content="done"),
        ]
    )
    tools = ToolRegistry()
    tools.register(CountingTool())
    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        run_budget_config=RunBudgetConfig(
            _env_file=None,
            warning_tokens=5,
            finalization_tokens=1_000,
            hard_tokens=2_000,
            warning_model_calls=100,
            finalization_model_calls=101,
            hard_model_calls=102,
        ),
    ).run("计数")

    assert result.ok is True
    first, second, third, fourth = adapter.requests
    assert all(m.name != RUNTIME_NOTICE_NAME for m in second.messages)
    # 预警出现的那一步：已发送前缀不变，预警在最末尾
    assert third.messages[: len(second.messages)] == second.messages
    assert third.messages[-1].name == RUNTIME_NOTICE_NAME
    assert "预算预警" in (third.messages[-1].content or "")
    # 之后一步：继续沿用已发送前缀，预警不重复追加
    assert fourth.messages[: len(third.messages)] == third.messages
    assert sum(m.name == RUNTIME_NOTICE_NAME for m in fourth.messages) == 1
    assert first.tools == second.tools == third.tools == fourth.tools
    # 按适配器转换后同样保持前缀（含“新用户消息会丢思考”的规则）
    assert _anthropic_prefix_kept(first, second)
    assert _anthropic_prefix_kept(second, third)
    assert _anthropic_prefix_kept(third, fourth)


# 工具调用写到一半被输出上限截断（实测 write_file 只剩 path、缺 content）：
# 不执行半截调用，丢弃这条回复，按截断恢复；下一次完整调用才真正执行。
@pytest.mark.asyncio
async def test_truncated_tool_call_is_not_executed_and_recovers() -> None:
    truncated_call = model_response(
        tool_calls=(ToolCall(id="c1", name="count", arguments={"value": 1}),)
    ).model_copy(update={"finish_reason": "max_tokens"})
    registry, adapter = fake_registry(
        [
            truncated_call,
            model_response(
                tool_calls=(ToolCall(id="c2", name="count", arguments={"value": 2}),)
            ),
            model_response(content="done"),
        ]
    )
    tool = CountingTool()
    tools = ToolRegistry()
    tools.register(tool)
    result = await AgentRuntime(
        registry, tools, provider="fake", max_output_tokens=2048
    ).run("work")

    assert result.ok
    assert result.content == "done"
    assert tool.executions == 1  # 只执行了完整的那次调用
    retry = adapter.requests[1]
    assert retry.messages[-1].name == RUNTIME_NOTICE_NAME
    assert "调用参数不完整" in (retry.messages[-1].content or "")
    # 被丢弃的半截调用不进入后续请求
    assert all(
        call.id != "c1"
        for message in adapter.requests[2].messages
        for call in message.tool_calls
    )
    assert _anthropic_prefix_kept(adapter.requests[0], adapter.requests[1])


@pytest.mark.asyncio
async def test_repeated_truncated_tool_calls_stop_with_clear_error() -> None:
    truncated_call = model_response(
        tool_calls=(ToolCall(id="c1", name="count", arguments={"value": 1}),)
    ).model_copy(update={"finish_reason": "max_tokens"})
    registry, adapter = fake_registry([truncated_call] * 3)
    tool = CountingTool()
    tools = ToolRegistry()
    tools.register(tool)
    result = await AgentRuntime(registry, tools, provider="fake").run("work")

    assert not result.ok
    assert tool.executions == 0
    assert len(adapter.requests) == 3
    assert result.error is not None
    assert "without complete tool calls" in result.error.message
    assert result.model_finish_reason == "max_tokens"
