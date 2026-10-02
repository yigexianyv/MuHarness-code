
from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import SecretStr

from app.domain.task import (
    FileTaskStore,
    TaskContextProvider,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
    register_task_tools,
)
from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    AgentMode,
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
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.runtime import AgentRuntime
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry


class ScriptedAdapter(ModelAdapter):
    # 函数说明：ScriptedAdapter.__init__
    # 用途：初始化 ScriptedAdapter；参数及实际保存的实例字段见下方说明。
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

    # 函数说明：ScriptedAdapter.complete
    # 用途：完成ScriptedAdapter，供回归测试与测试辅助使用。
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

    # 函数说明：ScriptedAdapter.close
    # 用途：关闭ScriptedAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class TrackingTool(BaseTool):

    # 函数说明：TrackingTool.__init__
    # 用途：初始化 TrackingTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   permission：所需权限等级，类型 `ToolPermission`；默认 `ToolPermission.ALLOWED`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._name`、`self.executions`、`self._permission`。
    def __init__(
        self,
        name: str,
        *,
        permission: ToolPermission = ToolPermission.ALLOWED,
    ) -> None:
        self._name = name
        self.executions = 0
        self._permission = permission

    # 函数说明：TrackingTool.definition
    # 用途：提供 TrackingTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"探测工具 {self._name}",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                    "command": {"type": "string"},
                    "url": {"type": "string"},
                },
            },
            permission=self._permission,
        )

    # 函数说明：TrackingTool.execute
    # 用途：执行TrackingTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, object]`。
    # 返回：类型 `str`；返回 `f'{self._name}-ok'`。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    async def execute(self, arguments: dict[str, object]) -> str:
        self.executions += 1
        return f"{self._name}-ok"


# 函数说明：_registry
# 用途：在回归测试与测试辅助中处理 `_registry`，通过 `registry.register` 完成首个内部处
# 理步骤。
# 参数：
#   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`。
# 返回：类型 `tuple[ModelAdapterRegistry, ScriptedAdapter]`；返回 `(registry, adapter)`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `ScriptedAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`。
def _registry(
    responses: Sequence[ModelResponse | Exception],
) -> tuple[ModelAdapterRegistry, ScriptedAdapter]:
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = ScriptedAdapter(config, responses)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)
    return registry, adapter


# 函数说明：_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str | None`；默认 `None`。
#   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`；默认 `()`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def _response(
    *,
    content: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
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
        usage=ModelUsage(),
    )


# 函数说明：_call
# 用途：返回 `ToolCall(id=f'call-{name}', name=name, arguments=arguments)`，提供 回归测
# 试与测试辅助 的派生值。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, object]`。
# 返回：类型 `ToolCall`；返回
# `ToolCall(id=f'call-{name}', name=name, arguments=arguments)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall`。
def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=arguments)


_STUB_NAMES = (
    "read_file",
    "write_file",
    "list_files",
    "web_search",
    "http_request",
    "shell",
    "current_time",
    "memory_read",
)


# 函数说明：_build_tools
# 用途：构建工具集合，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `tuple[ToolRegistry, FileTaskStore, dict[str, TrackingTool]]`；返回
# `(tools, task_store, stubs)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `TrackingTool` →
# `tools.register` → `FileTaskStore` → `task_store.initialize` → `register_task_tools`。
async def _build_tools(
    tmp_path,
) -> tuple[ToolRegistry, FileTaskStore, dict[str, TrackingTool]]:

    tools = ToolRegistry()
    stubs: dict[str, TrackingTool] = {}
    for name in _STUB_NAMES:
        stub = TrackingTool(name)
        stubs[name] = stub
        tools.register(stub)

    task_store = FileTaskStore(tmp_path / "tasks")
    await task_store.initialize()
    register_task_tools(tools, task_store)
    return tools, task_store, stubs


# 函数说明：_run
# 用途：运行回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   registry：工具、模型或能力注册表。
#   tools：可用工具定义或工具实例集合。
#   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
#   task_store：任务存储，类型 `FileTaskStore`。
#   conversation_id：目标会话标识，类型 `str | None`；默认 `'conv-1'`。
#   handler：请求或事件处理回调，类型 `InMemoryEventHandler | None`；默认 `None`。
# 返回：返回 `await runtime.run(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentRuntime` → `TaskContextProvider`
#  → `runtime.run`。
async def _run(
    registry,
    tools,
    *,
    mode: AgentMode,
    task_store: FileTaskStore,
    conversation_id: str | None = "conv-1",
    handler: InMemoryEventHandler | None = None,
):
    runtime = AgentRuntime(
        registry,
        tools,
        provider="fake",
        task_context_provider=TaskContextProvider(task_store),
    )
    return await runtime.run(
        "规划任务",
        conversation_id=conversation_id,
        run_id="run-1",
        event_handler=handler,
        mode=mode,
    )




# 函数说明：test_registry_plan_mode_filters_side_effect_tools
# 用途：回归验证回归测试与测试辅助中的 `registry_plan_mode_filters_side_effect_tools` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` →
# `tools.allowed_names_for_mode` → `tools.model_definitions_for_mode`。
# 分支与异常：
#   验证条件：`'write_file' in normal and 'shell' in normal`。
#   验证条件：`'read_file' in plan`。
#   验证条件：`'list_files' in plan`。
#   验证条件：`'web_search' in plan`。
async def test_registry_plan_mode_filters_side_effect_tools(tmp_path) -> None:
    tools, _, _ = await _build_tools(tmp_path)

    normal = tools.allowed_names_for_mode(AgentMode.NORMAL)
    assert "write_file" in normal and "shell" in normal

    plan = tools.allowed_names_for_mode(AgentMode.PLAN)
    assert "read_file" in plan
    assert "list_files" in plan
    assert "web_search" in plan
    assert "current_time" in plan
    assert "task_create" in plan
    assert "task_update" in plan
    assert "task_get" in plan
    assert "task_list" in plan
    for forbidden in (
        "write_file",
        "shell",
        "http_request",
        "automation_create",
        "automation_pause",
        "memory_create",
        "memory_update",
        "memory_archive",
        "core_memory_update",
        "core_memory_remove",
        "skill_read",
        "tool_search",
    ):
        assert forbidden not in plan

    plan_defs = {d.name for d in tools.model_definitions_for_mode(AgentMode.PLAN)}
    assert "write_file" not in plan_defs
    assert "read_file" in plan_defs




# 函数说明：test_plan_mode_allows_read_and_search
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_allows_read_and_search` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` → `_registry` →
# `_response` → `_call` → `_run`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`stubs['read_file'].executions == 1`。
#   验证条件：`stubs['web_search'].executions == 1`。
#   验证条件：`all((record.result.success for record in result.tool_calls))`。
async def test_plan_mode_allows_read_and_search(tmp_path) -> None:
    tools, task_store, stubs = await _build_tools(tmp_path)
    registry, _ = _registry(
        [
            _response(
                tool_calls=(
                    _call("read_file", {"path": "a.py"}),
                    _call("web_search", {"query": "agent runtime"}),
                )
            ),
            _response(content="计划说明"),
        ]
    )
    result = await _run(
        registry, tools, mode=AgentMode.PLAN, task_store=task_store
    )

    assert result.ok is True
    assert stubs["read_file"].executions == 1
    assert stubs["web_search"].executions == 1
    assert all(record.result.success for record in result.tool_calls)




# 函数说明：test_plan_mode_creates_pending_task
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_creates_pending_task` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` → `_registry` →
# `_response` → `_call` → `_run`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.plan_task_id is not None`。
#   验证条件：`'without creating a task' not in (result.content or '')`。
#   验证条件：`task is not None`。
async def test_plan_mode_creates_pending_task(tmp_path) -> None:
    tools, task_store, _ = await _build_tools(tmp_path)
    registry, _ = _registry(
        [
            _response(
                tool_calls=(
                    _call(
                        "task_create",
                        {
                            "title": "实现 Evidence Runtime",
                            "goal": "实现 Evidence Runtime V1",
                            "steps": [
                                {"title": "定义 protocol"},
                                {"title": "实现 observe"},
                                {"title": "实现 click/type"},
                            ],
                        },
                    ),
                )
            ),
            _response(content="计划已形成"),
        ]
    )
    result = await _run(
        registry, tools, mode=AgentMode.PLAN, task_store=task_store
    )

    assert result.ok is True
    assert result.plan_task_id is not None
    assert "without creating a task" not in (result.content or "")

    task = await task_store.get(result.plan_task_id)
    assert task is not None
    assert task.status is TaskStatus.PENDING  
    assert task.goal == "实现 Evidence Runtime V1"
    assert len(task.steps) == 3
    assert all(step.status is TaskStepStatus.TODO for step in task.steps)




# 函数说明：test_plan_mode_can_update_plan_content
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_can_update_plan_content` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` → `task_store.create` →
#  `_registry` → `_response` → `_call` → `_run`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.plan_task_id == created.id`。
#   验证条件：`updated is not None`。
#   验证条件：`updated.goal == '细化后的目标'`。
async def test_plan_mode_can_update_plan_content(tmp_path) -> None:
    tools, task_store, _ = await _build_tools(tmp_path)
    created = await task_store.create(
        title="初始计划",
        owner_conversation_id="conv-1",
    )
    registry, _ = _registry(
        [
            _response(
                tool_calls=(
                    _call(
                        "task_update",
                        {
                            "task_id": created.id,
                            "goal": "细化后的目标",
                            "constraints": ["不改动核心模块"],
                            "steps": [{"title": "第一步"}, {"title": "第二步"}],
                        },
                    ),
                )
            ),
            _response(content="计划已细化"),
        ]
    )
    result = await _run(
        registry, tools, mode=AgentMode.PLAN, task_store=task_store
    )

    assert result.ok is True
    assert result.plan_task_id == created.id
    updated = await task_store.get(created.id)
    assert updated is not None
    assert updated.goal == "细化后的目标"
    assert updated.constraints == ("不改动核心模块",)
    assert len(updated.steps) == 2
    assert updated.status is TaskStatus.PENDING



# 函数说明：test_plan_mode_task_update_cannot_change_status
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_task_update_cannot_change_status` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` → `task_store.create` →
#  `pytest.raises` → `update_tool.execute_with_context` → `ToolExecutionContext` → `TC`
# 。
# 分支与异常：
#   验证条件：`(await task_store.get(created.id)).status is TaskStatus.PENDING`。
#   预期异常：`pytest.raises(ValueError, match='改变任务状态')`。
async def test_plan_mode_task_update_cannot_change_status(tmp_path) -> None:

    from app.domain.task.tools import TaskUpdateTool
    from app.models.types import ToolCall as TC
    from app.tools.hooks import ToolExecutionContext

    tools, task_store, _ = await _build_tools(tmp_path)
    created = await task_store.create(
        title="初始计划",
        owner_conversation_id="conv-1",
    )
    update_tool: TaskUpdateTool = tools.get("task_update")

    status_args = {"task_id": created.id, "status": "active"}
    with pytest.raises(ValueError, match="改变任务状态"):
        await update_tool.execute_with_context(
            status_args,
            ToolExecutionContext(
                tool_call=TC(id="t1", name="task_update", arguments=status_args),
                run_id="run-1",
                conversation_id="conv-1",
                mode=AgentMode.PLAN,
            ),
        )
    assert (await task_store.get(created.id)).status is TaskStatus.PENDING




# 函数说明：test_plan_mode_blocks_side_effect_tools
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_blocks_side_effect_tools` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` → `_registry` →
# `_response` → `_call` → `_run`。
# 分支与异常：
#   验证条件：`stubs['write_file'].executions == 0`。
#   验证条件：`stubs['shell'].executions == 0`。
#   验证条件：`stubs['http_request'].executions == 0`。
#   验证条件：`record.result.success is False`。
async def test_plan_mode_blocks_side_effect_tools(tmp_path) -> None:
    tools, task_store, stubs = await _build_tools(tmp_path)
    registry, _ = _registry(
        [
            _response(
                tool_calls=(
                    _call("write_file", {"path": "evil.txt"}),
                    _call("shell", {"command": "rm -rf /"}),
                    _call("http_request", {"url": "https://example.com"}),
                )
            ),
            _response(content="我不应能执行副作用工具"),
        ]
    )
    result = await _run(
        registry, tools, mode=AgentMode.PLAN, task_store=task_store
    )

    assert stubs["write_file"].executions == 0
    assert stubs["shell"].executions == 0
    assert stubs["http_request"].executions == 0
    for record in result.tool_calls:
        assert record.result.success is False
        assert "not allowed in plan mode" in (record.result.error or "")




# 函数说明：test_plan_mode_without_task_returns_clear_message
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_without_task_returns_clear_message` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` → `_registry` →
# `_response` → `_run`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.plan_task_id is None`。
#   验证条件：`'Plan mode finished without creating a task' in (result.content or '')`。
async def test_plan_mode_without_task_returns_clear_message(tmp_path) -> None:
    tools, task_store, _ = await _build_tools(tmp_path)
    registry, _ = _registry([_response(content="没有形成计划")])
    result = await _run(
        registry, tools, mode=AgentMode.PLAN, task_store=task_store
    )

    assert result.ok is True
    assert result.plan_task_id is None
    assert "Plan mode finished without creating a task" in (result.content or "")




# 函数说明：test_plan_mode_invalid_pending_task_is_not_success
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_invalid_pending_task_is_not_success`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` → `_registry` →
# `_response` → `_call` → `_run` → `task_store.list`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.plan_task_id is None`。
#   验证条件：`'without a valid pending task' in (result.content or '')`。
#   验证条件：`len(tasks) == 1`。
async def test_plan_mode_invalid_pending_task_is_not_success(tmp_path) -> None:

    tools, task_store, _ = await _build_tools(tmp_path)
    registry, _ = _registry(
        [
            _response(
                tool_calls=(_call("task_create", {"title": "空计划"}),)
            ),
            _response(content="计划完成"),
        ]
    )
    result = await _run(
        registry, tools, mode=AgentMode.PLAN, task_store=task_store
    )

    assert result.ok is True
    assert result.plan_task_id is None  
    assert "without a valid pending task" in (result.content or "")

    tasks = await task_store.list()
    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.PENDING


# 函数说明：test_plan_mode_valid_pending_task_passes
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_valid_pending_task_passes` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` → `_registry` →
# `_response` → `_call` → `_run`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.plan_task_id is not None`。
#   验证条件：`'without a valid pending task' not in (result.content or '')`。
#   验证条件：`task is not None and task.status is TaskStatus.PENDING`。
async def test_plan_mode_valid_pending_task_passes(tmp_path) -> None:

    tools, task_store, _ = await _build_tools(tmp_path)
    registry, _ = _registry(
        [
            _response(
                tool_calls=(
                    _call(
                        "task_create",
                        {
                            "title": "实现 Evidence Runtime",
                            "goal": "实现 Evidence Runtime V1",
                            "steps": [
                                {"title": "定义 protocol"},
                                {"title": "实现 observe"},
                            ],
                        },
                    ),
                )
            ),
            _response(content="计划已形成"),
        ]
    )
    result = await _run(
        registry, tools, mode=AgentMode.PLAN, task_store=task_store
    )

    assert result.ok is True
    assert result.plan_task_id is not None
    assert "without a valid pending task" not in (result.content or "")
    task = await task_store.get(result.plan_task_id)
    assert task is not None and task.status is TaskStatus.PENDING


# 函数说明：test_pending_plan_validation_conditions
# 用途：回归验证回归测试与测试辅助中的 `pending_plan_validation_conditions` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` → `store.initialize` →
#  `TaskContextProvider` → `store.create` → `TaskStep` →
# `provider.pending_plan_is_valid`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`await provider.pending_plan_is_valid('conv-1', valid.id) is True`。
#   验证条件：`await provider.pending_plan_is_valid('conv-1', no_goal.id) is False`。
#   验证条件：`await provider.pending_plan_is_valid('conv-1', no_steps.id) is False`。
#   验证条件：`await provider.pending_plan_is_valid('conv-1', with_done.id) is False`。
async def test_pending_plan_validation_conditions(tmp_path) -> None:

    store = FileTaskStore(tmp_path / "tasks")
    await store.initialize()
    provider = TaskContextProvider(store)

    valid = await store.create(
        title="T",
        goal="G",
        steps=(TaskStep(id="s1", title="s1"),),
        owner_conversation_id="conv-1",
    )
    assert await provider.pending_plan_is_valid("conv-1", valid.id) is True

    no_goal = await store.create(
        title="T",
        steps=(TaskStep(id="s2", title="s2"),),
        owner_conversation_id="conv-1",
    )
    assert await provider.pending_plan_is_valid("conv-1", no_goal.id) is False

    no_steps = await store.create(
        title="T",
        goal="G",
        owner_conversation_id="conv-1",
    )
    assert await provider.pending_plan_is_valid("conv-1", no_steps.id) is False

    with_done = await store.create(
        title="T",
        goal="G",
        steps=(
            TaskStep(
                id="s3",
                title="s3",
                status=TaskStepStatus.DONE,
                note="已完成",
            ),
        ),
        owner_conversation_id="conv-1",
    )
    assert await provider.pending_plan_is_valid("conv-1", with_done.id) is False
    with_progress = await store.create(
        title="T",
        goal="G",
        steps=(
            TaskStep(
                id="s4",
                title="s4",
                status=TaskStepStatus.IN_PROGRESS,
            ),
        ),
        owner_conversation_id="conv-1",
    )
    assert await provider.pending_plan_is_valid("conv-1", with_progress.id) is False

    await store.plan_accept(valid.id)
    assert await provider.pending_plan_is_valid("conv-1", valid.id) is False

    other = await store.create(
        title="T",
        goal="G",
        steps=(TaskStep(id="s5", title="s5"),),
        owner_conversation_id="conv-2",
    )
    assert await provider.pending_plan_is_valid("conv-1", other.id) is False
    assert await provider.pending_plan_is_valid("conv-1", "0" * 32) is False
    assert await provider.pending_plan_is_valid(None, "0" * 32) is False
    assert await provider.pending_plan_is_valid("conv-1", "") is False




# 函数说明：test_plan_accept_pending_to_active
# 用途：回归验证回归测试与测试辅助中的 `plan_accept_pending_to_active` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` → `store.initialize` →
#  `store.create` → `store.plan_accept`。
# 分支与异常：
#   验证条件：`task.status is TaskStatus.PENDING`。
#   验证条件：`accepted.status is TaskStatus.ACTIVE`。
#   验证条件：`accepted.completed_at is None`。
#   验证条件：`accepted.revision == task.revision + 1`。
async def test_plan_accept_pending_to_active(tmp_path) -> None:
    store = FileTaskStore(tmp_path / "tasks")
    await store.initialize()
    task = await store.create(title="计划", owner_conversation_id="conv-1")
    assert task.status is TaskStatus.PENDING

    accepted = await store.plan_accept(task.id)
    assert accepted.status is TaskStatus.ACTIVE
    assert accepted.completed_at is None
    assert accepted.revision == task.revision + 1


# 函数说明：test_plan_reject_pending_to_cancelled
# 用途：回归验证回归测试与测试辅助中的 `plan_reject_pending_to_cancelled` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` → `store.initialize` →
#  `store.create` → `store.plan_reject`。
# 分支与异常：
#   验证条件：`rejected.status is TaskStatus.CANCELLED`。
#   验证条件：`rejected.completed_at is not None`。
async def test_plan_reject_pending_to_cancelled(tmp_path) -> None:
    store = FileTaskStore(tmp_path / "tasks")
    await store.initialize()
    task = await store.create(title="计划", owner_conversation_id="conv-1")

    rejected = await store.plan_reject(task.id)
    assert rejected.status is TaskStatus.CANCELLED
    assert rejected.completed_at is not None


# 函数说明：test_non_pending_task_cannot_accept_or_reject
# 用途：回归验证回归测试与测试辅助中的 `non_pending_task_cannot_accept_or_reject` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` → `store.initialize` →
#  `store.create` → `store.plan_accept` → `pytest.raises` → `store.plan_reject`；另有 1
# 个调用点。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='only pending')`。
async def test_non_pending_task_cannot_accept_or_reject(tmp_path) -> None:
    store = FileTaskStore(tmp_path / "tasks")
    await store.initialize()
    task = await store.create(title="计划", owner_conversation_id="conv-1")
    await store.plan_accept(task.id)  

    with pytest.raises(ValueError, match="only pending"):
        await store.plan_accept(task.id)
    with pytest.raises(ValueError, match="only pending"):
        await store.plan_reject(task.id)

    other = await store.create(title="另一计划", owner_conversation_id="conv-1")
    await store.set_status(other.id, TaskStatus.COMPLETED)
    with pytest.raises(ValueError, match="only pending"):
        await store.plan_accept(other.id)
    with pytest.raises(ValueError, match="only pending"):
        await store.plan_reject(other.id)




# 函数说明：test_accept_then_injected_via_task_context
# 用途：回归验证回归测试与测试辅助中的 `accept_then_injected_via_task_context` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` → `store.initialize` →
#  `store.create` → `TaskContextProvider` → `provider.message_for` → `store.plan_accept`
# 。
# 分支与异常：
#   验证条件：`await provider.message_for('conv-1') is None`。
#   验证条件：`message is not None and task.id in (message.content or '')`。
async def test_accept_then_injected_via_task_context(tmp_path) -> None:
    store = FileTaskStore(tmp_path / "tasks")
    await store.initialize()
    task = await store.create(title="计划", owner_conversation_id="conv-1")
    provider = TaskContextProvider(store)

    assert await provider.message_for("conv-1") is None  

    await store.plan_accept(task.id)
    message = await provider.message_for("conv-1")
    assert message is not None and task.id in (message.content or "")




# 函数说明：test_run_persists_mode
# 用途：回归验证回归测试与测试辅助中的 `run_persists_mode` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `conversation_store.initialize` → `conversation_store.create` → `SQLiteTraceStore` →
# `trace_store.initialize` → `SQLiteRunStore`；另有 12 个调用点。
# 分支与异常：
#   验证条件：`dispatch.run.mode is AgentMode.NORMAL`。
#   验证条件：`dispatch2.run.mode is AgentMode.PLAN`。
#   验证条件：`persisted is not None`。
#   验证条件：`persisted.mode is AgentMode.PLAN`。
async def test_run_persists_mode(tmp_path) -> None:
    from app.domain.conversation.service import ConversationService
    from app.domain.conversation.store import SQLiteConversationStore
    from app.records.trace import SQLiteTraceStore
    from app.runtime.checkpoint import SQLiteCheckpointStore
    from app.runtime.run import RunManager, SQLiteRunStore

    database = tmp_path / "muharness.db"
    conversation_store = SQLiteConversationStore(database)
    await conversation_store.initialize()
    conversation = await conversation_store.create()
    trace_store = SQLiteTraceStore(database)
    await trace_store.initialize()
    run_store = SQLiteRunStore(database)
    await run_store.initialize()
    checkpoint_store = SQLiteCheckpointStore(database)
    await checkpoint_store.initialize()

    tools, _, _ = await _build_tools(tmp_path)
    registry, _ = _registry([_response(content="普通回答")])
    runtime = AgentRuntime(registry, tools, provider="fake")
    run_manager = RunManager(run_store, checkpoint_store, runtime)
    service = ConversationService(conversation_store, run_manager, trace_store)
    dispatch = await service.dispatch(
        conversation_id=conversation.id,
        content="你好",
    )
    assert dispatch.run.mode is AgentMode.NORMAL

    tools2, _, _ = await _build_tools(tmp_path)
    registry2, _ = _registry(
        [
            _response(tool_calls=(_call("task_create", {"title": "计划"}),)),
            _response(content="计划说明"),
        ]
    )
    runtime2 = AgentRuntime(registry2, tools2, provider="fake")
    run_manager2 = RunManager(run_store, checkpoint_store, runtime2)
    service2 = ConversationService(conversation_store, run_manager2, trace_store)
    dispatch2 = await service2.dispatch(
        conversation_id=conversation.id,
        content="规划一下",
        mode=AgentMode.PLAN,
    )
    assert dispatch2.run.mode is AgentMode.PLAN

    persisted = await run_store.get(dispatch2.run.id)
    assert persisted is not None
    assert persisted.mode is AgentMode.PLAN




# 函数说明：test_plan_mode_emits_events
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_emits_events` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_tools` → `_registry` →
# `_response` → `_call` → `InMemoryEventHandler` → `_run`。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`AgentEventType.AGENT_STARTED in types`。
#   验证条件：`AgentEventType.AGENT_COMPLETED in types`。
#   验证条件：`AgentEventType.TOOL_STARTED in types`。
async def test_plan_mode_emits_events(tmp_path) -> None:
    tools, task_store, _ = await _build_tools(tmp_path)
    registry, _ = _registry(
        [
            _response(tool_calls=(_call("read_file", {"path": "a.py"}),)),
            _response(tool_calls=(_call("task_create", {"title": "计划"}),)),
            _response(content="计划完成"),
        ]
    )
    handler = InMemoryEventHandler()
    result = await _run(
        registry,
        tools,
        mode=AgentMode.PLAN,
        task_store=task_store,
        handler=handler,
    )

    assert result.ok is True
    types = [event.type for event in handler.events]
    assert AgentEventType.AGENT_STARTED in types
    assert AgentEventType.AGENT_COMPLETED in types
    assert AgentEventType.TOOL_STARTED in types
    assert AgentEventType.TOOL_COMPLETED in types
    assert AgentEventType.MODEL_STARTED in types




# 函数说明：test_automation_dispatch_defaults_to_normal
# 用途：回归验证回归测试与测试辅助中的 `automation_dispatch_defaults_to_normal` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `conversation_store.initialize` → `conversation_store.create` → `SQLiteTraceStore` →
# `trace_store.initialize` → `SQLiteRunStore`；另有 11 个调用点。
# 分支与异常：
#   验证条件：`dispatch.run.mode is AgentMode.NORMAL`。
#   验证条件：`dispatch.run.source == 'automation'`。
#   验证条件：`dispatch.run.source_id == 'auto-1'`。
async def test_automation_dispatch_defaults_to_normal(tmp_path) -> None:
    from app.domain.conversation.inputs import ConversationSource, TriggerContext
    from app.domain.conversation.service import ConversationService
    from app.domain.conversation.store import SQLiteConversationStore
    from app.records.trace import SQLiteTraceStore
    from app.runtime.checkpoint import SQLiteCheckpointStore
    from app.runtime.run import RunManager, SQLiteRunStore

    database = tmp_path / "muharness.db"
    conversation_store = SQLiteConversationStore(database)
    await conversation_store.initialize()
    conversation = await conversation_store.create()
    trace_store = SQLiteTraceStore(database)
    await trace_store.initialize()
    run_store = SQLiteRunStore(database)
    await run_store.initialize()
    checkpoint_store = SQLiteCheckpointStore(database)
    await checkpoint_store.initialize()

    tools, _, _ = await _build_tools(tmp_path)
    registry, _ = _registry([_response(content="自动化回答")])
    runtime = AgentRuntime(registry, tools, provider="fake")
    run_manager = RunManager(run_store, checkpoint_store, runtime)
    service = ConversationService(conversation_store, run_manager, trace_store)

    dispatch = await service.dispatch(
        conversation_id=conversation.id,
        content="定时任务",
        trigger=TriggerContext(
            source=ConversationSource.AUTOMATION,
            automation_id="auto-1",
        ),
    )
    assert dispatch.run.mode is AgentMode.NORMAL
    assert dispatch.run.source == "automation"
    assert dispatch.run.source_id == "auto-1"
