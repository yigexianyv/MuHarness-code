
from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

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
from app.runtime.agent.runtime import AgentRuntime
from app.safety.sandbox import (
    DockerSandboxBackend,
    SandboxLaunchSpec,
    SandboxSupervisor,
)
from app.tools.approval import ApprovalGate, ApprovalRequest, ApprovalResponse
from app.tools.base import BaseTool
from app.tools.builtin.shell import ShellCommandTool
from app.tools.builtin.write_file import WriteFileTool
from app.tools.executor import ToolExecutor
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry
from app.tools.role_boundary import (
    AUDIT_ALLOWED_TOOLS,
    EXECUTE_ALLOWED_TOOLS,
    ROLE_REJECTION_PREFIX,
    count_role_rejections,
)

# ---------------------------------------------------------------------------
# 夹具：脚本化模型 + 计数工具（同 test_plan_mode.py 的写法）
# ---------------------------------------------------------------------------


class ScriptedAdapter(ModelAdapter):
    # 函数说明：ScriptedAdapter.__init__
    # 用途：初始化 ScriptedAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.requests`。
    def __init__(self, config: ProviderConfig, responses: Sequence[ModelResponse]) -> None:
        super().__init__(config)
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    # 函数说明：ScriptedAdapter.complete
    # 用途：完成ScriptedAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `self.responses.pop(0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.responses.pop`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)

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
    #   更新对象字段：`self._name`、`self._permission`、`self.executions`。
    def __init__(
        self,
        name: str,
        *,
        permission: ToolPermission = ToolPermission.ALLOWED,
    ) -> None:
        self._name = name
        self._permission = permission
        self.executions = 0

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
                "properties": {"path": {"type": "string"}, "query": {"type": "string"}},
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


class RecordingGate(ApprovalGate):
    # 函数说明：RecordingGate.__init__
    # 用途：初始化 RecordingGate；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.requests`。
    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    # 函数说明：RecordingGate.request_approval
    # 用途：在回归测试与测试辅助中处理 `request_approval`，通过 `self.requests.append`
    # 完成首个内部处理步骤。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；不返回结果值（隐式 None）。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        self.requests.append(request)
        raise AssertionError("role boundary must reject before any approval request")


# 函数说明：_model_registry
# 用途：在回归测试与测试辅助中处理 `_model_registry`，通过 `registry.register` 完成首个
# 内部处理步骤。
# 参数：
#   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse]`。
# 返回：类型 `tuple[ModelAdapterRegistry, ScriptedAdapter]`；返回 `(registry, adapter)`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `ScriptedAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`。
def _model_registry(responses: Sequence[ModelResponse]) -> tuple[ModelAdapterRegistry, ScriptedAdapter]:
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
def _response(*, content: str | None = None, tool_calls: tuple[ToolCall, ...] = ()) -> ModelResponse:
    return ModelResponse(
        id="fake-response",
        provider="fake",
        model="fake-model",
        message=Message(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls),
        usage=ModelUsage(),
    )


# 函数说明：_call
# 用途：返回 `ToolCall(id=f'call-{name}{suffix}', name=name, arguments=arguments)`，提供
#  回归测试与测试辅助 的派生值。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, object]`。
#   suffix：`suffix`输入或配置值，类型 `str`；默认 `''`。
# 返回：类型 `ToolCall`；返回
# `ToolCall(id=f'call-{name}{suffix}', name=name, arguments=arguments)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall`。
def _call(name: str, arguments: dict[str, object], suffix: str = "") -> ToolCall:
    return ToolCall(id=f"call-{name}{suffix}", name=name, arguments=arguments)


# 覆盖白名单内外的典型工具；evidence_* 与真实注册方式一致，是延迟工具
_STUBS = (
    "read_file",
    "list_files",
    "get_current_time",
    "task_get",
    "task_update",
    "memory_create",
    "core_memory_update",
    "automation_create",
    "http_request",
    "web_search",
    "history_search",
)
_DEFERRED_STUBS = ("evidence_search", "evidence_read")


# 函数说明：_tools
# 用途：在回归测试与测试辅助中处理 `_tools`，通过 `tools.register` 完成首个内部处理步骤
# 。
# 参数：
#   workspace：目标工作区，类型 `Path`。
# 返回：类型 `tuple[ToolRegistry, dict[str, TrackingTool]]`；返回 `(tools, stubs)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `TrackingTool` →
# `tools.register` → `WriteFileTool`。
def _tools(workspace: Path) -> tuple[ToolRegistry, dict[str, TrackingTool]]:
    tools = ToolRegistry()
    stubs: dict[str, TrackingTool] = {}
    for name in _STUBS:
        stubs[name] = TrackingTool(name)
        tools.register(stubs[name])
    for name in _DEFERRED_STUBS:
        stubs[name] = TrackingTool(name)
        tools.register(stubs[name], deferred=True)
    tools.register(WriteFileTool(workspace))
    return tools, stubs


# 函数说明：_run
# 用途：运行回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   model_registry：模型适配器注册表，类型 `ModelAdapterRegistry`。
#   tools：可用工具定义或工具实例集合，类型 `ToolRegistry`。
#   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
# 返回：返回
# `await runtime.run('子任务', conversation_id='conv-1', run_id='run-1', mode=mode)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentRuntime` → `runtime.run`。
async def _run(model_registry: ModelAdapterRegistry, tools: ToolRegistry, mode: AgentMode):
    runtime = AgentRuntime(model_registry, tools, provider="fake")
    return await runtime.run("子任务", conversation_id="conv-1", run_id="run-1", mode=mode)


# ---------------------------------------------------------------------------
# 第一层：模型看到的工具
# ---------------------------------------------------------------------------


# 函数说明：test_model_definitions_follow_role_whitelists
# 用途：回归验证回归测试与测试辅助中的 `model_definitions_follow_role_whitelists` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools` → `tools.register` →
# `TrackingTool` → `tools.model_definitions_for_mode` → `tools.is_available_for_mode` →
# `tools.allowed_names_for_mode`。
# 分支与异常：
#   验证条件：`manage == set()`。
#   验证条件：`execute == {'read_file', 'list_files', 'get_current_time', 'write_file',
# 'evidence_search', 'evidence_read'}`。
#   验证条件：`audit == {'read_file', 'list_files', 'get_current_time', 'task_get', '
# evidence_search', 'evidence_read'}`。
#   验证条件：`execute <= EXECUTE_ALLOWED_TOOLS and audit <= AUDIT_ALLOWED_TOOLS`。
def test_model_definitions_follow_role_whitelists(tmp_path: Path) -> None:
    tools, _ = _tools(tmp_path)
    tools.register(TrackingTool("mcp_github_create_issue"), deferred=True)

    manage = {d.name for d in tools.model_definitions_for_mode(AgentMode.MANAGE)}
    execute = {d.name for d in tools.model_definitions_for_mode(AgentMode.EXECUTE)}
    audit = {d.name for d in tools.model_definitions_for_mode(AgentMode.AUDIT)}

    assert manage == set()
    # 白名单内的延迟工具（evidence_*）不需要 tool_search 激活
    assert execute == {"read_file", "list_files", "get_current_time", "write_file",
                       "evidence_search", "evidence_read"}
    assert audit == {"read_file", "list_files", "get_current_time", "task_get",
                     "evidence_search", "evidence_read"}
    assert execute <= EXECUTE_ALLOWED_TOOLS and audit <= AUDIT_ALLOWED_TOOLS

    # 白名单外的工具即使“已激活”也不可用
    assert not tools.is_available_for_mode(
        "mcp_github_create_issue", AgentMode.EXECUTE, activated_names={"mcp_github_create_issue"}
    )
    # NORMAL / PLAN 行为不变
    assert "memory_create" in tools.allowed_names_for_mode(AgentMode.NORMAL)
    assert "write_file" not in tools.allowed_names_for_mode(AgentMode.PLAN)


# ---------------------------------------------------------------------------
# 第二层：执行入口拒绝（构造非法调用，而不是只看工具列表）
# ---------------------------------------------------------------------------


# 函数说明：test_audit_write_file_call_is_rejected_and_file_not_created
# 用途：回归验证回归测试与测试辅助中的
# `audit_write_file_call_is_rejected_and_file_not_created` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools` → `_model_registry` →
# `_response` → `_call` → `_run` → `(tmp_path / 'evil.txt').exists`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`not (tmp_path / 'evil.txt').exists()`。
#   验证条件：`record.result.success is False`。
#   验证条件：`(record.result.error or '').startswith(ROLE_REJECTION_PREFIX)`。
#   验证条件：`'write_file' not in {tool.name for tool in adapter.requests[0].tools}`。
async def test_audit_write_file_call_is_rejected_and_file_not_created(tmp_path: Path) -> None:
    tools, _ = _tools(tmp_path)
    model_registry, adapter = _model_registry(
        [
            _response(tool_calls=(_call("write_file", {"path": "evil.txt", "content": "x"}),)),
            _response(content="审计结束"),
        ]
    )

    result = await _run(model_registry, tools, AgentMode.AUDIT)

    assert not (tmp_path / "evil.txt").exists()
    [record] = result.tool_calls
    assert record.result.success is False
    assert (record.result.error or "").startswith(ROLE_REJECTION_PREFIX)
    assert "write_file" not in {tool.name for tool in adapter.requests[0].tools}
    assert count_role_rejections(r.result for r in result.tool_calls) == {"write_file": 1}


# 函数说明：test_execute_side_effect_tools_are_rejected_without_running
# 用途：回归验证回归测试与测试辅助中的
# `execute_side_effect_tools_are_rejected_without_running` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools` → `_model_registry` →
# `_response` → `_call` → `_run` →
# `(result.tool_calls[0].result.error or '').startswith`。
# 分支与异常：
#   验证条件：`stubs[name].executions == 0`。
#   验证条件：
# `(result.tool_calls[0].result.error or '').startswith(ROLE_REJECTION_PREFIX)`。
@pytest.mark.parametrize(
    "name",
    ["memory_create", "core_memory_update", "task_update", "automation_create",
     "http_request", "web_search", "history_search"],
)
async def test_execute_side_effect_tools_are_rejected_without_running(
    tmp_path: Path, name: str
) -> None:
    tools, stubs = _tools(tmp_path)
    model_registry, _ = _model_registry(
        [_response(tool_calls=(_call(name, {"query": "x"}),)), _response(content="完成")]
    )

    result = await _run(model_registry, tools, AgentMode.EXECUTE)

    assert stubs[name].executions == 0
    assert (result.tool_calls[0].result.error or "").startswith(ROLE_REJECTION_PREFIX)


# 函数说明：test_tool_registered_after_runtime_creation_is_rejected
# 用途：回归验证回归测试与测试辅助中的
# `tool_registered_after_runtime_creation_is_rejected` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools` → `_model_registry` →
# `_response` → `_call` → `AgentRuntime` → `TrackingTool`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`late_mcp_tool.executions == 0`。
#   验证条件：`[r.result.success for r in result.tool_calls] == [False, False]`。
#   验证条件：`count_role_rejections((r.result for r in result.tool_calls)) == {'
# tool_search': 1, 'mcp_github_create_issue': 1}`。
#   验证条件：`'tool_search' not in names and 'mcp_github_create_issue' not in names`。
async def test_tool_registered_after_runtime_creation_is_rejected(tmp_path: Path) -> None:
    tools, _ = _tools(tmp_path)
    model_registry, adapter = _model_registry(
        [
            _response(tool_calls=(_call("tool_search", {"query": "github"}),)),
            _response(tool_calls=(_call("mcp_github_create_issue", {"query": "x"}),)),
            _response(content="完成"),
        ]
    )
    runtime = AgentRuntime(model_registry, tools, provider="fake")
    late_mcp_tool = TrackingTool("mcp_github_create_issue")
    tools.register(late_mcp_tool, deferred=True)  # 模拟运行中新连上的 MCP Server

    result = await runtime.run("子任务", conversation_id="conv-1", run_id="run-1",
                               mode=AgentMode.EXECUTE)

    assert late_mcp_tool.executions == 0
    assert [r.result.success for r in result.tool_calls] == [False, False]
    assert count_role_rejections(r.result for r in result.tool_calls) == {
        "tool_search": 1,
        "mcp_github_create_issue": 1,
    }
    for request in adapter.requests:
        names = {tool.name for tool in request.tools}
        assert "tool_search" not in names and "mcp_github_create_issue" not in names


# 函数说明：test_manage_mode_has_no_tools
# 用途：回归验证回归测试与测试辅助中的 `manage_mode_has_no_tools` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools` → `_model_registry` →
# `_response` → `_call` → `_run` →
# `(result.tool_calls[0].result.error or '').startswith`。
# 分支与异常：
#   验证条件：`adapter.requests[0].tools == ()`。
#   验证条件：`stubs['read_file'].executions == 0`。
#   验证条件：
# `(result.tool_calls[0].result.error or '').startswith(ROLE_REJECTION_PREFIX)`。
async def test_manage_mode_has_no_tools(tmp_path: Path) -> None:
    tools, stubs = _tools(tmp_path)
    model_registry, adapter = _model_registry(
        [_response(tool_calls=(_call("read_file", {"path": "a.py"}),)), _response(content="计划")]
    )

    result = await _run(model_registry, tools, AgentMode.MANAGE)

    assert adapter.requests[0].tools == ()
    assert stubs["read_file"].executions == 0
    assert (result.tool_calls[0].result.error or "").startswith(ROLE_REJECTION_PREFIX)


# 函数说明：test_whitelisted_deferred_tool_runs_in_execute_without_activation
# 用途：回归验证回归测试与测试辅助中的
# `whitelisted_deferred_tool_runs_in_execute_without_activation` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools` → `_model_registry` →
# `_response` → `_call` → `_run`。
# 分支与异常：
#   验证条件：`result.tool_calls[0].result.success is True`。
#   验证条件：`stubs['evidence_search'].executions == 1`。
async def test_whitelisted_deferred_tool_runs_in_execute_without_activation(tmp_path: Path) -> None:
    tools, stubs = _tools(tmp_path)
    model_registry, _ = _model_registry(
        [_response(tool_calls=(_call("evidence_search", {"query": "pytest"}),)),
         _response(content="完成")]
    )

    result = await _run(model_registry, tools, AgentMode.EXECUTE)

    assert result.tool_calls[0].result.success is True
    assert stubs["evidence_search"].executions == 1


# 函数说明：test_plan_mode_message_is_unchanged
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_message_is_unchanged` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools` → `_model_registry` →
# `_response` → `_call` → `_run`。
# 分支与异常：
#   验证条件：`stubs['memory_create'].executions == 0`。
#   验证条件：`'not allowed in plan mode' in (result.tool_calls[0].result.error or '')`
# 。
async def test_plan_mode_message_is_unchanged(tmp_path: Path) -> None:
    tools, stubs = _tools(tmp_path)
    model_registry, _ = _model_registry(
        [_response(tool_calls=(_call("memory_create", {"query": "x"}),)), _response(content="计划")]
    )

    result = await _run(model_registry, tools, AgentMode.PLAN)

    assert stubs["memory_create"].executions == 0
    assert "not allowed in plan mode" in (result.tool_calls[0].result.error or "")


# ---------------------------------------------------------------------------
# RoleBoundaryHook：绕过 ToolRoundExecutor 直接调 ToolExecutor
# ---------------------------------------------------------------------------


# 函数说明：test_direct_executor_call_is_rejected_by_hook
# 用途：回归验证回归测试与测试辅助中的 `direct_executor_call_is_rejected_by_hook` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools` → `ToolExecutor` → `_call` →
# `executor.execute` → `ToolExecutionContext` → `(rejected.error or '').startswith`；另
# 有 2 个调用点。
# 分支与异常：
#   验证条件：`rejected.success is False`。
#   验证条件：`(rejected.error or '').startswith(ROLE_REJECTION_PREFIX)`。
#   验证条件：`not (tmp_path / 'direct.txt').exists()`。
#   验证条件：`allowed.success is True`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'direct.txt').read_text`。
async def test_direct_executor_call_is_rejected_by_hook(tmp_path: Path) -> None:
    tools, _ = _tools(tmp_path)
    executor = ToolExecutor(tools)
    call = _call("write_file", {"path": "direct.txt", "content": "x"})

    rejected = await executor.execute(
        call, context=ToolExecutionContext(tool_call=call, run_id="r", mode=AgentMode.AUDIT)
    )
    assert rejected.success is False
    assert (rejected.error or "").startswith(ROLE_REJECTION_PREFIX)
    assert not (tmp_path / "direct.txt").exists()

    allowed = await executor.execute(
        call, context=ToolExecutionContext(tool_call=call, run_id="r", mode=AgentMode.NORMAL)
    )
    assert allowed.success is True
    assert (tmp_path / "direct.txt").read_text(encoding="utf-8") == "x"


# 函数说明：test_hook_rejects_before_approval_is_requested
# 用途：回归验证回归测试与测试辅助中的 `hook_rejects_before_approval_is_requested` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `TrackingTool` →
# `tools.register` → `RecordingGate` → `ToolExecutor` → `_call`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`(result.error or '').startswith(ROLE_REJECTION_PREFIX)`。
#   验证条件：`gate.requests == []`。
#   验证条件：`guarded.executions == 0`。
async def test_hook_rejects_before_approval_is_requested(tmp_path: Path) -> None:
    tools = ToolRegistry()
    guarded = TrackingTool("http_request", permission=ToolPermission.HUMAN_APPROVAL)
    tools.register(guarded)
    gate = RecordingGate()
    executor = ToolExecutor(tools, approval_gate=gate)
    call = _call("http_request", {"query": "x"})

    result = await executor.execute(
        call, context=ToolExecutionContext(tool_call=call, run_id="r", mode=AgentMode.EXECUTE)
    )

    assert result.success is False
    assert (result.error or "").startswith(ROLE_REJECTION_PREFIX)
    assert gate.requests == []
    assert guarded.executions == 0


# ---------------------------------------------------------------------------
# 第三层：工具自身
# ---------------------------------------------------------------------------


# 函数说明：test_write_file_refuses_in_read_only_modes
# 用途：回归验证回归测试与测试辅助中的 `write_file_refuses_in_read_only_modes` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`WriteFileTool` → `_call` →
# `pytest.raises` → `tool.execute_with_context` → `ToolExecutionContext` →
# `(tmp_path / 'a.txt').exists`。
# 分支与异常：
#   验证条件：`not (tmp_path / 'a.txt').exists()`。
#   验证条件：`(tmp_path / 'a.txt').exists()`。
#   预期异常：`pytest.raises(PermissionError)`。
@pytest.mark.parametrize("mode", [AgentMode.AUDIT, AgentMode.MANAGE])
async def test_write_file_refuses_in_read_only_modes(tmp_path: Path, mode: AgentMode) -> None:
    tool = WriteFileTool(tmp_path)
    arguments = {"path": "a.txt", "content": "x"}
    call = _call("write_file", arguments)

    with pytest.raises(PermissionError):
        await tool.execute_with_context(arguments, ToolExecutionContext(tool_call=call, mode=mode))
    assert not (tmp_path / "a.txt").exists()

    await tool.execute_with_context(
        arguments, ToolExecutionContext(tool_call=call, mode=AgentMode.EXECUTE)
    )
    assert (tmp_path / "a.txt").exists()


class CapturingSupervisor(SandboxSupervisor):
    """生成真实的 Docker 启动参数并记录下来，然后改为本地执行一个空命令，不依赖 Docker。"""

    # 函数说明：CapturingSupervisor.__init__
    # 用途：初始化 CapturingSupervisor；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   workspace：目标工作区，类型 `Path`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super` →
    # `DockerSandboxBackend`。
    # 副作用与资源：
    #   更新对象字段：`self.launches`。
    def __init__(self, workspace: Path) -> None:
        super().__init__(
            workspace,
            native_backend=DockerSandboxBackend(
                workspace, docker_command=sys.executable, image="test-muharness-sandbox:latest"
            ),
        )
        self.launches: list[SandboxLaunchSpec] = []

    # 函数说明：CapturingSupervisor.prepare_launch
    # 用途：准备`launch`，供回归测试与测试辅助使用。
    # 参数：
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：类型 `SandboxLaunchSpec`；返回 `SandboxLaunchSpec(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().prepare_launch` → `super`
    #  → `shutil.rmtree` → `SandboxLaunchSpec`。
    def prepare_launch(self, **kwargs) -> SandboxLaunchSpec:  # type: ignore[override]
        spec = super().prepare_launch(**kwargs)
        self.launches.append(spec)
        for path in spec.cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)
        return SandboxLaunchSpec(
            command=sys.executable,
            args=("-c", "pass"),
            cwd=spec.cwd,
            env=spec.env,
            backend="capture",
            sandboxed=True,
        )


# 函数说明：_workspace_mount
# 用途：在回归测试与测试辅助中处理 `_workspace_mount`，通过 `m.endswith` 完成首个内部处
# 理步骤。
# 参数：
#   spec：`spec`输入或配置值，类型 `SandboxLaunchSpec`。
# 返回：类型 `str`；返回 `next((m for m in mounts if m.endswith('target=/workspace') or
# 'target=/workspace,' in m))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`next` → `m.endswith`。
def _workspace_mount(spec: SandboxLaunchSpec) -> str:
    mounts = [spec.args[i + 1] for i, value in enumerate(spec.args) if value == "--mount"]
    return next(m for m in mounts if m.endswith("target=/workspace") or "target=/workspace," in m)


# 函数说明：test_shell_mounts_workspace_readonly_only_for_audit
# 用途：回归验证回归测试与测试辅助中的 `shell_mounts_workspace_readonly_only_for_audit`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
#   readonly：`readonly`输入或配置值，类型 `bool`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`CapturingSupervisor` →
# `ShellCommandTool` → `tool.execute_with_context` → `ToolExecutionContext` → `_call` →
# `_workspace_mount(spec).endswith`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`output['exit_code'] == 0`。
#   验证条件：`spec.backend == 'docker'`。
#   验证条件：`_workspace_mount(spec).endswith(',readonly') is readonly`。
@pytest.mark.parametrize(
    ("mode", "readonly"),
    [(AgentMode.AUDIT, True), (AgentMode.EXECUTE, False), (AgentMode.NORMAL, False)],
)
async def test_shell_mounts_workspace_readonly_only_for_audit(
    tmp_path: Path, mode: AgentMode, readonly: bool
) -> None:
    supervisor = CapturingSupervisor(tmp_path)
    tool = ShellCommandTool(tmp_path, sandbox_supervisor=supervisor)
    arguments = {"command": "touch should-fail"}

    output = await tool.execute_with_context(
        arguments, ToolExecutionContext(tool_call=_call("run_shell_command", arguments), mode=mode)
    )

    assert output["exit_code"] == 0
    [spec] = supervisor.launches
    assert spec.backend == "docker"
    assert _workspace_mount(spec).endswith(",readonly") is readonly
