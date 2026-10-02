
from __future__ import annotations

from typing import Any

import pytest

from app.models.types import ToolCall, ToolDefinition, ToolPermission
from app.tools import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    AutoApproveGate,
    BaseTool,
    ToolExecutor,
    ToolRegistry,
)


class StubTool(BaseTool):
    # 函数说明：StubTool.__init__
    # 用途：初始化 StubTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   permission：所需权限等级，类型 `ToolPermission`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._name`、`self._permission`、`self.executions`。
    def __init__(self, name: str, permission: ToolPermission) -> None:
        self._name = name
        self._permission = permission
        self.executions = 0

    # 函数说明：StubTool.definition
    # 用途：提供 StubTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"stub {self._name}",
            permission=self._permission,
        )

    # 函数说明：StubTool.execute
    # 用途：执行StubTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `str`；返回 `f'ran:{self._name}'`。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        self.executions += 1
        return f"ran:{self._name}"


class RecordingGate(ApprovalGate):
    # 函数说明：RecordingGate.__init__
    # 用途：初始化 RecordingGate；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   decision：权限、上下文或审计决策，类型 `ApprovalDecision`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.decision`、`self.requests`。
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    # 函数说明：RecordingGate.request_approval
    # 用途：在回归测试与测试辅助中处理 `request_approval`，通过 `self.requests.append`
    # 完成首个内部处理步骤。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；返回 `ApprovalResponse(decision=self.decision)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalResponse`。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        self.requests.append(request)
        return ApprovalResponse(decision=self.decision)


# 函数说明：build
# 用途：构建回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   *tools：额外位置参数，按实现向内部调用传递。
# 返回：类型 `ToolRegistry`；返回 `registry`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register`。
def build(*tools: StubTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# 函数说明：test_forbidden_tools_are_hidden_from_model_definitions
# 用途：回归验证回归测试与测试辅助中的
# `forbidden_tools_are_hidden_from_model_definitions` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build` → `StubTool` →
# `registry.definitions`。
# 分支与异常：
#   验证条件：`names == {'allowed_tool', 'review_tool'}`。
#   验证条件：`'secret_tool' not in names`。
@pytest.mark.asyncio
async def test_forbidden_tools_are_hidden_from_model_definitions() -> None:
    registry = build(
        StubTool("allowed_tool", ToolPermission.ALLOWED),
        StubTool("review_tool", ToolPermission.HUMAN_APPROVAL),
        StubTool("secret_tool", ToolPermission.FORBIDDEN),
    )

    names = {definition.name for definition in registry.definitions()}
    assert names == {"allowed_tool", "review_tool"}
    assert "secret_tool" not in names


# 函数说明：test_forbidden_tool_is_blocked_even_with_auto_approve
# 用途：回归验证回归测试与测试辅助中的
# `forbidden_tool_is_blocked_even_with_auto_approve` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StubTool` → `ToolExecutor` → `build`
# → `AutoApproveGate` → `executor.execute` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`'forbidden' in (result.error or '')`。
#   验证条件：`secret.executions == 0`。
@pytest.mark.asyncio
async def test_forbidden_tool_is_blocked_even_with_auto_approve() -> None:
    secret = StubTool("secret_tool", ToolPermission.FORBIDDEN)
    executor = ToolExecutor(build(secret), approval_gate=AutoApproveGate())

    result = await executor.execute(
        ToolCall(id="s-1", name="secret_tool", arguments={})
    )

    assert result.success is False
    assert "forbidden" in (result.error or "")
    assert secret.executions == 0


# 函数说明：test_human_approval_denied_by_default_gate
# 用途：回归验证回归测试与测试辅助中的 `human_approval_denied_by_default_gate` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StubTool` →
# `ToolExecutor(build(tool)).execute` → `ToolExecutor` → `build` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`'human approval' in (result.error or '')`。
#   验证条件：`tool.executions == 0`。
@pytest.mark.asyncio
async def test_human_approval_denied_by_default_gate() -> None:
    tool = StubTool("review_tool", ToolPermission.HUMAN_APPROVAL)

    result = await ToolExecutor(build(tool)).execute(
        ToolCall(id="r-1", name="review_tool", arguments={})
    )

    assert result.success is False
    assert "human approval" in (result.error or "")
    assert tool.executions == 0


# 函数说明：test_human_approval_denied_captures_gate_request
# 用途：回归验证回归测试与测试辅助中的 `human_approval_denied_captures_gate_request` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RecordingGate` → `ToolExecutor` →
# `build` → `StubTool` → `executor.execute` → `ToolCall`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`len(gate.requests) == 1`。
#   验证条件：`gate.requests[0].tool_name == 'review_tool'`。
#   验证条件：`gate.requests[0].arguments == {'a': 1}`。
@pytest.mark.asyncio
async def test_human_approval_denied_captures_gate_request() -> None:
    gate = RecordingGate(ApprovalDecision.DENIED)
    executor = ToolExecutor(
        build(StubTool("review_tool", ToolPermission.HUMAN_APPROVAL)),
        approval_gate=gate,
    )

    result = await executor.execute(
        ToolCall(id="r-1", name="review_tool", arguments={"a": 1})
    )

    assert result.success is False
    assert len(gate.requests) == 1
    assert gate.requests[0].tool_name == "review_tool"
    assert gate.requests[0].arguments == {"a": 1}
    assert "stub review_tool" in gate.requests[0].summary()
    assert '{"a":1}' in gate.requests[0].summary()


# 函数说明：test_human_approval_approved_executes
# 用途：回归验证回归测试与测试辅助中的 `human_approval_approved_executes` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RecordingGate` → `StubTool` →
# `ToolExecutor` → `build` → `executor.execute` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`result.output == 'ran:review_tool'`。
#   验证条件：`tool.executions == 1`。
@pytest.mark.asyncio
async def test_human_approval_approved_executes() -> None:
    gate = RecordingGate(ApprovalDecision.APPROVED)
    tool = StubTool("review_tool", ToolPermission.HUMAN_APPROVAL)
    executor = ToolExecutor(build(tool), approval_gate=gate)

    result = await executor.execute(
        ToolCall(id="r-1", name="review_tool", arguments={})
    )

    assert result.success is True
    assert result.output == "ran:review_tool"
    assert tool.executions == 1


# 函数说明：test_auto_approve_gate_runs_human_approval_tools
# 用途：回归验证回归测试与测试辅助中的 `auto_approve_gate_runs_human_approval_tools` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StubTool` → `ToolExecutor` → `build`
# → `AutoApproveGate` → `executor.execute` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`tool.executions == 1`。
@pytest.mark.asyncio
async def test_auto_approve_gate_runs_human_approval_tools() -> None:
    tool = StubTool("review_tool", ToolPermission.HUMAN_APPROVAL)
    executor = ToolExecutor(build(tool), approval_gate=AutoApproveGate())

    result = await executor.execute(
        ToolCall(id="r-1", name="review_tool", arguments={})
    )

    assert result.success is True
    assert tool.executions == 1


# 函数说明：test_execution_records_capture_success_and_failures
# 用途：回归验证回归测试与测试辅助中的 `execution_records_capture_success_and_failures`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolExecutor` → `build` → `StubTool`
# → `executor.execute` → `ToolCall` → `(missing.error or '').lower`。
# 分支与异常：
#   验证条件：`len(records) == 3`。
#   验证条件：`ok.tool_name == 'ok_tool'`。
#   验证条件：`ok.success is True`。
#   验证条件：`ok.error is None`。
@pytest.mark.asyncio
async def test_execution_records_capture_success_and_failures() -> None:
    executor = ToolExecutor(
        build(
            StubTool("ok_tool", ToolPermission.ALLOWED),
            StubTool("bad_tool", ToolPermission.FORBIDDEN),
        )
    )

    await executor.execute(ToolCall(id="1", name="ok_tool", arguments={}))
    await executor.execute(ToolCall(id="2", name="bad_tool", arguments={}))
    await executor.execute(ToolCall(id="3", name="missing_tool", arguments={}))

    records = executor.execution_records
    assert len(records) == 3

    ok = records[0]
    assert ok.tool_name == "ok_tool"
    assert ok.success is True
    assert ok.error is None
    assert ok.permission == ToolPermission.ALLOWED.value
    assert ok.duration_ms >= 0

    bad = records[1]
    assert bad.success is False
    assert bad.error is not None
    assert "forbidden" in bad.error

    missing = records[2]
    assert missing.success is False
    assert "not found" in (missing.error or "").lower()


# 函数说明：test_in_memory_logger_respects_maxlen
# 用途：回归验证回归测试与测试辅助中的 `in_memory_logger_respects_maxlen` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`InMemoryExecutionLogger` →
# `ToolExecutor` → `build` → `StubTool` → `executor.execute` → `ToolCall`；另有 1 个调用
# 点。
# 分支与异常：
#   验证条件：`logger.count == 2`。
#   验证条件：`[r.tool_call_id for r in logger.recent()] == ['1', '2']`。
@pytest.mark.asyncio
async def test_in_memory_logger_respects_maxlen() -> None:
    from app.tools import InMemoryExecutionLogger

    logger = InMemoryExecutionLogger(maxlen=2)
    executor = ToolExecutor(build(StubTool("t", ToolPermission.ALLOWED)), logger=logger)

    for index in range(3):
        await executor.execute(
            ToolCall(id=f"{index}", name="t", arguments={})
        )

    assert logger.count == 2
    assert [r.tool_call_id for r in logger.recent()] == ["1", "2"]


# 函数说明：test_registry_name_validation_and_unregister
# 用途：回归验证回归测试与测试辅助中的 `registry_name_validation_and_unregister` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `pytest.raises` →
# `registry.register` → `StubTool` → `registry.names` → `registry.unregister`。
# 分支与异常：
#   验证条件：`registry.names() == ('good_name',)`。
#   验证条件：`registry.unregister('good_name') is tool`。
#   预期异常：`pytest.raises(ValueError, match='empty')`。
#   预期异常：`pytest.raises(ValueError, match='letters, digits')`。
#   预期异常：`pytest.raises(KeyError)`。
def test_registry_name_validation_and_unregister() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="empty"):
        registry.register(StubTool("", ToolPermission.ALLOWED))
    with pytest.raises(ValueError, match="letters, digits"):
        registry.register(StubTool("bad name", ToolPermission.ALLOWED))

    tool = StubTool("good_name", ToolPermission.ALLOWED)
    registry.register(tool)
    assert registry.names() == ("good_name",)
    assert registry.unregister("good_name") is tool
    with pytest.raises(KeyError):
        registry.get("good_name")
