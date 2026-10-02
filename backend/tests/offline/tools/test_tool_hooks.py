
from __future__ import annotations

from typing import Any

import pytest

from app.models.types import ToolCall, ToolDefinition, ToolPermission, ToolResult
from app.tools import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    BaseTool,
    ToolExecutionContext,
    ToolExecutor,
    ToolHook,
    ToolRegistry,
)


class HookTool(BaseTool):
    # 函数说明：HookTool.__init__
    # 用途：初始化 HookTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   permission：所需权限等级，类型 `ToolPermission`；默认 `ToolPermission.ALLOWED`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._permission`、`self.executions`。
    def __init__(self, permission: ToolPermission = ToolPermission.ALLOWED) -> None:
        self._permission = permission
        self.executions = 0

    # 函数说明：HookTool.definition
    # 用途：提供 HookTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="hook_tool",
            description="用于验证 Hook 生命周期",
            permission=self._permission,
        )

    # 函数说明：HookTool.execute
    # 用途：执行HookTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `value`
    # 。
    # 返回：类型 `str`；返回 `f"value:{arguments['value']}"`。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        self.executions += 1
        return f"value:{arguments['value']}"


class RecordingHook(ToolHook):
    # 函数说明：RecordingHook.__init__
    # 用途：初始化 RecordingHook；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.events`、`self.contexts`、`self.results`。
    def __init__(self) -> None:
        self.events: list[str] = []
        self.contexts: list[ToolExecutionContext] = []
        self.results: list[ToolResult] = []

    # 函数说明：RecordingHook.before_execute
    # 用途：在 `execute` 前后执行 RecordingHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def before_execute(self, context: ToolExecutionContext) -> None:
        self.events.append("before")
        self.contexts.append(context)

    # 函数说明：RecordingHook.on_approval_required
    # 用途：处理 `approval_required` 事件；调用路径和状态变化见下方说明。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def on_approval_required(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
    ) -> None:
        self.events.append("approval_required")

    # 函数说明：RecordingHook.on_approval_completed
    # 用途：处理 `approval_completed` 事件；调用路径和状态变化见下方说明。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    #   decision：权限、上下文或审计决策，类型 `ApprovalDecision`。
    #   rule：待匹配的权限规则，类型 `Any`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def on_approval_completed(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        rule: Any = None,
    ) -> None:
        suffix = f":{rule.id}" if rule is not None else ""
        self.events.append(f"approval_completed:{decision.value}{suffix}")

    # 函数说明：RecordingHook.after_execute
    # 用途：在 `execute` 前后执行 RecordingHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def after_execute(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        self.events.append("after")
        self.results.append(result)


class FailingHook(ToolHook):
    # 函数说明：FailingHook.before_execute
    # 用途：在 `execute` 前后执行 FailingHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def before_execute(self, context: ToolExecutionContext) -> None:
        raise RuntimeError("before unavailable")

    # 函数说明：FailingHook.after_execute
    # 用途：在 `execute` 前后执行 FailingHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def after_execute(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        raise RuntimeError("after unavailable")


class FailingCriticalHook(ToolHook):
    critical = True

    # 函数说明：FailingCriticalHook.before_execute
    # 用途：在 `execute` 前后执行 FailingCriticalHook 的生命周期钩子。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def before_execute(self, context: ToolExecutionContext) -> None:
        raise RuntimeError("security policy unavailable")


class FixedGate(ApprovalGate):
    # 函数说明：FixedGate.__init__
    # 用途：初始化 FixedGate；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   decision：权限、上下文或审计决策，类型 `ApprovalDecision`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._decision`。
    def __init__(self, decision: ApprovalDecision) -> None:
        self._decision = decision

    # 函数说明：FixedGate.request_approval
    # 用途：返回 `ApprovalResponse(decision=self._decision)`，提供 FixedGate 的派生值。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；返回 `ApprovalResponse(decision=self._decision)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalResponse`。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(decision=self._decision)


class FailingGate(ApprovalGate):
    # 函数说明：FailingGate.request_approval
    # 用途：处理回归测试与测试辅助中的 `request_approval` 数据；结果及边界条件见下方说明
    # 。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；不返回结果值（隐式 None）。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        raise RuntimeError("approval service unavailable")


# 函数说明：build_registry
# 用途：构建`registry`，供回归测试与测试辅助使用。
# 参数：
#   tool：目标工具实例，类型 `BaseTool`。
# 返回：类型 `ToolRegistry`；返回 `registry`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register`。
def build_registry(tool: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


# 函数说明：test_hooks_receive_context_and_success_result
# 用途：回归验证回归测试与测试辅助中的 `hooks_receive_context_and_success_result` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`HookTool` → `RecordingHook` →
# `ToolExecutor` → `build_registry` → `ToolCall` → `executor.execute`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`hook.events == ['before', 'after']`。
#   验证条件：`hook.results == [result]`。
#   验证条件：`context.run_id == 'run-1'`。
@pytest.mark.asyncio
async def test_hooks_receive_context_and_success_result() -> None:
    tool = HookTool()
    hook = RecordingHook()
    executor = ToolExecutor(build_registry(tool), hooks=(hook,))
    call = ToolCall(id="hook-1", name="hook_tool", arguments={"value": 7})

    result = await executor.execute(
        call,
        context=ToolExecutionContext(
            tool_call=call,
            run_id="run-1",
            conversation_id="conversation-1",
            step=2,
        ),
    )

    assert result.success is True
    assert hook.events == ["before", "after"]
    assert hook.results == [result]
    context = hook.contexts[0]
    assert context.run_id == "run-1"
    assert context.conversation_id == "conversation-1"
    assert context.step == 2
    assert context.tool_definition == tool.definition
    assert context.arguments == {"value": 7}
    assert "started_at" in context.metadata


# 函数说明：test_approval_lifecycle_is_dispatched_in_order
# 用途：回归验证回归测试与测试辅助中的 `approval_lifecycle_is_dispatched_in_order` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`HookTool` → `RecordingHook` →
# `ToolExecutor` → `build_registry` → `FixedGate` → `executor.execute`；另有 1 个调用点
# 。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`hook.events == ['before', 'approval_required', 'approval_completed:
# approved', 'after']`。
@pytest.mark.asyncio
async def test_approval_lifecycle_is_dispatched_in_order() -> None:
    tool = HookTool(ToolPermission.HUMAN_APPROVAL)
    hook = RecordingHook()
    executor = ToolExecutor(
        build_registry(tool),
        approval_gate=FixedGate(ApprovalDecision.APPROVED),
        hooks=(hook,),
    )

    result = await executor.execute(
        ToolCall(id="hook-2", name="hook_tool", arguments={"value": 8})
    )

    assert result.success is True
    assert hook.events == [
        "before",
        "approval_required",
        "approval_completed:approved",
        "after",
    ]


# 函数说明：test_observer_hook_failure_does_not_change_tool_result
# 用途：回归验证回归测试与测试辅助中的
# `observer_hook_failure_does_not_change_tool_result` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`HookTool` → `ToolExecutor` →
# `build_registry` → `FailingHook` → `executor.execute` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`result.output == 'value:9'`。
#   验证条件：`tool.executions == 1`。
#   验证条件：`len(executor.execution_records) == 1`。
@pytest.mark.asyncio
async def test_observer_hook_failure_does_not_change_tool_result() -> None:
    tool = HookTool()
    executor = ToolExecutor(build_registry(tool), hooks=(FailingHook(),))

    result = await executor.execute(
        ToolCall(id="hook-3", name="hook_tool", arguments={"value": 9})
    )

    assert result.success is True
    assert result.output == "value:9"
    assert tool.executions == 1
    assert len(executor.execution_records) == 1


# 函数说明：test_custom_hooks_cannot_bypass_forbidden_permission
# 用途：回归验证回归测试与测试辅助中的 `custom_hooks_cannot_bypass_forbidden_permission`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`HookTool` → `RecordingHook` →
# `ToolExecutor` → `build_registry` → `executor.execute` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`'forbidden' in (result.error or '')`。
#   验证条件：`tool.executions == 0`。
#   验证条件：`hook.events == ['before', 'after']`。
@pytest.mark.asyncio
async def test_custom_hooks_cannot_bypass_forbidden_permission() -> None:
    tool = HookTool(ToolPermission.FORBIDDEN)
    hook = RecordingHook()
    executor = ToolExecutor(build_registry(tool), hooks=(hook,))

    result = await executor.execute(
        ToolCall(id="hook-4", name="hook_tool", arguments={"value": 10})
    )

    assert result.success is False
    assert "forbidden" in (result.error or "")
    assert tool.executions == 0
    assert hook.events == ["before", "after"]


# 函数说明：test_critical_hook_failure_denies_execution
# 用途：回归验证回归测试与测试辅助中的 `critical_hook_failure_denies_execution` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`HookTool` → `ToolExecutor` →
# `build_registry` → `FailingCriticalHook` → `executor.execute` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`'Critical tool hook failed' in (result.error or '')`。
#   验证条件：`tool.executions == 0`。
@pytest.mark.asyncio
async def test_critical_hook_failure_denies_execution() -> None:
    tool = HookTool()
    executor = ToolExecutor(
        build_registry(tool),
        hooks=(FailingCriticalHook(),),
    )

    result = await executor.execute(
        ToolCall(id="hook-critical", name="hook_tool", arguments={"value": 10})
    )

    assert result.success is False
    assert "Critical tool hook failed" in (result.error or "")
    assert tool.executions == 0


# 函数说明：test_permission_gate_failure_is_closed_and_recorded
# 用途：回归验证回归测试与测试辅助中的 `permission_gate_failure_is_closed_and_recorded`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`HookTool` → `RecordingHook` →
# `ToolExecutor` → `build_registry` → `FailingGate` → `executor.execute`；另有 1 个调用
# 点。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`'Permission check failed' in (result.error or '')`。
#   验证条件：`tool.executions == 0`。
#   验证条件：`hook.events == ['before', 'approval_required', 'after']`。
@pytest.mark.asyncio
async def test_permission_gate_failure_is_closed_and_recorded() -> None:
    tool = HookTool(ToolPermission.HUMAN_APPROVAL)
    hook = RecordingHook()
    executor = ToolExecutor(
        build_registry(tool),
        approval_gate=FailingGate(),
        hooks=(hook,),
    )

    result = await executor.execute(
        ToolCall(id="hook-5", name="hook_tool", arguments={"value": 11})
    )

    assert result.success is False
    assert "Permission check failed" in (result.error or "")
    assert tool.executions == 0
    assert hook.events == ["before", "approval_required", "after"]
    assert executor.execution_records[0].error == result.error
