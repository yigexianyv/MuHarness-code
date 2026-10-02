from unittest.mock import patch

import pytest

from app.models.types import ToolCall, ToolDefinition, ToolPermission
from app.tools.approval import AutoApproveGate
from app.tools.base import BaseTool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


# 函数说明：test_approval_wait_is_separate_from_execution
# 用途：回归验证回归测试与测试辅助中的 `approval_wait_is_separate_from_execution` 场景，
# 下方断言说明列出实际通过条件。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `Tool` → `patch` → `ToolExecutor(registry, approval_gate=Gate()).execute` →
# `ToolExecutor`；另有 2 个调用点。
# 资源/并发边界：
# `patch('app.tools.executor.perf_counter', side_effect=lambda: clock[0])`，上下文退出时
# 执行相应清理。
# 分支与异常：
#   验证条件：`result.success`。
#   验证条件：`result.duration_ms == 52000`。
#   验证条件：`result.approval_wait_ms == 48000`。
#   验证条件：`result.execution_duration_ms == 4000`。
@pytest.mark.asyncio
async def test_approval_wait_is_separate_from_execution():
    clock = [0.0]

    class Gate(AutoApproveGate):
        # 函数说明：test_approval_wait_is_separate_from_execution.Gate.request_approval
        # 用途：在回归测试与测试辅助中处理 `request_approval`，通过
        # `super().request_approval` 完成首个内部处理步骤。
        # 参数：
        #   request：待处理的请求对象。
        # 返回：返回 `await super().request_approval(request)`。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().request_approval` →
        # `super`。
        # 闭包依赖：从外层读取 `clock`。
        async def request_approval(self, request):
            clock[0] += 48.0
            return await super().request_approval(request)

    class Tool(BaseTool):
        # 函数说明：test_approval_wait_is_separate_from_execution.Tool.definition
        # 用途：提供 Tool 的模型可见定义，包含名称、说明、参数结构及权限声明。
        # 返回：返回 `ToolDefinition(…)`。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
        @property
        def definition(self):
            return ToolDefinition(
                name="timed_tool", description="Timing test", parameters={},
                permission=ToolPermission.HUMAN_APPROVAL,
            )

        # 函数说明：test_approval_wait_is_separate_from_execution.Tool.execute
        # 用途：执行Tool，供回归测试与测试辅助使用。
        # 参数：
        #   arguments：工具调用的参数对象或 JSON 文本。
        # 返回：返回 `'done'`。
        # 闭包依赖：从外层读取 `clock`。
        async def execute(self, arguments):
            clock[0] += 4.0
            return "done"

    registry = ToolRegistry()
    registry.register(Tool())
    with patch("app.tools.executor.perf_counter", side_effect=lambda: clock[0]):
        result = await ToolExecutor(registry, approval_gate=Gate()).execute(
            ToolCall(id="timed", name="timed_tool", arguments={})
        )
    assert result.success
    assert result.duration_ms == 52000
    assert result.approval_wait_ms == 48000
    assert result.execution_duration_ms == 4000
