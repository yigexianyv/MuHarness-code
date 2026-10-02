"""长任务里的审批策略：沙箱内的 shell 调用自动批准，其余照常请示。

run_shell_command 在长任务里总是在禁网的 Docker 沙箱中执行：Auditor 只读挂载 workspace，
Executor 只能写 workspace。逐条人工审批这些命令负担很重，却几乎不增加安全性；
真正越界的是网络和外部系统（http_request、MCP 工具等），它们仍然走原来的审批。
已保存的“拒绝”规则在审批之前生效，这里不会绕过它们。
"""

from __future__ import annotations

from collections.abc import Collection

from app.tools.approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
)

SANDBOXED_TOOLS = frozenset({"run_shell_command"})


class SandboxAutoApproveGate(ApprovalGate):
    """对沙箱内工具直接批准，其余请求交给原来的审批入口。"""

    # 函数说明：SandboxAutoApproveGate.__init__
    # 用途：初始化 SandboxAutoApproveGate；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   inner：`inner`输入或配置值，类型 `ApprovalGate`。
    #   tools：可用工具定义或工具实例集合，类型 `Collection[str]`；默认
    # `SANDBOXED_TOOLS`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`frozenset`。
    # 副作用与资源：
    #   更新对象字段：`self._inner`、`self._tools`。
    def __init__(self, inner: ApprovalGate, tools: Collection[str] = SANDBOXED_TOOLS) -> None:
        self._inner = inner
        self._tools = frozenset(tools)

    # 函数说明：SandboxAutoApproveGate.request_approval
    # 用途：在规划、执行、审计协作中处理 `request_approval`，通过
    # `self._inner.request_approval` 完成首个内部处理步骤。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；按分支返回
    # `ApprovalResponse(decision=ApprovalDecision.APPROVED)`；
    # `await self._inner.request_approval(request)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalResponse` →
    # `self._inner.request_approval`。
    # 分支与异常：
    #   当 `request.tool_name in self._tools` 时，返回
    # `ApprovalResponse(decision=ApprovalDecision.APPROVED)`。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        if request.tool_name in self._tools:
            return ApprovalResponse(decision=ApprovalDecision.APPROVED)
        return await self._inner.request_approval(request)


__all__ = ["SANDBOXED_TOOLS", "SandboxAutoApproveGate"]
