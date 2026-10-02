
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"


class ApprovalScope(StrEnum):

    ONCE = "once"
    RUN = "run"
    CONVERSATION = "conversation"


class ApprovalResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    decision: ApprovalDecision
    scope: ApprovalScope = ApprovalScope.ONCE


@dataclass(frozen=True)
class ApprovalRequest:

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    description: str = ""
    run_id: str | None = None
    conversation_id: str | None = None
    # 函数说明：ApprovalRequest.summary
    # 用途：在工具注册、执行与权限钩子中处理 `summary`，通过 `json.dumps` 完成首个内部处
    # 理步骤。
    # 参数：
    #   max_arguments：调用参数输入或配置值，类型 `int`；默认 `500`。
    # 返回：类型 `str`；返回 `'\n'.join(lines)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
    def summary(self, *, max_arguments: int = 500) -> str:
        serialized = json.dumps(
            self.arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(serialized) > max_arguments:
            serialized = serialized[:max_arguments] + "…"
        lines = [
            f"工具: {self.tool_name}",
            f"说明: {self.description or '(无)'}",
            f"参数: {serialized}",
        ]
        return "\n".join(lines)


class ApprovalGate(ABC):

    # 函数说明：ApprovalGate.request_approval
    # 用途：处理工具注册、执行与权限钩子中的 `request_approval` 数据；结果及边界条件见下
    # 方说明。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；不返回结果值（隐式 None）。
    @abstractmethod
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        pass


class AutoApproveGate(ApprovalGate):

    # 函数说明：AutoApproveGate.request_approval
    # 用途：返回 `ApprovalResponse(decision=ApprovalDecision.APPROVED)`，提供
    # AutoApproveGate 的派生值。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；返回
    # `ApprovalResponse(decision=ApprovalDecision.APPROVED)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalResponse`。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(decision=ApprovalDecision.APPROVED)


class DenyAllGate(ApprovalGate):

    # 函数说明：DenyAllGate.request_approval
    # 用途：返回 `ApprovalResponse(decision=ApprovalDecision.DENIED)`，提供 DenyAllGate
    # 的派生值。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；返回
    # `ApprovalResponse(decision=ApprovalDecision.DENIED)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalResponse`。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(decision=ApprovalDecision.DENIED)


class ConsoleApprovalGate(ApprovalGate):

    # 函数说明：ConsoleApprovalGate.__init__
    # 用途：初始化 ConsoleApprovalGate；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   prompt_prefix：`prompt_prefix`输入或配置值，类型 `str`；默认 `'[人工审核]'`。
    #   rule_label_factory：规则构造工厂，类型 `Callable[[ApprovalRequest], str] | None`
    # ；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._prompt_prefix`、`self._rule_label_factory`。
    def __init__(
        self,
        *,
        prompt_prefix: str = "[人工审核]",
        rule_label_factory: Callable[[ApprovalRequest], str] | None = None,
    ) -> None:
        self._prompt_prefix = prompt_prefix
        self._rule_label_factory = rule_label_factory

    # 函数说明：ConsoleApprovalGate.request_approval
    # 用途：在工具注册、执行与权限钩子中处理 `request_approval`，通过
    # `self._rule_label_factory` 完成首个内部处理步骤。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；按分支返回
    # `ApprovalResponse(decision=ApprovalDecision.APPROVED, scope=ApprovalScope.RUN)`；`
    # ApprovalResponse(decision=ApprovalDecision.APPROVED, scope=ApprovalScope.
    # CONVERSATION)`；`ApprovalResponse(decision=ApprovalDecision.APPROVED)`；
    # `ApprovalResponse(decision=ApprovalDecision.DENIED)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._rule_label_factory` →
    # `request.summary` → `asyncio.to_thread` → `ApprovalResponse`。
    # 分支与异常：
    #   当 `answer == '2'` 时，返回 `ApprovalResponse(…)`。
    #   当 `answer == '3'` 时，返回 `ApprovalResponse(…)`。
    #   当 `answer == '1'` 时，返回
    # `ApprovalResponse(decision=ApprovalDecision.APPROVED)`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        label = (
            self._rule_label_factory(request)
            if self._rule_label_factory is not None
            else "当前工作记住该安全规则（相同参数自动通过）"
        )
        prompt = (
            f"\n{self._prompt_prefix}\n"
            f"{request.summary()}\n"
            f"1. 仅允许这一次\n"
            f"2. 当前 Run 内允许完全相同的操作\n"
            f"3. {label}\n"
            f"4. 拒绝\n"
            f"请选择 [1/2/3/4]: "
        )
        answer = (await asyncio.to_thread(input, prompt)).strip()
        if answer == "2":
            return ApprovalResponse(
                decision=ApprovalDecision.APPROVED,
                scope=ApprovalScope.RUN,
            )
        if answer == "3":
            return ApprovalResponse(
                decision=ApprovalDecision.APPROVED,
                scope=ApprovalScope.CONVERSATION,
            )
        if answer == "1":
            return ApprovalResponse(decision=ApprovalDecision.APPROVED)
        return ApprovalResponse(decision=ApprovalDecision.DENIED)


__all__ = [
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalScope",
    "AutoApproveGate",
    "ConsoleApprovalGate",
    "DenyAllGate",
]
