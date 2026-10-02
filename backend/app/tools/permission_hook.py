
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.models.types import ToolPermission

from .approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalScope,
)
from .hooks import ToolExecutionContext, ToolHook, ToolHookDecision
from .permissions.models import PermissionEffect, PermissionRule
from .permissions.policy import PermissionPolicyEngine
from .permissions.rule_factory import build_safe_rule
from .permissions.store import PermissionRuleStore

RuleFactory = Callable[..., PermissionRule]


@dataclass(frozen=True, slots=True)
class _ApprovalOutcome:

    response: ApprovalResponse
    rule: PermissionRule | None = None


# 函数说明：_scope_ids
# 用途：处理工具注册、执行与权限钩子中的 `_scope_ids` 数据；结果及边界条件见下方说明。
# 参数：
#   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
# 返回：类型 `tuple[str, ...]`；返回 `tuple((scope_id for scope_id in ids if scope_id))`
# 。
def _scope_ids(context: ToolExecutionContext) -> tuple[str, ...]:

    ids = (context.run_id, context.conversation_id)
    return tuple(scope_id for scope_id in ids if scope_id)


# 函数说明：_scope_id_for
# 用途：处理工具注册、执行与权限钩子中的 `_scope_id_for` 数据；结果及边界条件见下方说明
# 。
# 参数：
#   scope：记忆、规则或查询作用域，类型 `ApprovalScope`。
#   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
# 返回：类型 `str | None`；按分支返回 `context.run_id`；`context.conversation_id`；
# `None`。
# 分支与异常：
#   当 `scope is ApprovalScope.RUN` 时，返回 `context.run_id`。
#   当 `scope is ApprovalScope.CONVERSATION` 时，返回 `context.conversation_id`。
def _scope_id_for(scope: ApprovalScope, context: ToolExecutionContext) -> str | None:
    if scope is ApprovalScope.RUN:
        return context.run_id
    if scope is ApprovalScope.CONVERSATION:
        return context.conversation_id
    return None


class PermissionHook(ToolHook):

    critical = True

    # 函数说明：PermissionHook.__init__
    # 用途：初始化 PermissionHook；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   approval_gate：工具审批门，类型 `ApprovalGate`。
    #   policy：权限决策策略，类型 `PermissionPolicyEngine | None`；默认 `None`。
    #   rule_store：权限规则存储，类型 `PermissionRuleStore | None`；默认 `None`。
    #   rule_factory：根据审批结果构建规则的回调，类型 `RuleFactory`；默认
    # `build_safe_rule`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._approval_gate`、`self._policy`、`self._rule_store`、
    # `self._rule_factory`。
    def __init__(
        self,
        approval_gate: ApprovalGate,
        *,
        policy: PermissionPolicyEngine | None = None,
        rule_store: PermissionRuleStore | None = None,
        rule_factory: RuleFactory = build_safe_rule,
    ) -> None:
        self._approval_gate = approval_gate
        self._policy = policy
        self._rule_store = rule_store
        self._rule_factory = rule_factory

    # 函数说明：PermissionHook.before_execute
    # 用途：按禁用规则与已存权限规则评估工具调用，必要时提出审批请求。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `ToolHookDecision | None`；按分支返回 `None`；`ToolHookDecision(…)`；
    # `ToolHookDecision(approval_request=request, matched_rule=verdict.rule)`；
    # `ToolHookDecision(approval_request=request)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolHookDecision` →
    # `ApprovalRequest` → `self._policy.evaluate` → `_scope_ids`。
    # 分支与异常：
    #   当 `definition is None` 时，返回 `None`。
    #   当 `permission is ToolPermission.FORBIDDEN` 时，返回 `ToolHookDecision(…)`。
    #   当 `permission is not ToolPermission.HUMAN_APPROVAL` 时，返回 `None`。
    #   当 `verdict.effect is PermissionEffect.DENY` 时，返回 `ToolHookDecision(…)`。
    async def before_execute(
        self,
        context: ToolExecutionContext,
    ) -> ToolHookDecision | None:
        """按禁用规则与已存权限规则评估工具调用，必要时提出审批请求。"""
        definition = context.tool_definition
        if definition is None:
            return None

        permission = definition.permission
        if permission is ToolPermission.FORBIDDEN:
            return ToolHookDecision(
                denied_reason=(
                    f"Tool '{context.tool_call.name}' is forbidden "
                    "for model execution."
                )
            )
        if permission is not ToolPermission.HUMAN_APPROVAL:
            return None

        request = ApprovalRequest(
            tool_call_id=context.tool_call.id,
            tool_name=context.tool_call.name,
            arguments=context.arguments or {},
            description=definition.description,
            run_id=context.run_id,
            conversation_id=context.conversation_id,
        )

        if self._policy is not None:
            verdict = await self._policy.evaluate(
                tool_name=context.tool_call.name,
                arguments=context.arguments or {},
                scope_ids=_scope_ids(context),
            )
            if verdict.effect is PermissionEffect.DENY:
                return ToolHookDecision(
                    denied_reason=(
                        f"Tool '{context.tool_call.name}' execution was denied "
                        "by a stored permission rule."
                    )
                )
            if verdict.effect is PermissionEffect.ALLOW and verdict.rule_id:
                return ToolHookDecision(
                    approval_request=request,
                    matched_rule=verdict.rule,
                )

        return ToolHookDecision(approval_request=request)

    # 函数说明：PermissionHook.request_approval
    # 用途：等待用户审批，并按所选范围保存后续可复用的规则。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext | None`；默认 `None`。
    # 返回：类型 `_ApprovalOutcome`；按分支返回
    # `_ApprovalOutcome(response=response, rule=rule)`；
    # `_ApprovalOutcome(response=response)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._approval_gate.request_approval` → `self._create_rule` → `_ApprovalOutcome`
    # 。
    # 分支与异常：
    #   当 `rule is not None` 时，返回 `_ApprovalOutcome(response=response, rule=rule)`
    # 。
    async def request_approval(
        self,
        request: ApprovalRequest,
        *,
        context: ToolExecutionContext | None = None,
    ) -> _ApprovalOutcome:

        """等待用户审批，并按所选范围保存后续可复用的规则。"""
        response = await self._approval_gate.request_approval(request)
        if (
            response.decision is ApprovalDecision.APPROVED
            and response.scope in (ApprovalScope.RUN, ApprovalScope.CONVERSATION)
            and context is not None
        ):
            rule = await self._create_rule(request, response.scope, context)
            if rule is not None:
                return _ApprovalOutcome(response=response, rule=rule)
        return _ApprovalOutcome(response=response)

    # 函数说明：PermissionHook._create_rule
    # 用途：创建规则，供工具注册、执行与权限钩子使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    #   scope：记忆、规则或查询作用域，类型 `ApprovalScope`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `PermissionRule | None`；按分支返回 `None`；`rule`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_scope_id_for` →
    # `self._rule_factory` → `self._rule_store.add`。
    # 分支与异常：
    #   当 `self._rule_store is None` 时，返回 `None`。
    #   当 `not scope_id` 时，返回 `None`。
    async def _create_rule(
        self,
        request: ApprovalRequest,
        scope: ApprovalScope,
        context: ToolExecutionContext,
    ) -> PermissionRule | None:
        if self._rule_store is None:
            return None
        scope_id = _scope_id_for(scope, context)
        if not scope_id:
            return None
        rule = self._rule_factory(
            tool_name=request.tool_name,
            arguments=request.arguments,
            scope=scope,
            scope_id=scope_id,
        )
        await self._rule_store.add(rule)
        return rule

    # 函数说明：PermissionHook.clear_run_rules
    # 用途：清除指定运行关联的临时权限规则，返回删除数量。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `int`；按分支返回 `0`；
    # `await self._rule_store.remove_scope(ApprovalScope.RUN, run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._rule_store.remove_scope`。
    # 分支与异常：
    #   当 `self._rule_store is None` 时，返回 `0`。
    async def clear_run_rules(self, run_id: str) -> int:

        if self._rule_store is None:
            return 0
        return await self._rule_store.remove_scope(ApprovalScope.RUN, run_id)

    # 函数说明：PermissionHook.denied_reason
    # 用途：处理工具注册、执行与权限钩子中的 `denied_reason` 数据；结果及边界条件见下方
    # 说明。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   decision：权限、上下文或审计决策，类型 `ApprovalDecision`。
    # 返回：类型 `str | None`；按分支返回 `None`；`f"Tool '{context.tool_call.name}'
    # execution was denied (requires human approval)."`。
    # 分支与异常：
    #   当 `decision is ApprovalDecision.APPROVED` 时，返回 `None`。
    @staticmethod
    def denied_reason(
        context: ToolExecutionContext,
        decision: ApprovalDecision,
    ) -> str | None:

        if decision is ApprovalDecision.APPROVED:
            return None
        return (
            f"Tool '{context.tool_call.name}' execution was denied "
            "(requires human approval)."
        )


__all__ = ["PermissionHook"]
