
from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any, Literal

from app.models.types import AgentMode, ToolCall, ToolResult

from .approval import ApprovalDecision, ApprovalGate, DenyAllGate
from .base import BaseTool
from .hooks import ToolExecutionContext, ToolHook, ToolHookDecision, ToolHookRunner
from .observability import (
    InMemoryExecutionLogger,
    ObservabilityHook,
    ToolExecutionLogger,
    ToolExecutionRecord,
    _now_iso,
)
from .output import ToolOutputRecorder
from .permission_hook import PermissionHook, RuleFactory
from .permissions.policy import PermissionPolicyEngine
from .permissions.rule_factory import build_safe_rule
from .permissions.store import PermissionRuleStore
from .registry import ToolRegistry
from .role_boundary import RoleBoundaryHook

MAX_TOOL_OUTPUT_CHARS = 20_000


class ToolExecutor:
    # 函数说明：ToolExecutor.__init__
    # 用途：初始化 ToolExecutor；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
    #   timeout_seconds：等待或执行超时，单位为秒，类型 `float`；默认 `30.0`。
    #   max_output_chars：工具输出的字符数上限，类型 `int`；默认 `MAX_TOOL_OUTPUT_CHARS`
    # 。
    #   approval_gate：工具审批门，类型 `ApprovalGate | None`；默认 `None`。
    #   logger：日志或工具执行记录器，类型 `ToolExecutionLogger | None`；默认 `None`。
    #   hooks：附加工具生命周期钩子，类型 `Sequence[ToolHook]`；默认 `()`。
    #   policy_engine：权限规则决策引擎，类型 `PermissionPolicyEngine | None`；默认
    # `None`。
    #   rule_store：权限规则存储，类型 `PermissionRuleStore | None`；默认 `None`。
    #   rule_factory：根据审批结果构建规则的回调，类型 `RuleFactory`；默认
    # `build_safe_rule`。
    #   output_recorder：原始工具输出证据记录器，类型 `ToolOutputRecorder | None`；默认
    # `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`PermissionPolicyEngine` →
    # `InMemoryExecutionLogger` → `PermissionHook` → `DenyAllGate` → `RoleBoundaryHook`
    # → `ObservabilityHook`。
    # 分支与异常：
    #   当 `timeout_seconds <= 0` 时，抛出
    # `ValueError('timeout_seconds must be greater than zero')`。
    #   当 `max_output_chars <= 0` 时，抛出
    # `ValueError('max_output_chars must be greater than zero')`。
    #   当 `max_output_chars > MAX_TOOL_OUTPUT_CHARS` 时，抛出 `ValueError(…)`。
    #   当 `policy_engine is not None and rule_store is not None and (…` 时，抛出
    # `ValueError(…)`。
    # 副作用与资源：
    #   更新对象字段：`self._registry`、`self._timeout_seconds`、
    # `self._max_output_chars`、`self.logger`、`self._permission_hook`、`self._hooks`、
    # `self._output_recorder`。
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = 30.0,
        max_output_chars: int = MAX_TOOL_OUTPUT_CHARS,
        approval_gate: ApprovalGate | None = None,
        logger: ToolExecutionLogger | None = None,
        hooks: Sequence[ToolHook] = (),
        policy_engine: PermissionPolicyEngine | None = None,
        rule_store: PermissionRuleStore | None = None,
        rule_factory: RuleFactory = build_safe_rule,
        output_recorder: ToolOutputRecorder | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be greater than zero")
        if max_output_chars > MAX_TOOL_OUTPUT_CHARS:
            raise ValueError(f"max_output_chars cannot exceed {MAX_TOOL_OUTPUT_CHARS}")
        if (
            policy_engine is not None
            and rule_store is not None
            and policy_engine.store is not rule_store
        ):
            raise ValueError("policy_engine and rule_store must use the same store")
        resolved_store = rule_store or (
            policy_engine.store if policy_engine is not None else None
        )
        resolved_policy = policy_engine or (
            PermissionPolicyEngine(resolved_store)
            if resolved_store is not None
            else None
        )
        self._registry = registry
        self._timeout_seconds = timeout_seconds
        self._max_output_chars = max_output_chars
        self.logger = logger or InMemoryExecutionLogger()
        self._permission_hook = PermissionHook(
            approval_gate or DenyAllGate(),
            policy=resolved_policy,
            rule_store=resolved_store,
            rule_factory=rule_factory,
        )
        self._hooks = (
            RoleBoundaryHook(allows=self._role_allows),
            self._permission_hook,
            ObservabilityHook(self.logger),
            *hooks,
        )
        self._output_recorder = output_recorder

    # 函数说明：ToolExecutor._role_allows
    # 用途：检查当前角色模式是否允许调用指定工具；非角色模式直接允许。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `bool`；按分支返回 `True`；
    # `self._registry.is_allowed_for_mode(name, mode)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._registry.is_role_mode` →
    # `self._registry.is_allowed_for_mode`。
    # 分支与异常：
    #   当 `not self._registry.is_role_mode(mode)` 时，返回 `True`。
    def _role_allows(self, mode: AgentMode, name: str) -> bool:
        if not self._registry.is_role_mode(mode):
            return True
        return self._registry.is_allowed_for_mode(name, mode)

    # 函数说明：ToolExecutor.execution_records
    # 用途：读取内存日志中的工具执行记录；其他日志实现返回空元组。
    # 返回：类型 `tuple[ToolExecutionRecord, ...]`；按分支返回 `self.logger.records`；
    # `()`。
    # 分支与异常：
    #   当 `isinstance(self.logger, InMemoryExecutionLogger)` 时，返回
    # `self.logger.records`。
    @property
    def execution_records(self) -> tuple[ToolExecutionRecord, ...]:
        if isinstance(self.logger, InMemoryExecutionLogger):
            return self.logger.records
        return ()

    # 函数说明：ToolExecutor.clear_run_rules
    # 用途：清除指定运行关联的临时权限规则，返回删除数量。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `int`；返回 `await self._permission_hook.clear_run_rules(run_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._permission_hook.clear_run_rules`。
    async def clear_run_rules(self, run_id: str) -> int:

        return await self._permission_hook.clear_run_rules(run_id)

    # 函数说明：ToolExecutor.execute
    # 用途：按工具查找、前置钩子、权限审批、限时调用和后置记录顺序执行一次结构化工具请求
    # 。
    # 参数：
    #   tool_call：单次结构化工具调用，类型 `ToolCall`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext | None`；默认 `None`。
    #   hooks：附加工具生命周期钩子，类型 `Sequence[ToolHook]`；默认 `()`。
    # 返回：类型 `ToolResult`；返回
    # `await self._complete(execution_context, result, hook_runner)`。
    # 设计约束：前置钩子先获得规范化上下文；工具不存在或授权失败时也通过后置钩子完成记录
    # 。
    # 设计约束：审批等待时长与真正工具执行时长分别统计；调用者得到统一 ToolResult，而非
    # 原始异常。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`perf_counter` → `_now_iso` →
    # `self._lookup_tool` → `_safe_arguments` → `ToolExecutionContext` → `replace`；另有
    #  7 个调用点。
    # 分支与异常：
    #   `tool is None` 分支在完成前置处理后返回
    # `await self._complete(execution_context, result, hook_runner)`。
    #   `denied_reason is not None` 分支在完成前置处理后返回
    # `await self._complete(execution_context, result, hook_runner)`。
    async def execute(
        self,
        tool_call: ToolCall,
        *,
        context: ToolExecutionContext | None = None,
        hooks: Sequence[ToolHook] = (),
    ) -> ToolResult:
        """依次完成工具查找、权限检查、实际调用和结果记录，统一返回 ToolResult。"""
        started_at = perf_counter()
        started_iso = _now_iso()

        tool = self._lookup_tool(tool_call)
        arguments = _safe_arguments(tool_call.arguments)
        base_context = context or ToolExecutionContext(tool_call=tool_call)
        execution_context = replace(
            base_context,
            tool_call=tool_call,
            tool_definition=tool.definition if tool is not None else None,
            arguments=arguments,
            metadata={**base_context.metadata, "started_at": started_iso},
        )
        hook_runner = ToolHookRunner(*self._hooks, *hooks)
        permission_check = await hook_runner.before_execute(execution_context)

        if tool is None:
            result = self._failure(
                tool_call,
                f"Tool not found: {tool_call.name}",
                started_at,
            )
            return await self._complete(execution_context, result, hook_runner)

        authorization_started_at = perf_counter()
        denied_reason = await self._authorize(
            execution_context,
            hook_runner,
            permission_check,
        )
        approval_wait_ms = (
            _duration_ms(authorization_started_at)
            if permission_check is not None
            and permission_check.approval_request is not None
            and permission_check.matched_rule is None
            else 0.0
        )
        if denied_reason is not None:
            result = self._failure(tool_call, denied_reason, started_at)
            return await self._complete(execution_context, result, hook_runner)

        execution_started_at = perf_counter()
        result = await self._dispatch(
            tool,
            tool_call,
            execution_context,
            started_at,
        )
        result = result.model_copy(update={
            "approval_wait_ms": approval_wait_ms,
            "execution_duration_ms": _duration_ms(execution_started_at),
        })
        return await self._complete(execution_context, result, hook_runner)

    # 函数说明：ToolExecutor._lookup_tool
    # 用途：查找工具，供工具注册、执行与权限钩子使用。
    # 参数：
    #   tool_call：单次结构化工具调用，类型 `ToolCall`。
    # 返回：类型 `BaseTool | None`；按分支返回 `self._registry.get(tool_call.name)`；
    # `None`。
    # 分支与异常：
    #   捕获 `KeyError` 后，返回 `None`。
    def _lookup_tool(self, tool_call: ToolCall) -> BaseTool | None:
        try:
            return self._registry.get(tool_call.name)
        except KeyError:
            return None

    # 函数说明：ToolExecutor._authorize
    # 用途：复用已匹配权限规则或等待人工审批，通知审批生命周期钩子，并返回拒绝原因或
    # None。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   hook_runner：工具生命周期钩子执行器，类型 `ToolHookRunner`。
    #   check：前置工具钩子返回的权限决策，类型 `ToolHookDecision | None`。
    # 返回：类型 `str | None`；按分支返回 `None`；`check.denied_reason`；
    # `self._permission_hook.denied_reason(context, outcome.response.decision)`；
    # `f'Permission check failed: {type(exc).__name__}: {exc}'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `hook_runner.on_approval_completed` → `hook_runner.on_approval_required` →
    # `self._permission_hook.request_approval` → `self._permission_hook.denied_reason`。
    # 分支与异常：
    #   当 `check is None` 时，返回 `None`。
    #   当 `check.denied_reason is not None` 时，返回 `check.denied_reason`。
    #   当 `check.approval_request is None` 时，返回 `None`。
    #   `check.matched_rule is not None` 分支在完成前置处理后返回 `None`。
    #   捕获 `Exception` 后，返回
    # `f'Permission check failed: {type(exc).__name__}: {exc}'`。
    async def _authorize(
        self,
        context: ToolExecutionContext,
        hook_runner: ToolHookRunner,
        check: ToolHookDecision | None,
    ) -> str | None:
        """处理规则匹配与人工审批；返回拒绝原因或允许继续。"""
        try:
            if check is None:
                return None
            if check.denied_reason is not None:
                return check.denied_reason
            if check.approval_request is None:
                return None

            request = check.approval_request

            if check.matched_rule is not None:
                await hook_runner.on_approval_completed(
                    context,
                    request,
                    ApprovalDecision.APPROVED,
                    rule=check.matched_rule,
                )
                return None

            await hook_runner.on_approval_required(context, request)
            outcome = await self._permission_hook.request_approval(
                request,
                context=context,
            )
            await hook_runner.on_approval_completed(
                context,
                request,
                outcome.response.decision,
                rule=outcome.rule,
            )
            return self._permission_hook.denied_reason(
                context,
                outcome.response.decision,
            )
        except Exception as exc:
            return f"Permission check failed: {type(exc).__name__}: {exc}"

    def _timeout_for(self, tool: BaseTool, arguments: dict[str, Any]) -> float:
        """外层时限取统一时限与工具自报时限中较大者，不让外层抢先取消工具。"""
        try:
            requested = tool.execution_timeout(arguments)
        except Exception:
            requested = None
        if requested is None or requested <= 0:
            return self._timeout_seconds
        return max(self._timeout_seconds, float(requested))

    # 函数说明：ToolExecutor._dispatch
    # 用途：解析工具参数并限时执行；把参数错误、超时和工具异常转换为失败结果，保存原始证
    # 据后截断展示输出。
    # 参数：
    #   tool：目标工具实例，类型 `BaseTool`。
    #   tool_call：单次结构化工具调用，类型 `ToolCall`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   started_at：开始时间，用于计算时长，类型 `float`。
    # 返回：类型 `ToolResult`；按分支返回
    # `self._failure(tool_call, f'Invalid arguments: {exc}', started_at)`；
    # `self._failure(…)`；`ToolResult(…)`。
    # 设计约束：证据记录使用未截断的序列化输出；展示截断不影响已保存证据，证据失败单独写
    # 入 evidence_error。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_parse_arguments` →
    # `self._failure` → `asyncio.timeout` → `tool.execute_with_context` →
    # `_serialize_output` → `self._output_recorder.record`；另有 3 个调用点。
    # 资源/并发边界：`asyncio.timeout(self._timeout_seconds)`，上下文退出时执行相应清理
    # 。
    # 分支与异常：
    #   捕获 `(TypeError, ValueError)` 后，返回
    # `self._failure(tool_call, f'Invalid arguments: {exc}', started_at)`。
    #   捕获 `TimeoutError` 后，返回 `self._failure(…)`。
    #   捕获 `(KeyError, TypeError, ValueError)` 后，返回
    # `self._failure(tool_call, f'Invalid arguments: {exc}', started_at)`。
    #   捕获 `Exception` 后，返回 `self._failure(…)`。
    #   捕获 `Exception` 后，执行异常处理调用 `type`。
    async def _dispatch(
        self,
        tool: BaseTool,
        tool_call: ToolCall,
        context: ToolExecutionContext,
        started_at: float,
    ) -> ToolResult:
        """调用工具并将超时或异常转换为标准工具结果。"""
        try:
            arguments = _parse_arguments(tool_call.arguments)
        except (TypeError, ValueError) as exc:
            return self._failure(
                tool_call,
                f"Invalid arguments: {exc}",
                started_at,
            )

        timeout_seconds = self._timeout_for(tool, arguments)
        try:
            async with asyncio.timeout(timeout_seconds):
                output = await tool.execute_with_context(arguments, context)
        except TimeoutError:
            return self._failure(
                tool_call,
                f"Tool timed out after {timeout_seconds:g} seconds.",
                started_at,
                execution_outcome="unknown",
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._failure(
                tool_call,
                f"Invalid arguments: {exc}",
                started_at,
                execution_outcome="unknown",
            )
        except Exception as exc:
            return self._failure(
                tool_call,
                f"Tool execution failed: {type(exc).__name__}: {exc}",
                started_at,
                execution_outcome="unknown",
            )

        serialized_output = _serialize_output(output)
        evidence_id: str | None = None
        output_sha256: str | None = None
        evidence_error: str | None = None
        if self._output_recorder is not None:
            try:
                recorded = await self._output_recorder.record(
                    context,
                    serialized_output,
                )
            except Exception as exc:
                evidence_error = f"{type(exc).__name__}: {exc}"
            else:
                if recorded is not None:
                    evidence_id = recorded.id
                    output_sha256 = recorded.sha256

        output_truncated = len(serialized_output) > self._max_output_chars
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            success=True,
            output=_truncate(serialized_output, self._max_output_chars),
            error=None,
            duration_ms=_duration_ms(started_at),
            evidence_id=evidence_id,
            output_chars=(
                len(serialized_output)
                if evidence_id is not None
                or output_truncated
                or evidence_error is not None
                else None
            ),
            output_sha256=output_sha256,
            output_truncated=True if output_truncated else None,
            evidence_error=evidence_error,
        )

    # 函数说明：ToolExecutor._failure
    # 用途：返回 `ToolResult(…)`，提供 ToolExecutor 的派生值。
    # 参数：
    #   tool_call：单次结构化工具调用，类型 `ToolCall`。
    #   error：异常或错误信息，类型 `str`。
    #   started_at：开始时间，用于计算时长，类型 `float`。
    # 返回：类型 `ToolResult`；返回 `ToolResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolResult` → `_duration_ms`。
    def _failure(
        self,
        tool_call: ToolCall,
        error: str,
        started_at: float,
        *,
        execution_outcome: Literal["not_started", "unknown"] = "not_started",
    ) -> ToolResult:
        """失败结果携带执行阶段；调用已开始时不承诺副作用未发生。"""
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            success=False,
            output=None,
            error=error,
            duration_ms=_duration_ms(started_at),
            execution_outcome=execution_outcome,
            retry_advice=(
                "Execution started, but its final effects are unknown. "
                "Do not blindly repeat this operation. Verify its state first; "
                "retry only after confirming it is safe or idempotent."
                if execution_outcome == "unknown"
                else "The tool body was not invoked. Fix the reported error "
                "or obtain authorization before trying again."
            ),
        )

    # 函数说明：ToolExecutor._complete
    # 用途：运行执行后钩子并记录工具调用结果。
    # 参数：
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    #   result：上一步计算或执行得到的结果，类型 `ToolResult`。
    #   hook_runner：工具生命周期钩子执行器，类型 `ToolHookRunner`。
    # 返回：类型 `ToolResult`；返回 `result`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`hook_runner.after_execute`。
    async def _complete(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
        hook_runner: ToolHookRunner,
    ) -> ToolResult:

        """运行执行后钩子并记录工具调用结果。"""
        await hook_runner.after_execute(context, result)
        return result


# 函数说明：_parse_arguments
# 用途：保留已有参数字典，或解码 JSON 文本并校验结果必须是对象。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any] | str`。
# 返回：类型 `dict[str, Any]`；按分支返回 `arguments`；`parsed`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads`。
# 分支与异常：
#   当 `isinstance(arguments, dict)` 时，返回 `arguments`。
#   捕获 `json.JSONDecodeError` 后，转换或抛出
# `ValueError(f'arguments are not valid JSON: {exc.msg}')`。
#   当 `not isinstance(parsed, dict)` 时，抛出
# `TypeError('arguments must be a JSON object')`。
def _parse_arguments(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments are not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise TypeError("arguments must be a JSON object")
    return parsed


# 函数说明：_safe_arguments
# 用途：尝试将工具参数解析为字典；参数类型或 JSON 格式错误时返回空字典供前置钩子使用。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any] | str`。
# 返回：类型 `dict[str, Any]`；按分支返回 `_parse_arguments(arguments)`；`{}`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_parse_arguments`。
# 分支与异常：
#   捕获 `(TypeError, ValueError)` 后，返回 `{}`。
def _safe_arguments(arguments: dict[str, Any] | str) -> dict[str, Any]:
    try:
        return _parse_arguments(arguments)
    except (TypeError, ValueError):
        return {}


# 函数说明：_serialize_output
# 用途：序列化输出，供工具注册、执行与权限钩子使用。
# 参数：
#   output：工具、模型或转换步骤的输出，类型 `Any`。
# 返回：类型 `str`；按分支返回 `output`；
# `json.dumps(output, ensure_ascii=False, separators=(',', ':'))`；`str(output)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
# 分支与异常：
#   当 `isinstance(output, str)` 时，返回 `output`。
#   捕获 `(TypeError, ValueError)` 后，返回 `str(output)`。
def _serialize_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(output)


# 函数说明：_truncate
# 用途：截断工具注册、执行与权限钩子，供工具注册、执行与权限钩子使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
#   limit：本次返回或处理的数量上限，类型 `int`。
# 返回：类型 `str`；返回 `value[:limit]`。
def _truncate(value: str, limit: int) -> str:
    return value[:limit]


# 函数说明：_duration_ms
# 用途：返回 `max(0.0, (perf_counter() - started_at) * 1000)`，提供 工具注册、执行与权限
# 钩子 的派生值。
# 参数：
#   started_at：开始时间，用于计算时长，类型 `float`。
# 返回：类型 `float`；返回 `max(0.0, (perf_counter() - started_at) * 1000)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`perf_counter`。
def _duration_ms(started_at: float) -> float:
    return max(0.0, (perf_counter() - started_at) * 1000)
