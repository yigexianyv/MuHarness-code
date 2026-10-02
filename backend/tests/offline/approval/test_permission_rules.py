
from __future__ import annotations

from typing import Any

import pytest

from app.models.types import ToolCall, ToolDefinition, ToolPermission
from app.tools import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalScope,
    BaseTool,
    InMemoryPermissionRuleStore,
    PermissionEffect,
    PermissionPolicyEngine,
    PermissionRule,
    SQLitePermissionRuleStore,
    ToolExecutor,
    ToolRegistry,
    build_matcher,
    build_safe_rule,
    describe_safe_rule,
)
from app.tools.hooks import ToolExecutionContext, ToolHook
from app.tools.permissions.matchers import ExactArgumentsMatcher


class ScriptedGate(ApprovalGate):

    # 函数说明：ScriptedGate.__init__
    # 用途：初始化 ScriptedGate；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   *responses：额外位置参数，按实现向内部调用传递。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.calls`。
    def __init__(self, *responses: ApprovalResponse) -> None:
        self.responses = list(responses)
        self.calls = 0

    # 函数说明：ScriptedGate.request_approval
    # 用途：在回归测试与测试辅助中处理 `request_approval`，通过 `self.responses.pop` 完
    # 成首个内部处理步骤。
    # 参数：
    #   request：待处理的请求对象，类型 `ApprovalRequest`。
    # 返回：类型 `ApprovalResponse`；按分支返回
    # `ApprovalResponse(decision=ApprovalDecision.DENIED)`；`self.responses.pop(0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalResponse` →
    # `self.responses.pop`。
    # 分支与异常：
    #   当 `not self.responses` 时，返回
    # `ApprovalResponse(decision=ApprovalDecision.DENIED)`。
    # 副作用与资源：
    #   更新对象字段：`self.calls`。
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        self.calls += 1
        if not self.responses:
            return ApprovalResponse(decision=ApprovalDecision.DENIED)
        return self.responses.pop(0)


class ShellStub(BaseTool):

    # 函数说明：ShellStub.definition
    # 用途：提供 ShellStub 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_shell_command",
            description="run a shell command",
            permission=ToolPermission.HUMAN_APPROVAL,
        )

    # 函数说明：ShellStub.__init__
    # 用途：初始化 ShellStub；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    def __init__(self) -> None:
        self.executions = 0

    # 函数说明：ShellStub.execute
    # 用途：执行ShellStub，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `command`。
    # 返回：类型 `str`；返回 `f"ran:{arguments.get('command')}"`。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        self.executions += 1
        return f"ran:{arguments.get('command')}"


class RecordingApprovalHook(ToolHook):
    # 函数说明：RecordingApprovalHook.__init__
    # 用途：初始化 RecordingApprovalHook；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.completed`。
    def __init__(self) -> None:
        self.completed: list[tuple[Any, Any]] = []

    # 函数说明：RecordingApprovalHook.on_approval_completed
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
        self.completed.append((decision, rule))



# 函数说明：test_exact_arguments_matcher
# 用途：回归验证回归测试与测试辅助中的 `exact_arguments_matcher` 场景，下方断言说明列出
# 实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ExactArgumentsMatcher` →
# `matcher.matches`。
# 分支与异常：
#   验证条件：`matcher.matches({'command': 'pytest tests'})`。
#   验证条件：`not matcher.matches({'command': 'pytest other'})`。
#   验证条件：`not matcher.matches({})`。
def test_exact_arguments_matcher() -> None:
    matcher = ExactArgumentsMatcher({"command": "pytest tests"})

    assert matcher.matches({"command": "pytest tests"})
    assert not matcher.matches({"command": "pytest other"})
    assert not matcher.matches({})


# 函数说明：test_build_matcher_dispatches_by_type
# 用途：回归验证回归测试与测试辅助中的 `build_matcher_dispatches_by_type` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_matcher` → `pytest.raises`。
# 分支与异常：
#   验证条件：`isinstance(build_matcher('exact_arguments', {'arguments': {}}),
# ExactArgumentsMatcher)`。
#   预期异常：`pytest.raises(ValueError, match='Unknown matcher')`。
def test_build_matcher_dispatches_by_type() -> None:
    assert isinstance(
        build_matcher("exact_arguments", {"arguments": {}}),
        ExactArgumentsMatcher,
    )
    with pytest.raises(ValueError, match="Unknown matcher"):
        build_matcher("command_prefix", {"prefix": "pytest"})
    with pytest.raises(ValueError, match="Unknown matcher"):
        build_matcher("host_exact", {"host": "example.com"})
    with pytest.raises(ValueError, match="Unknown matcher"):
        build_matcher("nope", {})



# 函数说明：test_rule_factory_run_scope_uses_exact_arguments
# 用途：回归验证回归测试与测试辅助中的 `rule_factory_run_scope_uses_exact_arguments` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_safe_rule`。
# 分支与异常：
#   验证条件：`rule.scope is ApprovalScope.RUN`。
#   验证条件：`rule.scope_id == 'run-1'`。
#   验证条件：`rule.matcher_type == 'exact_arguments'`。
#   验证条件：`rule.matcher == {'arguments': {'command': 'pytest tests'}}`。
def test_rule_factory_run_scope_uses_exact_arguments() -> None:
    rule = build_safe_rule(
        tool_name="run_shell_command",
        arguments={"command": "pytest tests"},
        scope=ApprovalScope.RUN,
        scope_id="run-1",
    )

    assert rule.scope is ApprovalScope.RUN
    assert rule.scope_id == "run-1"
    assert rule.matcher_type == "exact_arguments"
    assert rule.matcher == {"arguments": {"command": "pytest tests"}}
    assert rule.effect is PermissionEffect.ALLOW


# 函数说明：test_rule_factory_conversation_shell_uses_exact_arguments
# 用途：回归验证回归测试与测试辅助中的
# `rule_factory_conversation_shell_uses_exact_arguments` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_safe_rule`。
# 分支与异常：
#   验证条件：`rule.matcher_type == 'exact_arguments'`。
#   验证条件：
# `rule.matcher == {'arguments': {'command': 'pytest tests/test_runtime.py'}}`。
def test_rule_factory_conversation_shell_uses_exact_arguments() -> None:
    rule = build_safe_rule(
        tool_name="run_shell_command",
        arguments={"command": "pytest tests/test_runtime.py"},
        scope=ApprovalScope.CONVERSATION,
        scope_id="conv-1",
    )

    assert rule.matcher_type == "exact_arguments"
    assert rule.matcher == {"arguments": {"command": "pytest tests/test_runtime.py"}}


# 函数说明：test_rule_factory_conversation_http_uses_exact_arguments
# 用途：回归验证回归测试与测试辅助中的
# `rule_factory_conversation_http_uses_exact_arguments` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_safe_rule`。
# 分支与异常：
#   验证条件：`rule.matcher_type == 'exact_arguments'`。
#   验证条件：`rule.matcher == {'arguments': {'url': 'https://example.com/a'}}`。
def test_rule_factory_conversation_http_uses_exact_arguments() -> None:
    rule = build_safe_rule(
        tool_name="http_request",
        arguments={"url": "https://example.com/a"},
        scope=ApprovalScope.CONVERSATION,
        scope_id="conv-1",
    )

    assert rule.matcher_type == "exact_arguments"
    assert rule.matcher == {
        "arguments": {"url": "https://example.com/a"}
    }


# 函数说明：test_describe_safe_rule
# 用途：回归验证回归测试与测试辅助中的 `describe_safe_rule` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ApprovalRequest` →
# `describe_safe_rule`。
# 分支与异常：
#   验证条件：`describe_safe_rule(shell) == expected`。
#   验证条件：`describe_safe_rule(http) == expected`。
#   验证条件：`describe_safe_rule(other) == expected`。
def test_describe_safe_rule() -> None:
    shell = ApprovalRequest(
        tool_call_id="1",
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
    )
    http = ApprovalRequest(
        tool_call_id="2",
        tool_name="http_request",
        arguments={"url": "https://a.com/"},
    )
    other = ApprovalRequest(tool_call_id="3", tool_name="other", arguments={})

    expected = "当前会话记住该操作（仅完整参数相同时自动通过）"
    assert describe_safe_rule(shell) == expected
    assert describe_safe_rule(http) == expected
    assert describe_safe_rule(other) == expected


# 函数说明：test_once_scope_cannot_be_persisted_as_rule
# 用途：回归验证回归测试与测试辅助中的 `once_scope_cannot_be_persisted_as_rule` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `PermissionRule`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='ONCE')`。
def test_once_scope_cannot_be_persisted_as_rule() -> None:
    with pytest.raises(ValueError, match="ONCE"):
        PermissionRule(
            id="rule-once",
            tool_name="run_shell_command",
            scope=ApprovalScope.ONCE,
            scope_id="run-1",
            matcher_type="exact_arguments",
            matcher={"arguments": {"command": "pytest x"}},
            description="非法临时规则",
        )



# 函数说明：test_policy_engine_allows_matching_rule_in_scope
# 用途：回归验证回归测试与测试辅助中的 `policy_engine_allows_matching_rule_in_scope` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`InMemoryPermissionRuleStore` →
# `build_safe_rule` → `store.add` → `PermissionPolicyEngine` → `engine.evaluate`。
# 分支与异常：
#   验证条件：`verdict.effect is PermissionEffect.ALLOW`。
#   验证条件：`verdict.rule_id == rule.id`。
@pytest.mark.asyncio
async def test_policy_engine_allows_matching_rule_in_scope() -> None:
    store = InMemoryPermissionRuleStore()
    rule = build_safe_rule(
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
        scope=ApprovalScope.CONVERSATION,
        scope_id="conv-1",
    )
    await store.add(rule)
    engine = PermissionPolicyEngine(store)

    verdict = await engine.evaluate(
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
        scope_ids=("conv-1",),
    )

    assert verdict.effect is PermissionEffect.ALLOW
    assert verdict.rule_id == rule.id


# 函数说明：test_policy_engine_asks_when_no_rule_matches
# 用途：回归验证回归测试与测试辅助中的 `policy_engine_asks_when_no_rule_matches` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`PermissionPolicyEngine` →
# `InMemoryPermissionRuleStore` → `engine.evaluate`。
# 分支与异常：
#   验证条件：`verdict.effect is PermissionEffect.ASK`。
#   验证条件：`verdict.rule_id is None`。
@pytest.mark.asyncio
async def test_policy_engine_asks_when_no_rule_matches() -> None:
    engine = PermissionPolicyEngine(InMemoryPermissionRuleStore())

    verdict = await engine.evaluate(
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
        scope_ids=("run-1",),
    )

    assert verdict.effect is PermissionEffect.ASK
    assert verdict.rule_id is None


# 函数说明：test_policy_engine_ignores_rules_outside_scope
# 用途：回归验证回归测试与测试辅助中的 `policy_engine_ignores_rules_outside_scope` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`InMemoryPermissionRuleStore` →
# `store.add` → `build_safe_rule` → `PermissionPolicyEngine` → `engine.evaluate`。
# 分支与异常：
#   验证条件：`verdict.effect is PermissionEffect.ASK`。
@pytest.mark.asyncio
async def test_policy_engine_ignores_rules_outside_scope() -> None:
    store = InMemoryPermissionRuleStore()
    await store.add(
        build_safe_rule(
            tool_name="run_shell_command",
            arguments={"command": "pytest x"},
            scope=ApprovalScope.CONVERSATION,
            scope_id="conv-1",
        )
    )
    engine = PermissionPolicyEngine(store)

    verdict = await engine.evaluate(
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
        scope_ids=("other-conv",),
    )

    assert verdict.effect is PermissionEffect.ASK


# 函数说明：test_policy_engine_deny_rule_wins_over_allow_rule
# 用途：回归验证回归测试与测试辅助中的 `policy_engine_deny_rule_wins_over_allow_rule` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`InMemoryPermissionRuleStore` →
# `store.add` → `build_safe_rule` → `PermissionPolicyEngine(store).evaluate` →
# `PermissionPolicyEngine`。
# 分支与异常：
#   验证条件：`verdict.effect is PermissionEffect.DENY`。
#   验证条件：`verdict.rule_id == deny_rule.id`。
@pytest.mark.asyncio
async def test_policy_engine_deny_rule_wins_over_allow_rule() -> None:
    store = InMemoryPermissionRuleStore()
    arguments = {"command": "pytest x"}
    await store.add(
        build_safe_rule(
            tool_name="run_shell_command",
            arguments=arguments,
            scope=ApprovalScope.CONVERSATION,
            scope_id="conv-1",
            effect=PermissionEffect.ALLOW,
        )
    )
    deny_rule = build_safe_rule(
        tool_name="run_shell_command",
        arguments=arguments,
        scope=ApprovalScope.CONVERSATION,
        scope_id="conv-1",
        effect=PermissionEffect.DENY,
    )
    await store.add(deny_rule)

    verdict = await PermissionPolicyEngine(store).evaluate(
        tool_name="run_shell_command",
        arguments=arguments,
        scope_ids=("conv-1",),
    )

    assert verdict.effect is PermissionEffect.DENY
    assert verdict.rule_id == deny_rule.id


# 函数说明：test_http_conversation_rule_does_not_expand_method_or_path
# 用途：回归验证回归测试与测试辅助中的
# `http_conversation_rule_does_not_expand_method_or_path` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`InMemoryPermissionRuleStore` →
# `store.add` → `build_safe_rule` → `PermissionPolicyEngine` → `engine.evaluate`。
# 分支与异常：
#   验证条件：`same.effect is PermissionEffect.ALLOW`。
#   验证条件：`changed.effect is PermissionEffect.ASK`。
@pytest.mark.asyncio
async def test_http_conversation_rule_does_not_expand_method_or_path() -> None:
    store = InMemoryPermissionRuleStore()
    await store.add(
        build_safe_rule(
            tool_name="http_request",
            arguments={
                "url": "https://api.example.com/status",
                "method": "GET",
            },
            scope=ApprovalScope.CONVERSATION,
            scope_id="conv-1",
        )
    )
    engine = PermissionPolicyEngine(store)

    same = await engine.evaluate(
        tool_name="http_request",
        arguments={
            "url": "https://api.example.com/status",
            "method": "GET",
        },
        scope_ids=("conv-1",),
    )
    changed = await engine.evaluate(
        tool_name="http_request",
        arguments={
            "url": "https://api.example.com/delete",
            "method": "POST",
        },
        scope_ids=("conv-1",),
    )

    assert same.effect is PermissionEffect.ALLOW
    assert changed.effect is PermissionEffect.ASK



# 函数说明：_shell_context
# 用途：返回 `ToolExecutionContext(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   call_id：工具调用标识，类型 `str`。
#   run_id：目标运行标识，类型 `str`。
#   conversation_id：目标会话标识，类型 `str`。
# 返回：类型 `ToolExecutionContext`；返回 `ToolExecutionContext(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolExecutionContext` → `ToolCall`。
def _shell_context(
    call_id: str,
    run_id: str,
    conversation_id: str,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_call=ToolCall(
            id=call_id,
            name="run_shell_command",
            arguments={"command": "pytest x"},
        ),
        run_id=run_id,
        conversation_id=conversation_id,
    )


# 函数说明：test_run_scope_rule_remembers_exact_operation
# 用途：回归验证回归测试与测试辅助中的 `run_scope_rule_remembers_exact_operation` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ShellStub` → `ToolRegistry` →
# `registry.register` → `InMemoryPermissionRuleStore` → `PermissionPolicyEngine` →
# `ScriptedGate`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`first.success is True`。
#   验证条件：`gate.calls == 1`。
#   验证条件：`second.success is True`。
@pytest.mark.asyncio
async def test_run_scope_rule_remembers_exact_operation() -> None:
    tool = ShellStub()
    registry = ToolRegistry()
    registry.register(tool)
    store = InMemoryPermissionRuleStore()
    engine = PermissionPolicyEngine(store)
    gate = ScriptedGate(
        ApprovalResponse(
            decision=ApprovalDecision.APPROVED,
            scope=ApprovalScope.RUN,
        )
    )
    executor = ToolExecutor(
        registry,
        approval_gate=gate,
        policy_engine=engine,
        rule_store=store,
    )
    context = _shell_context("c1", "run-1", "conv-1")

    first = await executor.execute(
        ToolCall(id="c1", name="run_shell_command", arguments={"command": "pytest x"}),
        context=context,
    )
    assert first.success is True
    assert gate.calls == 1

    second = await executor.execute(
        ToolCall(id="c2", name="run_shell_command", arguments={"command": "pytest x"}),
        context=context,
    )
    assert second.success is True
    assert gate.calls == 1
    assert tool.executions == 2

    third = await executor.execute(
        ToolCall(id="c3", name="run_shell_command", arguments={"command": "pytest y"}),
        context=context,
    )
    assert third.success is False
    assert gate.calls == 2


# 函数说明：test_conversation_scope_only_allows_same_operation
# 用途：回归验证回归测试与测试辅助中的 `conversation_scope_only_allows_same_operation`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ShellStub` → `ToolRegistry` →
# `registry.register` → `InMemoryPermissionRuleStore` → `PermissionPolicyEngine` →
# `ScriptedGate`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`first.success is True`。
#   验证条件：`gate.calls == 1`。
#   验证条件：`second.success is True`。
@pytest.mark.asyncio
async def test_conversation_scope_only_allows_same_operation() -> None:
    tool = ShellStub()
    registry = ToolRegistry()
    registry.register(tool)
    store = InMemoryPermissionRuleStore()
    engine = PermissionPolicyEngine(store)
    gate = ScriptedGate(
        ApprovalResponse(
            decision=ApprovalDecision.APPROVED,
            scope=ApprovalScope.CONVERSATION,
        )
    )
    executor = ToolExecutor(
        registry,
        approval_gate=gate,
        policy_engine=engine,
        rule_store=store,
    )
    context = _shell_context("c1", "run-1", "conv-1")

    first = await executor.execute(
        ToolCall(id="c1", name="run_shell_command", arguments={"command": "pytest x"}),
        context=context,
    )
    assert first.success is True
    assert gate.calls == 1

    second = await executor.execute(
        ToolCall(
            id="c2",
            name="run_shell_command",
            arguments={"command": "pytest x"},
        ),
        context=_shell_context("c2", "run-2", "conv-1"),
    )
    assert second.success is True
    assert gate.calls == 1

    third = await executor.execute(
        ToolCall(
            id="c3",
            name="run_shell_command",
            arguments={"command": "pytest x; rm -rf /tmp/x"},
        ),
        context=_shell_context("c3", "run-3", "conv-1"),
    )
    assert third.success is False
    assert gate.calls == 2


# 函数说明：test_once_scope_does_not_create_rule
# 用途：回归验证回归测试与测试辅助中的 `once_scope_does_not_create_rule` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ShellStub` → `ToolRegistry` →
# `registry.register` → `InMemoryPermissionRuleStore` → `PermissionPolicyEngine` →
# `ScriptedGate`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`first.success is True`。
#   验证条件：`await store.list() == ()`。
#   验证条件：`second.success is False`。
#   验证条件：`gate.calls == 2`。
@pytest.mark.asyncio
async def test_once_scope_does_not_create_rule() -> None:
    tool = ShellStub()
    registry = ToolRegistry()
    registry.register(tool)
    store = InMemoryPermissionRuleStore()
    engine = PermissionPolicyEngine(store)
    gate = ScriptedGate(ApprovalResponse(decision=ApprovalDecision.APPROVED))
    executor = ToolExecutor(
        registry,
        approval_gate=gate,
        policy_engine=engine,
        rule_store=store,
    )
    context = _shell_context("c1", "run-1", "conv-1")

    first = await executor.execute(
        ToolCall(id="c1", name="run_shell_command", arguments={"command": "pytest x"}),
        context=context,
    )
    assert first.success is True
    assert await store.list() == ()  

    second = await executor.execute(
        ToolCall(id="c2", name="run_shell_command", arguments={"command": "pytest x"}),
        context=context,
    )
    assert second.success is False
    assert gate.calls == 2


# 函数说明：test_approval_completed_event_carries_rule_id
# 用途：回归验证回归测试与测试辅助中的 `approval_completed_event_carries_rule_id` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ShellStub` → `ToolRegistry` →
# `registry.register` → `InMemoryPermissionRuleStore` → `PermissionPolicyEngine` →
# `ScriptedGate`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`len(hook.completed) == 1`。
#   验证条件：`decision is ApprovalDecision.APPROVED`。
#   验证条件：`rule is not None and rule.scope is ApprovalScope.RUN`。
#   验证条件：`len(hook.completed) == 2`。
@pytest.mark.asyncio
async def test_approval_completed_event_carries_rule_id() -> None:
    tool = ShellStub()
    registry = ToolRegistry()
    registry.register(tool)
    store = InMemoryPermissionRuleStore()
    engine = PermissionPolicyEngine(store)
    gate = ScriptedGate(
        ApprovalResponse(
            decision=ApprovalDecision.APPROVED,
            scope=ApprovalScope.RUN,
        )
    )
    executor = ToolExecutor(
        registry,
        approval_gate=gate,
        policy_engine=engine,
        rule_store=store,
    )
    hook = RecordingApprovalHook()
    context = _shell_context("c1", "run-1", "conv-1")

    await executor.execute(
        ToolCall(id="c1", name="run_shell_command", arguments={"command": "pytest x"}),
        context=context,
        hooks=(hook,),
    )
    assert len(hook.completed) == 1
    decision, rule = hook.completed[0]
    assert decision is ApprovalDecision.APPROVED
    assert rule is not None and rule.scope is ApprovalScope.RUN

    await executor.execute(
        ToolCall(id="c2", name="run_shell_command", arguments={"command": "pytest x"}),
        context=context,
        hooks=(hook,),
    )
    assert len(hook.completed) == 2
    assert hook.completed[1][1] is not None



# 函数说明：test_sqlite_rule_store_persists_and_queries
# 用途：回归验证回归测试与测试辅助中的 `sqlite_rule_store_persists_and_queries` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLitePermissionRuleStore` →
# `store.initialize` → `build_safe_rule` → `store.add` → `reopened.initialize` →
# `reopened.list`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`len(rules) == 1`。
#   验证条件：`rules[0].id == rule.id`。
#   验证条件：`rules[0].matcher == rule.matcher`。
#   验证条件：`rules[0].description == rule.description`。
@pytest.mark.asyncio
async def test_sqlite_rule_store_persists_and_queries(tmp_path) -> None:
    database_path = tmp_path / "muharness.db"
    store = SQLitePermissionRuleStore(database_path)
    await store.initialize()
    rule = build_safe_rule(
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
        scope=ApprovalScope.CONVERSATION,
        scope_id="conv-1",
    )
    await store.add(rule)

    reopened = SQLitePermissionRuleStore(database_path)
    await reopened.initialize()
    rules = await reopened.list(scope_ids=("conv-1",))
    assert len(rules) == 1
    assert rules[0].id == rule.id
    assert rules[0].matcher == rule.matcher
    assert rules[0].description == rule.description

    assert await reopened.remove(rule.id) is True
    assert await reopened.list(scope_ids=("conv-1",)) == ()


# 函数说明：test_sqlite_empty_scope_cannot_read_unrelated_rules
# 用途：回归验证回归测试与测试辅助中的 `sqlite_empty_scope_cannot_read_unrelated_rules`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLitePermissionRuleStore` →
# `store.initialize` → `build_safe_rule` → `store.add` → `store.list` →
# `PermissionPolicyEngine(store).evaluate`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`await store.list(scope_ids=()) == ()`。
#   验证条件：`verdict.effect is PermissionEffect.ASK`。
@pytest.mark.asyncio
async def test_sqlite_empty_scope_cannot_read_unrelated_rules(tmp_path) -> None:
    store = SQLitePermissionRuleStore(tmp_path / "muharness.db")
    await store.initialize()
    rule = build_safe_rule(
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
        scope=ApprovalScope.CONVERSATION,
        scope_id="unrelated-conversation",
    )
    await store.add(rule)

    assert await store.list(scope_ids=()) == ()
    verdict = await PermissionPolicyEngine(store).evaluate(
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
        scope_ids=(),
    )
    assert verdict.effect is PermissionEffect.ASK


# 函数说明：test_rule_store_get_and_remove_scope
# 用途：回归验证回归测试与测试辅助中的 `rule_store_get_and_remove_scope` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLitePermissionRuleStore` →
# `store.initialize` → `build_safe_rule` → `store.add` → `store.remove_scope`。
# 分支与异常：
#   验证条件：`await store.get(first.id) == first`。
#   验证条件：`await store.remove_scope(ApprovalScope.CONVERSATION, 'conv-1') == 1`。
#   验证条件：`await store.get(first.id) is None`。
#   验证条件：`await store.get(second.id) == second`。
@pytest.mark.asyncio
async def test_rule_store_get_and_remove_scope(tmp_path) -> None:
    store = SQLitePermissionRuleStore(tmp_path / "muharness.db")
    await store.initialize()
    first = build_safe_rule(
        tool_name="run_shell_command",
        arguments={"command": "pytest x"},
        scope=ApprovalScope.CONVERSATION,
        scope_id="conv-1",
    )
    second = build_safe_rule(
        tool_name="run_shell_command",
        arguments={"command": "pytest y"},
        scope=ApprovalScope.CONVERSATION,
        scope_id="conv-2",
    )
    await store.add(first)
    await store.add(second)

    assert await store.get(first.id) == first
    assert await store.remove_scope(ApprovalScope.CONVERSATION, "conv-1") == 1
    assert await store.get(first.id) is None
    assert await store.get(second.id) == second


# 函数说明：test_sqlite_initialize_invalidates_legacy_broad_rules
# 用途：回归验证回归测试与测试辅助中的
# `sqlite_initialize_invalidates_legacy_broad_rules` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLitePermissionRuleStore` →
# `store.initialize` → `PermissionRule` → `store.add` → `reopened.initialize`。
# 分支与异常：
#   验证条件：`await store.get(legacy.id) is not None`。
#   验证条件：`await reopened.get(legacy.id) is None`。
@pytest.mark.asyncio
async def test_sqlite_initialize_invalidates_legacy_broad_rules(tmp_path) -> None:
    database_path = tmp_path / "muharness.db"
    store = SQLitePermissionRuleStore(database_path)
    await store.initialize()
    legacy = PermissionRule(
        id="legacy-prefix-rule",
        tool_name="run_shell_command",
        scope=ApprovalScope.CONVERSATION,
        scope_id="conv-1",
        matcher_type="command_prefix",
        matcher={"prefix": "pytest"},
        description="旧版宽泛命令规则",
    )
    await store.add(legacy)
    assert await store.get(legacy.id) is not None

    reopened = SQLitePermissionRuleStore(database_path)
    await reopened.initialize()

    assert await reopened.get(legacy.id) is None


# 函数说明：test_executor_rejects_mismatched_policy_and_rule_stores
# 用途：回归验证回归测试与测试辅助中的
# `executor_rejects_mismatched_policy_and_rule_stores` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`InMemoryPermissionRuleStore` →
# `pytest.raises` → `ToolExecutor` → `ToolRegistry` → `PermissionPolicyEngine`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='same store')`。
def test_executor_rejects_mismatched_policy_and_rule_stores() -> None:
    first_store = InMemoryPermissionRuleStore()
    second_store = InMemoryPermissionRuleStore()

    with pytest.raises(ValueError, match="same store"):
        ToolExecutor(
            ToolRegistry(),
            policy_engine=PermissionPolicyEngine(first_store),
            rule_store=second_store,
        )
