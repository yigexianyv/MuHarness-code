"""第 5 步的接线部件：额外工具授权、角色 Runtime 缓存、恢复记录、状态推送、子 Run 网关。"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from app.models.types import (
    AgentMode,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from app.records.trace import SQLiteTraceEventHandler, SQLiteTraceStore
from app.runtime.agent.budget import RunBudgetConfig
from app.runtime.agent.events import AgentEvent, AgentEventType
from app.runtime.checkpoint import SQLiteCheckpointStore
from app.runtime.mea import (
    ExtraToolsError,
    MeaEvents,
    MeaRunGateway,
    RecoveryInfoProvider,
    RoleLimits,
    RoleRuntimes,
    SQLiteMeaStore,
    optional_executor_tools,
    validate_extra_tools,
)
from app.runtime.mea.models import MeaRound, MeaRun, RoundPhase
from app.runtime.mea.requirements import build_requirements
from app.tools.base import BaseTool
from app.tools.executor import ToolExecutor
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry
from app.tools.role_boundary import ROLE_REJECTION_PREFIX


class NamedTool(BaseTool):
    # 函数说明：NamedTool.__init__
    # 用途：初始化 NamedTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    # 副作用与资源：
    #   更新对象字段：`self._definition`、`self.calls`。
    def __init__(self, name: str) -> None:
        self._definition = ToolDefinition(
            name=name, description=name, parameters={"type": "object", "properties": {}}
        )
        self.calls = 0

    # 函数说明：NamedTool.definition
    # 用途：提供 NamedTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `self._definition`。
    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    # 函数说明：NamedTool.execute
    # 用途：执行NamedTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `str`；返回 `'ok'`。
    # 副作用与资源：
    #   更新对象字段：`self.calls`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        self.calls += 1
        return "ok"


# 函数说明：_registry
# 用途：在回归测试与测试辅助中处理 `_registry`，通过 `registry.register` 完成首个内部处
# 理步骤。
# 参数：
#   *names：额外位置参数，按实现向内部调用传递。
# 返回：类型 `ToolRegistry`；返回 `registry`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `NamedTool`。
def _registry(*names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register(NamedTool(name))
    return registry


# 函数说明：_context
# 用途：返回 `ToolExecutionContext(tool_call=call, run_id='r', mode=mode)`，提供 回归测
# 试与测试辅助 的派生值。
# 参数：
#   call：调用输入或配置值，类型 `ToolCall`。
#   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
# 返回：类型 `ToolExecutionContext`；返回
# `ToolExecutionContext(tool_call=call, run_id='r', mode=mode)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolExecutionContext`。
def _context(call: ToolCall, mode: AgentMode) -> ToolExecutionContext:
    return ToolExecutionContext(tool_call=call, run_id="r", mode=mode)


# ---------------------------------------------------------------- 额外工具


# 函数说明：test_validate_extra_tools_accepts_only_network_and_mcp
# 用途：回归验证回归测试与测试辅助中的
# `validate_extra_tools_accepts_only_network_and_mcp` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` → `validate_extra_tools` →
#  `optional_executor_tools` → `pytest.raises`。
# 分支与异常：
#   验证条件：`validate_extra_tools(['web_search', ' web_search ', 'mcp__docs__search',
# 'read_file'], registry) == ('mcp__docs__search', '…`。
#   验证条件：`optional_executor_tools(registry) == ('http_request', 'mcp__docs__search'
# , 'web_search')`。
#   预期异常：`pytest.raises(ExtraToolsError)`。
def test_validate_extra_tools_accepts_only_network_and_mcp() -> None:
    registry = _registry("read_file", "web_search", "http_request", "memory_create",
                         "task_update", "mcp__docs__search")

    assert validate_extra_tools(["web_search", " web_search ", "mcp__docs__search", "read_file"],
                                registry) == ("mcp__docs__search", "web_search")
    assert optional_executor_tools(registry) == ("http_request", "mcp__docs__search", "web_search")
    for bad in (["memory_create"], ["task_update"], ["mcp__missing__x"], ["web_search", "nope"]):
        with pytest.raises(ExtraToolsError):
            validate_extra_tools(bad, registry)


# 函数说明：test_extra_tool_view_allows_the_tool_only_for_execute
# 用途：回归验证回归测试与测试辅助中的
# `extra_tool_view_allows_the_tool_only_for_execute` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` →
# `registry.with_role_extras` → `ToolCall` → `ToolExecutor(registry).execute` →
# `ToolExecutor` → `_context`；另有 9 个调用点。
# 分支与异常：
#   验证条件：`base_result.success is False`。
#   验证条件：`(base_result.error or '').startswith(ROLE_REJECTION_PREFIX)`。
#   验证条件：`view_result.success is True`。
#   验证条件：`registry.get('web_search').calls == 1`。
async def test_extra_tool_view_allows_the_tool_only_for_execute() -> None:
    registry = _registry("read_file", "web_search")
    view = registry.with_role_extras({AgentMode.EXECUTE: {"web_search"}})
    call = ToolCall(id="c1", name="web_search", arguments={})

    base_result = await ToolExecutor(registry).execute(call, context=_context(call, AgentMode.EXECUTE))
    assert base_result.success is False
    assert (base_result.error or "").startswith(ROLE_REJECTION_PREFIX)

    view_result = await ToolExecutor(view).execute(call, context=_context(call, AgentMode.EXECUTE))
    assert view_result.success is True
    assert registry.get("web_search").calls == 1

    audit_result = await ToolExecutor(view).execute(call, context=_context(call, AgentMode.AUDIT))
    assert (audit_result.error or "").startswith(ROLE_REJECTION_PREFIX)

    names = {d.name for d in view.model_definitions_for_mode(AgentMode.EXECUTE)}
    assert "web_search" in names
    assert "web_search" not in {d.name for d in registry.model_definitions_for_mode(AgentMode.EXECUTE)}

    # 视图和原注册表共享工具：之后注册的工具在视图里也存在，但不在白名单里仍然被拒
    registry.register(NamedTool("mcp__late__tool"))
    late = ToolCall(id="c2", name="mcp__late__tool", arguments={})
    late_result = await ToolExecutor(view).execute(late, context=_context(late, AgentMode.EXECUTE))
    assert "mcp__late__tool" in view.names()
    assert (late_result.error or "").startswith(ROLE_REJECTION_PREFIX)


# 函数说明：test_role_runtimes_cache_executor_per_tool_set
# 用途：回归验证回归测试与测试辅助中的 `role_runtimes_cache_executor_per_tool_set` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` → `RoleRuntimes` →
# `runtimes.executor_for` → `built[3][1].is_allowed_for_mode` →
# `registry.is_allowed_for_mode` → `runtimes.timeouts`。
# 分支与异常：
#   验证条件：`len(built) == 3 and set(runtimes) == {AgentMode.MANAGE, AgentMode.EXECUTE
# , AgentMode.AUDIT}`。
#   验证条件：`runtimes.executor_for(()) is runtimes[AgentMode.EXECUTE]`。
#   验证条件：`runtimes.executor_for(('web_search',)) is first`。
#   验证条件：`runtimes.executor_for(('http_request', 'web_search')) is not first`。
def test_role_runtimes_cache_executor_per_tool_set() -> None:
    registry = _registry("read_file", "web_search", "http_request")
    built: list[tuple[AgentMode, ToolRegistry]] = []
    approvals: list[tuple[AgentMode, bool]] = []

    # 函数说明：test_role_runtimes_cache_executor_per_tool_set.build
    # 用途：构建回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    #   tools：可用工具定义或工具实例集合，类型 `ToolRegistry`。
    #   limits：上限输入或配置值，类型 `RoleLimits`。
    #   auto：`auto`输入或配置值，类型 `bool`。
    # 返回：类型 `object`；返回 `object()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`object`。
    # 闭包依赖：从外层读取 `approvals`、`built`。
    def build(mode: AgentMode, tools: ToolRegistry, limits: RoleLimits, auto: bool) -> object:
        built.append((mode, tools))
        approvals.append((mode, auto))
        return object()

    runtimes = RoleRuntimes(registry, build)
    assert len(built) == 3 and set(runtimes) == {AgentMode.MANAGE, AgentMode.EXECUTE, AgentMode.AUDIT}
    assert runtimes.executor_for(()) is runtimes[AgentMode.EXECUTE]
    first = runtimes.executor_for(("web_search",))
    assert runtimes.executor_for(("web_search",)) is first
    assert runtimes.executor_for(("http_request", "web_search")) is not first
    assert len(built) == 5
    assert built[3][1].is_allowed_for_mode("web_search", AgentMode.EXECUTE)
    assert not registry.is_allowed_for_mode("web_search", AgentMode.EXECUTE)
    assert runtimes.timeouts()[AgentMode.EXECUTE] == 30 * 60
    # Auditor 的只读 shell 总是自动批准；Executor 默认请示，按长任务的选择取自动批准版本
    assert approvals[:3] == [(AgentMode.MANAGE, False), (AgentMode.EXECUTE, False), (AgentMode.AUDIT, True)]
    auto = runtimes.executor_for((), True)
    assert auto is not runtimes[AgentMode.EXECUTE]
    assert runtimes.executor_for((), True) is auto
    assert runtimes.executor_for(("web_search",), True) is not first
    assert approvals[-1] == (AgentMode.EXECUTE, True)


# 函数说明：test_role_limits_replace_call_thresholds_only
# 用途：回归验证回归测试与测试辅助中的 `role_limits_replace_call_thresholds_only` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RunBudgetConfig` →
# `RoleLimits(max_steps=40, max_tool_rounds=40, hard_model_calls=40,…` → `RoleLimits`。
# 分支与异常：
#   验证条件：`(budget.warning_model_calls, budget.finalization_model_calls, budget.
# hard_model_calls) == (30, 36, 40)`。
#   验证条件：`budget.hard_tokens == 200000`。
def test_role_limits_replace_call_thresholds_only() -> None:
    base = RunBudgetConfig(_env_file=None, hard_tokens=200_000, hard_model_calls=15)
    budget = RoleLimits(max_steps=40, max_tool_rounds=40, hard_model_calls=40,
                        finalization_model_calls=36, warning_model_calls=30).budget(base)
    assert (budget.warning_model_calls, budget.finalization_model_calls, budget.hard_model_calls) == (30, 36, 40)
    assert budget.hard_tokens == 200_000


# ---------------------------------------------------------------- 恢复记录


# 函数说明：test_recovery_info_lists_completed_and_pending_calls
# 用途：回归验证回归测试与测试辅助中的 `recovery_info_lists_completed_and_pending_calls`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteCheckpointStore` →
# `SQLiteTraceStore` → `checkpoints.initialize` → `traces.initialize` →
# `checkpoints.start` → `Message`；另有 11 个调用点。
# 分支与异常：
#   验证条件：`len(completed) == 1`。
#   验证条件：`'write_file' in completed[0] and 'log.txt' in completed[0] and ('ev-1' in
#  completed[0])`。
#   验证条件：`len(pending) == 1`。
#   验证条件：`'run_shell_command' in pending[0] and 'python import.py' in pending[0]`。
# 副作用与资源：
#   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
async def test_recovery_info_lists_completed_and_pending_calls(tmp_path) -> None:
    database = tmp_path / "m.db"
    checkpoints = SQLiteCheckpointStore(database)
    traces = SQLiteTraceStore(database)
    await checkpoints.initialize()
    await traces.initialize()
    run_id = "exec-run"
    await checkpoints.start(run_id, conversation_id="conv-1",
                            user_message=Message(role=MessageRole.USER, content="执行"))
    await checkpoints.before_model(run_id, step=1)
    append = ToolCall(id="c1", name="write_file", arguments={"path": "log.txt", "content": "line\n"})
    shell = ToolCall(id="c2", name="run_shell_command", arguments={"command": "python import.py"})
    await checkpoints.before_tools(run_id, step=1, tool_calls=(append, shell))
    await checkpoints.complete_tool(
        run_id,
        ToolResult(tool_call_id="c1", tool_name="write_file", success=True, output="ok",
                   duration_ms=1, evidence_id="ev-1"),
    )
    handler = SQLiteTraceEventHandler(traces)
    for index, call in enumerate((append, shell), start=1):
        await handler.emit(AgentEvent(run_id=run_id, conversation_id="conv-1", sequence=index,
                                      type=AgentEventType.TOOL_STARTED, tool_call=call))
    await checkpoints.recover_running()  # “进程重启”

    completed, pending = await RecoveryInfoProvider(checkpoints, traces)(run_id)

    assert len(completed) == 1
    assert "write_file" in completed[0] and "log.txt" in completed[0] and "ev-1" in completed[0]
    assert len(pending) == 1
    assert "run_shell_command" in pending[0] and "python import.py" in pending[0]
    assert await RecoveryInfoProvider(checkpoints, traces)("missing-run") == ([], [])


# ---------------------------------------------------------------- 推送与网关


# 函数说明：test_store_changes_are_broadcast_without_report_bodies
# 用途：回归验证回归测试与测试辅助中的
# `store_changes_are_broadcast_without_report_bodies` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteMeaStore` → `store.initialize`
# → `MeaEvents` → `store.add_listener` → `datetime.now` → `MeaRun`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`sent == []`。
#   验证条件：`methods == ['mea.round', 'mea.status']`。
#   验证条件：`round_payload['mea_id'] == 'm1'`。
#   验证条件：`round_payload['round']['ref'] == 'round_001'`。
async def test_store_changes_are_broadcast_without_report_bodies(tmp_path) -> None:
    store = SQLiteMeaStore(tmp_path / "m.db")
    await store.initialize()
    sent: list[tuple[str, Any]] = []

    # 函数说明：test_store_changes_are_broadcast_without_report_bodies.broadcast
    # 用途：广播回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `Any`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 闭包依赖：从外层读取 `sent`。
    async def broadcast(method: str, params: Any) -> None:
        sent.append((method, params))

    events = MeaEvents()
    store.add_listener(events.on_store_change)
    now = datetime.now(UTC)
    run = MeaRun(id="m1", task_id="task-1", conversation_id="conv-1", created_at=now, updated_at=now)
    await store.create_run(run, build_requirements(original_request="导入", goal="导入"))
    assert sent == []  # 还没有广播函数：静默

    events.set_broadcaster(broadcast)
    await store.save_round(
        MeaRound(mea_run_id="m1", index=1, phase=RoundPhase.AUDITED, auditor_report="很长的报告",
                 audit_status="complete", created_at=now, updated_at=now)
    )
    await store.save_run(run.model_copy(update={"pause_requested": True}))

    methods = [method for method, _ in sent]
    assert methods == ["mea.round", "mea.status"]
    round_payload = sent[0][1]
    assert round_payload["mea_id"] == "m1"
    assert round_payload["round"]["ref"] == "round_001"
    assert round_payload["round"]["audit_status"] == "complete"
    assert "auditor_report" not in round_payload["round"]
    assert sent[1][1]["mea"]["pause_requested"] is True


# 函数说明：test_store_listener_failure_does_not_break_writes
# 用途：回归验证回归测试与测试辅助中的 `store_listener_failure_does_not_break_writes` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteMeaStore` → `store.initialize`
# → `store.add_listener` → `datetime.now` → `MeaRun` → `store.create_run`；另有 2 个调用
# 点。
# 分支与异常：
#   验证条件：`(await store.require('m1')).id == 'm1'`。
async def test_store_listener_failure_does_not_break_writes(tmp_path) -> None:
    store = SQLiteMeaStore(tmp_path / "m.db")
    await store.initialize()

    # 函数说明：test_store_listener_failure_does_not_break_writes.broken
    # 用途：处理回归测试与测试辅助中的 `broken` 数据；结果及边界条件见下方说明。
    # 参数：
    #   run：当前运行记录。
    #   rnd：当前测试模型轮次。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def broken(run, rnd) -> None:
        raise RuntimeError("socket closed")

    store.add_listener(broken)
    now = datetime.now(UTC)
    run = MeaRun(id="m1", task_id="task-1", conversation_id="conv-1", created_at=now, updated_at=now)
    await store.create_run(run, build_requirements(original_request="导入", goal="导入"))
    assert (await store.require("m1")).id == "m1"


class RecordingRunManager:
    # 函数说明：RecordingRunManager.__init__
    # 用途：初始化 RecordingRunManager；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.started`。
    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []

    # 函数说明：RecordingRunManager.start
    # 用途：启动RecordingRunManager，供回归测试与测试辅助使用。
    # 参数：
    #   user_message：当前用户消息，类型 `str`。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：返回 `(kwargs.get('run_id', 'r'), None)`。
    async def start(self, user_message: str, **kwargs: Any):
        self.started.append({"user_message": user_message, **kwargs})
        return kwargs.get("run_id", "r"), None

    # 函数说明：RecordingRunManager.active_run_ids
    # 用途：运行`ids`，供回归测试与测试辅助使用。
    # 返回：类型 `tuple[str, ...]`；返回 `('r',)`。
    @property
    def active_run_ids(self) -> tuple[str, ...]:
        return ("r",)

    # 函数说明：RecordingRunManager.forget_results
    # 用途：清除结果集合，供回归测试与测试辅助使用。
    # 参数：
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def forget_results(self, run_ids: tuple[str, ...]) -> None:
        pass


# 函数说明：test_gateway_attaches_trace_and_mea_event_handlers
# 用途：回归验证回归测试与测试辅助中的 `gateway_attaches_trace_and_mea_event_handlers`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteTraceStore` →
# `traces.initialize` → `MeaEvents` → `events.set_broadcaster` → `RecordingRunManager` →
#  `MeaRunGateway`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`gateway.active_run_ids == ('r',)`。
#   验证条件：`sent[0][0] == 'mea.agent_event'`。
#   验证条件：`sent[0][1]['mea_id'] == 'm1' and sent[0][1]['role'] == 'executor'`。
#   验证条件：`sent[0][1]['event']['run_id'] == 'exec-1'`。
# 副作用与资源：
#   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
async def test_gateway_attaches_trace_and_mea_event_handlers(tmp_path) -> None:
    traces = SQLiteTraceStore(tmp_path / "m.db")
    await traces.initialize()
    sent: list[tuple[str, Any]] = []

    # 函数说明：test_gateway_attaches_trace_and_mea_event_handlers.broadcast
    # 用途：广播回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `Any`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 闭包依赖：从外层读取 `sent`。
    async def broadcast(method: str, params: Any) -> None:
        sent.append((method, params))

    events = MeaEvents()
    events.set_broadcaster(broadcast)
    manager = RecordingRunManager()
    gateway = MeaRunGateway(
        manager, trace_handler_factory=lambda: SQLiteTraceEventHandler(traces), events=events
    )

    await gateway.start("执行", run_id="exec-1", source="mea:executor", source_id="m1",
                        conversation_id="conv-1", mode=AgentMode.EXECUTE)
    handler = manager.started[0]["event_handler"]
    await handler.emit(AgentEvent(run_id="exec-1", conversation_id="conv-1",
                                  type=AgentEventType.AGENT_STARTED))

    assert gateway.active_run_ids == ("r",)
    assert sent[0][0] == "mea.agent_event"
    assert sent[0][1]["mea_id"] == "m1" and sent[0][1]["role"] == "executor"
    assert sent[0][1]["event"]["run_id"] == "exec-1"
    assert await traces.get("exec-1") is not None  # Trace 同时被记录


# 函数说明：test_data_paths_inside_workspace_are_excluded_from_snapshots
# 用途：回归验证回归测试与测试辅助中的
# `data_paths_inside_workspace_are_excluded_from_snapshots` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_data_paths_inside` →
# `inside.resolve`。
# 分支与异常：
#   验证条件：
# `_data_paths_inside(workspace, inside, outside, workspace) == (inside.resolve(),)`。
def test_data_paths_inside_workspace_are_excluded_from_snapshots(tmp_path) -> None:
    from app.application import _data_paths_inside

    workspace = tmp_path / "ws"
    inside = workspace / ".muharness"
    outside = tmp_path / "data"
    assert _data_paths_inside(workspace, inside, outside, workspace) == (inside.resolve(),)


# 函数说明：test_gateway_tracks_pending_approvals
# 用途：回归验证回归测试与测试辅助中的 `gateway_tracks_pending_approvals` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RecordingRunManager` →
# `MeaRunGateway` → `gateway.start` → `handler.emit` → `AgentEvent` →
# `gateway.awaiting_approval`。
# 分支与异常：
#   验证条件：`gateway.awaiting_approval('exec-1')`。
#   验证条件：`not gateway.awaiting_approval('exec-1')`。
# 副作用与资源：
#   向事件发射器、广播器或连接发送结果/通知，可能影响订阅方可见状态。
async def test_gateway_tracks_pending_approvals() -> None:
    manager = RecordingRunManager()
    gateway = MeaRunGateway(manager)
    await gateway.start("执行", run_id="exec-1", source="mea:executor", source_id="m1")
    handler = manager.started[0]["event_handler"]

    await handler.emit(AgentEvent(run_id="exec-1", type=AgentEventType.TOOL_APPROVAL_REQUIRED))
    assert gateway.awaiting_approval("exec-1")
    await handler.emit(AgentEvent(run_id="exec-1", type=AgentEventType.TOOL_APPROVAL_COMPLETED))
    assert not gateway.awaiting_approval("exec-1")


# 函数说明：test_role_output_budgets_are_capped_by_the_model
# 用途：回归验证回归测试与测试辅助中的 `role_output_budgets_are_capped_by_the_model` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MeaSettings` → `role_limits` →
# `roomy[AgentMode.AUDIT].budget` → `RunBudgetConfig`。
# 分支与异常：
#   验证条件：`roomy[AgentMode.EXECUTE].max_output_tokens == 32768`。
#   验证条件：`roomy[AgentMode.AUDIT].max_output_tokens == 32768`。
#   验证条件：`roomy[AgentMode.MANAGE].max_output_tokens == 16384`。
#   验证条件：`roomy[AgentMode.AUDIT].finalization_max_output_tokens == 8192`。
def test_role_output_budgets_are_capped_by_the_model() -> None:
    from app.runtime.mea.runtimes import MeaSettings, role_limits

    settings = MeaSettings(_env_file=None)
    roomy = role_limits(settings, model_max_output_tokens=65_536)
    assert roomy[AgentMode.EXECUTE].max_output_tokens == 32_768
    assert roomy[AgentMode.AUDIT].max_output_tokens == 32_768
    assert roomy[AgentMode.MANAGE].max_output_tokens == 16_384
    assert roomy[AgentMode.AUDIT].finalization_max_output_tokens == 8_192
    assert roomy[AgentMode.AUDIT].max_steps == 20  # 步数等其余上限不变

    small = role_limits(settings, model_max_output_tokens=4_096)
    assert {limit.max_output_tokens for limit in small.values()} == {4_096}
    assert {limit.finalization_max_output_tokens for limit in small.values()} == {4_096}

    custom = role_limits(MeaSettings(_env_file=None, role_max_output_tokens=20_000))
    assert custom[AgentMode.EXECUTE].max_output_tokens == 20_000

    budget = roomy[AgentMode.AUDIT].budget(RunBudgetConfig(_env_file=None))
    assert budget.finalization_max_output_tokens == 8_192  # 聊天默认是 1200


# 函数说明：test_forced_final_audit_report_keeps_the_role_output_budget
# 用途：审计者用完工具轮次后被迫收尾：这次输出就是审计报告，必须拿到角色的收尾预算而不是
#  1200。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `ModelAdapterRegistry` → `ModelSettings` → `registry.register` → `role_limits`；另有 7
#  个调用点。
# 分支与异常：
#   验证条件：`requests[0].max_output_tokens == 4096`。
#   验证条件：`not final.tools`。
#   验证条件：`final.max_output_tokens == 4096`。
#   验证条件：`default.finalization_max_output_tokens == 1200`。
async def test_forced_final_audit_report_keeps_the_role_output_budget(tmp_path) -> None:
    """审计者用完工具轮次后被迫收尾：这次输出就是审计报告，必须拿到角色的收尾预算而不是 1200。"""

    from pydantic import SecretStr

    from app.models.adapter import ModelAdapter
    from app.models.config import ModelSettings, ProviderConfig
    from app.models.registry import ModelAdapterRegistry
    from app.models.types import ApiStyle, ModelResponse, ModelUsage
    from app.runtime.agent.runtime import AgentRuntime
    from app.runtime.mea.runtimes import MeaSettings, role_limits

    requests: list[Any] = []

    class ReadingAuditor(ModelAdapter):
        # 函数说明：test_forced_final_audit_report_keeps_the_role_output_budget.
        # ReadingAuditor.complete
        # 用途：完成ReadingAuditor，供回归测试与测试辅助使用。
        # 参数：
        #   request：待处理的请求对象。
        # 返回：返回 `ModelResponse(…)`。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCall` → `Message` →
        # `ModelResponse` → `ModelUsage`。
        # 闭包依赖：从外层读取 `requests`。
        async def complete(self, request):
            requests.append(request)
            if request.tools:
                call = ToolCall(id=f"c{len(requests)}", name="read_file", arguments={})
                message = Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))
            else:
                message = Message(role=MessageRole.ASSISTANT, content="状态: complete")
            return ModelResponse(id="r", provider="fake", model="fake-model", message=message,
                                 usage=ModelUsage())

        # 函数说明：test_forced_final_audit_report_keeps_the_role_output_budget.
        # ReadingAuditor.close
        # 用途：关闭ReadingAuditor，供回归测试与测试辅助使用。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        async def close(self) -> None:
            pass

    config = ProviderConfig(provider="fake", model="fake-model", api_key=SecretStr("k"),
                            api_style=ApiStyle.CHAT_COMPLETIONS)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: ReadingAuditor(config), config=config)
    # 离线假模型走保守能力（单次输出最多 4096），和 Application 一样按模型上限封顶
    limits = role_limits(MeaSettings(_env_file=None), model_max_output_tokens=4_096)[AgentMode.AUDIT]
    limits = replace(limits, max_steps=4, max_tool_rounds=2)
    runtime = AgentRuntime(
        registry,
        _registry("read_file"),
        provider="fake",
        max_steps=limits.max_steps,
        max_tool_rounds=limits.max_tool_rounds,
        max_output_tokens=limits.max_output_tokens,
        run_budget_config=limits.budget(RunBudgetConfig(_env_file=None)),
    )

    async for _ in runtime.run_stream("审计 s1", mode=AgentMode.AUDIT):
        pass

    assert requests[0].max_output_tokens == 4_096
    final = requests[-1]
    assert not final.tools
    assert final.max_output_tokens == 4_096  # 以前被压到 1200

    default = RunBudgetConfig(_env_file=None)
    assert default.finalization_max_output_tokens == 1_200



# 函数说明：test_sandbox_gate_approves_shell_and_defers_everything_else
# 用途：回归验证回归测试与测试辅助中的
# `sandbox_gate_approves_shell_and_defers_everything_else` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SandboxAutoApproveGate` →
# `DenyAllGate` → `ApprovalRequest` → `gate.request_approval`。
# 分支与异常：
#   验证条件：
# `(await gate.request_approval(shell)).decision is ApprovalDecision.APPROVED`。
#   验证条件：
# `(await gate.request_approval(network)).decision is ApprovalDecision.DENIED`。
async def test_sandbox_gate_approves_shell_and_defers_everything_else() -> None:
    from app.runtime.mea.approvals import SandboxAutoApproveGate
    from app.tools.approval import ApprovalDecision, ApprovalRequest, DenyAllGate

    gate = SandboxAutoApproveGate(DenyAllGate())
    shell = ApprovalRequest(tool_call_id="c1", tool_name="run_shell_command", arguments={"command": "pytest"})
    network = ApprovalRequest(tool_call_id="c2", tool_name="http_request", arguments={"url": "https://x"})

    assert (await gate.request_approval(shell)).decision is ApprovalDecision.APPROVED
    assert (await gate.request_approval(network)).decision is ApprovalDecision.DENIED


# 函数说明：test_executor_shell_runs_without_asking_when_auto_approved
# 用途：回归验证回归测试与测试辅助中的
# `executor_shell_runs_without_asking_when_auto_approved` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `Shell` →
# `registry.register` → `AskingGate` → `ToolCall` →
# `ToolExecutor(registry, approval_gate=SandboxAutoApproveGate(human)).execute`；另有 3
# 个调用点。
# 分支与异常：
#   验证条件：`result.success is True and shell.calls == 1 and (human.asked == [])`。
async def test_executor_shell_runs_without_asking_when_auto_approved() -> None:
    from app.models.types import ToolPermission
    from app.runtime.mea.approvals import SandboxAutoApproveGate
    from app.tools.approval import ApprovalGate, ApprovalRequest, ApprovalResponse

    class AskingGate(ApprovalGate):
        # 函数说明：
        # test_executor_shell_runs_without_asking_when_auto_approved.AskingGate.__init__
        # 用途：初始化 AskingGate；参数及实际保存的实例字段见下方说明。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        # 副作用与资源：
        #   更新对象字段：`self.asked`。
        def __init__(self) -> None:
            self.asked: list[str] = []

        # 函数说明：test_executor_shell_runs_without_asking_when_auto_approved.
        # AskingGate.request_approval
        # 用途：在回归测试与测试辅助中处理 `request_approval`，通过 `self.asked.append`
        # 完成首个内部处理步骤。
        # 参数：
        #   request：待处理的请求对象，类型 `ApprovalRequest`。
        # 返回：类型 `ApprovalResponse`；不返回结果值（隐式 None）。
        async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
            self.asked.append(request.tool_name)
            raise AssertionError("sandboxed shell should not reach the human gate")

    class Shell(NamedTool):
        # 函数说明：
        # test_executor_shell_runs_without_asking_when_auto_approved.Shell.__init__
        # 用途：初始化 Shell；参数及实际保存的实例字段见下方说明。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
        # 副作用与资源：
        #   更新对象字段：`self._definition`。
        def __init__(self) -> None:
            super().__init__("run_shell_command")
            self._definition = self._definition.model_copy(
                update={"permission": ToolPermission.HUMAN_APPROVAL}
            )

    registry = ToolRegistry()
    shell = Shell()
    registry.register(shell)
    human = AskingGate()
    call = ToolCall(id="c1", name="run_shell_command", arguments={})

    result = await ToolExecutor(registry, approval_gate=SandboxAutoApproveGate(human)).execute(
        call, context=_context(call, AgentMode.EXECUTE)
    )

    assert result.success is True and shell.calls == 1 and human.asked == []


# 函数说明：test_runner_picks_executor_runtime_by_approval_choice
# 用途：回归验证回归测试与测试辅助中的
# `runner_picks_executor_runtime_by_approval_choice` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Runtimes` → `MeaRunner` →
# `SQLiteMeaStore` → `datetime.now` → `MeaRun` → `runner._runtime_for`。
# 分支与异常：
#   验证条件：`runner._runtime_for(manual, AgentMode.EXECUTE) == 'executor-False'`。
#   验证条件：`runner._runtime_for(auto, AgentMode.EXECUTE) == 'executor-True'`。
#   验证条件：`runner._runtime_for(auto, AgentMode.AUDIT) == 'base-audit'`。
#   验证条件：`runtimes.asked == [((), False), (('web_search',), True)]`。
def test_runner_picks_executor_runtime_by_approval_choice(tmp_path) -> None:
    from app.runtime.mea.runner import MeaRunner

    class Runtimes(dict):
        # 函数说明：
        # test_runner_picks_executor_runtime_by_approval_choice.Runtimes.__init__
        # 用途：初始化 Runtimes；参数及实际保存的实例字段见下方说明。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
        # 副作用与资源：
        #   更新对象字段：`self.asked`。
        def __init__(self) -> None:
            super().__init__({mode: f"base-{mode.value}" for mode in (
                AgentMode.MANAGE, AgentMode.EXECUTE, AgentMode.AUDIT)})
            self.asked: list[tuple[tuple[str, ...], bool]] = []

        # 函数说明：
        # test_runner_picks_executor_runtime_by_approval_choice.Runtimes.executor_for
        # 用途：在回归测试与测试辅助中处理 `executor_for`，通过 `self.asked.append` 完成
        # 首个内部处理步骤。
        # 参数：
        #   extra_tools：传给 `tuple` 的输入。
        #   auto_approve_sandbox：`auto_approve_sandbox`输入或配置值。
        # 返回：返回 `f'executor-{auto_approve_sandbox}'`。
        def executor_for(self, extra_tools, auto_approve_sandbox):
            self.asked.append((tuple(extra_tools), auto_approve_sandbox))
            return f"executor-{auto_approve_sandbox}"

    runtimes = Runtimes()
    runner = MeaRunner(store=SQLiteMeaStore(tmp_path / "m.db"), tasks=None, runs=None,  # type: ignore[arg-type]
                       runtimes=runtimes, workspace_root=tmp_path)
    now = datetime.now(UTC)
    manual = MeaRun(id="m1", task_id="t", conversation_id="c", auto_approve_sandbox=False,
                    created_at=now, updated_at=now)
    auto = manual.model_copy(update={"auto_approve_sandbox": True, "extra_tools": ("web_search",)})

    assert runner._runtime_for(manual, AgentMode.EXECUTE) == "executor-False"
    assert runner._runtime_for(auto, AgentMode.EXECUTE) == "executor-True"
    assert runner._runtime_for(auto, AgentMode.AUDIT) == "base-audit"
    assert runtimes.asked == [((), False), (("web_search",), True)]
