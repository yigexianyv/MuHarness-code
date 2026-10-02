
from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import SecretStr

from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
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
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.runtime import AgentRuntime
from app.safety.approval import (
    ApprovalRequestStatus,
    SQLiteApprovalStore,
    WebApprovalGate,
)
from app.tools.approval import (
    ApprovalDecision,
)
from app.tools.approval import (
    ApprovalRequest as ApprovalSubmission,
)
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry


# 函数说明：approval_store
# 用途：保存审批，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：返回 `store`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteApprovalStore` →
# `store.initialize`。
@pytest.fixture
async def approval_store(tmp_path):
    store = SQLiteApprovalStore(tmp_path / "muharness.db")
    await store.initialize()
    return store


# 函数说明：_create
# 用途：创建回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   store：持久化存储依赖。
#   **overrides：额外关键字参数，按实现处理或转交。
# 返回：类型 `object`；返回 `await store.create(**params)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`params.update` → `store.create`。
async def _create(store, **overrides) -> object:
    params = {
        "run_id": "run-1",
        "conversation_id": "conv-1",
        "tool_name": "run_shell_command",
        "tool_call_id": "call-1",
        "arguments": {"command": "pytest"},
        "reason": "运行测试",
    }
    params.update(overrides)
    return await store.create(**params)




# 函数说明：test_create_approval_request
# 用途：回归验证回归测试与测试辅助中的 `create_approval_request` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create`。
# 分支与异常：
#   验证条件：`record.id`。
#   验证条件：`record.status is ApprovalRequestStatus.PENDING`。
#   验证条件：`record.resolved_at is None`。
#   验证条件：`record.tool_name == 'run_shell_command'`。
async def test_create_approval_request(approval_store) -> None:
    record = await _create(approval_store)

    assert record.id
    assert record.status is ApprovalRequestStatus.PENDING
    assert record.resolved_at is None
    assert record.tool_name == "run_shell_command"
    assert record.tool_call_id == "call-1"
    assert record.arguments == {"command": "pytest"}
    assert record.reason == "运行测试"
    assert record.run_id == "run-1"
    assert record.conversation_id == "conv-1"

    fetched = await approval_store.get(record.id)
    assert fetched == record


# 函数说明：test_approve
# 用途：回归验证回归测试与测试辅助中的 `approve` 场景，下方断言说明列出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create` → `approval_store.resolve`。
# 分支与异常：
#   验证条件：`resolved.status is ApprovalRequestStatus.APPROVED`。
#   验证条件：`resolved.resolved_at is not None`。
#   验证条件：`resolved.created_at is not None`。
async def test_approve(approval_store) -> None:
    record = await _create(approval_store)

    resolved = await approval_store.resolve(
        record.id,
        ApprovalRequestStatus.APPROVED,
    )
    assert resolved.status is ApprovalRequestStatus.APPROVED
    assert resolved.resolved_at is not None
    assert resolved.created_at is not None


# 函数说明：test_deny
# 用途：回归验证回归测试与测试辅助中的 `deny` 场景，下方断言说明列出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create` → `approval_store.resolve`。
# 分支与异常：
#   验证条件：`resolved.status is ApprovalRequestStatus.DENIED`。
#   验证条件：`resolved.resolved_at is not None`。
async def test_deny(approval_store) -> None:
    record = await _create(approval_store)

    resolved = await approval_store.resolve(
        record.id,
        ApprovalRequestStatus.DENIED,
    )
    assert resolved.status is ApprovalRequestStatus.DENIED
    assert resolved.resolved_at is not None


# 函数说明：test_duplicate_resolve_rejected
# 用途：回归验证回归测试与测试辅助中的 `duplicate_resolve_rejected` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create` → `approval_store.resolve` →
#  `pytest.raises`。
# 分支与异常：
#   验证条件：`fetched.status is ApprovalRequestStatus.APPROVED`。
#   预期异常：`pytest.raises(ValueError, match='already resolved')`。
async def test_duplicate_resolve_rejected(approval_store) -> None:
    record = await _create(approval_store)
    await approval_store.resolve(record.id, ApprovalRequestStatus.APPROVED)

    with pytest.raises(ValueError, match="already resolved"):
        await approval_store.resolve(record.id, ApprovalRequestStatus.DENIED)
    with pytest.raises(ValueError, match="already resolved"):
        await approval_store.resolve(record.id, ApprovalRequestStatus.APPROVED)

    fetched = await approval_store.get(record.id)
    assert fetched.status is ApprovalRequestStatus.APPROVED


# 函数说明：test_resolve_to_pending_rejected
# 用途：回归验证回归测试与测试辅助中的 `resolve_to_pending_rejected` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create` → `pytest.raises` →
# `approval_store.resolve`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='PENDING')`。
async def test_resolve_to_pending_rejected(approval_store) -> None:
    record = await _create(approval_store)
    with pytest.raises(ValueError, match="PENDING"):
        await approval_store.resolve(record.id, ApprovalRequestStatus.PENDING)


# 函数说明：test_list_filters
# 用途：回归验证回归测试与测试辅助中的 `list_filters` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create` → `approval_store.resolve` →
#  `approval_store.list`。
# 分支与异常：
#   验证条件：`[item.id for item in pending] == [second.id]`。
#   验证条件：`{item.id for item in all_records} == {first.id, second.id}`。
#   验证条件：`[item.id for item in by_run] == [first.id]`。
async def test_list_filters(approval_store) -> None:
    first = await _create(approval_store, run_id="run-1")
    second = await _create(
        approval_store,
        run_id="run-2",
        tool_name="http_request",
        tool_call_id="call-2",
    )
    await approval_store.resolve(first.id, ApprovalRequestStatus.APPROVED)

    pending = await approval_store.list(status=ApprovalRequestStatus.PENDING)
    assert [item.id for item in pending] == [second.id]

    all_records = await approval_store.list()
    assert {item.id for item in all_records} == {first.id, second.id}

    by_run = await approval_store.list(run_id="run-1")
    assert [item.id for item in by_run] == [first.id]


# 函数说明：test_cancelled_approval_cannot_resolve
# 用途：回归验证回归测试与测试辅助中的 `cancelled_approval_cannot_resolve` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create` → `approval_store.resolve` →
#  `pytest.raises`。
# 分支与异常：
#   验证条件：`resolved.status is ApprovalRequestStatus.CANCELLED`。
#   预期异常：`pytest.raises(ValueError, match='already resolved')`。
async def test_cancelled_approval_cannot_resolve(approval_store) -> None:

    record = await _create(approval_store, run_id="run-1")
    resolved = await approval_store.resolve(
        record.id,
        ApprovalRequestStatus.CANCELLED,
    )
    assert resolved.status is ApprovalRequestStatus.CANCELLED

    with pytest.raises(ValueError, match="already resolved"):
        await approval_store.resolve(record.id, ApprovalRequestStatus.APPROVED)
    with pytest.raises(ValueError, match="already resolved"):
        await approval_store.resolve(record.id, ApprovalRequestStatus.DENIED)


# 函数说明：test_cancel_pending_for_run
# 用途：回归验证回归测试与测试辅助中的 `cancel_pending_for_run` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create` → `approval_store.resolve` →
#  `approval_store.cancel_pending_for_run`。
# 分支与异常：
#   验证条件：`cancelled == 1`。
#   验证条件：
# `(await approval_store.get(target.id)).status is ApprovalRequestStatus.CANCELLED`。
#   验证条件：
# `(await approval_store.get(other_run.id)).status is ApprovalRequestStatus.PENDING`。
#   验证条件：
# `(await approval_store.get(resolved.id)).status is ApprovalRequestStatus.APPROVED`。
async def test_cancel_pending_for_run(approval_store) -> None:

    target = await _create(approval_store, run_id="run-cancel", tool_call_id="c1")
    other_run = await _create(
        approval_store,
        run_id="run-other",
        tool_call_id="c2",
    )
    resolved = await _create(approval_store, run_id="run-cancel", tool_call_id="c3")
    await approval_store.resolve(resolved.id, ApprovalRequestStatus.APPROVED)

    cancelled = await approval_store.cancel_pending_for_run("run-cancel")

    assert cancelled == 1  
    assert (await approval_store.get(target.id)).status is (
        ApprovalRequestStatus.CANCELLED
    )
    assert (await approval_store.get(other_run.id)).status is (
        ApprovalRequestStatus.PENDING
    )
    assert (await approval_store.get(resolved.id)).status is (
        ApprovalRequestStatus.APPROVED
    )


# 函数说明：test_reconcile_orphans_cancels_orphan_pending
# 用途：回归验证回归测试与测试辅助中的 `reconcile_orphans_cancels_orphan_pending` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create` → `approval_store.resolve` →
#  `approval_store.reconcile_orphans`。
# 分支与异常：
#   验证条件：`cancelled == 2`。
#   验证条件：
# `(await approval_store.get(live.id)).status is ApprovalRequestStatus.PENDING`。
#   验证条件：
# `(await approval_store.get(orphan.id)).status is ApprovalRequestStatus.CANCELLED`。
#   验证条件：
# `(await approval_store.get(no_run.id)).status is ApprovalRequestStatus.CANCELLED`。
async def test_reconcile_orphans_cancels_orphan_pending(approval_store) -> None:

    live = await _create(approval_store, run_id="active-run", tool_call_id="c1")
    orphan = await _create(approval_store, run_id="dead-run", tool_call_id="c2")
    no_run = await _create(approval_store, run_id=None, tool_call_id="c3")
    resolved = await _create(approval_store, run_id="dead-run", tool_call_id="c4")
    await approval_store.resolve(resolved.id, ApprovalRequestStatus.APPROVED)

    cancelled = await approval_store.reconcile_orphans({"active-run"})

    assert cancelled == 2  
    assert (await approval_store.get(live.id)).status is (
        ApprovalRequestStatus.PENDING
    )
    assert (await approval_store.get(orphan.id)).status is (
        ApprovalRequestStatus.CANCELLED
    )
    assert (await approval_store.get(no_run.id)).status is (
        ApprovalRequestStatus.CANCELLED
    )
    assert (await approval_store.get(resolved.id)).status is (
        ApprovalRequestStatus.APPROVED
    )




# 函数说明：test_approve_right_after_record_created_wakes_run
# 用途：回归验证回归测试与测试辅助中的 `approve_right_after_record_created_wakes_run` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`WebApprovalGate` →
# `ApprovalSubmission` → `asyncio.create_task` → `gate.request_approval` →
# `approval_store.list` → `asyncio.sleep`；另有 3 个调用点。
# 分支与异常：
#   当 `pending` 时，结束当前循环。
#   验证条件：`pending`。
#   验证条件：`not waiting.done()`。
#   验证条件：`response.decision is ApprovalDecision.APPROVED`。
#   验证条件：`gate.pending_count == 0`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
async def test_approve_right_after_record_created_wakes_run(approval_store) -> None:

    gate = WebApprovalGate(approval_store)
    submission = ApprovalSubmission(
        tool_call_id="call-1",
        tool_name="run_shell_command",
        arguments={"command": "pytest"},
        run_id="run-1",
    )
    waiting = asyncio.create_task(gate.request_approval(submission))

    for _ in range(500):
        pending = await approval_store.list(
            status=ApprovalRequestStatus.PENDING,
        )
        if pending:
            break
        await asyncio.sleep(0)
    assert pending, "应创建 PENDING ApprovalRequest"
    assert not waiting.done()

    await gate.approve(pending[0].id)
    response = await asyncio.wait_for(waiting, timeout=5)
    assert response.decision is ApprovalDecision.APPROVED
    assert gate.pending_count == 0


# 函数说明：test_cancel_run_cancels_pending_approval
# 用途：回归验证回归测试与测试辅助中的 `cancel_run_cancels_pending_approval` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `_BlockingAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`；
# 另有 14 个调用点。
# 分支与异常：
#   当 `adapter.started.is_set()` 时，结束当前循环。
#   验证条件：`adapter.started.is_set()`。
#   验证条件：`record.status is ApprovalRequestStatus.PENDING`。
#   验证条件：`after is not None`。
#   验证条件：`after.status is ApprovalRequestStatus.CANCELLED`。
async def test_cancel_run_cancels_pending_approval(tmp_path) -> None:

    from app.runtime.agent.runtime import AgentRuntime
    from app.runtime.checkpoint import SQLiteCheckpointStore
    from app.runtime.run import RunManager, SQLiteRunStore

    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = _BlockingAdapter(config)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)

    database = tmp_path / "muharness.db"
    run_store = SQLiteRunStore(database)
    await run_store.initialize()
    checkpoint_store = SQLiteCheckpointStore(database)
    await checkpoint_store.initialize()
    approval_store = SQLiteApprovalStore(database)
    await approval_store.initialize()

    runtime = AgentRuntime(registry, ToolRegistry(), provider="fake")
    manager = RunManager(
        run_store,
        checkpoint_store,
        runtime,
        approval_store=approval_store,
    )
    run_id, _ = await manager.start("阻塞", conversation_id="conv-1")
    for _ in range(200):
        if adapter.started.is_set():
            break
        await asyncio.sleep(0)
    assert adapter.started.is_set()

    record = await approval_store.create(
        run_id=run_id,
        conversation_id="conv-1",
        tool_name="run_shell_command",
        tool_call_id="call-1",
        arguments={"command": "pytest"},
    )
    assert record.status is ApprovalRequestStatus.PENDING

    await manager.cancel(run_id)

    after = await approval_store.get(record.id)
    assert after is not None
    assert after.status is ApprovalRequestStatus.CANCELLED


class _BlockingAdapter(ModelAdapter):

    # 函数说明：_BlockingAdapter.__init__
    # 用途：初始化 _BlockingAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super` →
    # `asyncio.Event`。
    # 副作用与资源：
    #   更新对象字段：`self.started`、`self.cancelled`、`self.requests`。
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.started = asyncio.Event()
        self.cancelled = False
        self.requests: list[ModelRequest] = []

    # 函数说明：_BlockingAdapter.complete
    # 用途：完成_BlockingAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.started.set` →
    # `asyncio.Event().wait` → `asyncio.Event`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，重新抛出原异常。
    # 副作用与资源：
    #   更新对象字段：`self.cancelled`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("阻塞模型不应正常完成")

    # 函数说明：_BlockingAdapter.close
    # 用途：关闭_BlockingAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：test_web_gate_waits_until_approve
# 用途：回归验证回归测试与测试辅助中的 `web_gate_waits_until_approve` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`WebApprovalGate` → `asyncio.Event` →
# `gate.set_broadcaster` → `ApprovalSubmission` → `asyncio.create_task` →
# `gate.request_approval`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`pending`。
#   验证条件：`notifications[0][0] == 'approval.required'`。
#   验证条件：`not waiting.done()`。
#   验证条件：`response.decision is ApprovalDecision.APPROVED`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
async def test_web_gate_waits_until_approve(approval_store) -> None:
    gate = WebApprovalGate(approval_store)
    notifications: list[tuple[str, object]] = []
    broadcasted = asyncio.Event()

    # 函数说明：test_web_gate_waits_until_approve.broadcaster
    # 用途：在回归测试与测试辅助中处理 `broadcaster`，通过 `notifications.append` 完成首
    # 个内部处理步骤。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `object`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`broadcasted.set`。
    # 闭包依赖：从外层读取 `broadcasted`、`notifications`。
    async def broadcaster(method: str, params: object) -> None:
        notifications.append((method, params))
        broadcasted.set()

    gate.set_broadcaster(broadcaster)

    submission = ApprovalSubmission(
        tool_call_id="call-1",
        tool_name="run_shell_command",
        arguments={"command": "pytest"},
        description="运行测试",
        run_id="run-1",
        conversation_id="conv-1",
    )
    waiting = asyncio.create_task(gate.request_approval(submission))

    await asyncio.wait_for(broadcasted.wait(), timeout=5)

    pending = await approval_store.list(status=ApprovalRequestStatus.PENDING)
    assert pending, "应创建 PENDING ApprovalRequest"
    assert notifications[0][0] == "approval.required"
    assert not waiting.done()

    approved = await gate.approve(pending[0].id)
    response = await asyncio.wait_for(waiting, timeout=5)

    assert response.decision is ApprovalDecision.APPROVED
    assert approved.status is ApprovalRequestStatus.APPROVED
    assert notifications[-1][0] == "approval.resolved"
    assert gate.pending_count == 0


# 函数说明：test_web_gate_deny_returns_denied
# 用途：回归验证回归测试与测试辅助中的 `web_gate_deny_returns_denied` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`WebApprovalGate` →
# `ApprovalSubmission` → `asyncio.create_task` → `gate.request_approval` →
# `approval_store.list` → `asyncio.sleep`；另有 2 个调用点。
# 分支与异常：
#   当 `pending` 时，结束当前循环。
#   验证条件：`response.decision is ApprovalDecision.DENIED`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
async def test_web_gate_deny_returns_denied(approval_store) -> None:
    gate = WebApprovalGate(approval_store)
    submission = ApprovalSubmission(
        tool_call_id="call-1",
        tool_name="run_shell_command",
        arguments={"command": "pytest"},
        run_id="run-1",
    )
    waiting = asyncio.create_task(gate.request_approval(submission))

    for _ in range(200):
        pending = await approval_store.list(
            status=ApprovalRequestStatus.PENDING,
        )
        if pending:
            break
        await asyncio.sleep(0)

    await gate.deny(pending[0].id)
    response = await asyncio.wait_for(waiting, timeout=5)
    assert response.decision is ApprovalDecision.DENIED


# 函数说明：test_gate_resolve_only_once
# 用途：回归验证回归测试与测试辅助中的 `gate_resolve_only_once` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`WebApprovalGate` →
# `ApprovalSubmission` → `asyncio.create_task` → `gate.request_approval` →
# `approval_store.list` → `asyncio.sleep`；另有 4 个调用点。
# 分支与异常：
#   当 `pending` 时，结束当前循环。
#   预期异常：`pytest.raises(ValueError, match='already resolved')`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
async def test_gate_resolve_only_once(approval_store) -> None:

    gate = WebApprovalGate(approval_store)
    submission = ApprovalSubmission(
        tool_call_id="call-1",
        tool_name="run_shell_command",
        arguments={"command": "pytest"},
        run_id="run-1",
    )
    waiting = asyncio.create_task(gate.request_approval(submission))
    for _ in range(200):
        pending = await approval_store.list(
            status=ApprovalRequestStatus.PENDING,
        )
        if pending:
            break
        await asyncio.sleep(0)
    approval_id = pending[0].id

    await gate.approve(approval_id)
    await asyncio.wait_for(waiting, timeout=5)
    with pytest.raises(ValueError, match="already resolved"):
        await gate.approve(approval_id)
    with pytest.raises(ValueError, match="already resolved"):
        await gate.deny(approval_id)


# 函数说明：test_disconnect_does_not_auto_decide
# 用途：回归验证回归测试与测试辅助中的 `disconnect_does_not_auto_decide` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   approval_store：审批记录存储。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`WebApprovalGate` →
# `ApprovalSubmission` → `asyncio.create_task` → `gate.request_approval` →
# `approval_store.list` → `asyncio.sleep`；另有 3 个调用点。
# 分支与异常：
#   当 `pending` 时，结束当前循环。
#   验证条件：`pending and pending[0].status is ApprovalRequestStatus.PENDING`。
#   验证条件：`not waiting.done()`。
#   验证条件：`response.decision is ApprovalDecision.DENIED`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
async def test_disconnect_does_not_auto_decide(approval_store) -> None:

    gate = WebApprovalGate(approval_store)
    submission = ApprovalSubmission(
        tool_call_id="call-1",
        tool_name="run_shell_command",
        arguments={"command": "pytest"},
        run_id="run-1",
    )
    waiting = asyncio.create_task(gate.request_approval(submission))
    for _ in range(200):
        pending = await approval_store.list(
            status=ApprovalRequestStatus.PENDING,
        )
        if pending:
            break
        await asyncio.sleep(0)

    assert pending and pending[0].status is ApprovalRequestStatus.PENDING
    assert not waiting.done()

    await gate.deny(pending[0].id)
    response = await asyncio.wait_for(waiting, timeout=5)
    assert response.decision is ApprovalDecision.DENIED




class ApprovalProbeTool(BaseTool):
    definition = ToolDefinition(
        name="approval_probe",
        description="需要审批的探测工具",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        permission=ToolPermission.HUMAN_APPROVAL,
    )

    # 函数说明：ApprovalProbeTool.__init__
    # 用途：初始化 ApprovalProbeTool；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    def __init__(self) -> None:
        self.executions = 0

    # 函数说明：ApprovalProbeTool.execute
    # 用途：执行ApprovalProbeTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, object]`；读取键
    # `value`。
    # 返回：类型 `str`；返回 `f"probe-{arguments.get('value')}"`。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    async def execute(self, arguments: dict[str, object]) -> str:
        self.executions += 1
        return f"probe-{arguments.get('value')}"


class ScriptedAdapter(ModelAdapter):
    # 函数说明：ScriptedAdapter.__init__
    # 用途：初始化 ScriptedAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`
    # 。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.requests`。
    def __init__(
        self,
        config: ProviderConfig,
        responses: Sequence[ModelResponse | Exception],
    ) -> None:
        super().__init__(config)
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    # 函数说明：ScriptedAdapter.complete
    # 用途：完成ScriptedAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.responses.pop`。
    # 分支与异常：
    #   当 `isinstance(response, Exception)` 时，抛出 `response`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    # 函数说明：ScriptedAdapter.close
    # 用途：关闭ScriptedAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str | None`；默认 `None`。
#   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`；默认 `()`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def _response(
    *,
    content: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
) -> ModelResponse:
    return ModelResponse(
        id="fake-response",
        provider="fake",
        model="fake-model",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        ),
        usage=ModelUsage(),
    )


# 函数说明：_registry
# 用途：在回归测试与测试辅助中处理 `_registry`，通过 `registry.register` 完成首个内部处
# 理步骤。
# 参数：
#   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`。
# 返回：类型 `tuple[ModelAdapterRegistry, ScriptedAdapter]`；返回 `(registry, adapter)`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `ScriptedAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`。
def _registry(
    responses: Sequence[ModelResponse | Exception],
) -> tuple[ModelAdapterRegistry, ScriptedAdapter]:
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


# 函数说明：test_agent_waits_for_approval_then_continues
# 用途：回归验证回归测试与测试辅助中的 `agent_waits_for_approval_then_continues` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` → `_response` → `ToolCall`
#  → `ApprovalProbeTool` → `ToolRegistry` → `tools.register`；另有 12 个调用点。
# 分支与异常：
#   当 `pending` 时，结束当前循环。
#   验证条件：`pending and pending[0].tool_name == 'approval_probe'`。
#   验证条件：`pending[0].run_id == 'run-1'`。
#   验证条件：`tool.executions == 0`。
#   验证条件：`not result_task.done()`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
async def test_agent_waits_for_approval_then_continues(tmp_path) -> None:
    registry, _ = _registry(
        [
            _response(
                tool_calls=(
                    ToolCall(
                        id="ap-1",
                        name="approval_probe",
                        arguments={"value": 7},
                    ),
                )
            ),
            _response(content="审批通过，任务完成"),
        ]
    )
    tool = ApprovalProbeTool()
    tools = ToolRegistry()
    tools.register(tool)
    store = SQLiteApprovalStore(tmp_path / "muharness.db")
    await store.initialize()
    gate = WebApprovalGate(store)
    handler = InMemoryEventHandler()

    runtime = AgentRuntime(
        registry,
        tools,
        provider="fake",
        approval_gate=gate,
    )
    result_task = asyncio.create_task(
        runtime.run(
            "执行需要审批的任务",
            run_id="run-1",
            event_handler=handler,
        )
    )

    for _ in range(500):
        pending = await store.list(status=ApprovalRequestStatus.PENDING)
        if pending:
            break
        await asyncio.sleep(0)
    assert pending and pending[0].tool_name == "approval_probe"
    assert pending[0].run_id == "run-1"
    assert tool.executions == 0
    assert not result_task.done()

    await gate.approve(pending[0].id)
    result = await asyncio.wait_for(result_task, timeout=10)

    assert result.ok is True
    assert tool.executions == 1
    assert result.tool_calls[0].result.success is True

    approval_events = [
        event
        for event in handler.events
        if event.type
        in {
            AgentEventType.TOOL_APPROVAL_REQUIRED,
            AgentEventType.TOOL_APPROVAL_COMPLETED,
        }
    ]
    assert [event.type for event in approval_events] == [
        AgentEventType.TOOL_APPROVAL_REQUIRED,
        AgentEventType.TOOL_APPROVAL_COMPLETED,
    ]
    assert approval_events[1].approval_decision is ApprovalDecision.APPROVED


# 函数说明：test_agent_denied_tool_not_executed
# 用途：回归验证回归测试与测试辅助中的 `agent_denied_tool_not_executed` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_registry` → `_response` → `ToolCall`
#  → `ApprovalProbeTool` → `ToolRegistry` → `tools.register`；另有 10 个调用点。
# 分支与异常：
#   当 `pending` 时，结束当前循环。
#   验证条件：`tool.executions == 0`。
#   验证条件：`result.tool_calls[0].result.success is False`。
#   验证条件：`'denied' in (result.tool_calls[0].result.error or '')`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
async def test_agent_denied_tool_not_executed(tmp_path) -> None:
    registry, _ = _registry(
        [
            _response(
                tool_calls=(
                    ToolCall(
                        id="ap-1",
                        name="approval_probe",
                        arguments={"value": 1},
                    ),
                )
            ),
            _response(content="审批被拒绝"),
        ]
    )
    tool = ApprovalProbeTool()
    tools = ToolRegistry()
    tools.register(tool)
    store = SQLiteApprovalStore(tmp_path / "muharness.db")
    await store.initialize()
    gate = WebApprovalGate(store)

    runtime = AgentRuntime(
        registry,
        tools,
        provider="fake",
        approval_gate=gate,
    )
    result_task = asyncio.create_task(
        runtime.run("执行需要审批的任务", run_id="run-1")
    )
    for _ in range(500):
        pending = await store.list(status=ApprovalRequestStatus.PENDING)
        if pending:
            break
        await asyncio.sleep(0)

    await gate.deny(pending[0].id)
    result = await asyncio.wait_for(result_task, timeout=10)

    assert tool.executions == 0
    assert result.tool_calls[0].result.success is False
    assert "denied" in (result.tool_calls[0].result.error or "")
