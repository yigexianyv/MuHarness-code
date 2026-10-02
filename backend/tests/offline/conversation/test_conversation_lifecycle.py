
from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from app.domain.artifact import ArtifactService, SQLiteArtifactStore
from app.domain.automation import Schedule, ScheduleKind, SQLiteAutomationStore
from app.domain.conversation.coordinator import ConversationOperationCoordinator
from app.domain.conversation.lifecycle import ConversationLifecycleService
from app.domain.conversation.store import SQLiteConversationStore
from app.domain.task import FileTaskStore
from app.models.types import Message, MessageRole
from app.records.evidence import SQLiteEvidenceStore
from app.records.trace import SQLiteTraceStore
from app.runtime.agent.events import AgentEvent, AgentEventType
from app.runtime.checkpoint import SQLiteCheckpointStore
from app.runtime.context import (
    ConversationSummaryState,
    RollingConversationSummary,
    SQLiteConversationSummaryStore,
)
from app.runtime.run import SQLiteRunStore
from app.safety.approval import SQLiteApprovalStore
from app.tools import ApprovalScope, PermissionEffect, PermissionRule
from app.tools.permissions import SQLitePermissionRuleStore


class _RunManagerStub:
    # 函数说明：_RunManagerStub.cancel_for_conversation
    # 用途：取消会话，供回归测试与测试辅助使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `tuple[()]`；返回 `()`。
    async def cancel_for_conversation(self, conversation_id: str) -> tuple[()]:
        return ()

    # 函数说明：_RunManagerStub.forget_results
    # 用途：清除结果集合，供回归测试与测试辅助使用。
    # 参数：
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def forget_results(self, run_ids: tuple[str, ...]) -> None:
        pass


class _AutomationSchedulerStub:
    # 函数说明：_AutomationSchedulerStub.__init__
    # 用途：初始化 _AutomationSchedulerStub；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteAutomationStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.store`。
    def __init__(self, store: SQLiteAutomationStore) -> None:
        self.store = store

    # 函数说明：_AutomationSchedulerStub.delete_for_conversation
    # 用途：删除会话，供回归测试与测试辅助使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `int`；返回 `await self.store.delete_for_conversation(conversation_id)`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self.store.delete_for_conversation`。
    async def delete_for_conversation(self, conversation_id: str) -> int:
        return await self.store.delete_for_conversation(conversation_id)


class _PostRunProcessorStub:
    # 函数说明：_PostRunProcessorStub.cancel_for_conversation
    # 用途：取消会话，供回归测试与测试辅助使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `int`；返回 `0`。
    async def cancel_for_conversation(self, conversation_id: str) -> int:
        return 0


# 函数说明：test_delete_removes_all_conversation_private_data
# 用途：回归验证回归测试与测试辅助中的 `delete_removes_all_conversation_private_data` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `SQLiteConversationStore` → `SQLiteConversationSummaryStore` → `SQLiteRunStore` →
# `SQLiteCheckpointStore` → `SQLiteTraceStore`；另有 47 个调用点。
# 分支与异常：
#   验证条件：`artifact_path is not None and artifact_path.is_file()`。
#   验证条件：`result is not None`。
#   验证条件：`result.deleted is True`。
#   验证条件：`result.deleted_runs == 1`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / 'report.txt').write_text`。
@pytest.mark.asyncio
async def test_delete_removes_all_conversation_private_data(tmp_path) -> None:
    database = tmp_path / "muharness.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tasks_dir = tmp_path / "tasks"

    conversations = SQLiteConversationStore(database)
    summaries = SQLiteConversationSummaryStore(database)
    runs = SQLiteRunStore(database)
    checkpoints = SQLiteCheckpointStore(database)
    traces = SQLiteTraceStore(database)
    evidence = SQLiteEvidenceStore(database)
    approvals = SQLiteApprovalStore(database)
    artifacts_store = SQLiteArtifactStore(database)
    artifacts = ArtifactService(
        artifacts_store,
        workspace,
        managed_dir=tmp_path / "artifacts",
    )
    tasks = FileTaskStore(tasks_dir)
    rules = SQLitePermissionRuleStore(database)
    automations = SQLiteAutomationStore(database)

    for store in (
        conversations,
        summaries,
        runs,
        checkpoints,
        traces,
        evidence,
        approvals,
        artifacts_store,
        tasks,
        rules,
        automations,
    ):
        await store.initialize()

    conversation_a = await conversations.create(
        messages=(Message(role=MessageRole.USER, content="仅属于 A"),)
    )
    conversation_b = await conversations.create(
        messages=(Message(role=MessageRole.USER, content="仅属于 B"),)
    )
    await summaries.save(
        conversation_a.id,
        ConversationSummaryState(
            summary=RollingConversationSummary(current_objective="删除测试"),
            covered_message_count=1,
        ),
    )

    run_a = await runs.create(
        conversation_id=conversation_a.id,
        user_message="执行 A",
    )
    await runs.mark_started(run_a.id)
    await runs.mark_completed(run_a.id)
    run_b = await runs.create(
        conversation_id=conversation_b.id,
        user_message="执行 B",
    )
    await runs.mark_started(run_b.id)
    await runs.mark_completed(run_b.id)

    await checkpoints.start(
        run_a.id,
        conversation_id=conversation_a.id,
        user_message=Message(role=MessageRole.USER, content="执行 A"),
    )

    await traces.record_event(
        AgentEvent(
            run_id=run_a.id,
            conversation_id=conversation_a.id,
            sequence=0,
            type=AgentEventType.AGENT_STARTED,
        )
    )

    await traces.record_event(
        AgentEvent(
            run_id=run_b.id,
            conversation_id=conversation_b.id,
            sequence=0,
            type=AgentEventType.AGENT_STARTED,
        )
    )

    content = "A 的不可变工具证据"
    await evidence.create(
        conversation_id=conversation_a.id,
        run_id=run_a.id,
        tool_call_id="tool-a",
        tool_name="read_file",
        content=content,
        sha256=hashlib.sha256(content.encode()).hexdigest(),
    )
    await approvals.create(
        run_id=run_a.id,
        conversation_id=conversation_a.id,
        tool_name="run_shell_command",
        tool_call_id="approval-a",
    )
    task_a = await tasks.create(
        title="A 的任务",
        owner_conversation_id=conversation_a.id,
    )
    task_b = await tasks.create(
        title="B 的任务",
        owner_conversation_id=conversation_b.id,
    )

    (workspace / "report.txt").write_text("artifact", encoding="utf-8")
    artifact_a = await artifacts.publish_file(
        path="report.txt",
        run_id=run_a.id,
        conversation_id=None,
        task_id=task_a.id,
    )
    artifact_path = await artifacts.file_path(artifact_a.id)
    assert artifact_path is not None and artifact_path.is_file()

    await rules.add(
        PermissionRule(
            id="conversation-rule-a",
            tool_name="run_shell_command",
            scope=ApprovalScope.CONVERSATION,
            scope_id=conversation_a.id,
            effect=PermissionEffect.ALLOW,
            matcher_type="arguments_exact",
            matcher={"command": "pwd"},
            description="A 会话规则",
        )
    )
    await rules.add(
        PermissionRule(
            id="run-rule-a",
            tool_name="run_shell_command",
            scope=ApprovalScope.RUN,
            scope_id=run_a.id,
            effect=PermissionEffect.ALLOW,
            matcher_type="arguments_exact",
            matcher={"command": "pwd"},
            description="A Run 规则",
        )
    )
    await automations.create(
        title="A 自动化",
        prompt="继续 A",
        conversation_id=conversation_a.id,
        schedule=Schedule(
            kind=ScheduleKind.ONCE,
            run_at=datetime.now(UTC) + timedelta(days=1),
        ),
        next_run_at=datetime.now(UTC) + timedelta(days=1),
    )

    coordinator = ConversationOperationCoordinator()
    lifecycle = ConversationLifecycleService(
        conversations,
        coordinator,
        _RunManagerStub(),  
        runs,
        checkpoints,
        traces,
        evidence,
        approvals,
        artifacts,
        tasks,
        rules,
        _AutomationSchedulerStub(automations),  
        _PostRunProcessorStub(),  
    )

    result = await lifecycle.delete(conversation_a.id)

    assert result is not None
    assert result.deleted is True
    assert result.deleted_runs == 1
    assert result.deleted_checkpoints == 1
    assert result.deleted_traces == 1
    assert result.deleted_evidence == 1
    assert result.deleted_approvals == 1
    assert result.deleted_permission_rules == 2
    assert result.deleted_tasks == 1
    assert result.deleted_artifacts == 1
    assert result.deleted_automations == 1
    assert result.audit_records_retained is False

    assert await conversations.get(conversation_a.id) is None
    assert await summaries.load(conversation_a.id) is None
    assert await runs.get(run_a.id) is None
    assert await checkpoints.get(run_a.id) is None
    assert await traces.get(run_a.id) is None
    assert await evidence.list_recent(conversation_id=conversation_a.id) == ()
    assert await approvals.list(conversation_id=conversation_a.id) == ()
    assert await artifacts_store.get(artifact_a.id) is None
    assert not artifact_path.exists()
    assert await tasks.get(task_a.id) is None
    assert await rules.list(scope_ids=(conversation_a.id, run_a.id)) == ()
    assert await automations.list(conversation_id=conversation_a.id) == ()

    assert await conversations.get(conversation_b.id) is not None
    assert await runs.get(run_b.id) is not None
    assert await traces.get(run_b.id) is not None
    assert await tasks.get(task_b.id) is not None


# 函数说明：test_delete_missing_conversation_is_noop
# 用途：回归验证回归测试与测试辅助中的 `delete_missing_conversation_is_noop` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `conversations.initialize` → `ConversationLifecycleService` →
# `ConversationOperationCoordinator` → `_MustNotRun` → `lifecycle.delete`。
# 分支与异常：
#   验证条件：`await lifecycle.delete('missing') is None`。
@pytest.mark.asyncio
async def test_delete_missing_conversation_is_noop(tmp_path) -> None:
    database = tmp_path / "muharness.db"
    conversations = SQLiteConversationStore(database)
    await conversations.initialize()

    class _MustNotRun:
        # 函数说明：test_delete_missing_conversation_is_noop._MustNotRun.__getattr__
        # 用途：处理回归测试与测试辅助中的 `__getattr__` 数据；结果及边界条件见下方说明
        # 。
        # 参数：
        #   name：目标对象、工具或配置项名称，类型 `str`。
        # 返回：不返回结果值（隐式 None）。
        def __getattr__(self, name: str):
            raise AssertionError(f"不应调用：{name}")

    lifecycle = ConversationLifecycleService(
        conversations,
        ConversationOperationCoordinator(),
        _MustNotRun(),  
        _MustNotRun(),  
        _MustNotRun(),  
        _MustNotRun(),  
        _MustNotRun(),  
        _MustNotRun(),  
        _MustNotRun(),  
        _MustNotRun(),  
        _MustNotRun(),  
        _MustNotRun(),  
        _MustNotRun(),  
    )

    assert await lifecycle.delete("missing") is None


# 函数说明：test_deletion_blocks_new_conversation_execution
# 用途：回归验证回归测试与测试辅助中的 `deletion_blocks_new_conversation_execution` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ConversationOperationCoordinator` →
# `asyncio.Event` → `asyncio.create_task` → `delete` → `stop_started.wait` →
# `pytest.raises`；另有 2 个调用点。
# 分支与异常：
#   预期异常：`pytest.raises(KeyError, match='正在删除')`。
# 副作用与资源：
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
@pytest.mark.asyncio
async def test_deletion_blocks_new_conversation_execution() -> None:
    coordinator = ConversationOperationCoordinator()
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()

    # 函数说明：test_deletion_blocks_new_conversation_execution.stop_active_work
    # 用途：停止活跃项，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`stop_started.set` →
    # `allow_stop.wait`。
    # 闭包依赖：从外层读取 `allow_stop`、`stop_started`。
    async def stop_active_work() -> None:
        stop_started.set()
        await allow_stop.wait()

    # 函数说明：test_deletion_blocks_new_conversation_execution.delete
    # 用途：删除回归测试与测试辅助，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`coordinator.deletion`。
    # 闭包依赖：从外层读取 `coordinator`。
    async def delete() -> None:
        async with coordinator.deletion(
            "conversation-a",
            stop_active_work=stop_active_work,
        ):
            pass

    deletion = asyncio.create_task(delete())
    await stop_started.wait()
    with pytest.raises(KeyError, match="正在删除"):
        async with coordinator.execution("conversation-a"):
            pass
    allow_stop.set()
    await deletion
