
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactService
from app.domain.automation import AutomationScheduler
from app.domain.task import FileTaskStore
from app.records.evidence import SQLiteEvidenceStore
from app.records.trace import SQLiteTraceStore
from app.runtime.agent.post_run_processor import PostRunProcessor
from app.runtime.checkpoint import SQLiteCheckpointStore
from app.runtime.run import RunManager, SQLiteRunStore
from app.safety.approval import SQLiteApprovalStore
from app.tools import ApprovalScope, PermissionRuleStore

from .coordinator import ConversationOperationCoordinator
from .store import SQLiteConversationStore


class ConversationDeletionResult(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    deleted: bool = True
    cancelled_runs: int = Field(default=0, ge=0)
    cancelled_post_run_jobs: int = Field(default=0, ge=0)
    deleted_automations: int = Field(default=0, ge=0)
    deleted_approvals: int = Field(default=0, ge=0)
    deleted_permission_rules: int = Field(default=0, ge=0)
    deleted_tasks: int = Field(default=0, ge=0)
    deleted_artifacts: int = Field(default=0, ge=0)
    deleted_evidence: int = Field(default=0, ge=0)
    deleted_checkpoints: int = Field(default=0, ge=0)
    deleted_traces: int = Field(default=0, ge=0)
    deleted_runs: int = Field(default=0, ge=0)
    cancelled_meas: int = Field(default=0, ge=0)
    deleted_meas: int = Field(default=0, ge=0)
    audit_records_retained: bool = False


class ConversationLifecycleService:

    # 函数说明：ConversationLifecycleService.__init__
    # 用途：初始化 ConversationLifecycleService；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   conversation_store：会话历史存储，类型 `SQLiteConversationStore`。
    #   operation_coordinator：`operation_coordinator`输入或配置值，类型
    # `ConversationOperationCoordinator`。
    #   run_manager：运行管理者输入或配置值，类型 `RunManager`。
    #   run_store：运行持久化存储依赖，类型 `SQLiteRunStore`。
    #   checkpoint_store：运行检查点存储，类型 `SQLiteCheckpointStore`。
    #   trace_store：执行轨迹存储，类型 `SQLiteTraceStore`。
    #   evidence_store：原始证据持久化存储依赖，类型 `SQLiteEvidenceStore`。
    #   approval_store：审批记录存储，类型 `SQLiteApprovalStore`。
    #   artifact_service：交付物输入或配置值，类型 `ArtifactService`。
    #   task_store：任务存储，类型 `FileTaskStore`。
    #   permission_rule_store：权限规则持久化存储依赖，类型 `PermissionRuleStore`。
    #   automation_scheduler：自动化任务输入或配置值，类型 `AutomationScheduler`。
    #   post_run_processor：运行输入或配置值，类型 `PostRunProcessor`。
    #   mea_runner：`mea_runner`输入或配置值，类型 `Any | None`；默认 `None`。
    #   mea_store：`mea`持久化存储依赖，类型 `Any | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._conversation_store`、`self._operations`、
    # `self._run_manager`、`self._run_store`、`self._checkpoint_store`、
    # `self._trace_store`、`self._evidence_store`、`self._approval_store` 等 15 个字段。
    def __init__(
        self,
        conversation_store: SQLiteConversationStore,
        operation_coordinator: ConversationOperationCoordinator,
        run_manager: RunManager,
        run_store: SQLiteRunStore,
        checkpoint_store: SQLiteCheckpointStore,
        trace_store: SQLiteTraceStore,
        evidence_store: SQLiteEvidenceStore,
        approval_store: SQLiteApprovalStore,
        artifact_service: ArtifactService,
        task_store: FileTaskStore,
        permission_rule_store: PermissionRuleStore,
        automation_scheduler: AutomationScheduler,
        post_run_processor: PostRunProcessor,
        *,
        mea_runner: Any | None = None,
        mea_store: Any | None = None,
        run_message_store: Any | None = None,
    ) -> None:
        self._conversation_store = conversation_store
        self._operations = operation_coordinator
        self._run_manager = run_manager
        self._run_store = run_store
        self._checkpoint_store = checkpoint_store
        self._trace_store = trace_store
        self._evidence_store = evidence_store
        self._approval_store = approval_store
        self._artifact_service = artifact_service
        self._task_store = task_store
        self._permission_rule_store = permission_rule_store
        self._automation_scheduler = automation_scheduler
        self._post_run_processor = post_run_processor
        self._mea_runner = mea_runner
        self._mea_store = mea_store
        self._run_message_store = run_message_store

    # 函数说明：ConversationLifecycleService.delete
    # 用途：先停止会话相关运行，再删除消息及其关联资源。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `ConversationDeletionResult | None`；按分支返回 `None`；
    # `ConversationDeletionResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._operations.deletion` →
    # `self._run_store.list_for_conversation` → `self._task_store.list_for_conversation`
    #  → `self._artifact_service.delete_for_conversation` →
    # `self._task_store.delete_for_conversation` →
    # `self._permission_rule_store.remove_scope`；另有 9 个调用点。
    # 分支与异常：
    #   当 `not normalized` 时，抛出 `ValueError('conversation_id cannot be empty')`。
    #   当 `await self._conversation_store.get(normalized) is None` 时，返回 `None`。
    #   当 `not deleted` 时，抛出 `RuntimeError(f'删除会话失败：{normalized}')`。
    async def delete(
        self,
        conversation_id: str,
    ) -> ConversationDeletionResult | None:

        """先停止会话相关运行，再删除消息及其关联资源。"""
        normalized = conversation_id.strip()
        if not normalized:
            raise ValueError("conversation_id cannot be empty")
        if await self._conversation_store.get(normalized) is None:
            return None

        deleted_automations = 0
        cancelled_runs = 0
        cancelled_post_run_jobs = 0
        cancelled_meas = 0

        # 函数说明：ConversationLifecycleService.delete.stop_active_work
        # 用途：停止活跃项，供会话生命周期与历史持久化使用。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：
        # `self._mea_runner.cancel_for_conversation` →
        # `self._automation_scheduler.delete_for_conversation` →
        # `self._run_manager.cancel_for_conversation` →
        # `self._post_run_processor.cancel_for_conversation` →
        # `self._approval_store.cancel_pending_for_conversation`。
        # 闭包依赖：从外层读取 `normalized`。
        async def stop_active_work() -> None:
            nonlocal deleted_automations
            nonlocal cancelled_runs
            nonlocal cancelled_post_run_jobs
            nonlocal cancelled_meas

            # 先停长任务循环，否则它会在子 Run 被取消后继续启动新的子 Run
            if self._mea_runner is not None:
                cancelled_meas = await self._mea_runner.cancel_for_conversation(normalized)
            deleted_automations = (
                await self._automation_scheduler.delete_for_conversation(normalized)
            )
            cancelled_runs = len(
                await self._run_manager.cancel_for_conversation(normalized)
            )
            cancelled_post_run_jobs = (
                await self._post_run_processor.cancel_for_conversation(normalized)
            )
            await self._approval_store.cancel_pending_for_conversation(normalized)

        async with self._operations.deletion(
            normalized,
            stop_active_work=stop_active_work,
        ):
            if await self._conversation_store.get(normalized) is None:
                return None

            runs = await self._run_store.list_for_conversation(normalized)
            run_ids = tuple(run.id for run in runs)
            owned_tasks = await self._task_store.list_for_conversation(normalized)
            task_ids = tuple(task.id for task in owned_tasks)
            artifact_ids = await self._artifact_service.delete_for_conversation(
                normalized,
                run_ids=run_ids,
                task_ids=task_ids,
            )
            deleted_task_ids = (
                await self._task_store.delete_for_conversation(normalized)
            )

            deleted_permission_rules = await self._permission_rule_store.remove_scope(
                ApprovalScope.CONVERSATION,
                normalized,
            )
            for run_id in run_ids:
                deleted_permission_rules += (
                    await self._permission_rule_store.remove_scope(
                        ApprovalScope.RUN,
                        run_id,
                    )
                )

            deleted_approvals = (
                await self._approval_store.delete_for_conversation(
                    normalized,
                    run_ids=run_ids,
                )
            )
            deleted_evidence = (
                await self._evidence_store.delete_for_conversation(normalized)
            )
            deleted_checkpoints = (
                await self._checkpoint_store.delete_for_conversation(
                    normalized,
                    run_ids=run_ids,
                )
            )
            deleted_traces = (
                await self._trace_store.delete_for_conversation(
                    normalized,
                    run_ids=run_ids,
                )
            )
            deleted_meas = (
                await self._mea_store.delete_for_conversation(normalized)
                if self._mea_store is not None
                else 0
            )
            if self._run_message_store is not None:
                await self._run_message_store.delete_for_conversation(
                    normalized,
                    run_ids=run_ids,
                )
            deleted_runs = await self._run_store.delete_for_conversation(normalized)
            self._run_manager.forget_results(run_ids)
            deleted = await self._conversation_store.delete(normalized)
            if not deleted:
                raise RuntimeError(f"删除会话失败：{normalized}")

        return ConversationDeletionResult(
            conversation_id=normalized,
            cancelled_runs=cancelled_runs,
            cancelled_post_run_jobs=cancelled_post_run_jobs,
            deleted_automations=deleted_automations,
            deleted_approvals=deleted_approvals,
            deleted_permission_rules=deleted_permission_rules,
            deleted_tasks=len(deleted_task_ids),
            deleted_artifacts=len(artifact_ids),
            deleted_evidence=deleted_evidence,
            deleted_checkpoints=deleted_checkpoints,
            deleted_traces=deleted_traces,
            deleted_runs=deleted_runs,
            cancelled_meas=cancelled_meas,
            deleted_meas=deleted_meas,
        )
__all__ = ["ConversationDeletionResult", "ConversationLifecycleService"]
