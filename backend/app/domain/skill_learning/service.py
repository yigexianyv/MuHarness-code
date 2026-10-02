
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.skills import SkillScope, SkillStore
from app.domain.task import FileTaskStore, Task, TaskStatus
from app.models.registry import ModelAdapterRegistry
from app.models.types import ModelUsage, add_model_usage
from app.records.trace.store import SQLiteTraceStore

from .config import SkillLearningSettings
from .distiller import DistillationOutcome, ProcedureDistiller
from .evidence import TraceEvidenceBuilder
from .miner import PatternMiningOutcome, TaskPatternMiner
from .models import (
    SkillCandidate,
    SkillCandidateAction,
    SkillCandidateStatus,
    TaskCard,
    TaskPatternCluster,
)
from .store import InflightBatch, MiningWatermark, SkillCandidateStore
from .trace_selector import TaskTraceSelector

logger = logging.getLogger("muharness.skill_learning.service")

_MAX_COMPLETED_TASKS = 1_000_000


class DistillationRecord(BaseModel):

    model_config = ConfigDict(extra="forbid")

    cluster_name: str
    action: str | None = None
    reason: str | None = None
    proposed_name: str | None = None
    existing_skill_name: str | None = None
    related_skill_names: tuple[str, ...] = ()
    raw_output: str | None = None
    adjudication_raw_output: str | None = None
    error: str | None = None


class SkillLearningOutcome(BaseModel):

    model_config = ConfigDict(extra="forbid")

    triggered: bool = False
    skipped_reason: str | None = None
    pending_count: int = 0
    scanned_task_count: int = 0
    cluster_count: int = 0
    clusters: tuple[TaskPatternCluster, ...] = ()
    pattern_mining_raw_output: str | None = None
    candidate_count: int = 0
    distillations: tuple[DistillationRecord, ...] = ()
    usage: ModelUsage = Field(default_factory=ModelUsage)
    pattern_mining_calls: int = 0
    distillation_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    pattern_mining_duration_ms: float = 0.0
    distillation_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    error: str | None = None


class SkillLearningService:

    # 函数说明：SkillLearningService.__init__
    # 用途：初始化 SkillLearningService；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   task_store：任务存储，类型 `FileTaskStore`。
    #   trace_store：执行轨迹存储，类型 `SQLiteTraceStore`。
    #   skill_store：技能存储，类型 `SkillStore`。
    #   candidate_store：候选持久化存储依赖，类型 `SkillCandidateStore`。
    #   registry：工具、模型或能力注册表，类型 `ModelAdapterRegistry`。
    #   settings：业务或模型设置，类型 `SkillLearningSettings | None`；默认 `None`。
    #   default_provider：未指定服务商时的默认值，类型 `str | None`；默认 `None`。
    #   default_model：未指定模型时的默认值，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillLearningSettings` →
    # `TaskPatternMiner` → `ProcedureDistiller` → `TraceEvidenceBuilder` →
    # `TaskTraceSelector` → `_BatchPlanner`；另有 2 个调用点。
    # 副作用与资源：
    #   更新对象字段：`self.task_store`、`self.trace_store`、`self.skill_store`、
    # `self.candidate_store`、`self._registry`、`self.settings`、
    # `self._default_provider`、`self._default_model` 等 15 个字段。
    def __init__(
        self,
        task_store: FileTaskStore,
        trace_store: SQLiteTraceStore,
        skill_store: SkillStore,
        candidate_store: SkillCandidateStore,
        registry: ModelAdapterRegistry,
        *,
        settings: SkillLearningSettings | None = None,
        default_provider: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self.task_store = task_store
        self.trace_store = trace_store
        self.skill_store = skill_store
        self.candidate_store = candidate_store
        self._registry = registry
        self.settings = settings or SkillLearningSettings()
        self._default_provider = default_provider
        self._default_model = default_model
        self.miner = TaskPatternMiner(
            registry,
            settings=self.settings,
            default_provider=default_provider,
            default_model=default_model,
        )
        self.distiller = ProcedureDistiller(
            registry,
            settings=self.settings,
            default_provider=default_provider,
            default_model=default_model,
        )
        self.evidence_builder = TraceEvidenceBuilder(self.settings)
        self.trace_selector = TaskTraceSelector()
        self._batch_planner = _BatchPlanner()
        self._pipeline = _MiningPipeline()
        self._candidate_review = _CandidateReview()


    # 函数说明：SkillLearningService.maybe_run_mining
    # 用途：根据任务证据决定是否挖掘候选技能，并记录蒸馏结果。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
    #   model：模型名称，类型 `str | None`；默认 `None`。
    # 返回：类型 `SkillLearningOutcome`；按分支返回
    # `SkillLearningOutcome(skipped_reason='disabled')`；`SkillLearningOutcome(…)`；
    # `report.to_outcome(planned, error=error)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillLearningOutcome` →
    # `self._batch_planner.prepare` → `self._batch_planner.complete` →
    # `self._pipeline.execute` → `self._batch_planner.fail` → `report.to_outcome`。
    # 分支与异常：
    #   当 `not self.settings.skill_learning_enabled` 时，返回
    # `SkillLearningOutcome(skipped_reason='disabled')`。
    #   当 `isinstance(planned, _WaitingBatch)` 时，返回 `SkillLearningOutcome(…)`。
    #   `not planned.cards` 分支在完成前置处理后返回 `SkillLearningOutcome(…)`。
    async def maybe_run_mining(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> SkillLearningOutcome:

        """根据任务证据决定是否挖掘候选技能，并记录蒸馏结果。"""
        if not self.settings.skill_learning_enabled:
            return SkillLearningOutcome(skipped_reason="disabled")
        planned = await self._batch_planner.prepare(self)
        if isinstance(planned, _WaitingBatch):
            return SkillLearningOutcome(
                pending_count=planned.pending_count,
                skipped_reason="batch_not_ready",
            )
        if not planned.cards:
            await self._batch_planner.complete(self, planned, clear_error=False)
            return SkillLearningOutcome(
                triggered=True,
                pending_count=len(planned.new_pending),
                scanned_task_count=0,
                error="no readable completed tasks to scan",
            )

        report = await self._pipeline.execute(self, planned)
        error = report.error
        if error:
            error = await self._batch_planner.fail(
                self, planned, error=error, stage=report.failure_stage
            )
        else:
            await self._batch_planner.complete(self, planned)
        return report.to_outcome(planned, error=error)

    # 函数说明：SkillLearningService._load_task_events
    # 用途：加载任务事件序列，供技能候选提炼与审核使用。
    # 参数：
    #   task：当前任务记录，类型 `Task`。
    # 返回：类型 `tuple`；按分支返回 `()`；`self.trace_selector.select(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.trace_store.load_events` →
    # `self.trace_selector.select`。
    # 分支与异常：
    #   捕获 `(KeyError, ValueError, OSError)` 后，跳过当前循环项，继续处理后续项。
    #   当 `not run_events` 时，返回 `()`。
    async def _load_task_events(self, task: Task) -> tuple:

        run_events: dict[str, tuple] = {}
        for run_id in task.run_ids:
            try:
                run_events[run_id] = await self.trace_store.load_events(run_id)
            except (KeyError, ValueError, OSError):
                continue
        if not run_events:
            return ()
        return self.trace_selector.select(
            task,
            run_events,
            max_events=self.settings.skill_learning_max_events_per_task,
        )


    # 函数说明：SkillLearningService.list_candidates
    # 用途：列出候选集合，供技能候选提炼与审核使用。
    # 参数：
    #   status：目标状态，类型 `SkillCandidateStatus | None`；默认 `None`。
    # 返回：类型 `tuple[SkillCandidate, ...]`；返回
    # `await self.candidate_store.list(status=status)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.candidate_store.list`。
    async def list_candidates(
        self,
        *,
        status: SkillCandidateStatus | None = None,
    ) -> tuple[SkillCandidate, ...]:
        return await self.candidate_store.list(status=status)

    # 函数说明：SkillLearningService.get_candidate
    # 用途：获取候选，供技能候选提炼与审核使用。
    # 参数：
    #   candidate_id：待审核技能候选标识，类型 `str`。
    # 返回：类型 `SkillCandidate | None`；返回
    # `await self.candidate_store.get(candidate_id)`。
    async def get_candidate(self, candidate_id: str) -> SkillCandidate | None:
        return await self.candidate_store.get(candidate_id)


    # 函数说明：SkillLearningService.accept
    # 用途：接受候选技能，创建或更新托管技能并记录决策。
    # 参数：
    #   candidate_id：待审核技能候选标识，类型 `str`。
    #   scope：记忆、规则或查询作用域，类型 `str | None`；默认 `None`。
    # 返回：类型 `tuple[SkillCandidate, Path | None]`；返回
    # `await self._candidate_review.accept(self, candidate_id, scope=scope)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._candidate_review.accept`。
    async def accept(
        self,
        candidate_id: str,
        *,
        scope: str | None = None,
    ) -> tuple[SkillCandidate, Path | None]:

        """接受候选技能，创建或更新托管技能并记录决策。"""
        return await self._candidate_review.accept(self, candidate_id, scope=scope)

    # 函数说明：SkillLearningService.reject
    # 用途：拒绝候选技能并保存其最终状态。
    # 参数：
    #   candidate_id：待审核技能候选标识，类型 `str`。
    # 返回：类型 `SkillCandidate`；返回
    # `await self._candidate_review.reject(self, candidate_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._candidate_review.reject`。
    async def reject(self, candidate_id: str) -> SkillCandidate:

        """拒绝候选技能并保存其最终状态。"""
        return await self._candidate_review.reject(self, candidate_id)

    # 函数说明：SkillLearningService._create_skill
    # 用途：创建技能，供技能候选提炼与审核使用。
    # 参数：
    #   candidate：候选记录，类型 `SkillCandidate`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    # 返回：类型 `Path`；返回
    # `await self._candidate_review.create_skill(self, candidate, scope)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._candidate_review.create_skill`。
    async def _create_skill(
        self,
        candidate: SkillCandidate,
        scope: SkillScope,
    ) -> Path:
        return await self._candidate_review.create_skill(self, candidate, scope)

    # 函数说明：SkillLearningService._update_skill
    # 用途：更新技能，供技能候选提炼与审核使用。
    # 参数：
    #   candidate：候选记录，类型 `SkillCandidate`。
    # 返回：类型 `Path`；返回
    # `await self._candidate_review.update_skill(self, candidate)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self._candidate_review.update_skill`。
    async def _update_skill(self, candidate: SkillCandidate) -> Path:
        return await self._candidate_review.update_skill(self, candidate)


    # 函数说明：SkillLearningService.render_candidate_details
    # 用途：生成展示文本候选，供技能候选提炼与审核使用。
    # 参数：
    #   candidate：候选记录，类型 `SkillCandidate`。
    # 返回：类型 `str`；返回 `'\n'.join(lines)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`candidate.action.value.upper`。
    def render_candidate_details(self, candidate: SkillCandidate) -> str:

        lines = [
            f"Proposed Skill: {candidate.proposed_name}",
            f"Origin: {candidate.origin.value}",
            f"Action: {candidate.action.value.upper()}",
            f"Status: {candidate.status.value}",
            "",
            f"Why: {candidate.reason}",
            "",
            f"Source Tasks: {' '.join(candidate.source_task_ids) or '（无）'}",
            f"Source Runs: {' '.join(candidate.source_run_ids) or '（无）'}",
            f"Source Conversation: {candidate.source_conversation_id or '（无）'}",
            "",
            "Common Procedure:",
        ]
        for index, step in enumerate(candidate.procedure, 1):
            lines.append(f"{index}. {step}")
        if candidate.pitfalls:
            lines.append("")
            lines.append("Repeated Problems:")
            lines.extend(f"- {item}" for item in candidate.pitfalls)
        if candidate.verification:
            lines.append("")
            lines.append("Verification:")
            lines.extend(f"- {item}" for item in candidate.verification)
        if candidate.evidence_summary:
            lines.append("")
            lines.append("Evidence Summary:")
            lines.append(candidate.evidence_summary)
        return "\n".join(lines)




@dataclass(frozen=True, slots=True)
class _WaitingBatch:
    pending_count: int


@dataclass(frozen=True, slots=True)
class _BatchPlan:
    # Keep the initially loaded watermark, not either preparation write.
    watermark: MiningWatermark
    processed: frozenset[str]
    new_pending: tuple[str, ...]
    inflight: InflightBatch
    scan_ids: tuple[str, ...]
    by_id: dict[str, Task]
    cards: tuple[TaskCard, ...]

    # 函数说明：_BatchPlan.processed_after_scan
    # 用途：返回 `tuple(sorted(self.processed | set(self.scan_ids)))`，提供 _BatchPlan
    # 的派生值。
    # 返回：类型 `tuple[str, ...]`；返回
    # `tuple(sorted(self.processed | set(self.scan_ids)))`。
    @property
    def processed_after_scan(self) -> tuple[str, ...]:
        return tuple(sorted(self.processed | set(self.scan_ids)))


class _BatchPlanner:
    """Own queue selection and the existing watermark transition boundaries."""

    # 函数说明：_BatchPlanner.prepare
    # 用途：准备_BatchPlanner，供技能候选提炼与审核使用。
    # 参数：
    #   service：业务服务依赖，类型 `SkillLearningService`。
    # 返回：类型 `_BatchPlan | _WaitingBatch`；按分支返回
    # `_WaitingBatch(pending_count=len(pending))`；`_BatchPlan(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`service.task_store.list` →
    # `service.candidate_store.load_watermark` →
    # `service.candidate_store.save_watermark` → `_WaitingBatch` → `InflightBatch` →
    # `uuid4`；另有 4 个调用点。
    # 分支与异常：
    #   当 `len(pending) < service.settings.skill_learning_batch_size` 时，返回
    # `_WaitingBatch(pending_count=len(pending))`。
    async def prepare(
        self, service: SkillLearningService
    ) -> _BatchPlan | _WaitingBatch:
        completed = await service.task_store.list(
            status=TaskStatus.COMPLETED,
            limit=_MAX_COMPLETED_TASKS,
        )
        by_id = {task.id: task for task in completed}
        watermark = await service.candidate_store.load_watermark()
        processed = set(watermark.processed_task_ids)
        pending = list(watermark.pending_task_ids)
        inflight = watermark.inflight

        if inflight is None:
            known = processed | set(pending)
            new_ids = [task.id for task in completed if task.id not in known]
            for task_id in new_ids:
                if task_id not in pending:
                    pending.append(task_id)
            await service.candidate_store.save_watermark(
                watermark.model_copy(update={"pending_task_ids": tuple(pending)})
            )
            if len(pending) < service.settings.skill_learning_batch_size:
                return _WaitingBatch(pending_count=len(pending))
            scan_ids = tuple(
                pending[: service.settings.skill_learning_max_tasks_per_scan]
            )
            scan_set = set(scan_ids)
            new_pending = tuple(
                task_id for task_id in pending if task_id not in scan_set
            )
            inflight = InflightBatch(
                batch_id=uuid4().hex,
                task_ids=scan_ids,
                started_at=datetime.now(UTC),
            )
            await service.candidate_store.save_watermark(
                watermark.model_copy(
                    update={
                        "pending_task_ids": new_pending,
                        "inflight": inflight,
                        "last_error": None,
                    }
                )
            )
        else:
            scan_ids = inflight.task_ids
            new_pending = tuple(pending)

        cards = tuple(
            _to_card(by_id[task_id])
            for task_id in scan_ids
            if task_id in by_id
        )
        return _BatchPlan(
            watermark=watermark,
            processed=frozenset(processed),
            new_pending=new_pending,
            inflight=inflight,
            scan_ids=scan_ids,
            by_id=by_id,
            cards=cards,
        )

    # 函数说明：_BatchPlanner.complete
    # 用途：完成_BatchPlanner，供技能候选提炼与审核使用。
    # 参数：
    #   service：业务服务依赖，类型 `SkillLearningService`。
    #   planned：`planned`输入或配置值，类型 `_BatchPlan`。
    #   clear_error：错误输入或配置值，类型 `bool`；默认 `True`。
    #   error：异常或错误信息，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` →
    # `service.candidate_store.save_watermark`。
    async def complete(
        self,
        service: SkillLearningService,
        planned: _BatchPlan,
        *,
        clear_error: bool = True,
        error: str | None = None,
    ) -> None:
        updates = {
            "processed_task_ids": planned.processed_after_scan,
            "pending_task_ids": planned.new_pending,
            "inflight": None,
            "last_mining_at": datetime.now(UTC),
        }
        if clear_error:
            updates["last_error"] = error
        await service.candidate_store.save_watermark(
            planned.watermark.model_copy(update=updates)
        )

    # 函数说明：_BatchPlanner.fail
    # 用途：记录失败_BatchPlanner，供技能候选提炼与审核使用。
    # 参数：
    #   service：业务服务依赖，类型 `SkillLearningService`。
    #   planned：传给 `self.complete` 的输入，类型 `_BatchPlan`。
    #   error：异常或错误信息，类型 `str`。
    #   stage：`stage`输入或配置值，类型 `str`。
    # 返回：类型 `str`；按分支返回 `f'{stage} failed after {service.settings.
    # skill_learning_max_attempts} attempts: {error}'`；`error`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.complete` →
    # `service.candidate_store.save_watermark` → `logger.warning`。
    # 分支与异常：
    #   `new_attempt >= service.settings.skill_learning_max_attempts` 分支在完成前置处理
    # 后返回 `f'{stage} failed after {…`。
    async def fail(
        self,
        service: SkillLearningService,
        planned: _BatchPlan,
        *,
        error: str,
        stage: str,
    ) -> str:
        new_attempt = planned.inflight.attempt + 1
        if new_attempt >= service.settings.skill_learning_max_attempts:
            await self.complete(service, planned, error=error)
            return (
                f"{stage} failed after "
                f"{service.settings.skill_learning_max_attempts} attempts: {error}"
            )
        await service.candidate_store.save_watermark(
            planned.watermark.model_copy(
                update={
                    "inflight": planned.inflight.model_copy(
                        update={"attempt": new_attempt, "last_error": error}
                    ),
                    "last_error": error,
                }
            )
        )
        logger.warning(
            f"{stage} failed (attempt %s/%s): %s",
            new_attempt,
            service.settings.skill_learning_max_attempts,
            error,
        )
        return error


@dataclass(frozen=True, slots=True)
class _PipelineReport:
    mining: PatternMiningOutcome
    usage: ModelUsage
    pattern_duration: float
    total_duration: float
    error: str | None = None
    failure_stage: str = "pattern mining"
    created: tuple[SkillCandidate, ...] = ()
    distillations: tuple[DistillationRecord, ...] = ()
    distill_calls: int = 0
    distill_duration: float = 0.0

    # 函数说明：_PipelineReport.to_outcome
    # 用途：处理技能候选提炼与审核中的 `to_outcome` 数据；结果及边界条件见下方说明。
    # 参数：
    #   planned：`planned`输入或配置值，类型 `_BatchPlan`。
    #   error：异常或错误信息，类型 `str | None`。
    # 返回：类型 `SkillLearningOutcome`；按分支返回
    # `SkillLearningOutcome(**common, error=error)`；
    # `SkillLearningOutcome(**common, cluster_count=0, clusters=())`；
    # `SkillLearningOutcome(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillLearningOutcome`。
    # 分支与异常：
    #   `self.failure_stage == 'pattern mining'` 分支在完成前置处理后返回
    # `SkillLearningOutcome(**common, cluster_count=0, clusters=())`。
    #   当 `self.error` 时，返回 `SkillLearningOutcome(**common, error=error)`。
    def to_outcome(
        self, planned: _BatchPlan, *, error: str | None
    ) -> SkillLearningOutcome:
        common = {
            "triggered": True,
            "pending_count": len(planned.new_pending),
            "scanned_task_count": len(planned.cards),
            "pattern_mining_raw_output": self.mining.raw_output,
            "usage": self.usage,
            "pattern_mining_calls": 1,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "pattern_mining_duration_ms": self.pattern_duration,
            "total_duration_ms": self.total_duration,
        }
        # Preserve explicit-field sets as well as values for exclude_unset users.
        if self.failure_stage == "pattern mining":
            if self.error:
                return SkillLearningOutcome(**common, error=error)
            return SkillLearningOutcome(**common, cluster_count=0, clusters=())
        return SkillLearningOutcome(
            **common,
            cluster_count=len(self.mining.clusters),
            clusters=self.mining.clusters,
            candidate_count=len(self.created),
            distillations=self.distillations,
            distillation_calls=self.distill_calls,
            distillation_duration_ms=self.distill_duration,
            error=error,
        )


class _MiningPipeline:
    """Run model stages serially and retain candidates before batch finalization."""

    # 函数说明：_MiningPipeline.execute
    # 用途：执行_MiningPipeline，供技能候选提炼与审核使用。
    # 参数：
    #   service：业务服务依赖，类型 `SkillLearningService`。
    #   planned：`planned`输入或配置值，类型 `_BatchPlan`。
    # 返回：类型 `_PipelineReport`；返回 `_PipelineReport(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`service.miner.mine` →
    # `_PipelineReport` → `ModelUsage` → `service.list_candidates` →
    # `service.skill_store.catalog` → `service.candidate_store.find_duplicate_source`；
    # 另有 8 个调用点。
    # 分支与异常：
    #   当 `mining.error or not mining.clusters` 时，返回 `_PipelineReport(…)`。
    #   当 `await service.candidate_store.find_duplicate_source(…)` 时，跳过当前循环项。
    #   当 `task is None` 时，跳过当前循环项。
    #   `distill.error` 分支在完成前置处理后跳过当前循环项。
    async def execute(
        self, service: SkillLearningService, planned: _BatchPlan
    ) -> _PipelineReport:
        mining: PatternMiningOutcome = await service.miner.mine(planned.cards)
        usage = mining.usage
        pattern_duration = mining.duration_ms
        if mining.error or not mining.clusters:
            return _PipelineReport(
                mining=mining,
                usage=usage,
                pattern_duration=pattern_duration,
                total_duration=pattern_duration,
                error=mining.error,
            )

        created: list[SkillCandidate] = []
        errors: list[str] = []
        distillations: list[DistillationRecord] = []
        distill_usage = ModelUsage()
        distill_duration = 0.0
        distill_calls = 0
        pending_candidates = await service.list_candidates(
            status=SkillCandidateStatus.PENDING
        )
        catalog = await service.skill_store.catalog()
        for cluster in mining.clusters:
            if await service.candidate_store.find_duplicate_source(cluster.task_ids):
                continue
            evidence_map: dict[str, str] = {}
            run_ids_map: dict[str, tuple[str, ...]] = {}
            for task_id in cluster.task_ids:
                task = planned.by_id.get(task_id)
                if task is None:
                    continue
                run_ids_map[task_id] = task.run_ids
                events = await service._load_task_events(task)
                evidence_map[task_id] = service.evidence_builder.build(task, events)
            logger.debug(
                "cluster %s evidence chars: %s",
                cluster.pattern_name,
                {k: len(v) for k, v in evidence_map.items()},
            )
            distill: DistillationOutcome = await service.distiller.distill(
                cluster,
                evidence=evidence_map,
                run_ids=run_ids_map,
                catalog=catalog,
                pending_candidates=pending_candidates,
                skill_loader=service.skill_store.load,
            )
            distillations.append(
                DistillationRecord(
                    cluster_name=cluster.pattern_name,
                    action=distill.action,
                    reason=distill.reason,
                    proposed_name=distill.proposed_name,
                    existing_skill_name=distill.existing_skill_name,
                    related_skill_names=distill.related_skill_names,
                    raw_output=distill.raw_output,
                    adjudication_raw_output=distill.adjudication_raw_output,
                    error=distill.error,
                )
            )
            distill_calls += distill.model_call_count
            distill_usage = _add_usage(distill_usage, distill.usage)
            distill_duration += distill.duration_ms
            if distill.error:
                errors.append(f"{cluster.pattern_name}: {distill.error}")
                continue
            if distill.candidate is None:
                continue
            if _pending_name_exists(
                pending_candidates,
                distill.candidate.proposed_name,
            ):
                continue
            await service.candidate_store.create(distill.candidate)
            created.append(distill.candidate)
            pending_candidates = pending_candidates + (distill.candidate,)

        return _PipelineReport(
            mining=mining,
            usage=_add_usage(usage, distill_usage),
            pattern_duration=pattern_duration,
            total_duration=pattern_duration + distill_duration,
            error="; ".join(errors) or None,
            failure_stage="distillation",
            created=tuple(created),
            distillations=tuple(distillations),
            distill_calls=distill_calls,
            distill_duration=distill_duration,
        )


class _CandidateReview:
    """Keep human decisions after successful materialization of formal skills."""

    # 函数说明：_CandidateReview.accept
    # 用途：接受_CandidateReview，供技能候选提炼与审核使用。
    # 参数：
    #   service：业务服务依赖，类型 `SkillLearningService`。
    #   candidate_id：待审核技能候选标识，类型 `str`。
    #   scope：记忆、规则或查询作用域，类型 `str | None`。
    # 返回：类型 `tuple[SkillCandidate, Path | None]`；返回 `(updated, target)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._pending_candidate` →
    # `_resolve_scope` → `service._create_skill` → `service._update_skill` →
    # `datetime.now` → `service.candidate_store.update`。
    async def accept(
        self,
        service: SkillLearningService,
        candidate_id: str,
        *,
        scope: str | None,
    ) -> tuple[SkillCandidate, Path | None]:
        candidate = await self._pending_candidate(service, candidate_id)
        resolved_scope = _resolve_scope(
            scope or service.settings.skill_learning_default_scope
        )
        target: Path | None = None
        if candidate.action is SkillCandidateAction.CREATE:
            target = await service._create_skill(candidate, resolved_scope)
        else:
            target = await service._update_skill(candidate)
        updated = candidate.model_copy(
            update={
                "status": SkillCandidateStatus.ACCEPTED,
                "reviewed_at": datetime.now(UTC),
            }
        )
        await service.candidate_store.update(updated)
        return updated, target

    # 函数说明：_CandidateReview.reject
    # 用途：拒绝_CandidateReview，供技能候选提炼与审核使用。
    # 参数：
    #   service：业务服务依赖，类型 `SkillLearningService`。
    #   candidate_id：待审核技能候选标识，类型 `str`。
    # 返回：类型 `SkillCandidate`；返回 `updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._pending_candidate` →
    # `datetime.now` → `service.candidate_store.update`。
    async def reject(
        self, service: SkillLearningService, candidate_id: str
    ) -> SkillCandidate:
        candidate = await self._pending_candidate(service, candidate_id)
        updated = candidate.model_copy(
            update={
                "status": SkillCandidateStatus.REJECTED,
                "reviewed_at": datetime.now(UTC),
            }
        )
        await service.candidate_store.update(updated)
        return updated

    # 函数说明：_CandidateReview._pending_candidate
    # 用途：在技能候选提炼与审核中处理 `_pending_candidate`，通过
    # `service.candidate_store.get` 完成首个内部处理步骤。
    # 参数：
    #   service：业务服务依赖，类型 `SkillLearningService`。
    #   candidate_id：待审核技能候选标识，类型 `str`。
    # 返回：类型 `SkillCandidate`；返回 `candidate`。
    # 分支与异常：
    #   当 `candidate is None` 时，抛出
    # `KeyError(f'candidate not found: {candidate_id}')`。
    #   当 `candidate.status is not SkillCandidateStatus.PENDING` 时，抛出
    # `ValueError(…)`。
    async def _pending_candidate(
        self, service: SkillLearningService, candidate_id: str
    ) -> SkillCandidate:
        candidate = await service.candidate_store.get(candidate_id)
        if candidate is None:
            raise KeyError(f"candidate not found: {candidate_id}")
        if candidate.status is not SkillCandidateStatus.PENDING:
            raise ValueError(f"candidate is not pending: {candidate.status.value}")
        return candidate

    # 函数说明：_CandidateReview.create_skill
    # 用途：创建技能，供技能候选提炼与审核使用。
    # 参数：
    #   service：业务服务依赖，类型 `SkillLearningService`。
    #   candidate：候选记录，类型 `SkillCandidate`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    # 返回：类型 `Path`；返回 `installed.metadata.location`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`service.skill_store.install` →
    # `_render_candidate_body`。
    async def create_skill(
        self,
        service: SkillLearningService,
        candidate: SkillCandidate,
        scope: SkillScope,
    ) -> Path:
        installed = await service.skill_store.install(
            name=candidate.proposed_name,
            description=candidate.description,
            instructions=_render_candidate_body(candidate),
            scope=scope,
        )
        return installed.metadata.location

    # 函数说明：_CandidateReview.update_skill
    # 用途：更新技能，供技能候选提炼与审核使用。
    # 参数：
    #   service：业务服务依赖，类型 `SkillLearningService`。
    #   candidate：候选记录，类型 `SkillCandidate`。
    # 返回：类型 `Path`；返回 `updated.metadata.location`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`service.skill_store.update` →
    # `_render_candidate_body`。
    # 分支与异常：
    #   当 `not name` 时，抛出
    # `ValueError('update candidate requires existing_skill_name')`。
    # 副作用与资源：
    #   数据库操作：UPDATE candidate；连接与事务边界以 with/提交语句为准。
    async def update_skill(
        self, service: SkillLearningService, candidate: SkillCandidate
    ) -> Path:
        name = candidate.existing_skill_name
        if not name:
            raise ValueError("update candidate requires existing_skill_name")
        updated = await service.skill_store.update(
            name=name,
            description=candidate.description,
            instructions=_render_candidate_body(candidate),
        )
        return updated.metadata.location


# 函数说明：_add_usage
# 用途：添加用量，供技能候选提炼与审核使用。
# 参数：
#   total：传给 `add_model_usage` 的输入，类型 `ModelUsage`。
#   current：当前值或状态，类型 `ModelUsage`。
# 返回：类型 `ModelUsage`；返回 `add_model_usage(total, current)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`add_model_usage`。
def _add_usage(total: ModelUsage, current: ModelUsage) -> ModelUsage:

    return add_model_usage(total, current)


# 函数说明：_pending_name_exists
# 用途：处理技能候选提炼与审核中的 `_pending_name_exists` 数据；结果及边界条件见下方说明
# 。
# 参数：
#   candidates：候选集合输入或配置值，类型 `Sequence[SkillCandidate]`。
#   proposed_name：名称输入或配置值，类型 `str`。
# 返回：类型 `bool`；按分支返回 `False`；
# `any((candidate.proposed_name == proposed_name for candidate in candidates))`。
# 分支与异常：
#   当 `not proposed_name` 时，返回 `False`。
def _pending_name_exists(
    candidates: Sequence[SkillCandidate],
    proposed_name: str,
) -> bool:

    if not proposed_name:
        return False
    return any(
        candidate.proposed_name == proposed_name for candidate in candidates
    )


# 函数说明：_to_card
# 用途：返回 `TaskCard(…)`，提供 技能候选提炼与审核 的派生值。
# 参数：
#   task：当前任务记录，类型 `Task`。
# 返回：类型 `TaskCard`；返回 `TaskCard(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskCard`。
def _to_card(task: Task) -> TaskCard:
    return TaskCard(
        task_id=task.id,
        title=task.title,
        description=task.description,
        goal=task.goal,
        constraints=task.constraints,
        key_facts=task.key_facts,
        final_steps=tuple(
            step.title for step in task.steps if step.status.value == "done"
        ),
        created_at=task.created_at,
        completed_at=task.completed_at,
        run_count=len(task.run_ids),
    )


# 函数说明：_resolve_scope
# 用途：解析或定位作用域，供技能候选提炼与审核使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `SkillScope`；按分支返回 `SkillScope.PROJECT`；`SkillScope.USER`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().lower`。
# 分支与异常：
#   当 `normalized in ('project', 'project_scope',…` 时，返回 `SkillScope.PROJECT`。
#   当 `normalized in ('user', 'user_scope', SkillScope.USER.value)` 时，返回
# `SkillScope.USER`。
def _resolve_scope(value: str) -> SkillScope:
    normalized = value.strip().lower()
    if normalized in ("project", "project_scope", SkillScope.PROJECT.value):
        return SkillScope.PROJECT
    if normalized in ("user", "user_scope", SkillScope.USER.value):
        return SkillScope.USER
    raise ValueError(f"invalid skill scope: {value}")


# 函数说明：_render_candidate_body
# 用途：生成展示文本候选，供技能候选提炼与审核使用。
# 参数：
#   candidate：候选记录，类型 `SkillCandidate`。
# 返回：类型 `str`；返回 `'\n'.join(lines) + '\n'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`name.replace('-', ' ').title` →
# `name.replace`。
def _render_candidate_body(candidate: SkillCandidate) -> str:

    name = candidate.existing_skill_name or candidate.proposed_name
    title = name.replace("-", " ").title()
    lines = [
        f"# {title}",
        "",
    ]
    lines.append("## Procedure")
    lines.append("")
    for index, step in enumerate(candidate.procedure, 1):
        lines.append(f"{index}. {step}")
    if candidate.pitfalls:
        lines.append("")
        lines.append("## Pitfalls")
        lines.append("")
        lines.extend(f"- {item}" for item in candidate.pitfalls)
    if candidate.verification:
        lines.append("")
        lines.append("## Verification")
        lines.append("")
        lines.extend(f"- {item}" for item in candidate.verification)
    return "\n".join(lines) + "\n"


__all__ = ["SkillLearningOutcome", "SkillLearningService"]
