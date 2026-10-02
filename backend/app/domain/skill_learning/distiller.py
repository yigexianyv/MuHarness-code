
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.skills import Skill, SkillMetadata
from app.models.registry import ModelAdapterRegistry
from app.models.types import ModelUsage, add_model_usage

from ._call import ModelCallResult, call_model, parse_strict_json
from .config import SkillLearningSettings
from .models import (
    SkillCandidate,
    SkillCandidateAction,
    SkillCandidateStatus,
    TaskPatternCluster,
)
from .prompts import (
    _DISTILLATION_PROMPT,
    _OVERLAP_ADJUDICATION_PROMPT,
    _RELEVANCE_PROMPT,
)

_MAX_RELATED_SKILLS = 3
_MAX_SKILL_BODY_CHARS = 4000


class _Distilled(BaseModel):

    model_config = ConfigDict(extra="forbid")

    action: str
    proposed_name: str | None = None
    description: str | None = None
    reason: str | None = None
    procedure: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()
    existing_skill_name: str | None = None

    # 函数说明：_Distilled.valid_action
    # 用途：校验并规范化模型字段 'action'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().lower`。
    # 分支与异常：
    #   当 `normalized not in {'none', 'create', 'update'}` 时，抛出
    # `ValueError(f'invalid action: {value}')`。
    @field_validator("action")
    @classmethod
    def valid_action(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"none", "create", "update"}:
            raise ValueError(f"invalid action: {value}")
        return normalized

    # 函数说明：_Distilled.normalize_list_field
    # 用途：校验并规范化模型字段 'procedure'、'pitfalls'、'verification'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `object`；按分支返回 `()`；`(value,)`；`value`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `()`。
    #   当 `isinstance(value, str)` 时，返回 `(value,)`。
    @field_validator("procedure", "pitfalls", "verification", mode="before")
    @classmethod
    def normalize_list_field(cls, value: object) -> object:

        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return value


class DistillationOutcome(BaseModel):

    model_config = ConfigDict(extra="forbid")

    candidate: SkillCandidate | None = None
    action: str | None = None
    reason: str | None = None
    proposed_name: str | None = None
    existing_skill_name: str | None = None
    related_skill_names: tuple[str, ...] = ()
    model_call_count: int = 1
    provider: str | None = None
    model: str | None = None
    duration_ms: float = 0.0
    usage: ModelUsage = Field(default_factory=ModelUsage)
    raw_output: str | None = None
    adjudication_raw_output: str | None = None
    error: str | None = None


class _RelevanceOutcome(BaseModel):

    model_config = ConfigDict(extra="forbid")

    selected: tuple[str, ...] = ()
    usage: ModelUsage = Field(default_factory=ModelUsage)
    duration_ms: float = 0.0
    error: str | None = None


class _OverlapDecision(BaseModel):

    model_config = ConfigDict(extra="forbid")

    relationship: str
    existing_skill_name: str | None = None
    reason: str

    # 函数说明：_OverlapDecision.valid_relationship
    # 用途：校验并规范化模型字段 'relationship'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `normalized`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().lower`。
    # 分支与异常：
    #   当 `normalized not in {'same', 'different'}` 时，抛出
    # `ValueError(f'invalid relationship: {value}')`。
    @field_validator("relationship")
    @classmethod
    def valid_relationship(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"same", "different"}:
            raise ValueError(f"invalid relationship: {value}")
        return normalized


class ProcedureDistiller:

    # 函数说明：ProcedureDistiller.__init__
    # 用途：初始化 ProcedureDistiller；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ModelAdapterRegistry`。
    #   settings：业务或模型设置，类型 `SkillLearningSettings`。
    #   default_provider：未指定服务商时的默认值，类型 `str | None`；默认 `None`。
    #   default_model：未指定模型时的默认值，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._registry`、`self.settings`、`self._default_provider`、
    # `self._default_model`。
    def __init__(
        self,
        registry: ModelAdapterRegistry,
        *,
        settings: SkillLearningSettings,
        default_provider: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self._registry = registry
        self.settings = settings
        self._default_provider = default_provider
        self._default_model = default_model

    # 函数说明：ProcedureDistiller.distill
    # 用途：提炼ProcedureDistiller，供技能候选提炼与审核使用。
    # 参数：
    #   cluster：传给 `self._select_related_skills` 的输入，类型 `TaskPatternCluster`。
    #   evidence：传给 `self._to_candidate` 的输入，类型 `dict[str, str]`。
    #   run_ids：待处理的运行标识集合，类型 `dict[str, tuple[str, ...]]`。
    #   catalog：传给 `self._select_related_skills` 的输入，类型
    # `Sequence[SkillMetadata]`；默认 `()`。
    #   pending_candidates：待处理项候选集合输入或配置值，类型
    # `Sequence[SkillCandidate]`；默认 `()`。
    #   skill_loader：技能输入或配置值，类型
    # `Callable[[str], Awaitable[Skill | None]] | None`；默认 `None`。
    # 返回：类型 `DistillationOutcome`；返回 `DistillationOutcome(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelUsage` →
    # `self._select_related_skills` → `skill_loader` → `_clip_skill_body` →
    # `_merge_usage` → `json.dumps`；另有 6 个调用点。
    # 分支与异常：
    #   当 `not result.ok` 时，返回 `DistillationOutcome(…)`。
    #   当 `payload is None` 时，返回 `DistillationOutcome(…)`。
    #   捕获 `Exception` 后，返回 `DistillationOutcome(…)`。
    #   当 `adjudicated is None` 时，返回 `DistillationOutcome(…)`。
    async def distill(
        self,
        cluster: TaskPatternCluster,
        *,
        evidence: dict[str, str],
        run_ids: dict[str, tuple[str, ...]],
        catalog: Sequence[SkillMetadata] = (),
        pending_candidates: Sequence[SkillCandidate] = (),
        skill_loader: Callable[[str], Awaitable[Skill | None]] | None = None,
    ) -> DistillationOutcome:

        related_names: tuple[str, ...] = ()
        related_bodies: dict[str, str] = {}
        extra_usage = ModelUsage()
        model_call_count = 1
        relevance_duration = 0.0
        if catalog and skill_loader is not None:
            relevance = await self._select_related_skills(cluster, catalog)
            if relevance.error is None and relevance.selected:
                selected = set(relevance.selected)
                names = tuple(
                    item.name
                    for item in catalog
                    if item.name in selected
                )[:_MAX_RELATED_SKILLS]
                bodies: dict[str, str] = {}
                for name in names:
                    skill = await skill_loader(name)
                    if skill is not None:
                        bodies[name] = _clip_skill_body(skill.content)
                related_names = names
                related_bodies = bodies
            extra_usage = _merge_usage(extra_usage, relevance.usage)
            model_call_count += 1
            relevance_duration = relevance.duration_ms

        user_payload: dict[str, Any] = {
            "cluster": cluster.model_dump(mode="json"),
            "evidence": evidence,
            "catalog": [
                {"name": item.name, "description": item.description}
                for item in catalog
            ],
            "related_skills": [
                {"name": name, "body": body}
                for name, body in related_bodies.items()
            ],
            "pending_candidates": [
                {
                    "id": item.id,
                    "action": item.action.value,
                    "proposed_name": item.proposed_name,
                    "description": item.description,
                    "existing_skill_name": item.existing_skill_name,
                    "reason": item.reason[:300],
                }
                for item in pending_candidates
            ],
        }
        user_content = json.dumps(
            user_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        result: ModelCallResult = await call_model(
            self._registry,
            system_prompt=_DISTILLATION_PROMPT,
            user_content=user_content,
            settings=self.settings,
            default_provider=self._default_provider,
            default_model=self._default_model,
        )
        total_usage = _merge_usage(extra_usage, result.usage)
        total_duration = relevance_duration + result.duration_ms
        if not result.ok:
            return DistillationOutcome(
                provider=result.provider,
                model=result.model,
                duration_ms=total_duration,
                usage=total_usage,
                raw_output=result.raw_output,
                related_skill_names=related_names,
                model_call_count=model_call_count,
                error=result.error,
            )
        payload = parse_strict_json(result.raw_output or "")
        if payload is None:
            return DistillationOutcome(
                provider=result.provider,
                model=result.model,
                duration_ms=total_duration,
                usage=total_usage,
                raw_output=result.raw_output,
                related_skill_names=related_names,
                model_call_count=model_call_count,
                error="distillation returned non-JSON output",
            )
        try:
            distilled = _Distilled.model_validate(payload)
        except Exception as exc:
            return DistillationOutcome(
                provider=result.provider,
                model=result.model,
                duration_ms=total_duration,
                usage=total_usage,
                raw_output=result.raw_output,
                related_skill_names=related_names,
                model_call_count=model_call_count,
                error=f"invalid distillation schema: {type(exc).__name__}: {exc}",
            )
        if (
            distilled.action == "update"
            and not distilled.existing_skill_name
            and len(related_names) == 1
        ):
            distilled = distilled.model_copy(
                update={"existing_skill_name": related_names[0]}
            )
        adjudication_raw_output: str | None = None
        if distilled.action == "create" and related_bodies:
            adjudicated, adjudication_result = await self._adjudicate_overlap(
                cluster,
                distilled=distilled,
                related_bodies=related_bodies,
            )
            total_usage = _merge_usage(total_usage, adjudication_result.usage)
            total_duration += adjudication_result.duration_ms
            model_call_count += 1
            adjudication_raw_output = adjudication_result.raw_output
            if adjudicated is None:
                return DistillationOutcome(
                    action="create",
                    provider=adjudication_result.provider or result.provider,
                    model=adjudication_result.model or result.model,
                    duration_ms=total_duration,
                    usage=total_usage,
                    raw_output=result.raw_output,
                    adjudication_raw_output=adjudication_raw_output,
                    reason=distilled.reason,
                    proposed_name=distilled.proposed_name,
                    related_skill_names=related_names,
                    model_call_count=model_call_count,
                    error=(
                        adjudication_result.error
                        or "overlap adjudication returned an invalid decision"
                    ),
                )
            if adjudicated.relationship == "same":
                target = adjudicated.existing_skill_name
                if target is None and len(related_names) == 1:
                    target = related_names[0]
                if target not in related_bodies:
                    return DistillationOutcome(
                        action="create",
                        provider=adjudication_result.provider or result.provider,
                        model=adjudication_result.model or result.model,
                        duration_ms=total_duration,
                        usage=total_usage,
                        raw_output=result.raw_output,
                        adjudication_raw_output=adjudication_raw_output,
                        reason=adjudicated.reason,
                        proposed_name=distilled.proposed_name,
                        related_skill_names=related_names,
                        model_call_count=model_call_count,
                        error=(
                            "overlap adjudication selected an unknown existing "
                            f"skill: {target!r}"
                        ),
                    )
                distilled = distilled.model_copy(
                    update={
                        "action": "update",
                        "proposed_name": target,
                        "existing_skill_name": target,
                        "reason": adjudicated.reason,
                    }
                )
        if distilled.action == "none":
            return DistillationOutcome(
                action="none",
                provider=result.provider,
                model=result.model,
                duration_ms=total_duration,
                usage=total_usage,
                raw_output=result.raw_output,
                adjudication_raw_output=adjudication_raw_output,
                reason=distilled.reason,
                proposed_name=distilled.proposed_name,
                existing_skill_name=distilled.existing_skill_name,
                related_skill_names=related_names,
                model_call_count=model_call_count,
            )
        try:
            candidate = self._to_candidate(
                cluster, distilled, evidence, run_ids, catalog
            )
        except Exception as exc:
            return DistillationOutcome(
                action=distilled.action,
                provider=result.provider,
                model=result.model,
                duration_ms=total_duration,
                usage=total_usage,
                raw_output=result.raw_output,
                adjudication_raw_output=adjudication_raw_output,
                reason=distilled.reason,
                proposed_name=distilled.proposed_name,
                existing_skill_name=distilled.existing_skill_name,
                related_skill_names=related_names,
                model_call_count=model_call_count,
                error=f"invalid candidate: {type(exc).__name__}: {exc}",
            )
        return DistillationOutcome(
            candidate=candidate,
            action=distilled.action,
            reason=candidate.reason,
            proposed_name=candidate.proposed_name,
            existing_skill_name=candidate.existing_skill_name,
            provider=result.provider,
            model=result.model,
            duration_ms=total_duration,
            usage=total_usage,
            raw_output=result.raw_output,
            adjudication_raw_output=adjudication_raw_output,
            related_skill_names=related_names,
            model_call_count=model_call_count,
        )

    # 函数说明：ProcedureDistiller._adjudicate_overlap
    # 用途：判定`overlap`，供技能候选提炼与审核使用。
    # 参数：
    #   cluster：`cluster`输入或配置值，类型 `TaskPatternCluster`。
    #   distilled：`distilled`输入或配置值，类型 `_Distilled`。
    #   related_bodies：`related_bodies`输入或配置值，类型 `dict[str, str]`。
    # 返回：类型 `tuple[_OverlapDecision | None, ModelCallResult]`；按分支返回
    # `(None, result)`；
    # `(None, replace(result, error='overlap adjudication returned non-JSON output'))`；
    # `(_OverlapDecision.model_validate(payload), result)`；`(None, replace(result,
    # error=f'invalid overlap adjudication schema: {type(exc).__name__}:…`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps` → `call_model` →
    # `parse_strict_json` → `replace` → `_OverlapDecision.model_validate`。
    # 分支与异常：
    #   当 `not result.ok` 时，返回 `(None, result)`。
    #   当 `payload is None` 时，返回
    # `(None, replace(result, error='overlap adjudication returned…`。
    #   捕获 `Exception` 后，返回 `(None, replace(result, error=f'invalid overlap
    # adjudication schema: {type(exc).__name__}:…`。
    async def _adjudicate_overlap(
        self,
        cluster: TaskPatternCluster,
        *,
        distilled: _Distilled,
        related_bodies: dict[str, str],
    ) -> tuple[_OverlapDecision | None, ModelCallResult]:

        user_content = json.dumps(
            {
                "cluster": cluster.model_dump(mode="json"),
                "proposed_candidate": distilled.model_dump(mode="json"),
                "related_skills": [
                    {"name": name, "body": body}
                    for name, body in related_bodies.items()
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        result = await call_model(
            self._registry,
            system_prompt=_OVERLAP_ADJUDICATION_PROMPT,
            user_content=user_content,
            settings=self.settings,
            default_provider=self._default_provider,
            default_model=self._default_model,
        )
        if not result.ok:
            return None, result
        payload = parse_strict_json(result.raw_output or "")
        if payload is None:
            return None, replace(
                result,
                error="overlap adjudication returned non-JSON output",
            )
        try:
            return _OverlapDecision.model_validate(payload), result
        except Exception as exc:
            return None, replace(
                result,
                error=(
                    "invalid overlap adjudication schema: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # 函数说明：ProcedureDistiller._select_related_skills
    # 用途：选取技能集合，供技能候选提炼与审核使用。
    # 参数：
    #   cluster：`cluster`输入或配置值，类型 `TaskPatternCluster`。
    #   catalog：`catalog`输入或配置值，类型 `Sequence[SkillMetadata]`。
    # 返回：类型 `_RelevanceOutcome`；按分支返回 `_RelevanceOutcome(usage=result.usage,
    # duration_ms=result.duration_ms, error=result.error)`；`_RelevanceOutcome(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps` → `call_model` →
    # `_RelevanceOutcome` → `parse_strict_json`。
    # 分支与异常：
    #   当 `not result.ok` 时，返回 `_RelevanceOutcome(…)`。
    #   当 `payload is None or not isinstance(payload.get('…` 时，返回
    # `_RelevanceOutcome(…)`。
    async def _select_related_skills(
        self,
        cluster: TaskPatternCluster,
        catalog: Sequence[SkillMetadata],
    ) -> _RelevanceOutcome:

        user_payload: dict[str, Any] = {
            "cluster": {
                "pattern_name": cluster.pattern_name,
                "description": cluster.description,
                "similarity_reason": cluster.similarity_reason,
            },
            "catalog": [
                {"name": item.name, "description": item.description}
                for item in catalog
            ],
        }
        user_content = json.dumps(
            user_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        result: ModelCallResult = await call_model(
            self._registry,
            system_prompt=_RELEVANCE_PROMPT,
            user_content=user_content,
            settings=self.settings,
            default_provider=self._default_provider,
            default_model=self._default_model,
        )
        if not result.ok:
            return _RelevanceOutcome(
                usage=result.usage,
                duration_ms=result.duration_ms,
                error=result.error,
            )
        payload = parse_strict_json(result.raw_output or "")
        if payload is None or not isinstance(payload.get("related_skills"), list):
            return _RelevanceOutcome(
                usage=result.usage,
                duration_ms=result.duration_ms,
                error="relevance returned non-JSON output",
            )
        selected: list[str] = []
        for item in payload["related_skills"]:
            if isinstance(item, str) and item.strip():
                name = item.strip()
                if name not in selected:
                    selected.append(name)
        return _RelevanceOutcome(
            selected=tuple(selected),
            usage=result.usage,
            duration_ms=result.duration_ms,
        )

    # 函数说明：ProcedureDistiller._to_candidate
    # 用途：在技能候选提炼与审核中处理 `_to_candidate`，通过 `source_run_ids.extend` 完
    # 成首个内部处理步骤。
    # 参数：
    #   cluster：传给 `_evidence_summary` 的输入，类型 `TaskPatternCluster`。
    #   distilled：`distilled`输入或配置值，类型 `_Distilled`。
    #   evidence：传给 `_evidence_summary` 的输入，类型 `dict[str, str]`。
    #   run_ids：待处理的运行标识集合，类型 `dict[str, tuple[str, ...]]`。
    #   catalog：传给 `_catalog_description` 的输入，类型 `Sequence[SkillMetadata]`；默
    # 认 `()`。
    # 返回：类型 `SkillCandidate`；返回 `SkillCandidate(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_evidence_summary` →
    # `SkillCandidateAction` → `_catalog_description` → `SkillCandidate` → `uuid4` →
    # `datetime.now`。
    # 分支与异常：
    #   当 `not description` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   数据库操作：CREATE；连接与事务边界以 with/提交语句为准。
    def _to_candidate(
        self,
        cluster: TaskPatternCluster,
        distilled: _Distilled,
        evidence: dict[str, str],
        run_ids: dict[str, tuple[str, ...]],
        catalog: Sequence[SkillMetadata] = (),
    ) -> SkillCandidate:
        source_run_ids: list[str] = []
        for task_id in cluster.task_ids:
            source_run_ids.extend(run_ids.get(task_id, ()))
        unique_runs: list[str] = []
        for run_id in source_run_ids:
            if run_id not in unique_runs:
                unique_runs.append(run_id)
        evidence_summary = _evidence_summary(evidence, cluster)
        action = SkillCandidateAction(distilled.action)
        proposed_name = distilled.proposed_name or ""
        if action is SkillCandidateAction.UPDATE and not proposed_name:
            proposed_name = distilled.existing_skill_name or ""
        description = distilled.description or ""
        if action is SkillCandidateAction.UPDATE:
            if not description:
                description = _catalog_description(
                    catalog, distilled.existing_skill_name
                )
                if not description:
                    raise ValueError(
                        "update candidate requires a non-empty description: model "
                        f"did not provide one and existing_skill_name "
                        f"{distilled.existing_skill_name!r} was not found in the "
                        "skill catalog"
                    )
        else:
            if not description:
                raise ValueError(
                    "create candidate requires a non-empty description"
                )
        return SkillCandidate(
            id=uuid4().hex,
            action=action,
            proposed_name=proposed_name,
            description=description,
            reason=distilled.reason or "",
            procedure=distilled.procedure,
            pitfalls=distilled.pitfalls,
            verification=distilled.verification,
            source_task_ids=tuple(cluster.task_ids),
            source_run_ids=tuple(unique_runs),
            existing_skill_name=distilled.existing_skill_name,
            status=SkillCandidateStatus.PENDING,
            created_at=datetime.now(UTC),
            evidence_summary=evidence_summary,
        )


# 函数说明：_evidence_summary
# 用途：在技能候选提炼与审核中处理 `_evidence_summary`，通过 `','.join` 完成首个内部处理
# 步骤。
# 参数：
#   evidence：原始证据输入或配置值，类型 `dict[str, str]`。
#   cluster：`cluster`输入或配置值，类型 `TaskPatternCluster`。
# 返回：类型 `str`；返回 `'\n'.join(lines)`。
# 分支与异常：
#   当 `not text` 时，跳过当前循环项。
def _evidence_summary(
    evidence: dict[str, str],
    cluster: TaskPatternCluster,
) -> str:

    lines = [
        f"cluster={cluster.pattern_name}",
        f"tasks={','.join(cluster.task_ids)}",
        f"similarity={cluster.similarity_reason}",
    ]
    for task_id in cluster.task_ids:
        text = evidence.get(task_id, "")
        if not text:
            continue
        first = " ".join(text.split())[:240]
        lines.append(f"evidence[{task_id}] {first}")
    return "\n".join(lines)


# 函数说明：_clip_skill_body
# 用途：限制长度技能，供技能候选提炼与审核使用。
# 参数：
#   content：内容正文，类型 `str`。
# 返回：类型 `str`；按分支返回 `text`；`text[:_MAX_SKILL_BODY_CHARS] + '…[截断]'`。
# 分支与异常：
#   当 `len(text) <= _MAX_SKILL_BODY_CHARS` 时，返回 `text`。
def _clip_skill_body(content: str) -> str:

    text = " ".join(content.split())
    if len(text) <= _MAX_SKILL_BODY_CHARS:
        return text
    return text[:_MAX_SKILL_BODY_CHARS] + "…[截断]"


# 函数说明：_catalog_description
# 用途：处理技能候选提炼与审核中的 `_catalog_description` 数据；结果及边界条件见下方说明
# 。
# 参数：
#   catalog：`catalog`输入或配置值，类型 `Sequence[SkillMetadata]`。
#   name：目标对象、工具或配置项名称，类型 `str | None`。
# 返回：类型 `str`；按分支返回 `''`；`item.description`。
# 分支与异常：
#   当 `not name` 时，返回 `''`。
#   当 `item.name == name` 时，返回 `item.description`。
def _catalog_description(
    catalog: Sequence[SkillMetadata],
    name: str | None,
) -> str:

    if not name:
        return ""
    for item in catalog:
        if item.name == name:
            return item.description
    return ""


# 函数说明：_merge_usage
# 用途：合并用量，供技能候选提炼与审核使用。
# 参数：
#   total：传给 `add_model_usage` 的输入，类型 `ModelUsage`。
#   current：当前值或状态，类型 `ModelUsage`。
# 返回：类型 `ModelUsage`；返回 `add_model_usage(total, current)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`add_model_usage`。
def _merge_usage(total: ModelUsage, current: ModelUsage) -> ModelUsage:

    return add_model_usage(total, current)


__all__ = ["DistillationOutcome", "ProcedureDistiller"]
