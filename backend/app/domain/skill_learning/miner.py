
from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from app.models.registry import ModelAdapterRegistry
from app.models.types import ModelUsage

from ._call import ModelCallResult, call_model, parse_strict_json
from .config import SkillLearningSettings
from .models import PatternMiningResult, TaskCard, TaskPatternCluster
from .prompts import _PATTERN_MINING_PROMPT


class PatternMiningOutcome(BaseModel):

    model_config = ConfigDict(extra="forbid")

    clusters: tuple[TaskPatternCluster, ...] = ()
    provider: str | None = None
    model: str | None = None
    duration_ms: float = 0.0
    usage: ModelUsage = Field(default_factory=ModelUsage)
    raw_output: str | None = None
    error: str | None = None


class TaskPatternMiner:

    # 函数说明：TaskPatternMiner.__init__
    # 用途：初始化 TaskPatternMiner；参数及实际保存的实例字段见下方说明。
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

    # 函数说明：TaskPatternMiner.mine
    # 用途：挖掘TaskPatternMiner，供技能候选提炼与审核使用。
    # 参数：
    #   cards：`cards`输入或配置值，类型 `tuple[TaskCard, ...]`。
    # 返回：类型 `PatternMiningOutcome`；按分支返回 `PatternMiningOutcome()`；
    # `PatternMiningOutcome(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`PatternMiningOutcome` →
    # `json.dumps` → `call_model` → `parse_strict_json` →
    # `PatternMiningResult.model_validate` → `set(cluster.task_ids).issubset`。
    # 分支与异常：
    #   当 `not cards` 时，返回 `PatternMiningOutcome()`。
    #   当 `not result.ok` 时，返回 `PatternMiningOutcome(…)`。
    #   当 `payload is None` 时，返回 `PatternMiningOutcome(…)`。
    #   捕获 `Exception` 后，返回 `PatternMiningOutcome(…)`。
    async def mine(
        self,
        cards: tuple[TaskCard, ...],
    ) -> PatternMiningOutcome:

        if not cards:
            return PatternMiningOutcome()
        user_content = json.dumps(
            [card.model_dump(mode="json") for card in cards],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        result: ModelCallResult = await call_model(
            self._registry,
            system_prompt=_PATTERN_MINING_PROMPT,
            user_content=user_content,
            settings=self.settings,
            default_provider=self._default_provider,
            default_model=self._default_model,
        )
        if not result.ok:
            return PatternMiningOutcome(
                provider=result.provider,
                model=result.model,
                duration_ms=result.duration_ms,
                usage=result.usage,
                raw_output=result.raw_output,
                error=result.error,
            )
        payload = parse_strict_json(result.raw_output or "")
        if payload is None:
            return PatternMiningOutcome(
                provider=result.provider,
                model=result.model,
                duration_ms=result.duration_ms,
                usage=result.usage,
                raw_output=result.raw_output,
                error="pattern mining returned non-JSON output",
            )
        try:
            parsed = PatternMiningResult.model_validate(payload)
        except Exception as exc:
            return PatternMiningOutcome(
                provider=result.provider,
                model=result.model,
                duration_ms=result.duration_ms,
                usage=result.usage,
                raw_output=result.raw_output,
                error=f"invalid pattern mining schema: {type(exc).__name__}: {exc}",
            )
        valid_ids = {card.task_id for card in cards}
        clusters = tuple(
            cluster
            for cluster in parsed.clusters
            if len(cluster.task_ids) >= self.settings.skill_learning_min_cluster_size
            and set(cluster.task_ids).issubset(valid_ids)
        )
        return PatternMiningOutcome(
            clusters=clusters,
            provider=result.provider,
            model=result.model,
            duration_ms=result.duration_ms,
            usage=result.usage,
            raw_output=result.raw_output,
        )


__all__ = ["PatternMiningOutcome", "TaskPatternMiner"]
