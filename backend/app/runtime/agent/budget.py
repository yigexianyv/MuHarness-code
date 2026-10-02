
from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.types import ModelUsage

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class RunBudgetStatus(StrEnum):

    DISABLED = "disabled"
    ACTIVE = "active"
    WARNING = "warning"
    FINALIZING = "finalizing"
    EXCEEDED = "exceeded"


class RunBudgetReason(StrEnum):

    TOKENS = "tokens"
    MODEL_CALLS = "model_calls"


class RunBudgetConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="RUN_BUDGET_",
        extra="ignore",
    )

    enabled: bool = True
    warning_tokens: int = Field(default=80_000, ge=1)
    finalization_tokens: int = Field(default=120_000, ge=1)
    hard_tokens: int = Field(default=160_000, ge=1)
    warning_model_calls: int | None = Field(default=None, ge=1)
    finalization_model_calls: int | None = Field(default=None, ge=1)
    hard_model_calls: int = Field(default=15, ge=1)
    finalization_max_output_tokens: int = Field(default=1_200, ge=1)

    # 函数说明：RunBudgetConfig.validate_thresholds
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `RunBudgetConfig`；返回 `self`。
    # 分支与异常：
    #   当 `not self.warning_tokens < self.finalization_tokens <…` 时，抛出
    # `ValueError(…)`。
    #   当 `any(…)` 时，抛出 `ValueError(…)`。
    @model_validator(mode="after")
    def validate_thresholds(self) -> RunBudgetConfig:

        if not (
            self.warning_tokens
            < self.finalization_tokens
            < self.hard_tokens
        ):
            raise ValueError(
                "run budget token thresholds must satisfy warning < "
                "finalization < hard"
            )
        call_thresholds = [
            threshold
            for threshold in (
                self.warning_model_calls,
                self.finalization_model_calls,
                self.hard_model_calls,
            )
            if threshold is not None
        ]
        if any(
            left >= right
            for left, right in zip(call_thresholds, call_thresholds[1:])
        ):
            raise ValueError(
                "configured run budget model call thresholds must be "
                "strictly increasing"
            )
        return self


class RunBudgetDecision(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: RunBudgetStatus
    reason: RunBudgetReason | None = None
    chargeable_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)

    # 函数说明：RunBudgetDecision.should_warn
    # 用途：返回 `self.status is RunBudgetStatus.WARNING`，提供 RunBudgetDecision 的派生
    # 值。
    # 返回：类型 `bool`；返回 `self.status is RunBudgetStatus.WARNING`。
    @property
    def should_warn(self) -> bool:
        return self.status is RunBudgetStatus.WARNING

    # 函数说明：RunBudgetDecision.should_finalize
    # 用途：结束`should`，供模型与工具执行循环使用。
    # 返回：类型 `bool`；返回 `self.status is RunBudgetStatus.FINALIZING`。
    @property
    def should_finalize(self) -> bool:
        return self.status is RunBudgetStatus.FINALIZING

    # 函数说明：RunBudgetDecision.exceeded
    # 用途：返回 `self.status is RunBudgetStatus.EXCEEDED`，提供 RunBudgetDecision 的派
    # 生值。
    # 返回：类型 `bool`；返回 `self.status is RunBudgetStatus.EXCEEDED`。
    @property
    def exceeded(self) -> bool:
        return self.status is RunBudgetStatus.EXCEEDED


class RunBudget:

    # 函数说明：RunBudget.__init__
    # 用途：初始化 RunBudget；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `RunBudgetConfig | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`RunBudgetConfig`。
    # 副作用与资源：
    #   更新对象字段：`self.config`。
    def __init__(self, config: RunBudgetConfig | None = None) -> None:
        self.config = config or RunBudgetConfig()

    # 函数说明：RunBudget.evaluate
    # 用途：根据累计用量和调用次数决定是否警告、收尾或终止本次运行。
    # 参数：
    #   usage：模型调用用量统计，类型 `ModelUsage`。
    #   chargeable_tokens_override：Token 用量输入或配置值，类型 `int | None`；默认
    # `None`。
    #   model_calls_override：模型调用集合输入或配置值，类型 `int | None`；默认 `None`。
    # 返回：类型 `RunBudgetDecision`；返回 `RunBudgetDecision(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`chargeable_tokens` →
    # `RunBudgetDecision`。
    # 分支与异常：
    #   当 `not self.config.enabled` 时，返回 `RunBudgetDecision(…)`。
    #   当 `token_hit or call_hit` 时，返回 `RunBudgetDecision(…)`。
    def evaluate(
        self,
        usage: ModelUsage,
        *,
        chargeable_tokens_override: int | None = None,
        model_calls_override: int | None = None,
    ) -> RunBudgetDecision:

        """根据累计用量和调用次数决定是否警告、收尾或终止本次运行。"""
        chargeable = (
            chargeable_tokens(usage)
            if chargeable_tokens_override is None
            else chargeable_tokens_override
        )
        calls = (
            usage.model_calls
            if model_calls_override is None
            else model_calls_override
        )
        if not self.config.enabled:
            return RunBudgetDecision(
                status=RunBudgetStatus.DISABLED,
                chargeable_tokens=chargeable,
                model_calls=calls,
            )
        for status, token_limit, call_limit in (
            (
                RunBudgetStatus.EXCEEDED,
                self.config.hard_tokens,
                self.config.hard_model_calls,
            ),
            (
                RunBudgetStatus.FINALIZING,
                self.config.finalization_tokens,
                self.config.finalization_model_calls,
            ),
            (
                RunBudgetStatus.WARNING,
                self.config.warning_tokens,
                self.config.warning_model_calls,
            ),
        ):
            token_hit = chargeable >= token_limit
            call_hit = call_limit is not None and calls >= call_limit
            if token_hit or call_hit:
                return RunBudgetDecision(
                    status=status,
                    reason=(
                        RunBudgetReason.TOKENS
                        if token_hit
                        else RunBudgetReason.MODEL_CALLS
                    ),
                    chargeable_tokens=chargeable,
                    model_calls=calls,
                )
        return RunBudgetDecision(
            status=RunBudgetStatus.ACTIVE,
            chargeable_tokens=chargeable,
            model_calls=calls,
        )


# 函数说明：chargeable_tokens
# 用途：计算用于运行预算的可计费 Token 总量。
# 参数：
#   usage：模型调用用量统计，类型 `ModelUsage`。
# 返回：类型 `int`；返回 `input_tokens + usage.output_tokens`。
def chargeable_tokens(usage: ModelUsage) -> int:

    input_tokens = (
        usage.uncached_input_tokens
        if usage.uncached_input_tokens is not None
        else usage.input_tokens
    )
    return input_tokens + usage.output_tokens


__all__ = [
    "RunBudget",
    "RunBudgetConfig",
    "RunBudgetDecision",
    "RunBudgetReason",
    "RunBudgetStatus",
    "chargeable_tokens",
]
