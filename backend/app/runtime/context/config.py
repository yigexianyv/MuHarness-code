from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class ContextSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    context_window_default: int = Field(default=128_000, gt=0)
    context_window_openai: int = Field(default=200_000, gt=0)
    context_window_qwen: int = Field(default=1_000_000, gt=0)
    context_window_deepseek: int = Field(default=1_048_576, gt=0)
    context_window_anthropic: int = Field(default=200_000, gt=0)

    context_safety_margin_tokens: int = Field(default=4_096, ge=0)
    context_trigger_ratio: float = Field(default=0.80, gt=0.0, lt=1.0)
    context_target_ratio: float = Field(default=0.60, gt=0.0, lt=1.0)
    context_preferred_input_tokens: int = Field(default=64_000, gt=0)
    context_working_trigger_ratio: float = Field(default=0.80, gt=0.0, le=1.0)
    context_working_target_ratio: float = Field(default=0.45, gt=0.0, lt=1.0)
    context_compact_input_tokens: int | None = Field(default=None, gt=0)
    context_tool_result_budget_ratio: float = Field(default=0.35, gt=0.0, lt=1.0)
    context_keep_recent_tool_rounds: int = Field(default=2, ge=0)
    context_keep_recent_conversation_blocks: int = Field(default=4, ge=0)
    context_summary_max_output_tokens: int = Field(default=1_024, gt=0)
    context_large_fold_span_tokens: int = Field(default=50_000, ge=0)
    context_summary_max_output_tokens_large_fold: int = Field(default=2_048, gt=0)
    context_max_tool_result_chars: int = Field(default=8_000, gt=0)
    context_tool_result_head_chars: int = Field(default=4_000, ge=0)
    context_tool_result_tail_chars: int = Field(default=2_000, ge=0)

    context_override_provider: str | None = None
    context_override_model: str | None = None
    context_window_override: int | None = Field(default=None, gt=0)
    max_output_tokens_override: int | None = Field(default=None, gt=0)

    # 函数说明：ContextSettings.validate_tool_result_segments
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `ContextSettings`；返回 `self`。
    # 分支与异常：
    #   当 `retained > self.context_max_tool_result_chars` 时，抛出 `ValueError(…)`。
    #   当 `self.context_working_target_ratio >=…` 时，抛出 `ValueError(…)`。
    #   当 `self.context_summary_max_output_tokens_large_fold <…` 时，抛出
    # `ValueError(…)`。
    @model_validator(mode="after")
    def validate_tool_result_segments(self) -> ContextSettings:

        retained = (
            self.context_tool_result_head_chars + self.context_tool_result_tail_chars
        )
        if retained > self.context_max_tool_result_chars:
            raise ValueError(
                "context tool result head/tail chars cannot exceed "
                "context_max_tool_result_chars"
            )
        if self.context_working_target_ratio >= self.context_working_trigger_ratio:
            raise ValueError(
                "context_working_target_ratio must be lower than "
                "context_working_trigger_ratio"
            )
        if (
            self.context_summary_max_output_tokens_large_fold
            < self.context_summary_max_output_tokens
        ):
            raise ValueError(
                "context_summary_max_output_tokens_large_fold must not be lower "
                "than context_summary_max_output_tokens"
            )
        return self


class ContextSummaryModelConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="CONTEXT_SUMMARY_",
        extra="ignore",
    )

    enabled: bool = True
    provider: str | None = None
    model: str | None = None

    # 函数说明：ContextSummaryModelConfig.normalize_optional_text
    # 用途：校验并规范化模型字段 'provider'、'model'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`value.strip() or None`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('summary provider and model must be strings')`。
    @field_validator("provider", "model", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("summary provider and model must be strings")
        return value.strip() or None


__all__ = ["ContextSettings", "ContextSummaryModelConfig"]
