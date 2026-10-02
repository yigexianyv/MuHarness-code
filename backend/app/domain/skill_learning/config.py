
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.paths import runtime_data_path

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

_DEFAULT_DATA_DIR = runtime_data_path("skill-learning")


class SkillLearningSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    skill_learning_enabled: bool = True
    skill_learning_batch_size: int = Field(default=20, ge=1)
    skill_learning_min_cluster_size: int = Field(default=3, ge=2)
    skill_learning_max_tasks_per_scan: int = Field(default=20, ge=1)
    skill_learning_max_attempts: int = Field(default=3, ge=1)
    skill_learning_provider: str | None = None
    skill_learning_model: str | None = None
    skill_learning_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    skill_learning_max_output_tokens: int = Field(default=2_000, ge=1)
    skill_learning_timeout_seconds: float = Field(default=60.0, gt=0.0)
    skill_learning_disable_thinking: bool | None = None
    skill_learning_default_scope: str = "project"
    skill_learning_data_dir: Path = _DEFAULT_DATA_DIR
    skill_learning_max_events_per_task: int = Field(default=200, ge=1)
    skill_learning_max_evidence_chars: int = Field(default=3_000, ge=200)


__all__ = ["SkillLearningSettings"]
