
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class SkillSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    skill_context_max_tokens: int = Field(default=4_096, gt=0)
    skill_max_active: int = Field(default=4, gt=0)
    skill_catalog_max_tokens: int = Field(default=2_048, gt=0)


__all__ = ["SkillSettings"]
