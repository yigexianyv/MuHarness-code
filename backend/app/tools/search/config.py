
from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class SearchProviderName(StrEnum):
    AUTO = "auto"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"


class SearchSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    search_provider: SearchProviderName = SearchProviderName.AUTO
    tavily_api_key: SecretStr | None = None
    search_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    search_max_results: int = Field(default=5, ge=1, le=10)

    # 函数说明：SearchSettings.tavily_api_key_value
    # 用途：在网页搜索服务商与降级中处理 `tavily_api_key_value`，通过
    # `self.tavily_api_key.get_secret_value().strip` 完成首个内部处理步骤。
    # 返回：类型 `str | None`；按分支返回 `None`；`value or None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self.tavily_api_key.get_secret_value`。
    # 分支与异常：
    #   当 `self.tavily_api_key is None` 时，返回 `None`。
    def tavily_api_key_value(self) -> str | None:
        if self.tavily_api_key is None:
            return None
        value = self.tavily_api_key.get_secret_value().strip()
        return value or None


__all__ = ["SearchProviderName", "SearchSettings"]
