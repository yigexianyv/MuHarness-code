
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class MemoryEmbeddingSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="MEMORY_EMBEDDING_",
        extra="ignore",
    )

    enabled: bool = False
    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    dimensions: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=1, ge=0)
    batch_size: int = Field(default=16, ge=1)
    min_similarity: float | None = Field(default=None, ge=0.0, lt=1.0)

    # 函数说明：MemoryEmbeddingSettings.normalize_optional_text
    # 用途：校验并规范化模型字段 'base_url'、'model'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`value.strip() or None`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('embedding base_url and model must be strings')`。
    @field_validator("base_url", "model", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("embedding base_url and model must be strings")
        return value.strip() or None

    # 函数说明：MemoryEmbeddingSettings.adapter_configured
    # 用途：返回 `self.enabled and bool(self.model) and (self.api_key is not None)`，提
    # 供 MemoryEmbeddingSettings 的派生值。
    # 返回：类型 `bool`；返回
    # `self.enabled and bool(self.model) and (self.api_key is not None)`。
    def adapter_configured(self) -> bool:

        return self.enabled and bool(self.model) and self.api_key is not None


@runtime_checkable
class EmbeddingAdapter(Protocol):

    # 函数说明：EmbeddingAdapter.model_name
    # 用途：处理长期记忆管理与检索中的 `model_name` 数据；结果及边界条件见下方说明。
    # 返回：类型 `str`；不返回结果值（隐式 None）。
    @property
    def model_name(self) -> str:
        pass

    # 函数说明：EmbeddingAdapter.dimensions
    # 用途：处理长期记忆管理与检索中的 `dimensions` 数据；结果及边界条件见下方说明。
    # 返回：类型 `int | None`；不返回结果值（隐式 None）。
    @property
    def dimensions(self) -> int | None:
        pass

    # 函数说明：EmbeddingAdapter.embed_documents
    # 用途：生成向量`documents`，供长期记忆管理与检索使用。
    # 参数：
    #   texts：待处理的文本集合，类型 `tuple[str, ...]`。
    # 返回：类型 `tuple[tuple[float, ...], ...]`；不返回结果值（隐式 None）。
    async def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        pass

    # 函数说明：EmbeddingAdapter.embed_query
    # 用途：生成向量查询，供长期记忆管理与检索使用。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `tuple[float, ...]`；不返回结果值（隐式 None）。
    async def embed_query(self, text: str) -> tuple[float, ...]:
        pass

    # 函数说明：EmbeddingAdapter.close
    # 用途：关闭EmbeddingAdapter，供长期记忆管理与检索使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class OpenAICompatibleEmbeddingAdapter:

    # 函数说明：OpenAICompatibleEmbeddingAdapter.__init__
    # 用途：初始化 OpenAICompatibleEmbeddingAdapter；参数及实际保存的实例字段见下方说明
    # 。
    # 参数：
    #   base_url：传给 `_is_local_base_url` 的输入，类型 `str | None`。
    #   api_key：键输入或配置值，类型 `str`。
    #   model：模型名称，类型 `str`。
    #   dimensions：`dimensions`输入或配置值，类型 `int | None`；默认 `None`。
    #   timeout_seconds：等待或执行超时，单位为秒，类型 `float`；默认 `30.0`。
    #   max_retries：`max_retries`输入或配置值，类型 `int`；默认 `1`。
    #   batch_size：传给 `max` 的输入，类型 `int`；默认 `16`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_is_local_base_url` →
    # `httpx.AsyncClient` → `AsyncOpenAI`。
    # 分支与异常：
    #   当 `not model` 时，抛出 `ValueError('embedding model is required')`。
    # 副作用与资源：
    #   更新对象字段：`self._client`、`self._model`、`self._dimensions`、
    # `self._batch_size`。
    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str,
        model: str,
        dimensions: int | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        batch_size: int = 16,
    ) -> None:
        import httpx
        from openai import AsyncOpenAI

        if not model:
            raise ValueError("embedding model is required")
        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        if base_url and _is_local_base_url(base_url):
            client_kwargs["http_client"] = httpx.AsyncClient(trust_env=False)
        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)

    # 函数说明：OpenAICompatibleEmbeddingAdapter.model_name
    # 用途：返回 `self._model`，提供 OpenAICompatibleEmbeddingAdapter 的派生值。
    # 返回：类型 `str`；返回 `self._model`。
    @property
    def model_name(self) -> str:
        return self._model

    # 函数说明：OpenAICompatibleEmbeddingAdapter.dimensions
    # 用途：返回 `self._dimensions`，提供 OpenAICompatibleEmbeddingAdapter 的派生值。
    # 返回：类型 `int | None`；返回 `self._dimensions`。
    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    # 函数说明：OpenAICompatibleEmbeddingAdapter.embed_documents
    # 用途：生成向量`documents`，供长期记忆管理与检索使用。
    # 参数：
    #   texts：待处理的文本集合，类型 `tuple[str, ...]`。
    # 返回：类型 `tuple[tuple[float, ...], ...]`；返回 `tuple(vectors)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._embed`。
    async def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed(list(batch)))
        return tuple(vectors)

    # 函数说明：OpenAICompatibleEmbeddingAdapter.embed_query
    # 用途：生成向量查询，供长期记忆管理与检索使用。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `tuple[float, ...]`；返回 `vectors[0]`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._embed`。
    async def embed_query(self, text: str) -> tuple[float, ...]:
        vectors = await self._embed([text])
        return vectors[0]

    # 函数说明：OpenAICompatibleEmbeddingAdapter._embed
    # 用途：生成向量OpenAICompatibleEmbeddingAdapter，供长期记忆管理与检索使用。
    # 参数：
    #   texts：待处理的文本集合，类型 `list[str]`。
    # 返回：类型 `list[tuple[float, ...]]`；返回
    # `tuple((tuple((float(value) for value in item.embedding)) for item in ordered))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._client.embeddings.create`。
    async def _embed(
        self,
        texts: list[str],
    ) -> list[tuple[float, ...]]:
        request: dict[str, object] = {
            "model": self._model,
            "input": texts,
        }
        if self._dimensions is not None:
            request["dimensions"] = self._dimensions
        response = await self._client.embeddings.create(**request)
        ordered = sorted(response.data, key=lambda item: item.index)
        return tuple(
            tuple(float(value) for value in item.embedding) for item in ordered
        )

    # 函数说明：OpenAICompatibleEmbeddingAdapter.close
    # 用途：关闭OpenAICompatibleEmbeddingAdapter，供长期记忆管理与检索使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._client.close`。
    async def close(self) -> None:
        await self._client.close()


class FakeEmbeddingAdapter:

    # 函数说明：FakeEmbeddingAdapter.__init__
    # 用途：初始化 FakeEmbeddingAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   dimensions：`dimensions`输入或配置值，类型 `int`；默认 `1024`。
    #   model_name：模型名称，类型 `str`；默认 `'fake-embedding'`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `dimensions <= 0` 时，抛出
    # `ValueError('fake embedding dimensions must be positive')`。
    # 副作用与资源：
    #   更新对象字段：`self._dimensions`、`self._model_name`。
    def __init__(
        self,
        *,
        dimensions: int = 1024,
        model_name: str = "fake-embedding",
    ) -> None:
        if dimensions <= 0:
            raise ValueError("fake embedding dimensions must be positive")
        self._dimensions = dimensions
        self._model_name = model_name

    # 函数说明：FakeEmbeddingAdapter.model_name
    # 用途：返回 `self._model_name`，提供 FakeEmbeddingAdapter 的派生值。
    # 返回：类型 `str`；返回 `self._model_name`。
    @property
    def model_name(self) -> str:
        return self._model_name

    # 函数说明：FakeEmbeddingAdapter.dimensions
    # 用途：返回 `self._dimensions`，提供 FakeEmbeddingAdapter 的派生值。
    # 返回：类型 `int | None`；返回 `self._dimensions`。
    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    # 函数说明：FakeEmbeddingAdapter.embed_documents
    # 用途：生成向量`documents`，供长期记忆管理与检索使用。
    # 参数：
    #   texts：待处理的文本集合，类型 `tuple[str, ...]`。
    # 返回：类型 `tuple[tuple[float, ...], ...]`；返回
    # `tuple((self._vector(text) for text in texts))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._vector`。
    async def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vector(text) for text in texts)

    # 函数说明：FakeEmbeddingAdapter.embed_query
    # 用途：生成向量查询，供长期记忆管理与检索使用。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `tuple[float, ...]`；返回 `self._vector(text)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._vector`。
    async def embed_query(self, text: str) -> tuple[float, ...]:
        return self._vector(text)

    # 函数说明：FakeEmbeddingAdapter.close
    # 用途：关闭FakeEmbeddingAdapter，供长期记忆管理与检索使用。
    # 返回：类型 `None`；无结果值，显式返回 None。
    async def close(self) -> None:
        return None

    # 函数说明：FakeEmbeddingAdapter._vector
    # 用途：在长期记忆管理与检索中处理 `_vector`，通过 `counts.get` 完成首个内部处理步骤
    # 。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `tuple[float, ...]`；返回 `tuple(vector)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_semantic_tokens` →
    # `_hash_bucket` → `math.sqrt`。
    def _vector(self, text: str) -> tuple[float, ...]:
        counts: dict[int, float] = {}
        for token in _semantic_tokens(text):
            bucket = _hash_bucket(token, self._dimensions)
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        vector = [0.0] * self._dimensions
        for bucket, weight in counts.items():
            vector[bucket] = weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return tuple(vector)


_TOKEN_SEPARATOR = re.compile(
    r"[\s,.;:!?，。；：！？、()\[\]{}\"'`|/\\<>@#$%^&*+=~\-_""]+"
)

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"})


# 函数说明：_is_local_base_url
# 用途：在长期记忆管理与检索中处理 `_is_local_base_url`，通过
# `(urlsplit(base_url).hostname or '').strip().lower` 完成首个内部处理步骤。
# 参数：
#   base_url：传给 `urlsplit` 的输入，类型 `str`。
# 返回：类型 `bool`；返回 `host in _LOCAL_HOSTS`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `(urlsplit(base_url).hostname or '').strip().lower` → `urlsplit`。
def _is_local_base_url(base_url: str) -> bool:

    from urllib.parse import urlsplit

    host = (urlsplit(base_url).hostname or "").strip().lower()
    return host in _LOCAL_HOSTS


# 函数说明：_semantic_tokens
# 用途：在长期记忆管理与检索中处理 `_semantic_tokens`，通过 `_TOKEN_SEPARATOR.split` 完
# 成首个内部处理步骤。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `list[str]`；返回 `tokens`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`text.casefold` → `raw.isascii`。
# 分支与异常：
#   当 `not raw` 时，跳过当前循环项。
#   `raw.isascii()` 分支在完成前置处理后跳过当前循环项。
def _semantic_tokens(text: str) -> list[str]:

    tokens: list[str] = []
    for raw in _TOKEN_SEPARATOR.split(text.casefold()):
        if not raw:
            continue
        if raw.isascii():
            tokens.append(raw)
            continue
        tokens.extend(raw[i : i + 2] for i in range(len(raw) - 1))
        if len(raw) == 1:
            tokens.append(raw)
    return tokens


# 函数说明：_hash_bucket
# 用途：在长期记忆管理与检索中处理 `_hash_bucket`，通过
# `hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest` 完成首个内部处理步骤。
# 参数：
#   token：Token输入或配置值，类型 `str`。
#   dimensions：`dimensions`输入或配置值，类型 `int`。
# 返回：类型 `int`；返回 `int.from_bytes(digest, 'big') % dimensions`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest` → `hashlib.blake2b` →
# `token.encode` → `int.from_bytes`。
def _hash_bucket(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


# 函数说明：build_embedding_adapter
# 用途：构建向量，供长期记忆管理与检索使用。
# 参数：
#   settings：业务或模型设置，类型 `MemoryEmbeddingSettings`。
# 返回：类型 `OpenAICompatibleEmbeddingAdapter | None`；按分支返回 `None`；
# `OpenAICompatibleEmbeddingAdapter(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`settings.adapter_configured` →
# `OpenAICompatibleEmbeddingAdapter` → `settings.api_key.get_secret_value`。
# 分支与异常：
#   当 `not settings.adapter_configured()` 时，返回 `None`。
def build_embedding_adapter(
    settings: MemoryEmbeddingSettings,
) -> OpenAICompatibleEmbeddingAdapter | None:

    if not settings.adapter_configured():
        return None
    return OpenAICompatibleEmbeddingAdapter(
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),
        model=settings.model or "",
        dimensions=settings.dimensions,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        batch_size=settings.batch_size,
    )


__all__ = [
    "EmbeddingAdapter",
    "FakeEmbeddingAdapter",
    "MemoryEmbeddingSettings",
    "OpenAICompatibleEmbeddingAdapter",
    "build_embedding_adapter",
]
