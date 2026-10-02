
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from array import array
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import aiosqlite

from .embedding import EmbeddingAdapter
from .models import MemoryRecord, MemoryStatus

logger = logging.getLogger("muharness.memory.search_index")

DEFAULT_SEARCH_DATABASE_NAME = "search.sqlite"
_SCHEMA_VERSION = 2
_RRF_K = 60


class SearchMode(StrEnum):

    HYBRID = "hybrid"
    VECTOR = "vector"
    FTS = "fts"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class MemorySearchCandidate:

    memory_id: str
    title: str
    summary: str
    revision: int
    snippet: str
    rrf_score: float
    matched_by_vector: bool
    matched_by_fts: bool


@dataclass(frozen=True, slots=True)
class MemorySearchResult:

    mode: SearchMode
    candidates: tuple[MemorySearchCandidate, ...]
    query: str
    degrade_reason: str | None = None


class MemorySearchSettings:

    # 函数说明：MemorySearchSettings.__init__
    # 用途：初始化 MemorySearchSettings；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   top_k：保留的检索候选数量，类型 `int`；默认 `5`。
    #   chunk_chars：字符数量或字符预算，类型 `int`；默认 `900`。
    #   chunk_overlap_chars：字符数量或字符预算，类型 `int`；默认 `180`。
    #   max_chunks_per_memory：记忆输入或配置值，类型 `int`；默认 `16`。
    #   candidate_multiplier：候选输入或配置值，类型 `int`；默认 `8`。
    #   min_vector_similarity：`min_vector_similarity`输入或配置值，类型 `float`；默认
    # `0.12`。
    #   snippet_chars：字符数量或字符预算，类型 `int`；默认 `360`。
    #   recall_message_max_chars：字符数量或字符预算，类型 `int`；默认 `2400`。
    #   query_max_chars：字符数量或字符预算，类型 `int`；默认 `1600`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `top_k <= 0 or chunk_chars <= 0` 时，抛出
    # `ValueError('search settings limits must be positive')`。
    #   当 `not 0 <= chunk_overlap_chars < chunk_chars` 时，抛出
    # `ValueError('chunk overlap must be within [0, chunk_chars)')`。
    #   当 `not 0.0 <= min_vector_similarity < 1.0` 时，抛出
    # `ValueError('min_vector_similarity must be within [0, 1)')`。
    # 副作用与资源：
    #   更新对象字段：`self.top_k`、`self.chunk_chars`、`self.chunk_overlap_chars`、
    # `self.max_chunks_per_memory`、`self.candidate_multiplier`、
    # `self.min_vector_similarity`、`self.snippet_chars`、
    # `self.recall_message_max_chars` 等 9 个字段。
    def __init__(
        self,
        *,
        top_k: int = 5,
        chunk_chars: int = 900,
        chunk_overlap_chars: int = 180,
        max_chunks_per_memory: int = 16,
        candidate_multiplier: int = 8,
        min_vector_similarity: float = 0.12,
        snippet_chars: int = 360,
        recall_message_max_chars: int = 2_400,
        query_max_chars: int = 1_600,
    ) -> None:
        if top_k <= 0 or chunk_chars <= 0:
            raise ValueError("search settings limits must be positive")
        if not 0 <= chunk_overlap_chars < chunk_chars:
            raise ValueError("chunk overlap must be within [0, chunk_chars)")
        if not 0.0 <= min_vector_similarity < 1.0:
            raise ValueError("min_vector_similarity must be within [0, 1)")
        self.top_k = top_k
        self.chunk_chars = chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.max_chunks_per_memory = max_chunks_per_memory
        self.candidate_multiplier = candidate_multiplier
        self.min_vector_similarity = min_vector_similarity
        self.snippet_chars = snippet_chars
        self.recall_message_max_chars = recall_message_max_chars
        self.query_max_chars = query_max_chars


_SCHEMA_CHUNKS = """
CREATE TABLE IF NOT EXISTS memory_chunks (
    memory_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    embedding_model TEXT,
    embedding_dim INTEGER,
    embedding BLOB,
    PRIMARY KEY (memory_id, chunk_index)
);
"""

_SCHEMA_META = """
CREATE TABLE IF NOT EXISTS search_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# 函数说明：chunk_memory_text
# 用途：按段落和重叠字符预算切分记忆内容，保留标题与摘要，限制块数量并为每块计算 SHA-256
# 。
# 参数：
#   record：待处理的数据记录，类型 `MemoryRecord`。
#   settings：业务或模型设置，类型 `MemorySearchSettings`。
# 返回：类型 `tuple[tuple[str, str], ...]`；返回 `tuple(chunks)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `'\n\n'.join(current)[-settings.chunk_overlap_chars:].lstrip` →
# `hashlib.sha256(text.encode('utf-8')).hexdigest` → `hashlib.sha256` → `text.encode`。
# 分支与异常：
#   当 `not paragraph` 时，跳过当前循环项。
def chunk_memory_text(
    record: MemoryRecord,
    *,
    settings: MemorySearchSettings,
) -> tuple[tuple[str, str], ...]:

    header = f"{record.title} | {record.summary}"
    paragraphs = [part.strip() for part in record.content.split("\n\n")]
    bodies: list[str] = []
    current: list[str] = []
    used = 0
    for paragraph in paragraphs:
        if not paragraph:
            continue
        addition = len(paragraph) + (1 if current else 0)
        if current and used + addition > settings.chunk_chars:
            bodies.append("\n\n".join(current))
            overlap = "\n\n".join(current)[-settings.chunk_overlap_chars :].lstrip()
            current = [overlap] if overlap else []
            used = len(overlap)
        current.append(paragraph)
        used += addition
    if current:
        bodies.append("\n\n".join(current))
    if not bodies:
        bodies = [record.content]
    bodies = bodies[: settings.max_chunks_per_memory]
    chunks: list[tuple[str, str]] = []
    for body in bodies:
        text = f"{header}\n{body}"[: max(settings.chunk_chars * 2, len(header) + 200)]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunks.append((text, digest))
    return tuple(chunks)


# 函数说明：_fts_match_expression
# 用途：清理查询词并限制词数，给每个词加引号后以 OR 组合，避免直接把输入当成 FTS 查询语
# 法。
# 参数：
#   query：检索查询文本，类型 `str`。
#   max_terms：`max_terms`输入或配置值，类型 `int`；默认 `12`。
# 返回：类型 `str | None`；按分支返回 `None`；`' OR '.join(expressions)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`query.replace('\r', ' ').replace` →
# `query.replace`。
# 分支与异常：
#   当 `len(term) < 2` 时，跳过当前循环项。
#   当 `not terms` 时，返回 `None`。
def _fts_match_expression(query: str, *, max_terms: int = 12) -> str | None:

    terms: list[str] = []
    for raw in query.replace("\r", " ").replace("\n", " ").split():
        term = raw.strip().strip("\"'()*")
        if len(term) < 2:
            continue
        terms.append(term)
    if not terms:
        return None
    expressions = [f'"{term}"' for term in terms[:max_terms]]
    return " OR ".join(expressions)


# 函数说明：_cosine_similarity
# 用途：计算已归一化向量的点积作为余弦相似度；空向量或维度不同返回 0，本函数不执行归一化
# 。
# 参数：
#   left：传给 `len` 的输入，类型 `array`。
#   right：传给 `len` 的输入，类型 `array`。
# 返回：类型 `float`；按分支返回 `0.0`；`dot`。
# 分支与异常：
#   当 `len(left) != len(right) or not len(left)` 时，返回 `0.0`。
def _cosine_similarity(
    left: array,
    right: array,
) -> float:
    if len(left) != len(right) or not len(left):
        return 0.0
    dot = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
    return dot


# 函数说明：_decode_embedding
# 用途：解码向量，供长期记忆管理与检索使用。
# 参数：
#   blob：传给 `values.frombytes` 的输入，类型 `bytes`。
# 返回：类型 `array`；返回 `values`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`array` → `values.frombytes`。
def _decode_embedding(blob: bytes) -> array:
    values = array("f")
    values.frombytes(blob)
    return values


# 函数说明：_normalize_vector
# 用途：规范化`vector`，供长期记忆管理与检索使用。
# 参数：
#   vector：传给 `tuple` 的输入，类型 `tuple[float, ...]`。
# 返回：类型 `tuple[tuple[float, ...], bytes]`；返回
# `(normalized, array('f', normalized).tobytes())`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`math.sqrt` →
# `array('f', normalized).tobytes` → `array`。
def _normalize_vector(vector: tuple[float, ...]) -> tuple[tuple[float, ...], bytes]:
    norm = math.sqrt(sum(value * value for value in vector))
    normalized = (
        tuple(value / norm for value in vector) if norm > 0 else tuple(vector)
    )
    return normalized, array("f", normalized).tobytes()


class MemorySearchIndex:

    # 函数说明：MemorySearchIndex.__init__
    # 用途：初始化 MemorySearchIndex；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   database_path：SQLite 数据库路径，类型 `str | Path`。
    #   embedding：向量输入或配置值，类型 `EmbeddingAdapter | None`。
    #   settings：业务或模型设置，类型 `MemorySearchSettings | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(database_path).expanduser().resolve` → `Path(database_path).expanduser` →
    # `Path` → `MemorySearchSettings` → `asyncio.Lock`。
    # 副作用与资源：
    #   更新对象字段：`self.database_path`、`self.embedding`、`self.settings`、
    # `self._write_lock`、`self._fts_available`、`self._embeddings_ready`、
    # `self._initialized`。
    def __init__(
        self,
        database_path: str | Path,
        *,
        embedding: EmbeddingAdapter | None,
        settings: MemorySearchSettings | None = None,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.embedding = embedding
        self.settings = settings or MemorySearchSettings()
        self._write_lock = asyncio.Lock()
        self._fts_available = False
        self._embeddings_ready = False
        self._initialized = False


    # 函数说明：MemorySearchIndex.initialize
    # 用途：初始化MemorySearchIndex，供长期记忆管理与检索使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.database_path.parent.mkdir`
    # → `self._open_schema` → `logger.warning` → `self.database_path.unlink`。
    # 分支与异常：
    #   捕获 `aiosqlite.Error` 后，执行异常处理调用 `logger.warning`、
    # `self.database_path.unlink`、`self._open_schema`。
    # 副作用与资源：
    #   更新对象字段：`self._initialized`。
    #   文件或资源访问：`self.database_path.parent.mkdir`、`self.database_path.unlink`。
    async def initialize(self) -> None:

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self._open_schema()
        except aiosqlite.Error:
            logger.warning(
                "memory search index unreadable, rebuilding: %s",
                self.database_path,
            )
            self.database_path.unlink(missing_ok=True)
            await self._open_schema()
        self._initialized = True

    # 函数说明：MemorySearchIndex._open_schema
    # 用途：打开结构定义，供长期记忆管理与检索使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `database.executescript` → `self._meta_get` →
    # `self._meta_set` → `self._ensure_fts_table`；另有 1 个调用点。
    # 副作用与资源：
    #   数据库操作：PRAGMA、DROP IF、DELETE memory_chunks；连接与事务边界以 with/提交语
    # 句为准。
    async def _open_schema(self) -> None:
        async with self._connect() as database:
            await database.execute("PRAGMA journal_mode=WAL")
            await database.executescript(_SCHEMA_META)
            await database.executescript(_SCHEMA_CHUNKS)
            stored_version = await self._meta_get(database, "schema_version")
            if stored_version != str(_SCHEMA_VERSION):
                await database.execute("DROP TABLE IF EXISTS memory_fts")
                await database.execute("DELETE FROM memory_chunks")
                await self._meta_set(database, "schema_version", _SCHEMA_VERSION)
            tokenizer = await self._ensure_fts_table(database)
            await self._meta_set(database, "fts_tokenizer", tokenizer)
            await database.commit()

    # 函数说明：MemorySearchIndex._ensure_fts_table
    # 用途：确保`fts_table`，供长期记忆管理与检索使用。
    # 参数：
    #   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
    # 返回：类型 `str`；按分支返回 `tokenizer`；`'none'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute` →
    # `database.commit` → `logger.info`。
    # 分支与异常：
    #   捕获 `aiosqlite.Error` 后，执行异常处理调用 `logger.info`。
    # 副作用与资源：
    #   更新对象字段：`self._fts_available`。
    async def _ensure_fts_table(self, database: aiosqlite.Connection) -> str:
        for tokenizer in ("trigram", "unicode61"):
            try:
                await database.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
                    "text, memory_id UNINDEXED, chunk_index UNINDEXED, "
                    f"tokenize='{tokenizer}')"
                )
                await database.commit()
                self._fts_available = True
                return tokenizer
            except aiosqlite.Error as exc:
                logger.info("fts5 tokenizer %s unavailable: %s", tokenizer, exc)
        self._fts_available = False
        return "none"

    # 函数说明：MemorySearchIndex.reconcile
    # 用途：核对并协调MemorySearchIndex，供长期记忆管理与检索使用。
    # 参数：
    #   records：记录集合输入或配置值，类型 `tuple[MemoryRecord, ...]`。
    #   generate_embeddings：向量集合输入或配置值，类型 `bool`；默认 `True`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute_fetchall` → `self._remove_rows` → `self._upsert_rows`。
    # 资源/并发边界：`self._write_lock`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `not self._initialized` 时，返回 `None`。
    #   当 `record.status is not MemoryStatus.ACTIVE` 时，跳过当前循环项。
    # 副作用与资源：
    #   数据库操作：SELECT memory_chunks；连接与事务边界以 with/提交语句为准。
    async def reconcile(
        self,
        records: tuple[MemoryRecord, ...],
        *,
        generate_embeddings: bool = True,
    ) -> None:

        if not self._initialized:
            return
        active_ids = {
            record.id for record in records if record.status is MemoryStatus.ACTIVE
        }
        async with self._write_lock:
            async with self._connect() as database:
                indexed_ids = {
                    row[0]
                    for row in await database.execute_fetchall(
                        "SELECT DISTINCT memory_id FROM memory_chunks"
                    )
                }
            for stale_id in sorted(indexed_ids - active_ids):
                await self._remove_rows(stale_id)
            for record in records:
                if record.status is not MemoryStatus.ACTIVE:
                    continue
                await self._upsert_rows(
                    record,
                    generate_embeddings=generate_embeddings,
                )


    # 函数说明：MemorySearchIndex.upsert
    # 用途：新增或更新MemorySearchIndex，供长期记忆管理与检索使用。
    # 参数：
    #   record：待处理的数据记录，类型 `MemoryRecord`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._upsert_rows`。
    # 资源/并发边界：`self._write_lock`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `not self._initialized` 时，返回 `None`。
    async def upsert(self, record: MemoryRecord) -> None:
        if not self._initialized:
            return
        async with self._write_lock:
            await self._upsert_rows(record)

    # 函数说明：MemorySearchIndex.remove
    # 用途：移除MemorySearchIndex，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._remove_rows`。
    # 资源/并发边界：`self._write_lock`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `not self._initialized` 时，返回 `None`。
    async def remove(self, memory_id: str) -> None:
        if not self._initialized:
            return
        async with self._write_lock:
            await self._remove_rows(memory_id)

    # 函数说明：MemorySearchIndex._upsert_rows
    # 用途：新增或更新数据库行集合，供长期记忆管理与检索使用。
    # 参数：
    #   record：待处理的数据记录，类型 `MemoryRecord`。
    #   generate_embeddings：向量集合输入或配置值，类型 `bool`；默认 `True`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`chunk_memory_text` →
    # `self._connect` → `database.execute_fetchall` → `self._delete_chunk` →
    # `database.execute` → `self._embed_texts`；另有 3 个调用点。
    # 分支与异常：
    #   捕获 `Exception` 后，执行异常处理调用 `logger.warning`。
    # 副作用与资源：
    #   数据库操作：SELECT memory_chunks、INSERT memory_chunks、INSERT memory_fts；连接
    # 与事务边界以 with/提交语句为准。
    async def _upsert_rows(
        self,
        record: MemoryRecord,
        *,
        generate_embeddings: bool = True,
    ) -> None:
        chunks = chunk_memory_text(record, settings=self.settings)
        model_name = self._embedding_model_name
        try:
            async with self._connect() as database:
                rows = await database.execute_fetchall(
                    "SELECT chunk_index, text_sha256, embedding_model, embedding "
                    "FROM memory_chunks WHERE memory_id = ?",
                    (record.id,),
                )
                existing = {row[0]: (row[1], row[2], row[3]) for row in rows}
                reusable: list[int] = []
                pending: list[tuple[int, str, str]] = []
                for index, (text, digest) in enumerate(chunks):
                    current = existing.get(index)
                    if (
                        current is not None
                        and current[0] == digest
                        and current[1] == model_name
                        and (current[2] is not None or self.embedding is None)
                    ):
                        reusable.append(index)
                    else:
                        pending.append((index, text, digest))
                for index in existing:
                    if index >= len(chunks):
                        await self._delete_chunk(database, record.id, index)
                if reusable:
                    placeholders = ",".join("?" * len(reusable))
                    await database.execute(
                        "UPDATE memory_chunks SET revision = ? "
                        f"WHERE memory_id = ? AND chunk_index IN ({placeholders})",
                        (record.revision, record.id, *reusable),
                    )
                if pending:
                    vectors = (
                        await self._embed_texts(tuple(item[1] for item in pending))
                        if generate_embeddings
                        else None
                    )
                    for position, (index, text, digest) in enumerate(pending):
                        vector = vectors[position] if vectors is not None else None
                        if vector is not None:
                            normalized, blob = _normalize_vector(vector)
                            dim = len(normalized)
                            model = model_name
                        else:
                            blob = None
                            dim = None
                            model = None
                        await self._delete_chunk(database, record.id, index)
                        await database.execute(
                            "INSERT OR REPLACE INTO memory_chunks ("
                            "memory_id, chunk_index, revision, title, summary, "
                            "text, text_sha256, embedding_model, embedding_dim, "
                            "embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                record.id,
                                index,
                                record.revision,
                                record.title,
                                record.summary,
                                text,
                                digest,
                                model,
                                dim,
                                blob,
                            ),
                        )
                        if self._fts_available:
                            await database.execute(
                                "INSERT INTO memory_fts ("
                                "text, memory_id, chunk_index) VALUES (?, ?, ?)",
                                (text, record.id, index),
                            )
                await database.commit()
        except Exception as exc:
            logger.warning(
                "memory search index sync failed for %s: %s",
                record.id,
                exc,
            )

    # 函数说明：MemorySearchIndex.backfill_embeddings
    # 用途：为尚无向量的记忆块补齐嵌入并写回索引。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute_fetchall` → `self._embed_texts` → `logger.warning` →
    # `_normalize_vector` → `database.executemany`；另有 1 个调用点。
    # 资源/并发边界：`self._write_lock`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `not self._initialized or self.embedding is None` 时，返回 `None`。
    #   `not rows` 分支在完成前置处理后返回 `None`。
    #   当 `vectors is None` 时，返回 `None`。
    #   `len(vectors) != len(rows)` 分支在完成前置处理后返回 `None`。
    #   捕获 `Exception` 后，执行异常处理调用 `logger.warning`。
    # 副作用与资源：
    #   更新对象字段：`self._embeddings_ready`。
    #   数据库操作：SELECT memory_chunks、UPDATE memory_chunks；连接与事务边界以 with/提
    # 交语句为准。
    async def backfill_embeddings(self) -> None:

        """为尚无向量的记忆块补齐嵌入并写回索引。"""
        if not self._initialized or self.embedding is None:
            return
        model_name = self._embedding_model_name
        try:
            async with self._connect() as database:
                rows = await database.execute_fetchall(
                    "SELECT memory_id, chunk_index, text, text_sha256 "
                    "FROM memory_chunks WHERE embedding IS NULL "
                    "OR embedding_model IS NULL OR embedding_model != ?",
                    (model_name,),
                )
            if not rows:
                self._embeddings_ready = True
                return
            texts = tuple(str(row[2]) for row in rows)
            vectors = await self._embed_texts(texts)
            if vectors is None:
                return
            if len(vectors) != len(rows):
                logger.warning(
                    "memory embedding response count mismatch: expected=%s actual=%s",
                    len(rows),
                    len(vectors),
                )
                return
            updates: list[tuple[str | None, int, bytes, str, int, str]] = []
            for row, vector in zip(rows, vectors, strict=True):
                normalized, blob = _normalize_vector(vector)
                updates.append(
                    (
                        model_name, len(normalized), blob,
                        str(row[0]), int(row[1]), str(row[3]),
                    )
                )
            async with self._write_lock:
                async with self._connect() as database:
                    # 仍按摘要校验，检索期间被更新的文本不会写入过期向量。
                    await database.executemany(
                        "UPDATE memory_chunks SET embedding_model = ?, "
                        "embedding_dim = ?, embedding = ? WHERE memory_id = ? "
                        "AND chunk_index = ? AND text_sha256 = ?",
                        updates,
                    )
                    await database.commit()
            self._embeddings_ready = True
        except Exception as exc:
            logger.warning("memory embedding backfill failed: %s", exc)

    # 函数说明：MemorySearchIndex._remove_rows
    # 用途：移除数据库行集合，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `database.commit` → `logger.warning`。
    # 分支与异常：
    #   捕获 `aiosqlite.Error` 后，执行异常处理调用 `logger.warning`。
    # 副作用与资源：
    #   数据库操作：DELETE memory_chunks、DELETE memory_fts；连接与事务边界以 with/提交
    # 语句为准。
    async def _remove_rows(self, memory_id: str) -> None:
        try:
            async with self._connect() as database:
                await database.execute(
                    "DELETE FROM memory_chunks WHERE memory_id = ?",
                    (memory_id,),
                )
                if self._fts_available:
                    await database.execute(
                        "DELETE FROM memory_fts WHERE memory_id = ?",
                        (memory_id,),
                    )
                await database.commit()
        except aiosqlite.Error as exc:
            logger.warning(
                "memory search index remove failed for %s: %s",
                memory_id,
                exc,
            )

    # 函数说明：MemorySearchIndex._delete_chunk
    # 用途：删除`chunk`，供长期记忆管理与检索使用。
    # 参数：
    #   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
    #   memory_id：目标记忆标识，类型 `str`。
    #   chunk_index：索引输入或配置值，类型 `int`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute`。
    # 副作用与资源：
    #   数据库操作：DELETE memory_chunks、DELETE memory_fts；连接与事务边界以 with/提交
    # 语句为准。
    async def _delete_chunk(
        self,
        database: aiosqlite.Connection,
        memory_id: str,
        chunk_index: int,
    ) -> None:
        await database.execute(
            "DELETE FROM memory_chunks WHERE memory_id = ? AND chunk_index = ?",
            (memory_id, chunk_index),
        )
        if self._fts_available:
            await database.execute(
                "DELETE FROM memory_fts WHERE memory_id = ? AND chunk_index = ?",
                (memory_id, chunk_index),
            )

    # 函数说明：MemorySearchIndex._embed_texts
    # 用途：生成向量`texts`，供长期记忆管理与检索使用。
    # 参数：
    #   texts：待处理的文本集合，类型 `tuple[str, ...]`。
    # 返回：类型 `tuple[tuple[float, ...], ...] | None`；按分支返回 `None`；
    # `await self.embedding.embed_documents(texts)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.embedding.embed_documents` →
    #  `logger.warning`。
    # 分支与异常：
    #   当 `self.embedding is None` 时，返回 `None`。
    #   捕获 `Exception` 后，返回 `None`。
    async def _embed_texts(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...] | None:
        if self.embedding is None:
            return None
        try:
            return await self.embedding.embed_documents(texts)
        except Exception as exc:
            logger.warning("memory embedding batch failed: %s", exc)
            return None


    # 函数说明：MemorySearchIndex.search
    # 用途：组合全文与向量匹配结果，返回排序后的记忆命中项。
    # 参数：
    #   query：检索查询文本，类型 `str`。
    #   limit：本次返回或处理的数量上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `MemorySearchResult`；返回 `MemorySearchResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MemorySearchResult` →
    # `self._vector_search` → `self._fts_search` → `_first_hit_per_memory` →
    # `matched_ids.add` → `MemorySearchCandidate`。
    # 分支与异常：
    #   当 `not self._initialized` 时，返回 `MemorySearchResult(…)`。
    #   当 `fts_hits is None and vector_hits is None` 时，返回 `MemorySearchResult(…)`。
    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> MemorySearchResult:

        """组合全文与向量匹配结果，返回排序后的记忆命中项。"""
        if not self._initialized:
            return MemorySearchResult(
                mode=SearchMode.UNAVAILABLE,
                candidates=(),
                query=query,
                degrade_reason="index not initialized",
            )
        effective_limit = limit or self.settings.top_k
        fetch = max(
            effective_limit * self.settings.candidate_multiplier,
            effective_limit,
        )
        degrade_reason: str | None = None

        vector_hits = await self._vector_search(query, fetch)
        if vector_hits is None:
            degrade_reason = "embedding unavailable or failed"
        fts_hits = await self._fts_search(query, fetch)
        if fts_hits is None and vector_hits is None:
            return MemorySearchResult(
                mode=SearchMode.UNAVAILABLE,
                candidates=(),
                query=query,
                degrade_reason=degrade_reason or "no retrieval path available",
            )

        scores: dict[str, float] = {}
        best_hits: dict[str, _ChunkHit] = {}
        matched_vector: set[str] = set()
        matched_fts: set[str] = set()
        for hits, matched_ids in (
            (vector_hits, matched_vector), (fts_hits, matched_fts),
        ):
            # 每条检索路径只给同一记忆计分一次，保留原来的 RRF 权重。
            unique_hits = _first_hit_per_memory(hits or ())
            for rank, hit in enumerate(unique_hits, start=1):
                memory_id = hit.memory_id
                matched_ids.add(memory_id)
                scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (_RRF_K + rank)
                previous = best_hits.get(memory_id)
                if previous is None or hit.score > previous.score:
                    best_hits[memory_id] = hit

        ordered = sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))
        candidates: list[MemorySearchCandidate] = []
        for memory_id in ordered[:effective_limit]:
            hit = best_hits[memory_id]
            candidates.append(
                MemorySearchCandidate(
                    memory_id=memory_id,
                    title=hit.title or "",
                    summary="",
                    revision=0,
                    snippet=hit.text[: self.settings.snippet_chars],
                    rrf_score=scores[memory_id],
                    matched_by_vector=memory_id in matched_vector,
                    matched_by_fts=memory_id in matched_fts,
                )
            )
        if vector_hits is not None and fts_hits is not None:
            mode = SearchMode.HYBRID
        elif vector_hits is not None:
            mode = SearchMode.VECTOR
        else:
            mode = SearchMode.FTS
        return MemorySearchResult(
            mode=mode,
            candidates=tuple(candidates),
            query=query,
            degrade_reason=degrade_reason,
        )

    # 函数说明：MemorySearchIndex._vector_search
    # 用途：检索`vector`，供长期记忆管理与检索使用。
    # 参数：
    #   query：检索查询文本，类型 `str`。
    #   fetch：`fetch`输入或配置值，类型 `int`。
    # 返回：类型 `tuple[_ChunkHit, ...] | None`；按分支返回 `None`；
    # `tuple(hits[:fetch])`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.embedding.embed_query` →
    # `logger.warning` → `_normalize_vector` → `array` → `self._connect` →
    # `database.execute_fetchall`；另有 4 个调用点。
    # 分支与异常：
    #   当 `self.embedding is None or not self._embeddings_ready` 时，返回 `None`。
    #   捕获 `Exception` 后，返回 `None`。
    #   捕获 `aiosqlite.Error` 后，返回 `None`。
    #   当 `len(vector) != len(query_vector)` 时，跳过当前循环项。
    # 副作用与资源：
    #   数据库操作：SELECT memory_chunks；连接与事务边界以 with/提交语句为准。
    async def _vector_search(
        self,
        query: str,
        fetch: int,
    ) -> tuple[_ChunkHit, ...] | None:
        if self.embedding is None or not self._embeddings_ready:
            return None
        try:
            query_vector_raw = await self.embedding.embed_query(query)
        except Exception as exc:
            logger.warning("memory query embedding failed: %s", exc)
            return None
        normalized_query, _ = _normalize_vector(query_vector_raw)
        query_vector = array("f", normalized_query)
        try:
            async with self._connect() as database:
                rows = await database.execute_fetchall(
                    "SELECT memory_id, chunk_index, title, text, embedding "
                    "FROM memory_chunks WHERE embedding IS NOT NULL"
                )
        except aiosqlite.Error as exc:
            logger.warning("memory vector search failed: %s", exc)
            return None
        hits: list[_ChunkHit] = []
        for memory_id, chunk_index, title, text, blob in rows:
            vector = _decode_embedding(blob)
            if len(vector) != len(query_vector):
                continue
            similarity = _cosine_similarity(query_vector, vector)
            if similarity < self.settings.min_vector_similarity:
                continue
            hits.append(
                _ChunkHit(
                    memory_id=memory_id,
                    chunk_index=chunk_index,
                    title=title or "",
                    text=text or "",
                    score=similarity,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.memory_id, hit.chunk_index))
        return tuple(hits[:fetch])

    # 函数说明：MemorySearchIndex._fts_search
    # 用途：检索`fts`，供长期记忆管理与检索使用。
    # 参数：
    #   query：检索查询文本，类型 `str`。
    #   fetch：`fetch`输入或配置值，类型 `int`。
    # 返回：类型 `tuple[_ChunkHit, ...] | None`；按分支返回 `None`；`tuple(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_fts_match_expression` →
    # `self._connect` → `database.execute_fetchall` → `logger.warning` → `_ChunkHit`。
    # 分支与异常：
    #   当 `not self._fts_available` 时，返回 `None`。
    #   当 `expression is None` 时，返回 `None`。
    #   捕获 `aiosqlite.Error` 后，返回 `None`。
    # 副作用与资源：
    #   数据库操作：SELECT memory_fts；连接与事务边界以 with/提交语句为准。
    async def _fts_search(
        self,
        query: str,
        fetch: int,
    ) -> tuple[_ChunkHit, ...] | None:
        if not self._fts_available:
            return None
        expression = _fts_match_expression(query)
        if expression is None:
            return None
        try:
            async with self._connect() as database:
                rows = await database.execute_fetchall(
                    "SELECT memory_id, chunk_index, text FROM memory_fts "
                    "WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
                    (expression, fetch),
                )
        except aiosqlite.Error as exc:
            logger.warning("memory fts search failed: %s", exc)
            return None
        return tuple(
            _ChunkHit(
                memory_id=memory_id,
                chunk_index=chunk_index,
                title="",
                text=text or "",
                score=0.0,
            )
            for memory_id, chunk_index, text in rows
        )


    # 函数说明：MemorySearchIndex._embedding_model_name
    # 用途：返回 `self.embedding.model_name if self.embedding is not None else None`，提
    # 供 MemorySearchIndex 的派生值。
    # 返回：类型 `str | None`；返回
    # `self.embedding.model_name if self.embedding is not None else None`。
    @property
    def _embedding_model_name(self) -> str | None:
        return self.embedding.model_name if self.embedding is not None else None

    # 函数说明：MemorySearchIndex._connect
    # 用途：连接MemorySearchIndex，供长期记忆管理与检索使用。
    # 返回：类型 `aiosqlite.Connection`；返回 `aiosqlite.connect(self.database_path)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`aiosqlite.connect`。
    def _connect(self) -> aiosqlite.Connection:
        return aiosqlite.connect(self.database_path)

    # 函数说明：MemorySearchIndex._meta_get
    # 用途：获取`meta`，供长期记忆管理与检索使用。
    # 参数：
    #   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
    #   key：字段名或查询键，类型 `str`。
    # 返回：类型 `str | None`；返回 `rows[0][0] if rows else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute_fetchall`。
    # 副作用与资源：
    #   数据库操作：SELECT search_meta；连接与事务边界以 with/提交语句为准。
    @staticmethod
    async def _meta_get(
        database: aiosqlite.Connection,
        key: str,
    ) -> str | None:
        rows = await database.execute_fetchall(
            "SELECT value FROM search_meta WHERE key = ?",
            (key,),
        )
        return rows[0][0] if rows else None

    # 函数说明：MemorySearchIndex._meta_set
    # 用途：设置`meta`，供长期记忆管理与检索使用。
    # 参数：
    #   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
    #   key：字段名或查询键，类型 `str`。
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute`。
    # 副作用与资源：
    #   数据库操作：INSERT search_meta；连接与事务边界以 with/提交语句为准。
    @staticmethod
    async def _meta_set(
        database: aiosqlite.Connection,
        key: str,
        value: str,
    ) -> None:
        await database.execute(
            "INSERT OR REPLACE INTO search_meta (key, value) VALUES (?, ?)",
            (key, value),
        )


@dataclass(frozen=True, slots=True)
class _ChunkHit:

    memory_id: str
    chunk_index: int
    title: str
    text: str
    score: float


# 函数说明：_first_hit_per_memory
# 用途：在长期记忆管理与检索中处理 `_first_hit_per_memory`，通过 `seen.add` 完成首个内部
# 处理步骤。
# 参数：
#   hits：`hits`输入或配置值，类型 `tuple[_ChunkHit, ...]`。
# 返回：类型 `tuple[_ChunkHit, ...]`；返回 `tuple(unique)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`seen.add`。
# 分支与异常：
#   当 `hit.memory_id in seen` 时，跳过当前循环项。
def _first_hit_per_memory(hits: tuple[_ChunkHit, ...]) -> tuple[_ChunkHit, ...]:

    seen: set[str] = set()
    unique: list[_ChunkHit] = []
    for hit in hits:
        if hit.memory_id in seen:
            continue
        seen.add(hit.memory_id)
        unique.append(hit)
    return tuple(unique)


__all__ = [
    "DEFAULT_SEARCH_DATABASE_NAME",
    "MemorySearchCandidate",
    "MemorySearchIndex",
    "MemorySearchResult",
    "MemorySearchSettings",
    "SearchMode",
]
