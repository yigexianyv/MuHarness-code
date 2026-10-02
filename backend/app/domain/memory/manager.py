
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

from app.models.types import Message, MessageRole

from .core import DEFAULT_MAX_CORE_TOKENS, CoreMemoryEntry, CoreMemoryManager
from .embedding import EmbeddingAdapter
from .index import MemoryIndex
from .maintenance import MemoryMaintenance
from .models import MemoryRecord, MemoryStatus
from .prompts import (
    CORE_MEMORY_HEADER,
    MEMORY_POLICY_PROMPT,
)
from .recall import (
    MemoryRecallQueryInputs,
    MemoryRecallService,
    MemoryRecallSnapshot,
)
from .search_index import (
    DEFAULT_SEARCH_DATABASE_NAME,
    MemorySearchIndex,
    MemorySearchResult,
    MemorySearchSettings,
    SearchMode,
)
from .store import DEFAULT_MEMORY_DIR, MemoryStore

logger = logging.getLogger("muharness.memory.manager")

_DIRECTORY_LOCKS: dict[Path, asyncio.Lock] = {}

CORE_MEMORY_MESSAGE_NAME = "muharness_core_memory"
MEMORY_INDEX_MESSAGE_NAME = "muharness_memory_index"
MEMORY_POLICY_MESSAGE_NAME = "muharness_memory_policy"


class MemoryManager:

    # 函数说明：MemoryManager.__init__
    # 用途：初始化 MemoryManager；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   memory_dir：记忆文件目录，类型 `str | Path`；默认 `DEFAULT_MEMORY_DIR`。
    #   max_active：允许同时活跃的数量上限，类型 `int`；默认 `25`。
    #   max_core_tokens：Token 数量或 Token 预算，类型 `int`；默认
    # `DEFAULT_MAX_CORE_TOKENS`。
    #   embedding：向量输入或配置值，类型 `EmbeddingAdapter | None`；默认 `None`。
    #   search_settings：检索设置输入或配置值，类型 `MemorySearchSettings | None`；默认
    # `None`。
    #   min_vector_similarity：`min_vector_similarity`输入或配置值，类型 `float | None`
    # ；默认 `None`。
    #   hybrid_search_enabled：检索输入或配置值，类型 `bool`；默认 `True`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(memory_dir).expanduser().resolve` → `Path(memory_dir).expanduser` → `Path` →
    #  `MemoryStore` → `CoreMemoryManager` → `MemoryIndex`；另有 6 个调用点。
    # 副作用与资源：
    #   更新对象字段：`self.memory_dir`、`self.max_active`、`self.store`、`self.core`、
    # `self.index`、`self.maintenance`、`self.search_settings`、
    # `self._hybrid_search_enabled` 等 14 个字段。
    def __init__(
        self,
        memory_dir: str | Path = DEFAULT_MEMORY_DIR,
        *,
        max_active: int = 25,
        max_core_tokens: int = DEFAULT_MAX_CORE_TOKENS,
        embedding: EmbeddingAdapter | None = None,
        search_settings: MemorySearchSettings | None = None,
        min_vector_similarity: float | None = None,
        hybrid_search_enabled: bool = True,
    ) -> None:
        self.memory_dir = Path(memory_dir).expanduser().resolve()
        self.max_active = max_active
        self.store = MemoryStore(self.memory_dir, max_active=max_active)
        self.core = CoreMemoryManager(
            self.memory_dir,
            max_tokens=max_core_tokens,
        )
        self.index = MemoryIndex(self.memory_dir)
        self.maintenance = MemoryMaintenance(max_active=max_active)
        if search_settings is None and min_vector_similarity is not None:
            search_settings = MemorySearchSettings(
                min_vector_similarity=min_vector_similarity
            )
        self.search_settings = search_settings or MemorySearchSettings()
        self._hybrid_search_enabled = hybrid_search_enabled
        self._search_index = MemorySearchIndex(
            self.memory_dir / DEFAULT_SEARCH_DATABASE_NAME,
            embedding=embedding,
            settings=self.search_settings,
        )
        self._recall_service = MemoryRecallService(self)
        self._search_ok = False
        self._search_backfill_task: asyncio.Task[None] | None = None
        self._lock = _DIRECTORY_LOCKS.setdefault(self.memory_dir, asyncio.Lock())
        self._lock_path = self.memory_dir / ".memory.lock"

    # 函数说明：MemoryManager.hybrid_recall_enabled
    # 用途：召回`enabled`，供长期记忆管理与检索使用。
    # 返回：类型 `bool`；返回 `self._hybrid_search_enabled and self._search_ok`。
    @property
    def hybrid_recall_enabled(self) -> bool:

        return self._hybrid_search_enabled and self._search_ok

    # 函数说明：MemoryManager.initialize
    # 用途：初始化记忆文件、核心记忆和搜索索引，并同步现有条目。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.store.initialize` → `self.core.initialize` → `self._rebuild_index` →
    # `self._search_index.initialize` → `self._search_index.reconcile`；另有 4 个调用点
    # 。
    # 分支与异常：
    #   捕获 `Exception` 后，执行异常处理调用 `logger.warning`。
    # 副作用与资源：
    #   更新对象字段：`self._search_ok`、`self._search_backfill_task`。
    #   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
    async def initialize(self) -> None:

        """初始化记忆文件、核心记忆和搜索索引，并同步现有条目。"""
        async with self._mutation_guard():
            await self.store.initialize()
            await self.core.initialize()
            await self._rebuild_index()
        if self._hybrid_search_enabled:
            try:
                await self._search_index.initialize()
                await self._search_index.reconcile(
                    await self.store.list_active(),
                    generate_embeddings=False,
                )
                self._search_ok = True
                self._search_backfill_task = asyncio.create_task(
                    self._search_index.backfill_embeddings(),
                    name="memory-embedding-backfill",
                )
            except Exception as exc:
                self._search_ok = False
                logger.warning(
                    "memory search index unavailable, fallback to index cues: %s",
                    exc,
                )

    # 函数说明：MemoryManager.close
    # 用途：关闭MemoryManager，供长期记忆管理与检索使用。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`task.done` → `task.cancel` →
    # `suppress`。
    # 分支与异常：
    #   当 `task is None` 时，返回 `None`。
    # 副作用与资源：
    #   更新对象字段：`self._search_backfill_task`。
    async def close(self) -> None:

        task = self._search_backfill_task
        self._search_backfill_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task


    # 函数说明：MemoryManager.context_messages
    # 用途：将适用记忆转成模型上下文消息。
    # 参数：
    #   recall：`recall`输入或配置值，类型 `MemoryRecallSnapshot | None`；默认 `None`。
    # 返回：类型 `tuple[Message, ...]`；返回 `tuple(messages)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.core.load` →
    # `core_content.startswith` → `Message` → `self.index.load` →
    # `recall.render_message`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def context_messages(
        self,
        *,
        recall: MemoryRecallSnapshot | None = None,
    ) -> tuple[Message, ...]:

        """将适用记忆转成模型上下文消息。"""
        async with self._lock:
            messages: list[Message] = []
            core_text = await self.core.load()
            if core_text.strip():
                core_content = core_text.strip()
                if not core_content.startswith(CORE_MEMORY_HEADER):
                    core_content = f"{CORE_MEMORY_HEADER}\n\n{core_content}"
                messages.append(
                    Message(
                        role=MessageRole.SYSTEM,
                        name=CORE_MEMORY_MESSAGE_NAME,
                        content=core_content,
                    )
                )
            if recall is None:
                index_text = await self.index.load()
                if index_text is not None:
                    messages.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            name=MEMORY_INDEX_MESSAGE_NAME,
                            content=index_text,
                        )
                    )
            else:
                recall_message = recall.render_message(
                    max_chars=self.search_settings.recall_message_max_chars
                )
                if recall_message is not None:
                    messages.append(recall_message)
            messages.append(
                Message(
                    role=MessageRole.SYSTEM,
                    name=MEMORY_POLICY_MESSAGE_NAME,
                    content=MEMORY_POLICY_PROMPT,
                )
            )
            return tuple(messages)

    # 函数说明：MemoryManager.reflection_context
    # 用途：在长期记忆管理与检索中处理 `reflection_context`，通过 `self.core.load` 完成
    # 首个内部处理步骤。
    # 返回：类型 `tuple[str, str]`；返回 `(core_text, index_text or '')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.core.load` →
    # `self.index.load`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def reflection_context(self) -> tuple[str, str]:

        async with self._lock:
            core_text = await self.core.load()
            index_text = await self.index.load()
            return core_text, index_text or ""


    # 函数说明：MemoryManager.read
    # 用途：读取MemoryManager，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    # 返回：类型 `MemoryRecord | None`；返回 `await self.store.read(memory_id)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.store.read`。
    async def read(self, memory_id: str) -> MemoryRecord | None:

        async with self._mutation_guard():
            return await self.store.read(memory_id)

    # 函数说明：MemoryManager.list
    # 用途：列出MemoryManager，供长期记忆管理与检索使用。
    # 返回：类型 `tuple[MemoryRecord, ...]`；返回 `await self.store.list_active()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.store.list_active`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def list(self) -> tuple[MemoryRecord, ...]:

        async with self._lock:
            return await self.store.list_active()

    # 函数说明：MemoryManager.list_archived
    # 用途：列出`archived`，供长期记忆管理与检索使用。
    # 返回：类型 `tuple[MemoryRecord, ...]`；返回 `await self.store.list_archived()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.store.list_archived`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def list_archived(self) -> tuple[MemoryRecord, ...]:

        async with self._lock:
            return await self.store.list_archived()


    # 函数说明：MemoryManager.search
    # 用途：搜索记忆条目，在向量检索不可用时使用可用的索引路径。
    # 参数：
    #   query：检索查询文本，类型 `str`。
    #   limit：本次返回或处理的数量上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `MemorySearchResult`；返回 `MemorySearchResult(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MemorySearchResult` →
    # `self._search_index.search` → `self.store.load` → `replace`。
    # 分支与异常：
    #   当 `not self._search_ok` 时，返回 `MemorySearchResult(…)`。
    #   当 `record is None or record.status is not MemoryStatus.ACTIVE` 时，跳过当前循环
    # 项。
    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> MemorySearchResult:

        """搜索记忆条目，在向量检索不可用时使用可用的索引路径。"""
        if not self._search_ok:
            return MemorySearchResult(
                mode=SearchMode.UNAVAILABLE,
                candidates=(),
                query=query,
                degrade_reason="search index unavailable",
            )
        result = await self._search_index.search(query, limit=limit)
        enriched: list = []
        for candidate in result.candidates:
            record = await self.store.load(candidate.memory_id)
            if record is None or record.status is not MemoryStatus.ACTIVE:
                continue
            enriched.append(
                replace(
                    candidate,
                    title=record.title,
                    summary=record.summary,
                    revision=record.revision,
                )
            )
        return MemorySearchResult(
            mode=result.mode,
            candidates=tuple(enriched),
            query=result.query,
            degrade_reason=result.degrade_reason,
        )

    # 函数说明：MemoryManager.recall
    # 用途：根据当前任务与对话线索召回相关记忆快照。
    # 参数：
    #   inputs：传给 `self._recall_service.recall` 的输入，类型
    # `MemoryRecallQueryInputs`。
    # 返回：类型 `MemoryRecallSnapshot`；返回
    # `await self._recall_service.recall(inputs)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._recall_service.recall`。
    async def recall(
        self,
        inputs: MemoryRecallQueryInputs,
    ) -> MemoryRecallSnapshot:

        """根据当前任务与对话线索召回相关记忆快照。"""
        return await self._recall_service.recall(inputs)

    # 函数说明：MemoryManager.create
    # 用途：创建记忆并同步搜索索引。
    # 参数：
    #   title：面向用户的标题，类型 `str`。
    #   summary：已有或新生成的摘要，类型 `str`。
    #   content：内容正文，类型 `str`。
    # 返回：类型 `MemoryRecord`；返回 `record`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.store.count_active` → `self.store.create` → `self._rebuild_index` →
    # `self._sync_search_index`。
    # 分支与异常：
    #   当 `await self.store.count_active() >= self.max_active` 时，抛出
    # `ValueError('active memory capacity is full')`。
    async def create(
        self,
        *,
        title: str,
        summary: str,
        content: str,
    ) -> MemoryRecord:

        """创建记忆并同步搜索索引。"""
        async with self._mutation_guard():
            if await self.store.count_active() >= self.max_active:
                raise ValueError("active memory capacity is full")
            record = await self.store.create(
                title=title,
                summary=summary,
                content=content,
            )
            await self._rebuild_index()
            await self._sync_search_index(record)
            return record

    # 函数说明：MemoryManager.create_if_capacity
    # 用途：容量检查通过后创建记忆；容量不足时返回相应结果。
    # 参数：
    #   title：面向用户的标题，类型 `str`。
    #   summary：已有或新生成的摘要，类型 `str`。
    #   content：内容正文，类型 `str`。
    # 返回：类型 `MemoryRecord | None`；按分支返回 `None`；`record`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.store.count_active` → `self.store.create` → `self._rebuild_index` →
    # `self._sync_search_index`。
    # 分支与异常：
    #   当 `await self.store.count_active() >= self.max_active` 时，返回 `None`。
    async def create_if_capacity(
        self,
        *,
        title: str,
        summary: str,
        content: str,
    ) -> MemoryRecord | None:

        """容量检查通过后创建记忆；容量不足时返回相应结果。"""
        async with self._mutation_guard():
            if await self.store.count_active() >= self.max_active:
                return None
            record = await self.store.create(
                title=title,
                summary=summary,
                content=content,
            )
            await self._rebuild_index()
            await self._sync_search_index(record)
            return record

    # 函数说明：MemoryManager.validate_create
    # 用途：校验`create`，供长期记忆管理与检索使用。
    # 参数：
    #   title：面向用户的标题，类型 `str`。
    #   summary：已有或新生成的摘要，类型 `str`。
    #   content：内容正文，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` → `MemoryRecord`。
    @staticmethod
    def validate_create(
        *,
        title: str,
        summary: str,
        content: str,
    ) -> None:

        now = datetime.now(UTC)
        MemoryRecord(
            id="M000",
            title=title,
            summary=summary,
            content=content,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )

    # 函数说明：MemoryManager.update
    # 用途：更新MemoryManager，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    #   title：面向用户的标题，类型 `str | None`；默认 `None`。
    #   summary：已有或新生成的摘要，类型 `str | None`；默认 `None`。
    #   content：内容正文，类型 `str`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    # 返回：类型 `MemoryRecord`；返回 `record`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.store.update` → `self._rebuild_index` → `self._sync_search_index`。
    async def update(
        self,
        memory_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        content: str,
        reason: str,
    ) -> MemoryRecord:

        async with self._mutation_guard():
            record = await self.store.update(
                memory_id,
                title=title,
                summary=summary,
                content=content,
                reason=reason,
            )
            await self._rebuild_index()
            await self._sync_search_index(record)
            return record

    # 函数说明：MemoryManager.update_if_revision
    # 用途：仅在修订版本匹配时更新记忆，避免覆盖并发修改。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    #   expected_revision：`expected_revision`输入或配置值，类型 `int`。
    #   title：面向用户的标题，类型 `str`。
    #   summary：已有或新生成的摘要，类型 `str`。
    #   content：内容正文，类型 `str`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    # 返回：类型 `MemoryRecord`；返回 `record`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.store.update` → `self._rebuild_index` → `self._sync_search_index`。
    async def update_if_revision(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        title: str,
        summary: str,
        content: str,
        reason: str,
    ) -> MemoryRecord:

        """仅在修订版本匹配时更新记忆，避免覆盖并发修改。"""
        async with self._mutation_guard():
            record = await self.store.update(
                memory_id,
                title=title,
                summary=summary,
                content=content,
                reason=reason,
                expected_revision=expected_revision,
            )
            await self._rebuild_index()
            await self._sync_search_index(record)
            return record

    # 函数说明：MemoryManager.archive
    # 用途：归档MemoryManager，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    # 返回：类型 `MemoryRecord`；返回 `record`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.store.archive` → `self._rebuild_index` → `self._drop_search_index`。
    async def archive(self, memory_id: str, *, reason: str) -> MemoryRecord:

        async with self._mutation_guard():
            record = await self.store.archive(memory_id, reason=reason)
            await self._rebuild_index()
            await self._drop_search_index(record.id)
            return record

    # 函数说明：MemoryManager.archive_if_unchanged
    # 用途：仅在内容仍与预期一致时归档记忆。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    #   expected_record：记录输入或配置值，类型 `MemoryRecord`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    # 返回：类型 `MemoryRecord`；返回 `record`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.store.load` → `self.store.archive` → `self._rebuild_index` →
    # `self._drop_search_index`。
    # 分支与异常：
    #   当 `current is None` 时，抛出 `KeyError(f"memory '{memory_id}' not found")`。
    #   当 `current != expected_record` 时，抛出 `ValueError(…)`。
    async def archive_if_unchanged(
        self,
        memory_id: str,
        *,
        expected_record: MemoryRecord,
        reason: str,
    ) -> MemoryRecord:

        """仅在内容仍与预期一致时归档记忆。"""
        async with self._mutation_guard():
            current = await self.store.load(memory_id)
            if current is None:
                raise KeyError(f"memory '{memory_id}' not found")
            if current != expected_record:
                raise ValueError(
                    f"memory '{memory_id}' changed since maintenance snapshot"
                )
            record = await self.store.archive(memory_id, reason=reason)
            await self._rebuild_index()
            await self._drop_search_index(record.id)
            return record

    # 函数说明：MemoryManager.upsert_core
    # 用途：新增或更新`core`，供长期记忆管理与检索使用。
    # 参数：
    #   key：字段名或查询键，类型 `str`。
    #   value：待校验、规范化或转换的值，类型 `str`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    #   source_statement：`source_statement`输入或配置值，类型 `str`。
    # 返回：类型 `tuple[CoreMemoryEntry, bool]`；返回 `await self.core.upsert(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.core.upsert`。
    async def upsert_core(
        self,
        *,
        key: str,
        value: str,
        reason: str,
        source_statement: str,
    ) -> tuple[CoreMemoryEntry, bool]:

        async with self._mutation_guard():
            return await self.core.upsert(
                key=key,
                value=value,
                reason=reason,
                source_statement=source_statement,
            )

    # 函数说明：MemoryManager.remove_core
    # 用途：移除`core`，供长期记忆管理与检索使用。
    # 参数：
    #   key：字段名或查询键，类型 `str`。
    # 返回：类型 `CoreMemoryEntry`；返回 `await self.core.remove(key)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._mutation_guard` →
    # `self.core.remove`。
    async def remove_core(self, key: str) -> CoreMemoryEntry:

        async with self._mutation_guard():
            return await self.core.remove(key)


    # 函数说明：MemoryManager.maintenance_required
    # 用途：在长期记忆管理与检索中处理 `maintenance_required`，通过
    # `self.maintenance.exceeds_capacity` 完成首个内部处理步骤。
    # 返回：类型 `bool`；返回
    # `self.maintenance.exceeds_capacity(await self.store.count_active())`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self.maintenance.exceeds_capacity` → `self.store.count_active`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def maintenance_required(self) -> bool:

        async with self._lock:
            return self.maintenance.exceeds_capacity(
                await self.store.count_active()
            )

    # 函数说明：MemoryManager.active_count
    # 用途：统计活跃项，供长期记忆管理与检索使用。
    # 返回：类型 `int`；返回 `await self.store.count_active()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.store.count_active`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def active_count(self) -> int:

        async with self._lock:
            return await self.store.count_active()

    # 函数说明：MemoryManager.has_capacity
    # 用途：判断`capacity`是否满足当前实现的条件。
    # 参数：
    #   required_slots：`required_slots`输入或配置值，类型 `int`；默认 `1`。
    # 返回：类型 `bool`；返回 `active_count + required_slots <= self.max_active`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.store.count_active`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `required_slots < 0` 时，抛出
    # `ValueError('required_slots cannot be negative')`。
    async def has_capacity(self, *, required_slots: int = 1) -> bool:

        if required_slots < 0:
            raise ValueError("required_slots cannot be negative")
        async with self._lock:
            active_count = await self.store.count_active()
            return active_count + required_slots <= self.max_active

    # 函数说明：MemoryManager.retention_candidates
    # 用途：在长期记忆管理与检索中处理 `retention_candidates`，通过
    # `self.store.list_active` 完成首个内部处理步骤。
    # 参数：
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `5`。
    # 返回：类型 `tuple[MemoryRecord, ...]`；返回
    # `self.maintenance.select_candidates(active, limit=limit)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.store.list_active` →
    # `self.maintenance.select_candidates`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    async def retention_candidates(
        self,
        *,
        limit: int = 5,
    ) -> tuple[MemoryRecord, ...]:

        async with self._lock:
            active = await self.store.list_active()
            return self.maintenance.select_candidates(active, limit=limit)

    # 函数说明：MemoryManager._rebuild_index
    # 用途：重建索引，供长期记忆管理与检索使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.index.rebuild` →
    # `self.store.list_active`。
    async def _rebuild_index(self) -> None:
        await self.index.rebuild(await self.store.list_active())

    # 函数说明：MemoryManager._sync_search_index
    # 用途：检索索引，供长期记忆管理与检索使用。
    # 参数：
    #   record：待处理的数据记录，类型 `MemoryRecord`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._search_index.upsert` →
    # `logger.warning`。
    # 分支与异常：
    #   当 `not self._search_ok` 时，返回 `None`。
    #   捕获 `Exception` 后，执行异常处理调用 `logger.warning`。
    async def _sync_search_index(self, record: MemoryRecord) -> None:

        if not self._search_ok:
            return
        try:
            await self._search_index.upsert(record)
        except Exception as exc:
            logger.warning(
                "memory search index sync failed for %s: %s",
                record.id,
                exc,
            )

    # 函数说明：MemoryManager._drop_search_index
    # 用途：检索索引，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._search_index.remove` →
    # `logger.warning`。
    # 分支与异常：
    #   当 `not self._search_ok` 时，返回 `None`。
    #   捕获 `Exception` 后，执行异常处理调用 `logger.warning`。
    async def _drop_search_index(self, memory_id: str) -> None:

        if not self._search_ok:
            return
        try:
            await self._search_index.remove(memory_id)
        except Exception as exc:
            logger.warning(
                "memory search index remove failed for %s: %s",
                memory_id,
                exc,
            )

    # 函数说明：MemoryManager._mutation_guard
    # 用途：检查边界`mutation`，供长期记忆管理与检索使用。
    # 返回：异步生成器，逐项产出 `None`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread` →
    # `self._acquire_file_lock` → `self._release_file_lock`。
    # 资源/并发边界：`self._lock`，上下文退出时执行相应清理。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    @asynccontextmanager
    async def _mutation_guard(self) -> AsyncIterator[None]:

        async with self._lock:
            await asyncio.to_thread(self.memory_dir.mkdir, parents=True, exist_ok=True)
            handle = await asyncio.to_thread(self._lock_path.open, "a+b")
            try:
                await self._acquire_file_lock(handle)
                yield
            finally:
                await self._release_file_lock(handle)

    # 函数说明：MemoryManager._acquire_file_lock
    # 用途：获取文件，供长期记忆管理与检索使用。
    # 参数：
    #   handle：传给 `asyncio.to_thread` 的输入，类型 `BinaryIO`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread` →
    # `handle.fileno`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    @staticmethod
    async def _acquire_file_lock(handle: BinaryIO) -> None:
        if fcntl is not None:
            await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            await asyncio.to_thread(_lock_windows_file, handle)

    # 函数说明：MemoryManager._release_file_lock
    # 用途：释放文件，供长期记忆管理与检索使用。
    # 参数：
    #   handle：传给 `asyncio.to_thread` 的输入，类型 `BinaryIO`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread` →
    # `handle.fileno`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    @staticmethod
    async def _release_file_lock(handle: BinaryIO) -> None:
        try:
            if fcntl is not None:
                await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                await asyncio.to_thread(_unlock_windows_file, handle)
        finally:
            await asyncio.to_thread(handle.close)


# 函数说明：_lock_windows_file
# 用途：获取锁文件，供长期记忆管理与检索使用。
# 参数：
#   handle：`handle`输入或配置值，类型 `BinaryIO`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`handle.seek` → `handle.read` →
# `handle.write` → `handle.flush` → `msvcrt.locking` → `handle.fileno`。
def _lock_windows_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if not handle.read(1):
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)


# 函数说明：_unlock_windows_file
# 用途：释放锁文件，供长期记忆管理与检索使用。
# 参数：
#   handle：`handle`输入或配置值，类型 `BinaryIO`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`handle.seek` → `msvcrt.locking` →
# `handle.fileno`。
def _unlock_windows_file(handle: BinaryIO) -> None:
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


__all__ = [
    "CORE_MEMORY_MESSAGE_NAME",
    "MEMORY_INDEX_MESSAGE_NAME",
    "MEMORY_POLICY_MESSAGE_NAME",
    "MemoryManager",
    "MemoryRecallQueryInputs",
    "MemoryRecallSnapshot",
    "MemorySearchResult",
    "MemorySearchSettings",
    "SearchMode",
]
