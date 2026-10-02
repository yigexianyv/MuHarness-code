
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.domain.memory import (
    CORE_MEMORY_MESSAGE_NAME,
    MEMORY_INDEX_MESSAGE_NAME,
    MEMORY_RECALL_MESSAGE_NAME,
    MEMORY_SEARCH_TOOL_NAME,
    FakeEmbeddingAdapter,
    MemoryManager,
    MemoryRecallQueryInputs,
    MemoryRecord,
    SearchMode,
)
from app.domain.memory.recall import MemoryRecallCandidate, MemoryRecallSnapshot
from app.domain.memory.search_index import MemorySearchIndex, _ChunkHit
from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResult,
)
from app.runtime.agent.context_session import RuntimeContextSession
from app.runtime.agent.events import AgentEventType, InMemoryEventHandler
from app.runtime.agent.result import AgentStopReason, ToolCallRecord
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.agent.runtime_helpers import recalled_memory_revisions
from app.tools.registry import ToolRegistry


class CountingFakeEmbedding(FakeEmbeddingAdapter):

    # 函数说明：CountingFakeEmbedding.__init__
    # 用途：初始化 CountingFakeEmbedding；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.query_calls`。
    def __init__(self) -> None:
        super().__init__()
        self.query_calls = 0

    # 函数说明：CountingFakeEmbedding.embed_query
    # 用途：生成向量查询，供回归测试与测试辅助使用。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `tuple[float, ...]`；返回 `await super().embed_query(text)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().embed_query` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.query_calls`。
    async def embed_query(self, text: str) -> tuple[float, ...]:
        self.query_calls += 1
        return await super().embed_query(text)


class FailingEmbedding:

    # 函数说明：FailingEmbedding.model_name
    # 用途：返回 `'failing-embedding'`，提供 FailingEmbedding 的派生值。
    # 返回：类型 `str`；返回 `'failing-embedding'`。
    @property
    def model_name(self) -> str:
        return "failing-embedding"

    # 函数说明：FailingEmbedding.dimensions
    # 用途：返回 `8`，提供 FailingEmbedding 的派生值。
    # 返回：类型 `int | None`；返回 `8`。
    @property
    def dimensions(self) -> int | None:
        return 8

    # 函数说明：FailingEmbedding.embed_documents
    # 用途：生成向量`documents`，供回归测试与测试辅助使用。
    # 参数：
    #   texts：待处理的文本集合。
    # 返回：不返回结果值（隐式 None）。
    async def embed_documents(self, texts):
        raise RuntimeError("embedding service offline")

    # 函数说明：FailingEmbedding.embed_query
    # 用途：生成向量查询，供回归测试与测试辅助使用。
    # 参数：
    #   text：待处理的文本，类型 `str`。
    # 返回：类型 `tuple[float, ...]`；不返回结果值（隐式 None）。
    async def embed_query(self, text: str) -> tuple[float, ...]:
        raise RuntimeError("embedding service offline")

    # 函数说明：FailingEmbedding.close
    # 用途：关闭FailingEmbedding，供回归测试与测试辅助使用。
    # 返回：类型 `None`；无结果值，显式返回 None。
    async def close(self) -> None:
        return None


class BlockingEmbedding(FakeEmbeddingAdapter):

    # 函数说明：BlockingEmbedding.__init__
    # 用途：初始化 BlockingEmbedding；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super` →
    # `asyncio.Event`。
    # 副作用与资源：
    #   更新对象字段：`self.started`、`self.release`。
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    # 函数说明：BlockingEmbedding.embed_documents
    # 用途：生成向量`documents`，供回归测试与测试辅助使用。
    # 参数：
    #   texts：待处理的文本集合。
    # 返回：返回 `await super().embed_documents(texts)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.started.set` →
    # `self.release.wait` → `super().embed_documents` → `super`。
    async def embed_documents(self, texts):
        self.started.set()
        await self.release.wait()
        return await super().embed_documents(texts)


# 函数说明：memory_root
# 用途：返回 `tmp_path / 'memory'`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `Path`；返回 `tmp_path / 'memory'`。
@pytest.fixture
def memory_root(tmp_path: Path) -> Path:
    return tmp_path / "memory"


# 函数说明：_seed
# 用途：在回归测试与测试辅助中处理 `_seed`，通过 `manager.create` 完成首个内部处理步骤。
# 参数：
#   manager：当前业务管理器，类型 `MemoryManager`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`manager.create`。
async def _seed(manager: MemoryManager) -> None:
    await manager.create(
        title="数据库迁移决定",
        summary="生产库从 MySQL 迁移到 PostgreSQL 的最终决定",
        content=(
            "2024-12 定稿：生产环境数据库从 MySQL 整体迁移到 PostgreSQL 16，"
            "原因是事务隔离与窗口函数需求。迁移脚本放在 deploy/pg-migration。"
        ),
    )
    await manager.create(
        title="前端构建决定",
        summary="前端打包从 webpack 换成 vite",
        content=(
            "2024-11 定稿：前端构建工具从 webpack 5 迁移到 vite 5，"
            "提升本地开发热更新速度。"
        ),
    )




# 函数说明：test_chinese_paraphrase_recall_ranks_relevant_memory_first
# 用途：回归验证回归测试与测试辅助中的
# `chinese_paraphrase_recall_ranks_relevant_memory_first` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`CountingFakeEmbedding` →
# `MemoryManager` → `manager.initialize` → `_seed` → `manager.search`。
# 分支与异常：
#   验证条件：`result.mode is SearchMode.HYBRID`。
#   验证条件：`result.candidates`。
#   验证条件：`result.candidates[0].memory_id == 'M001'`。
#   验证条件：`result.candidates[0].title == '数据库迁移决定'`。
@pytest.mark.asyncio
async def test_chinese_paraphrase_recall_ranks_relevant_memory_first(
    memory_root: Path,
) -> None:
    embedding = CountingFakeEmbedding()
    manager = MemoryManager(memory_root, embedding=embedding)
    await manager.initialize()
    await _seed(manager)

    result = await manager.search("当初数据库迁移到 PG 是怎么定的")

    assert result.mode is SearchMode.HYBRID
    assert result.candidates, "语义改写查询应召回相关记忆"
    assert result.candidates[0].memory_id == "M001"
    assert result.candidates[0].title == "数据库迁移决定"
    assert result.candidates[0].revision == 1




# 函数说明：test_exact_keyword_recall_via_fts5_without_embedding
# 用途：回归验证回归测试与测试辅助中的 `exact_keyword_recall_via_fts5_without_embedding`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` → `manager.initialize`
#  → `_seed` → `manager.search`。
# 分支与异常：
#   验证条件：`result.mode is SearchMode.FTS`。
#   验证条件：`[candidate.memory_id for candidate in result.candidates] == ['M001']`。
#   验证条件：`result.candidates[0].matched_by_fts is True`。
#   验证条件：`result.candidates[0].matched_by_vector is False`。
@pytest.mark.asyncio
async def test_exact_keyword_recall_via_fts5_without_embedding(
    memory_root: Path,
) -> None:
    manager = MemoryManager(memory_root, embedding=None)
    await manager.initialize()
    await _seed(manager)

    result = await manager.search("pg-migration")

    assert result.mode is SearchMode.FTS
    assert [candidate.memory_id for candidate in result.candidates] == ["M001"]
    assert result.candidates[0].matched_by_fts is True
    assert result.candidates[0].matched_by_vector is False


# 函数说明：test_recall_message_counts_separator_and_final_newlines
# 用途：回归验证回归测试与测试辅助中的
# `recall_message_counts_separator_and_final_newlines` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryRecallCandidate` →
# `MemoryRecallSnapshot` → `snapshot.render_message`。
# 分支与异常：
#   验证条件：`full is not None and full.content is not None`。
#   验证条件：`exact is not None`。
#   验证条件：`exact.content == full.content`。
#   验证条件：`snapshot.render_message(max_chars=len(full.content) - 1) is None`。
def test_recall_message_counts_separator_and_final_newlines() -> None:
    candidate = MemoryRecallCandidate(
        memory_id="M001", title="记录标题", summary="记录摘要", revision=1,
        snippet="独立的正文片段", rrf_score=1.0,
        matched_by_vector=False, matched_by_fts=True,
    )
    snapshot = MemoryRecallSnapshot(
        query="记录", mode=SearchMode.FTS, candidates=(candidate,),
    )
    full = snapshot.render_message(max_chars=10_000)
    assert full is not None and full.content is not None

    exact = snapshot.render_message(max_chars=len(full.content))
    assert exact is not None
    assert exact.content == full.content
    assert snapshot.render_message(max_chars=len(full.content) - 1) is None


# 函数说明：test_recall_message_keeps_whole_candidates_within_budget
# 用途：回归验证回归测试与测试辅助中的
# `recall_message_keeps_whole_candidates_within_budget` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryRecallCandidate` →
# `MemoryRecallSnapshot` → `single.render_message` → `combined.render_message`。
# 分支与异常：
#   验证条件：`expected is not None and expected.content is not None`。
#   验证条件：`message is not None and message.content is not None`。
#   验证条件：`message.content == expected.content`。
#   验证条件：`len(message.content) <= budget`。
def test_recall_message_keeps_whole_candidates_within_budget() -> None:
    first = MemoryRecallCandidate(
        memory_id="M001", title="第一条", summary="摘要一", revision=1,
        snippet="", rrf_score=1.0,
        matched_by_vector=False, matched_by_fts=True,
    )
    second = MemoryRecallCandidate(
        memory_id="M002", title="第二条", summary="摘要二", revision=2,
        snippet="另一条记忆", rrf_score=0.5,
        matched_by_vector=False, matched_by_fts=True,
    )
    single = MemoryRecallSnapshot("记录", SearchMode.FTS, (first,))
    expected = single.render_message(max_chars=10_000)
    assert expected is not None and expected.content is not None
    budget = len(expected.content)
    combined = MemoryRecallSnapshot("记录", SearchMode.FTS, (first, second))
    message = combined.render_message(max_chars=budget)

    assert message is not None and message.content is not None
    assert message.content == expected.content
    assert len(message.content) <= budget
    assert "M002" not in message.content


# 函数说明：test_backfill_does_not_attach_stale_vector_after_text_update
# 用途：回归验证回归测试与测试辅助中的
# `backfill_does_not_attach_stale_vector_after_text_update` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`BlockingEmbedding` →
# `MemorySearchIndex` → `index.initialize` → `datetime.now` → `MemoryRecord` →
# `index.reconcile`；另有 9 个调用点。
# 分支与异常：
#   验证条件：`len(rows) == 1`。
#   验证条件：`'新部署流程' in text`。
#   验证条件：`revision == 2`。
#   验证条件：`vector is None`。
# 副作用与资源：
#   数据库操作：SELECT memory_chunks；连接与事务边界以 with/提交语句为准。
#   创建后台异步任务；任务取消、等待和异常处理由本函数及调用方的生命周期代码管理。
@pytest.mark.asyncio
async def test_backfill_does_not_attach_stale_vector_after_text_update(
    tmp_path: Path,
) -> None:
    embedding = BlockingEmbedding()
    index = MemorySearchIndex(tmp_path / "search.sqlite", embedding=embedding)
    await index.initialize()
    now = datetime.now(UTC)
    record = MemoryRecord(
        id="M001", title="部署流程", summary="流程摘要", content="旧部署流程",
        created_at=now, updated_at=now, last_accessed_at=now,
    )
    await index.reconcile((record,), generate_embeddings=False)
    backfill = asyncio.create_task(index.backfill_embeddings())
    try:
        await asyncio.wait_for(embedding.started.wait(), timeout=5)
        fields = record.model_dump()
        fields.update(content="新部署流程", revision=2)
        updated = MemoryRecord.model_validate(fields)
        await index.reconcile((updated,), generate_embeddings=False)
    finally:
        embedding.release.set()
        await asyncio.wait_for(backfill, timeout=5)

    async with index._connect() as database:
        rows = await database.execute_fetchall(
            "SELECT text, revision, embedding FROM memory_chunks WHERE memory_id = ?",
            (record.id,),
        )
    assert len(rows) == 1
    text, revision, vector = rows[0]
    assert "新部署流程" in text
    assert revision == 2
    assert vector is None




# 函数说明：test_hybrid_rrf_merges_and_dedupes_by_memory_id
# 用途：回归验证回归测试与测试辅助中的 `hybrid_rrf_merges_and_dedupes_by_memory_id` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` → `manager.search`。
# 分支与异常：
#   验证条件：`len(ids) == len(set(ids))`。
#   验证条件：`result.mode is SearchMode.HYBRID`。
#   验证条件：`top.memory_id == 'M001'`。
#   验证条件：`top.matched_by_vector and top.matched_by_fts`。
@pytest.mark.asyncio
async def test_hybrid_rrf_merges_and_dedupes_by_memory_id(
    memory_root: Path,
) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)

    result = await manager.search("数据库 迁移 PostgreSQL 部署脚本")

    ids = [candidate.memory_id for candidate in result.candidates]
    assert len(ids) == len(set(ids)), "同一 memory 的多个 Chunk 必须合并"
    assert result.mode is SearchMode.HYBRID
    top = result.candidates[0]
    assert top.memory_id == "M001"
    assert top.matched_by_vector and top.matched_by_fts
    if len(result.candidates) > 1:
        assert top.rrf_score > result.candidates[1].rrf_score


# 函数说明：test_rrf_prefers_dual_path_hit_over_single_path
# 用途：回归验证回归测试与测试辅助中的 `rrf_prefers_dual_path_hit_over_single_path` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `manager.create` → `manager.search` →
# `next`。
# 分支与异常：
#   验证条件：`'M002' in ids`。
#   验证条件：`m002.rrf_score >= m001.rrf_score`。
@pytest.mark.asyncio
async def test_rrf_prefers_dual_path_hit_over_single_path(
    memory_root: Path,
) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await manager.create(
        title="唯一关键词",
        summary="包含独占代码标识",
        content="冷门标识 XYZUNIQUE-42 只出现在这条记忆里。",
    )
    await manager.create(
        title="共享关键词",
        summary="同时包含主题词与独占标识",
        content="数据库迁移主题；另一条独占标识是 XYZUNIQUE-43。",
    )

    result = await manager.search("数据库 迁移 XYZUNIQUE-42")

    ids = [candidate.memory_id for candidate in result.candidates]
    assert "M002" in ids
    if "M001" in ids and len(ids) > 1:
        m002 = next(c for c in result.candidates if c.memory_id == "M002")
        m001 = next(c for c in result.candidates if c.memory_id == "M001")
        assert m002.rrf_score >= m001.rrf_score




# 函数说明：test_update_invalidates_stale_vectors
# 用途：回归验证回归测试与测试辅助中的 `update_invalidates_stale_vectors` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `manager.create` → `manager.search` →
# `manager.update`。
# 分支与异常：
#   验证条件：`[candidate.memory_id for candidate in before.candidates] == ['M001']`。
#   验证条件：`[candidate.memory_id for candidate in after_old.candidates] == []`。
#   验证条件：`[candidate.memory_id for candidate in after_new.candidates] == ['M001']`
# 。
#   验证条件：`after_new.candidates[0].revision == 2`。
@pytest.mark.asyncio
async def test_update_invalidates_stale_vectors(memory_root: Path) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    record = await manager.create(
        title="部署配置",
        summary="线上部署脚本位置与回滚流程",
        content="部署脚本 deploy/RELEASE-9F27.sh，回滚用 deploy/ROLLBACK-9F27.sh。",
    )

    before = await manager.search("RELEASE-9F27")
    assert [candidate.memory_id for candidate in before.candidates] == ["M001"]

    await manager.update(
        record.id,
        content="部署改为 GitHub Actions 流水线，回滚走 re-deploy 旧版本 tag。",
        reason="迁移 CI",
    )

    after_old = await manager.search("RELEASE-9F27")
    assert [candidate.memory_id for candidate in after_old.candidates] == []
    after_new = await manager.search("GitHub Actions 流水线")
    assert [candidate.memory_id for candidate in after_new.candidates] == ["M001"]
    assert after_new.candidates[0].revision == 2


# 函数说明：test_archive_removes_memory_from_search
# 用途：回归验证回归测试与测试辅助中的 `archive_removes_memory_from_search` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` → `manager.archive` →
# `manager.search`。
# 分支与异常：
#   验证条件：`'M002' not in [candidate.memory_id for candidate in result.candidates]`。
@pytest.mark.asyncio
async def test_archive_removes_memory_from_search(memory_root: Path) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)
    await manager.archive("M002", reason="过时")

    result = await manager.search("vite webpack 前端构建")
    assert "M002" not in [candidate.memory_id for candidate in result.candidates]




# 函数说明：test_manual_markdown_edit_rebuilds_index
# 用途：回归验证回归测试与测试辅助中的 `manual_markdown_edit_rebuilds_index` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` →
# `path.read_text(encoding='utf-8').replace` → `path.read_text`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result.candidates`。
#   验证条件：`result.candidates[0].memory_id == 'M002'`。
#   验证条件：`'M002' not in [candidate.memory_id for candidate in stale.candidates]`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`、`path.write_text`。
@pytest.mark.asyncio
async def test_manual_markdown_edit_rebuilds_index(memory_root: Path) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)

    path = memory_root / "active" / "M002.md"
    text = path.read_text(encoding="utf-8").replace("webpack", "Rspack")
    path.write_text(text, encoding="utf-8")

    rebuilt = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await rebuilt.initialize()

    result = await rebuilt.search("Rspack 构建")
    assert result.candidates
    assert result.candidates[0].memory_id == "M002"
    stale = await rebuilt.search("webpack")
    assert "M002" not in [candidate.memory_id for candidate in stale.candidates]


# 函数说明：test_deleted_projection_file_is_rebuilt
# 用途：回归验证回归测试与测试辅助中的 `deleted_projection_file_is_rebuilt` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` →
# `(memory_root / 'search.sqlite').unlink` → `rebuilt.initialize`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`rebuilt.hybrid_recall_enabled is True`。
#   验证条件：`[candidate.memory_id for candidate in result.candidates] == ['M001']`。
# 副作用与资源：
#   文件或资源访问：`(memory_root / 'search.sqlite').unlink`。
@pytest.mark.asyncio
async def test_deleted_projection_file_is_rebuilt(memory_root: Path) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)
    (memory_root / "search.sqlite").unlink()

    rebuilt = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await rebuilt.initialize()
    assert rebuilt.hybrid_recall_enabled is True

    result = await rebuilt.search("pg-migration")
    assert [candidate.memory_id for candidate in result.candidates] == ["M001"]




# 函数说明：test_embedding_failure_degrades_to_fts
# 用途：回归验证回归测试与测试辅助中的 `embedding_failure_degrades_to_fts` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` → `FailingEmbedding` →
#  `manager.initialize` → `_seed` → `manager.search`。
# 分支与异常：
#   验证条件：`result.mode is SearchMode.FTS`。
#   验证条件：`result.degrade_reason is not None`。
#   验证条件：`'M001' in [candidate.memory_id for candidate in result.candidates]`。
@pytest.mark.asyncio
async def test_embedding_failure_degrades_to_fts(memory_root: Path) -> None:
    manager = MemoryManager(memory_root, embedding=FailingEmbedding())
    await manager.initialize()
    await _seed(manager)

    result = await manager.search("PostgreSQL 迁移")

    assert result.mode is SearchMode.FTS
    assert result.degrade_reason is not None
    assert "M001" in [candidate.memory_id for candidate in result.candidates]


# 函数说明：test_embedding_backfill_does_not_block_manager_initialize
# 用途：回归验证回归测试与测试辅助中的
# `embedding_backfill_does_not_block_manager_initialize` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `seed_manager.initialize` → `seed_manager.create` → `seed_manager.close` →
# `BlockingEmbedding` → `asyncio.wait_for`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`before.mode is SearchMode.FTS`。
#   验证条件：`manager._search_backfill_task is not None`。
#   验证条件：`after.mode is SearchMode.HYBRID`。
@pytest.mark.asyncio
async def test_embedding_backfill_does_not_block_manager_initialize(
    memory_root: Path,
) -> None:
    seed_manager = MemoryManager(memory_root, embedding=None)
    await seed_manager.initialize()
    await seed_manager.create(
        title="启动期向量补全",
        summary="Host 不应等待远程 Embedding",
        content="FTS 投影先可用，向量随后在后台补齐。",
    )
    await seed_manager.close()

    embedding = BlockingEmbedding()
    manager = MemoryManager(memory_root, embedding=embedding)
    await asyncio.wait_for(manager.initialize(), timeout=0.5)

    await asyncio.wait_for(embedding.started.wait(), timeout=0.5)
    before = await manager.search("启动期向量补全")
    assert before.mode is SearchMode.FTS

    embedding.release.set()
    assert manager._search_backfill_task is not None
    await asyncio.wait_for(manager._search_backfill_task, timeout=0.5)
    after = await manager.search("启动期向量补全")
    assert after.mode is SearchMode.HYBRID
    await manager.close()


# 函数说明：test_corrupted_index_does_not_break_markdown
# 用途：回归验证回归测试与测试辅助中的 `corrupted_index_does_not_break_markdown` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` →
# `(memory_root / 'search.sqlite').write_bytes` → `manager.search`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`result.mode is SearchMode.UNAVAILABLE`。
#   验证条件：`record.id == 'M003'`。
#   验证条件：`loaded is not None`。
# 副作用与资源：
#   文件或资源访问：`(memory_root / 'search.sqlite').write_bytes`。
@pytest.mark.asyncio
async def test_corrupted_index_does_not_break_markdown(memory_root: Path) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)

    (memory_root / "search.sqlite").write_bytes(b"not a sqlite file")

    result = await manager.search("PostgreSQL")
    assert result.mode is SearchMode.UNAVAILABLE
    record = await manager.create(
        title="新增记忆",
        summary="索引损坏后仍可写入",
        content="Markdown 是唯一权威存储，索引损坏不阻塞写入。",
    )
    assert record.id == "M003"
    loaded = await manager.store.load("M003")
    assert loaded is not None




# 函数说明：test_search_and_recall_do_not_increment_access_count
# 用途：回归验证回归测试与测试辅助中的 `search_and_recall_do_not_increment_access_count`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` → `manager.search` →
# `manager.recall`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`snapshot.candidates`。
#   验证条件：`record is not None`。
#   验证条件：`record.access_count == 0`。
@pytest.mark.asyncio
async def test_search_and_recall_do_not_increment_access_count(
    memory_root: Path,
) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)

    await manager.search("数据库迁移")
    snapshot = await manager.recall(
        MemoryRecallQueryInputs(user_message="数据库迁移怎么定的")
    )
    assert snapshot.candidates

    record = await manager.store.load("M001")
    assert record is not None
    assert record.access_count == 0

    await manager.read("M001")
    record = await manager.store.load("M001")
    assert record is not None
    assert record.access_count == 1




# 函数说明：test_context_messages_hybrid_injects_recall_instead_of_index
# 用途：回归验证回归测试与测试辅助中的
# `context_messages_hybrid_injects_recall_instead_of_index` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `manager.core.update` → `_seed` →
# `manager.recall`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`snapshot.candidates`。
#   验证条件：`MEMORY_INDEX_MESSAGE_NAME not in names`。
#   验证条件：`MEMORY_RECALL_MESSAGE_NAME in names`。
#   验证条件：`CORE_MEMORY_MESSAGE_NAME in names`。
@pytest.mark.asyncio
async def test_context_messages_hybrid_injects_recall_instead_of_index(
    memory_root: Path,
) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await manager.core.update("用户身份：开发者")
    await _seed(manager)

    snapshot = await manager.recall(
        MemoryRecallQueryInputs(user_message="前端构建工具换成了什么")
    )
    assert snapshot.candidates

    hybrid_messages = await manager.context_messages(recall=snapshot)
    names = [message.name for message in hybrid_messages]
    assert MEMORY_INDEX_MESSAGE_NAME not in names
    assert MEMORY_RECALL_MESSAGE_NAME in names
    assert CORE_MEMORY_MESSAGE_NAME in names
    recall_message = next(
        message
        for message in hybrid_messages
        if message.name == MEMORY_RECALL_MESSAGE_NAME
    )
    content = recall_message.content or ""
    assert "可能相关的记忆" in content
    assert "自动召回不算读取" in content
    assert "memory_read" in content
    assert "2024-11 定稿" not in content.replace("Snippet: ", "") or (
        content.count("2024-11 定稿") <= 1
    )

    legacy_messages = await manager.context_messages()
    legacy_names = [message.name for message in legacy_messages]
    assert MEMORY_INDEX_MESSAGE_NAME in legacy_names
    assert MEMORY_RECALL_MESSAGE_NAME not in legacy_names


# 函数说明：test_runtime_index_failure_falls_back_to_legacy_index
# 用途：回归验证回归测试与测试辅助中的
# `runtime_index_failure_falls_back_to_legacy_index` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` →
# `(memory_root / 'search.sqlite').write_bytes` → `RuntimeContextSession`；另有 3 个调用
# 点。
# 分支与异常：
#   验证条件：`MEMORY_INDEX_MESSAGE_NAME in names`。
#   验证条件：`MEMORY_RECALL_MESSAGE_NAME not in names`。
#   验证条件：`context.recall_mode == SearchMode.UNAVAILABLE.value`。
# 副作用与资源：
#   文件或资源访问：`(memory_root / 'search.sqlite').write_bytes`。
@pytest.mark.asyncio
async def test_runtime_index_failure_falls_back_to_legacy_index(
    memory_root: Path,
) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)
    (memory_root / "search.sqlite").write_bytes(b"broken at runtime")

    session = RuntimeContextSession(
        memory_manager=manager,
        skill_store=None,
        skill_context_provider=None,
        task_context_provider=None,
        recall_query=MemoryRecallQueryInputs(user_message="数据库迁移决定是什么"),
    )
    context = await session.build(
        conversation_id="conversation-runtime-fallback",
        recovery_checkpoint=None,
        trailing_system_messages=(),
    )

    names = [message.name for message in context.messages]
    assert MEMORY_INDEX_MESSAGE_NAME in names
    assert MEMORY_RECALL_MESSAGE_NAME not in names
    assert context.recall_mode == SearchMode.UNAVAILABLE.value
    await manager.close()


# 函数说明：test_recall_message_respects_top5_and_char_budget
# 用途：回归验证回归测试与测试辅助中的 `recall_message_respects_top5_and_char_budget` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `manager.create` → `manager.recall` →
# `MemoryRecallQueryInputs`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`message is not None`。
#   验证条件：`len(snapshot.candidates) <= 5`。
#   验证条件：`len(content) <= manager.search_settings.recall_message_max_chars + 200`。
@pytest.mark.asyncio
async def test_recall_message_respects_top5_and_char_budget(
    memory_root: Path,
) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    for index in range(7):
        await manager.create(
            title=f"共享主题 {index}",
            summary=f"关于部署流程的第 {index} 条记忆",
            content=(
                f"部署流程说明 {index}：先跑测试，再构建镜像，最后滚动发布。"
                "重复的正文用于验证预算截断行为。"
            ),
        )

    snapshot = await manager.recall(
        MemoryRecallQueryInputs(user_message="部署流程是怎么规定的")
    )
    message = snapshot.render_message(
        max_chars=manager.search_settings.recall_message_max_chars
    )
    assert message is not None
    assert len(snapshot.candidates) <= 5
    content = message.content or ""
    assert len(content) <= manager.search_settings.recall_message_max_chars + 200


# 函数说明：test_empty_recall_returns_no_message
# 用途：回归验证回归测试与测试辅助中的 `empty_recall_returns_no_message` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` → `manager.recall` →
# `MemoryRecallQueryInputs`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`message is None`。
@pytest.mark.asyncio
async def test_empty_recall_returns_no_message(memory_root: Path) -> None:
    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)

    snapshot = await manager.recall(
        MemoryRecallQueryInputs(user_message="今天天气怎么样")
    )
    message = snapshot.render_message()
    assert message is None




# 函数说明：test_memory_search_tool_returns_candidates_without_full_content
# 用途：回归验证回归测试与测试辅助中的
# `memory_search_tool_returns_candidates_without_full_content` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` → `MemorySearchTool` →
# `tool.execute`。
# 分支与异常：
#   验证条件：`output['available'] is True`。
#   验证条件：`output['mode'] == 'hybrid'`。
#   验证条件：`results`。
#   验证条件：`results[0]['memory_id'] == 'M001'`。
@pytest.mark.asyncio
async def test_memory_search_tool_returns_candidates_without_full_content(
    memory_root: Path,
) -> None:
    from app.domain.memory import MemorySearchTool

    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)

    tool = MemorySearchTool(manager)
    output = await tool.execute({"query": "数据库迁移 PostgreSQL"})

    assert output["available"] is True
    assert output["mode"] == "hybrid"
    results = output["results"]
    assert results
    assert results[0]["memory_id"] == "M001"
    assert results[0]["revision"] == 1
    assert "content" not in results[0]
    assert "snippet" in results[0]

    missing = await tool.execute({"query": "不存在的主题 XYZNONE"})
    assert missing["results"] == []


# 函数说明：test_memory_search_tool_reports_unavailable_when_index_down
# 用途：回归验证回归测试与测试辅助中的
# `memory_search_tool_reports_unavailable_when_index_down` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` →
# `(memory_root / 'search.sqlite').write_bytes` → `MemorySearchTool`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`output['available'] is False`。
#   验证条件：`output['reason']`。
# 副作用与资源：
#   文件或资源访问：`(memory_root / 'search.sqlite').write_bytes`。
@pytest.mark.asyncio
async def test_memory_search_tool_reports_unavailable_when_index_down(
    memory_root: Path,
) -> None:
    from app.domain.memory import MemorySearchTool

    manager = MemoryManager(memory_root, embedding=FakeEmbeddingAdapter())
    await manager.initialize()
    await _seed(manager)
    (memory_root / "search.sqlite").write_bytes(b"broken")

    tool = MemorySearchTool(manager)
    output = await tool.execute({"query": "数据库"})

    assert output["available"] is False
    assert output["reason"]


# 函数说明：test_memory_search_registered_and_not_deferred
# 用途：回归验证回归测试与测试辅助中的 `memory_search_registered_and_not_deferred` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `MemoryManager` →
# `Path` → `register_memory_tools` → `registry.definitions`。
# 分支与异常：
#   验证条件：`MEMORY_SEARCH_TOOL_NAME in names`。
#   验证条件：`MEMORY_SEARCH_TOOL_NAME not in DEFAULT_DEFERRED_MEMORY_TOOL_NAMES`。
def test_memory_search_registered_and_not_deferred() -> None:
    from app.domain.memory import (
        DEFAULT_DEFERRED_MEMORY_TOOL_NAMES,
        register_memory_tools,
    )

    registry = ToolRegistry()
    manager = MemoryManager(Path("/nonexistent-muharness-memory"))
    register_memory_tools(registry, manager)

    names = {
        definition.name
        for definition in registry.definitions(for_model=True)
    }
    assert MEMORY_SEARCH_TOOL_NAME in names
    assert MEMORY_SEARCH_TOOL_NAME not in DEFAULT_DEFERRED_MEMORY_TOOL_NAMES




class _CountingRecallManager(MemoryManager):
    # 函数说明：_CountingRecallManager.__init__
    # 用途：初始化 _CountingRecallManager；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   *args：额外位置参数，按实现向内部调用传递。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.recall_calls`。
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.recall_calls = 0

    # 函数说明：_CountingRecallManager.recall
    # 用途：召回_CountingRecallManager，供回归测试与测试辅助使用。
    # 参数：
    #   inputs：传给 `super().recall` 的输入。
    # 返回：返回 `await super().recall(inputs)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().recall` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.recall_calls`。
    async def recall(self, inputs):
        self.recall_calls += 1
        return await super().recall(inputs)


# 函数说明：_recall_messages
# 用途：召回消息序列，供回归测试与测试辅助使用。
# 参数：
#   messages：本次处理的消息序列。
# 返回：类型 `list`；返回
# `[message for message in messages if message.name == MEMORY_RECALL_MESSAGE_NAME]`。
def _recall_messages(messages) -> list:
    return [
        message
        for message in messages
        if message.name == MEMORY_RECALL_MESSAGE_NAME
    ]


# 函数说明：test_session_recalls_once_and_reuses_across_steps
# 用途：回归验证回归测试与测试辅助中的 `session_recalls_once_and_reuses_across_steps` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`CountingFakeEmbedding` →
# `_CountingRecallManager` → `manager.initialize` → `_seed` → `RuntimeContextSession` →
# `MemoryRecallQueryInputs`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`manager.recall_calls == 1`。
#   验证条件：`embedding.query_calls == 1`。
#   验证条件：`len(first_recall) == len(second_recall) == 1`。
#   验证条件：`first_recall[0].content == second_recall[0].content`。
@pytest.mark.asyncio
async def test_session_recalls_once_and_reuses_across_steps(
    memory_root: Path,
) -> None:
    embedding = CountingFakeEmbedding()
    manager = _CountingRecallManager(memory_root, embedding=embedding)
    await manager.initialize()
    await _seed(manager)

    session = RuntimeContextSession(
        memory_manager=manager,
        skill_store=None,
        skill_context_provider=None,
        task_context_provider=None,
        recall_query=MemoryRecallQueryInputs(user_message="数据库迁移决定是什么"),
    )
    first = await session.build(
        conversation_id=None,
        recovery_checkpoint=None,
        trailing_system_messages=(),
    )
    second = await session.build(
        conversation_id=None,
        recovery_checkpoint=None,
        trailing_system_messages=(),
    )

    assert manager.recall_calls == 1
    assert embedding.query_calls == 1
    first_recall = _recall_messages(first.messages)
    second_recall = _recall_messages(second.messages)
    assert len(first_recall) == len(second_recall) == 1
    assert first_recall[0].content == second_recall[0].content
    assert first.recall_candidate_ids == second.recall_candidate_ids
    assert first.recall_mode == SearchMode.HYBRID.value


# 函数说明：test_rrf_scores_each_memory_once_per_retrieval_path
# 用途：回归验证回归测试与测试辅助中的 `rrf_scores_each_memory_once_per_retrieval_path`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   monkeypatch：pytest 提供的临时替换依赖夹具，类型 `pytest.MonkeyPatch`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemorySearchIndex` →
# `index.initialize` → `monkeypatch.setattr` → `index.search` → `pytest.approx`。
# 分支与异常：
#   验证条件：
# `[candidate.memory_id for candidate in result.candidates] == ['M002', 'M001']`。
#   验证条件：`result.candidates[1].rrf_score == pytest.approx(1 / 61)`。
@pytest.mark.asyncio
async def test_rrf_scores_each_memory_once_per_retrieval_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = MemorySearchIndex(tmp_path / "rrf.sqlite", embedding=None)
    await index.initialize()

    # 函数说明：test_rrf_scores_each_memory_once_per_retrieval_path.vector_hits
    # 用途：处理回归测试与测试辅助中的 `vector_hits` 数据；结果及边界条件见下方说明。
    # 参数：
    #   query：检索查询文本，类型 `str`。
    #   fetch：`fetch`输入或配置值，类型 `int`。
    # 返回：返回 `(_ChunkHit('M001', 0, '长记忆', 'chunk 0', 0.9), _ChunkHit('M001', 1,
    # '长记忆', 'chunk 1', 0.8)…`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_ChunkHit`。
    async def vector_hits(query: str, fetch: int):
        del query, fetch
        return (
            _ChunkHit("M001", 0, "长记忆", "chunk 0", 0.9),
            _ChunkHit("M001", 1, "长记忆", "chunk 1", 0.8),
            _ChunkHit("M002", 0, "双路命中", "vector chunk", 0.7),
        )

    # 函数说明：test_rrf_scores_each_memory_once_per_retrieval_path.fts_hits
    # 用途：处理回归测试与测试辅助中的 `fts_hits` 数据；结果及边界条件见下方说明。
    # 参数：
    #   query：检索查询文本，类型 `str`。
    #   fetch：`fetch`输入或配置值，类型 `int`。
    # 返回：返回 `(_ChunkHit('M002', 0, '', 'exact keyword', 0.0),)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_ChunkHit`。
    async def fts_hits(query: str, fetch: int):
        del query, fetch
        return (_ChunkHit("M002", 0, "", "exact keyword", 0.0),)

    monkeypatch.setattr(index, "_vector_search", vector_hits)
    monkeypatch.setattr(index, "_fts_search", fts_hits)

    result = await index.search("query", limit=2)

    assert [candidate.memory_id for candidate in result.candidates] == [
        "M002",
        "M001",
    ]
    assert result.candidates[1].rrf_score == pytest.approx(1 / 61)




class RecordingModelAdapter(ModelAdapter):

    # 函数说明：RecordingModelAdapter.__init__
    # 用途：初始化 RecordingModelAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置。
    #   responses：预设的模型或服务响应序列。
    # 返回：不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.requests`。
    def __init__(self, config, responses):
        super().__init__(config)
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    # 函数说明：RecordingModelAdapter.complete
    # 用途：完成RecordingModelAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；不返回结果值（隐式 None）。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    # 函数说明：RecordingModelAdapter.complete_stream
    # 用途：完成事件流，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    #   on_text_delta：增量文本回调。
    #   on_reasoning_delta：增量推理文本回调；默认 `None`。
    # 返回：返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`on_text_delta` →
    # `self.responses.pop`。
    # 分支与异常：
    #   验证条件：`isinstance(response, ModelResponse)`。
    async def complete_stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta,
        on_reasoning_delta=None,
    ):
        self.requests.append(request)
        await on_text_delta("完成")
        response = self.responses.pop(0)
        assert isinstance(response, ModelResponse)
        return response

    # 函数说明：RecordingModelAdapter.close
    # 用途：关闭RecordingModelAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；无结果值，显式返回 None。
    async def close(self) -> None:
        return None


# 函数说明：_offline_registry
# 用途：在回归测试与测试辅助中处理 `_offline_registry`，通过 `registry.register` 完成首
# 个内部处理步骤。
# 参数：
#   responses：预设的模型或服务响应序列。
# 返回：返回 `(registry, adapter)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `RecordingModelAdapter` → `ModelAdapterRegistry` → `ModelSettings` →
# `registry.register`。
def _offline_registry(responses):
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = RecordingModelAdapter(config, responses)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)
    return registry, adapter


# 函数说明：test_runtime_recall_not_persisted_into_history_or_summary
# 用途：回归验证回归测试与测试辅助中的
# `runtime_recall_not_persisted_into_history_or_summary` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'note.txt').write_text` →
#  `_offline_registry` → `ModelResponse` → `Message` → `ToolCall` → `ModelUsage`；另有 1
# 4 个调用点。
# 分支与异常：
#   验证条件：`result.stop_reason is AgentStopReason.FINAL_ANSWER`。
#   验证条件：`manager.recall_calls == 1`。
#   验证条件：`embedding.query_calls == 1`。
#   验证条件：`len(adapter.requests) == 2`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'note.txt').write_text`。
@pytest.mark.asyncio
async def test_runtime_recall_not_persisted_into_history_or_summary(
    memory_root: Path,
    tmp_path: Path,
) -> None:
    from app.tools.builtin.read_file import ReadFileTool

    (tmp_path / "note.txt").write_text("工具输入", encoding="utf-8")
    registry, adapter = _offline_registry(
        [
            ModelResponse(
                id="r1",
                provider="fake",
                model="fake-model",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="read-1",
                            name="read_file",
                            arguments={"path": "note.txt"},
                        ),
                    ),
                ),
                usage=ModelUsage(),
            ),
            ModelResponse(
                id="r2",
                provider="fake",
                model="fake-model",
                message=Message(role=MessageRole.ASSISTANT, content="已读取"),
                usage=ModelUsage(),
            ),
        ]
    )
    embedding = CountingFakeEmbedding()
    manager = _CountingRecallManager(memory_root, embedding=embedding)
    await manager.initialize()
    await _seed(manager)

    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    events = InMemoryEventHandler()

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        memory_manager=manager,
    ).run(
        "讲讲数据库迁移的决定",
        history=(),
        conversation_id="conversation-recall",
        event_handler=events,
    )

    assert result.stop_reason is AgentStopReason.FINAL_ANSWER
    assert manager.recall_calls == 1
    assert embedding.query_calls == 1
    assert len(adapter.requests) == 2
    for request in adapter.requests:
        names = [message.name for message in request.messages]
        assert MEMORY_RECALL_MESSAGE_NAME in names
        assert MEMORY_INDEX_MESSAGE_NAME not in names
    first_names = [message.name for message in adapter.requests[0].messages]
    second_names = [message.name for message in adapter.requests[1].messages]
    first_content = next(
        message.content
        for message in adapter.requests[0].messages
        if message.name == MEMORY_RECALL_MESSAGE_NAME
    )
    second_content = next(
        message.content
        for message in adapter.requests[1].messages
        if message.name == MEMORY_RECALL_MESSAGE_NAME
    )
    assert first_content == second_content
    assert first_names.count(MEMORY_RECALL_MESSAGE_NAME) == 1
    assert second_names.count(MEMORY_RECALL_MESSAGE_NAME) == 1
    persisted_names = [message.name for message in result.messages]
    assert MEMORY_RECALL_MESSAGE_NAME not in persisted_names
    started = [
        event for event in events.events if event.type is AgentEventType.MODEL_STARTED
    ]
    assert len(started) == 2
    assert started[0].recall_candidate_ids == ("M001",)
    assert started[0].recall_mode == SearchMode.HYBRID.value
    record = await manager.store.load("M001")
    assert record is not None
    assert record.access_count == 0




# 函数说明：_tool_record
# 用途：记录工具，供回归测试与测试辅助使用。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
#   output：工具、模型或转换步骤的输出，类型 `dict`。
#   success：执行是否成功，类型 `bool`；默认 `True`。
# 返回：类型 `ToolCallRecord`；返回 `ToolCallRecord(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCallRecord` → `ToolCall` →
# `ToolResult` → `json.dumps`。
def _tool_record(
    name: str,
    arguments: dict,
    output: dict,
    *,
    success: bool = True,
) -> ToolCallRecord:
    return ToolCallRecord(
        round_index=0,
        tool_call=ToolCall(id=f"{name}-1", name=name, arguments=arguments),
        result=ToolResult(
            tool_call_id=f"{name}-1",
            tool_name=name,
            success=success,
            output=json.dumps(output, ensure_ascii=False),
            duration_ms=1.0,
        ),
    )


# 函数说明：test_memory_search_hit_does_not_authorize_reflection_update
# 用途：回归验证回归测试与测试辅助中的
# `memory_search_hit_does_not_authorize_reflection_update` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_record` →
# `recalled_memory_revisions`。
# 分支与异常：
#   验证条件：`recalled_memory_revisions((search_record,)) == {}`。
def test_memory_search_hit_does_not_authorize_reflection_update() -> None:
    search_record = _tool_record(
        "memory_search",
        {"query": "数据库迁移"},
        {
            "available": True,
            "query": "数据库迁移",
            "mode": "hybrid",
            "results": [
                {
                    "memory_id": "M001",
                    "title": "数据库迁移决定",
                    "revision": 1,
                    "snippet": "…",
                }
            ],
        },
    )

    assert recalled_memory_revisions((search_record,)) == {}


# 函数说明：test_memory_read_success_authorizes_reflection_update
# 用途：回归验证回归测试与测试辅助中的
# `memory_read_success_authorizes_reflection_update` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_record` →
# `recalled_memory_revisions`。
# 分支与异常：
#   验证条件：`recalled_memory_revisions((read_record, failed_read)) == {'M001': 3}`。
def test_memory_read_success_authorizes_reflection_update() -> None:
    read_record = _tool_record(
        "memory_read",
        {"memory_id": "M001"},
        {"found": True, "id": "M001", "revision": 3, "content": "…"},
    )
    failed_read = _tool_record(
        "memory_read",
        {"memory_id": "M002"},
        {"found": False, "memory_id": "M002"},
        success=False,
    )

    assert recalled_memory_revisions((read_record, failed_read)) == {"M001": 3}




# 函数说明：test_fake_embedding_is_deterministic
# 用途：回归验证回归测试与测试辅助中的 `fake_embedding_is_deterministic` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeEmbeddingAdapter` →
# `fake.embed_query`。
# 分支与异常：
#   验证条件：`first == second`。
#   验证条件：`first != other`。
#   验证条件：`len(first) == fake.dimensions`。
@pytest.mark.asyncio
async def test_fake_embedding_is_deterministic() -> None:
    fake = FakeEmbeddingAdapter()
    first = await fake.embed_query("生产数据库迁移到 PostgreSQL")
    second = await fake.embed_query("生产数据库迁移到 PostgreSQL")
    other = await fake.embed_query("completely unrelated text")

    assert first == second
    assert first != other
    assert len(first) == fake.dimensions




# 函数说明：test_min_vector_similarity_threshold_is_wired
# 用途：回归验证回归测试与测试辅助中的 `min_vector_similarity_threshold_is_wired` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` →
# `FakeEmbeddingAdapter` → `manager.initialize` → `_seed` → `manager.search`。
# 分支与异常：
#   验证条件：`manager.search_settings.min_vector_similarity == 0.99`。
#   验证条件：`[candidate.memory_id for candidate in result.candidates] == ['M001']`。
#   验证条件：`result.candidates[0].matched_by_vector is False`。
#   验证条件：`result.candidates[0].matched_by_fts is True`。
@pytest.mark.asyncio
async def test_min_vector_similarity_threshold_is_wired(
    memory_root: Path,
) -> None:

    manager = MemoryManager(
        memory_root,
        embedding=FakeEmbeddingAdapter(),
        min_vector_similarity=0.99,
    )
    await manager.initialize()
    await _seed(manager)

    assert manager.search_settings.min_vector_similarity == 0.99
    result = await manager.search("pg-migration")
    assert [candidate.memory_id for candidate in result.candidates] == ["M001"]
    assert result.candidates[0].matched_by_vector is False
    assert result.candidates[0].matched_by_fts is True
