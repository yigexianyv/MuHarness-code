
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from app.domain.memory import (
    CORE_MEMORY_MESSAGE_NAME,
    MEMORY_INDEX_MESSAGE_NAME,
    MEMORY_POLICY_MESSAGE_NAME,
    MEMORY_POLICY_PROMPT,
    CoreMemoryManager,
    MemoryManager,
    MemoryRecord,
    MemoryStatus,
    parse_memory_markdown,
    register_memory_tools,
    register_memory_write_tools,
)
from app.domain.memory import store as memory_store
from app.domain.memory.store import MemoryStore
from app.models.types import ToolCall
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry


# 函数说明：memory_root
# 用途：返回 `tmp_path / 'memory'`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `Path`；返回 `tmp_path / 'memory'`。
@pytest.fixture
def memory_root(tmp_path: Path) -> Path:
    return tmp_path / "memory"


# 函数说明：_manager
# 用途：在回归测试与测试辅助中处理 `_manager`，通过 `manager.initialize` 完成首个内部处
# 理步骤。
# 参数：
#   root：当前操作的根目录，类型 `Path`。
# 返回：类型 `MemoryManager`；返回 `manager`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` → `manager.initialize`
# 。
async def _manager(root: Path) -> MemoryManager:
    manager = MemoryManager(root)
    await manager.initialize()
    return manager




# 函数说明：test_core_loads_empty_when_missing
# 用途：回归验证回归测试与测试辅助中的 `core_loads_empty_when_missing` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.core.load`。
# 分支与异常：
#   验证条件：`await manager.core.load() == ''`。
@pytest.mark.asyncio
async def test_core_loads_empty_when_missing(memory_root: Path) -> None:
    manager = await _manager(memory_root)

    assert await manager.core.load() == ""


# 函数说明：test_core_update_and_load_roundtrip
# 用途：回归验证回归测试与测试辅助中的 `core_update_and_load_roundtrip` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.core.update` →
# `manager.core.load`。
# 分支与异常：
#   验证条件：`'使用中文交流' in await manager.core.load()`。
@pytest.mark.asyncio
async def test_core_update_and_load_roundtrip(memory_root: Path) -> None:
    manager = await _manager(memory_root)

    await manager.core.update("用户长期偏好：使用中文交流。")

    assert "使用中文交流" in await manager.core.load()


# 函数说明：test_core_upsert_preserves_manual_content_and_other_entries
# 用途：回归验证回归测试与测试辅助中的
# `core_upsert_preserves_manual_content_and_other_entries` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.core.update` →
# `manager.upsert_core` → `manager.core.load` → `(memory_root / 'CORE.md').read_text`。
# 分支与异常：
#   验证条件：`first.key == 'communication.language'`。
#   验证条件：`first_created is True`。
#   验证条件：`second_created is True`。
#   验证条件：`created_again is False`。
# 副作用与资源：
#   文件或资源访问：`(memory_root / 'CORE.md').read_text`。
@pytest.mark.asyncio
async def test_core_upsert_preserves_manual_content_and_other_entries(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    await manager.core.update("# Core Memory\n\n人工维护的长期约束。")

    first, first_created = await manager.upsert_core(
        key="communication.language",
        value="始终使用中文交流。",
        reason="用户明确表达长期语言偏好",
        source_statement="以后都使用中文和我交流",
    )
    _, second_created = await manager.upsert_core(
        key="code.comment_language",
        value="代码注释使用中文。",
        reason="用户明确表达长期代码约束",
        source_statement="以后代码注释都使用中文",
    )
    updated, created_again = await manager.upsert_core(
        key="communication.language",
        value="默认使用简体中文交流。",
        reason="用户更新了语言偏好",
        source_statement="以后默认使用简体中文",
    )

    visible = await manager.core.load()
    raw = (memory_root / "CORE.md").read_text(encoding="utf-8")
    assert first.key == "communication.language"
    assert first_created is True
    assert second_created is True
    assert created_again is False
    assert updated.value == "默认使用简体中文交流。"
    assert "人工维护的长期约束" in visible
    assert "默认使用简体中文交流" in visible
    assert "代码注释使用中文" in visible
    assert "始终使用中文交流" not in visible
    assert "用户更新了语言偏好" in raw
    assert "以后默认使用简体中文" in raw
    assert "用户更新了语言偏好" not in visible
    assert "以后默认使用简体中文" not in visible


# 函数说明：test_core_upsert_failure_does_not_change_file
# 用途：回归验证回归测试与测试辅助中的 `core_upsert_failure_does_not_change_file` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` → `manager.initialize`
#  → `manager.upsert_core` → `(memory_root / 'CORE.md').read_text` → `pytest.raises`。
# 分支与异常：
#   验证条件：`(memory_root / 'CORE.md').read_text(encoding='utf-8') == before`。
#   预期异常：`pytest.raises(ValueError, match='core memory exceeds token limit')`。
# 副作用与资源：
#   文件或资源访问：`(memory_root / 'CORE.md').read_text`。
@pytest.mark.asyncio
async def test_core_upsert_failure_does_not_change_file(memory_root: Path) -> None:
    manager = MemoryManager(memory_root, max_core_tokens=20)
    await manager.initialize()
    await manager.upsert_core(
        key="communication.language",
        value="中文",
        reason="用户明确要求",
        source_statement="以后使用中文",
    )
    before = (memory_root / "CORE.md").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="core memory exceeds token limit"):
        await manager.upsert_core(
            key="identity.background",
            value="非常长的身份信息" * 100,
            reason="用户明确更新身份",
            source_statement="我的身份背景已经更新",
        )

    assert (memory_root / "CORE.md").read_text(encoding="utf-8") == before


# 函数说明：_core_document
# 用途：返回
# `f"---\nformat: {format_marker}\nupdated_at: '2026-08-12T07:00:58+00:00'\naudit:\n…`，
# 提供 回归测试与测试辅助 的派生值。
# 参数：
#   format_marker：`format_marker`输入或配置值，类型 `str`；默认 `'muharness-core-v1'`。
#   value：待校验、规范化或转换的值，类型 `str`；默认 `'始终使用中文交流。'`。
#   manual_body：`manual_body`输入或配置值，类型 `str`；默认 `'人工维护的长期约束。'`。
# 返回：类型 `str`；返回
# `f"---\nformat: {format_marker}\nupdated_at: '2026-08-12T07:00:58+00:00'\naudit:\n…`。
def _core_document(
    *,
    format_marker: str = "muharness-core-v1",
    value: str = "始终使用中文交流。",
    manual_body: str = "人工维护的长期约束。",
) -> str:

    return f"""---
format: {format_marker}
updated_at: '2026-08-12T07:00:58+00:00'
audit:
  verified_by: user
entries:
- key: communication.language
  value: {value}
  reason: 用户明确表达长期语言偏好
  source_statement: 以后都使用中文和我交流
  updated_at: '2026-08-12T07:00:58+00:00'
---
# Core Memory

{manual_body}

## Managed Core Entries

### communication.language

{value}
"""


# 函数说明：test_core_initialize_preserves_current_document
# 用途：回归验证回归测试与测试辅助中的 `core_initialize_preserves_current_document` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`memory_root.mkdir` → `_core_document`
#  → `path.write_text` → `CoreMemoryManager` → `manager.initialize` → `path.read_text`；
# 另有 2 个调用点。
# 分支与异常：
#   验证条件：`path.read_text(encoding='utf-8') == document`。
#   验证条件：`'始终使用中文交流' in await manager.load()`。
#   验证条件：`created is False`。
#   验证条件：`'format: muharness-core-v1' in path.read_text(encoding='utf-8')`。
# 副作用与资源：
#   文件或资源访问：`memory_root.mkdir`、`path.write_text`、`path.read_text`。
@pytest.mark.asyncio
async def test_core_initialize_preserves_current_document(memory_root: Path) -> None:
    memory_root.mkdir(parents=True)
    path = memory_root / "CORE.md"
    document = _core_document()
    path.write_text(document, encoding="utf-8")
    manager = CoreMemoryManager(memory_root)

    await manager.initialize()
    assert path.read_text(encoding="utf-8") == document
    assert "始终使用中文交流" in await manager.load()

    _, created = await manager.upsert(
        key="communication.language",
        value="默认使用中文交流。",
        reason="用户更新了长期语言偏好",
        source_statement="以后默认使用中文",
    )
    assert created is False
    assert "format: muharness-core-v1" in path.read_text(encoding="utf-8")


# 函数说明：test_core_load_hides_current_front_matter
# 用途：回归验证回归测试与测试辅助中的 `core_load_hides_current_front_matter` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`memory_root.mkdir` →
# `(memory_root / 'CORE.md').write_text` → `_core_document` → `CoreMemoryManager` →
# `manager.load`。
# 分支与异常：
#   验证条件：`'始终使用中文交流' in visible`。
#   验证条件：`'人工维护的长期约束' in visible`。
#   验证条件：`'muharness-core-v1' not in visible`。
#   验证条件：`'source_statement' not in visible`。
# 副作用与资源：
#   文件或资源访问：`memory_root.mkdir`、`(memory_root / 'CORE.md').write_text`。
@pytest.mark.asyncio
async def test_core_load_hides_current_front_matter(
    memory_root: Path,
) -> None:
    memory_root.mkdir(parents=True)
    (memory_root / "CORE.md").write_text(
        _core_document(),
        encoding="utf-8",
    )
    manager = CoreMemoryManager(memory_root)

    visible = await manager.load()

    assert "始终使用中文交流" in visible
    assert "人工维护的长期约束" in visible
    assert "muharness-core-v1" not in visible
    assert "source_statement" not in visible
    assert "verified_by" not in visible


# 函数说明：test_core_rejects_unsupported_format_without_rewriting
# 用途：回归验证回归测试与测试辅助中的
# `core_rejects_unsupported_format_without_rewriting` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`memory_root.mkdir` → `_core_document`
#  → `path.write_text` → `path.read_bytes` → `CoreMemoryManager` → `pytest.raises`；另有
#  1 个调用点。
# 分支与异常：
#   验证条件：`path.read_bytes() == before`。
#   预期异常：`pytest.raises(ValueError, match='unsupported CORE.md format')`。
# 副作用与资源：
#   文件或资源访问：`memory_root.mkdir`、`path.write_text`、`path.read_bytes`。
@pytest.mark.asyncio
async def test_core_rejects_unsupported_format_without_rewriting(
    memory_root: Path,
) -> None:
    memory_root.mkdir(parents=True)
    path = memory_root / "CORE.md"
    document = _core_document(format_marker="unsupported-core-v1")
    path.write_text(document, encoding="utf-8")
    before = path.read_bytes()
    manager = CoreMemoryManager(memory_root)

    operations = (
        manager.initialize,
        manager.load,
        lambda: manager.upsert(
            key="communication.language",
            value="默认使用中文交流。",
            reason="用户更新了长期语言偏好",
            source_statement="以后默认使用中文",
        ),
        lambda: manager.remove("communication.language"),
    )
    for operation in operations:
        with pytest.raises(ValueError, match="unsupported CORE.md format"):
            await operation()
        assert path.read_bytes() == before


# 函数说明：test_core_preserves_manual_yaml_without_format
# 用途：回归验证回归测试与测试辅助中的 `core_preserves_manual_yaml_without_format` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`memory_root.mkdir` →
# `path.write_text` → `CoreMemoryManager` → `manager.initialize` → `manager.load` →
# `path.read_text`。
# 分支与异常：
#   验证条件：`await manager.load() == document.strip()`。
#   验证条件：`path.read_text(encoding='utf-8') == document`。
# 副作用与资源：
#   文件或资源访问：`memory_root.mkdir`、`path.write_text`、`path.read_text`。
@pytest.mark.asyncio
async def test_core_preserves_manual_yaml_without_format(memory_root: Path) -> None:
    memory_root.mkdir(parents=True)
    path = memory_root / "CORE.md"
    document = "---\npreferences:\n  language: 中文\n---\n人工维护的长期约束。\n"
    path.write_text(document, encoding="utf-8")
    manager = CoreMemoryManager(memory_root)

    await manager.initialize()
    assert await manager.load() == document.strip()
    assert path.read_text(encoding="utf-8") == document


# 函数说明：test_core_mutations_preserve_current_document_body
# 用途：回归验证回归测试与测试辅助中的 `core_mutations_preserve_current_document_body`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`memory_root.mkdir` →
# `path.write_text` → `_core_document` → `_manager` → `manager.upsert_core` →
# `manager.remove_core`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`created is False`。
#   验证条件：`updated.value == '默认使用简体中文交流。'`。
#   验证条件：`added_created is True`。
#   验证条件：`added.key == 'safety.confirmation'`。
# 副作用与资源：
#   文件或资源访问：`memory_root.mkdir`、`path.write_text`、`path.read_text`。
@pytest.mark.asyncio
async def test_core_mutations_preserve_current_document_body(
    memory_root: Path,
) -> None:
    memory_root.mkdir(parents=True)
    path = memory_root / "CORE.md"
    path.write_text(_core_document(), encoding="utf-8")
    manager = await _manager(memory_root)

    updated, created = await manager.upsert_core(
        key="communication.language",
        value="默认使用简体中文交流。",
        reason="用户更新了全局语言偏好",
        source_statement="以后默认使用简体中文",
    )
    added, added_created = await manager.upsert_core(
        key="safety.confirmation",
        value="高风险操作执行前必须确认。",
        reason="用户明确表达全局安全约束",
        source_statement="高风险操作执行前必须问我",
    )
    removed = await manager.remove_core("communication.language")

    raw = path.read_text(encoding="utf-8")
    visible = await manager.core.load()
    assert created is False
    assert updated.value == "默认使用简体中文交流。"
    assert added_created is True
    assert added.key == "safety.confirmation"
    assert removed.key == "communication.language"
    assert raw.count("format: muharness-core-v1") == 1
    assert raw.count("---") == 2
    assert "人工维护的长期约束" in visible
    assert "高风险操作执行前必须确认" in visible
    assert "默认使用简体中文交流" not in visible


# 函数说明：test_core_load_token_limit_failure_does_not_change_file
# 用途：回归验证回归测试与测试辅助中的
# `core_load_token_limit_failure_does_not_change_file` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`memory_root.mkdir` → `_core_document`
#  → `path.write_text` → `CoreMemoryManager` → `pytest.raises` → `manager.load`；另有 1
# 个调用点。
# 分支与异常：
#   验证条件：`path.read_text(encoding='utf-8') == document`。
#   预期异常：`pytest.raises(ValueError, match='core memory exceeds token limit')`。
# 副作用与资源：
#   文件或资源访问：`memory_root.mkdir`、`path.write_text`、`path.read_text`。
@pytest.mark.asyncio
async def test_core_load_token_limit_failure_does_not_change_file(
    memory_root: Path,
) -> None:
    memory_root.mkdir(parents=True)
    path = memory_root / "CORE.md"
    document = _core_document(manual_body="很长的人工正文" * 200)
    path.write_text(document, encoding="utf-8")
    manager = CoreMemoryManager(memory_root, max_tokens=10)

    with pytest.raises(ValueError, match="core memory exceeds token limit"):
        await manager.load()

    assert path.read_text(encoding="utf-8") == document


# 函数说明：test_core_initialize_and_load_reject_symbolic_links
# 用途：回归验证回归测试与测试辅助中的 `core_initialize_and_load_reject_symbolic_links`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
#   monkeypatch：pytest 提供的临时替换依赖夹具，类型 `pytest.MonkeyPatch`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`memory_root.mkdir` → `_core_document`
#  → `path.write_text` → `monkeypatch.setattr` → `CoreMemoryManager` → `pytest.raises`；
# 另有 3 个调用点。
# 分支与异常：
#   验证条件：`path.read_text(encoding='utf-8') == document`。
#   预期异常：`pytest.raises(ValueError, match='cannot be a symbolic link')`。
# 副作用与资源：
#   文件或资源访问：`memory_root.mkdir`、`path.write_text`、`path.read_text`。
@pytest.mark.asyncio
async def test_core_initialize_and_load_reject_symbolic_links(
    memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_root.mkdir(parents=True)
    path = memory_root / "CORE.md"
    document = _core_document()
    path.write_text(document, encoding="utf-8")
    monkeypatch.setattr(Path, "is_symlink", lambda candidate: candidate == path)
    manager = CoreMemoryManager(memory_root)

    with pytest.raises(ValueError, match="cannot be a symbolic link"):
        await manager.initialize()
    with pytest.raises(ValueError, match="cannot be a symbolic link"):
        await manager.load()
    assert path.read_text(encoding="utf-8") == document


# 函数说明：test_core_does_not_count_towards_active_limit
# 用途：回归验证回归测试与测试辅助中的 `core_does_not_count_towards_active_limit` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.core.update` →
# `manager.create` → `manager.store.count_active` → `manager.maintenance_required`。
# 分支与异常：
#   验证条件：`await manager.store.count_active() == 25`。
#   验证条件：`await manager.maintenance_required() is False`。
@pytest.mark.asyncio
async def test_core_does_not_count_towards_active_limit(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    await manager.core.update("# Core Memory\n\n用户身份：开发者")

    for index in range(25):
        await manager.create(
            title=f"记忆 {index}",
            summary=f"cue {index}",
            content=f"内容 {index}",
        )

    assert await manager.store.count_active() == 25
    assert await manager.maintenance_required() is False


# 函数说明：test_core_not_archived_with_memories
# 用途：回归验证回归测试与测试辅助中的 `core_not_archived_with_memories` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.core.update` →
# `manager.create` → `manager.archive` → `(memory_root / 'CORE.md').is_file` →
# `manager.core.load`。
# 分支与异常：
#   验证条件：`(memory_root / 'CORE.md').is_file()`。
#   验证条件：`await manager.core.load() != ''`。
@pytest.mark.asyncio
async def test_core_not_archived_with_memories(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    await manager.core.update("用户身份：开发者")
    record = await manager.create(
        title="临时记忆",
        summary="cue",
        content="内容",
    )

    await manager.archive(record.id, reason="不再需要")

    assert (memory_root / "CORE.md").is_file()
    assert await manager.core.load() != ""




# 函数说明：test_create_assigns_incrementing_ids
# 用途：回归验证回归测试与测试辅助中的 `create_assigns_incrementing_ids` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `(memory_root / 'active' / 'M001.md').is_file` →
# `(memory_root / 'active' / 'M002.md').is_file`。
# 分支与异常：
#   验证条件：`first.id == 'M001'`。
#   验证条件：`second.id == 'M002'`。
#   验证条件：`(memory_root / 'active' / 'M001.md').is_file()`。
#   验证条件：`(memory_root / 'active' / 'M002.md').is_file()`。
@pytest.mark.asyncio
async def test_create_assigns_incrementing_ids(memory_root: Path) -> None:
    manager = await _manager(memory_root)

    first = await manager.create(title="A", summary="a", content="内容A")
    second = await manager.create(title="B", summary="b", content="内容B")

    assert first.id == "M001"
    assert second.id == "M002"
    assert (memory_root / "active" / "M001.md").is_file()
    assert (memory_root / "active" / "M002.md").is_file()


# 函数说明：test_concurrent_create_assigns_unique_ids
# 用途：回归验证回归测试与测试辅助中的 `concurrent_create_assigns_unique_ids` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `asyncio.gather` →
# `manager.create` → `manager.store.count_active`。
# 分支与异常：
#   验证条件：
# `{record.id for record in records} == {f'M{index:03d}' for index in range(1, 11)}`。
#   验证条件：`await manager.store.count_active() == 10`。
@pytest.mark.asyncio
async def test_concurrent_create_assigns_unique_ids(memory_root: Path) -> None:

    import asyncio

    manager = await _manager(memory_root)
    records = await asyncio.gather(
        *(
            manager.create(
                title=f"记忆 {index}",
                summary=f"cue {index}",
                content=f"内容 {index}",
            )
            for index in range(10)
        )
    )

    assert {record.id for record in records} == {
        f"M{index:03d}" for index in range(1, 11)
    }
    assert await manager.store.count_active() == 10


# 函数说明：test_read_returns_record
# 用途：回归验证回归测试与测试辅助中的 `read_returns_record` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.read`。
# 分支与异常：
#   验证条件：`loaded is not None`。
#   验证条件：`loaded.title == 'A'`。
#   验证条件：`loaded.content == '内容A'`。
@pytest.mark.asyncio
async def test_read_returns_record(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="A", summary="a", content="内容A")

    loaded = await manager.read(record.id)

    assert loaded is not None
    assert loaded.title == "A"
    assert loaded.content == "内容A"


# 函数说明：test_read_missing_returns_none
# 用途：回归验证回归测试与测试辅助中的 `read_missing_returns_none` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.read`。
# 分支与异常：
#   验证条件：`await manager.read('M999') is None`。
@pytest.mark.asyncio
async def test_read_missing_returns_none(memory_root: Path) -> None:
    manager = await _manager(memory_root)

    assert await manager.read("M999") is None


# 函数说明：test_memory_id_rejects_path_traversal
# 用途：回归验证回归测试与测试辅助中的 `memory_id_rejects_path_traversal` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `pytest.raises` →
# `manager.read`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='memory id')`。
@pytest.mark.asyncio
async def test_memory_id_rejects_path_traversal(memory_root: Path) -> None:
    manager = await _manager(memory_root)

    with pytest.raises(ValueError, match="memory id"):
        await manager.read("../../CORE")


# 函数说明：test_create_rejects_content_larger_than_tool_read_budget
# 用途：回归验证回归测试与测试辅助中的
# `create_rejects_content_larger_than_tool_read_budget` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `pytest.raises` →
# `manager.create` → `manager.list`。
# 分支与异常：
#   验证条件：`await manager.list() == ()`。
#   预期异常：`pytest.raises(ValueError, match='memory content exceeds')`。
@pytest.mark.asyncio
async def test_create_rejects_content_larger_than_tool_read_budget(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)

    with pytest.raises(ValueError, match="memory content exceeds"):
        await manager.create(title="过长", summary="过长正文", content="x" * 12_001)

    assert await manager.list() == ()


# 函数说明：test_update_changes_content
# 用途：回归验证回归测试与测试辅助中的 `update_changes_content` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.update` → `manager.read`。
# 分支与异常：
#   验证条件：`updated.content == '新内容'`。
#   验证条件：`updated.revision == record.revision + 1`。
#   验证条件：`(await manager.read(record.id)).content == '新内容'`。
@pytest.mark.asyncio
async def test_update_changes_content(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="A", summary="a", content="旧内容")

    updated = await manager.update(record.id, content="新内容", reason="修正事实")

    assert updated.content == "新内容"
    assert updated.revision == record.revision + 1
    assert (await manager.read(record.id)).content == "新内容"


# 函数说明：test_update_changes_recall_cue_and_rebuilds_index
# 用途：回归验证回归测试与测试辅助中的 `update_changes_recall_cue_and_rebuilds_index` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.update_if_revision` → `manager.index.load`。
# 分支与异常：
#   验证条件：`updated.title == '新标题'`。
#   验证条件：`updated.summary == '新 cue'`。
#   验证条件：`'[M001] 新标题' in index`。
#   验证条件：`'Cue: 新 cue' in index`。
@pytest.mark.asyncio
async def test_update_changes_recall_cue_and_rebuilds_index(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="旧标题", summary="旧 cue", content="旧正文")

    updated = await manager.update_if_revision(
        record.id,
        expected_revision=record.revision,
        title="新标题",
        summary="新 cue",
        content="新正文",
        reason="架构已经变化",
    )

    index = await manager.index.load() or ""
    assert updated.title == "新标题"
    assert updated.summary == "新 cue"
    assert "[M001] 新标题" in index
    assert "Cue: 新 cue" in index
    assert "旧 cue" not in index


# 函数说明：test_update_rejects_stale_revision_without_changing_file_or_index
# 用途：回归验证回归测试与测试辅助中的
# `update_rejects_stale_revision_without_changing_file_or_index` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.update` → `memory_path.read_text` → `manager.index.load` → `pytest.raises`；
# 另有 1 个调用点。
# 分支与异常：
#   验证条件：`memory_path.read_text(encoding='utf-8') == before_file`。
#   验证条件：`await manager.index.load() == before_index`。
#   预期异常：`pytest.raises(ValueError, match='revision conflict')`。
# 副作用与资源：
#   文件或资源访问：`memory_path.read_text`。
@pytest.mark.asyncio
async def test_update_rejects_stale_revision_without_changing_file_or_index(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="标题", summary="cue", content="初始正文")
    await manager.update(record.id, content="并发写入", reason="另一个 Run")
    memory_path = memory_root / "active" / f"{record.id}.md"
    before_file = memory_path.read_text(encoding="utf-8")
    before_index = await manager.index.load()

    with pytest.raises(ValueError, match="revision conflict"):
        await manager.update_if_revision(
            record.id,
            expected_revision=record.revision,
            title="过期标题",
            summary="过期 cue",
            content="过期正文",
            reason="过期 Run",
        )

    assert memory_path.read_text(encoding="utf-8") == before_file
    assert await manager.index.load() == before_index


# 函数说明：test_update_missing_raises
# 用途：回归验证回归测试与测试辅助中的 `update_missing_raises` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `pytest.raises` →
# `manager.update`。
# 分支与异常：
#   预期异常：`pytest.raises(KeyError)`。
@pytest.mark.asyncio
async def test_update_missing_raises(memory_root: Path) -> None:
    manager = await _manager(memory_root)

    with pytest.raises(KeyError):
        await manager.update("M999", content="内容", reason="修正")


# 函数说明：test_archive_moves_to_archive_and_sets_status
# 用途：回归验证回归测试与测试辅助中的 `archive_moves_to_archive_and_sets_status` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.archive` → `(memory_root / 'active' / f'{record.id}.md').exists` →
# `(memory_root / 'archive' / f'{record.id}.md').is_file` → `manager.list`。
# 分支与异常：
#   验证条件：`archived.status is MemoryStatus.ARCHIVED`。
#   验证条件：`not (memory_root / 'active' / f'{record.id}.md').exists()`。
#   验证条件：`(memory_root / 'archive' / f'{record.id}.md').is_file()`。
#   验证条件：`all((item.id != record.id for item in await manager.list()))`。
@pytest.mark.asyncio
async def test_archive_moves_to_archive_and_sets_status(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="A", summary="a", content="内容A")

    archived = await manager.archive(record.id, reason="已经过时")

    assert archived.status is MemoryStatus.ARCHIVED
    assert not (memory_root / "active" / f"{record.id}.md").exists()
    assert (memory_root / "archive" / f"{record.id}.md").is_file()
    assert all(item.id != record.id for item in await manager.list())


# 函数说明：test_archived_memory_cannot_be_updated_or_return_to_active
# 用途：回归验证回归测试与测试辅助中的
# `archived_memory_cannot_be_updated_or_return_to_active` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.archive` → `archived_path.read_text` → `pytest.raises` → `manager.update`；另
# 有 1 个调用点。
# 分支与异常：
#   验证条件：`archived_path.read_text(encoding='utf-8') == before`。
#   验证条件：`not (memory_root / 'active' / f'{record.id}.md').exists()`。
#   预期异常：`pytest.raises(ValueError, match='only active memory')`。
# 副作用与资源：
#   文件或资源访问：`archived_path.read_text`。
@pytest.mark.asyncio
async def test_archived_memory_cannot_be_updated_or_return_to_active(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="A", summary="a", content="原内容")
    await manager.archive(record.id, reason="已经过时")
    archived_path = memory_root / "archive" / f"{record.id}.md"
    before = archived_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="only active memory"):
        await manager.update(record.id, content="错误恢复", reason="不应成功")

    assert archived_path.read_text(encoding="utf-8") == before
    assert not (memory_root / "active" / f"{record.id}.md").exists()


# 函数说明：test_initialize_repairs_interrupted_archive
# 用途：回归验证回归测试与测试辅助中的 `initialize_repairs_interrupted_archive` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `MemoryRecord` → `active_path.write_text` → `interrupted.render_markdown` →
# `active_path.exists`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`not active_path.exists()`。
#   验证条件：`(memory_root / 'archive' / f'{record.id}.md').is_file()`。
#   验证条件：`await restarted.list() == ()`。
#   验证条件：`f'[{record.id}]' not in (await restarted.index.load() or '')`。
# 副作用与资源：
#   文件或资源访问：`active_path.write_text`。
@pytest.mark.asyncio
async def test_initialize_repairs_interrupted_archive(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="A", summary="a", content="原内容")
    interrupted = MemoryRecord(
        **{
            **record.model_dump(),
            "status": MemoryStatus.ARCHIVED,
            "archive_reason": "模拟移动前中断",
        }
    )
    active_path = memory_root / "active" / f"{record.id}.md"
    active_path.write_text(interrupted.render_markdown(), encoding="utf-8")

    restarted = await _manager(memory_root)

    assert not active_path.exists()
    assert (memory_root / "archive" / f"{record.id}.md").is_file()
    assert await restarted.list() == ()
    assert f"[{record.id}]" not in (await restarted.index.load() or "")


# 函数说明：test_update_and_archive_reasons_are_persisted
# 用途：回归验证回归测试与测试辅助中的 `update_and_archive_reasons_are_persisted` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.update` → `manager.archive` → `manager.store.load`。
# 分支与异常：
#   验证条件：`loaded is not None`。
#   验证条件：`archived.last_update_reason == '用户修正了事实'`。
#   验证条件：`loaded.last_update_reason == '用户修正了事实'`。
#   验证条件：`loaded.archive_reason == '该背景已经失效'`。
@pytest.mark.asyncio
async def test_update_and_archive_reasons_are_persisted(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="A", summary="a", content="旧内容")

    await manager.update(record.id, content="新内容", reason="用户修正了事实")
    archived = await manager.archive(record.id, reason="该背景已经失效")
    loaded = await manager.store.load(record.id)

    assert loaded is not None
    assert archived.last_update_reason == "用户修正了事实"
    assert loaded.last_update_reason == "用户修正了事实"
    assert loaded.archive_reason == "该背景已经失效"


# 函数说明：test_list_returns_only_active
# 用途：回归验证回归测试与测试辅助中的 `list_returns_only_active` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.archive` → `manager.list`。
# 分支与异常：
#   验证条件：`ids == {'M002'}`。
@pytest.mark.asyncio
async def test_list_returns_only_active(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    first = await manager.create(title="A", summary="a", content="内容A")
    await manager.create(title="B", summary="b", content="内容B")
    await manager.archive(first.id, reason="清理测试数据")

    ids = {record.id for record in await manager.list()}

    assert ids == {"M002"}




# 函数说明：test_read_increments_access_count
# 用途：回归验证回归测试与测试辅助中的 `read_increments_access_count` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.read`。
# 分支与异常：
#   验证条件：`loaded is not None`。
#   验证条件：`loaded.access_count == 3`。
@pytest.mark.asyncio
async def test_read_increments_access_count(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="A", summary="a", content="内容A")

    await manager.read(record.id)
    await manager.read(record.id)

    loaded = await manager.read(record.id)
    assert loaded is not None
    assert loaded.access_count == 3


# 函数说明：test_read_updates_last_accessed_at
# 用途：回归验证回归测试与测试辅助中的 `read_updates_last_accessed_at` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.read`。
# 分支与异常：
#   验证条件：`loaded is not None`。
#   验证条件：`loaded.last_accessed_at >= before`。
@pytest.mark.asyncio
async def test_read_updates_last_accessed_at(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="A", summary="a", content="内容A")
    before = record.last_accessed_at

    loaded = await manager.read(record.id)

    assert loaded is not None
    assert loaded.last_accessed_at >= before


# 函数说明：test_memory_policy_requires_read_before_using_index_cue
# 用途：回归验证回归测试与测试辅助中的
# `memory_policy_requires_read_before_using_index_cue` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'只是发现线索，不是权威正文' in MEMORY_POLICY_PROMPT`。
#   验证条件：`'依赖它回答、决策或执行前也必须调用 memory_read' in MEMORY_POLICY_PROMPT`
# 。
#   验证条件：`'只有成功的 memory_read 才算普通记忆已读取' in MEMORY_POLICY_PROMPT`。
def test_memory_policy_requires_read_before_using_index_cue() -> None:

    assert "只是发现线索，不是权威正文" in MEMORY_POLICY_PROMPT
    assert "依赖它回答、决策或执行前也必须调用 memory_read" in MEMORY_POLICY_PROMPT
    assert "只有成功的 memory_read 才算普通记忆已读取" in MEMORY_POLICY_PROMPT


# 函数说明：test_update_refreshes_updated_at
# 用途：回归验证回归测试与测试辅助中的 `update_refreshes_updated_at` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.update`。
# 分支与异常：
#   验证条件：`updated.updated_at >= before`。
@pytest.mark.asyncio
async def test_update_refreshes_updated_at(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    record = await manager.create(title="A", summary="a", content="旧内容")
    before = record.updated_at

    updated = await manager.update(record.id, content="新内容", reason="修正事实")

    assert updated.updated_at >= before


# 函数说明：test_legacy_memory_without_revision_defaults_to_first_revision
# 用途：回归验证回归测试与测试辅助中的
# `legacy_memory_without_revision_defaults_to_first_revision` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` → `MemoryRecord` →
# `record.render_markdown().replace` → `record.render_markdown` →
# `parse_memory_markdown`。
# 分支与异常：
#   验证条件：`loaded.revision == 1`。
def test_legacy_memory_without_revision_defaults_to_first_revision() -> None:
    now = datetime.now(UTC)
    record = MemoryRecord(
        id="M001",
        title="旧记忆",
        summary="旧格式",
        content="正文",
        created_at=now,
        updated_at=now,
        last_accessed_at=now,
    )
    legacy = record.render_markdown().replace("revision: 1\n", "")

    loaded = parse_memory_markdown(legacy)

    assert loaded.revision == 1




# 函数说明：test_create_rebuilds_index
# 用途：回归验证回归测试与测试辅助中的 `create_rebuilds_index` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.index.load`。
# 分支与异常：
#   验证条件：`index_text is not None`。
#   验证条件：`'[M001] 第一' in index_text`。
#   验证条件：`'cue-1' in index_text`。
@pytest.mark.asyncio
async def test_create_rebuilds_index(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    await manager.create(title="第一", summary="cue-1", content="内容1")

    index_text = await manager.index.load()
    assert index_text is not None
    assert "[M001] 第一" in index_text
    assert "cue-1" in index_text


# 函数说明：test_initialize_repairs_stale_index_from_active_files
# 用途：回归验证回归测试与测试辅助中的
# `initialize_repairs_stale_index_from_active_files` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `(memory_root / 'INDEX.md').write_text` → `restarted.index.load`。
# 分支与异常：
#   验证条件：`index_text is not None`。
#   验证条件：`'[M001] 第一' in index_text`。
#   验证条件：`'stale index' not in index_text`。
# 副作用与资源：
#   文件或资源访问：`(memory_root / 'INDEX.md').write_text`。
@pytest.mark.asyncio
async def test_initialize_repairs_stale_index_from_active_files(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    await manager.create(title="第一", summary="cue-1", content="内容1")
    (memory_root / "INDEX.md").write_text("stale index", encoding="utf-8")

    restarted = await _manager(memory_root)
    index_text = await restarted.index.load()

    assert index_text is not None
    assert "[M001] 第一" in index_text
    assert "stale index" not in index_text


# 函数说明：test_archive_removes_from_index
# 用途：回归验证回归测试与测试辅助中的 `archive_removes_from_index` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.archive` → `manager.index.load`。
# 分支与异常：
#   验证条件：`index_text is not None`。
#   验证条件：`'[M001]' not in index_text`。
#   验证条件：`'[M002] 第二' in index_text`。
@pytest.mark.asyncio
async def test_archive_removes_from_index(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    first = await manager.create(title="第一", summary="cue-1", content="内容1")
    await manager.create(title="第二", summary="cue-2", content="内容2")

    await manager.archive(first.id, reason="已经过时")

    index_text = await manager.index.load()
    assert index_text is not None
    assert "[M001]" not in index_text
    assert "[M002] 第二" in index_text


# 函数说明：test_index_contains_cue_not_full_content
# 用途：回归验证回归测试与测试辅助中的 `index_contains_cue_not_full_content` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.index.load`。
# 分支与异常：
#   验证条件：`index_text is not None`。
#   验证条件：`'短提示' in index_text`。
#   验证条件：`'完整记忆正文内容A' not in index_text`。
@pytest.mark.asyncio
async def test_index_contains_cue_not_full_content(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    await manager.create(title="第一", summary="短提示", content="完整记忆正文内容A")

    index_text = await manager.index.load()
    assert index_text is not None
    assert "短提示" in index_text
    assert "完整记忆正文内容A" not in index_text




# 函数说明：test_active_below_limit_does_not_require_maintenance
# 用途：回归验证回归测试与测试辅助中的 `active_below_limit_does_not_require_maintenance`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `manager.maintenance_required`。
# 分支与异常：
#   验证条件：`await manager.maintenance_required() is False`。
@pytest.mark.asyncio
async def test_active_below_limit_does_not_require_maintenance(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    for index in range(25):
        await manager.create(title=f"m{index}", summary=f"s{index}", content="c")

    assert await manager.maintenance_required() is False


# 函数说明：test_26th_memory_is_rejected_by_hard_capacity
# 用途：回归验证回归测试与测试辅助中的 `26th_memory_is_rejected_by_hard_capacity` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.create` →
# `pytest.raises` → `manager.store.count_active` → `manager.maintenance_required`。
# 分支与异常：
#   验证条件：`await manager.store.count_active() == 25`。
#   验证条件：`await manager.maintenance_required() is False`。
#   预期异常：`pytest.raises(ValueError, match='capacity is full')`。
@pytest.mark.asyncio
async def test_26th_memory_is_rejected_by_hard_capacity(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    for index in range(25):
        await manager.create(title=f"m{index}", summary=f"s{index}", content="c")

    with pytest.raises(ValueError, match="capacity is full"):
        await manager.create(title="m25", summary="s25", content="c")

    assert await manager.store.count_active() == 25
    assert await manager.maintenance_required() is False


# 函数说明：test_two_managers_share_capacity_lock
# 用途：回归验证回归测试与测试辅助中的 `two_managers_share_capacity_lock` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` → `asyncio.gather` →
# `first_manager.initialize` → `second_manager.initialize` →
# `first_manager.create_if_capacity` → `second_manager.create_if_capacity`；另有 1 个调
# 用点。
# 分支与异常：
#   验证条件：`await first_manager.active_count() == 1`。
#   验证条件：`sum((record is not None for record in (first, second))) == 1`。
@pytest.mark.asyncio
async def test_two_managers_share_capacity_lock(memory_root: Path) -> None:
    first_manager = MemoryManager(memory_root, max_active=1)
    second_manager = MemoryManager(memory_root, max_active=1)
    await asyncio.gather(first_manager.initialize(), second_manager.initialize())

    first, second = await asyncio.gather(
        first_manager.create_if_capacity(title="A", summary="A", content="A"),
        second_manager.create_if_capacity(title="B", summary="B", content="B"),
    )

    assert await first_manager.active_count() == 1
    assert sum(record is not None for record in (first, second)) == 1


# 函数说明：test_internal_create_tool_cannot_bypass_capacity
# 用途：回归验证回归测试与测试辅助中的 `internal_create_tool_cannot_bypass_capacity` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `ToolRegistry` →
# `register_memory_tools` → `register_memory_write_tools` → `manager.create` →
# `pytest.raises`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`await manager.store.count_active() == 25`。
#   预期异常：`pytest.raises(ValueError, match='capacity is full')`。
@pytest.mark.asyncio
async def test_internal_create_tool_cannot_bypass_capacity(
    memory_root: Path,
) -> None:

    manager = await _manager(memory_root)
    registry = ToolRegistry()
    register_memory_tools(registry, manager)
    register_memory_write_tools(registry, manager)
    for index in range(25):
        await manager.create(title=f"m{index}", summary=f"s{index}", content="c")

    with pytest.raises(ValueError, match="capacity is full"):
        await registry.get("memory_create").execute(
            {"title": "第 26 条", "summary": "触发维护", "content": "新内容"}
        )

    assert await manager.store.count_active() == 25


# 函数说明：test_maintenance_selects_least_retained_candidates
# 用途：回归验证回归测试与测试辅助中的 `maintenance_selects_least_retained_candidates`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` → `_manager` →
# `MemoryRecord` → `timedelta` → `manager.maintenance.select_candidates`。
# 分支与异常：
#   验证条件：`[item.id for item in candidates] == ['M001', 'M002', 'M003']`。
@pytest.mark.asyncio
async def test_maintenance_selects_least_retained_candidates(
    memory_root: Path,
) -> None:
    now = datetime.now(UTC)
    manager = await _manager(memory_root)
    records = [
        MemoryRecord(
            id=f"M{index:03d}",
            title=f"t{index}",
            summary=f"s{index}",
            content="c",
            created_at=now,
            updated_at=now - timedelta(hours=(6 - index) * 10),
            last_accessed_at=now - timedelta(hours=(6 - index) * 20),
            access_count=index,
        )
        for index in range(1, 6)
    ]

    candidates = manager.maintenance.select_candidates(records, limit=3)

    assert [item.id for item in candidates] == ["M001", "M002", "M003"]




# 函数说明：test_context_messages_include_core_index_policy
# 用途：回归验证回归测试与测试辅助中的 `context_messages_include_core_index_policy` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.core.update` →
# `manager.create` → `manager.context_messages` → `next` →
# `(core_message.content or '').count`。
# 分支与异常：
#   验证条件：`CORE_MEMORY_MESSAGE_NAME in names`。
#   验证条件：`MEMORY_INDEX_MESSAGE_NAME in names`。
#   验证条件：`MEMORY_POLICY_MESSAGE_NAME in names`。
#   验证条件：`not any(('完整正文' in (message.content or '') for message in messages))`
# 。
@pytest.mark.asyncio
async def test_context_messages_include_core_index_policy(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    await manager.core.update("用户身份：开发者")
    await manager.create(title="第一", summary="cue-1", content="完整正文")

    messages = await manager.context_messages()
    names = {message.name for message in messages}

    assert CORE_MEMORY_MESSAGE_NAME in names
    assert MEMORY_INDEX_MESSAGE_NAME in names
    assert MEMORY_POLICY_MESSAGE_NAME in names
    assert not any(
        "完整正文" in (message.content or "") for message in messages
    )
    core_message = next(
        message for message in messages if message.name == CORE_MEMORY_MESSAGE_NAME
    )
    assert (core_message.content or "").count("# Core Memory") == 1


# 函数说明：test_policy_message_guides_model_directed_recall
# 用途：回归验证回归测试与测试辅助中的 `policy_message_guides_model_directed_recall` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'memory_read' in MEMORY_POLICY_PROMPT`。
#   验证条件：`'在运行结束后整理' in MEMORY_POLICY_PROMPT`。
#   验证条件：`'core_memory_update' in MEMORY_POLICY_PROMPT`。
#   验证条件：`'完全无关的项目或仓库' in MEMORY_POLICY_PROMPT`。
@pytest.mark.asyncio
async def test_policy_message_guides_model_directed_recall() -> None:
    assert "memory_read" in MEMORY_POLICY_PROMPT
    assert "在运行结束后整理" in MEMORY_POLICY_PROMPT
    assert "core_memory_update" in MEMORY_POLICY_PROMPT
    assert "完全无关的项目或仓库" in MEMORY_POLICY_PROMPT
    assert "项目或仓库专属" in MEMORY_POLICY_PROMPT
    assert "也属于普通记忆" in MEMORY_POLICY_PROMPT
    assert "全局安全/隐私约束" in MEMORY_POLICY_PROMPT
    assert "先调用 tool_search" in MEMORY_POLICY_PROMPT
    assert '"core memory update"' in MEMORY_POLICY_PROMPT
    assert "只有 core_memory_update 成功回执才能确认已保存" in (
        MEMORY_POLICY_PROMPT
    )
    assert "搜索或修改失败时明确说未保存" in MEMORY_POLICY_PROMPT
    assert "Task" in MEMORY_POLICY_PROMPT
    assert "Skills" in MEMORY_POLICY_PROMPT


# 函数说明：test_write_policy_rejects_transient_and_procedural_content
# 用途：回归验证回归测试与测试辅助中的
# `write_policy_rejects_transient_and_procedural_content` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'不是当前任务的临时进度或临时约束' in MEMORY_WRITE_POLICY`。
#   验证条件：`'不是应由 Skills 保存的操作流程' in MEMORY_WRITE_POLICY`。
#   验证条件：`'有疑问就不创建' in MEMORY_WRITE_POLICY`。
#   验证条件：`'仅当以下条件全部满足时创建记忆' in MEMORY_WRITE_POLICY`。
def test_write_policy_rejects_transient_and_procedural_content() -> None:
    from app.domain.memory import MEMORY_WRITE_POLICY

    assert "不是当前任务的临时进度或临时约束" in MEMORY_WRITE_POLICY
    assert "不是应由 Skills 保存的操作流程" in MEMORY_WRITE_POLICY
    assert "有疑问就不创建" in MEMORY_WRITE_POLICY
    assert "仅当以下条件全部满足时创建记忆" in MEMORY_WRITE_POLICY
    assert "用户的耐久要求" in MEMORY_WRITE_POLICY




# 函数说明：test_register_memory_tools_exposes_only_main_agent_tools
# 用途：回归验证回归测试与测试辅助中的
# `register_memory_tools_exposes_only_main_agent_tools` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `ToolRegistry` →
# `register_memory_tools` → `registry.names`。
# 分支与异常：
#   验证条件：`set(registry.names()) == {'memory_read', 'memory_search', 'memory_list',
# 'core_memory_update', 'core_memory_remove'}`。
@pytest.mark.asyncio
async def test_register_memory_tools_exposes_only_main_agent_tools(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    registry = ToolRegistry()

    register_memory_tools(registry, manager)

    assert set(registry.names()) == {
        "memory_read",
        "memory_search",
        "memory_list",
        "core_memory_update",
        "core_memory_remove",
    }


# 函数说明：test_memory_create_and_read_tools_roundtrip
# 用途：回归验证回归测试与测试辅助中的 `memory_create_and_read_tools_roundtrip` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `ToolRegistry` →
# `register_memory_tools` → `register_memory_write_tools` →
# `registry.get('memory_create').execute` → `registry.get('memory_read').execute`；另有
# 1 个调用点。
# 分支与异常：
#   验证条件：`created['id'] == 'M001'`。
#   验证条件：`read['found'] is True`。
#   验证条件：`'使用中文交流' in read['content']`。
#   验证条件：`loaded is not None`。
@pytest.mark.asyncio
async def test_memory_create_and_read_tools_roundtrip(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    registry = ToolRegistry()
    register_memory_tools(registry, manager)
    register_memory_write_tools(registry, manager)

    created = await registry.get("memory_create").execute(
        {
            "title": "用户偏好",
            "summary": "偏好中文",
            "content": "用户长期偏好使用中文交流。",
        }
    )
    assert created["id"] == "M001"

    read = await registry.get("memory_read").execute(
        {"memory_id": "M001"}
    )
    assert read["found"] is True
    assert "使用中文交流" in read["content"]
    loaded = await manager.read("M001")
    assert loaded is not None
    assert loaded.access_count == 2


# 函数说明：test_memory_list_returns_cues_without_content
# 用途：回归验证回归测试与测试辅助中的 `memory_list_returns_cues_without_content` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `ToolRegistry` →
# `register_memory_tools` → `manager.create` → `registry.get('memory_list').execute`。
# 分支与异常：
#   验证条件：`result['memories'][0]['title'] == '第一'`。
#   验证条件：`'机密正文' not in str(result)`。
@pytest.mark.asyncio
async def test_memory_list_returns_cues_without_content(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    registry = ToolRegistry()
    register_memory_tools(registry, manager)
    await manager.create(title="第一", summary="cue-1", content="机密正文")

    result = await registry.get("memory_list").execute({})

    assert result["memories"][0]["title"] == "第一"
    assert "机密正文" not in str(result)


# 函数说明：test_memory_update_and_archive_tools
# 用途：回归验证回归测试与测试辅助中的 `memory_update_and_archive_tools` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `ToolRegistry` →
# `register_memory_tools` → `register_memory_write_tools` → `manager.create` →
# `registry.get('memory_update').execute`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`updated['updated'] is True`。
#   验证条件：`archived['status'] == 'archived'`。
#   验证条件：`await manager.read(record.id) is None`。
@pytest.mark.asyncio
async def test_memory_update_and_archive_tools(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    registry = ToolRegistry()
    register_memory_tools(registry, manager)
    register_memory_write_tools(registry, manager)
    record = await manager.create(title="第一", summary="cue-1", content="旧正文")

    updated = await registry.get("memory_update").execute(
        {
            "memory_id": record.id,
            "title": "更新后的标题",
            "summary": "更新后的 cue",
            "content": "新正文",
            "reason": "修正",
            "expected_revision": record.revision,
        }
    )
    assert updated["updated"] is True

    archived = await registry.get("memory_archive").execute(
        {"memory_id": record.id, "reason": "过时"}
    )
    assert archived["status"] == "archived"
    assert (await manager.read(record.id)) is None


# 函数说明：test_memory_tools_validate_required_arguments
# 用途：回归验证回归测试与测试辅助中的 `memory_tools_validate_required_arguments` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `ToolRegistry` →
# `register_memory_tools` → `register_memory_write_tools` → `pytest.raises` →
# `registry.get('memory_read').execute`；另有 1 个调用点。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match="'memory_id'")`。
#   预期异常：`pytest.raises(ValueError, match="'content'")`。
@pytest.mark.asyncio
async def test_memory_tools_validate_required_arguments(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    registry = ToolRegistry()
    register_memory_tools(registry, manager)
    register_memory_write_tools(registry, manager)

    with pytest.raises(ValueError, match="'memory_id'"):
        await registry.get("memory_read").execute({})
    with pytest.raises(ValueError, match="'content'"):
        await registry.get("memory_create").execute(
            {"title": "t", "summary": "s", "content": ""}
        )


# 函数说明：test_core_update_tool_requires_exact_current_user_statement
# 用途：回归验证回归测试与测试辅助中的
# `core_update_tool_requires_exact_current_user_statement` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `ToolRegistry` →
# `register_memory_tools` → `pytest.raises` → `tool.execute_with_context` →
# `ToolExecutionContext`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`await manager.core.load() == ''`。
#   预期异常：`pytest.raises(ValueError, match='copied exactly')`。
@pytest.mark.asyncio
async def test_core_update_tool_requires_exact_current_user_statement(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    registry = ToolRegistry()
    register_memory_tools(registry, manager)
    tool = registry.get("core_memory_update")
    arguments = {
        "key": "communication.language",
        "value": "始终使用中文交流。",
        "reason": "用户明确表达全局长期偏好",
        "explicit_user_statement": "以后都使用中文和我交流",
    }

    with pytest.raises(ValueError, match="copied exactly"):
        await tool.execute_with_context(
            arguments,
            ToolExecutionContext(
                tool_call=ToolCall(id="core-1", name="core_memory_update"),
                user_input="请帮我检查代码",
            ),
        )

    assert await manager.core.load() == ""


# 函数说明：test_core_update_tool_writes_through_harness
# 用途：回归验证回归测试与测试辅助中的 `core_update_tool_writes_through_harness` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `ToolRegistry` →
# `register_memory_tools` → `tool.execute_with_context` → `ToolExecutionContext` →
# `ToolCall`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result['created'] is True`。
#   验证条件：`'始终使用中文交流' in await manager.core.load()`。
@pytest.mark.asyncio
async def test_core_update_tool_writes_through_harness(memory_root: Path) -> None:
    manager = await _manager(memory_root)
    registry = ToolRegistry()
    register_memory_tools(registry, manager)
    tool = registry.get("core_memory_update")
    statement = "以后都使用中文和我交流"

    result = await tool.execute_with_context(
        {
            "key": "communication.language",
            "value": "始终使用中文交流。",
            "reason": "用户明确表达全局长期偏好",
            "explicit_user_statement": statement,
        },
        ToolExecutionContext(
            tool_call=ToolCall(id="core-1", name="core_memory_update"),
            user_input=f"请记住，{statement}。",
        ),
    )

    assert result["created"] is True
    assert "始终使用中文交流" in await manager.core.load()


# 函数说明：test_core_update_tool_description_explains_classification_litmus
# 用途：回归验证回归测试与测试辅助中的
# `core_update_tool_description_explains_classification_litmus` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `ToolRegistry` →
# `register_memory_tools`。
# 分支与异常：
#   验证条件：`'换到完全无关的项目仍须适用' in description`。
#   验证条件：`'全局安全/隐私约束' in description`。
#   验证条件：`'项目架构、选型、路径、实现限制和历史决定' in description`。
#   验证条件：`'属于普通记忆' in description`。
@pytest.mark.asyncio
async def test_core_update_tool_description_explains_classification_litmus(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    registry = ToolRegistry()
    register_memory_tools(registry, manager)

    description = registry.get("core_memory_update").definition.description

    assert "换到完全无关的项目仍须适用" in description
    assert "全局安全/隐私约束" in description
    assert "项目架构、选型、路径、实现限制和历史决定" in description
    assert "属于普通记忆" in description


# 函数说明：test_core_remove_tool_requires_current_user_revocation
# 用途：回归验证回归测试与测试辅助中的
# `core_remove_tool_requires_current_user_revocation` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   memory_root：长期记忆根目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager` → `manager.upsert_core` →
# `ToolRegistry` → `register_memory_tools` → `pytest.raises` →
# `tool.execute_with_context`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result == {'key': 'communication.language', 'removed': True}`。
#   验证条件：`'始终使用中文交流' not in await manager.core.load()`。
#   预期异常：`pytest.raises(ValueError, match='copied exactly')`。
@pytest.mark.asyncio
async def test_core_remove_tool_requires_current_user_revocation(
    memory_root: Path,
) -> None:
    manager = await _manager(memory_root)
    await manager.upsert_core(
        key="communication.language",
        value="始终使用中文交流。",
        reason="用户明确表达全局长期偏好",
        source_statement="以后都使用中文和我交流",
    )
    registry = ToolRegistry()
    register_memory_tools(registry, manager)
    tool = registry.get("core_memory_remove")
    arguments = {
        "key": "communication.language",
        "reason": "用户撤销了长期语言偏好",
        "explicit_user_statement": "不用再记住语言偏好",
    }

    with pytest.raises(ValueError, match="copied exactly"):
        await tool.execute_with_context(
            arguments,
            ToolExecutionContext(
                tool_call=ToolCall(id="core-2", name="core_memory_remove"),
                user_input="继续检查代码",
            ),
        )

    result = await tool.execute_with_context(
        arguments,
        ToolExecutionContext(
            tool_call=ToolCall(id="core-3", name="core_memory_remove"),
            user_input="不用再记住语言偏好",
        ),
    )

    assert result == {"key": "communication.language", "removed": True}
    assert "始终使用中文交流" not in await manager.core.load()


# 函数说明：test_atomic_write_failure_preserves_file_and_removes_temporary
# 用途：回归验证回归测试与测试辅助中的
# `atomic_write_failure_preserves_file_and_removes_temporary` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   monkeypatch：pytest 提供的临时替换依赖夹具，类型 `pytest.MonkeyPatch`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryStore` → `store.initialize` →
# `store.create` → `target.read_bytes` → `monkeypatch.setattr` → `pytest.raises`；另有 2
#  个调用点。
# 分支与异常：
#   验证条件：`target.read_bytes() == before`。
#   验证条件：`list(store.active_dir.iterdir()) == [target]`。
#   预期异常：`pytest.raises(OSError, match='simulated write failure')`。
# 副作用与资源：
#   文件或资源访问：`target.read_bytes`。
@pytest.mark.asyncio
async def test_atomic_write_failure_preserves_file_and_removes_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path)
    await store.initialize()
    record = await store.create(title="原记录", summary="原摘要", content="原正文")
    target = store.active_dir / f"{record.id}.md"
    before = target.read_bytes()

    # 函数说明：
    # test_atomic_write_failure_preserves_file_and_removes_temporary.reject_replace
    # 用途：拒绝`replace`，供回归测试与测试辅助使用。
    # 参数：
    #   source：输入来源或原始数据，类型 `Path`。
    #   destination：`destination`输入或配置值，类型 `Path`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def reject_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(memory_store.os, "replace", reject_replace)
    with pytest.raises(OSError, match="simulated write failure"):
        await store.update(record.id, content="新正文", reason="修正")

    assert target.read_bytes() == before
    assert list(store.active_dir.iterdir()) == [target]


# 函数说明：test_parallel_atomic_writes_use_distinct_temporary_files
# 用途：回归验证回归测试与测试辅助中的
# `parallel_atomic_writes_use_distinct_temporary_files` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   monkeypatch：pytest 提供的临时替换依赖夹具，类型 `pytest.MonkeyPatch`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryStore` → `Barrier` →
# `monkeypatch.setattr` → `asyncio.gather` → `asyncio.to_thread` → `target.read_text`；
# 另有 1 个调用点。
# 分支与异常：
#   验证条件：`len(set(temporary_paths)) == 2`。
#   验证条件：`target.read_text(encoding='utf-8') in {'first complete write', 'second
# complete write'}`。
#   验证条件：`list(tmp_path.iterdir()) == [target]`。
# 副作用与资源：
#   文件或资源访问：`target.read_text`。
#   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
@pytest.mark.asyncio
async def test_parallel_atomic_writes_use_distinct_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path)
    target = tmp_path / "M001.md"
    barrier = Barrier(2)
    temporary_paths: list[Path] = []
    original_replace = memory_store.os.replace

    # 函数说明：
    # test_parallel_atomic_writes_use_distinct_temporary_files.synchronized_replace
    # 用途：替换`synchronized`，供回归测试与测试辅助使用。
    # 参数：
    #   source：输入来源或原始数据，类型 `Path`。
    #   destination：传给 `original_replace` 的输入，类型 `Path`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`barrier.wait` →
    # `original_replace`。
    # 闭包依赖：从外层读取 `barrier`、`original_replace`、`temporary_paths`。
    def synchronized_replace(source: Path, destination: Path) -> None:
        temporary_paths.append(source)
        barrier.wait(timeout=10)
        original_replace(source, destination)

    monkeypatch.setattr(memory_store.os, "replace", synchronized_replace)
    await asyncio.gather(
        asyncio.to_thread(store._write_bytes, "first complete write", target),
        asyncio.to_thread(store._write_bytes, "second complete write", target),
    )

    assert len(set(temporary_paths)) == 2
    assert target.read_text(encoding="utf-8") in {
        "first complete write", "second complete write",
    }
    assert list(tmp_path.iterdir()) == [target]
