
from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.paths import runtime_data_path

from .models import (
    MemoryRecord,
    MemoryStatus,
    next_memory_id,
    normalize_memory_id,
    parse_memory_markdown,
)

DEFAULT_MEMORY_DIR = runtime_data_path("memory")
_MAX_MEMORY_FILE_BYTES = 512_000

logger = logging.getLogger("muharness.memory.store")


class MemoryStore:

    # 函数说明：MemoryStore.__init__
    # 用途：初始化 MemoryStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   memory_dir：记忆文件目录，类型 `str | Path`；默认 `DEFAULT_MEMORY_DIR`。
    #   max_active：允许同时活跃的数量上限，类型 `int`；默认 `25`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(memory_dir).expanduser().resolve` → `Path(memory_dir).expanduser` → `Path`。
    # 分支与异常：
    #   当 `max_active <= 0` 时，抛出
    # `ValueError('max_active must be greater than zero')`。
    # 副作用与资源：
    #   更新对象字段：`self.memory_dir`、`self.active_dir`、`self.archive_dir`、
    # `self.max_active`。
    def __init__(
        self,
        memory_dir: str | Path = DEFAULT_MEMORY_DIR,
        *,
        max_active: int = 25,
    ) -> None:
        self.memory_dir = Path(memory_dir).expanduser().resolve()
        self.active_dir = self.memory_dir / "active"
        self.archive_dir = self.memory_dir / "archive"
        self.max_active = max_active
        if max_active <= 0:
            raise ValueError("max_active must be greater than zero")

    # 函数说明：MemoryStore.initialize
    # 用途：初始化MemoryStore，供长期记忆管理与检索使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def initialize(self) -> None:

        await asyncio.to_thread(self.active_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(self.archive_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(self._repair_interrupted_archives)

    # 函数说明：MemoryStore.create
    # 用途：创建MemoryStore，供长期记忆管理与检索使用。
    # 参数：
    #   title：面向用户的标题，类型 `str`。
    #   summary：已有或新生成的摘要，类型 `str`。
    #   content：内容正文，类型 `str`。
    # 返回：类型 `MemoryRecord`；返回 `record`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._all_ids` → `datetime.now` →
    #  `MemoryRecord` → `next_memory_id` → `self._write`。
    async def create(
        self,
        *,
        title: str,
        summary: str,
        content: str,
    ) -> MemoryRecord:

        existing_ids = await self._all_ids()
        now = datetime.now(UTC)
        record = MemoryRecord(
            id=next_memory_id(existing_ids),
            title=title,
            summary=summary,
            content=content,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )
        await self._write(record)
        return record

    # 函数说明：MemoryStore.load
    # 用途：加载MemoryStore，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    # 返回：类型 `MemoryRecord | None`；按分支返回 `None`；
    # `await asyncio.to_thread(_read_record, path)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`normalize_memory_id` →
    # `self._resolve_path` → `asyncio.to_thread`。
    # 分支与异常：
    #   当 `path is None or not await asyncio.to_thread(path.is_file)` 时，返回 `None`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def load(self, memory_id: str) -> MemoryRecord | None:

        normalized = normalize_memory_id(memory_id)
        path = await self._resolve_path(normalized)
        if path is None or not await asyncio.to_thread(path.is_file):
            return None
        return await asyncio.to_thread(_read_record, path)

    # 函数说明：MemoryStore.read
    # 用途：读取MemoryStore，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    # 返回：类型 `MemoryRecord | None`；按分支返回 `None`；`updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.load` → `datetime.now` →
    # `MemoryRecord.model_validate` → `self._write`。
    # 分支与异常：
    #   当 `record is None or record.status is MemoryStatus.ARCHIVED` 时，返回 `None`。
    async def read(self, memory_id: str) -> MemoryRecord | None:

        record = await self.load(memory_id)
        if record is None or record.status is MemoryStatus.ARCHIVED:
            return None
        fields = record.model_dump()
        fields["access_count"] += 1
        fields["last_accessed_at"] = datetime.now(UTC)
        updated = MemoryRecord.model_validate(fields)
        await self._write(updated)
        return updated

    # 函数说明：MemoryStore.update
    # 用途：更新MemoryStore，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    #   title：面向用户的标题，类型 `str | None`；默认 `None`。
    #   summary：已有或新生成的摘要，类型 `str | None`；默认 `None`。
    #   content：内容正文，类型 `str`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    #   expected_revision：`expected_revision`输入或配置值，类型 `int | None`；默认
    # `None`。
    # 返回：类型 `MemoryRecord`；返回 `updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.load` → `fields.update` →
    # `datetime.now` → `MemoryRecord.model_validate` → `self._write`。
    # 分支与异常：
    #   当 `record is None` 时，抛出 `KeyError(f"memory '{memory_id}' not found")`。
    #   当 `record.status is not MemoryStatus.ACTIVE` 时，抛出
    # `ValueError('only active memory can be updated')`。
    #   当 `expected_revision is not None and record.revision !=…` 时，抛出
    # `ValueError(…)`。
    async def update(
        self,
        memory_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        content: str,
        reason: str,
        expected_revision: int | None = None,
    ) -> MemoryRecord:

        record = await self.load(memory_id)
        if record is None:
            raise KeyError(f"memory '{memory_id}' not found")
        if record.status is not MemoryStatus.ACTIVE:
            raise ValueError("only active memory can be updated")
        if expected_revision is not None and record.revision != expected_revision:
            raise ValueError(
                f"memory '{record.id}' revision conflict: "
                f"expected {expected_revision}, current {record.revision}"
            )
        fields = record.model_dump()
        fields.update(
            title=record.title if title is None else title,
            summary=record.summary if summary is None else summary,
            content=content,
            last_update_reason=reason,
            updated_at=datetime.now(UTC),
            revision=record.revision + 1,
        )
        updated = MemoryRecord.model_validate(fields)
        await self._write(updated)
        return updated

    # 函数说明：MemoryStore.archive
    # 用途：归档MemoryStore，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    # 返回：类型 `MemoryRecord`；按分支返回 `record`；`updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.load` → `fields.update` →
    # `datetime.now` → `MemoryRecord.model_validate` → `asyncio.to_thread` →
    # `updated.render_markdown`；另有 1 个调用点。
    # 分支与异常：
    #   当 `record is None` 时，抛出 `KeyError(f"memory '{memory_id}' not found")`。
    #   当 `record.status is MemoryStatus.ARCHIVED` 时，返回 `record`。
    #   捕获 `BaseException` 后，重新抛出原异常。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def archive(self, memory_id: str, *, reason: str) -> MemoryRecord:

        record = await self.load(memory_id)
        if record is None:
            raise KeyError(f"memory '{memory_id}' not found")
        if record.status is MemoryStatus.ARCHIVED:
            return record
        fields = record.model_dump()
        fields.update(
            status=MemoryStatus.ARCHIVED,
            archive_reason=reason,
            updated_at=datetime.now(UTC),
            revision=record.revision + 1,
        )
        updated = MemoryRecord.model_validate(fields)
        source = self.active_dir / f"{record.id}.md"
        target = self.archive_dir / f"{record.id}.md"
        await asyncio.to_thread(self._write_bytes, updated.render_markdown(), source)
        try:
            await asyncio.to_thread(os.replace, source, target)
        except BaseException:
            await asyncio.to_thread(self._write_bytes, record.render_markdown(), source)
            raise
        return updated

    # 函数说明：MemoryStore.list_active
    # 用途：列出活跃项，供长期记忆管理与检索使用。
    # 返回：类型 `tuple[MemoryRecord, ...]`；返回
    # `tuple(sorted(records, key=lambda record: record.id))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.active_dir.glob` →
    # `asyncio.to_thread` → `logger.warning`。
    # 分支与异常：
    #   当 `await asyncio.to_thread(path.is_symlink)` 时，跳过当前循环项。
    #   捕获 `(ValueError, OSError)` 后，执行异常处理调用 `logger.warning`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def list_active(self) -> tuple[MemoryRecord, ...]:

        records: list[MemoryRecord] = []
        for path in sorted(self.active_dir.glob("M*.md")):
            if await asyncio.to_thread(path.is_symlink):
                continue
            try:
                record = await asyncio.to_thread(_read_record, path)
                if record.status is MemoryStatus.ACTIVE:
                    records.append(record)
            except (ValueError, OSError) as exc:
                logger.warning("skip unreadable memory %s: %s", path.name, exc)
        return tuple(sorted(records, key=lambda record: record.id))

    # 函数说明：MemoryStore.list_archived
    # 用途：列出`archived`，供长期记忆管理与检索使用。
    # 返回：类型 `tuple[MemoryRecord, ...]`；返回
    # `tuple(sorted(records, key=lambda record: record.updated_at, reverse=True))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.archive_dir.glob` →
    # `asyncio.to_thread` → `logger.warning`。
    # 分支与异常：
    #   当 `await asyncio.to_thread(path.is_symlink)` 时，跳过当前循环项。
    #   捕获 `(ValueError, OSError)` 后，执行异常处理调用 `logger.warning`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def list_archived(self) -> tuple[MemoryRecord, ...]:

        records: list[MemoryRecord] = []
        for path in self.archive_dir.glob("M*.md"):
            if await asyncio.to_thread(path.is_symlink):
                continue
            try:
                record = await asyncio.to_thread(_read_record, path)
                if record.status is MemoryStatus.ARCHIVED:
                    records.append(record)
            except (ValueError, OSError) as exc:
                logger.warning("skip unreadable archived memory %s: %s", path.name, exc)
        return tuple(
            sorted(records, key=lambda record: record.updated_at, reverse=True)
        )

    # 函数说明：MemoryStore.count_active
    # 用途：统计活跃项，供长期记忆管理与检索使用。
    # 返回：类型 `int`；返回 `len(await self.list_active())`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.list_active`。
    async def count_active(self) -> int:
        return len(await self.list_active())

    # 函数说明：MemoryStore._all_ids
    # 用途：在长期记忆管理与检索中处理 `_all_ids`，通过 `directory.glob` 完成首个内部处
    # 理步骤。
    # 返回：类型 `set[str]`；返回 `ids`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`directory.glob` → `self.load` →
    # `ids.add`。
    async def _all_ids(self) -> set[str]:
        ids: set[str] = set()
        for directory in (self.active_dir, self.archive_dir):
            for path in directory.glob("M*.md"):
                record = await self.load(path.stem)
                if record is not None:
                    ids.add(record.id)
        return ids

    # 函数说明：MemoryStore._resolve_path
    # 用途：解析或定位路径，供长期记忆管理与检索使用。
    # 参数：
    #   memory_id：目标记忆标识，类型 `str`。
    # 返回：类型 `Path | None`；按分支返回 `path`；`None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread`。
    # 分支与异常：
    #   当 `await asyncio.to_thread(path.is_file) and (not await…` 时，返回 `path`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def _resolve_path(self, memory_id: str) -> Path | None:
        for directory in (self.active_dir, self.archive_dir):
            path = directory / f"{memory_id}.md"
            if await asyncio.to_thread(path.is_file) and not await asyncio.to_thread(
                path.is_symlink
            ):
                return path
        return None

    # 函数说明：MemoryStore._write
    # 用途：写入MemoryStore，供长期记忆管理与检索使用。
    # 参数：
    #   record：待处理的数据记录，类型 `MemoryRecord`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread` →
    # `record.render_markdown`。
    # 分支与异常：
    #   当 `record.status is not MemoryStatus.ACTIVE` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def _write(self, record: MemoryRecord) -> None:
        if record.status is not MemoryStatus.ACTIVE:
            raise ValueError("inactive memory cannot be written to active directory")
        target = self.active_dir / f"{record.id}.md"
        await asyncio.to_thread(self._write_bytes, record.render_markdown(), target)

    # 函数说明：MemoryStore._write_bytes
    # 用途：写入`bytes`，供长期记忆管理与检索使用。
    # 参数：
    #   content：内容正文，类型 `str`。
    #   target：传给 `os.replace` 的输入，类型 `Path`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`content.encode` →
    # `target.parent.mkdir` → `NamedTemporaryFile` → `Path` → `stream.write` →
    # `os.replace`；另有 1 个调用点。
    # 分支与异常：
    #   当 `len(encoded) > _MAX_MEMORY_FILE_BYTES` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   文件或资源访问：`target.parent.mkdir`、`os.replace`、`temporary.unlink`。
    def _write_bytes(self, content: str, target: Path) -> None:
        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_MEMORY_FILE_BYTES:
            raise ValueError(
                f"memory file exceeds {_MAX_MEMORY_FILE_BYTES} bytes: {target.name}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            # 同目录临时文件保证原子替换；随机名称避免同进程写入时重名。
            with NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    # 函数说明：MemoryStore._repair_interrupted_archives
    # 用途：修复`interrupted_archives`，供长期记忆管理与检索使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.active_dir.glob` →
    # `path.is_symlink` → `path.is_file` → `path.stat` → `parse_memory_markdown` →
    # `path.read_text`；另有 2 个调用点。
    # 分支与异常：
    #   当 `path.is_symlink() or not path.is_file()` 时，跳过当前循环项。
    #   当 `path.stat().st_size > _MAX_MEMORY_FILE_BYTES` 时，跳过当前循环项。
    #   当 `record.id != path.stem or record.status is not…` 时，跳过当前循环项。
    #   捕获 `(OSError, ValueError)` 后，执行异常处理调用 `logger.warning`。
    # 副作用与资源：
    #   文件或资源访问：`path.read_text`、`os.replace`。
    def _repair_interrupted_archives(self) -> None:

        for path in self.active_dir.glob("M*.md"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.stat().st_size > _MAX_MEMORY_FILE_BYTES:
                    continue
                record = parse_memory_markdown(path.read_text(encoding="utf-8"))
                if record.id != path.stem or record.status is not MemoryStatus.ARCHIVED:
                    continue
                os.replace(path, self.archive_dir / path.name)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "failed to repair interrupted memory archive %s: %s",
                    path.name,
                    exc,
                )


# 函数说明：_read_record
# 用途：读取记录，供长期记忆管理与检索使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
# 返回：类型 `MemoryRecord`；返回 `record`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.stat` → `path.read_text` →
# `parse_memory_markdown`。
# 分支与异常：
#   当 `path.stat().st_size > _MAX_MEMORY_FILE_BYTES` 时，抛出
# `ValueError(f'memory file too large: {path.name}')`。
#   当 `record.id != path.stem` 时，抛出 `ValueError(…)`。
#   当 `record.status is not expected_status` 时，抛出 `ValueError(…)`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
def _read_record(path: Path) -> MemoryRecord:
    if path.stat().st_size > _MAX_MEMORY_FILE_BYTES:
        raise ValueError(f"memory file too large: {path.name}")
    text = path.read_text(encoding="utf-8")
    record = parse_memory_markdown(text)
    if record.id != path.stem:
        raise ValueError(
            f"memory id does not match filename: {record.id} != {path.stem}"
        )
    expected_status = (
        MemoryStatus.ARCHIVED
        if path.parent.name == "archive"
        else MemoryStatus.ACTIVE
    )
    if record.status is not expected_status:
        raise ValueError(
            f"memory status does not match directory: {record.status.value}"
        )
    return record


__all__ = ["DEFAULT_MEMORY_DIR", "MemoryStore"]
