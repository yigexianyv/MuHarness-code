"""工作区文件快照：按原始字节保存，用于"回到第 N 步之前"时恢复文件。

不依赖 git，也不碰用户自己项目的 git：文件内容按 sha256 去重存进数据目录，
每个快照只是一份"相对路径 → 内容哈希"的清单。没有变化的文件不会重复保存。

范围和上限（超过就不拍快照，界面上明确提示"此后不能回退文件"）：
- 跳过依赖、缓存、版本库目录和数据目录，跳过符号链接；
- 单个文件超过 1 MiB 只记为"已跳过"，回退时既不恢复也不删除；
- 一个快照最多 1000 个文件、20 MiB。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat as stat_module
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

MAX_FILE_BYTES = 1024 * 1024
MAX_SNAPSHOT_BYTES = 20 * 1024 * 1024
MAX_SNAPSHOT_FILES = 1000

EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".next",
        "dist",
        "build",
    }
)

SKIPPED_TOO_LARGE = "too_large"
SKIPPED_SYMLINK = "symlink"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspace_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class FileEntry:
    """sha256 为空表示文件在范围内但未保存内容（skipped 说明原因）。"""

    sha256: str | None = None
    size: int = 0
    skipped: str | None = None


Manifest = Mapping[str, FileEntry]


@dataclass(frozen=True, slots=True)
class CaptureResult:
    snapshot_id: str | None
    error: str | None = None
    manifest: Manifest = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RestorePlan:
    """把工作区恢复到目标快照需要做的事。"""

    restore: tuple[str, ...] = ()
    delete: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.restore or self.delete)


class SnapshotLimitExceeded(Exception):
    pass


class WorkspaceSnapshotStore:

    def __init__(
        self,
        database_path: str | Path,
        workspace_root: str | Path,
        *,
        blob_dir: str | Path | None = None,
        exclude_paths: Iterable[str | Path] = (),
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.blob_dir = (
            Path(blob_dir).expanduser().resolve()
            if blob_dir is not None
            else self.database_path.parent / "workspace_snapshots"
        )
        excluded = [Path(path).expanduser().resolve() for path in exclude_paths]
        excluded.append(self.blob_dir)
        excluded.append(self.database_path.parent)
        self._exclude_paths = tuple(
            path
            for path in excluded
            if path != self.workspace_root and _is_inside(path, self.workspace_root)
        )
        # (相对路径) -> (mtime_ns, size, sha256)：未变化的文件不重复计算哈希
        self._hash_cache: dict[str, tuple[int, int, str]] = {}

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.database_path) as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    async def capture(self) -> CaptureResult:
        try:
            manifest = await asyncio.to_thread(self._scan, True)
        except SnapshotLimitExceeded as exc:
            return CaptureResult(snapshot_id=None, error=str(exc))
        except OSError as exc:
            return CaptureResult(
                snapshot_id=None,
                error=f"读取工作区失败：{type(exc).__name__}: {exc}",
            )
        manifest_json = _manifest_json(manifest)
        snapshot_id = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        async with aiosqlite.connect(self.database_path) as database:
            await database.execute(
                """
                INSERT OR IGNORE INTO workspace_snapshots (
                    snapshot_id, manifest_json, created_at
                ) VALUES (?, ?, ?)
                """,
                (snapshot_id, manifest_json, datetime.now(UTC).isoformat()),
            )
            await database.commit()
        return CaptureResult(snapshot_id=snapshot_id, manifest=manifest)

    async def current_manifest(self) -> Manifest:
        """只计算当前工作区清单，不保存文件内容。"""
        return await asyncio.to_thread(self._scan, False)

    async def load(self, snapshot_id: str) -> Manifest | None:
        async with aiosqlite.connect(self.database_path) as database:
            cursor = await database.execute(
                "SELECT manifest_json FROM workspace_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            path: FileEntry(**entry) for path, entry in json.loads(row[0]).items()
        }

    @staticmethod
    def plan(target: Manifest, current: Manifest) -> RestorePlan:
        restore: list[str] = []
        delete: list[str] = []
        skipped: list[str] = []
        for path, entry in sorted(target.items()):
            if entry.sha256 is None:
                skipped.append(path)
                continue
            now = current.get(path)
            if now is None or now.sha256 != entry.sha256:
                restore.append(path)
        for path, entry in sorted(current.items()):
            if path in target or entry.sha256 is None:
                # 过大或符号链接：不知道它在目标时刻的内容，保持原样
                continue
            delete.append(path)
        return RestorePlan(
            restore=tuple(restore),
            delete=tuple(delete),
            skipped=tuple(skipped),
        )

    async def restore(self, target: Manifest) -> RestorePlan:
        current = await self.current_manifest()
        plan = self.plan(target, current)
        await asyncio.to_thread(self._apply, target, plan)
        return plan

    def _apply(self, target: Manifest, plan: RestorePlan) -> None:
        for path in plan.restore:
            entry = target[path]
            assert entry.sha256 is not None
            data = self._blob_path(entry.sha256).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry.sha256:
                raise OSError(f"快照内容损坏：{path}")
            destination = self._resolve(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.rewind-tmp")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            self._hash_cache.pop(path, None)
        for path in plan.delete:
            destination = self._resolve(path)
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            self._hash_cache.pop(path, None)

    def _scan(self, store_blobs: bool) -> dict[str, FileEntry]:
        manifest: dict[str, FileEntry] = {}
        total = 0
        if not self.workspace_root.exists():
            return manifest
        for directory, dir_names, file_names in os.walk(self.workspace_root):
            base = Path(directory)
            dir_names[:] = sorted(
                name
                for name in dir_names
                if name not in EXCLUDED_DIR_NAMES
                and not (base / name).is_symlink()
                and not self._excluded(base / name)
            )
            for name in sorted(file_names):
                full = base / name
                if self._excluded(full) or name.endswith(".rewind-tmp"):
                    continue
                relative = full.relative_to(self.workspace_root).as_posix()
                info = os.lstat(full)
                if stat_module.S_ISLNK(info.st_mode):
                    manifest[relative] = FileEntry(skipped=SKIPPED_SYMLINK)
                    continue
                if not stat_module.S_ISREG(info.st_mode):
                    continue
                if info.st_size > MAX_FILE_BYTES:
                    manifest[relative] = FileEntry(
                        size=info.st_size, skipped=SKIPPED_TOO_LARGE
                    )
                    continue
                total += info.st_size
                if len(manifest) + 1 > MAX_SNAPSHOT_FILES:
                    raise SnapshotLimitExceeded(
                        f"工作区文件超过 {MAX_SNAPSHOT_FILES} 个，未保存文件快照"
                    )
                if total > MAX_SNAPSHOT_BYTES:
                    raise SnapshotLimitExceeded(
                        f"工作区文件超过 {MAX_SNAPSHOT_BYTES // (1024 * 1024)} MiB，"
                        "未保存文件快照"
                    )
                manifest[relative] = FileEntry(
                    sha256=self._file_hash(relative, full, info, store_blobs),
                    size=info.st_size,
                )
        return manifest

    def _file_hash(
        self,
        relative: str,
        full: Path,
        info: os.stat_result,
        store_blob: bool,
    ) -> str:
        cached = self._hash_cache.get(relative)
        if cached is not None and cached[:2] == (info.st_mtime_ns, info.st_size):
            digest = cached[2]
            if not store_blob or self._blob_path(digest).exists():
                return digest
        data = full.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if store_blob:
            blob = self._blob_path(digest)
            if not blob.exists():
                blob.parent.mkdir(parents=True, exist_ok=True)
                # 不同会话的运行可能同时拍快照：临时文件名各不相同
                temporary = blob.with_name(f"{blob.name}.{uuid4().hex}.tmp")
                temporary.write_bytes(data)
                os.replace(temporary, blob)
        self._hash_cache[relative] = (info.st_mtime_ns, info.st_size, digest)
        return digest

    def _blob_path(self, digest: str) -> Path:
        return self.blob_dir / digest[:2] / digest

    def _resolve(self, relative: str) -> Path:
        destination = (self.workspace_root / relative).resolve()
        if not _is_inside(destination, self.workspace_root):
            raise OSError(f"路径越出工作区：{relative}")
        return destination

    def _excluded(self, path: Path) -> bool:
        return any(
            path == excluded or _is_inside(path, excluded)
            for excluded in self._exclude_paths
        )


def manifest_digest(manifest: Manifest) -> str:
    return hashlib.sha256(_manifest_json(manifest).encode("utf-8")).hexdigest()


def _manifest_json(manifest: Manifest) -> str:
    return json.dumps(
        {
            path: {
                key: value
                for key, value in (
                    ("sha256", entry.sha256),
                    ("size", entry.size),
                    ("skipped", entry.skipped),
                )
                if value is not None
            }
            for path, entry in sorted(manifest.items())
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "MAX_FILE_BYTES",
    "MAX_SNAPSHOT_BYTES",
    "MAX_SNAPSHOT_FILES",
    "CaptureResult",
    "FileEntry",
    "Manifest",
    "RestorePlan",
    "WorkspaceSnapshotStore",
    "manifest_digest",
]
