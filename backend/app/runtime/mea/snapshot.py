
"""Auditor 只读守卫用的 workspace 快照（移植 LHH adapters/claude_permissions.py）。

审计前后各做一次快照：记录路径、类型、mode、size、mtime，4MB 以下文件再加 sha256。
两次快照有任何差异，本轮审计就作废。

和 LHH 的两处不同，都是为了在 Windows 上不误报：
- 元数据用 ``os.lstat`` 现取，不用 ``DirEntry.stat()``。Windows 上后者读的是父目录索引里
  缓存的时间戳，NTFS 会在子项句柄关闭后才回写，所以连续两次快照之间目录 mtime 会自己变。
- 目录只记类型和 mode，不记 mtime。目录内容的变化已经由子项的新增、删除体现。
"""

from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

MAX_DIGEST_BYTES = 4 * 1024 * 1024
_MAX_ERRORS = 100

Record = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:

    records: Mapping[str, Record]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SnapshotDiff:

    added: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    type_changed: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    # 函数说明：SnapshotDiff.mutated
    # 用途：返回 `bool(self.added or self.deleted or self.changed or self.type_changed)`
    # ，提供 SnapshotDiff 的派生值。
    # 返回：类型 `bool`；返回
    # `bool(self.added or self.deleted or self.changed or self.type_changed)`。
    @property
    def mutated(self) -> bool:
        return bool(self.added or self.deleted or self.changed or self.type_changed)

    # 函数说明：SnapshotDiff.paths
    # 用途：处理规划、执行、审计协作中的 `paths` 数据；结果及边界条件见下方说明。
    # 参数：
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `20`。
    # 返回：类型 `list[str]`；返回 `combined[:limit]`。
    def paths(self, limit: int = 20) -> list[str]:
        combined = [
            *(f"+ {path}" for path in self.added),
            *(f"- {path}" for path in self.deleted),
            *(f"~ {path}" for path in self.changed),
            *(f"! {path}" for path in self.type_changed),
        ]
        return combined[:limit]


# 函数说明：snapshot_workspace
# 用途：遍历 workspace（不跟随符号链接）。
# 参数：
#   workspace：目标工作区，类型 `str | os.PathLike[str]`。
#   exclude：`exclude`输入或配置值，类型 `Iterable[str | os.PathLike[str]]`；默认 `()`。
# 返回：类型 `WorkspaceSnapshot`；按分支返回
# `WorkspaceSnapshot(records={}, errors=(f'workspace does not exist: {root}',))`；
# `WorkspaceSnapshot(records=records, errors=tuple(errors[:_MAX_ERRORS]))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(workspace).expanduser().resolve`
#  → `Path(workspace).expanduser` → `Path` → `root.exists` → `WorkspaceSnapshot` →
# `_resolve_excluded`；另有 10 个调用点。
# 分支与异常：
#   当 `not root.exists()` 时，返回 `WorkspaceSnapshot(…)`。
#   捕获 `OSError` 后，跳过当前循环项，继续处理后续项。
#   当 `path in excluded` 时，跳过当前循环项。
#   捕获 `OSError` 后，执行异常处理调用 `errors.append`、`type`。
def snapshot_workspace(
    workspace: str | os.PathLike[str],
    exclude: Iterable[str | os.PathLike[str]] = (),
) -> WorkspaceSnapshot:
    """遍历 workspace（不跟随符号链接）。``exclude`` 可写相对 workspace 的路径或绝对路径。"""

    root = Path(workspace).expanduser().resolve()
    if not root.exists():
        return WorkspaceSnapshot(records={}, errors=(f"workspace does not exist: {root}",))
    excluded = {_resolve_excluded(root, item) for item in exclude}
    records: dict[str, Record] = {}
    errors: list[str] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            errors.append(f"{directory}: {type(exc).__name__}: {exc}")
            continue
        for entry in entries:
            path = Path(entry.path)
            if path in excluded:
                continue
            try:
                relative = path.relative_to(root).as_posix()
                info = os.lstat(path)
                mode = info.st_mode
                if stat_module.S_ISLNK(mode):
                    records[relative] = ("symlink", mode, info.st_mtime_ns, os.readlink(path))
                elif stat_module.S_ISDIR(mode):
                    records[relative] = ("dir", mode)
                    stack.append(path)
                elif stat_module.S_ISREG(mode):
                    records[relative] = (
                        "file",
                        mode,
                        info.st_size,
                        info.st_mtime_ns,
                        _small_file_digest(path, info.st_size),
                    )
                else:
                    records[relative] = ("other", mode, info.st_size, info.st_mtime_ns)
            except OSError as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return WorkspaceSnapshot(records=records, errors=tuple(errors[:_MAX_ERRORS]))


# 函数说明：snapshot_diff
# 用途：比较快照，供规划、执行、审计协作使用。
# 参数：
#   before：`before`输入或配置值，类型 `WorkspaceSnapshot`。
#   after：`after`输入或配置值，类型 `WorkspaceSnapshot`。
# 返回：类型 `SnapshotDiff`；返回 `SnapshotDiff(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SnapshotDiff`。
# 分支与异常：
#   当 `old == new` 时，跳过当前循环项。
def snapshot_diff(before: WorkspaceSnapshot, after: WorkspaceSnapshot) -> SnapshotDiff:
    before_paths = set(before.records)
    after_paths = set(after.records)
    changed: list[str] = []
    type_changed: list[str] = []
    for path in sorted(before_paths & after_paths):
        old = before.records[path]
        new = after.records[path]
        if old == new:
            continue
        if old[0] != new[0]:
            type_changed.append(path)
        else:
            changed.append(path)
    return SnapshotDiff(
        added=tuple(sorted(after_paths - before_paths)),
        deleted=tuple(sorted(before_paths - after_paths)),
        changed=tuple(changed),
        type_changed=tuple(type_changed),
        errors=(*before.errors, *after.errors)[:_MAX_ERRORS],
    )


# 函数说明：snapshot_digest
# 用途：快照的稳定摘要，存进 round 作为审计轨迹。
# 参数：
#   snapshot：快照输入或配置值，类型 `WorkspaceSnapshot`。
# 返回：类型 `str`；返回 `hashlib.sha256(payload.encode('utf-8')).hexdigest()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps` →
# `hashlib.sha256(payload.encode('utf-8')).hexdigest` → `hashlib.sha256` →
# `payload.encode`。
def snapshot_digest(snapshot: WorkspaceSnapshot) -> str:
    """快照的稳定摘要，存进 round 作为审计轨迹。"""

    payload = json.dumps(
        {path: list(record) for path, record in sorted(snapshot.records.items())},
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# 函数说明：_resolve_excluded
# 用途：解析或定位`excluded`，供规划、执行、审计协作使用。
# 参数：
#   root：当前操作的根目录，类型 `Path`。
#   item：当前集合元素，类型 `str | os.PathLike[str]`。
# 返回：类型 `Path`；返回 `path.resolve()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(item).expanduser` → `Path` →
# `path.is_absolute` → `path.resolve`。
def _resolve_excluded(root: Path, item: str | os.PathLike[str]) -> Path:
    path = Path(item).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


# 函数说明：_small_file_digest
# 用途：在规划、执行、审计协作中处理 `_small_file_digest`，通过 `hashlib.sha256` 完成首
# 个内部处理步骤。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
#   size：`size`输入或配置值，类型 `int`。
# 返回：类型 `str | None`；按分支返回 `None`；`digest.hexdigest()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`hashlib.sha256` → `path.open` →
# `handle.read` → `digest.update` → `digest.hexdigest`。
# 分支与异常：
#   当 `size > MAX_DIGEST_BYTES` 时，返回 `None`。
# 副作用与资源：
#   文件或资源访问：`path.open`。
def _small_file_digest(path: Path, size: int) -> str | None:
    if size > MAX_DIGEST_BYTES:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(128 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "MAX_DIGEST_BYTES",
    "SnapshotDiff",
    "WorkspaceSnapshot",
    "snapshot_diff",
    "snapshot_digest",
    "snapshot_workspace",
]
