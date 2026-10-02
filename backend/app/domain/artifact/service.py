
from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.tools.builtin._workspace import (
    resolve_workspace_path,
    workspace_root_path,
)

from .models import Artifact, ArtifactKind
from .store import SQLiteArtifactStore

logger = logging.getLogger("muharness.artifact")

MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024
_UNSAFE_FILENAME_RE = re.compile(r"[\x00-\x1f\x7f/\\]")
_ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_PORTABLE_MIME_TYPES = {
    ".csv": "text/csv",
}

Broadcaster = Callable[[str, Any], Awaitable[None]]


class ArtifactTooLargeError(ValueError):
    pass


class ArtifactService:

    # 函数说明：ArtifactService.__init__
    # 用途：初始化 ArtifactService；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SQLiteArtifactStore`。
    #   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path | None`；默认
    # `None`。
    #   managed_dir：受管交付物保存目录，类型 `str | Path | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace_root_path` →
    # `Path(managed_dir).expanduser().resolve` → `Path(managed_dir).expanduser` → `Path`
    # 。
    # 副作用与资源：
    #   更新对象字段：`self.store`、`self.workspace_root`、`self.managed_dir`、
    # `self._broadcaster`。
    def __init__(
        self,
        store: SQLiteArtifactStore,
        workspace_root: str | Path | None = None,
        *,
        managed_dir: str | Path | None = None,
    ) -> None:
        self.store = store
        self.workspace_root = workspace_root_path(workspace_root)
        if managed_dir is None:
            managed_dir = self.workspace_root.parent / "artifacts"
        self.managed_dir = Path(managed_dir).expanduser().resolve()
        self._broadcaster: Broadcaster | None = None

    # 函数说明：ArtifactService.set_broadcaster
    # 用途：设置`broadcaster`，供交付物发布与存储使用。
    # 参数：
    #   broadcaster：通知广播回调，类型 `Broadcaster`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._broadcaster`。
    def set_broadcaster(self, broadcaster: Broadcaster) -> None:

        self._broadcaster = broadcaster


    # 函数说明：ArtifactService.publish_file
    # 用途：校验工作区文件和大小上限，复制到受管目录并计算摘要，原子发布副本后登记交付物
    # 及通知。
    # 参数：
    #   path：目标文件或目录路径，类型 `str`。
    #   title：面向用户的标题，类型 `str`；默认 `''`。
    #   description：补充描述，类型 `str | None`；默认 `None`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   task_id：目标任务标识，类型 `str | None`；默认 `None`。
    # 返回：类型 `Artifact`；返回 `artifact`。
    # 设计约束：发布的是独立受管副本；工作区源文件后续修改不会改变已发布内容。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`resolve_workspace_path` →
    # `asyncio.to_thread` → `_safe_filename` → `Path` → `Path(filename).suffix.lower` →
    # `mimetypes.guess_type`；另有 5 个调用点。
    # 分支与异常：
    #   当 `not await asyncio.to_thread(source.is_file)` 时，抛出 `ValueError(…)`。
    #   当 `size_bytes > MAX_ARTIFACT_BYTES` 时，抛出 `ArtifactTooLargeError(…)`。
    #   捕获 `Exception` 后，重新抛出原异常。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def publish_file(
        self,
        *,
        path: str,
        title: str = "",
        description: str | None = None,
        run_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> Artifact:

        """复制并校验本地文件，登记为可通过服务访问的产物。"""
        source = resolve_workspace_path(self.workspace_root, path)
        if not await asyncio.to_thread(source.is_file):
            raise ValueError("artifact path must reference a file in the workspace")
        size_bytes = (await asyncio.to_thread(source.stat)).st_size
        if size_bytes > MAX_ARTIFACT_BYTES:
            raise ArtifactTooLargeError(
                f"artifact exceeds size limit ({size_bytes} > {MAX_ARTIFACT_BYTES})"
            )

        filename = _safe_filename(Path(path).name)
        mime_type = _PORTABLE_MIME_TYPES.get(
            Path(filename).suffix.lower(),
            mimetypes.guess_type(filename)[0] or "application/octet-stream",
        )

        artifact_id = uuid.uuid4().hex
        artifact_dir = self.managed_dir / artifact_id
        await asyncio.to_thread(artifact_dir.mkdir, parents=True, exist_ok=False)
        final_path = artifact_dir / filename
        temporary_path = artifact_dir / f".{artifact_id}.tmp"
        stored = False

        try:
            size_bytes, sha256 = await asyncio.to_thread(
                self._copy_and_digest,
                source,
                temporary_path,
            )
            await asyncio.to_thread(os.replace, temporary_path, final_path)
            artifact = Artifact(
                id=artifact_id,
                kind=ArtifactKind.FILE,
                title=title or filename,
                description=description,
                filename=filename,
                mime_type=mime_type,
                size_bytes=size_bytes,
                sha256=sha256,
                run_id=run_id,
                conversation_id=conversation_id,
                task_id=task_id,
                source_url=None,
                created_at=datetime.now(UTC),
            )
            try:
                await self.store.create(artifact)
                stored = True
            except Exception:
                await asyncio.to_thread(_cleanup_artifact_dir, artifact_dir)
                raise
        finally:
            await asyncio.to_thread(_cleanup_temp_file, temporary_path)
            if not stored:
                await asyncio.to_thread(_cleanup_artifact_dir, artifact_dir)

        await self._notify(artifact)
        return artifact

    # 函数说明：ArtifactService.publish_url
    # 用途：登记外部 URL 产物并通知订阅方。
    # 参数：
    #   url：目标 HTTP 地址，类型 `str`。
    #   title：面向用户的标题，类型 `str`；默认 `''`。
    #   description：补充描述，类型 `str | None`；默认 `None`。
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   task_id：目标任务标识，类型 `str | None`；默认 `None`。
    # 返回：类型 `Artifact`；返回 `artifact`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`urlsplit` → `parsed.scheme.lower`
    #  → `Artifact` → `datetime.now` → `self.store.create` → `self._notify`。
    # 分支与异常：
    #   当 `parsed.scheme.lower() not in ('http', 'https') or not…` 时，抛出
    # `ValueError('artifact url must be http(s)')`。
    async def publish_url(
        self,
        *,
        url: str,
        title: str = "",
        description: str | None = None,
        run_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> Artifact:

        """登记外部 URL 产物并通知订阅方。"""
        parsed = urlsplit(url.strip())
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            raise ValueError("artifact url must be http(s)")
        normalized_url = url.strip()

        artifact = Artifact(
            kind=ArtifactKind.URL,
            title=title or normalized_url,
            description=description,
            filename=None,
            mime_type=None,
            size_bytes=0,
            sha256=None,
            run_id=run_id,
            conversation_id=conversation_id,
            task_id=task_id,
            source_url=normalized_url,
            created_at=datetime.now(UTC),
        )
        await self.store.create(artifact)
        await self._notify(artifact)
        return artifact


    # 函数说明：ArtifactService.file_path
    # 用途：在交付物发布与存储中处理 `file_path`，通过 `self.store.get` 完成首个内部处理
    # 步骤。
    # 参数：
    #   artifact_id：交付物标识，类型 `str`。
    # 返回：类型 `Path | None`；按分支返回 `None`；
    # `self.managed_dir / artifact.id / artifact.filename`。
    # 分支与异常：
    #   当 `artifact is None or artifact.kind is not ArtifactKind.FILE` 时，返回 `None`
    # 。
    #   当 `not artifact.filename` 时，返回 `None`。
    async def file_path(self, artifact_id: str) -> Path | None:

        artifact = await self.store.get(artifact_id)
        if artifact is None or artifact.kind is not ArtifactKind.FILE:
            return None
        if not artifact.filename:
            return None
        return self.managed_dir / artifact.id / artifact.filename

    # 函数说明：ArtifactService.delete_for_conversation
    # 用途：删除会话，供交付物发布与存储使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`；默认 `()`。
    #   task_ids：任务输入或配置值，类型 `tuple[str, ...]`；默认 `()`。
    # 返回：类型 `tuple[str, ...]`；返回 `artifact_ids`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `self.store.list_related_to_conversation` → `_ARTIFACT_ID_RE.fullmatch` →
    # `asyncio.to_thread` → `self.store.delete_many`。
    # 分支与异常：
    #   当 `artifact.kind is not ArtifactKind.FILE` 时，跳过当前循环项。
    #   当 `not _ARTIFACT_ID_RE.fullmatch(artifact.id)` 时，抛出
    # `ValueError(f'invalid managed artifact id: {artifact.id}')`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def delete_for_conversation(
        self,
        conversation_id: str,
        *,
        run_ids: tuple[str, ...] = (),
        task_ids: tuple[str, ...] = (),
    ) -> tuple[str, ...]:

        artifacts = await self.store.list_related_to_conversation(
            conversation_id,
            run_ids=run_ids,
            task_ids=task_ids,
        )
        for artifact in artifacts:
            if artifact.kind is not ArtifactKind.FILE:
                continue
            if not _ARTIFACT_ID_RE.fullmatch(artifact.id):
                raise ValueError(f"invalid managed artifact id: {artifact.id}")
            artifact_dir = self.managed_dir / artifact.id
            await asyncio.to_thread(_delete_managed_artifact_dir, artifact_dir)
        artifact_ids = tuple(artifact.id for artifact in artifacts)
        await self.store.delete_many(artifact_ids)
        return artifact_ids


    # 函数说明：ArtifactService._copy_and_digest
    # 用途：复制文件内容时累计摘要，返回复制内容对应的哈希。
    # 参数：
    #   source：输入来源或原始数据，类型 `Path`。
    #   destination：`destination`输入或配置值，类型 `Path`。
    # 返回：类型 `tuple[int, str]`；返回 `(size_bytes, digest.hexdigest())`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`hashlib.sha256` → `source.open` →
    #  `destination.open` → `src.read` → `digest.update` → `dst.write`；另有 4 个调用点
    # 。
    # 分支与异常：
    #   当 `not block` 时，结束当前循环。
    #   当 `size_bytes > MAX_ARTIFACT_BYTES` 时，抛出 `ArtifactTooLargeError(…)`。
    # 副作用与资源：
    #   文件或资源访问：`source.open`、`destination.open`。
    @staticmethod
    def _copy_and_digest(source: Path, destination: Path) -> tuple[int, str]:

        digest = hashlib.sha256()
        size_bytes = 0
        with source.open("rb") as src, destination.open("xb") as dst:
            while True:
                block = src.read(_CHUNK_SIZE)
                if not block:
                    break
                size_bytes += len(block)
                if size_bytes > MAX_ARTIFACT_BYTES:
                    raise ArtifactTooLargeError(
                        "artifact exceeds size limit "
                        f"({size_bytes} > {MAX_ARTIFACT_BYTES})"
                    )
                digest.update(block)
                dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        return size_bytes, digest.hexdigest()

    # 函数说明：ArtifactService._notify
    # 用途：通知ArtifactService，供交付物发布与存储使用。
    # 参数：
    #   artifact：交付物输入或配置值，类型 `Artifact`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._broadcaster` →
    # `artifact.public_dict` → `logger.warning`。
    # 分支与异常：
    #   捕获 `Exception` 后，执行异常处理调用 `logger.warning`。
    async def _notify(self, artifact: Artifact) -> None:
        if self._broadcaster is not None:
            try:
                await self._broadcaster(
                    "artifact.created", {"artifact": artifact.public_dict()}
                )
            except Exception as exc:
                logger.warning("artifact.created broadcast failed: %s", exc)


# 函数说明：_safe_filename
# 用途：规范化文件名称，具体替换字符及兜底名称见实现。
# 参数：
#   filename：传给 `_UNSAFE_FILENAME_RE.sub` 的输入，类型 `str`。
# 返回：类型 `str`；返回 `cleaned[:240] or 'artifact'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_UNSAFE_FILENAME_RE.sub`。
def _safe_filename(filename: str) -> str:

    cleaned = _UNSAFE_FILENAME_RE.sub("_", filename).strip().strip(".")
    return cleaned[:240] or "artifact"


# 函数说明：_cleanup_temp_file
# 用途：清理文件，供交付物发布与存储使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`suppress` → `path.unlink`。
# 副作用与资源：
#   文件或资源访问：`path.unlink`。
def _cleanup_temp_file(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


# 函数说明：_cleanup_artifact_dir
# 用途：清理交付物，供交付物发布与存储使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.is_symlink` → `suppress` →
# `path.unlink` → `path.exists` → `path.iterdir` → `child.unlink`；另有 1 个调用点。
# 分支与异常：
#   `path.is_symlink()` 分支在完成前置处理后返回 `None`。
# 副作用与资源：
#   文件或资源访问：`path.unlink`、`child.unlink`、`path.rmdir`。
def _cleanup_artifact_dir(path: Path) -> None:

    if path.is_symlink():
        with suppress(OSError):
            path.unlink()
        return
    for child in path.iterdir() if path.exists() else ():
        with suppress(OSError):
            child.unlink()
    with suppress(OSError):
        path.rmdir()


# 函数说明：_delete_managed_artifact_dir
# 用途：删除交付物，供交付物发布与存储使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.is_symlink` → `path.unlink` →
# `path.exists` → `path.iterdir` → `child.is_dir` → `child.is_symlink`；另有 2 个调用点
# 。
# 分支与异常：
#   `path.is_symlink()` 分支在完成前置处理后返回 `None`。
#   当 `not path.exists()` 时，返回 `None`。
#   当 `child.is_dir() and (not child.is_symlink())` 时，抛出 `RuntimeError(…)`。
# 副作用与资源：
#   文件或资源访问：`path.unlink`、`child.unlink`、`path.rmdir`。
def _delete_managed_artifact_dir(path: Path) -> None:

    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            raise RuntimeError(f"unexpected nested artifact directory: {child}")
        child.unlink()
    path.rmdir()
