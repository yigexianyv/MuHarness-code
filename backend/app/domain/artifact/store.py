
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH

from .models import Artifact, ArtifactKind

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    description TEXT,
    filename TEXT,
    mime_type TEXT,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT,
    run_id TEXT,
    conversation_id TEXT,
    task_id TEXT,
    source_url TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_created
ON artifacts(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_id
ON artifacts(run_id);

CREATE INDEX IF NOT EXISTS idx_artifacts_conversation_id
ON artifacts(conversation_id);
"""


class SQLiteArtifactStore:

    # 函数说明：SQLiteArtifactStore.__init__
    # 用途：初始化 SQLiteArtifactStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   database_path：SQLite 数据库路径，类型 `str | Path`；默认
    # `DEFAULT_DATABASE_PATH`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(database_path).expanduser().resolve` → `Path(database_path).expanduser` →
    # `Path`。
    # 副作用与资源：
    #   更新对象字段：`self.database_path`。
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    # 函数说明：SQLiteArtifactStore.initialize
    # 用途：初始化SQLiteArtifactStore，供交付物发布与存储使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.database_path.parent.mkdir`
    # → `self._connect` → `database.executescript` → `database.commit`。
    # 副作用与资源：
    #   文件或资源访问：`self.database_path.parent.mkdir`。
    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    # 函数说明：SQLiteArtifactStore.create
    # 用途：创建SQLiteArtifactStore，供交付物发布与存储使用。
    # 参数：
    #   artifact：交付物输入或配置值，类型 `Artifact`。
    # 返回：类型 `Artifact`；返回 `artifact`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `artifact.created_at.isoformat` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：INSERT artifacts；连接与事务边界以 with/提交语句为准。
    async def create(self, artifact: Artifact) -> Artifact:

        async with self._connect() as database:
            await database.execute(
                """
                INSERT INTO artifacts (
                    id, kind, title, description, filename, mime_type,
                    size_bytes, sha256, run_id, conversation_id, task_id,
                    source_url, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.id,
                    artifact.kind.value,
                    artifact.title,
                    artifact.description,
                    artifact.filename,
                    artifact.mime_type,
                    artifact.size_bytes,
                    artifact.sha256,
                    artifact.run_id,
                    artifact.conversation_id,
                    artifact.task_id,
                    artifact.source_url,
                    artifact.created_at.isoformat(),
                ),
            )
            await database.commit()
        return artifact

    # 函数说明：SQLiteArtifactStore.get
    # 用途：获取SQLiteArtifactStore，供交付物发布与存储使用。
    # 参数：
    #   artifact_id：交付物标识，类型 `str`。
    # 返回：类型 `Artifact | None`；按分支返回 `None`；`_row_to_artifact(row)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchone` → `_row_to_artifact`。
    # 分支与异常：
    #   当 `row is None` 时，返回 `None`。
    # 副作用与资源：
    #   数据库操作：SELECT artifacts；连接与事务边界以 with/提交语句为准。
    async def get(self, artifact_id: str) -> Artifact | None:
        async with self._connect() as database:
            async with database.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_artifact(row)

    # 函数说明：SQLiteArtifactStore.list
    # 用途：列出SQLiteArtifactStore，供交付物发布与存储使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str | None`；默认 `None`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   task_id：目标任务标识，类型 `str | None`；默认 `None`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `50`。
    # 返回：类型 `tuple[Artifact, ...]`；返回
    # `tuple((_row_to_artifact(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `_row_to_artifact`。
    async def list(
        self,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        limit: int = 50,
    ) -> tuple[Artifact, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        async with self._connect() as database:
            async with database.execute(
                f"""
                SELECT * FROM artifacts
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ) as cursor:
                rows = await cursor.fetchall()
        return tuple(_row_to_artifact(row) for row in rows)

    # 函数说明：SQLiteArtifactStore.list_related_to_conversation
    # 用途：列出会话，供交付物发布与存储使用。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`；默认 `()`。
    #   task_ids：传给 `parameters.extend` 的输入，类型 `tuple[str, ...]`；默认 `()`。
    # 返回：类型 `tuple[Artifact, ...]`；返回
    # `tuple((_row_to_artifact(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `_row_to_artifact`。
    # 分支与异常：
    #   当 `not normalized` 时，抛出 `ValueError('conversation_id cannot be empty')`。
    async def list_related_to_conversation(
        self,
        conversation_id: str,
        *,
        run_ids: tuple[str, ...] = (),
        task_ids: tuple[str, ...] = (),
    ) -> tuple[Artifact, ...]:

        normalized = conversation_id.strip()
        if not normalized:
            raise ValueError("conversation_id cannot be empty")
        clauses = ["conversation_id = ?"]
        parameters: list[object] = [normalized]
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            clauses.append(f"run_id IN ({placeholders})")
            parameters.extend(run_ids)
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            clauses.append(f"task_id IN ({placeholders})")
            parameters.extend(task_ids)
        query = (
            "SELECT * FROM artifacts WHERE "
            + " OR ".join(f"({clause})" for clause in clauses)
            + " ORDER BY created_at ASC"
        )
        async with self._connect() as database:
            cursor = await database.execute(query, tuple(parameters))
            rows = await cursor.fetchall()
        return tuple(_row_to_artifact(row) for row in rows)

    # 函数说明：SQLiteArtifactStore.delete_many
    # 用途：删除`many`，供交付物发布与存储使用。
    # 参数：
    #   artifact_ids：传给 `database.execute` 的输入，类型 `tuple[str, ...]`。
    # 返回：类型 `int`；按分支返回 `0`；`max(cursor.rowcount, 0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `database.commit`。
    # 分支与异常：
    #   当 `not artifact_ids` 时，返回 `0`。
    async def delete_many(self, artifact_ids: tuple[str, ...]) -> int:

        if not artifact_ids:
            return 0
        placeholders = ",".join("?" for _ in artifact_ids)
        async with self._connect() as database:
            cursor = await database.execute(
                f"DELETE FROM artifacts WHERE id IN ({placeholders})",
                artifact_ids,
            )
            await database.commit()
        return max(cursor.rowcount, 0)

    # 函数说明：SQLiteArtifactStore._connect
    # 用途：连接SQLiteArtifactStore，供交付物发布与存储使用。
    # 返回：异步生成器，逐项产出 `connection`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`aiosqlite.connect` →
    # `connection.close`。
    # 副作用与资源：
    #   更新对象字段：`connection.row_factory`。
    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        connection = await aiosqlite.connect(self.database_path)
        connection.row_factory = aiosqlite.Row
        try:
            yield connection
        finally:
            await connection.close()


# 函数说明：_row_to_artifact
# 用途：把数据库查询行转换为交付物记录。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `created_at`、`id`、
# `kind`、`title`、`description`、`filename`、`mime_type`、`size_bytes`、`sha256`、
# `run_id`、`conversation_id`、`task_id`、`source_url`。
# 返回：类型 `Artifact`；返回 `Artifact(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `datetime.fromisoformat(created_at).astimezone` → `datetime.fromisoformat` →
# `datetime.now` → `Artifact` → `ArtifactKind`。
def _row_to_artifact(row: aiosqlite.Row) -> Artifact:
    created_at = row["created_at"]
    created = (
        datetime.fromisoformat(created_at).astimezone(UTC)
        if created_at
        else datetime.now(UTC)
    )
    return Artifact(
        id=row["id"],
        kind=ArtifactKind(row["kind"]),
        title=row["title"] or "",
        description=row["description"],
        filename=row["filename"],
        mime_type=row["mime_type"],
        size_bytes=row["size_bytes"] or 0,
        sha256=row["sha256"],
        run_id=row["run_id"],
        conversation_id=row["conversation_id"],
        task_id=row["task_id"],
        source_url=row["source_url"],
        created_at=created,
    )
