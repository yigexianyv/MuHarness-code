
"""MEA 状态的 SQLite 存储。

每次长任务一行 ``mea_runs``，每轮一行 ``mea_rounds``，权威要求的每个版本一行
``mea_requirements``（只追加）。键和常用过滤条件单独成列，完整内容存在 ``data`` JSON 里，
由 pydantic 模型校验；新增字段不需要改表。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH

from .models import MeaRound, MeaRun, MeaStatus, now_utc
from .requirements import Requirements

logger = logging.getLogger("muharness.mea.store")

# 每次成功写库后调用：(run, round)，其中一个可以为 None。用于向前端推送状态。
StoreListener = Callable[[MeaRun | None, MeaRound | None], Awaitable[None]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mea_runs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    status TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mea_runs_conversation
ON mea_runs(conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_mea_runs_status
ON mea_runs(status);

CREATE TABLE IF NOT EXISTS mea_requirements (
    mea_run_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    snapshot TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (mea_run_id, revision)
);

CREATE TABLE IF NOT EXISTS mea_rounds (
    mea_run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    kind TEXT NOT NULL,
    phase TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (mea_run_id, round_index)
);
"""


class SQLiteMeaStore:

    # 函数说明：SQLiteMeaStore.__init__
    # 用途：初始化 SQLiteMeaStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   database_path：SQLite 数据库路径，类型 `str | Path`；默认
    # `DEFAULT_DATABASE_PATH`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(database_path).expanduser().resolve` → `Path(database_path).expanduser` →
    # `Path`。
    # 副作用与资源：
    #   更新对象字段：`self.database_path`、`self._listeners`。
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self._listeners: list[StoreListener] = []

    # 函数说明：SQLiteMeaStore.add_listener
    # 用途：添加`listener`，供规划、执行、审计协作使用。
    # 参数：
    #   listener：传给 `self._listeners.append` 的输入，类型 `StoreListener`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def add_listener(self, listener: StoreListener) -> None:
        self._listeners.append(listener)

    # 函数说明：SQLiteMeaStore._notify
    # 用途：通知SQLiteMeaStore，供规划、执行、审计协作使用。
    # 参数：
    #   run：当前运行记录，类型 `MeaRun | None`。
    #   rnd：当前测试模型轮次，类型 `MeaRound | None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`listener` → `logger.exception`。
    # 分支与异常：
    #   捕获 `Exception` 后，执行异常处理调用 `logger.exception`。
    async def _notify(self, run: MeaRun | None, rnd: MeaRound | None) -> None:
        for listener in tuple(self._listeners):
            try:
                await listener(run, rnd)
            except Exception:  # 推送失败不能影响状态推进
                logger.exception("mea store listener failed")

    # 函数说明：SQLiteMeaStore.initialize
    # 用途：初始化SQLiteMeaStore，供规划、执行、审计协作使用。
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

    # ------------------------------------------------------------------ runs

    # 函数说明：SQLiteMeaStore.create_run
    # 用途：创建运行，供规划、执行、审计协作使用。
    # 参数：
    #   run：当前运行记录，类型 `MeaRun`。
    #   requirements：当前生效的任务要求，类型 `Requirements`。
    # 返回：类型 `MeaRun`；返回 `run`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `_upsert_run` → `_insert_requirements` → `database.commit` →
    # `self._notify`。
    # 分支与异常：
    #   当 `requirements.revision != run.requirements_revision` 时，抛出 `ValueError(…)`
    # 。
    async def create_run(self, run: MeaRun, requirements: Requirements) -> MeaRun:
        if requirements.revision != run.requirements_revision:
            raise ValueError("initial requirements revision must match the run")
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            await _upsert_run(database, run, insert=True)
            await _insert_requirements(database, run.id, requirements)
            await database.commit()
        await self._notify(run, None)
        return run

    # 函数说明：SQLiteMeaStore.get
    # 用途：获取SQLiteMeaStore，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `MeaRun | None`；返回
    # `MeaRun.model_validate_json(row['data']) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchone` → `MeaRun.model_validate_json`。
    # 副作用与资源：
    #   数据库操作：SELECT mea_runs；连接与事务边界以 with/提交语句为准。
    async def get(self, mea_id: str) -> MeaRun | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT data FROM mea_runs WHERE id = ?", (mea_id,)
            )
            row = await cursor.fetchone()
        return MeaRun.model_validate_json(row["data"]) if row is not None else None

    # 函数说明：SQLiteMeaStore.require
    # 用途：获取并校验必需的SQLiteMeaStore，供规划、执行、审计协作使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `MeaRun`；返回 `run`。
    # 分支与异常：
    #   当 `run is None` 时，抛出 `KeyError(f'长任务不存在：{mea_id}')`。
    async def require(self, mea_id: str) -> MeaRun:
        run = await self.get(mea_id)
        if run is None:
            raise KeyError(f"长任务不存在：{mea_id}")
        return run

    # 函数说明：SQLiteMeaStore.list_runs
    # 用途：列出运行集合，供规划、执行、审计协作使用。
    # 参数：
    #   status：目标状态，类型 `MeaStatus | None`；默认 `None`。
    #   conversation_id：目标会话标识，类型 `str | None`；默认 `None`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `50`。
    # 返回：类型 `tuple[MeaRun, ...]`；返回
    # `tuple((MeaRun.model_validate_json(row['data']) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MeaStatus` → `self._connect` →
    # `database.execute` → `cursor.fetchall` → `MeaRun.model_validate_json`。
    async def list_runs(
        self,
        *,
        status: MeaStatus | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
    ) -> tuple[MeaRun, ...]:
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(MeaStatus(status).value)
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self._connect() as database:
            cursor = await database.execute(
                f"SELECT data FROM mea_runs {where} ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            )
            rows = await cursor.fetchall()
        return tuple(MeaRun.model_validate_json(row["data"]) for row in rows)

    # 函数说明：SQLiteMeaStore.save_run
    # 用途：保存运行，供规划、执行、审计协作使用。
    # 参数：
    #   run：当前运行记录，类型 `MeaRun`。
    # 返回：类型 `MeaRun`；返回 `run`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`now_utc` → `self._connect` →
    # `_upsert_run` → `database.commit` → `self._notify`。
    async def save_run(self, run: MeaRun) -> MeaRun:
        run = run.model_copy(update={"updated_at": now_utc()})
        async with self._connect() as database:
            await _upsert_run(database, run)
            await database.commit()
        await self._notify(run, None)
        return run

    # 函数说明：SQLiteMeaStore.save_run_and_round
    # 用途：同一个事务里保存两者，例如终结决定：MEA → finalizing 且本轮 → applied。
    # 参数：
    #   run：当前运行记录，类型 `MeaRun`。
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `tuple[MeaRun, MeaRound]`；返回 `(run, rnd)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`now_utc` → `self._connect` →
    # `database.execute` → `_upsert_run` → `_upsert_round` → `database.commit`；另有 1
    # 个调用点。
    async def save_run_and_round(
        self,
        run: MeaRun,
        rnd: MeaRound,
    ) -> tuple[MeaRun, MeaRound]:
        """同一个事务里保存两者，例如终结决定：MEA → finalizing 且本轮 → applied。"""

        now = now_utc()
        run = run.model_copy(update={"updated_at": now})
        rnd = rnd.model_copy(update={"updated_at": now})
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            await _upsert_run(database, run)
            await _upsert_round(database, rnd)
            await database.commit()
        await self._notify(run, rnd)
        return run, rnd

    # ---------------------------------------------------------- requirements

    # 函数说明：SQLiteMeaStore.requirements
    # 用途：在规划、执行、审计协作中处理 `requirements`，通过 `self._connect` 完成首个内
    # 部处理步骤。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    #   revision：`revision`输入或配置值，类型 `int | None`；默认 `None`。
    # 返回：类型 `Requirements`；返回 `Requirements.from_json(row['snapshot'])`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchone` → `Requirements.from_json`。
    # 分支与异常：
    #   当 `row is None` 时，抛出
    # `KeyError(f"长任务 {mea_id} 没有要求版本 {revision or '(最新)'}")`。
    # 副作用与资源：
    #   数据库操作：SELECT mea_requirements；连接与事务边界以 with/提交语句为准。
    async def requirements(self, mea_id: str, revision: int | None = None) -> Requirements:
        async with self._connect() as database:
            if revision is None:
                cursor = await database.execute(
                    "SELECT snapshot FROM mea_requirements WHERE mea_run_id = ? "
                    "ORDER BY revision DESC LIMIT 1",
                    (mea_id,),
                )
            else:
                cursor = await database.execute(
                    "SELECT snapshot FROM mea_requirements "
                    "WHERE mea_run_id = ? AND revision = ?",
                    (mea_id, revision),
                )
            row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"长任务 {mea_id} 没有要求版本 {revision or '(最新)'}")
        return Requirements.from_json(row["snapshot"])

    # 函数说明：SQLiteMeaStore.save_requirements
    # 用途：追加一个新的要求版本，并在同一事务里更新 MEA 的当前版本号。
    # 参数：
    #   run：当前运行记录，类型 `MeaRun`。
    #   requirements：当前生效的任务要求，类型 `Requirements`。
    # 返回：类型 `MeaRun`；返回 `run`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`now_utc` → `self._connect` →
    # `database.execute` → `_insert_requirements` → `_upsert_run` → `database.commit`；
    # 另有 1 个调用点。
    # 分支与异常：
    #   当 `requirements.revision != run.requirements_revision + 1` 时，抛出
    # `ValueError(…)`。
    async def save_requirements(self, run: MeaRun, requirements: Requirements) -> MeaRun:
        """追加一个新的要求版本，并在同一事务里更新 MEA 的当前版本号。"""

        if requirements.revision != run.requirements_revision + 1:
            raise ValueError(
                f"requirements revision must be {run.requirements_revision + 1}, "
                f"got {requirements.revision}"
            )
        run = run.model_copy(
            update={"requirements_revision": requirements.revision, "updated_at": now_utc()}
        )
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            await _insert_requirements(database, run.id, requirements)
            await _upsert_run(database, run)
            await database.commit()
        await self._notify(run, None)
        return run

    # ---------------------------------------------------------------- rounds

    # 函数说明：SQLiteMeaStore.rounds
    # 用途：在规划、执行、审计协作中处理 `rounds`，通过 `self._connect` 完成首个内部处理
    # 步骤。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `tuple[MeaRound, ...]`；返回
    # `tuple((MeaRound.model_validate_json(row['data']) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `MeaRound.model_validate_json`。
    # 副作用与资源：
    #   数据库操作：SELECT mea_rounds；连接与事务边界以 with/提交语句为准。
    async def rounds(self, mea_id: str) -> tuple[MeaRound, ...]:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT data FROM mea_rounds WHERE mea_run_id = ? ORDER BY round_index",
                (mea_id,),
            )
            rows = await cursor.fetchall()
        return tuple(MeaRound.model_validate_json(row["data"]) for row in rows)

    # 函数说明：SQLiteMeaStore.last_round
    # 用途：在规划、执行、审计协作中处理 `last_round`，通过 `self._connect` 完成首个内部
    # 处理步骤。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：类型 `MeaRound | None`；返回
    # `MeaRound.model_validate_json(row['data']) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchone` → `MeaRound.model_validate_json`。
    # 副作用与资源：
    #   数据库操作：SELECT mea_rounds；连接与事务边界以 with/提交语句为准。
    async def last_round(self, mea_id: str) -> MeaRound | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT data FROM mea_rounds WHERE mea_run_id = ? "
                "ORDER BY round_index DESC LIMIT 1",
                (mea_id,),
            )
            row = await cursor.fetchone()
        return MeaRound.model_validate_json(row["data"]) if row is not None else None

    # 函数说明：SQLiteMeaStore.save_round
    # 用途：保存`round`，供规划、执行、审计协作使用。
    # 参数：
    #   rnd：当前测试模型轮次，类型 `MeaRound`。
    # 返回：类型 `MeaRound`；返回 `rnd`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`now_utc` → `self._connect` →
    # `_upsert_round` → `database.commit` → `self._notify`。
    async def save_round(self, rnd: MeaRound) -> MeaRound:
        rnd = rnd.model_copy(update={"updated_at": now_utc()})
        async with self._connect() as database:
            await _upsert_round(database, rnd)
            await database.commit()
        await self._notify(None, rnd)
        return rnd

    # 函数说明：SQLiteMeaStore.delete_for_conversation
    # 用途：删除会话下所有长任务及其轮次和要求版本，返回删除的长任务数。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    # 返回：类型 `int`；返回 `len(ids)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：SELECT mea_runs、DELETE mea_rounds、DELETE mea_requirements、DELETE
    # mea_runs；连接与事务边界以 with/提交语句为准。
    async def delete_for_conversation(self, conversation_id: str) -> int:
        """删除会话下所有长任务及其轮次和要求版本，返回删除的长任务数。"""

        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            cursor = await database.execute(
                "SELECT id FROM mea_runs WHERE conversation_id = ?", (conversation_id,)
            )
            ids = [row["id"] for row in await cursor.fetchall()]
            for mea_id in ids:
                await database.execute("DELETE FROM mea_rounds WHERE mea_run_id = ?", (mea_id,))
                await database.execute(
                    "DELETE FROM mea_requirements WHERE mea_run_id = ?", (mea_id,)
                )
            await database.execute(
                "DELETE FROM mea_runs WHERE conversation_id = ?", (conversation_id,)
            )
            await database.commit()
        return len(ids)

    # 函数说明：SQLiteMeaStore._connect
    # 用途：连接SQLiteMeaStore，供规划、执行、审计协作使用。
    # 返回：异步生成器，逐项产出 `database`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`aiosqlite.connect` →
    # `database.close`。
    # 副作用与资源：
    #   更新对象字段：`database.row_factory`。
    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        try:
            yield database
        finally:
            await database.close()


# 函数说明：_upsert_run
# 用途：新增或更新运行，供规划、执行、审计协作使用。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
#   run：当前运行记录，类型 `MeaRun`。
#   insert：`insert`输入或配置值，类型 `bool`；默认 `False`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`run.model_dump_json` →
# `run.created_at.isoformat` → `run.updated_at.isoformat` → `database.execute`。
# 分支与异常：
#   `insert` 分支在完成前置处理后返回 `None`。
# 副作用与资源：
#   数据库操作：INSERT mea_runs、INSERT mea_runs/SET；连接与事务边界以 with/提交语句为准
# 。
async def _upsert_run(
    database: aiosqlite.Connection,
    run: MeaRun,
    *,
    insert: bool = False,
) -> None:
    values = (
        run.id,
        run.task_id,
        run.conversation_id,
        run.status.value,
        run.model_dump_json(),
        run.created_at.isoformat(),
        run.updated_at.isoformat(),
    )
    if insert:
        await database.execute(
            "INSERT INTO mea_runs (id, task_id, conversation_id, status, data, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        return
    await database.execute(
        "INSERT INTO mea_runs (id, task_id, conversation_id, status, data, created_at, "
        "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET status = excluded.status, data = excluded.data, "
        "updated_at = excluded.updated_at",
        values,
    )


# 函数说明：_upsert_round
# 用途：新增或更新`round`，供规划、执行、审计协作使用。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
#   rnd：当前测试模型轮次，类型 `MeaRound`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute` →
# `rnd.model_dump_json` → `rnd.created_at.isoformat` → `rnd.updated_at.isoformat`。
# 副作用与资源：
#   数据库操作：INSERT mea_rounds/SET；连接与事务边界以 with/提交语句为准。
async def _upsert_round(database: aiosqlite.Connection, rnd: MeaRound) -> None:
    await database.execute(
        "INSERT INTO mea_rounds (mea_run_id, round_index, kind, phase, data, created_at, "
        "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(mea_run_id, round_index) DO UPDATE SET kind = excluded.kind, "
        "phase = excluded.phase, data = excluded.data, updated_at = excluded.updated_at",
        (
            rnd.mea_run_id,
            rnd.index,
            rnd.kind.value,
            rnd.phase.value,
            rnd.model_dump_json(),
            rnd.created_at.isoformat(),
            rnd.updated_at.isoformat(),
        ),
    )


# 函数说明：_insert_requirements
# 用途：插入`requirements`，供规划、执行、审计协作使用。
# 参数：
#   database：SQLite 数据库位置或连接，类型 `aiosqlite.Connection`。
#   mea_id：长任务协作记录标识，类型 `str`。
#   requirements：当前生效的任务要求，类型 `Requirements`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`database.execute` →
# `requirements.to_json` → `now_utc().isoformat` → `now_utc`。
# 副作用与资源：
#   数据库操作：INSERT mea_requirements；连接与事务边界以 with/提交语句为准。
async def _insert_requirements(
    database: aiosqlite.Connection,
    mea_id: str,
    requirements: Requirements,
) -> None:
    await database.execute(
        "INSERT INTO mea_requirements (mea_run_id, revision, snapshot, created_at) "
        "VALUES (?, ?, ?, ?)",
        (mea_id, requirements.revision, requirements.to_json(), now_utc().isoformat()),
    )


__all__ = ["SQLiteMeaStore", "StoreListener"]
