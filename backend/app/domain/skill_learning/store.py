
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import SkillCandidate, SkillCandidateStatus

logger = logging.getLogger("muharness.skill_learning.store")

WATERMARK_FILE_NAME = "skill_learning_watermark.json"
WATERMARK_VERSION = 1
MAX_CANDIDATE_FILE_BYTES = 500_000


class InflightBatch(BaseModel):

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    task_ids: tuple[str, ...]
    started_at: datetime
    attempt: int = Field(default=0, ge=0)
    last_error: str | None = None

    # 函数说明：InflightBatch.normalize_datetime
    # 用途：校验并规范化模型字段 'started_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime`。
    # 返回：类型 `datetime`；返回 `value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator("started_at")
    @classmethod
    def normalize_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("inflight started_at must include timezone information")
        return value.astimezone(UTC)


class MiningWatermark(BaseModel):

    model_config = ConfigDict(extra="forbid")

    version: int = WATERMARK_VERSION
    processed_task_ids: tuple[str, ...] = ()
    pending_task_ids: tuple[str, ...] = ()
    inflight: InflightBatch | None = None
    last_mining_at: datetime | None = None
    last_error: str | None = None

    # 函数说明：MiningWatermark.normalize_datetime
    # 用途：校验并规范化模型字段 'last_mining_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime | None`。
    # 返回：类型 `datetime | None`；按分支返回 `None`；`value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出 `ValueError(…)`
    # 。
    @field_validator("last_mining_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("watermark datetimes must include timezone information")
        return value.astimezone(UTC)

    # 函数说明：MiningWatermark.model_dump_json
    # 用途：返回 `super().model_dump_json()`，提供 MiningWatermark 的派生值。
    # 返回：类型 `str`；返回 `super().model_dump_json()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().model_dump_json` →
    # `super`。
    def model_dump_json(self) -> str:
        return super().model_dump_json()

    # 函数说明：MiningWatermark.model_validate_json
    # 用途：校验JSON 数据，供技能候选提炼与审核使用。
    # 参数：
    #   json_data：传给 `super().model_validate_json` 的输入，类型 `str`。
    # 返回：类型 `MiningWatermark`；返回 `super().model_validate_json(json_data)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().model_validate_json` →
    # `super`。
    @classmethod
    def model_validate_json(cls, json_data: str) -> MiningWatermark:
        return super().model_validate_json(json_data)


class SkillCandidateStore:

    # 函数说明：SkillCandidateStore.__init__
    # 用途：初始化 SkillCandidateStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   data_dir：相关数据的根目录或保存目录，类型 `str | Path`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(data_dir).expanduser().resolve` → `Path(data_dir).expanduser` → `Path`。
    # 副作用与资源：
    #   更新对象字段：`self.data_dir`、`self.candidates_dir`、`self.watermark_path`、
    # `self._locks`。
    def __init__(
        self,
        data_dir: str | Path,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.candidates_dir = self.data_dir / "candidates"
        self.watermark_path = self.data_dir / WATERMARK_FILE_NAME
        self._locks: dict[str, asyncio.Lock] = {}

    # 函数说明：SkillCandidateStore.initialize
    # 用途：初始化SkillCandidateStore，供技能候选提炼与审核使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def initialize(self) -> None:

        await asyncio.to_thread(self.candidates_dir.mkdir, parents=True, exist_ok=True)


    # 函数说明：SkillCandidateStore._candidate_path
    # 用途：返回 `self.candidates_dir / f'{candidate_id}.json'`，提供
    # SkillCandidateStore 的派生值。
    # 参数：
    #   candidate_id：待审核技能候选标识，类型 `str`。
    # 返回：类型 `Path`；返回 `self.candidates_dir / f'{candidate_id}.json'`。
    def _candidate_path(self, candidate_id: str) -> Path:
        return self.candidates_dir / f"{candidate_id}.json"

    # 函数说明：SkillCandidateStore.create
    # 用途：创建SkillCandidateStore，供技能候选提炼与审核使用。
    # 参数：
    #   candidate：候选记录，类型 `SkillCandidate`。
    # 返回：类型 `SkillCandidate`；返回 `candidate`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock_for` →
    # `self._candidate_path` → `asyncio.to_thread`。
    # 资源/并发边界：`self._lock_for(candidate.id)`，上下文退出时执行相应清理。
    # 分支与异常：
    #   当 `await asyncio.to_thread(path.is_file)` 时，抛出
    # `ValueError(f'candidate already exists: {candidate.id}')`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def create(self, candidate: SkillCandidate) -> SkillCandidate:

        async with self._lock_for(candidate.id):
            path = self._candidate_path(candidate.id)
            if await asyncio.to_thread(path.is_file):
                raise ValueError(f"candidate already exists: {candidate.id}")
            await asyncio.to_thread(
                _write_json,
                path,
                candidate.model_dump(mode="json"),
            )
        return candidate

    # 函数说明：SkillCandidateStore.get
    # 用途：获取SkillCandidateStore，供技能候选提炼与审核使用。
    # 参数：
    #   candidate_id：待审核技能候选标识，类型 `str`。
    # 返回：类型 `SkillCandidate | None`；按分支返回 `None`；
    # `await asyncio.to_thread(_read_candidate, path)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`candidate_id.strip().lower` →
    # `self._candidate_path` → `asyncio.to_thread`。
    # 分支与异常：
    #   当 `not normalized` 时，返回 `None`。
    #   当 `not await asyncio.to_thread(path.is_file)` 时，返回 `None`。
    #   当 `await asyncio.to_thread(path.is_symlink)` 时，返回 `None`。
    #   捕获 `(ValueError, OSError, UnicodeError)` 后，返回 `None`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def get(self, candidate_id: str) -> SkillCandidate | None:

        normalized = candidate_id.strip().lower()
        if not normalized:
            return None
        path = self._candidate_path(normalized)
        if not await asyncio.to_thread(path.is_file):
            return None
        if await asyncio.to_thread(path.is_symlink):
            return None
        try:
            return await asyncio.to_thread(_read_candidate, path)
        except (ValueError, OSError, UnicodeError):
            return None

    # 函数说明：SkillCandidateStore.list
    # 用途：列出SkillCandidateStore，供技能候选提炼与审核使用。
    # 参数：
    #   status：目标状态，类型 `SkillCandidateStatus | None`；默认 `None`。
    # 返回：类型 `tuple[SkillCandidate, ...]`；返回 `tuple(candidates)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread` →
    # `candidates.sort`。
    # 分支与异常：
    #   当 `candidate is None` 时，跳过当前循环项。
    #   当 `status is not None and candidate.status is not status` 时，跳过当前循环项。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def list(
        self,
        *,
        status: SkillCandidateStatus | None = None,
    ) -> tuple[SkillCandidate, ...]:

        candidates: list[SkillCandidate] = []
        for path in await asyncio.to_thread(self._list_candidate_files):
            candidate = await self.get(path.stem)
            if candidate is None:
                continue
            if status is not None and candidate.status is not status:
                continue
            candidates.append(candidate)
        candidates.sort(key=lambda item: item.created_at, reverse=True)
        return tuple(candidates)

    # 函数说明：SkillCandidateStore._list_candidate_files
    # 用途：列出候选文件集合，供技能候选提炼与审核使用。
    # 返回：类型 `list[Path]`；按分支返回 `[]`；
    # `sorted(self.candidates_dir.glob('*.json'))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.candidates_dir.is_dir` →
    # `self.candidates_dir.glob`。
    # 分支与异常：
    #   当 `not self.candidates_dir.is_dir()` 时，返回 `[]`。
    def _list_candidate_files(self) -> list[Path]:
        if not self.candidates_dir.is_dir():
            return []
        return sorted(self.candidates_dir.glob("*.json"))

    # 函数说明：SkillCandidateStore.update
    # 用途：更新SkillCandidateStore，供技能候选提炼与审核使用。
    # 参数：
    #   candidate：候选记录，类型 `SkillCandidate`。
    # 返回：类型 `SkillCandidate`；返回 `candidate`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock_for` →
    # `asyncio.to_thread` → `self._candidate_path`。
    # 资源/并发边界：`self._lock_for(candidate.id)`，上下文退出时执行相应清理。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def update(self, candidate: SkillCandidate) -> SkillCandidate:

        async with self._lock_for(candidate.id):
            await asyncio.to_thread(
                _write_json,
                self._candidate_path(candidate.id),
                candidate.model_dump(mode="json"),
            )
        return candidate

    # 函数说明：SkillCandidateStore.find_duplicate_source
    # 用途：在技能候选提炼与审核中处理 `find_duplicate_source`，通过 `self.list` 完成首
    # 个内部处理步骤。
    # 参数：
    #   source_task_ids：传给 `set` 的输入，类型 `tuple[str, ...]`。
    # 返回：类型 `SkillCandidate | None`；按分支返回 `candidate`；`None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.list`。
    # 分支与异常：
    #   当 `set(candidate.source_task_ids) == source_set` 时，返回 `candidate`。
    async def find_duplicate_source(
        self,
        source_task_ids: tuple[str, ...],
    ) -> SkillCandidate | None:

        source_set = set(source_task_ids)
        for candidate in await self.list():
            if set(candidate.source_task_ids) == source_set:
                return candidate
        return None


    # 函数说明：SkillCandidateStore.load_watermark
    # 用途：加载`watermark`，供技能候选提炼与审核使用。
    # 返回：类型 `MiningWatermark`；按分支返回 `MiningWatermark()`；
    # `MiningWatermark.model_validate_json(raw)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread` →
    # `MiningWatermark` → `MiningWatermark.model_validate_json` → `logger.warning`。
    # 分支与异常：
    #   当 `not await asyncio.to_thread(self.watermark_path.is_file)` 时，返回
    # `MiningWatermark()`。
    #   捕获 `(ValueError, OSError, UnicodeError)` 后，返回 `MiningWatermark()`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def load_watermark(self) -> MiningWatermark:

        if not await asyncio.to_thread(self.watermark_path.is_file):
            return MiningWatermark()
        try:
            raw = await asyncio.to_thread(
                self.watermark_path.read_text,
                encoding="utf-8",
            )
            return MiningWatermark.model_validate_json(raw)
        except (ValueError, OSError, UnicodeError):
            logger.warning("skill learning watermark is unreadable; resetting")
            return MiningWatermark()

    # 函数说明：SkillCandidateStore.save_watermark
    # 用途：保存`watermark`，供技能候选提炼与审核使用。
    # 参数：
    #   watermark：`watermark`输入或配置值，类型 `MiningWatermark`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._lock_for` →
    # `asyncio.to_thread`。
    # 资源/并发边界：`self._lock_for('watermark')`，上下文退出时执行相应清理。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def save_watermark(self, watermark: MiningWatermark) -> None:

        async with self._lock_for("watermark"):
            await asyncio.to_thread(
                _write_json,
                self.watermark_path,
                watermark.model_dump(mode="json"),
            )


    # 函数说明：SkillCandidateStore._lock_for
    # 用途：获取锁`for`，供技能候选提炼与审核使用。
    # 参数：
    #   key：字段名或查询键，类型 `str`。
    # 返回：类型 `asyncio.Lock`；返回 `self._locks[key]`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Lock`。
    def _lock_for(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]


# 函数说明：_read_candidate
# 用途：读取候选，供技能候选提炼与审核使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
# 返回：类型 `SkillCandidate`；返回
# `SkillCandidate.model_validate_json(path.read_text(encoding='utf-8'))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.stat` →
# `SkillCandidate.model_validate_json` → `path.read_text`。
# 分支与异常：
#   当 `path.stat().st_size > MAX_CANDIDATE_FILE_BYTES` 时，抛出
# `ValueError('candidate file too large')`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
def _read_candidate(path: Path) -> SkillCandidate:
    if path.stat().st_size > MAX_CANDIDATE_FILE_BYTES:
        raise ValueError("candidate file too large")
    return SkillCandidate.model_validate_json(path.read_text(encoding="utf-8"))


# 函数说明：_write_json
# 用途：写入JSON 数据，供技能候选提炼与审核使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
#   payload：传输或持久化载荷，类型 `dict[str, Any]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.parent.mkdir` →
# `path.with_suffix` → `datetime.now(UTC).strftime` → `datetime.now` →
# `temporary_path.write_text` → `json.dumps`；另有 1 个调用点。
# 副作用与资源：
#   文件或资源访问：`path.parent.mkdir`、`temporary_path.write_text`。
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(
        f".tmp.{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    )
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


__all__ = [
    "InflightBatch",
    "MiningWatermark",
    "SkillCandidateStore",
    "WATERMARK_VERSION",
]
