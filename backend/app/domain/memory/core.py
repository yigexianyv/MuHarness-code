
from __future__ import annotations

import asyncio
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from app.runtime.context.tokens import default_token_estimator

DEFAULT_MAX_CORE_TOKENS = 2_000
_CORE_FORMAT = "muharness-core-v1"
_CORE_HEADING = "# Core Memory"
_MANAGED_HEADING = "## Managed Core Entries"
_FRONT_MATTER_RE = re.compile(
    r"\A(?P<preamble>\ufeff?(?:[ \t]*\r?\n)*[ \t]*)"
    r"(?P<opening>---[ \t]*\r?\n)"
    r"(?P<metadata>.*?)"
    r"(?P<closing>^---[ \t]*(?:\r?\n|$))",
    re.DOTALL | re.MULTILINE,
)
_CORE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_MAX_CORE_VALUE_CHARS = 1_000
_MAX_CORE_REASON_CHARS = 1_000
_MAX_SOURCE_STATEMENT_CHARS = 2_000


class CoreMemoryEntry(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    value: str
    reason: str
    source_statement: str
    updated_at: datetime

    # 函数说明：CoreMemoryEntry.normalize_key
    # 用途：校验并规范化模型字段 'key'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalize_core_key(value)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`normalize_core_key`。
    @field_validator("key", mode="before")
    @classmethod
    def normalize_key(cls, value: object) -> str:
        return normalize_core_key(value)

    # 函数说明：CoreMemoryEntry.normalize_text
    # 用途：校验并规范化模型字段 'value'、'reason'、'source_statement'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('core memory value and reason must be strings')`。
    #   当 `not normalized` 时，抛出
    # `ValueError('core memory value and reason cannot be empty')`。
    @field_validator("value", "reason", "source_statement", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("core memory value and reason must be strings")
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("core memory value and reason cannot be empty")
        return normalized

    # 函数说明：CoreMemoryEntry.validate_value_length
    # 用途：校验并规范化模型字段 'value'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `value`。
    # 分支与异常：
    #   当 `len(value) > _MAX_CORE_VALUE_CHARS` 时，抛出 `ValueError(…)`。
    @field_validator("value")
    @classmethod
    def validate_value_length(cls, value: str) -> str:
        if len(value) > _MAX_CORE_VALUE_CHARS:
            raise ValueError(
                f"core memory value exceeds {_MAX_CORE_VALUE_CHARS} characters"
            )
        return value

    # 函数说明：CoreMemoryEntry.validate_reason_length
    # 用途：校验并规范化模型字段 'reason'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `value`。
    # 分支与异常：
    #   当 `len(value) > _MAX_CORE_REASON_CHARS` 时，抛出 `ValueError(…)`。
    @field_validator("reason")
    @classmethod
    def validate_reason_length(cls, value: str) -> str:
        if len(value) > _MAX_CORE_REASON_CHARS:
            raise ValueError(
                f"core memory reason exceeds {_MAX_CORE_REASON_CHARS} characters"
            )
        return value

    # 函数说明：CoreMemoryEntry.validate_source_statement_length
    # 用途：校验并规范化模型字段 'source_statement'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `value`。
    # 分支与异常：
    #   当 `len(value) > _MAX_SOURCE_STATEMENT_CHARS` 时，抛出 `ValueError(…)`。
    @field_validator("source_statement")
    @classmethod
    def validate_source_statement_length(cls, value: str) -> str:
        if len(value) > _MAX_SOURCE_STATEMENT_CHARS:
            raise ValueError(
                "core memory source statement exceeds "
                f"{_MAX_SOURCE_STATEMENT_CHARS} characters"
            )
        return value

    # 函数说明：CoreMemoryEntry.normalize_time
    # 用途：校验并规范化模型字段 'updated_at'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `datetime`。
    # 返回：类型 `datetime`；返回 `value.astimezone(UTC)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`value.utcoffset` →
    # `value.astimezone`。
    # 分支与异常：
    #   当 `value.tzinfo is None or value.utcoffset() is None` 时，抛出
    # `ValueError('core memory timestamp must include timezone')`。
    @field_validator("updated_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("core memory timestamp must include timezone")
        return value.astimezone(UTC)


class CoreMemoryManager:

    # 函数说明：CoreMemoryManager.__init__
    # 用途：初始化 CoreMemoryManager；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   memory_dir：记忆文件目录，类型 `str | Path`。
    #   max_tokens：Token 数量或 Token 预算，类型 `int`；默认 `DEFAULT_MAX_CORE_TOKENS`
    # 。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Path`。
    # 分支与异常：
    #   当 `max_tokens <= 0` 时，抛出
    # `ValueError('max_tokens must be greater than zero')`。
    # 副作用与资源：
    #   更新对象字段：`self.path`、`self.max_tokens`。
    def __init__(
        self,
        memory_dir: str | Path,
        *,
        max_tokens: int = DEFAULT_MAX_CORE_TOKENS,
    ) -> None:
        self.path = Path(memory_dir) / "CORE.md"
        self.max_tokens = max_tokens
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

    # 函数说明：CoreMemoryManager.initialize
    # 用途：初始化CoreMemoryManager，供长期记忆管理与检索使用。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread` →
    # `_parse_document`。
    # 分支与异常：
    #   当 `not await asyncio.to_thread(self.path.is_file)` 时，返回 `None`。
    #   当 `await asyncio.to_thread(self.path.is_symlink)` 时，抛出
    # `ValueError('CORE.md cannot be a symbolic link')`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def initialize(self) -> None:
        await asyncio.to_thread(self.path.parent.mkdir, parents=True, exist_ok=True)
        if not await asyncio.to_thread(self.path.is_file):
            return
        if await asyncio.to_thread(self.path.is_symlink):
            raise ValueError("CORE.md cannot be a symbolic link")
        content = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        _parse_document(content)

    # 函数说明：CoreMemoryManager.load
    # 用途：加载CoreMemoryManager，供长期记忆管理与检索使用。
    # 返回：类型 `str`；按分支返回 `''`；`visible`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread` →
    # `_parse_document` → `self._estimate_tokens`。
    # 分支与异常：
    #   当 `not await asyncio.to_thread(self.path.is_file)` 时，返回 `''`。
    #   当 `await asyncio.to_thread(self.path.is_symlink)` 时，抛出
    # `ValueError('CORE.md cannot be a symbolic link')`。
    #   当 `estimated > self.max_tokens` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def load(self) -> str:

        if not await asyncio.to_thread(self.path.is_file):
            return ""
        if await asyncio.to_thread(self.path.is_symlink):
            raise ValueError("CORE.md cannot be a symbolic link")
        content = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        _, visible = _parse_document(content)
        estimated = self._estimate_tokens(visible)
        if estimated > self.max_tokens:
            raise ValueError(
                f"core memory exceeds token limit: {estimated} > {self.max_tokens}"
            )
        return visible

    # 函数说明：CoreMemoryManager.update
    # 用途：更新CoreMemoryManager，供长期记忆管理与检索使用。
    # 参数：
    #   content：内容正文，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._estimate_tokens` →
    # `asyncio.to_thread`。
    # 分支与异常：
    #   当 `not normalized` 时，抛出 `ValueError('core memory content cannot be empty')`
    # 。
    #   当 `estimated > self.max_tokens` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def update(self, content: str) -> None:

        normalized = content.strip()
        if not normalized:
            raise ValueError("core memory content cannot be empty")
        estimated = self._estimate_tokens(normalized)
        if estimated > self.max_tokens:
            raise ValueError(
                f"core memory exceeds token limit: {estimated} > {self.max_tokens}"
            )
        await asyncio.to_thread(self._write_atomic, normalized + "\n")

    # 函数说明：CoreMemoryManager.upsert
    # 用途：新增或更新CoreMemoryManager，供长期记忆管理与检索使用。
    # 参数：
    #   key：字段名或查询键，类型 `str`。
    #   value：待校验、规范化或转换的值，类型 `str`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    #   source_statement：`source_statement`输入或配置值，类型 `str`。
    # 返回：类型 `tuple[CoreMemoryEntry, bool]`；返回 `(entry, created)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now` → `CoreMemoryEntry`
    #  → `asyncio.to_thread` → `_parse_document` → `_manual_content` →
    # `_render_document`；另有 1 个调用点。
    # 分支与异常：
    #   当 `await asyncio.to_thread(self.path.is_symlink)` 时，抛出
    # `ValueError('CORE.md cannot be a symbolic link')`。
    #   当 `estimated > self.max_tokens` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def upsert(
        self,
        *,
        key: str,
        value: str,
        reason: str,
        source_statement: str,
    ) -> tuple[CoreMemoryEntry, bool]:

        now = datetime.now(UTC)
        entry = CoreMemoryEntry(
            key=key,
            value=value,
            reason=reason,
            source_statement=source_statement,
            updated_at=now,
        )
        raw = ""
        if await asyncio.to_thread(self.path.is_file):
            if await asyncio.to_thread(self.path.is_symlink):
                raise ValueError("CORE.md cannot be a symbolic link")
            raw = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        entries, visible = _parse_document(raw)
        created = entry.key not in entries
        entries[entry.key] = entry
        manual_content = _manual_content(visible)
        rendered, visible_rendered = _render_document(
            entries, manual_content=manual_content
        )
        estimated = self._estimate_tokens(visible_rendered)
        if estimated > self.max_tokens:
            raise ValueError(
                f"core memory exceeds token limit: {estimated} > {self.max_tokens}"
            )
        await asyncio.to_thread(self._write_atomic, rendered)
        return entry, created

    # 函数说明：CoreMemoryManager.remove
    # 用途：移除CoreMemoryManager，供长期记忆管理与检索使用。
    # 参数：
    #   key：字段名或查询键，类型 `str`。
    # 返回：类型 `CoreMemoryEntry`；返回 `removed`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`normalize_core_key` →
    # `asyncio.to_thread` → `_parse_document` → `entries.pop` → `_render_document` →
    # `_manual_content`；另有 1 个调用点。
    # 分支与异常：
    #   当 `not await asyncio.to_thread(self.path.is_file)` 时，抛出
    # `KeyError(f'core memory key not found: {normalized_key}')`。
    #   当 `await asyncio.to_thread(self.path.is_symlink)` 时，抛出
    # `ValueError('CORE.md cannot be a symbolic link')`。
    #   当 `removed is None` 时，抛出
    # `KeyError(f'core memory key not found: {normalized_key}')`。
    #   当 `estimated > self.max_tokens` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def remove(self, key: str) -> CoreMemoryEntry:

        normalized_key = normalize_core_key(key)
        if not await asyncio.to_thread(self.path.is_file):
            raise KeyError(f"core memory key not found: {normalized_key}")
        if await asyncio.to_thread(self.path.is_symlink):
            raise ValueError("CORE.md cannot be a symbolic link")
        raw = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        entries, visible = _parse_document(raw)
        removed = entries.pop(normalized_key, None)
        if removed is None:
            raise KeyError(f"core memory key not found: {normalized_key}")
        rendered, visible_rendered = _render_document(
            entries,
            manual_content=_manual_content(visible),
        )
        estimated = self._estimate_tokens(visible_rendered)
        if estimated > self.max_tokens:
            raise ValueError(
                f"core memory exceeds token limit: {estimated} > {self.max_tokens}"
            )
        await asyncio.to_thread(self._write_atomic, rendered)
        return removed

    # 函数说明：CoreMemoryManager._estimate_tokens
    # 用途：估算Token 用量，供长期记忆管理与检索使用。
    # 参数：
    #   content：内容正文，类型 `str`。
    # 返回：类型 `int`；按分支返回 `estimator.estimate_text(content)`；
    # `len(content) // 2`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`default_token_estimator` →
    # `estimator.estimate_text`。
    # 分支与异常：
    #   捕获 `Exception` 后，返回 `len(content) // 2`。
    def _estimate_tokens(self, content: str) -> int:
        try:
            estimator = default_token_estimator()
            return estimator.estimate_text(content)
        except Exception:
            return len(content) // 2

    # 函数说明：CoreMemoryManager._write_atomic
    # 用途：写入`atomic`，供长期记忆管理与检索使用。
    # 参数：
    #   content：内容正文，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.path.parent.mkdir` →
    # `self.path.with_name` → `os.getpid` → `temporary.write_text` → `os.replace` →
    # `temporary.unlink`。
    # 副作用与资源：
    #   文件或资源访问：`self.path.parent.mkdir`、`temporary.write_text`、`os.replace`、
    # `temporary.unlink`。
    def _write_atomic(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


# 函数说明：_parse_document
# 用途：解析`document`，供长期记忆管理与检索使用。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `tuple[dict[str, CoreMemoryEntry], str]`；按分支返回 `({}, text.strip())`；
# `(_parse_entries(metadata), visible)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_split_front_matter` →
# `_parse_entries`。
# 分支与异常：
#   当 `parsed is None` 时，返回 `({}, text.strip())`。
#   当 `'format' not in metadata` 时，返回 `({}, text.strip())`。
#   当 `metadata['format'] != _CORE_FORMAT` 时，抛出 `ValueError(…)`。
def _parse_document(text: str) -> tuple[dict[str, CoreMemoryEntry], str]:

    parsed = _split_front_matter(text)
    if parsed is None:
        return {}, text.strip()
    metadata, visible = parsed
    if "format" not in metadata:
        return {}, text.strip()
    if metadata["format"] != _CORE_FORMAT:
        raise ValueError(f"unsupported CORE.md format; expected {_CORE_FORMAT}")
    return _parse_entries(metadata), visible


# 函数说明：_parse_entries
# 用途：解析条目，供长期记忆管理与检索使用。
# 参数：
#   metadata：关联元数据，类型 `dict[object, object]`；读取键 `entries`。
# 返回：类型 `dict[str, CoreMemoryEntry]`；返回 `{entry.key: entry for entry in (
# CoreMemoryEntry.model_validate(raw_entry) for raw_entry…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`CoreMemoryEntry.model_validate`。
# 分支与异常：
#   当 `not isinstance(raw_entries, list)` 时，抛出
# `ValueError('CORE.md entries metadata must be a list')`。
def _parse_entries(metadata: dict[object, object]) -> dict[str, CoreMemoryEntry]:

    raw_entries = metadata.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ValueError("CORE.md entries metadata must be a list")
    return {
        entry.key: entry
        for entry in (
            CoreMemoryEntry.model_validate(raw_entry) for raw_entry in raw_entries
        )
    }


# 函数说明：_split_front_matter
# 用途：拆分`front_matter`，供长期记忆管理与检索使用。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `tuple[dict[object, object], str] | None`；按分支返回 `None`；
# `(metadata, text[matched.end():].strip())`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_FRONT_MATTER_RE.match` →
# `yaml.safe_load` → `matched.group` → `matched.end`。
# 分支与异常：
#   当 `matched is None` 时，返回 `None`。
#   当 `not isinstance(metadata, dict)` 时，返回 `None`。
def _split_front_matter(
    text: str,
) -> tuple[dict[object, object], str] | None:

    matched = _FRONT_MATTER_RE.match(text)
    if matched is None:
        return None
    metadata = yaml.safe_load(matched.group("metadata"))
    if not isinstance(metadata, dict):
        return None
    return metadata, text[matched.end() :].strip()


# 函数说明：normalize_core_key
# 用途：规范化键，供长期记忆管理与检索使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `object`。
# 返回：类型 `str`；返回 `normalized`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().lower` →
# `_CORE_KEY_RE.fullmatch`。
# 分支与异常：
#   当 `not isinstance(value, str)` 时，抛出
# `TypeError('core memory key must be a string')`。
#   当 `not _CORE_KEY_RE.fullmatch(normalized)` 时，抛出 `ValueError(…)`。
def normalize_core_key(value: object) -> str:

    if not isinstance(value, str):
        raise TypeError("core memory key must be a string")
    normalized = value.strip().lower()
    if not _CORE_KEY_RE.fullmatch(normalized):
        raise ValueError("core memory key must be a lowercase dotted identifier")
    return normalized


# 函数说明：_manual_content
# 用途：在长期记忆管理与检索中处理 `_manual_content`，通过 `visible.strip` 完成首个内部
# 处理步骤。
# 参数：
#   visible：`visible`输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`normalized.startswith` →
# `normalized[len(_CORE_HEADING):].lstrip` →
# `normalized.split(_MANAGED_HEADING, 1)[0].rstrip`。
def _manual_content(visible: str) -> str:

    normalized = visible.strip()
    if normalized.startswith(_CORE_HEADING):
        normalized = normalized[len(_CORE_HEADING) :].lstrip()
    if _MANAGED_HEADING in normalized:
        normalized = normalized.split(_MANAGED_HEADING, 1)[0].rstrip()
    return normalized


# 函数说明：_render_document
# 用途：生成展示文本`document`，供长期记忆管理与检索使用。
# 参数：
#   entries：条目输入或配置值，类型 `dict[str, CoreMemoryEntry]`。
#   manual_content：正文输入或配置值，类型 `str`。
# 返回：类型 `tuple[str, str]`；返回 `(f'---\n{front}---\n{visible}', visible)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now(UTC).isoformat` →
# `datetime.now` → `entry.updated_at.isoformat` → `yaml.safe_dump` →
# `'\n'.join(body).rstrip`。
def _render_document(
    entries: dict[str, CoreMemoryEntry],
    *,
    manual_content: str,
) -> tuple[str, str]:
    metadata = {
        "format": _CORE_FORMAT,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "entries": [
            {
                "key": entry.key,
                "value": entry.value,
                "reason": entry.reason,
                "source_statement": entry.source_statement,
                "updated_at": entry.updated_at.isoformat(timespec="seconds"),
            }
            for entry in sorted(entries.values(), key=lambda item: item.key)
        ],
    }
    front = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
    body: list[str] = [_CORE_HEADING, ""]
    if manual_content:
        body.extend((manual_content, ""))
    body.extend((_MANAGED_HEADING, ""))
    for entry in sorted(entries.values(), key=lambda item: item.key):
        body.extend((f"### {entry.key}", "", entry.value, ""))
    visible = "\n".join(body).rstrip() + "\n"
    return f"---\n{front}---\n{visible}", visible


__all__ = [
    "CoreMemoryEntry",
    "CoreMemoryManager",
    "DEFAULT_MAX_CORE_TOKENS",
    "normalize_core_key",
]
