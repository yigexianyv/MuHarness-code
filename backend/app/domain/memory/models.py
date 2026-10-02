
from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_MEMORY_ID_PREFIX = "M"
_MEMORY_ID_RE = re.compile(r"^M[0-9]{3,}$")
_MAX_TITLE_CHARS = 200
_MAX_SUMMARY_CHARS = 500
_MAX_CONTENT_CHARS = 12_000
_MAX_REASON_CHARS = 1_000


class MemoryStatus(StrEnum):

    ACTIVE = "active"
    ARCHIVED = "archived"


class MemoryRecord(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^M[0-9]{3,}$")
    title: str
    summary: str
    content: str
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime
    access_count: int = Field(default=0, ge=0)
    revision: int = Field(default=1, ge=1)
    status: MemoryStatus = MemoryStatus.ACTIVE
    last_update_reason: str | None = None
    archive_reason: str | None = None

    # 函数说明：MemoryRecord.normalize_cue_text
    # 用途：校验并规范化模型字段 'title'、'summary'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('memory title and summary must be strings')`。
    #   当 `not normalized` 时，抛出
    # `ValueError('memory title and summary cannot be empty')`。
    @field_validator("title", "summary", mode="before")
    @classmethod
    def normalize_cue_text(cls, value: object) -> str:

        if not isinstance(value, str):
            raise TypeError("memory title and summary must be strings")
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("memory title and summary cannot be empty")
        return normalized

    # 函数说明：MemoryRecord.validate_title_length
    # 用途：校验并规范化模型字段 'title'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `value`。
    # 分支与异常：
    #   当 `len(value) > _MAX_TITLE_CHARS` 时，抛出 `ValueError(…)`。
    @field_validator("title")
    @classmethod
    def validate_title_length(cls, value: str) -> str:
        if len(value) > _MAX_TITLE_CHARS:
            raise ValueError(f"memory title exceeds {_MAX_TITLE_CHARS} characters")
        return value

    # 函数说明：MemoryRecord.validate_summary_length
    # 用途：校验并规范化模型字段 'summary'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `str`；返回 `value`。
    # 分支与异常：
    #   当 `len(value) > _MAX_SUMMARY_CHARS` 时，抛出 `ValueError(…)`。
    @field_validator("summary")
    @classmethod
    def validate_summary_length(cls, value: str) -> str:
        if len(value) > _MAX_SUMMARY_CHARS:
            raise ValueError(
                f"memory summary exceeds {_MAX_SUMMARY_CHARS} characters"
            )
        return value

    # 函数说明：MemoryRecord.normalize_content
    # 用途：校验并规范化模型字段 'content'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str`；返回 `normalized`。
    # 分支与异常：
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('memory content must be a string')`。
    #   当 `not normalized` 时，抛出 `ValueError('memory content cannot be empty')`。
    #   当 `len(normalized) > _MAX_CONTENT_CHARS` 时，抛出 `ValueError(…)`。
    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("memory content must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory content cannot be empty")
        if len(normalized) > _MAX_CONTENT_CHARS:
            raise ValueError(
                f"memory content exceeds {_MAX_CONTENT_CHARS} characters"
            )
        return normalized

    # 函数说明：MemoryRecord.normalize_reason
    # 用途：校验并规范化模型字段 'last_update_reason'、'archive_reason'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `str | None`；按分支返回 `None`；`normalized`。
    # 分支与异常：
    #   当 `value is None` 时，返回 `None`。
    #   当 `not isinstance(value, str)` 时，抛出
    # `TypeError('memory change reason must be a string')`。
    #   当 `not normalized` 时，返回 `None`。
    #   当 `len(normalized) > _MAX_REASON_CHARS` 时，抛出 `ValueError(…)`。
    @field_validator("last_update_reason", "archive_reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("memory change reason must be a string")
        normalized = " ".join(value.split())
        if not normalized:
            return None
        if len(normalized) > _MAX_REASON_CHARS:
            raise ValueError(
                f"memory change reason exceeds {_MAX_REASON_CHARS} characters"
            )
        return normalized

    # 函数说明：MemoryRecord.front_matter
    # 用途：处理长期记忆管理与检索中的 `front_matter` 数据；结果及边界条件见下方说明。
    # 返回：类型 `dict[str, object]`；返回 `metadata`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_iso`。
    def front_matter(self) -> dict[str, object]:

        metadata: dict[str, object] = {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "last_accessed_at": _iso(self.last_accessed_at),
            "access_count": self.access_count,
            "revision": self.revision,
            "status": self.status.value,
        }
        if self.last_update_reason is not None:
            metadata["last_update_reason"] = self.last_update_reason
        if self.archive_reason is not None:
            metadata["archive_reason"] = self.archive_reason
        return metadata

    # 函数说明：MemoryRecord.render_markdown
    # 用途：生成展示文本Markdown 文本，供长期记忆管理与检索使用。
    # 返回：类型 `str`；返回 `f'---\n{front}---\n{body}'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`yaml.safe_dump` →
    # `self.front_matter`。
    def render_markdown(self) -> str:

        front = yaml.safe_dump(
            self.front_matter(),
            allow_unicode=True,
            sort_keys=False,
        )
        body = "\n".join(
            (
                f"# {self.title}",
                "",
                "## Summary",
                "",
                self.summary,
                "",
                "## Memory",
                "",
                self.content,
                "",
            )
        )
        return f"---\n{front}---\n{body}"

    # 函数说明：MemoryRecord.render_full
    # 用途：生成展示文本`full`，供长期记忆管理与检索使用。
    # 返回：类型 `str`；返回
    # `f'# {self.title}\n\n## Summary\n\n{self.summary}\n\n## Memory\n\n{self.content}'`
    # 。
    def render_full(self) -> str:

        return (
            f"# {self.title}\n\n"
            f"## Summary\n\n{self.summary}\n\n"
            f"## Memory\n\n{self.content}"
        )


# 函数说明：parse_memory_markdown
# 用途：解析记忆Markdown 文本，供长期记忆管理与检索使用。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `MemoryRecord`；返回 `MemoryRecord(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_split_front_matter` →
# `yaml.safe_load` → `MemoryRecord` → `_extract_memory_section` → `_parse_iso` →
# `MemoryStatus`。
# 分支与异常：
#   当 `front is None` 时，抛出 `ValueError('memory file is missing YAML front matter')`
# 。
#   当 `not isinstance(front_matter, dict)` 时，抛出
# `ValueError('memory front matter must be a mapping')`。
def parse_memory_markdown(text: str) -> MemoryRecord:

    front, body = _split_front_matter(text)
    if front is None:
        raise ValueError("memory file is missing YAML front matter")
    front_matter = yaml.safe_load(front)
    if not isinstance(front_matter, dict):
        raise ValueError("memory front matter must be a mapping")
    return MemoryRecord(
        id=str(front_matter["id"]),
        title=str(front_matter["title"]),
        summary=str(front_matter.get("summary", "")),
        content=_extract_memory_section(body),
        created_at=_parse_iso(str(front_matter["created_at"])),
        updated_at=_parse_iso(str(front_matter["updated_at"])),
        last_accessed_at=_parse_iso(str(front_matter["last_accessed_at"])),
        access_count=int(front_matter.get("access_count", 0)),
        revision=int(front_matter.get("revision", 1)),
        status=MemoryStatus(
            str(front_matter.get("status", MemoryStatus.ACTIVE.value))
        ),
        last_update_reason=front_matter.get("last_update_reason"),
        archive_reason=front_matter.get("archive_reason"),
    )


# 函数说明：normalize_memory_id
# 用途：规范化记忆标识，供长期记忆管理与检索使用。
# 参数：
#   memory_id：目标记忆标识，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`memory_id.strip().upper` →
# `_MEMORY_ID_RE.fullmatch`。
# 分支与异常：
#   当 `not isinstance(memory_id, str)` 时，抛出
# `TypeError('memory id must be a string')`。
#   当 `not _MEMORY_ID_RE.fullmatch(normalized)` 时，抛出 `ValueError(…)`。
def normalize_memory_id(memory_id: str) -> str:

    if not isinstance(memory_id, str):
        raise TypeError("memory id must be a string")
    normalized = memory_id.strip().upper()
    if not _MEMORY_ID_RE.fullmatch(normalized):
        raise ValueError("memory id must match M followed by at least three digits")
    return normalized


# 函数说明：next_memory_id
# 用途：在长期记忆管理与检索中处理 `next_memory_id`，通过 `memory_id.removeprefix` 完成
# 首个内部处理步骤。
# 参数：
#   existing_ids：`existing_ids`输入或配置值，类型 `set[str]`。
# 返回：类型 `str`；返回 `f'{_MEMORY_ID_PREFIX}{highest + 1:03d}'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`memory_id.removeprefix` →
# `suffix.isdigit`。
def next_memory_id(existing_ids: set[str]) -> str:

    highest = 0
    for memory_id in existing_ids:
        suffix = memory_id.removeprefix(_MEMORY_ID_PREFIX)
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"{_MEMORY_ID_PREFIX}{highest + 1:03d}"


# 函数说明：_split_front_matter
# 用途：拆分`front_matter`，供长期记忆管理与检索使用。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `tuple[str | None, str]`；按分支返回 `(None, text)`；
# `('\n'.join(lines[1:index]), '\n'.join(lines[index + 1:]).strip())`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`text.splitlines`。
# 分支与异常：
#   当 `not lines or lines[0].strip() != '---'` 时，返回 `(None, text)`。
#   当 `lines[index].strip() == '---'` 时，返回
# `('\n'.join(lines[1:index]), '\n'.join(lines[index + 1:]).…`。
def _split_front_matter(text: str) -> tuple[str | None, str]:

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :]).strip()
    return None, text


# 函数说明：_extract_memory_section
# 用途：提取记忆，供长期记忆管理与检索使用。
# 参数：
#   body：请求正文或内容主体，类型 `str`。
# 返回：类型 `str`；按分支返回 `'\n'.join(lines[index + 1:]).strip()`；`body.strip()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`body.splitlines`。
# 分支与异常：
#   当 `line.strip() == '## Memory'` 时，返回 `'\n'.join(lines[index + 1:]).strip()`。
def _extract_memory_section(body: str) -> str:

    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "## Memory":
            return "\n".join(lines[index + 1 :]).strip()
    return body.strip()


# 函数说明：_iso
# 用途：返回 `value.astimezone(UTC).isoformat(timespec='seconds')`，提供 长期记忆管理与
# 检索 的派生值。
# 参数：
#   value：待校验、规范化或转换的值，类型 `datetime`。
# 返回：类型 `str`；返回 `value.astimezone(UTC).isoformat(timespec='seconds')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`value.astimezone(UTC).isoformat` →
# `value.astimezone`。
def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds")


# 函数说明：_parse_iso
# 用途：解析`iso`，供长期记忆管理与检索使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `datetime`；返回 `parsed.astimezone(UTC)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.fromisoformat` →
# `parsed.astimezone`。
# 分支与异常：
#   当 `parsed.tzinfo is None` 时，抛出 `ValueError(…)`。
def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("memory datetimes must include timezone information")
    return parsed.astimezone(UTC)


__all__ = [
    "MemoryRecord",
    "MemoryStatus",
    "next_memory_id",
    "normalize_memory_id",
    "parse_memory_markdown",
]
