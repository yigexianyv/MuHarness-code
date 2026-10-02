
from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_NAME_MAX_LENGTH = 64
SKILL_DESCRIPTION_MAX_LENGTH = 1024
SKILL_FILE_NAME = "SKILL.md"


class SkillScope(StrEnum):

    USER = "user"
    PROJECT = "project"


# 函数说明：validate_skill_name
# 用途：校验技能名称，供技能发现与激活使用。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_SKILL_NAME_RE.fullmatch`。
# 分支与异常：
#   当 `not normalized` 时，抛出 `ValueError('skill name cannot be empty')`。
#   当 `len(normalized) > SKILL_NAME_MAX_LENGTH` 时，抛出 `ValueError(…)`。
#   当 `not _SKILL_NAME_RE.fullmatch(normalized)` 时，抛出 `ValueError(…)`。
def validate_skill_name(name: str) -> str:

    normalized = name.strip()
    if not normalized:
        raise ValueError("skill name cannot be empty")
    if len(normalized) > SKILL_NAME_MAX_LENGTH:
        raise ValueError(
            f"skill name exceeds {SKILL_NAME_MAX_LENGTH} chars: {name!r}"
        )
    if not _SKILL_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "skill name must be lowercase letters/digits separated by single "
            f"hyphens: {name!r}"
        )
    return normalized


# 函数说明：valid_skill_name
# 用途：处理技能发现与激活中的 `valid_skill_name` 数据；结果及边界条件见下方说明。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `bool`；按分支返回 `True`；`False`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_skill_name`。
# 分支与异常：
#   捕获 `ValueError` 后，返回 `False`。
def valid_skill_name(name: str) -> bool:

    try:
        validate_skill_name(name)
        return True
    except ValueError:
        return False


class SkillMetadata(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    scope: SkillScope
    location: Path
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, object] | None = None
    allowed_tools: tuple[str, ...] = ()

    # 函数说明：SkillMetadata.render_catalog_entry
    # 用途：生成展示文本条目，供技能发现与激活使用。
    # 返回：类型 `str`；返回 `f'[{self.name}] {self.description}'`。
    def render_catalog_entry(self) -> str:

        return f"[{self.name}] {self.description}"


class SkillResources(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    scripts: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()

    # 函数说明：SkillResources.as_dict
    # 用途：将当前记录转为字典载荷，具体公开字段及转换规则由返回表达式确定。
    # 返回：类型 `dict[str, tuple[str, ...]]`；字典，包含字段 `references`、`scripts`、
    # `assets`。
    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {
            "references": self.references,
            "scripts": self.scripts,
            "assets": self.assets,
        }

    # 函数说明：SkillResources.is_empty
    # 用途：判断`empty`是否满足当前实现的条件。
    # 返回：类型 `bool`；返回 `not (self.scripts or self.references or self.assets)`。
    def is_empty(self) -> bool:
        return not (self.scripts or self.references or self.assets)


class Skill(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: SkillMetadata
    content: str
    root: Path
    resources: SkillResources = Field(default_factory=SkillResources)

    # 函数说明：Skill.render_instructions
    # 用途：生成展示文本`instructions`，供技能发现与激活使用。
    # 返回：类型 `str`；返回 `'\n'.join(body)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.resources.is_empty` →
    # `self.resources.as_dict`。
    def render_instructions(self) -> str:

        header = f"# Skill: {self.metadata.name}"
        body = [
            header,
            "\n技能依据：按当前权限与用户要求使用；加载不代表已执行。\n",
            self.content.strip(),
        ]
        if not self.resources.is_empty():
            body.extend(
                (
                    "",
                    "## 配套资源",
                    "",
                    "资源按需用 skill_resource_read 读取；未读取前不能假定其内容，"
                    "列出资源不表示自动加载或执行脚本：",
                )
            )
            for kind, items in self.resources.as_dict().items():
                if items:
                    body.append(f"- {kind}: " + ", ".join(items))
        return "\n".join(body)


__all__ = [
    "SKILL_DESCRIPTION_MAX_LENGTH",
    "SKILL_FILE_NAME",
    "SKILL_NAME_MAX_LENGTH",
    "Skill",
    "SkillMetadata",
    "SkillResources",
    "SkillScope",
    "valid_skill_name",
    "validate_skill_name",
]
