
from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from .models import (
    SKILL_DESCRIPTION_MAX_LENGTH,
    validate_skill_name,
)

_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "name",
        "description",
        "license",
        "compatibility",
        "metadata",
        "allowed-tools",
        "allowed_tools",
    }
)


class SkillParseError(ValueError):
    pass


class ParsedSkill(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, object] | None = None
    allowed_tools: tuple[str, ...] = ()
    body: str


# 函数说明：parse_skill_document
# 用途：解析技能前置元数据和正文，并校验名称及允许的工具。
# 参数：
#   text：待处理的文本，类型 `str`。
#   expected_name：名称输入或配置值，类型 `str`。
# 返回：类型 `ParsedSkill`；返回 `ParsedSkill(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_split_front_matter` →
# `yaml.safe_load` → `validate_skill_name` → `_optional_str` → `_allowed_tools` →
# `ParsedSkill`。
# 分支与异常：
#   当 `front is None` 时，抛出 `SkillParseError('missing YAML front matter')`。
#   捕获 `yaml.YAMLError` 后，转换或抛出
# `SkillParseError(f'invalid YAML front matter: {exc}')`。
#   当 `not isinstance(front_matter, dict)` 时，抛出
# `SkillParseError('front matter must be a mapping')`。
#   当 `unknown` 时，抛出 `SkillParseError(…)`。
#   捕获 `ValueError` 后，转换或抛出 `SkillParseError(str(exc))`。
def parse_skill_document(text: str, *, expected_name: str) -> ParsedSkill:

    """解析技能前置元数据和正文，并校验名称及允许的工具。"""
    front, body = _split_front_matter(text)
    if front is None:
        raise SkillParseError("missing YAML front matter")
    try:
        front_matter = yaml.safe_load(front)
    except yaml.YAMLError as exc:
        raise SkillParseError(f"invalid YAML front matter: {exc}") from exc
    if not isinstance(front_matter, dict):
        raise SkillParseError("front matter must be a mapping")

    unknown = sorted(set(front_matter) - _ALLOWED_TOP_LEVEL_FIELDS)
    if unknown:
        raise SkillParseError(
            f"unknown front matter field(s): {', '.join(unknown)}"
        )

    name = front_matter.get("name")
    if not isinstance(name, str):
        raise SkillParseError("missing or non-string 'name'")
    try:
        normalized_name = validate_skill_name(name)
    except ValueError as exc:
        raise SkillParseError(str(exc)) from exc
    if normalized_name != expected_name:
        raise SkillParseError(
            f"front matter name '{normalized_name}' does not match "
            f"directory name '{expected_name}'"
        )

    description = front_matter.get("description")
    if not isinstance(description, str):
        raise SkillParseError("missing or non-string 'description'")
    description = description.strip()
    if not description:
        raise SkillParseError("empty 'description'")
    if len(description) > SKILL_DESCRIPTION_MAX_LENGTH:
        raise SkillParseError(
            f"'description' exceeds {SKILL_DESCRIPTION_MAX_LENGTH} chars"
        )

    license_value = _optional_str(front_matter, "license", "license")
    compatibility = _optional_str(front_matter, "compatibility", "compatibility")

    metadata_value = front_matter.get("metadata")
    if metadata_value is not None and not isinstance(metadata_value, dict):
        raise SkillParseError("'metadata' must be a mapping")

    allowed_tools = _allowed_tools(front_matter)
    if not body.strip():
        raise SkillParseError("empty skill body")

    return ParsedSkill(
        name=normalized_name,
        description=description,
        license=license_value,
        compatibility=compatibility,
        metadata=metadata_value,
        allowed_tools=allowed_tools,
        body=body.strip(),
    )


# 函数说明：_optional_str
# 用途：在技能发现与激活中处理 `_optional_str`，通过 `front_matter.get` 完成首个内部处理
# 步骤。
# 参数：
#   front_matter：`front_matter`输入或配置值，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
#   label：`label`输入或配置值，类型 `str`。
# 返回：类型 `str | None`；按分支返回 `None`；`stripped`。
# 分支与异常：
#   当 `value is None` 时，返回 `None`。
#   当 `not isinstance(value, str)` 时，抛出
# `SkillParseError(f"'{label}' must be a string")`。
#   当 `not stripped` 时，抛出 `SkillParseError(f"'{label}' cannot be empty")`。
def _optional_str(
    front_matter: dict[str, Any],
    key: str,
    label: str,
) -> str | None:
    value = front_matter.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SkillParseError(f"'{label}' must be a string")
    stripped = value.strip()
    if not stripped:
        raise SkillParseError(f"'{label}' cannot be empty")
    return stripped


# 函数说明：_allowed_tools
# 用途：在技能发现与激活中处理 `_allowed_tools`，通过 `front_matter.get` 完成首个内部处
# 理步骤。
# 参数：
#   front_matter：`front_matter`输入或配置值，类型 `dict[str, Any]`；读取键
# `allowed-tools`、`allowed_tools`。
# 返回：类型 `tuple[str, ...]`；按分支返回 `()`；
# `tuple((str(item).strip() for item in raw))`。
# 分支与异常：
#   当 `hyphen_key is not None and underscore_key is not None` 时，抛出
# `SkillParseError(…)`。
#   当 `raw is None` 时，返回 `()`。
#   当 `not isinstance(raw, list) or not all((isinstance(item, str)…` 时，抛出
# `SkillParseError(…)`。
def _allowed_tools(front_matter: dict[str, Any]) -> tuple[str, ...]:
    hyphen_key = front_matter.get("allowed-tools")
    underscore_key = front_matter.get("allowed_tools")
    if hyphen_key is not None and underscore_key is not None:
        raise SkillParseError(
            "'allowed-tools' and 'allowed_tools' cannot both be present"
        )
    raw = hyphen_key if hyphen_key is not None else underscore_key
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and item.strip() for item in raw
    ):
        raise SkillParseError("'allowed-tools' must be a list of non-empty strings")
    return tuple(str(item).strip() for item in raw)


# 函数说明：_split_front_matter
# 用途：拆分`front_matter`，供技能发现与激活使用。
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


__all__ = ["ParsedSkill", "SkillParseError", "parse_skill_document"]
