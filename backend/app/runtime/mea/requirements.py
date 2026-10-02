
"""MEA 的权威要求：组装、追加修订、渲染和长度检查。

权威要求 = 触发 Plan 的原始请求 + Task.goal / description / constraints + 用户后续修订。
它原文进入三个角色和最终回复的输入，任何时候都不截断；超过上限直接报错，
由调用方拒绝启动或把 MEA 转入 waiting_user。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

MAX_REQUIREMENTS_CHARS = 16_000


class AmendmentKind(StrEnum):

    ANSWER = "answer"  # 请示用户后的回答
    NOTE = "note"  # 用户主动补充的持续有效要求


_KIND_LABELS = {
    AmendmentKind.ANSWER: "请示回答",
    AmendmentKind.NOTE: "补充要求",
}


class RequirementsError(ValueError):
    pass


class RequirementsTooLongError(RequirementsError):

    # 函数说明：RequirementsTooLongError.__init__
    # 用途：初始化 RequirementsTooLongError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   length：`length`输入或配置值，类型 `int`。
    #   limit：本次返回或处理的数量上限，类型 `int`；默认 `MAX_REQUIREMENTS_CHARS`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.length`、`self.limit`。
    def __init__(self, length: int, limit: int = MAX_REQUIREMENTS_CHARS) -> None:
        self.length = length
        self.limit = limit
        super().__init__(
            f"authoritative requirements are {length} chars, over the {limit} limit; "
            "they are never truncated"
        )


@dataclass(frozen=True, slots=True)
class Amendment:

    id: str  # A1、A2…，只追加，编号不复用
    kind: AmendmentKind
    text: str
    revision: int  # 该修订生效后的要求版本
    at: datetime

    # 函数说明：Amendment.to_dict
    # 用途：将当前记录转为字典载荷，具体公开字段及转换规则由返回表达式确定。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `id`、`kind`、`text`、`revision`、`at`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.at.isoformat`。
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "text": self.text,
            "revision": self.revision,
            "at": self.at.isoformat(),
        }

    # 函数说明：Amendment.from_dict
    # 用途：返回 `cls(…)`，提供 Amendment 的派生值。
    # 参数：
    #   data：待解析或写入的数据，类型 `Mapping[str, Any]`；读取键 `id`、`kind`、`text`
    # 、`revision`、`at`。
    # 返回：类型 `Amendment`；返回 `cls(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`cls` → `AmendmentKind` →
    # `_parse_datetime`。
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Amendment:
        return cls(
            id=str(data["id"]),
            kind=AmendmentKind(data["kind"]),
            text=str(data["text"]),
            revision=int(data["revision"]),
            at=_parse_datetime(data["at"]),
        )


@dataclass(frozen=True, slots=True)
class Requirements:

    original_request: str
    goal: str
    description: str | None = None
    constraints: tuple[str, ...] = ()
    amendments: tuple[Amendment, ...] = field(default_factory=tuple)
    revision: int = 1

    # 函数说明：Requirements.amendment
    # 用途：在规划、执行、审计协作中处理 `amendment`，通过 `amendment_id.strip().upper`
    # 完成首个内部处理步骤。
    # 参数：
    #   amendment_id：`amendment`标识，类型 `str`。
    # 返回：类型 `Amendment | None`；返回
    # `next((item for item in self.amendments if item.id == wanted), None)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`amendment_id.strip().upper` →
    # `next`。
    def amendment(self, amendment_id: str) -> Amendment | None:
        wanted = amendment_id.strip().upper()
        return next((item for item in self.amendments if item.id == wanted), None)

    # 函数说明：Requirements.to_json
    # 用途：返回 `json.dumps(…)`，提供 Requirements 的派生值。
    # 返回：类型 `str`；返回 `json.dumps(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps` → `item.to_dict`。
    def to_json(self) -> str:
        return json.dumps(
            {
                "original_request": self.original_request,
                "goal": self.goal,
                "description": self.description,
                "constraints": list(self.constraints),
                "amendments": [item.to_dict() for item in self.amendments],
                "revision": self.revision,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    # 函数说明：Requirements.from_json
    # 用途：在规划、执行、审计协作中处理 `from_json`，通过 `json.loads` 完成首个内部处理
    # 步骤。
    # 参数：
    #   raw：传给 `json.loads` 的输入，类型 `str`。
    # 返回：类型 `Requirements`；返回 `cls(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` → `cls` →
    # `Amendment.from_dict`。
    # 分支与异常：
    #   当 `not isinstance(data, dict)` 时，抛出 `RequirementsError(…)`。
    @classmethod
    def from_json(cls, raw: str) -> Requirements:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RequirementsError("requirements snapshot must be a JSON object")
        return cls(
            original_request=str(data.get("original_request") or ""),
            goal=str(data.get("goal") or ""),
            description=data.get("description") or None,
            constraints=tuple(str(item) for item in data.get("constraints") or ()),
            amendments=tuple(
                Amendment.from_dict(item) for item in data.get("amendments") or ()
            ),
            revision=int(data.get("revision") or 1),
        )


# 函数说明：build_requirements
# 用途：组装第 1 版要求并检查长度。
# 参数：
#   original_request：传给 `_clean` 的输入，类型 `str`。
#   goal：`goal`输入或配置值，类型 `str | None`。
#   description：补充描述，类型 `str | None`；默认 `None`。
#   constraints：`constraints`输入或配置值，类型 `Iterable[str]`；默认 `()`。
# 返回：类型 `Requirements`；返回 `requirements`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_clean` → `Requirements` →
# `check_length`。
# 分支与异常：
#   当 `not resolved_goal` 时，抛出 `RequirementsError(…)`。
def build_requirements(
    *,
    original_request: str,
    goal: str | None,
    description: str | None = None,
    constraints: Iterable[str] = (),
) -> Requirements:
    """组装第 1 版要求并检查长度。goal 为空时用原始请求代替。"""

    request = _clean(original_request)
    resolved_goal = _clean(goal or "") or request
    if not resolved_goal:
        raise RequirementsError("requirements need an original request or a goal")
    requirements = Requirements(
        original_request=request,
        goal=resolved_goal,
        description=_clean(description or "") or None,
        constraints=tuple(item for item in (_clean(c) for c in constraints) if item),
        revision=1,
    )
    check_length(requirements)
    return requirements


# 函数说明：add_amendment
# 用途：追加一条持续有效的修订，返回新版本；超长时抛错，原版本不变。
# 参数：
#   requirements：当前生效的任务要求，类型 `Requirements`。
#   kind：传给 `AmendmentKind` 的输入，类型 `AmendmentKind | str`。
#   text：待处理的文本，类型 `str`。
#   at：`at`输入或配置值，类型 `datetime | None`；默认 `None`。
# 返回：类型 `Requirements`；返回 `updated`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_clean` → `Amendment` →
# `AmendmentKind` → `(at or datetime.now(UTC)).astimezone` → `datetime.now` → `replace`
# ；另有 1 个调用点。
# 分支与异常：
#   当 `not body` 时，抛出 `RequirementsError('amendment text cannot be empty')`。
def add_amendment(
    requirements: Requirements,
    *,
    kind: AmendmentKind | str,
    text: str,
    at: datetime | None = None,
) -> Requirements:
    """追加一条持续有效的修订，返回新版本；超长时抛错，原版本不变。"""

    body = _clean(text)
    if not body:
        raise RequirementsError("amendment text cannot be empty")
    revision = requirements.revision + 1
    amendment = Amendment(
        id=f"A{len(requirements.amendments) + 1}",
        kind=AmendmentKind(kind),
        text=body,
        revision=revision,
        at=(at or datetime.now(UTC)).astimezone(UTC),
    )
    updated = replace(
        requirements,
        amendments=(*requirements.amendments, amendment),
        revision=revision,
    )
    check_length(updated)
    return updated


# 函数说明：render_requirements
# 用途：渲染为提示词里的“权威要求”小节。
# 参数：
#   requirements：当前生效的任务要求，类型 `Requirements`。
# 返回：类型 `str`；返回 `text`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_render`。
# 分支与异常：
#   当 `len(text) > MAX_REQUIREMENTS_CHARS` 时，抛出
# `RequirementsTooLongError(len(text))`。
def render_requirements(requirements: Requirements) -> str:
    """渲染为提示词里的“权威要求”小节。结果超过上限时抛错，从不截断。"""

    text = _render(requirements)
    if len(text) > MAX_REQUIREMENTS_CHARS:
        raise RequirementsTooLongError(len(text))
    return text


# 函数说明：check_length
# 用途：检查`length`，供规划、执行、审计协作使用。
# 参数：
#   requirements：当前生效的任务要求，类型 `Requirements`。
# 返回：类型 `int`；返回 `length`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_render`。
# 分支与异常：
#   当 `length > MAX_REQUIREMENTS_CHARS` 时，抛出 `RequirementsTooLongError(length)`。
def check_length(requirements: Requirements) -> int:
    length = len(_render(requirements))
    if length > MAX_REQUIREMENTS_CHARS:
        raise RequirementsTooLongError(length)
    return length


# 函数说明：_render
# 用途：生成展示文本规划、执行、审计协作，供规划、执行、审计协作使用。
# 参数：
#   requirements：当前生效的任务要求，类型 `Requirements`。
# 返回：类型 `str`；返回 `'\n\n'.join(parts)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`item.at.astimezone(UTC).strftime` →
# `item.at.astimezone`。
def _render(requirements: Requirements) -> str:
    parts = [
        "权威要求（原文，不得改写、不得截断；任务契约是对它的解释，冲突时以这里为准）:",
        "[原始请求]\n" + (requirements.original_request or "(无)"),
        "[任务目标]\n" + requirements.goal,
    ]
    if requirements.description:
        parts.append("[任务说明]\n" + requirements.description)
    constraints = "\n".join(f"- {item}" for item in requirements.constraints)
    parts.append("[明确约束]\n" + (constraints or "(无)"))
    if requirements.amendments:
        lines = [
            f"- {item.id}（生效于要求 v{item.revision}，"
            f"{item.at.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')}，"
            f"{_KIND_LABELS[item.kind]}）: {item.text}"
            for item in requirements.amendments
        ]
        parts.append("[用户后续修订（按时间顺序，后者优先）]\n" + "\n".join(lines))
    parts.append(f"当前要求版本: v{requirements.revision}")
    return "\n\n".join(parts)


# 函数说明：_clean
# 用途：返回 `str(value or '').replace('\r\n', '\n').strip()`，提供 规划、执行、审计协作
#  的派生值。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `str`；返回 `str(value or '').replace('\r\n', '\n').strip()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`str(value or '').replace`。
def _clean(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


# 函数说明：_parse_datetime
# 用途：解析日期时间，供规划、执行、审计协作使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `object`。
# 返回：类型 `datetime`；返回 `parsed.astimezone(UTC)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.fromisoformat` →
# `parsed.replace` → `parsed.astimezone`。
def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = [
    "MAX_REQUIREMENTS_CHARS",
    "Amendment",
    "AmendmentKind",
    "Requirements",
    "RequirementsError",
    "RequirementsTooLongError",
    "add_amendment",
    "build_requirements",
    "check_length",
    "render_requirements",
]
