
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.runtime.mea.requirements import (
    MAX_REQUIREMENTS_CHARS,
    AmendmentKind,
    Requirements,
    RequirementsError,
    RequirementsTooLongError,
    add_amendment,
    build_requirements,
    render_requirements,
)

_AT = datetime(2026, 9, 28, 13, 30, tzinfo=UTC)


# 函数说明：_base
# 用途：返回 `build_requirements(…)`，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `Requirements`；返回 `build_requirements(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_requirements`。
def _base() -> Requirements:
    return build_requirements(
        original_request="把 data/users.csv 导入数据库，并导出一份 JSON 备份",
        goal="users 表包含 CSV 的全部数据",
        description="CSV 是 UTF-8，第一行是表头",
        constraints=["不要修改数据库结构", "  只使用 scripts/ 下的脚本  ", ""],
    )


# 函数说明：test_render_contains_every_authoritative_part
# 用途：回归验证回归测试与测试辅助中的 `render_contains_every_authoritative_part` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`render_requirements` → `_base` →
# `text.endswith`。
# 分支与异常：
#   验证条件：`'[原始请求]\n把 data/users.csv 导入数据库，并导出一份 JSON 备份' in text`
# 。
#   验证条件：`'[任务目标]\nusers 表包含 CSV 的全部数据' in text`。
#   验证条件：`'[任务说明]\nCSV 是 UTF-8，第一行是表头' in text`。
#   验证条件：`'- 不要修改数据库结构\n- 只使用 scripts/ 下的脚本' in text`。
def test_render_contains_every_authoritative_part() -> None:
    text = render_requirements(_base())

    assert "[原始请求]\n把 data/users.csv 导入数据库，并导出一份 JSON 备份" in text
    assert "[任务目标]\nusers 表包含 CSV 的全部数据" in text
    assert "[任务说明]\nCSV 是 UTF-8，第一行是表头" in text
    assert "- 不要修改数据库结构\n- 只使用 scripts/ 下的脚本" in text
    assert "用户后续修订" not in text
    assert text.endswith("当前要求版本: v1")


# 函数说明：test_description_is_omitted_when_empty
# 用途：回归验证回归测试与测试辅助中的 `description_is_omitted_when_empty` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_requirements` →
# `render_requirements`。
# 分支与异常：
#   验证条件：`'[任务说明]' not in text`。
#   验证条件：`'[任务目标]\n跑一遍测试' in text`。
#   验证条件：`'[明确约束]\n(无)' in text`。
def test_description_is_omitted_when_empty() -> None:
    requirements = build_requirements(original_request="跑一遍测试", goal=None)
    text = render_requirements(requirements)

    assert "[任务说明]" not in text
    assert "[任务目标]\n跑一遍测试" in text  # goal 为空时用原始请求
    assert "[明确约束]\n(无)" in text


# 函数说明：test_build_requires_request_or_goal
# 用途：回归验证回归测试与测试辅助中的 `build_requires_request_or_goal` 场景，下方断言说
# 明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `build_requirements`
# 。
# 分支与异常：
#   预期异常：`pytest.raises(RequirementsError)`。
def test_build_requires_request_or_goal() -> None:
    with pytest.raises(RequirementsError):
        build_requirements(original_request="  ", goal="")


# 函数说明：test_amendments_are_numbered_and_bump_revision
# 用途：回归验证回归测试与测试辅助中的 `amendments_are_numbered_and_bump_revision` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`add_amendment` → `_base` →
# `second.amendment` → `render_requirements` → `text.endswith`。
# 分支与异常：
#   验证条件：`[a.id for a in second.amendments] == ['A1', 'A2']`。
#   验证条件：`[a.revision for a in second.amendments] == [2, 3]`。
#   验证条件：`second.revision == 3`。
#   验证条件：`second.amendment('a1') is not None`。
def test_amendments_are_numbered_and_bump_revision() -> None:
    first = add_amendment(_base(), kind=AmendmentKind.NOTE, text="改为导出 CSV，不要 JSON", at=_AT)
    second = add_amendment(first, kind="answer", text="用测试库", at=_AT)

    assert [a.id for a in second.amendments] == ["A1", "A2"]
    assert [a.revision for a in second.amendments] == [2, 3]
    assert second.revision == 3
    assert second.amendment("a1") is not None
    text = render_requirements(second)
    assert "- A1（生效于要求 v2，2026-09-28 13:30 UTC，补充要求）: 改为导出 CSV，不要 JSON" in text
    assert "- A2（生效于要求 v3，2026-09-28 13:30 UTC，请示回答）: 用测试库" in text
    assert text.endswith("当前要求版本: v3")


# 函数说明：test_empty_amendment_is_rejected
# 用途：回归验证回归测试与测试辅助中的 `empty_amendment_is_rejected` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `add_amendment` →
# `_base`。
# 分支与异常：
#   预期异常：`pytest.raises(RequirementsError)`。
def test_empty_amendment_is_rejected() -> None:
    with pytest.raises(RequirementsError):
        add_amendment(_base(), kind=AmendmentKind.NOTE, text="   ")


# 函数说明：test_json_round_trip
# 用途：回归验证回归测试与测试辅助中的 `json_round_trip` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`add_amendment` → `_base` →
# `Requirements.from_json` → `requirements.to_json` → `render_requirements`。
# 分支与异常：
#   验证条件：`restored == requirements`。
#   验证条件：`render_requirements(restored) == render_requirements(requirements)`。
def test_json_round_trip() -> None:
    requirements = add_amendment(_base(), kind=AmendmentKind.NOTE, text="保留原 id", at=_AT)
    restored = Requirements.from_json(requirements.to_json())

    assert restored == requirements
    assert render_requirements(restored) == render_requirements(requirements)


# 函数说明：test_too_long_is_rejected_not_truncated
# 用途：回归验证回归测试与测试辅助中的 `too_long_is_rejected_not_truncated` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `build_requirements`
# 。
# 分支与异常：
#   验证条件：`excinfo.value.length > MAX_REQUIREMENTS_CHARS`。
#   预期异常：`pytest.raises(RequirementsTooLongError)`。
def test_too_long_is_rejected_not_truncated() -> None:
    long_constraint = "x" * (MAX_REQUIREMENTS_CHARS + 1)
    with pytest.raises(RequirementsTooLongError) as excinfo:
        build_requirements(original_request="任务", goal="目标", constraints=[long_constraint])
    assert excinfo.value.length > MAX_REQUIREMENTS_CHARS


# 函数说明：test_amendment_that_overflows_is_rejected_and_original_kept
# 用途：回归验证回归测试与测试辅助中的
# `amendment_that_overflows_is_rejected_and_original_kept` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_base` → `pytest.raises` →
# `add_amendment`。
# 分支与异常：
#   验证条件：`base.revision == 1`。
#   验证条件：`base.amendments == ()`。
#   预期异常：`pytest.raises(RequirementsTooLongError)`。
def test_amendment_that_overflows_is_rejected_and_original_kept() -> None:
    base = _base()
    with pytest.raises(RequirementsTooLongError):
        add_amendment(base, kind=AmendmentKind.NOTE, text="y" * MAX_REQUIREMENTS_CHARS)
    assert base.revision == 1
    assert base.amendments == ()


# 函数说明：test_limit_is_inclusive_and_text_is_kept_whole
# 用途：回归验证回归测试与测试辅助中的 `limit_is_inclusive_and_text_is_kept_whole` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`render_requirements` →
# `build_requirements` → `pytest.raises`。
# 分支与异常：
#   验证条件：`len(text) == MAX_REQUIREMENTS_CHARS`。
#   验证条件：`'z' * fits in text`。
#   预期异常：`pytest.raises(RequirementsTooLongError)`。
def test_limit_is_inclusive_and_text_is_kept_whole() -> None:
    base_length = len(render_requirements(build_requirements(original_request="任务", goal="目标")))
    # 约束从 "(无)"（3 字符）变成 "- " + n 个字符
    fits = MAX_REQUIREMENTS_CHARS - base_length + 1
    text = render_requirements(
        build_requirements(original_request="任务", goal="目标", constraints=["z" * fits])
    )
    assert len(text) == MAX_REQUIREMENTS_CHARS
    assert "z" * fits in text

    with pytest.raises(RequirementsTooLongError):
        build_requirements(original_request="任务", goal="目标", constraints=["z" * (fits + 1)])
