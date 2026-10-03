"""所有 YAML 用例都能加载，检查项名字都存在。不调用模型。"""

from __future__ import annotations

from tests.eval.checks import known_checks
from tests.eval.session import preflight
from tests.eval.spec import CASES_DIR, VARIANTS_DIR, load_cases, load_variant


# 函数说明：test_every_case_loads_and_uses_known_checks
# 用途：回归验证回归测试与测试辅助中的 `every_case_loads_and_uses_known_checks` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`load_cases` → `preflight`。
# 分支与异常：
#   验证条件：`cases`。
#   验证条件：`{case.suite for case in cases} == {'behavior', 'context', 'mea'}`。
def test_every_case_loads_and_uses_known_checks() -> None:
    cases = load_cases()
    assert cases, "cases/ 下没有用例"
    preflight(cases)
    assert {case.suite for case in cases} == {"behavior", "context", "mea"}


# 函数说明：test_case_ids_are_prefixed_by_suite_and_match_file_names
# 用途：回归验证回归测试与测试辅助中的
# `case_ids_are_prefixed_by_suite_and_match_file_names` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`load_cases` → `case.id.startswith`。
# 分支与异常：
#   验证条件：`case.id.startswith(prefixes[case.suite])`。
#   验证条件：`case.source == f'{case.suite}/{case.id}.yaml'`。
#   验证条件：`case.why`。
def test_case_ids_are_prefixed_by_suite_and_match_file_names() -> None:
    prefixes = {"behavior": "b", "context": "c", "mea": "m"}
    for case in load_cases():
        assert case.id.startswith(prefixes[case.suite]), case.id
        assert case.source == f"{case.suite}/{case.id}.yaml", case.source
        assert case.why, f"{case.id} 缺少 why：写清这个用例防的是哪种问题"


# 函数说明：test_variants_load_and_prompt_files_exist
# 用途：回归验证回归测试与测试辅助中的 `variants_load_and_prompt_files_exist` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`VARIANTS_DIR.glob` → `load_variant` →
#  `variant.system_prompt` → `load_variant('current').system_prompt`。
# 分支与异常：
#   验证条件：`variant.system_prompt()`。
#   验证条件：`load_variant('current').system_prompt() is None`。
def test_variants_load_and_prompt_files_exist() -> None:
    for path in sorted(VARIANTS_DIR.glob("*.yaml")):
        variant = load_variant(path.stem)
        if variant.system_prompt_file:
            assert variant.system_prompt(), variant.name
    assert load_variant("current").system_prompt() is None


# 函数说明：test_digest_changes_when_case_changes
# 用途：回归验证回归测试与测试辅助中的 `digest_changes_when_case_changes` 场景，下方断言
# 说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`load_cases` → `case.digest` →
# `changed.digest`。
# 分支与异常：
#   验证条件：`case.digest() != changed.digest()`。
def test_digest_changes_when_case_changes() -> None:
    case = load_cases(ids=["b01-greeting"])[0]
    changed = case.model_copy(update={"checks": [*case.checks, {"no_tools": True}]})
    assert case.digest() != changed.digest()


# 函数说明：test_generated_file_places_inserted_line_in_the_middle
# 用途：回归验证回归测试与测试辅助中的
# `generated_file_places_inserted_line_in_the_middle` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`load_cases` →
# `case.setup.files['logs/part-1.log'].render().splitlines` →
# `case.setup.files['logs/part-1.log'].render`。
# 分支与异常：
#   验证条件：`'E-5052' in lines[110]`。
#   验证条件：`len(lines) == 221`。
def test_generated_file_places_inserted_line_in_the_middle() -> None:
    case = load_cases(ids=["c03-middle-fact-recovered"])[0]
    lines = case.setup.files["logs/part-1.log"].render().splitlines()
    assert "E-5052" in lines[110]
    assert len(lines) == 221


# 函数说明：test_known_checks_cover_documented_names
# 用途：回归验证回归测试与测试辅助中的 `known_checks_cover_documented_names` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.is_file` → `path.read_text` →
# `known_checks`。
# 分支与异常：
#   当 `not path.is_file()` 时，返回 `None`。
#   验证条件：`not missing`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
def test_known_checks_cover_documented_names() -> None:
    path = CASES_DIR.parent / "README.md"
    if not path.is_file():  # 公开仓库按 .gitignore 不含 Markdown，这时跳过
        return
    readme = path.read_text(encoding="utf-8")
    missing = [name for name in known_checks() if f"`{name}`" not in readme]
    assert not missing, f"README 没有说明这些检查项：{missing}"
