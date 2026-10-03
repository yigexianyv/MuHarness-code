"""报告与对照的汇总逻辑。不调用模型。"""

from __future__ import annotations

from tests.eval.report import (
    AttemptRecord,
    RunReport,
    render_comparison,
    render_run,
    summarize,
)


# 函数说明：_attempt
# 用途：返回 `AttemptRecord(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   case_id：`case`标识，类型 `str`。
#   attempt：`attempt`输入或配置值，类型 `int`。
#   passed：`passed`输入或配置值，类型 `bool`。
#   tokens：Token 用量输入或配置值，类型 `int`。
#   digest：`digest`输入或配置值，类型 `str`；默认 `'d1'`。
# 返回：类型 `AttemptRecord`；返回 `AttemptRecord(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AttemptRecord`。
def _attempt(case_id: str, attempt: int, passed: bool, tokens: int, digest: str = "d1") -> AttemptRecord:
    return AttemptRecord(
        case_id=case_id,
        suite="behavior",
        title=case_id,
        digest=digest,
        attempt=attempt,
        status="ok",
        passed=passed,
        verdicts=[{"check": "answer_has", "passed": passed, "detail": "命中" if passed else "一个都没出现：17"}],
        duration_seconds=1.0,
        chargeable_tokens=tokens,
        model_calls=2,
    )


# 函数说明：_report
# 用途：返回 `RunReport(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   variant：`variant`输入或配置值，类型 `str`。
#   attempts：`attempts`输入或配置值，类型 `list[AttemptRecord]`。
# 返回：类型 `RunReport`；返回 `RunReport(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RunReport`。
def _report(variant: str, attempts: list[AttemptRecord]) -> RunReport:
    return RunReport(variant=variant, repeat=2, started_at="2026-09-30T12:00:00+00:00", attempts=attempts)


# 函数说明：test_summary_marks_stable_flaky_and_failing
# 用途：回归验证回归测试与测试辅助中的 `summary_marks_stable_flaky_and_failing` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_report` → `_attempt` → `summarize` →
#  `render_run`。
# 分支与异常：
#   验证条件：`[summaries[c].mark for c in ('b01', 'b02', 'b03')] == ['✅', '⚠️', '❌']`
# 。
#   验证条件：`summaries['b01'].median_tokens == 110`。
#   验证条件：`'3/6（50%）' in text`。
#   验证条件：`'第 2 次：answer_has：一个都没出现：17' in text`。
def test_summary_marks_stable_flaky_and_failing() -> None:
    report = _report(
        "current",
        [
            _attempt("b01", 1, True, 100),
            _attempt("b01", 2, True, 120),
            _attempt("b02", 1, True, 100),
            _attempt("b02", 2, False, 300),
            _attempt("b03", 1, False, 50),
            _attempt("b03", 2, False, 50),
        ],
    )
    summaries = summarize(report)
    assert [summaries[c].mark for c in ("b01", "b02", "b03")] == ["✅", "⚠️", "❌"]
    assert summaries["b01"].median_tokens == 110
    text = render_run(report)
    assert "3/6（50%）" in text
    assert "第 2 次：answer_has：一个都没出现：17" in text


# 函数说明：test_comparison_lists_better_worse_and_incomparable
# 用途：回归验证回归测试与测试辅助中的 `comparison_lists_better_worse_and_incomparable`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_report` → `_attempt` →
# `render_comparison`。
# 分支与异常：
#   验证条件：`'- 变好：b01' in text`。
#   验证条件：`'- 变差：b02' in text`。
#   验证条件：`'不可比：b03' in text`。
#   验证条件：`'| b01 | 0/1 | 1/1 | ↑ | 200 | 100 | -50% |' in text`。
def test_comparison_lists_better_worse_and_incomparable() -> None:
    base = _report(
        "prompt-v1",
        [_attempt("b01", 1, False, 200), _attempt("b02", 1, True, 100), _attempt("b03", 1, True, 100, "old")],
    )
    other = _report(
        "current",
        [_attempt("b01", 1, True, 100), _attempt("b02", 1, False, 150), _attempt("b03", 1, True, 100, "new")],
    )
    text = render_comparison(base, other)
    assert "- 变好：b01" in text
    assert "- 变差：b02" in text
    assert "不可比：b03" in text
    assert "| b01 | 0/1 | 1/1 | ↑ | 200 | 100 | -50% |" in text


# 函数说明：test_report_round_trips_through_json
# 用途：回归验证回归测试与测试辅助中的 `report_round_trips_through_json` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_report` → `_attempt` → `report.save`
#  → `RunReport.load` → `md_path.read_text(encoding='utf-8').startswith` →
# `md_path.read_text`。
# 分支与异常：
#   验证条件：`RunReport.load(json_path) == report`。
#   验证条件：`md_path.read_text(encoding='utf-8').startswith('# 评测报告：current')`。
# 副作用与资源：
#   文件或资源访问：`md_path.read_text`。
def test_report_round_trips_through_json(tmp_path) -> None:
    report = _report("current", [_attempt("b01", 1, True, 100)])
    json_path, md_path = report.save(tmp_path)
    assert RunReport.load(json_path) == report
    assert md_path.read_text(encoding="utf-8").startswith("# 评测报告：current")
