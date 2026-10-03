"""汇总与对照报告。

一次评测输出一个 JSON（机器可读，用来对照）和一个 Markdown（人读）。
对照报告按用例 id 对齐两次评测；用例内容的摘要不同时会标出来，那一行不可比。
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .checks import Verdict
from .outcome import Outcome
from .spec import Case

REPORT_FORMAT = 1


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    suite: str
    title: str
    digest: str
    attempt: int
    status: str
    error: str | None = None
    passed: bool
    verdicts: list[dict[str, Any]]
    duration_seconds: float
    chargeable_tokens: int
    model_calls: int
    outcome: dict[str, Any] | None = None  # 完整 Outcome，用 --keep-outcomes 时保存


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: int = REPORT_FORMAT
    variant: str
    variant_description: str = ""
    provider: str | None = None
    model: str | None = None
    repeat: int
    started_at: str
    finished_at: str | None = None
    attempts: list[AttemptRecord] = Field(default_factory=list)

    # 函数说明：RunReport.save
    # 用途：保存RunReport，供回归测试与测试辅助使用。
    # 参数：
    #   directory：目录输入或配置值，类型 `Path`。
    # 返回：类型 `tuple[Path, Path]`；返回 `(json_path, md_path)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`directory.mkdir` →
    # `self.started_at.replace(':', '').replace` → `self.started_at.replace` →
    # `base.with_suffix` → `json_path.write_text` → `self.model_dump_json`；另有 2 个调
    # 用点。
    # 副作用与资源：
    #   文件或资源访问：`directory.mkdir`、`json_path.write_text`、`md_path.write_text`
    # 。
    def save(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.replace(":", "").replace("-", "")[:15]
        base = directory / f"{stamp}-{self.variant}"
        json_path = base.with_suffix(".json")
        md_path = base.with_suffix(".md")
        json_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(render_run(self), encoding="utf-8")
        return json_path, md_path

    # 函数说明：RunReport.load
    # 用途：加载RunReport，供回归测试与测试辅助使用。
    # 参数：
    #   path：目标文件或目录路径，类型 `Path`。
    # 返回：类型 `RunReport`；返回
    # `cls.model_validate(json.loads(path.read_text(encoding='utf-8')))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`cls.model_validate` →
    # `json.loads` → `path.read_text`。
    # 副作用与资源：
    #   文件或资源访问：`path.read_text`。
    @classmethod
    def load(cls, path: Path) -> RunReport:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


# 函数说明：now_iso
# 用途：返回 `datetime.now(UTC).isoformat(timespec='seconds')`，提供 回归测试与测试辅助
# 的派生值。
# 返回：类型 `str`；返回 `datetime.now(UTC).isoformat(timespec='seconds')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`datetime.now(UTC).isoformat` →
# `datetime.now`。
def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# 函数说明：record
# 用途：记录回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   case：`case`输入或配置值，类型 `Case`。
#   outcome：执行或验收结果，类型 `Outcome`。
#   verdicts：`verdicts`输入或配置值，类型 `list[Verdict]`。
#   keep_outcome：`keep_outcome`输入或配置值，类型 `bool`。
# 返回：类型 `AttemptRecord`；返回 `AttemptRecord(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AttemptRecord` → `case.digest`。
def record(case: Case, outcome: Outcome, verdicts: list[Verdict], *, keep_outcome: bool) -> AttemptRecord:
    return AttemptRecord(
        case_id=case.id,
        suite=case.suite,
        title=case.title,
        digest=case.digest(),
        attempt=outcome.attempt,
        status=outcome.status,
        error=outcome.error,
        passed=outcome.status == "ok" and all(v.passed for v in verdicts),
        verdicts=[{"check": v.check, "passed": v.passed, "detail": v.detail} for v in verdicts],
        duration_seconds=outcome.duration_seconds,
        chargeable_tokens=outcome.chargeable_tokens,
        model_calls=outcome.model_calls,
        outcome=outcome.model_dump(mode="json") if keep_outcome else None,
    )


@dataclass(slots=True)
class CaseSummary:
    case_id: str
    suite: str
    title: str
    digest: str
    attempts: int = 0
    passes: int = 0
    skipped: bool = False
    tokens: list[int] = field(default_factory=list)
    calls: list[int] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    # 函数说明：CaseSummary.rate
    # 用途：返回 `self.passes / self.attempts if self.attempts else 0.0`，提供
    # CaseSummary 的派生值。
    # 返回：类型 `float`；返回 `self.passes / self.attempts if self.attempts else 0.0`。
    @property
    def rate(self) -> float:
        return self.passes / self.attempts if self.attempts else 0.0

    # 函数说明：CaseSummary.mark
    # 用途：标记CaseSummary，供回归测试与测试辅助使用。
    # 返回：类型 `str`；按分支返回 `'⏭'`；`'✅'`；`'⚠️' if self.passes else '❌'`。
    # 分支与异常：
    #   当 `self.skipped` 时，返回 `'⏭'`。
    #   当 `self.passes == self.attempts` 时，返回 `'✅'`。
    @property
    def mark(self) -> str:
        if self.skipped:
            return "⏭"
        if self.passes == self.attempts:
            return "✅"
        return "⚠️" if self.passes else "❌"

    # 函数说明：CaseSummary.median_tokens
    # 用途：返回 `int(statistics.median(self.tokens)) if self.tokens else 0`，提供
    # CaseSummary 的派生值。
    # 返回：类型 `int`；返回 `int(statistics.median(self.tokens)) if self.tokens else 0`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`statistics.median`。
    @property
    def median_tokens(self) -> int:
        return int(statistics.median(self.tokens)) if self.tokens else 0

    # 函数说明：CaseSummary.median_calls
    # 用途：返回 `statistics.median(self.calls) if self.calls else 0`，提供 CaseSummary
    # 的派生值。
    # 返回：类型 `float`；返回 `statistics.median(self.calls) if self.calls else 0`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`statistics.median`。
    @property
    def median_calls(self) -> float:
        return statistics.median(self.calls) if self.calls else 0

    # 函数说明：CaseSummary.median_seconds
    # 用途：返回 `round(statistics.median(self.seconds), 1) if self.seconds else 0.0`，
    # 提供 CaseSummary 的派生值。
    # 返回：类型 `float`；返回
    # `round(statistics.median(self.seconds), 1) if self.seconds else 0.0`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`round` → `statistics.median`。
    @property
    def median_seconds(self) -> float:
        return round(statistics.median(self.seconds), 1) if self.seconds else 0.0


# 函数说明：summarize
# 用途：生成摘要回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   report：报告输入或配置值，类型 `RunReport`。
# 返回：类型 `dict[str, CaseSummary]`；返回 `summaries`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`summaries.setdefault` → `CaseSummary`
#  → `next`。
# 分支与异常：
#   `item.status == 'skipped'` 分支在完成前置处理后跳过当前循环项。
# 副作用与资源：
#   更新对象字段：`summary.skipped`、`summary.attempts`、`summary.passes`。
def summarize(report: RunReport) -> dict[str, CaseSummary]:
    summaries: dict[str, CaseSummary] = {}
    for item in report.attempts:
        summary = summaries.setdefault(
            item.case_id, CaseSummary(item.case_id, item.suite, item.title, item.digest)
        )
        if item.status == "skipped":
            summary.skipped = True
            summary.failures.append(item.error or "跳过")
            continue
        summary.attempts += 1
        summary.passes += int(item.passed)
        summary.tokens.append(item.chargeable_tokens)
        summary.calls.append(item.model_calls)
        summary.seconds.append(item.duration_seconds)
        if not item.passed:
            reason = item.error or next(
                (f"{v['check']}：{v['detail']}" for v in item.verdicts if not v["passed"]), "未知"
            )
            summary.failures.append(f"第 {item.attempt} 次：{reason}")
    return summaries


# 函数说明：_suite_rows
# 用途：在回归测试与测试辅助中处理 `_suite_rows`，通过 `summaries.values` 完成首个内部处
# 理步骤。
# 参数：
#   summaries：`summaries`输入或配置值，类型 `dict[str, CaseSummary]`。
# 返回：类型 `list[tuple[str, int, int, int, int]]`；返回
# `[(suite, *values) for suite, values in sorted(suites.items())]`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`suites.setdefault`。
# 分支与异常：
#   当 `summary.skipped` 时，跳过当前循环项。
def _suite_rows(summaries: dict[str, CaseSummary]) -> list[tuple[str, int, int, int, int]]:
    suites: dict[str, list[int]] = {}
    for summary in summaries.values():
        if summary.skipped:
            continue
        row = suites.setdefault(summary.suite, [0, 0, 0, 0])
        row[0] += summary.passes
        row[1] += summary.attempts
        row[2] += int(summary.passes == summary.attempts)
        row[3] += 1
    return [(suite, *values) for suite, values in sorted(suites.items())]


# 函数说明：render_run
# 用途：生成展示文本运行，供回归测试与测试辅助使用。
# 参数：
#   report：传给 `summarize` 的输入，类型 `RunReport`。
# 返回：类型 `str`；返回 `'\n'.join(lines).rstrip() + '\n'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`summarize` → `_suite_rows` → `_pct` →
#  `'\n'.join(lines).rstrip`。
def render_run(report: RunReport) -> str:
    summaries = summarize(report)
    lines = [
        f"# 评测报告：{report.variant}",
        "",
        f"- 变体：{report.variant}{' — ' + report.variant_description if report.variant_description else ''}",
        f"- 模型：{report.provider or '默认'} / {report.model or '默认'}",
        f"- 每个用例重复：{report.repeat} 次",
        f"- 时间：{report.started_at} → {report.finished_at or '未完成'}",
        "",
        "## 按套件",
        "",
        "| 套件 | 通过次数 | 稳定通过的用例 |",
        "| --- | ---: | ---: |",
    ]
    total_pass = total_runs = 0
    for suite, passes, runs, stable, cases in _suite_rows(summaries):
        total_pass += passes
        total_runs += runs
        lines.append(f"| {suite} | {passes}/{runs}（{_pct(passes, runs)}） | {stable}/{cases} |")
    lines += [
        "",
        f"总计 {total_pass}/{total_runs}（{_pct(total_pass, total_runs)}）。"
        "✅ 每次都通过，⚠️ 时过时不过，❌ 从未通过，⏭ 跳过。",
        "",
        "## 按用例",
        "",
        "| 用例 | 结果 | Token（中位数） | 模型调用 | 耗时 s |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for summary in sorted(summaries.values(), key=lambda s: (s.suite, s.case_id)):
        result = "跳过" if summary.skipped else f"{summary.mark} {summary.passes}/{summary.attempts}"
        lines.append(
            f"| {summary.case_id}<br>{summary.title} | {result} | {summary.median_tokens} "
            f"| {summary.median_calls:g} | {summary.median_seconds:g} |"
        )
    failing = [s for s in summaries.values() if s.failures]
    if failing:
        lines += ["", "## 没通过的原因", ""]
        for summary in sorted(failing, key=lambda s: s.case_id):
            lines.append(f"### {summary.case_id}")
            lines += [f"- {reason}" for reason in summary.failures]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# 函数说明：render_comparison
# 用途：生成展示文本`comparison`，供回归测试与测试辅助使用。
# 参数：
#   base：传给 `summarize` 的输入，类型 `RunReport`。
#   other：传给 `summarize` 的输入，类型 `RunReport`。
# 返回：类型 `str`；返回 `'\n'.join(lines) + '\n'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`summarize` → `_delta_pct` → `_pct`。
# 分支与异常：
#   当 `a.skipped or b.skipped` 时，跳过当前循环项。
#   `a.digest != b.digest` 分支在完成前置处理后跳过当前循环项。
def render_comparison(base: RunReport, other: RunReport) -> str:
    left, right = summarize(base), summarize(other)
    shared = sorted(set(left) & set(right))
    lines = [
        f"# 对照：{base.variant} → {other.variant}",
        "",
        f"- 基准：{base.variant}（{base.provider or '默认'} / {base.model or '默认'}，{base.started_at}）",
        f"- 对照：{other.variant}（{other.provider or '默认'} / {other.model or '默认'}，{other.started_at}）",
        "",
        "| 用例 | 基准通过 | 对照通过 | 变化 | 基准 Token | 对照 Token | Token 变化 |",
        "| --- | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    better: list[str] = []
    worse: list[str] = []
    incomparable: list[str] = []
    sums = [0, 0, 0, 0, 0, 0]  # 基准通过、基准次数、对照通过、对照次数、基准 Token、对照 Token
    for case_id in shared:
        a, b = left[case_id], right[case_id]
        if a.skipped or b.skipped:
            continue
        if a.digest != b.digest:
            incomparable.append(case_id)
            continue
        delta = b.rate - a.rate
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        if delta > 0:
            better.append(case_id)
        elif delta < 0:
            worse.append(case_id)
        lines.append(
            f"| {case_id} | {a.passes}/{a.attempts} | {b.passes}/{b.attempts} | {arrow} "
            f"| {a.median_tokens} | {b.median_tokens} | {_delta_pct(a.median_tokens, b.median_tokens)} |"
        )
        for index, value in enumerate(
            (a.passes, a.attempts, b.passes, b.attempts, a.median_tokens, b.median_tokens)
        ):
            sums[index] += value
    lines += [
        "",
        f"通过率 {_pct(sums[0], sums[1])} → {_pct(sums[2], sums[3])}；"
        f"Token 中位数合计 {sums[4]} → {sums[5]}（{_delta_pct(sums[4], sums[5])}）。",
        "",
        f"- 变好：{'、'.join(better) or '无'}",
        f"- 变差：{'、'.join(worse) or '无'}",
    ]
    if incomparable:
        lines.append(f"- 用例内容已改动、不可比：{'、'.join(incomparable)}")
    only = sorted(set(left) ^ set(right))
    if only:
        lines.append(f"- 只在一侧出现：{'、'.join(only)}")
    lines += ["", "单个用例重复次数少时，一两次的差异可能只是随机波动，建议 --repeat 3 以上再下结论。"]
    return "\n".join(lines) + "\n"


# 函数说明：_pct
# 用途：返回 `f'{part / whole:.0%}' if whole else '—'`，提供 回归测试与测试辅助 的派生值
# 。
# 参数：
#   part：`part`输入或配置值，类型 `int`。
#   whole：`whole`输入或配置值，类型 `int`。
# 返回：类型 `str`；返回 `f'{part / whole:.0%}' if whole else '—'`。
def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "—"


# 函数说明：_delta_pct
# 用途：处理回归测试与测试辅助中的 `_delta_pct` 数据；结果及边界条件见下方说明。
# 参数：
#   before：`before`输入或配置值，类型 `int`。
#   after：`after`输入或配置值，类型 `int`。
# 返回：类型 `str`；按分支返回 `'—'`；`f'{(after - before) / before:+.0%}'`。
# 分支与异常：
#   当 `not before` 时，返回 `'—'`。
def _delta_pct(before: int, after: int) -> str:
    if not before:
        return "—"
    return f"{(after - before) / before:+.0%}"


__all__ = [
    "AttemptRecord",
    "CaseSummary",
    "RunReport",
    "now_iso",
    "record",
    "render_comparison",
    "render_run",
    "summarize",
]
