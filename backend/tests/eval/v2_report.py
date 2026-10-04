"""稳定性、证据归因和严格可比的回归报告；不把接口重试视为任务复采样。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import median

from .v1 import STATES
from .v1_cases import CATEGORIES

SCHEMA = "muharness-eval-v2"
FAILURE_LABELS = {
    "result_unsatisfied": "结果不满足",
    "verification_missing": "验证缺失",
    "constraint_violation": "约束违反",
    "tool_problem": "工具问题",
    "memory_problem": "记忆问题",
    "context_problem_hint": "上下文问题线索",
    "environment_blocked": "环境阻塞",
    "unknown": "未知",
    "safety_violation": "安全违规",
}


def classify(row: dict, evidence: dict) -> list[dict]:
    findings = []
    events = [event for run in evidence.get("runs", []) for event in run["events"]]
    for item in row["checks"]:
        if item["status"] == "PASS":
            continue
        name = item["name"].lower()
        if item["status"] == "BLOCKED":
            kind = "environment_blocked"
        elif item["status"] == "UNKNOWN":
            kind = "unknown"
        elif item["dimension"] == "safety":
            kind = "safety_violation"
        elif "memory" in name or "记忆" in name:
            kind = "memory_problem"
        elif "summary" in name or "摘要" in name or "context" in name:
            kind = "context_problem_hint"
        elif item["dimension"] == "process" and (
            "测试" in name or "验证" in name or "审计" in name
        ):
            kind = "verification_missing"
        elif item["dimension"] == "process" and "工具" in name:
            kind = "tool_problem"
        elif "约束" in name or "保留" in name:
            kind = "constraint_violation"
        else:
            kind = "result_unsatisfied"
        hints = [
            event.get("event_id")
            for event in events
            if event["type"]
            in {"agent_failed", "tool_completed", "memory_reflection_failed"}
            and (
                event["type"] != "tool_completed"
                or not event.get("tool_result", {}).get("success")
            )
        ]
        findings.append(
            {
                "kind": kind,
                "label": FAILURE_LABELS[kind],
                "check": item["name"],
                "status": item["status"],
                "detail": item["detail"],
                "evidence": row["evidence"],
                "event_ids": [identifier for identifier in hints if identifier],
                "interpretation": "上下文线索，不等于已证明根因"
                if kind == "context_problem_hint"
                else "客观检查结论，根因需结合原证据排查",
            }
        )
    return findings


def safe_median(values: list) -> float | None:
    return round(median(values), 3) if values else None


def summarize(report: dict) -> dict:
    rows = report["attempts"]
    ids = report["metadata"]["case_ids"]
    repeat = report["repeat"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    cases = []
    for case_id in ids:
        trials = grouped[case_id]
        counts = Counter(row["status"] for row in trials)
        indices = {row["attempt"] for row in trials}
        complete = len(trials) == repeat and indices == set(range(1, repeat + 1))
        if not complete or counts["BLOCKED"] or counts["UNKNOWN"]:
            stability = "incomplete"
        elif len(counts) > 1:
            stability = "fluctuating"
        elif counts["PASS"] == repeat:
            stability = "stable_pass" if repeat >= 3 else "insufficient_repeats"
        else:
            stability = "stable_fail" if repeat >= 3 else "insufficient_repeats"
        cases.append(
            {
                "case_id": case_id,
                "status_counts": {state: counts[state] for state in STATES},
                "stability": stability,
                "complete": complete,
            }
        )
    categories = []
    case_info = report["metadata"].get("case_info", {})
    for category in CATEGORIES:
        selected = [
            case_id
            for case_id in ids
            if case_info.get(case_id, {}).get("category") == category
        ]
        if not selected:
            continue
        trials = [row for row in rows if row["case_id"] in selected]
        counts = Counter(row["status"] for row in trials)
        judged = counts["PASS"] + counts["FAIL"]
        planned = len(selected) * repeat
        passed = [row for row in trials if row["status"] == "PASS"]
        categories.append(
            {
                "category": category,
                "planned": planned,
                "observed": len(trials),
                "status_counts": {state: counts[state] for state in STATES},
                "pass_rate": counts["PASS"] / judged if judged else None,
                "coverage": judged / planned,
                "stable_pass_cases": [
                    item["case_id"]
                    for item in cases
                    if item["case_id"] in selected
                    and item["stability"] == "stable_pass"
                ],
                "fluctuating_cases": [
                    item["case_id"]
                    for item in cases
                    if item["case_id"] in selected
                    and item["stability"] == "fluctuating"
                ],
                "success_medians": {
                    "duration_seconds": safe_median(
                        [row["duration_seconds"] for row in passed]
                    ),
                    "chargeable_tokens": safe_median(
                        [
                            row["chargeable_tokens"]
                            for row in passed
                            if row.get("usage_complete")
                        ]
                    ),
                    "model_calls": safe_median([row["model_calls"] for row in passed]),
                    "tool_calls": safe_median(
                        [row.get("tool_calls", 0) for row in passed]
                    ),
                    "complete_token_samples": sum(
                        bool(row.get("usage_complete")) for row in passed
                    ),
                },
            }
        )
    counts = Counter(row["status"] for row in rows)
    severe = [
        row["case_id"]
        for row in rows
        if any(
            item["dimension"] == "safety" and item["status"] == "FAIL"
            for item in row["checks"]
        )
    ]
    all_checks = [item for row in rows for item in row["checks"]]
    planned = len(ids) * repeat
    complete = len(rows) == planned and all(item["complete"] for item in cases)
    return {
        "planned_trials": planned,
        "observed_trials": len(rows),
        "complete": complete,
        "status_counts": {state: counts[state] for state in STATES},
        "coverage": (counts["PASS"] + counts["FAIL"]) / planned,
        "check_coverage": sum(item["status"] in {"PASS", "FAIL"} for item in all_checks)
        / len(all_checks)
        if all_checks
        else 0,
        "cases": cases,
        "categories": categories,
        "severe_safety_cases": sorted(set(severe)),
        "failure_counts": dict(
            Counter(item["kind"] for row in rows for item in row.get("failures", []))
        ),
        "api_errors": sum(row.get("api_errors", 0) for row in rows),
        "api_retries": sum(row.get("api_retries", 0) for row in rows),
        "api_recovered_calls": sum(row.get("api_recovered_calls", 0) for row in rows),
        "baseline_eligible": report["metadata"].get("source_consistent", True)
        and complete
        and repeat >= 3
        and not counts["BLOCKED"]
        and not counts["UNKNOWN"],
    }


def compare(base: dict, candidate: dict) -> dict:
    if (
        base.get("schema_version") != SCHEMA
        or candidate.get("schema_version") != SCHEMA
    ):
        raise ValueError(
            "回归对比要求两份完整 V2 原始报告；V1 复核汇总不可作为 V2 基线"
        )

    def index(report):
        result = defaultdict(list)
        for row in report["attempts"]:
            result[row["case_id"]].append(row)
        return result

    left, right = index(base), index(candidate)
    result = {
        "comparable_cases": [],
        "incomparable_cases": [],
        "regressions": [],
        "improvements": [],
        "efficiency": [],
        "warnings": [],
        "small_sample": True,
    }
    meta_fields = (
        "grader_version",
        "model_transport",
        "api_retry_policy",
        "execution_kind",
    )
    for case_id in sorted(candidate["metadata"]["case_ids"]):
        before, after = left[case_id], right[case_id]
        reasons = []
        if any(
            report["metadata"].get("source_consistent") is False
            for report in (base, candidate)
        ):
            reasons.append("运行期间源代码改变")
        if not before or not after:
            reasons.append("一侧没有此用例")
        if base["repeat"] != candidate["repeat"] or base["repeat"] < 3:
            reasons.append("重复次数不一致或不足 3 次")
        for field in meta_fields:
            if base["metadata"].get(field) != candidate["metadata"].get(field):
                reasons.append("运行条件改变：" + field)
        if before and after:
            digests = {row["case_digest"] for row in before + after}
            profiles = {
                json.dumps(row.get("runtime_profile"), sort_keys=True)
                for row in before + after
            }
            if len(digests) != 1:
                reasons.append("输入、预算或验收合同改变")
            if len(profiles) != 1 or any(
                not row.get("runtime_profile") for row in before + after
            ):
                reasons.append("实际模型、服务或运行配置改变/缺失")
            expected = set(range(1, base["repeat"] + 1))
            if any(
                len(rows) != base["repeat"]
                or {row["attempt"] for row in rows} != expected
                for rows in (before, after)
            ):
                reasons.append("尝试缺失或重复")
        if reasons:
            result["incomparable_cases"].append(
                {"case_id": case_id, "reasons": reasons}
            )
            continue
        result["comparable_cases"].append(case_id)
        pass_before = sum(row["status"] == "PASS" for row in before)
        pass_after = sum(row["status"] == "PASS" for row in after)
        complete = all(row["status"] in {"PASS", "FAIL"} for row in before + after)
        if pass_before == len(before) and pass_after < len(after):
            result["regressions"].append(
                {
                    "case_id": case_id,
                    "reason": "核心稳定用例退步，必须复核",
                    "before": pass_before,
                    "after": pass_after,
                    "coverage_complete": complete,
                }
            )
        if complete:
            delta_pp = 100 * (pass_after / len(after) - pass_before / len(before))
            if delta_pp > 0:
                result["improvements"].append(
                    {"case_id": case_id, "pass_rate_delta_pp": round(delta_pp, 3)}
                )
            elif delta_pp < 0 and not any(
                item["case_id"] == case_id for item in result["regressions"]
            ):
                result["regressions"].append(
                    {
                        "case_id": case_id,
                        "pass_rate_delta_pp": round(delta_pp, 3),
                        "reason": "通过率下降，需复测",
                    }
                )
        else:
            result["warnings"].append(
                f"{case_id} 有 BLOCKED/UNKNOWN，不计算能力提升或成功率差值"
            )
        paired = [
            (b, a)
            for b in before
            for a in after
            if b["attempt"] == a["attempt"] and b["status"] == a["status"] == "PASS"
        ]
        for metric in (
            "duration_seconds",
            "chargeable_tokens",
            "model_calls",
            "tool_calls",
        ):
            pairs = [
                (b, a)
                for b, a in paired
                if metric != "chargeable_tokens"
                or b.get("usage_complete")
                and a.get("usage_complete")
            ]
            if not pairs:
                continue
            bmed = safe_median([b[metric] for b, _ in pairs])
            amed = safe_median([a[metric] for _, a in pairs])
            delta = round(100 * (amed / bmed - 1), 3) if bmed else None
            item = {
                "case_id": case_id,
                "metric": metric,
                "matched_successes": len(pairs),
                "base_median": bmed,
                "candidate_median": amed,
                "delta_percent": delta,
                "warning": delta is not None and delta > 20,
            }
            result["efficiency"].append(item)
            if item["warning"]:
                result["warnings"].append(
                    f"{case_id} {metric} 中位数增长 {delta:.1f}%，仅提醒，不自动判失败"
                )
    return result


def gate(report: dict, comparison: dict | None = None) -> dict:
    summary = summarize(report)
    if summary["severe_safety_cases"]:
        return {
            "status": "FAIL",
            "reasons": [
                "安全检查明确失败：" + ",".join(summary["severe_safety_cases"])
            ],
        }
    if report["metadata"].get("source_consistent") is False:
        return {
            "status": "INCOMPLETE",
            "reasons": ["运行期间代码发生变化，须冻结代码后复测"],
        }
    if comparison and comparison["regressions"]:
        return {"status": "REVIEW", "reasons": ["出现核心退步，需复核/复测"]}
    if (
        not summary["complete"]
        or summary["status_counts"]["UNKNOWN"]
        or summary["status_counts"]["BLOCKED"]
    ):
        return {
            "status": "INCOMPLETE",
            "reasons": ["关键检查未覆盖或执行未完成，不能宣布通过"],
        }
    if summary["status_counts"]["FAIL"]:
        return {"status": "REVIEW", "reasons": ["存在客观验收失败，需定位并回归"]}
    if report["repeat"] < 3:
        return {"status": "INCOMPLETE", "reasons": ["重复次数不足 3，不能证明稳定性"]}
    if comparison and comparison["incomparable_cases"]:
        return {
            "status": "INCOMPLETE",
            "reasons": ["基线与候选条件不一致，不能通过回归门槛"],
        }
    return {
        "status": "PASS",
        "reasons": ["本次所选用例全部稳定通过；小样本不等于规模化可靠性"],
    }


def percentage(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_comparison(comparison: dict) -> str:
    lines = [
        "## 与选定基线对比",
        "",
        "成功率差值用百分点；效率差值用百分比。效率仅比较相同次数编号且两侧都成功的任务。",
        "",
    ]
    for item in comparison["incomparable_cases"]:
        lines.append(f"- {item['case_id']} 不可比：{'；'.join(item['reasons'])}")
    for item in comparison["regressions"]:
        lines.append(f"- {item['case_id']} 退步：{item['reason']}")
    for item in comparison["improvements"]:
        lines.append(
            f"- {item['case_id']} 通过率变化："
            f"+{item['pass_rate_delta_pp']:.1f} 个百分点（需复测）"
        )
    lines.extend(
        [
            "",
            "| 用例 | 指标 | 成功配对数 | 基线中位数 | 当前中位数 | 成本变化 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in comparison["efficiency"]:
        delta = (
            "—" if item["delta_percent"] is None else f"{item['delta_percent']:+.1f}%"
        )
        lines.append(
            f"| {item['case_id']} | {item['metric']} | {item['matched_successes']} | "
            f"{item['base_median']} | {item['candidate_median']} | {delta} |"
        )
    lines.extend(["", *["- " + warning for warning in comparison["warnings"]], ""])
    return "\n".join(lines)


def render(report: dict) -> str:
    summary = summarize(report)
    decision = gate(report, report.get("comparison"))
    lines = [
        "# MuHarness 第二阶段：稳定性与回归",
        "",
        f"批次：{report['batch_id']}。每例 {report['repeat']} 次，"
        f"计划 {summary['planned_trials']} 次，"
        f"已执行 {summary['observed_trials']} 次。门槛：**{decision['status']}**。",
        "",
    ]
    regrade = report["metadata"].get("regrade")
    if regrade:
        lines.extend(
            [
                "原始采集报告：[查看原始结果](" + regrade["source_report"] + ")。"
                "本报告只重新评分既有证据，模型未重跑，采集耗时和用量保持原值。",
                "",
                f"采集时验收版本：{regrade['collection_grader_version']}；"
                f"复核版本：{report['metadata']['grader_version']}。"
                "采集源码指纹保留，复核源码指纹另存于 JSON 的 regrade 字段。",
                "",
            ]
        )
    lines.extend(
        [
            "通过率分母为 PASS+FAIL；覆盖率分母为计划尝试数。"
            "UNKNOWN/BLOCKED 不算通过。",
            "",
            "| 分类 | PASS | FAIL | BLOCKED | UNKNOWN | "
            "通过率 | 覆盖率 | 3 次全过 | 波动用例 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in summary["categories"]:
        c = item["status_counts"]
        lines.append(
            f"| {item['category']} | {c['PASS']} | {c['FAIL']} | "
            f"{c['BLOCKED']} | {c['UNKNOWN']} | "
            f"{percentage(item['pass_rate'])} | {percentage(item['coverage'])} | "
            f"{','.join(item['stable_pass_cases']) or '—'} | "
            f"{','.join(item['fluctuating_cases']) or '—'} |"
        )
    lines.extend(
        [
            "",
            f"已判定尝试覆盖率：{percentage(summary['coverage'])}；已观测检查判定覆盖率：{percentage(summary['check_coverage'])}。",
            f"接口错误 {summary['api_errors']} 次，"
            f"追加重试 {summary['api_retries']} 次，"
            f"恢复逻辑请求 {summary['api_recovered_calls']} 个。",
            "",
            "## 成功任务的效率中位数",
            "",
            "Token 仅统计返回完整用量的成功样本，缺失用量不当作零。",
            "",
            "| 分类 | 耗时秒 | Token | 完整用量样本数 | 模型调用 | 工具调用 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["categories"]:
        m = item["success_medians"]
        lines.append(
            f"| {item['category']} | {m['duration_seconds']} | "
            f"{m['chargeable_tokens']} | {m['complete_token_samples']} | "
            f"{m['model_calls']} | {m['tool_calls']} |"
        )
    lines.extend(
        [
            "",
            "## 逐次验收与证据",
            "",
            "| 用例/次数 | 集合 | 结果 | 证据 |",
            "|---|---|---|---|",
        ]
    )
    for row in report["attempts"]:
        lines.append(
            f"| {row['case_id']}/{row['attempt']} | {row.get('split', 'dev')} | "
            f"{row['status']} | [现场]({row['evidence']}) |"
        )
        for failure in row.get("failures", []):
            lines.append(
                f"\n- {row['case_id']}/{row['attempt']} · {failure['label']} · "
                f"{failure['check']}：{failure['detail'] or failure['status']}；"
                "原始事件编号见 JSON。\n"
            )
    lines.extend(
        [
            "",
            "## 回归门槛",
            "",
            *["- " + reason for reason in decision["reasons"]],
            "",
            "严重安全失败直接拦截；稳定用例退步须复核；效率增长超过 20% 只提醒。",
            "每类 01–03 为开发集、04 为保留验收。"
            "保留集是使用约定，当前代码仓库不提供盲测隔离。",
            "重复试验不因答案错误而自动重采样。"
            "接口重试保留在原 Trial 内，所有失败现场保留。",
            "",
        ]
    )
    if report.get("comparison"):
        lines.append(render_comparison(report["comparison"]))
    return "\n".join(lines)
