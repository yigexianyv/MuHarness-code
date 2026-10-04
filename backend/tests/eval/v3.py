"""第三阶段：36 项、独立质量评分与可追溯的改进流程。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .spec import CURRENT
from .v1 import run_trial, run_v1, source_metadata, write_json
from .v2_report import classify, compare, gate, summarize
from .v2_report import render as render_v2
from .v3_cases import GRADER_VERSION, v3_cases
from .v3_driver import make_driver
from .v3_grading import grade_case_v3, grade_v3, runtime_state_v3
from .v3_judge import RUBRICS, QualityJudge, digest, make_packet, unknown

SCHEMA = "muharness-eval-v3"
REPORTS_DIR = Path(__file__).resolve().parent / "reports" / "v3"


def quality_summary(rows):
    counts = {
        name: dict(
            Counter(
                row.get("quality", {})
                .get("rubrics", {})
                .get(name, {})
                .get("status", "UNKNOWN")
                for row in rows
            )
        )
        for name in RUBRICS
    }
    usages = [row.get("quality", {}).get("usage") for row in rows]
    return {
        "rubric_counts": counts,
        "calibration_status": "UNCALIBRATED",
        "model_calls": sum(u.get("model_calls", 0) for u in usages if u),
        "total_tokens": sum(u.get("total_tokens", 0) for u in usages if u),
        "usage_complete": bool(usages) and all(usages),
        "invalid": sum(
            row.get("quality", {}).get("state") == "INVALID" for row in rows
        ),
    }


def delivery_gate(report):
    objective = gate(report, report.get("comparison"))
    if objective["status"] in {"FAIL", "REVIEW"}:
        return objective
    return {
        "status": "INCOMPLETE",
        "reasons": objective["reasons"]
        + ["AI 质量评分未完成人工校准；客观结果保持独立"],
    }


def render(report):
    text = render_v2(report).replace(
        "第二阶段：稳定性与回归", "第三阶段：质量与持续优化"
    )
    quality = quality_summary(report["attempts"])
    text += (
        "\n## 回答质量（独立、尚未人工校准）\n\nAI 评分不修改客观结"
        "果；主模型和裁判可能同源，不宣称独立裁判可靠性。\n\n| 规则 |"
        " PASS | FAIL | UNKNOWN | NOT_APP"
        "LICABLE |\n|---|---:|---:|---:|--"
        "-:|\n"
    )
    for name, counts in quality["rubric_counts"].items():
        text += (
            f"| {name} | "
            + " | ".join(
                str(counts.get(s, 0))
                for s in ("PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE")
            )
            + " |\n"
        )
    text += (
        f"\n裁判模型调用 {quality['model_calls']}，"
        f"返回 total_tokens {quality['total_tokens']}；"
        f"费用与 Agent 分开记录，用量完整={quality['usage_complete']}。\n"
    )
    text += (
        "\n质量包 packet.json 和 quality.json 分别保存资料选择、"
        "证据引用、裁判原文、配置和用量。\n"
    )
    if report.get("gate"):
        text += (
            f"\n整体交付门槛：**{report['gate']['status']}**；顶部为客观任务门槛。\n"
        )
    return text


def load_report(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA or not report.get("finished_at"):
        raise ValueError("需要完成的 V3 report.json")
    ids = report.get("metadata", {}).get("case_ids", [])
    if not ids or len(ids) != len(set(ids)) or not report.get("attempts"):
        raise ValueError("V3报告用例缺失或重复")
    seen = set()
    for row in report["attempts"]:
        key = (row["case_id"], row["attempt"])
        if (
            key in seen
            or row["case_id"] not in ids
            or not 1 <= row["attempt"] <= report["repeat"]
        ):
            raise ValueError("V3尝试重复或未声明")
        if row["case_digest"] != report["metadata"]["case_digests"].get(row["case_id"]):
            raise ValueError("V3合同指纹不一致")
        if row["status"] not in {"PASS", "FAIL", "BLOCKED", "UNKNOWN"}:
            raise ValueError("未知V3客观状态")
        seen.add(key)
    return report


async def run_v3(
    *,
    plans=None,
    factory=None,
    provider=None,
    model=None,
    repeat=3,
    non_stream=False,
    api_retries=3,
    out=REPORTS_DIR,
    variant=CURRENT,
    baseline=None,
    prerequisite=None,
    judge=None,
    skip_judge=False,
    judge_provider=None,
    judge_model=None,
    progress=print,
):
    plans = v3_cases() if plans is None else plans
    if not plans or len({p.case.id for p in plans}) != len(plans):
        raise ValueError("V3 用例不能为空或重复")
    reference = load_report(baseline) if baseline else None
    owned = judge is None and not skip_judge
    if owned:
        judge = QualityJudge(
            provider=judge_provider or provider,
            model=judge_model,
            api_retries=api_retries,
        )
    driver = make_driver(non_stream=non_stream, api_retries=api_retries)

    async def trial(plan, **kwargs):
        row = await run_trial(
            plan,
            **kwargs,
            driver=driver,
            observation_grader=grade_v3,
            case_evaluator=grade_case_v3,
            runtime_classifier=runtime_state_v3,
        )
        path = kwargs["folder"] / "evidence.json"
        evidence = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        )
        row["runtime_profile"] = (
            evidence.get("outcome", {})
            .get("diagnostics", {})
            .get("runtime_profile", {})
        )
        row["split"] = plan.split
        if plan.case.id == "S06":
            row["usage_complete"] = False
        row["failures"] = classify(row, evidence)
        packet = make_packet(evidence, row, secrets=getattr(judge, "secrets", ()))
        if judge and evidence:
            row["quality"] = await judge.assess(packet)
        else:
            row["quality"] = {
                "state": "SKIPPED" if skip_judge else "MISSING_EVIDENCE",
                "rubrics": unknown("未执行AI评审或缺证据"),
                "usage": None,
                "packet_sha256": digest(packet),
            }
        write_json(
            kwargs["folder"] / "packet.json",
            packet,
            secrets=getattr(judge, "secrets", ()),
        )
        write_json(
            kwargs["folder"] / "quality.json",
            row["quality"],
            secrets=getattr(judge, "secrets", ()),
        )
        write_json(kwargs["folder"] / "result.json", row)
        return row

    try:
        report, folder = await run_v1(
            plans=plans,
            factory=factory,
            provider=provider,
            model=model,
            repeat=repeat,
            non_stream=non_stream,
            api_retries=api_retries,
            out=out,
            variant=variant,
            prerequisite=prerequisite,
            progress=progress,
            trial_runner=trial,
            schema_version=SCHEMA,
            grader_version=GRADER_VERSION,
            report_renderer=render,
            metadata_extra={
                "execution_kind": "live" if factory is None else "injected_factory",
                "case_info": {
                    p.case.id: {
                        "category": p.category,
                        "split": p.split,
                        "scenario": p.scenario,
                    }
                    for p in plans
                },
                "split_protocol": "01–04 开发，05–06 保留；未做盲测隔离",
                "judge_config": judge.config if judge else {"enabled": False},
            },
        )
    finally:
        if owned:
            await judge.close()
    report["metadata"]["source_sha256_at_finish"] = source_metadata()["source_sha256"]
    report["metadata"]["source_consistent"] = (
        report["metadata"]["source_sha256"]
        == report["metadata"]["source_sha256_at_finish"]
    )
    for row in report["attempts"]:
        row.setdefault("tool_calls", 0)
        row.setdefault("failures", classify(row, {}))
    report["summary"] = summarize(report)
    report["quality_summary"] = quality_summary(report["attempts"])
    if reference:
        report["comparison"] = compare(reference, report)
        write_json(folder / "comparison.json", report["comparison"])
    report["objective_gate"] = gate(report, report.get("comparison"))
    report["gate"] = delivery_gate(report)
    report["baseline_eligible"] = report["summary"]["baseline_eligible"]
    write_json(folder / "report.json", report)
    write_json(folder / "baseline.json", report)
    (folder / "report.md").write_text(render(report), encoding="utf-8")
    return report, folder


def exit_code(report):
    state = report["gate"]["status"]
    return 0 if state == "PASS" else 2 if state == "INCOMPLETE" else 1
