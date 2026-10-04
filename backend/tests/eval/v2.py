"""V2 入口：24 例 × 默认 3 次，隔离试验、稳定性统计及显式基线回归。"""

from __future__ import annotations

import json
from pathlib import Path

from .spec import CURRENT, Variant
from .stage import AppFactory
from .v1 import run_trial, run_v1, source_metadata, write_json
from .v2_cases import GRADER_VERSION, V2Case, v2_cases
from .v2_driver import make_driver
from .v2_grading import grade_case_v2, grade_v2
from .v2_report import SCHEMA, compare, gate, render, render_comparison, summarize

REPORTS_DIR = Path(__file__).resolve().parent / "reports" / "v2"


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA or not report.get("finished_at"):
        raise ValueError("请选择完成执行的 V2 原始 report.json 或 baseline.json")
    if not report.get("metadata", {}).get("case_ids") or not report.get("attempts"):
        raise ValueError("V2 报告缺少用例或尝试")
    seen = set()
    if len(set(report["metadata"]["case_ids"])) != len(report["metadata"]["case_ids"]):
        raise ValueError("报告的用例编号重复")
    for row in report["attempts"]:
        key = (row["case_id"], row["attempt"])
        if key in seen or row["case_id"] not in report["metadata"]["case_ids"]:
            raise ValueError("报告包含重复或未声明尝试")
        if row["case_digest"] != report["metadata"]["case_digests"].get(row["case_id"]):
            raise ValueError("报告的用例指纹不一致")
        if row["status"] not in {"PASS", "FAIL", "BLOCKED", "UNKNOWN"}:
            raise ValueError("报告包含未知的验收状态")
        if not 1 <= row["attempt"] <= report["repeat"]:
            raise ValueError("报告的尝试次数编号越界")
        seen.add(key)
    return report


async def run_v2(
    *,
    plans: list[V2Case] | None = None,
    factory: AppFactory | None = None,
    provider: str | None = None,
    model: str | None = None,
    repeat: int = 3,
    non_stream: bool = False,
    api_retries: int = 3,
    out: Path = REPORTS_DIR,
    variant: Variant = CURRENT,
    baseline: Path | None = None,
    prerequisite: dict | None = None,
    progress=print,
) -> tuple[dict, Path]:
    reference = load_report(baseline) if baseline is not None else None
    plans = v2_cases() if plans is None else plans
    by_id = {plan.case.id: plan for plan in plans}
    if len(by_id) != len(plans):
        raise ValueError("V2 用例编号重复")
    driver = make_driver(non_stream=non_stream, api_retries=api_retries)

    async def trial(plan, **kwargs):
        row = await run_trial(
            plan,
            **kwargs,
            driver=driver,
            observation_grader=grade_v2,
            case_evaluator=grade_case_v2,
        )
        evidence_path = kwargs["folder"] / "evidence.json"
        evidence = (
            json.loads(evidence_path.read_text(encoding="utf-8"))
            if evidence_path.is_file()
            else {}
        )
        if not evidence_path.is_file():
            row["evidence"] = kwargs["folder"].name + "/driver-error.json"
        row["runtime_profile"] = (
            evidence.get("outcome", {})
            .get("diagnostics", {})
            .get("runtime_profile", {})
        )
        row["split"] = plan.split
        from .v2_report import classify

        row["failures"] = classify(row, evidence)
        write_json(kwargs["folder"] / "result.json", row)
        return row

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
                plan.case.id: {
                    "category": plan.category,
                    "split": plan.split,
                    "scenario": plan.scenario,
                }
                for plan in plans
            },
        },
    )
    report["metadata"].update(
        {
            "execution_kind": "live" if factory is None else "injected_factory",
            "case_info": {
                plan.case.id: {
                    "category": plan.category,
                    "split": plan.split,
                    "scenario": plan.scenario,
                }
                for plan in plans
            },
            "split_protocol": "每类 01–03 开发，04 保留验收；保留集未做盲测隔离",
            "baseline_source": str(baseline.resolve()) if baseline else None,
        }
    )
    report["metadata"]["source_sha256_at_finish"] = source_metadata()["source_sha256"]
    report["metadata"]["source_consistent"] = (
        report["metadata"]["source_sha256"]
        == report["metadata"]["source_sha256_at_finish"]
    )
    # Docker 阻塞项没有进入 trial；也补充完整的归因，不能静默从分母删除。
    from .v2_report import classify

    for row in report["attempts"]:
        plan = by_id[row["case_id"]]
        row.setdefault("split", plan.split)
        row.setdefault("runtime_profile", {})
        row.setdefault("tool_calls", 0)
        row.setdefault("failures", classify(row, {}))
        write_json(
            folder / f"{row['case_id']}-{row['attempt']:02d}" / "result.json", row
        )
    report["summary"] = summarize(report)
    report["baseline_eligible"] = report["summary"]["baseline_eligible"]
    if reference is not None:
        report["comparison"] = compare(reference, report)
        write_json(folder / "comparison.json", report["comparison"])
        (folder / "comparison.md").write_text(
            render_comparison(report["comparison"]), encoding="utf-8"
        )
    report["gate"] = gate(report, report.get("comparison"))
    write_json(folder / "report.json", report)
    write_json(folder / "baseline.json", report)
    (folder / "report.md").write_text(render(report), encoding="utf-8")
    return report, folder


def exit_code(report: dict) -> int:
    decision = gate(report, report.get("comparison"))["status"]
    return 0 if decision == "PASS" else 2 if decision == "INCOMPLETE" else 1
