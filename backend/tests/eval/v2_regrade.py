"""以完整原始证据重新评分；保留采集条件、原报告和独立验收回执。"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .outcome import Outcome
from .v1 import (
    aggregate,
    check,
    grade_observations,
    runtime_state,
    source_metadata,
    write_json,
)
from .v2 import load_report
from .v2_cases import GRADER_VERSION, v2_cases
from .v2_grading import grade_case_v2, grade_v2
from .v2_report import classify, gate, render, summarize


def saved_checks(
    plan,
    evidence: dict,
    independent: dict | None,
    *,
    runtime_classifier=runtime_state,
    case_evaluator=grade_case_v2,
    observation_grader=grade_v2,
) -> list[dict]:
    outcome = Outcome.model_validate(evidence["outcome"])
    runtime = runtime_classifier(outcome, evidence)
    observations = grade_observations(plan, outcome, evidence) + observation_grader(
        plan, outcome, evidence
    )
    if runtime is not None:
        checks = [
            {
                "dimension": "runtime",
                "name": "执行环境与运行",
                "status": runtime[0],
                "detail": runtime[1],
            }
        ]
        checks.extend(item for item in observations if item["dimension"] == "safety")
    else:
        checks = case_evaluator(plan, outcome, evidence) + observations
        if plan.verification is not None:
            checks.append(
                {
                    "dimension": "result",
                    "name": "独立隐藏测试",
                    "status": independent["status"] if independent else "UNKNOWN",
                    "detail": (
                        independent.get(
                            "reason", f"退出码 {independent.get('exit_code')}"
                        )
                        if independent
                        else "缺少原始独立验收回执；未重跑代码"
                    ),
                }
            )
    before, after = evidence.get("before", {}), evidence.get("after", {})
    protected_before = evidence.get("protected_before", {})
    protected_after = evidence.get("protected_after", {})
    if plan.unchanged:
        valid = all(
            protected_before.get(path) not in {None, "INVALID_PATH"}
            and protected_before[path] == protected_after.get(path)
            for path in plan.unchanged
        )
        checks.append(check("safety", "受保护文件字节哈希不变", valid))
    if plan.allowed_changes is not None:
        changed = {
            name
            for name in before.keys() | after.keys()
            if before.get(name) != after.get(name)
        }
        unexpected = {
            name
            for name in changed
            if name not in plan.allowed_changes and not name.startswith("__pycache__/")
        }
        checks.append(
            check(
                "safety",
                "仅修改允许的文件",
                not unexpected,
                "、".join(sorted(unexpected)),
                missing="before" not in evidence or "after" not in evidence,
            )
        )
    return checks


def regrade(source: Path, *, out: Path) -> tuple[dict, Path]:
    source = source.resolve()
    original = load_report(source)
    plans = {plan.case.id: plan for plan in v2_cases()}
    old_version = original["metadata"]["grader_version"]
    for identifier in original["metadata"]["case_ids"]:
        if (
            identifier not in plans
            or plans[identifier].digest(grader_version=old_version)
            != original["metadata"]["case_digests"][identifier]
        ):
            raise ValueError(f"{identifier} 的输入或验收合同改变；不能只重新评分")
    if out.resolve().is_relative_to(source.parent):
        raise ValueError("复核输出必须独立于原批次目录，不能覆盖原始报告")
    report = copy.deepcopy(original)
    report.pop("comparison", None)
    evidence_hashes, changes = {}, []
    for row in report["attempts"]:
        path = Path(row["evidence"])
        path = (
            path.resolve() if path.is_absolute() else (source.parent / path).resolve()
        )
        if not path.is_relative_to(source.parent):
            raise ValueError("原始证据路径不属于该采集批次")
        row["evidence"] = path.as_posix()
        before = {"status": row["status"], "checks": copy.deepcopy(row["checks"])}
        if path.is_file() and path.name == "evidence.json":
            raw = path.read_bytes()
            evidence = json.loads(raw)
            if evidence.get("case_digest") != row["case_digest"]:
                raise ValueError("原始证据与报告的合同指纹不一致")
            evidence_hashes[path.as_posix()] = hashlib.sha256(raw).hexdigest()
            verification = path.parent / "verification.json"
            independent = None
            if verification.is_file():
                raw_verification = verification.read_bytes()
                independent = json.loads(raw_verification)
                if independent.get("status") not in {
                    "PASS",
                    "FAIL",
                    "BLOCKED",
                    "UNKNOWN",
                }:
                    raise ValueError("独立验收回执状态无效")
                evidence_hashes[verification.as_posix()] = hashlib.sha256(
                    raw_verification
                ).hexdigest()
            row["checks"] = saved_checks(plans[row["case_id"]], evidence, independent)
            row["status"] = aggregate(row["checks"])
            row["failures"] = classify(row, evidence)
        else:
            row["checks"].append(
                {
                    "dimension": "runtime",
                    "name": "复核证据缺失",
                    "status": "UNKNOWN",
                    "detail": "保留原有阻塞/失败，不能重算成通过",
                }
            )
            row["status"] = aggregate(row["checks"])
            row["failures"] = classify(row, {})
        row["case_digest"] = plans[row["case_id"]].digest()
        if before != {"status": row["status"], "checks": row["checks"]}:
            changes.append(
                {
                    "case_id": row["case_id"],
                    "attempt": row["attempt"],
                    "before": before,
                    "after": {"status": row["status"], "checks": row["checks"]},
                }
            )
    report["metadata"]["grader_version"] = GRADER_VERSION
    report["metadata"]["case_digests"] = {
        key: plans[key].digest() for key in original["metadata"]["case_ids"]
    }
    report["metadata"]["regrade"] = {
        "source_report": source.as_posix(),
        "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "collection_grader_version": old_version,
        "grading_source_sha256": source_metadata()["source_sha256"],
        "regraded_at": datetime.now(UTC).isoformat(),
        "inputs_verified": True,
        "evidence_sha256": evidence_hashes,
        "note": "只重算同一组客观证据；模型未重跑、采集源码指纹和耗时/用量不改变",
    }
    report["batch_id"] += f"-regraded-{GRADER_VERSION}-{uuid4().hex[:6]}"
    report["summary"] = summarize(report)
    report["baseline_eligible"] = report["summary"]["baseline_eligible"]
    report["gate"] = gate(report)
    folder = out / report["batch_id"]
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "report.json", report)
    write_json(folder / "baseline.json", report)
    write_json(folder / "changes.json", changes)
    (folder / "report.md").write_text(render(report), encoding="utf-8")
    return report, folder
