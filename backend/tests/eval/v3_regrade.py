"""Regrade immutable V3 evidence; optionally verify hash-matched saved artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .outcome import Outcome
from .v1 import aggregate, source_metadata, write_json
from .v2_regrade import saved_checks
from .v2_report import classify, gate, summarize
from .v3 import delivery_gate, load_report, quality_summary, render
from .v3_cases import GRADER_VERSION, v3_cases
from .v3_grading import grade_case_v3, grade_v3, runtime_state_v3
from .v3_judge import digest, make_packet, unknown
from .verification import verify


def restore_verified_files(evidence, verification, workspace):
    """Reject missing or changed bytes; restore Windows CRLF when its hash matches."""
    for name in verification.files:
        target = workspace / name
        if not target.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("artifact path escapes workspace")
        text = evidence.get("outcome", {}).get("files", {}).get(name)
        expected = evidence.get("after", {}).get(name, {}).get("sha256")
        if text is None or not expected:
            raise ValueError("missing saved artifact or hash: " + name)
        candidates = [text.encode(), text.replace("\n", "\r\n").encode()]
        data = next(
            (
                candidate
                for candidate in candidates
                if hashlib.sha256(candidate).hexdigest() == expected
            ),
            None,
        )
        if data is None:
            raise ValueError("saved artifact does not match original hash: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


async def regrade(source: Path, *, out: Path, verify_missing=False):
    source = source.resolve()
    original = load_report(source)
    if out.resolve().is_relative_to(source.parent):
        raise ValueError("output must be outside the original batch")
    plans = {p.case.id: p for p in v3_cases()}
    old_version = original["metadata"]["grader_version"]
    for identifier, expected in original["metadata"]["case_digests"].items():
        if (
            identifier not in plans
            or plans[identifier].digest(grader_version=old_version) != expected
        ):
            raise ValueError("task contract changed: " + identifier)
    report = copy.deepcopy(original)
    report.pop("comparison", None)
    folder = out / (
        original["batch_id"] + f"-regraded-{GRADER_VERSION}-{uuid4().hex[:6]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    hashes, changes, added = {}, [], []
    for row in report["attempts"]:
        path = Path(row["evidence"])
        path = (
            path.resolve() if path.is_absolute() else (source.parent / path).resolve()
        )
        if not path.is_relative_to(source.parent):
            raise ValueError("evidence must belong to the collection batch")
        raw = path.read_bytes()
        evidence = json.loads(raw)
        if evidence.get("case_digest") != row["case_digest"]:
            raise ValueError("source evidence/contract mismatch")
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        plan = plans[row["case_id"]]
        outcome = Outcome.model_validate(evidence["outcome"])
        receipt_path = path.parent / "verification.json"
        independent = None
        if receipt_path.is_file():
            independent = json.loads(receipt_path.read_text(encoding="utf-8"))
            hashes[str(receipt_path)] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()
        elif (
            verify_missing
            and plan.verification
            and runtime_state_v3(outcome, evidence) is None
        ):
            with tempfile.TemporaryDirectory(
                prefix="muharness-evidence-regrade-"
            ) as temp:
                workspace = Path(temp)
                try:
                    restore_verified_files(evidence, plan.verification, workspace)
                    independent = await verify(workspace, plan.verification)
                except ValueError as exc:
                    independent = {"status": "UNKNOWN", "reason": str(exc)}
            independent["verification_kind"] = "post_hoc_saved_artifacts"
            independent["verified_at"] = datetime.now(UTC).isoformat()
            receipt_path = (
                folder / f"{row['case_id']}-{row['attempt']:02d}" / "verification.json"
            )
            write_json(receipt_path, independent)
            added.append(str(receipt_path.resolve()))
        previous = {"status": row["status"], "checks": copy.deepcopy(row["checks"])}
        row["checks"] = saved_checks(
            plan,
            evidence,
            independent,
            runtime_classifier=runtime_state_v3,
            case_evaluator=grade_case_v3,
            observation_grader=grade_v3,
        )
        row["status"] = aggregate(row["checks"])
        row["failures"] = classify(row, evidence)
        row["source_case_digest"] = row["case_digest"]
        row["case_digest"] = plan.digest()
        row["evidence"] = str(path)
        if previous != {"status": row["status"], "checks": row["checks"]}:
            changes.append(
                {
                    "case_id": row["case_id"],
                    "attempt": row["attempt"],
                    "before": previous,
                    "after": {"status": row["status"], "checks": row["checks"]},
                }
            )
        packet = make_packet(evidence, row)
        row["source_quality"] = row.get("quality")
        row["quality"] = {
            "state": "NOT_REASSESSED",
            "rubrics": unknown("客观复核后未重新调用AI裁判"),
            "usage": None,
            "packet_sha256": digest(packet),
        }
        write_json(
            folder / f"{row['case_id']}-{row['attempt']:02d}" / "packet.json", packet
        )
        write_json(
            folder / f"{row['case_id']}-{row['attempt']:02d}" / "quality.json",
            row["quality"],
        )
    report["metadata"]["grader_version"] = GRADER_VERSION
    report["metadata"]["case_digests"] = {
        key: plans[key].digest() for key in original["metadata"]["case_ids"]
    }
    report["metadata"]["regrade"] = {
        "source_report": str(source),
        "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "collection_grader_version": old_version,
        "grading_source_sha256": source_metadata()["source_sha256"],
        "regraded_at": datetime.now(UTC).isoformat(),
        "inputs_verified": True,
        "evidence_sha256": hashes,
        "added_verifications": added,
        "source_quality_summary": original.get("quality_summary"),
        "note": "模型未重跑；原始报告、证据、采集条件和接口错误保留。补验收单独标注。",
    }
    report["batch_id"] = folder.name
    report["summary"] = summarize(report)
    report["quality_summary"] = quality_summary(report["attempts"])
    report["objective_gate"] = gate(report)
    report["gate"] = delivery_gate(report)
    report["baseline_eligible"] = report["summary"]["baseline_eligible"]
    write_json(folder / "report.json", report)
    write_json(folder / "changes.json", changes)
    (folder / "report.md").write_text(render(report), encoding="utf-8")
    return report, folder
