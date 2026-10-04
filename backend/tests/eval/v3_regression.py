"""真实失败留档及最小重现；不以重试成功覆盖原失败。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .v1 import _json_answer, write_json
from .v3_cases import v3_cases


def capture(source: Path, *, case_id: str, attempt: int, out: Path):
    if out.exists():
        raise ValueError("不能覆盖已保存回归现场")
    report = json.loads(source.read_text(encoding="utf-8"))
    row = next(
        (
            r
            for r in report["attempts"]
            if r["case_id"] == case_id and r["attempt"] == attempt
        ),
        None,
    )
    if row is None or row["status"] != "FAIL":
        raise ValueError("仅接收实际 FAIL，不把 UNKNOWN 或成功改造成真实失败")
    if not report.get("finished_at"):
        raise ValueError("待批次完成后再固化回归现场")
    path = Path(row["evidence"])
    if not path.is_absolute():
        path = source.parent / path
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("case_digest") != row["case_digest"]:
        raise ValueError("失败现场与原合同不一致")
    plan = next(p for p in v3_cases() if p.case.id == case_id)
    turns = evidence["outcome"]["turns"]
    bundle = {
        "kind": "observed_failure_regression",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_row": row,
        "source_evidence": str(path.resolve()),
        "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_case_digest": evidence["case_digest"],
        "current_case_digest": plan.digest(),
        "task": turns[-1]["say"],
        "observed_answer": turns[-1]["answer"],
        "reproduction_contract": plan.case.model_dump(mode="json"),
        "contract_changed": evidence["case_digest"] != plan.digest(),
        "source_grader_version": report["metadata"].get("grader_version"),
        "repeatable_command": f".\\evaluate.ps1 -Stage V3 -Case {case_id} -Repeat 3",
        "fix_status": "NOT_MODIFIED",
        "original_preserved": True,
    }
    if case_id == "A03":
        bundle["minimal_replay"] = {
            "check": "strict_json",
            "expected": {"owner": "林舟", "version": 7},
            "parsed": _json_answer(turns[-1]["answer"]),
            "expected_verdict": "FAIL",
        }
    write_json(out / "regression.json", bundle)
    write_json(out / "original-evidence.json", evidence)
    return bundle
