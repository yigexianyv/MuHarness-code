"""真人标注导出及人机一致性；空白标注绝不能自动当作人类意见。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .v1 import write_json
from .v3_judge import RUBRICS, STATES, digest


def export_annotations(quality_report: Path, out: Path):
    if out.exists():
        raise ValueError("不能覆盖已有人工标注")
    report = json.loads(quality_report.read_text(encoding="utf-8"))
    samples = []
    for review in report["reviews"]:
        packet = json.loads(Path(review["packet"]).read_text(encoding="utf-8"))
        if digest(packet) != review["result"]["packet_sha256"]:
            raise ValueError("评审包已被修改")
        samples.append(
            {
                "sample_id": review["sample_id"],
                "packet_sha256": digest(packet),
                "category": review["category"],
                "origin": review.get("origin", "observed_agent"),
                "packet": review["packet"],
                "annotator": "",
                "rubrics": {
                    name: {"status": "", "reason": "", "event_ids": []}
                    for name in RUBRICS
                },
            }
        )
    payload = {
        "schema_version": "muharness-human-v3",
        "source": str(quality_report.resolve()),
        "rubric_version": report["reviews"][0]["result"]["judge_config"][
            "rubric_version"
        ]
        if samples
        else None,
        "instructions": (
            "由真人逐项填写 status、reason、event_ids "
            "和 annotator；空白保留未标注。AI 预评分不显示以减少"
            "锚定。"
        ),
        "samples": samples,
    }
    write_json(out, payload)
    return payload


def calibrate(quality_report: Path, annotations: Path):
    quality = json.loads(quality_report.read_text(encoding="utf-8"))
    humans = json.loads(annotations.read_text(encoding="utf-8"))
    if humans.get("schema_version") != "muharness-human-v3":
        raise ValueError("人工标注 schema 不匹配")
    machine = {row["sample_id"]: row for row in quality["reviews"]}
    seen = set()
    totals = {name: Counter() for name in RUBRICS}
    disagreements = []
    labeled_samples = set()
    labeled_categories = set()
    covered_kinds = set()
    for sample in humans["samples"]:
        identifier = sample["sample_id"]
        if identifier in seen or identifier not in machine:
            raise ValueError("人工标注包含重复或未提供的样本")
        seen.add(identifier)
        row = machine[identifier]
        packet = json.loads(Path(row["packet"]).read_text(encoding="utf-8"))
        if (
            sample["packet_sha256"] != digest(packet)
            or digest(packet) != row["result"]["packet_sha256"]
        ):
            raise ValueError("人工标注与机器评审资料不一致")
        valid_ids = {
            e["event_id"] for e in packet["events"] + packet["objective_checks"]
        }
        if set(sample["rubrics"]) != set(RUBRICS):
            raise ValueError("人工规则集不完整")
        for name, verdict in sample["rubrics"].items():
            status = verdict.get("status")
            if not status:
                continue
            if (
                status not in STATES
                or not sample.get("annotator", "").strip()
                or not verdict.get("reason", "").strip()
            ):
                raise ValueError("已标注条目需真人署名、合法状态和理由")
            refs = verdict.get("event_ids", [])
            if (
                any(ref not in valid_ids for ref in refs)
                or status in {"PASS", "FAIL"}
                and not refs
            ):
                raise ValueError("人工标注证据引用不合法")
            actual = row["result"]["rubrics"][name]["status"]
            totals[name]["labeled"] += 1
            totals[name]["agreement"] += actual == status
            totals[name]["false_accept"] += actual == "PASS" and status == "FAIL"
            totals[name]["machine_unknown"] += actual == "UNKNOWN"
            if actual != status:
                disagreements.append(
                    {
                        "sample_id": identifier,
                        "rubric": name,
                        "human": verdict,
                        "machine": row["result"]["rubrics"][name],
                        "packet": row["packet"],
                    }
                )
            labeled_samples.add(identifier)
            labeled_categories.add(row["category"])
            covered_kinds.add(
                "fake_success"
                if row.get("origin") == "calibration_fixture"
                else row.get("objective_status", "UNKNOWN")
            )
    metrics = {
        name: {
            **dict(values),
            "agreement_rate": values["agreement"] / values["labeled"]
            if values["labeled"]
            else None,
            "false_accept_rate": values["false_accept"] / values["labeled"]
            if values["labeled"]
            else None,
            "unknown_rate": values["machine_unknown"] / values["labeled"]
            if values["labeled"]
            else None,
        }
        for name, values in totals.items()
    }
    representative = {"PASS", "FAIL", "UNKNOWN", "fake_success"} <= covered_kinds
    complete = (
        len(labeled_samples) >= 20
        and len(labeled_categories) == 6
        and all(values["labeled"] >= 20 for values in totals.values())
        and representative
    )
    judge_ready = all(
        machine[identifier]["result"].get("state") == "VALID"
        for identifier in labeled_samples
    )
    acceptable = (
        complete
        and judge_ready
        and all(
            values["agreement_rate"] >= 0.9
            and not values.get("false_accept", 0)
            and values["unknown_rate"] <= 0.2
            for values in metrics.values()
        )
    )
    return {
        "status": "CALIBRATED_SMALL_SAMPLE"
        if acceptable
        else "PENDING_JUDGE"
        if complete and not judge_ready
        else "REVIEW"
        if complete
        else "PENDING_HUMAN",
        "labeled_samples": len(labeled_samples),
        "planned_samples": len(machine),
        "categories": sorted(labeled_categories),
        "covered_sample_kinds": sorted(covered_kinds),
        "judge_ready": judge_ready,
        "rubrics": metrics,
        "disagreements": disagreements,
        "criteria": (
            "至少20条、六类覆盖、裁判实际有效、每条规则一致率>=90%、"
            "错误放行0、UNKNOWN<=20%；仅小样本校准"
        ),
        "quality_report": str(quality_report.resolve()),
        "annotations": str(annotations.resolve()),
    }
