"""单因素实验：M03 的真实运行后记忆反思开关，原始两臂报告保留。"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from .v1 import write_json
from .v3 import run_v3
from .v3_cases import v3_cases


def invariant_profile(profile):
    result = copy.deepcopy(profile)
    result.pop("memory_reflection_enabled", None)
    if result.get("memory_reflection"):
        result["memory_reflection"].pop("enabled", None)
    return result


def analyze(off, on):
    reasons = []
    if off["repeat"] != on["repeat"] or off["repeat"] < 3:
        reasons.append("每臂须同样至少3次")
    for key in (
        "source_sha256",
        "grader_version",
        "model_transport",
        "api_retry_policy",
        "execution_kind",
    ):
        if off["metadata"].get(key) != on["metadata"].get(key):
            reasons.append("非实验条件改变：" + key)
    for report, enabled in ((off, False), (on, True)):
        if report["metadata"].get("source_consistent") is not True:
            reasons.append("实验期间源码不一致或缺证明")
        rows = report["attempts"]
        if len(rows) != report["repeat"] or {r["attempt"] for r in rows} != set(
            range(1, report["repeat"] + 1)
        ):
            reasons.append("尝试缺失/重复")
        for row in rows:
            profile = row.get("runtime_profile", {})
            if (
                profile.get("memory_reflection_enabled") is not enabled
                or (profile.get("memory_reflection") or {}).get("enabled")
                is not enabled
            ):
                reasons.append("反思开关未实际应用")
            if row["status"] in {"BLOCKED", "UNKNOWN"}:
                reasons.append("关键检查缺失")
    profiles = {
        json.dumps(invariant_profile(r.get("runtime_profile", {})), sort_keys=True)
        for r in off["attempts"] + on["attempts"]
    }
    if len(profiles) != 1 or profiles == {"{}"}:
        reasons.append("实际模型/预算/其余配置改变")
    contracts = {r["metadata"].get("experiment_contract_sha256") for r in (off, on)}
    if len(contracts) != 1 or None in contracts:
        reasons.append("实验输入/验收合同不同")
    counts = {
        label: sum(r["status"] == "PASS" for r in report["attempts"])
        for label, report in (("off", off), ("on", on))
    }
    return {
        "status": "COMPARABLE" if not reasons else "INCOMPARABLE",
        "reasons": sorted(set(reasons)),
        "factor": "memory_reflection_enabled",
        "counts": counts,
        "delta_pp": round(
            100 * (counts["on"] / on["repeat"] - counts["off"] / off["repeat"]), 3
        )
        if not reasons
        else None,
        "scope": "M03 用户纠正及跨会话新值读取，仅此机制的小样本，不推广为整体能力提升",
        "arm_order": "off_then_on; 顺序效应未消除",
        "subjective_quality_included": False,
    }


async def run_experiment(
    *,
    out: Path,
    repeat=3,
    provider=None,
    model=None,
    api_retries=3,
    non_stream=False,
    factory=None,
):
    folder = out / (
        datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid4().hex[:6]
    )
    folder.mkdir(parents=True)
    plan = next(p for p in v3_cases() if p.case.id == "M03")
    contract = {
        "case": plan.case.model_dump(mode="json", exclude={"source"}),
        "scenario": plan.scenario,
        "observation": plan.observation,
    }
    contract["case"]["env"].pop("MEMORY_REFLECTION_ENABLED", None)
    sha = hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    reports = []
    for enabled, label in ((False, "off"), (True, "on")):
        case = plan.case.model_copy(
            deep=True,
            update={
                "env": {
                    **plan.case.env,
                    "MEMORY_REFLECTION_ENABLED": str(enabled).lower(),
                }
            },
        )
        report, location = await run_v3(
            plans=[replace(plan, case=case)],
            repeat=repeat,
            out=folder / label,
            provider=provider,
            model=model,
            api_retries=api_retries,
            non_stream=non_stream,
            factory=factory,
            skip_judge=True,
        )
        report["metadata"]["experiment_contract_sha256"] = sha
        report["metadata"]["controlled_factor"] = {
            "name": "memory_reflection_enabled",
            "value": enabled,
        }
        report["metadata"]["report_path"] = str((location / "report.json").resolve())
        write_json(location / "report.json", report)
        reports.append(report)
    result = analyze(*reports)
    result["arms"] = {
        label: r["metadata"]["report_path"]
        for label, r in zip(("off", "on"), reports, strict=True)
    }
    write_json(folder / "experiment.json", result)
    (folder / "experiment.md").write_text(
        "# 运行后记忆反思对照\n\n"
        + json.dumps(result, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return result, folder
