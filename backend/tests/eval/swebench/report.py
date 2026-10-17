from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from .common import (
    VARIANTS,
    atomic_write,
    digest,
    load_experiment,
    read_json,
    write_json,
)
from .metrics import instance_metrics, patch_files, summarize
from .runner import completed_result, export


def paired_statistics(
    pairs: list[tuple[bool, bool]], *, seed: int = 20261015, samples: int = 10000
) -> dict:
    """Pairs are (mini, MuHarness). Positive differences favor MuHarness."""
    counts = Counter((a, b) for a, b in pairs)
    b = counts[(False, True)]
    c = counts[(True, False)]
    discordant = b + c
    p = min(
        1.0,
        2 * sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / 2**discordant,
    )
    differences = [int(mu) - int(mini) for mini, mu in pairs]
    if not pairs:
        return {
            "n": 0,
            "both": 0,
            "muharness_only": 0,
            "mini_only": 0,
            "neither": 0,
            "difference": None,
            "bootstrap_ci95": None,
            "mcnemar_exact_p": None,
        }
    rng = random.Random(seed)
    bootstrap = sorted(
        mean(rng.choices(differences, k=len(differences))) for _ in range(samples)
    )
    return {
        "n": len(pairs),
        "both": counts[(True, True)],
        "muharness_only": b,
        "mini_only": c,
        "neither": counts[(False, False)],
        "difference": mean(differences),
        "bootstrap_ci95": [
            bootstrap[int(0.025 * samples)],
            bootstrap[int(0.975 * samples)],
        ],
        "mcnemar_exact_p": p,
        "bootstrap_samples": samples,
        "seed": seed,
    }


def build_report(
    root: Path, *, prices: tuple[float, float, float] | None = None
) -> dict:
    manifest = load_experiment(root, verify_source=False)
    tasks = read_json(root / "tasks.json")
    selection = read_json(root / "selection.json")
    tiers = {
        entry["instance_id"]: entry.get("tier", "unknown")
        for entry in selection.get("instances", [])
    }
    # Reference-fix file sets, judge side only (localization metric).
    gold_files = {
        row["instance_id"]: patch_files(row.get("patch") or "")
        for row in read_json(root / "trusted" / "dataset.json")
    }
    outcomes, results, variants, details = {}, {}, {}, {}
    for variant in VARIANTS:
        path = export(root, variant)
        grade_path = root / "grades" / f"{variant}.json"
        grade = read_json(grade_path) if grade_path.exists() else None
        if grade and (
            grade["fingerprint"] != manifest["fingerprint"]
            or grade["predictions_sha256"] != digest(path.read_bytes())
        ):
            raise ValueError(f"Stale {variant} grading; run grade again")
        outcomes[variant] = grade["outcomes"] if grade else {}
        results[variant] = {
            task["instance_id"]: completed_result(
                root, variant, task["instance_id"], manifest["fingerprint"]
            )
            for task in tasks
        }
        records = [r for r in results[variant].values() if r]
        usage = {
            key: sum(r.get("usage", {}).get(key, 0) or 0 for r in records)
            for key in (
                "model_calls",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "usage_missing_calls",
            )
        }
        solved = sum(row["resolved"] for row in outcomes[variant].values())
        cost = None
        if prices is not None:
            uncached = max(0, usage["input_tokens"] - usage["cached_input_tokens"])
            cost = (
                uncached * prices[0]
                + usage["cached_input_tokens"] * prices[1]
                + usage["output_tokens"] * prices[2]
            ) / 1_000_000
        variants[variant] = {
            "selected": len(tasks),
            "attempted": len(records),
            "completed": sum(r["status"] == "completed" for r in records),
            "infra_errors": sum(r["status"] == "infra_error" for r in records),
            "pending": len(tasks) - len(records),
            "known_outcomes": len(outcomes[variant]),
            "official_reports": sum(
                r["source"] == "official" for r in outcomes[variant].values()
            ),
            "empty_predictions": sum(
                r["source"] == "empty_prediction" for r in outcomes[variant].values()
            ),
            "resolved": solved,
            "resolved_over_selected": solved / len(tasks),
            "score_complete": len(outcomes[variant]) == len(tasks),
            "usage": usage,
            "duration_seconds": sum(r["duration_seconds"] for r in records),
            "estimated_cost": cost,
            "cost_per_resolved": cost / solved if cost is not None and solved else None,
            "cost_incomplete": usage["usage_missing_calls"] > 0,
            "stop_reasons": dict(Counter(r["stop_reason"] for r in records)),
        }
        details[variant] = {
            task["instance_id"]: instance_metrics(
                root,
                variant,
                task["instance_id"],
                results[variant][task["instance_id"]],
                gold_files.get(task["instance_id"], set()),
            )
            for task in tasks
        }
        variants[variant]["metrics"] = summarize(
            details[variant], outcomes[variant], results[variant]
        )
    comparisons, per_repo, per_instance, per_tier = [], {}, [], {}
    for task in tasks:
        instance = task["instance_id"]
        a = outcomes["mini"].get(instance, {}).get("resolved")
        b = outcomes["muharness"].get(instance, {}).get("resolved")
        if a is not None and b is not None:
            comparisons.append((a, b))
        repo = per_repo.setdefault(
            task["repo"],
            {
                "selected": 0,
                "mini_resolved": 0,
                "muharness_resolved": 0,
                "complete_pairs": 0,
            },
        )
        repo["selected"] += 1
        repo["mini_resolved"] += a is True
        repo["muharness_resolved"] += b is True
        repo["complete_pairs"] += a is not None and b is not None
        tier = tiers.get(instance, "unknown")
        bucket = per_tier.setdefault(
            tier,
            {
                "selected": 0,
                "mini_resolved": 0,
                "muharness_resolved": 0,
                "complete_pairs": 0,
            },
        )
        bucket["selected"] += 1
        bucket["mini_resolved"] += a is True
        bucket["muharness_resolved"] += b is True
        bucket["complete_pairs"] += a is not None and b is not None
        per_instance.append(
            {
                "instance_id": instance,
                "repo": task["repo"],
                "tier": tier,
                "mini": a,
                "muharness": b,
                "outcome_source": {
                    v: outcomes[v].get(instance, {}).get("source") for v in VARIANTS
                },
                "process": {v: details[v][instance] for v in VARIANTS},
            }
        )
    report = {
        "fingerprint": manifest["fingerprint"],
        "selected": len(tasks),
        "variants": variants,
        "paired": paired_statistics(comparisons),
        "per_repo": per_repo,
        "per_tier": per_tier,
        "per_instance": per_instance,
        "prices_per_million": prices,
        "interpretation": "Only this frozen subset; no Lite300 performance estimate. "
        "Unknown outcomes stay unknown; partial scores are lower bounds. "
        "Paired statistics use complete pairs only; incomplete coverage can bias them. "
        "Cost uses supplied rates, excludes missing API usage and is not an invoice.",
    }
    write_json(root / "report.json", report)
    lines = [
        f"# SWE-bench 对照报告（{len(tasks)} 题）",
        "",
        "仅代表本次固定清单；未完成判分时，通过数 / 清单数是下界。",
        "",
        "| 组别 | 已运行 | 基建错误 | 已知结果 | 官方判分 | 空补丁 | 解决 |"
        " 模型调用 | 输入 / 输出 token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, value in variants.items():
        usage = value["usage"]
        lines.append(
            f"| {variant} | {value['attempted']} | {value['infra_errors']} | "
            f"{value['known_outcomes']}/{len(tasks)} | {value['official_reports']} | "
            f"{value['empty_predictions']} | {value['resolved']} | "
            f"{usage['model_calls']} | "
            f"{usage['input_tokens']} / {usage['output_tokens']} |"
        )
    paired = report["paired"]
    lines += [
        "",
        f"完整配对：{paired['n']}/{len(tasks)}。两组都解出 {paired['both']}；"
        f"仅 MuHarness {paired['muharness_only']}；仅 mini {paired['mini_only']}；"
        f"都未解出 {paired['neither']}。",
        "",
        f"配对差值（MuHarness − mini）：{paired['difference']}；"
        f"bootstrap 95% CI：{paired['bootstrap_ci95']}；"
        f"精确 McNemar p：{paired['mcnemar_exact_p']}。",
        "",
        "配对统计仅使用结果完整的题；覆盖不足时可能有偏差，不能据此宣称胜出。",
        "",
        "## 结果质量与过程",
        "",
        "| 指标 | " + " | ".join(variants) + " |",
        "|---|" + "---:|" * len(variants),
    ]

    def cell(value):
        if value is None:
            return "—"
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    def pair(variant, first, second):
        m = variants[variant]["metrics"]
        return f"{cell(m[first])} / {cell(m[second])}"

    def ratio(variant, numerator, denominator):
        m = variants[variant]["metrics"]
        return f"{m[numerator]}/{m[denominator]}"

    rows = [
        (
            "补丁应用失败（计未解决）",
            lambda v: cell(variants[v]["metrics"]["patch_apply_failed"]),
        ),
        (
            "判分测试超时（计未解决）",
            lambda v: cell(variants[v]["metrics"]["eval_timeouts"]),
        ),
        (
            "未解决题 F2P 平均通过比例",
            lambda v: cell(variants[v]["metrics"]["partial_f2p_mean"]),
        ),
        (
            "未解决但修好部分 F2P",
            lambda v: ratio(v, "partial_f2p_any", "unresolved_with_tests"),
        ),
        ("打挂 P2P（回归）", lambda v: ratio(v, "p2p_regressions", "applied_with_p2p")),
        (
            "改到参考补丁的文件",
            lambda v: ratio(v, "localization_hits", "localization_defined"),
        ),
        ("跑过测试", lambda v: ratio(v, "ran_tests", "runs")),
        ("跑过脚本", lambda v: ratio(v, "ran_script", "runs")),
        (
            "补丁行数中位数",
            lambda v: cell(variants[v]["metrics"]["median_patch_lines"]),
        ),
        (
            "模型调用 中位数 / P90",
            lambda v: pair(v, "median_model_calls", "p90_model_calls"),
        ),
        (
            "耗时秒 中位数 / P90",
            lambda v: pair(v, "median_duration_seconds", "p90_duration_seconds"),
        ),
        (
            "已恢复的接口错误",
            lambda v: cell(variants[v]["metrics"]["recovered_errors"]),
        ),
        (
            "清理过 git 历史的题",
            lambda v: cell(variants[v]["metrics"]["git_sanitized"]),
        ),
        (
            "查看 git 历史的命令",
            lambda v: cell(variants[v]["metrics"]["git_history_probes"]),
        ),
    ]
    for label, render in rows:
        lines.append(f"| {label} | " + " | ".join(render(v) for v in variants) + " |")
    lines += [
        "",
        "## 按难度层",
        "",
        "| 难度 | 题数 | mini | MuHarness | 完整配对 |",
        "|---|---:|---:|---:|---:|",
    ]
    for tier in ("easy", "medium", "hard", "unsolved", "unknown"):
        if tier in per_tier:
            t = per_tier[tier]
            lines.append(
                f"| {tier} | {t['selected']} | {t['mini_resolved']} | "
                f"{t['muharness_resolved']} | {t['complete_pairs']} |"
            )
    lines += [
        "",
        "F2P/P2P 来自官方逐题 tests_status；文件命中只在判分端用参考补丁计算，"
        "不进入 Agent 运行。git 历史在两组开始前已同样清理，探查命令仅供排查。",
        "费用需显式提供每百万 token 单价；缺失用量和未运行任务不计入费用。",
        "逐题结果、repo 分组、退出原因及完整用量见 report.json。",
        "",
    ]
    atomic_write(root / "report.md", "\n".join(lines).encode())
    return report
