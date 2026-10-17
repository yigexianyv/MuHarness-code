"""Judge-side process and quality metrics. Never fed back to an agent run."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median

DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
RAN_TESTS = re.compile(
    r"\b(pytest|py\.test|tox|unittest|runtests\.py|manage\.py\s+test|bin/test)\b"
)
RAN_SCRIPT = re.compile(r"\bpython[\d.]*\s+(?!-m\b|-c\b)[^\s;|&]+\.py\b")
# Ways to look past base_commit. History is sanitized, so a hit means the agent
# *tried*; it is reported to make any leak investigation possible.
GIT_HISTORY_PROBE = re.compile(
    r"\bgit\b[^\n;|&]*(--all\b|--reflog\b|\breflog\b|\bfsck\b|\borigin/"
    r"|--remotes\b|--contains\b|--tags\b|--branches\b|\bls-remote\b)"
)


def patch_files(text: str) -> set[str]:
    return {b for _, b in DIFF_HEADER.findall(text or "")}


def shell_commands(attempt: Path) -> list[str]:
    path = attempt / "trajectory.jsonl"
    if not path.exists():
        return []
    commands = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") != "shell":
            continue
        value = event.get("value") or {}
        # MuHarness logs {"arguments": {...}}, mini logs {"action": {...}}.
        source = value.get("arguments") or value.get("action") or {}
        command = source.get("command")
        if isinstance(command, str):
            commands.append(command)
    return commands


def behaviour(commands: list[str]) -> dict:
    return {
        "shell_commands": len(commands),
        "ran_tests": any(RAN_TESTS.search(c) for c in commands),
        "ran_script": any(RAN_SCRIPT.search(c) for c in commands),
        "git_history_probes": sum(bool(GIT_HISTORY_PROBE.search(c)) for c in commands),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def instance_metrics(
    root: Path,
    variant: str,
    instance: str,
    result: dict | None,
    gold_files: set[str],
) -> dict:
    if not result or not result.get("attempt"):
        return {}
    attempt = root / "runs" / variant / instance / result["attempt"]
    patch_path = attempt / "model.patch"
    patch = (
        patch_path.read_text(encoding="utf-8", errors="replace")
        if patch_path.exists()
        else ""
    )
    files = patch_files(patch)
    return {
        "files_changed": sorted(files),
        "patch_lines": sum(
            1
            for line in patch.splitlines()
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ),
        # Localization: did the agent edit at least one file the reference fix
        # touches? Only defined for non-empty patches.
        "touched_gold_file": bool(files & gold_files) if files else None,
        "recovered_errors": result.get("recovered_errors", 0),
        "git_sanitized": bool((result.get("git_audit") or {}).get("sanitized")),
        **behaviour(shell_commands(attempt)),
    }


def summarize(
    per_instance: dict[str, dict], outcomes: dict[str, dict], results: dict
) -> dict:
    rows = [m for m in per_instance.values() if m]
    known = list(outcomes.values())
    unresolved_with_f2p = [
        o["fail_to_pass"]
        for o in known
        if not o["resolved"] and (o.get("fail_to_pass") or [0, 0])[1] > 0
    ]
    applied = [o for o in known if o.get("patch_applied")]
    with_p2p = [o for o in applied if (o.get("pass_to_pass") or [0, 0])[1] > 0]
    localized = [
        m["touched_gold_file"] for m in rows if m["touched_gold_file"] is not None
    ]
    records = [r for r in results.values() if r]
    calls = [r.get("usage", {}).get("model_calls", 0) for r in records]
    durations = [r.get("duration_seconds", 0) for r in records]
    return {
        "patch_apply_failed": sum(o["source"] == "patch_apply_failed" for o in known),
        "eval_timeouts": sum(o["source"] == "eval_timeout" for o in known),
        # Partial progress on unresolved instances: share of FAIL_TO_PASS fixed.
        "unresolved_with_tests": len(unresolved_with_f2p),
        "partial_f2p_mean": (
            sum(p / t for p, t in unresolved_with_f2p) / len(unresolved_with_f2p)
            if unresolved_with_f2p
            else None
        ),
        "partial_f2p_any": sum(p > 0 for p, _ in unresolved_with_f2p),
        # Regressions: applied patches that broke at least one PASS_TO_PASS test.
        "p2p_regressions": sum(
            o["pass_to_pass"][0] < o["pass_to_pass"][1] for o in with_p2p
        ),
        "applied_with_p2p": len(with_p2p),
        "localization_hits": sum(localized),
        "localization_defined": len(localized),
        "ran_tests": sum(m["ran_tests"] for m in rows),
        "ran_script": sum(m["ran_script"] for m in rows),
        "git_history_probes": sum(m["git_history_probes"] for m in rows),
        "git_sanitized": sum(m["git_sanitized"] for m in rows),
        "recovered_errors": sum(m["recovered_errors"] for m in rows),
        "median_patch_lines": (
            median([m["patch_lines"] for m in rows if m["files_changed"]])
            if any(m["files_changed"] for m in rows)
            else None
        ),
        "median_model_calls": median(calls) if calls else None,
        "p90_model_calls": percentile(calls, 0.9),
        "median_duration_seconds": median(durations) if durations else None,
        "p90_duration_seconds": percentile(durations, 0.9),
        "runs": len(rows),
    }
