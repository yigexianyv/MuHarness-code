"""One command: preflight, frozen selection, dev gold, run, grade, report."""

from __future__ import annotations

import sys
from pathlib import Path

from .common import (
    PACKAGE,
    VARIANTS,
    digest,
    load_experiment,
    read_json,
    validate_config,
    write_json,
)
from .dataset import fetch_rows
from .runner import experiment_lock, export, prepare, probe
from .worker import credential


def execute(
    root: Path,
    *,
    config_path: Path,
    mini_python: str,
    judge_python: str,
    check_only: bool = False,
) -> int:
    # Import here so the standalone CLI can dispatch back to its existing stages.
    from .__main__ import main

    def stage(name: str, *arguments: str) -> None:
        print(f"\n[SWE-bench] {name}", flush=True)
        if main([name, *arguments]) != 0:
            raise RuntimeError(f"Stage failed: {name}. Inspect the experiment logs.")

    stage("selfcheck", "--mini-python", mini_python, "--judge-python", judge_python)
    if check_only:
        return 0
    config = validate_config(read_json(config_path))
    credential(config)  # Fail before downloads or gold tests if the key is absent.
    root = root.resolve()
    if (root / "manifest.json").exists():
        manifest = load_experiment(root)
        for kind, executable in (
            ("muharness", sys.executable),
            ("mini", mini_python),
            ("judge", judge_python),
        ):
            if probe(executable, kind) != manifest["environments"][kind]:
                raise ValueError(f"{kind} environment changed; choose a new experiment")
        frozen = read_json(root / "config.json")
        for key, value in config.items():
            if key == "context":
                if any(frozen["context"].get(k) != v for k, v in value.items()):
                    raise ValueError(
                        "Context config changed; choose a new --experiment"
                    )
            elif frozen[key] != value:
                raise ValueError(f"Config changed ({key}); choose a new --experiment")
        if read_json(root / "selection.json") != read_json(PACKAGE / "lite30.json"):
            raise ValueError("The all command expects the fixed Lite30 selection")
    else:
        stage(
            "prepare",
            "--out",
            str(root),
            "--config",
            str(config_path.resolve()),
            "--mini-python",
            mini_python,
            "--judge-python",
            judge_python,
        )
        manifest = load_experiment(root)

    # A separate DEV experiment gates the first paid call. Gold never enters the
    # test experiment, its worker requests, or any mounted agent workspace.
    gold_root = root.with_name(root.name + "-dev-gold")
    if not (gold_root / "manifest.json").exists():
        rows, source = fetch_rows("dev", manifest["source"]["revision"])
        if not rows:
            raise ValueError("The official dev split is empty")
        prepare(
            gold_root,
            [sorted(rows, key=lambda r: r["instance_id"])[0]],
            None,
            config,
            source,
            manifest["environments"],
        )
    gold_manifest = load_experiment(gold_root)
    if gold_manifest["source"].get("revision") != manifest["source"].get("revision"):
        raise ValueError("Dev gold snapshot differs; choose a new experiment directory")
    checkpoint = gold_root / "gold-check.json"
    checked = read_json(checkpoint) if checkpoint.exists() else {}
    if (
        checked.get("fingerprint") != gold_manifest["fingerprint"]
        or checked.get("resolved") is not True
    ):
        from .grading import selfcheck

        instance = read_json(gold_root / "tasks.json")[0]["instance_id"]
        print(f"\n[SWE-bench] Dev gold round-trip: {instance}", flush=True)
        result = selfcheck(gold_root, instance, judge_python)
        write_json(checkpoint, result | {"fingerprint": gold_manifest["fingerprint"]})
    else:
        print("\n[SWE-bench] Dev gold already passed", flush=True)

    stage(
        "run",
        "--experiment",
        str(root),
        "--mini-python",
        mini_python,
        "--retry-infra",
    )
    grading_errors = []
    from .grading import grade
    from .report import build_report

    for variant in VARIANTS:
        try:
            with experiment_lock(root):
                predictions = export(root, variant)
            path = root / "grades" / f"{variant}.json"
            cached = read_json(path) if path.exists() else {}
            if (
                cached.get("fingerprint") == manifest["fingerprint"]
                and cached.get("predictions_sha256") == digest(predictions.read_bytes())
                and len(cached.get("outcomes", {})) == manifest["selection"]["count"]
                and not cached.get("missing_official")
                and cached.get("official_returncode") == 0
            ):
                print(f"[SWE-bench] Reuse unchanged {variant} grading", flush=True)
                continue
            print(f"\n[SWE-bench] Grade {variant}", flush=True)
            result = grade(root, variant, judge_python)
            if result["missing_official"] or result["official_returncode"]:
                grading_errors.append(variant)
        except (RuntimeError, OSError, ValueError) as exc:
            grading_errors.append(variant)
            print(f"[SWE-bench] {variant} grading failed: {exc}", flush=True)
    with experiment_lock(root):
        report = build_report(root)
    print(f"\n[SWE-bench] Report: {root / 'report.md'}", flush=True)
    complete = all(v["score_complete"] for v in report["variants"].values())
    if grading_errors or not complete:
        print(
            "[SWE-bench] Report is partial; inspect infra_errors and missing outcomes.",
            flush=True,
        )
        return 1
    return 0
