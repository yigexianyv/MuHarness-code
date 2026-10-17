from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

from .common import (
    BACKEND,
    atomic_write,
    digest,
    load_experiment,
    now,
    python_executable,
    read_json,
    write_json,
)
from .container import TaskContainer, docker, pin_image
from .runner import completed_result, experiment_lock, export, probe


def official_reports(directory: Path, ids: set[str]) -> dict[str, bool]:
    """Read official per-instance reports; absence is unknown, never a failed test."""
    resolved = {}
    for path in directory.rglob("report.json"):
        report = read_json(path)
        for instance, value in report.items():
            if (
                instance in ids
                and isinstance(value, dict)
                and type(value.get("resolved")) is bool
                and not value.get("infra_failure", False)
            ):
                if instance in resolved and resolved[instance] != value["resolved"]:
                    raise ValueError(f"Conflicting official reports for {instance}")
                resolved[instance] = value["resolved"]
    return resolved


# Official harness (5.x) log markers. On these two paths it writes NO report.json,
# yet its own results.json counts the instance as unresolved (error_ids, same
# denominator). Treating them as "unknown" would silently drop agent failures
# from the paired comparison.
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
EVAL_TIMEOUT = re.compile(r"Test timed out after \d+ seconds")


def test_counts(status: object) -> dict:
    if not isinstance(status, dict):
        return {}
    counts = {}
    for key, name in (
        ("FAIL_TO_PASS", "fail_to_pass"),
        ("PASS_TO_PASS", "pass_to_pass"),
    ):
        group = status.get(key) or {}
        passed = len(group.get("success") or [])
        total = passed + len(group.get("failure") or [])
        counts[name] = [passed, total]
    return counts


def official_outcomes(directory: Path, ids: set[str]) -> dict[str, dict]:
    """Per-instance official outcome plus detail; missing means unknown."""
    outcomes: dict[str, dict] = {}
    for path in directory.rglob("report.json"):
        report = read_json(path)
        for instance, value in report.items():
            if (
                instance not in ids
                or not isinstance(value, dict)
                or type(value.get("resolved")) is not bool
                or value.get("infra_failure", False)
            ):
                continue
            entry = {
                "resolved": value["resolved"],
                "source": "official",
                "patch_applied": bool(value.get("patch_successfully_applied")),
                **test_counts(value.get("tests_status")),
            }
            if (
                instance in outcomes
                and outcomes[instance]["resolved"] != entry["resolved"]
            ):
                raise ValueError(f"Conflicting official reports for {instance}")
            outcomes[instance] = entry
    for log in directory.rglob("run_instance.log"):
        instance = log.parent.name
        if instance not in ids or instance in outcomes:
            continue
        if (log.parent / "report.json").exists():
            continue  # infra-flagged report: stays unknown
        text = log.read_text(encoding="utf-8", errors="replace")
        if APPLY_PATCH_FAIL in text:
            outcomes[instance] = {
                "resolved": False,
                "source": "patch_apply_failed",
                "patch_applied": False,
            }
        elif EVAL_TIMEOUT.search(text):
            outcomes[instance] = {
                "resolved": False,
                "source": "eval_timeout",
                "patch_applied": True,
            }
    return outcomes


def invoke_judge(
    root: Path,
    python: str,
    predictions: Path,
    *,
    label: str,
    ids: list[str],
    timeout: int,
    max_workers: int = 1,
) -> tuple[Path, int]:
    directory = root / "judge" / f"{label}-{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, mode=0o700)
    dataset = read_json(root / "trusted" / "dataset.json")
    for row in dataset:
        lock = root / "images" / f"{row['instance_id']}.json"
        if lock.exists():
            row["image"] = read_json(lock)["id"]
    write_json(directory / "dataset.json", dataset)
    run_id = f"mh-{label}-{uuid.uuid4().hex[:12]}"
    args = [
        python_executable(python),
        "-m",
        "tests.eval.swebench.judge_worker",
        "--dataset_name",
        str(directory / "dataset.json"),
        "--split",
        read_json(root / "manifest.json")["source"]["split"],
        "--predictions_path",
        str(predictions),
        "--instance_ids",
        *ids,
        "--max_workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--run_id",
        run_id,
    ]
    write_json(directory / "invocation.json", {"argv": args, "created_at": now()})
    with (directory / "judge.log").open("wb") as log:
        # Official grading creates FRESH containers; it never uses the agent's mount.
        result = subprocess.run(
            args,
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ)
            | {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": str(BACKEND),
            },
        )
    return directory, result.returncode


def grade(
    root: Path, variant: str, python: str, *, timeout: int = 1800, max_workers: int = 1
) -> dict:
    manifest = load_experiment(root, verify_source=False)
    if probe(python, "judge") != manifest["environments"]["judge"]:
        raise ValueError("Official harness environment changed since prepare")
    with experiment_lock(root):
        predictions = export(root, variant)
        rows = [
            json.loads(line)
            for line in predictions.read_text(encoding="utf-8").splitlines()
        ]
        expected, outcomes = [], {}
        for row in rows:
            instance = row["instance_id"]
            result = completed_result(root, variant, instance, manifest["fingerprint"])
            if not result or result["status"] != "completed":
                continue
            if not row["model_patch"]:
                outcomes[instance] = {"resolved": False, "source": "empty_prediction"}
                continue
            # Older official harnesses derive image names instead of reading the
            # image field. Reject tag drift, so they still use our pinned image.
            pinned = read_json(root / "images" / f"{instance}.json")
            current = json.loads(docker("image", "inspect", pinned["source"]).stdout)[
                0
            ]["Id"]
            if current != pinned["id"]:
                raise ValueError(
                    f"Image tag drifted before official grading: {instance}"
                )
            expected.append(instance)
        directory, returncode = (None, 0)
        if expected:
            directory, returncode = invoke_judge(
                root,
                python,
                predictions,
                label=variant,
                ids=expected,
                timeout=timeout,
                max_workers=max_workers,
            )
            outcomes.update(official_outcomes(directory, set(expected)))
        report = {
            "fingerprint": manifest["fingerprint"],
            "variant": variant,
            "predictions_sha256": digest(predictions.read_bytes()),
            "created_at": now(),
            "official_returncode": returncode,
            "directory": str(directory) if directory else None,
            "outcomes": outcomes,
            "missing_official": sorted(set(expected) - outcomes.keys()),
        }
        write_json(root / "grades" / f"{variant}.json", report)
        return report


def selfcheck(
    root: Path, instance: str, python: str, *, pull: bool = True, timeout: int = 1800
) -> dict:
    """Trusted dev gold round-trip, without model calls or runner gold exposure."""
    manifest = load_experiment(root)
    if manifest["source"]["split"] != "dev":
        raise ValueError(
            "Gold selfcheck is restricted to a dev experiment; do not tune on test"
        )
    if probe(python, "judge") != manifest["environments"]["judge"]:
        raise ValueError("Official harness environment changed since prepare")
    task = next(
        (
            row
            for row in read_json(root / "tasks.json")
            if row["instance_id"] == instance
        ),
        None,
    )
    if task is None:
        raise ValueError("Instance must be in the frozen dev experiment")
    gold = next(
        row
        for row in read_json(root / "trusted/dataset.json")
        if row["instance_id"] == instance
    )
    if not gold.get("patch"):
        raise ValueError("Gold patch absent from trusted dev snapshot")
    with experiment_lock(root):
        directory = root / "trusted" / f"selfcheck-{uuid.uuid4().hex[:12]}"
        directory.mkdir()
        pinned = pin_image(
            task["image"], root / "images" / f"{instance}.json", pull=pull
        )
        container = TaskContainer.create(
            pinned["id"], directory / "workspace", task["base_commit"]
        )
        try:
            docker(
                "exec",
                "-i",
                "-w",
                "/testbed",
                container.name,
                "git",
                "-c",
                "safe.directory=/testbed",
                "apply",
                "-",
                input=gold["patch"].encode(),
            )
            container.quiesce()
            _, patch = container.export_patch(
                container.patch_base or task["base_commit"]
            )
            atomic_write(directory / "roundtrip.patch", patch)
        finally:
            container.close()
        predictions = directory / "predictions.jsonl"
        atomic_write(
            predictions,
            json.dumps(
                {
                    "instance_id": instance,
                    "model_name_or_path": "gold-roundtrip",
                    "model_patch": patch.decode(),
                }
            ).encode()
            + b"\n",
        )
        judged, returncode = invoke_judge(
            root,
            python,
            predictions,
            label="gold-roundtrip",
            ids=[instance],
            timeout=timeout,
        )
        outcomes = official_reports(judged, {instance})
        report = {
            "instance_id": instance,
            "resolved": outcomes.get(instance),
            "returncode": returncode,
            "directory": str(judged),
        }
        write_json(directory / "result.json", report)
        if report["resolved"] is not True:
            raise RuntimeError(
                f"Gold round-trip selfcheck failed; inspect {judged / 'judge.log'}"
            )
        return report
