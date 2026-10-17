from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .common import digest

DATASET = "SWE-bench/SWE-bench_Lite"
ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+\Z")


def load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if (path.suffix == ".jsonl")
        else json.loads(text)
    )
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(
            "Dataset snapshot must be a JSON array or JSONL of task objects"
        )
    return rows


def fetch_rows(split: str, revision: str | None) -> tuple[list[dict], dict]:
    # Resolve a mutable branch to a SHA before downloading any rows.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(DATASET, revision=revision).sha
    rows = list(load_dataset(DATASET, split=split, revision=sha))
    return rows, {"dataset": DATASET, "split": split, "revision": sha}


def select_rows(
    rows: list[dict], selection: dict | None
) -> tuple[list[dict], list[dict]]:
    indexed = {}
    for row in rows:
        instance_id = row.get("instance_id", "")
        if not ID_PATTERN.fullmatch(instance_id) or instance_id in indexed:
            raise ValueError(f"Invalid/duplicate dataset instance_id: {instance_id!r}")
        indexed[instance_id] = row
    entries = (
        selection["instances"]
        if selection
        else [{"instance_id": row["instance_id"]} for row in rows]
    )
    ids = [entry["instance_id"] for entry in entries]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Selection must contain unique, nonempty instance IDs")
    missing = set(ids) - indexed.keys()
    if missing:
        raise ValueError(f"Selection IDs absent from frozen dataset: {sorted(missing)}")
    selected = []
    for entry in entries:
        row = indexed[entry["instance_id"]]
        for field in ("repo", "base_commit", "problem_statement"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing {field}: {row['instance_id']}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
            raise ValueError(f"Invalid base_commit: {row['instance_id']}")
        if row["repo"].replace("/", "__") != row["instance_id"].rsplit("-", 1)[0]:
            raise ValueError(f"Repo mismatch: {row['instance_id']}")
        if "hist_resolved" in entry:
            resolved, total = (int(v) for v in entry["hist_resolved"].split("/"))
            if total <= 0 or not 0 <= resolved <= total:
                raise ValueError(f"Invalid historical count: {row['instance_id']}")
            ratio = resolved / total
            tier = (
                "unsolved"
                if ratio == 0
                else ("hard" if ratio < 0.15 else "medium" if ratio < 0.5 else "easy")
            )
            if entry.get("tier") != tier:
                raise ValueError(f"Inconsistent difficulty tier: {row['instance_id']}")
        selected.append(row)
    return selected, entries


def public_task(row: dict) -> dict:
    """Explicit allowlist: gold patches/test names/hints never enter worker input."""
    return {
        key: row[key]
        for key in ("instance_id", "repo", "base_commit", "problem_statement")
    } | {
        "image": image_name(row),
    }


def image_name(row: dict) -> str:
    for field in ("image_name", "docker_image", "image"):
        if isinstance(row.get(field), str) and row[field]:
            return row[field]
    instance = row["instance_id"].replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{instance}:latest".lower()


def local_source(path: Path, split: str) -> dict:
    return {
        "dataset": DATASET,
        "split": split,
        "revision": None,
        "local_snapshot_sha256": digest(path.read_bytes()),
    }


def selection_summary(entries: list[dict]) -> dict:
    return {
        "count": len(entries),
        "tiers": dict(Counter(e.get("tier", "unknown") for e in entries)),
        "difficulty_provenance": "User supplied; historical submissions not recomputed",
    }
