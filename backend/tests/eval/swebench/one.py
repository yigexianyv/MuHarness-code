"""Run MuHarness alone on SWE-bench Lite: one instance, or the fixed Lite30 set.

    python -m tests.eval.swebench.one --judge-python PATH [--instance ID] [--split dev]
    python -m tests.eval.swebench.one --judge-python PATH --lite30 [--experiment DIR]

Model settings come from backend/.env (MODEL_DEFAULT_PROVIDER and its
*_MODEL / *_BASE_URL / *_API_KEY), unless --config is given. Every run gets a
fresh experiment directory, so nothing has to be cleaned up between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .common import BACKEND, PACKAGE, read_json, validate_config

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


MEA_OVERRIDES = {
    "agent_mode": "mea",
    # Plan turn + up to mea_rounds Manager/Executor/Auditor rounds.
    "model_calls": 400,
    "task_timeout_seconds": 5400,
    # MEA aborts after 2 consecutive failed sub-runs, so ride out short upstream
    # overloads (proxy auth cooldowns): waits 15s, 60s, 240s, 300s.
    "transient_retries": 4,
    "retry_backoff_seconds": 15,
}


def config_from_env(model_override: str | None = None, mode: str = "normal") -> dict:
    """Mirror the product's own model settings; environment overrides .env."""
    from dotenv import dotenv_values

    values = {**dotenv_values(BACKEND / ".env"), **os.environ}
    provider = (values.get("MODEL_DEFAULT_PROVIDER") or "").strip().lower()
    if provider not in DEFAULT_BASE_URLS:
        raise ValueError(
            f"MODEL_DEFAULT_PROVIDER={provider or '(empty)'} is not supported here. "
            "The benchmark needs an OpenAI-compatible Chat Completions provider "
            "(openai / deepseek / qwen); pass --config for anything else."
        )
    prefix = provider.upper()
    model = (model_override or values.get(f"{prefix}_MODEL") or "").strip()
    if not model:
        raise ValueError(f"Set {prefix}_MODEL in backend/.env")
    base_url = (values.get(f"{prefix}_BASE_URL") or "").strip()
    defaults = read_json(PACKAGE / "config.example.json")
    return validate_config(
        defaults
        | {
            "provider": provider,
            "model": model,
            "base_url": base_url or DEFAULT_BASE_URLS[provider],
            "api_key_env": f"{prefix}_API_KEY",
        }
        | (MEA_OVERRIDES if mode == "mea" else {})
    )


SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell_command",
        "description": "Run a shell command in the repository and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run."}
            },
            "required": ["command"],
        },
    },
}


def check_endpoint(config: dict, key: str) -> bool:
    """One small, realistic Chat Completions call with a tool, before Docker work.

    Only failures that prove the benchmark cannot work stop the run. A 400 that
    rejects this probe's *content* (e.g. a moderation false positive) means the
    endpoint itself is reachable, so it is reported and the run continues.
    """
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=config["base_url"], max_retries=0, timeout=90)
    where = f"  base_url={config['base_url']}  model={config['model']}"
    try:
        client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a software engineering assistant.",
                },
                {
                    "role": "user",
                    "content": "List the files in the current directory.",
                },
            ],
            max_tokens=512,
            tools=[SHELL_TOOL],
        )
        return True
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        name = type(exc).__name__
        if name in ("APIConnectionError", "APITimeoutError"):
            hint = "Cannot reach the endpoint: is the local proxy running?"
        elif status in (401, 403):
            hint = f"The key in {config['api_key_env']} was rejected."
        elif status in (404, 405):
            hint = (
                "The endpoint has no POST /chat/completions. The benchmark needs "
                "Chat Completions with tools (a Responses-only proxy will not work)."
            )
        elif status == 400:
            print(
                f"  WARNING: endpoint reachable, but this probe was rejected "
                f"({name}: {exc}).\n{where}\n  Continuing; if the real run hits the "
                "same error, it will show up as model_error in the result."
            )
            return False
        else:
            hint = "Unexpected error from the endpoint."
        raise RuntimeError(
            f"Model endpoint check failed ({name}: {exc}).\n{where}\n  {hint}"
        ) from None


def load_rows(split: str, needed: set[str]) -> tuple[list[dict], dict]:
    """Official rows with retries; fall back to a local copy if HF is unreachable.

    A successful download is cached under output/swe-cache. Without network,
    the cache is used, else a previous experiment's frozen trusted dataset that
    contains every needed instance. The source record says where rows came from.
    """
    import time

    from .common import write_json
    from .dataset import fetch_rows

    output = BACKEND.parent / "output"
    cache = output / "swe-cache" / f"lite-{split}.json"
    for attempt in range(1, 4):
        try:
            downloaded, source = fetch_rows(split, None)
            write_json(cache, {"source": source, "rows": downloaded})
            return downloaded, source
        except Exception as exc:
            print(f"  download failed ({type(exc).__name__}: {exc})", flush=True)
            if attempt < 3:
                print(f"  retry {attempt}/2 in {5 * attempt}s", flush=True)
                time.sleep(5 * attempt)
    if cache.exists():
        data = read_json(cache)
        if needed <= {row["instance_id"] for row in data["rows"]}:
            print(f"  using cached copy: {cache}", flush=True)
            return data["rows"], data["source"] | {"loaded_from": str(cache)}
    if needed:
        manifests = sorted(output.glob("*/manifest.json")) + sorted(
            output.glob("swe-one/*/manifest.json")
        )
        for manifest in reversed(manifests):
            source = read_json(manifest).get("source", {})
            trusted = manifest.parent / "trusted" / "dataset.json"
            if source.get("split") != split or not trusted.exists():
                continue
            frozen = read_json(trusted)
            if needed <= {row["instance_id"] for row in frozen}:
                print(f"  using rows frozen in {manifest.parent}", flush=True)
                return frozen, source | {"loaded_from": str(trusted)}
    raise RuntimeError(
        "Could not download SWE-bench Lite metadata and no local copy covers "
        "the needed instances. Check the network and run the command again."
    )


def pull_images(tasks: list[dict], attempts: int = 3) -> None:
    """Fetch every task image up front: no timeout, visible progress, retries.

    Large images (django) can take longer than any per-task budget on a slow
    link; pulling here keeps download time out of the agent runs entirely.
    """
    import subprocess

    images = list(dict.fromkeys(task["image"] for task in tasks))
    print(f"\n[SWE-bench one] Pull task images ({len(images)})", flush=True)
    for index, image in enumerate(images, 1):
        present = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True
        )
        if present.returncode == 0:
            print(f"  [{index}/{len(images)}] present  {image}", flush=True)
            continue
        for attempt in range(1, attempts + 1):
            print(f"  [{index}/{len(images)}] pulling  {image}", flush=True)
            # Inherit stdout/stderr so Docker's own progress is shown.
            pulled = subprocess.run(
                ["docker", "pull", "--platform", "linux/amd64", image]
            )
            if pulled.returncode == 0:
                break
            if attempt == attempts:
                raise RuntimeError(
                    f"docker pull failed {attempts} times: {image}. Run the "
                    "same command again later; finished pulls are kept."
                )
            print(f"  pull failed, retry {attempt}/{attempts - 1}", flush=True)


def summarize_run(root: Path) -> dict:
    from .report import build_report
    from .runner import experiment_lock

    with experiment_lock(root):
        return build_report(root)


def print_single(root: Path, report: dict) -> None:
    from .common import read_json as load

    row = report["per_instance"][0]
    instance = row["instance_id"]
    variant = report["variants"]["muharness"]
    result = load(root / "runs" / "muharness" / instance / "result.json")
    attempt = root / "runs" / "muharness" / instance / result["attempt"]
    process = row["process"]["muharness"]
    print(
        json.dumps(
            {
                "instance": instance,
                "resolved": row["muharness"],
                "outcome": row["outcome_source"]["muharness"],
                "status": result["status"],
                "stop_reason": result["stop_reason"],
                "model_calls": variant["usage"]["model_calls"],
                "input_tokens": variant["usage"]["input_tokens"],
                "output_tokens": variant["usage"]["output_tokens"],
                "duration_seconds": round(result["duration_seconds"], 1),
                "files_changed": process.get("files_changed"),
                "touched_reference_files": process.get("touched_gold_file"),
                "ran_tests": process.get("ran_tests"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n  report:     {root / 'report.md'}")
    print(f"  patch:      {attempt / 'model.patch'}")
    print(f"  trajectory: {attempt / 'trajectory.jsonl'}")


def print_batch(root: Path, report: dict) -> None:
    variant = report["variants"]["muharness"]
    metrics = variant["metrics"]
    total = report["selected"]
    print(
        f"  resolved {variant['resolved']}/{total}  "
        f"(known outcomes {variant['known_outcomes']}/{total}, "
        f"infra errors {variant['infra_errors']}, pending {variant['pending']})"
    )
    for tier in ("easy", "medium", "hard", "unsolved"):
        if tier in report["per_tier"]:
            t = report["per_tier"][tier]
            print(f"  {tier:<9} {t['muharness_resolved']}/{t['selected']}")
    print(
        f"  model calls median {metrics['median_model_calls']}, "
        f"tokens in/out {variant['usage']['input_tokens']}/"
        f"{variant['usage']['output_tokens']}, "
        f"patch apply failures {metrics['patch_apply_failed']}"
    )
    print()
    for row in report["per_instance"]:
        mark = {True: "PASS", False: "fail", None: " ?  "}[row["muharness"]]
        source = row["outcome_source"]["muharness"] or "-"
        print(f"  {mark}  {row['tier']:<9} {row['instance_id']}  ({source})")
    print(f"\n  report: {root / 'report.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge-python", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--instance", help="instance_id; default: first of split")
    parser.add_argument(
        "--lite30",
        action="store_true",
        help="Run the fixed 30-instance test selection (lite30.json)",
    )
    parser.add_argument("--config", type=Path, help="Override backend/.env model")
    parser.add_argument(
        "--model", help="Use this model name; endpoint/key still from backend/.env"
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        help="Experiment folder; an existing one is resumed",
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "mea"),
        default="normal",
        help="normal: one agent run; mea: long-task mode (plan + rounds)",
    )
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--grade-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.lite30 and args.instance:
        parser.error("--lite30 and --instance are mutually exclusive")

    from .dataset import DATASET
    from .grading import grade, selfcheck
    from .runner import MAX_INFRA_RETRIES, completed_result, prepare, probe, run
    from .worker import credential

    def step(text: str) -> None:
        print(f"\n[SWE-bench one] {text}", flush=True)

    try:
        step("Model settings")
        config = (
            validate_config(read_json(args.config))
            if args.config
            else config_from_env(args.model, args.mode)
        )
        print(
            f"  provider={config['provider']} model={config['model']} "
            f"base_url={config['base_url']} key={config['api_key_env']} "
            f"mode={config['agent_mode']}"
        )
        key = credential(config)
        step("Endpoint check (1 small request)")
        if check_endpoint(config, key):
            print("  ok")

        split = "test" if args.lite30 else args.split
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.lite30:
            selection = read_json(PACKAGE / "lite30.json")
            default_root = (
                BACKEND.parent
                / "output"
                / (
                    "swe-lite30-muharness"
                    + ("-mea" if config["agent_mode"] == "mea" else "")
                )
            )
        else:
            selection = None
            default_root = None
        root = args.experiment or default_root
        root = root.resolve() if root else None

        if root and (root / "manifest.json").exists():
            step(f"Resume {root}")
            frozen = read_json(root / "config.json")
            changed = [
                k
                for k in ("provider", "model", "base_url", "agent_mode")
                if frozen.get(k, "normal") != config[k]
            ]
            if changed:
                raise ValueError(
                    f"This experiment was frozen with a different {', '.join(changed)} "
                    f"(model={frozen['model']}). Results must not mix models: pass "
                    "a new --experiment folder."
                )
            tasks = read_json(root / "tasks.json")
            # Seal attempts left by a killed controller (no model calls); they
            # are then re-run below as infrastructure errors.
            from .recovery import recover

            sealed = recover(root)
            if sealed:
                print(f"  sealed {len(sealed)} interrupted attempt(s)")
        else:
            step(f"Download SWE-bench Lite {split} metadata")
            needed = (
                {e["instance_id"] for e in selection["instances"]}
                if selection
                else ({args.instance} if args.instance else set())
            )
            rows, source = load_rows(split, needed)
            if selection is None:
                ids = sorted(row["instance_id"] for row in rows)
                instance = args.instance or ids[0]
                if instance not in ids:
                    raise ValueError(f"{instance} is not in the {split} split")
                selection = {
                    "dataset": DATASET,
                    "split": split,
                    "instances": [{"instance_id": instance}],
                }
            if root is None:
                instance = selection["instances"][0]["instance_id"]
                root = (
                    BACKEND.parent
                    / "output"
                    / "swe-one"
                    / f"{instance}-{config['agent_mode']}-{stamp}"
                ).resolve()
            step(f"Prepare {root}")
            # MuHarness only: mini is never probed or run in this mode.
            environments = {
                "mini": {"config": {}, "skipped": True},
                "judge": probe(args.judge_python, "judge"),
                "muharness": probe(sys.executable, "muharness"),
            }
            prepare(root, rows, selection, config, source, environments)
            tasks = read_json(root / "tasks.json")
            if split == "dev" and not args.skip_gold:
                step("Gold check: reference fix through export + official judge")
                selfcheck(root, tasks[0]["instance_id"], args.judge_python)
                print("  resolved: environment and grading chain are OK")

        fingerprint = read_json(root / "manifest.json")["fingerprint"]
        pull_images(tasks)
        step(f"Run MuHarness on {len(tasks)} instance(s); finished ones are skipped")
        # First pass runs everything not yet recorded; further passes only re-run
        # attempts sealed as infrastructure errors (never agent outcomes).
        for _ in range(1 + MAX_INFRA_RETRIES):
            run(root, ["muharness"], mini_python=sys.executable, retry_infra=True)
            pending_infra = [
                t["instance_id"]
                for t in tasks
                if (
                    completed_result(root, "muharness", t["instance_id"], fingerprint)
                    or {}
                ).get("status")
                == "infra_error"
            ]
            if not pending_infra:
                break
            print(f"  infra errors to retry: {', '.join(pending_infra)}")

        step("Official grading")
        grade(root, "muharness", args.judge_python, max_workers=args.grade_workers)
        report = summarize_run(root)
        step("Result")
        if len(tasks) == 1:
            print_single(root, report)
        else:
            print_batch(root, report)
        return 0
    except Exception as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
