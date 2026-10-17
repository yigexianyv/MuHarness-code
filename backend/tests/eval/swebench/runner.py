from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import monotonic

from .common import (
    BACKEND,
    VARIANTS,
    atomic_write,
    command,
    digest,
    json_bytes,
    load_experiment,
    now,
    python_executable,
    read_json,
    source_fingerprint,
    validate_config,
    write_json,
)
from .container import TaskContainer, pin_image
from .dataset import public_task, select_rows, selection_summary

USER_TEMPLATE = """仓库位于 /testbed（你的工作区根目录即该仓库）。请解决下面的 issue：
修改非测试代码，使问题得到修复且不破坏现有功能。
可以运行仓库中的测试、编写临时复现脚本。完成后简要说明修改内容并结束。

<issue>
{problem_statement}
</issue>"""


def probe(python: str, kind: str) -> dict:
    script = """
import importlib.metadata as md, json, platform, hashlib
from pathlib import Path
result = {'python': platform.python_version()}
def package_hash(module):
 root = Path(module.__file__).parent
 paths = sorted(root.rglob('*.py')) + sorted(root.rglob('*.yaml'))
 data = b''.join(str(p.relative_to(root)).encode() + p.read_bytes() for p in paths)
 return hashlib.sha256(data).hexdigest()
"""
    if kind == "mini":
        script += """
import minisweagent, yaml
from minisweagent.config import builtin_config_dir
from minisweagent.agents.default import AgentConfig
from minisweagent.models.litellm_model import LitellmModel
assert hasattr(LitellmModel, '_parse_actions'), 'mini v2 native tool-call API required'
assert 'system_template' in AgentConfig.model_fields, 'mini v2 required'
path = builtin_config_dir / 'benchmarks' / 'swebench.yaml'
result.update(version=md.version('mini-swe-agent'),
              config=yaml.safe_load(path.read_text(encoding='utf-8')))
result['source_sha256'] = package_hash(minisweagent)
names = ['litellm','openai','pydantic','jinja2']
result['dependencies'] = {n: md.version(n) for n in names}
"""
    elif kind == "judge":
        script += """
import swebench, swebench.harness.run_evaluation
result['version'] = md.version('swebench')
result['source_sha256'] = package_hash(swebench)
result['dependencies'] = {n: md.version(n) for n in ['docker','datasets']}
"""
    else:
        script += """
names = ['openai','pydantic','pydantic-settings','tiktoken']
result['dependencies'] = {n: md.version(n) for n in names}
"""
    script += "\nprint('MH_SWE_PROBE=' + json.dumps(result))"
    result = command([python_executable(python), "-c", script], timeout=120)
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("MH_SWE_PROBE="):
            return json.loads(line.removeprefix("MH_SWE_PROBE="))
    raise RuntimeError(f"Could not probe {kind} environment")


def prepare(
    root: Path,
    rows: list[dict],
    selection: dict | None,
    config: dict,
    source: dict,
    environments: dict,
) -> dict:
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "Experiment directory must be empty; never overwrite a frozen experiment"
        )
    selected, entries = select_rows(rows, selection)
    config = validate_config(config)
    from app.application import DEFAULT_SYSTEM_PROMPT
    from app.runtime.context.config import ContextSettings

    # Persist the complete effective context configuration, including .env overrides.
    unknown = config["context"].keys() - ContextSettings.model_fields.keys()
    if unknown:
        raise ValueError(f"Unknown context settings: {sorted(unknown)}")
    config["context"] = ContextSettings(**config["context"]).model_dump(mode="json")
    mini_config = deepcopy(environments["mini"]["config"])
    mini_config.setdefault("agent", {}).update(
        step_limit=config["model_calls"],
        cost_limit=0,
        wall_time_limit_seconds=config["task_timeout_seconds"],
    )
    mini_config.setdefault("environment", {}).update(
        cwd="/testbed",
        timeout=config["command_timeout_seconds"],
        run_args=["--network=none"],
        interpreter=["bash", "-c"],
    )
    root.mkdir(parents=True, exist_ok=True)
    private = root / "trusted"
    private.mkdir(mode=0o700)
    # No trusted data is ever copied into the per-attempt mount or worker request.
    files = {
        "tasks.json": [public_task(row) for row in selected],
        "trusted/dataset.json": selected,
        "selection.json": selection or {"instances": entries},
        "config.json": config,
        "mini-config.json": mini_config,
        "prompts.json": {
            "system": DEFAULT_SYSTEM_PROMPT
            + "\n\n本次评测只使用当前提供的文件与 Shell 工具。"
            "Task、Memory、Skills、MCP、网页及产物发布工具未启用；直接修改工作区并结束即可。",
            "user": USER_TEMPLATE,
        },
    }
    for name, value in files.items():
        write_json(root / name, value)
    try:
        commit = (
            command(["git", "rev-parse", "HEAD"], cwd=BACKEND).stdout.decode().strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit = None
    manifest = {
        "schema": 1,
        "created_at": now(),
        "source": source,
        "source_sha256": source_fingerprint(),
        "git_commit": commit,
        "files": {name: digest((root / name).read_bytes()) for name in files},
        "selection": selection_summary(entries),
        "variants": list(VARIANTS),
        "environments": environments,
        "policy": {
            "network": "none",
            "api_style": "chat_completions",
            "sdk_retries": 0,
            "image_platform": "linux/amd64",
            "container_user": "0:0",
            "host_platform": sys.platform,
            "ignore_file_mode": os.name == "nt",
            "patch_base": "dataset.base_commit, or the image setup commit when "
            "it changes tracked files (what the official harness patches)",
            "patch_exclude_if_untracked": ["patch.txt"],
            "memory": False,
            "skills": False,
            "mea": False,
        },
    }
    manifest["fingerprint"] = digest(json_bytes(manifest))
    write_json(root / "manifest.json", manifest)
    return manifest


@contextmanager
def experiment_lock(root: Path):
    """Serialize controller mutations; model workers remain separate processes."""
    path = root / ".controller.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(
            "Experiment controller is locked. If it crashed, inspect .controller.lock "
            "and remove that file after confirming its process is gone."
        ) from None
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "time": now()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def completed_result(
    root: Path, variant: str, instance: str, fingerprint: str
) -> dict | None:
    path = root / "runs" / variant / instance / "result.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result["fingerprint"] != fingerprint:
        raise ValueError("Result belongs to a different frozen experiment")
    if result.get("patch_sha256") is not None:
        patch = root / "runs" / variant / instance / result["attempt"] / "model.patch"
        if digest(patch.read_bytes()) != result["patch_sha256"]:
            raise ValueError(f"Saved patch changed: {instance}")
    return result


MAX_INFRA_RETRIES = 2


def archived_attempts(directory: Path) -> set[str]:
    """Attempts sealed as infrastructure failures and superseded by a retry."""
    return {
        read_json(path)["attempt"] for path in directory.glob("result.infra-*.json")
    }


def live_attempts(directory: Path) -> list[Path]:
    archived = archived_attempts(directory)
    return [p for p in directory.glob("attempt-*") if p.name not in archived]


def worker_status(worker: dict) -> str:
    infrastructure = {
        "worker_error",
        "worker_crash",
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "APIConnectionError",
        "ServiceUnavailableError",
        "APIError",
        "APITimeoutError",
        "Timeout",
        "InternalServerError",
        "NotFoundError",
    }
    return (
        "infra_error"
        if (
            worker["stop_reason"] in infrastructure
            or worker.get("model_error_type") in infrastructure
        )
        else "completed"
    )


def wait_with_progress(process, progress: Path, timeout: float) -> None:
    """Wait for the worker, echoing its progress lines (judge-free) to the console."""
    deadline = time.monotonic() + timeout
    position = 0

    def drain() -> None:
        nonlocal position
        if not progress.exists():
            return
        with progress.open("rb") as stream:
            stream.seek(position)
            data = stream.read()
        # Only complete lines; a partial last line is printed next time.
        end = data.rfind(b"\n") + 1
        position += end
        for line in data[:end].decode("utf-8", errors="replace").splitlines():
            print(f"    {line}", flush=True)

    while True:
        try:
            process.wait(timeout=2)
            drain()
            return
        except subprocess.TimeoutExpired:
            drain()
            if time.monotonic() >= deadline:
                raise


def run_one(
    root: Path,
    task: dict,
    variant: str,
    manifest: dict,
    python: str,
    *,
    pull: bool = True,
    previous_attempts: list[str] | None = None,
) -> dict:
    config = read_json(root / "config.json")
    directory = root / "runs" / variant / task["instance_id"]
    attempt = "attempt-" + uuid.uuid4().hex[:12]
    output = directory / attempt
    output.mkdir(parents=True)
    start = monotonic()
    result = {
        "instance_id": task["instance_id"],
        "variant": variant,
        "attempt": attempt,
        "fingerprint": manifest["fingerprint"],
        "started_at": now(),
        "status": "infra_error",
        "stop_reason": "setup_error",
        "patch_sha256": None,
        "previous_infra_attempts": previous_attempts or [],
    }
    write_json(output / "status.json", result)
    container = None
    try:
        pinned = pin_image(
            task["image"], root / "images" / f"{task['instance_id']}.json", pull=pull
        )
        result["image"] = pinned
        container = TaskContainer.create(
            pinned["id"],
            output / "workspace",
            task["base_commit"],
            config["command_timeout_seconds"],
        )
        write_json(
            output / "container.json",
            {
                "name": container.name,
                "patch_base": getattr(container, "patch_base", None),
            },
        )
        result["git_audit"] = getattr(container, "git_audit", None)
        prompts = read_json(root / "prompts.json")
        request = {
            "variant": variant,
            "task": task,
            "config": config,
            "container": container.name,
            "workspace": str(container.workspace),
            "mini_config": read_json(root / "mini-config.json"),
            "system_prompt": prompts["system"],
            "user_template": prompts["user"],
        }
        write_json(output / "request.json", request)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(BACKEND)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        with (output / "worker.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    python,
                    "-m",
                    "tests.eval.swebench.worker",
                    str(output / "request.json"),
                ],
                cwd=BACKEND,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            write_json(
                output / "worker-process.json",
                {"pid": process.pid, "started_at": now()},
            )
            try:
                wait_with_progress(
                    process, output / "progress.log", config["task_timeout_seconds"]
                )
                worker_path = output / "worker-result.json"
                worker = (
                    read_json(worker_path)
                    if worker_path.exists()
                    else {
                        "stop_reason": "worker_crash",
                        "error": f"Worker exit code {process.returncode}",
                    }
                )
                result.update(worker)
                result["status"] = worker_status(worker)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
                result.update(status="completed", stop_reason="task_timeout")
            except BaseException:
                process.kill()
                process.wait(timeout=15)
                raise
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        # Keep the tool's own message (e.g. Docker's), not just the exit code.
        detail = getattr(exc, "stderr", None) or getattr(exc, "output", None)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        if detail:
            result["error_detail"] = str(detail)[-4000:]
    finally:
        if container is not None:
            try:
                container.quiesce()
                raw, patch = container.export_patch(
                    getattr(container, "patch_base", None) or task["base_commit"]
                )
                atomic_write(output / "raw.patch", raw)
                atomic_write(output / "model.patch", patch)
                result["patch_sha256"] = digest(patch)
                result["patch_bytes"] = len(patch)
            except Exception as exc:
                result.update(
                    status="infra_error", export_error=f"{type(exc).__name__}: {exc}"
                )
            finally:
                container.close()
        result["duration_seconds"] = monotonic() - start
        result["finished_at"] = now()
        usage = output / "usage.json"
        result["usage"] = (
            read_json(usage)
            if usage.exists()
            else {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "usage_missing_calls": 0,
            }
        )
        write_json(output / "status.json", result)
        write_json(directory / "result.json", result)
    return result


def run(
    root: Path,
    variants: list[str],
    *,
    mini_python: str,
    filter_pattern: str = ".*",
    resume: bool = False,
    pull: bool = True,
    retry_infra: bool = False,
) -> list[dict]:
    manifest = load_experiment(root)
    interpreters = {"muharness": sys.executable, "mini": python_executable(mini_python)}
    for variant in variants:
        expected = manifest["environments"][variant]
        if (
            probe(interpreters[variant], "mini" if variant == "mini" else "muharness")
            != expected
        ):
            raise ValueError(
                f"{variant} Python/dependencies/config changed since prepare"
            )
    # Check credentials before starting expensive containers or creating any results.
    from .worker import credential

    credential(read_json(root / "config.json"))
    pattern = re.compile(filter_pattern)
    all_tasks = read_json(root / "tasks.json")
    position = {task["instance_id"]: i for i, task in enumerate(all_tasks)}
    tasks = [task for task in all_tasks if pattern.search(task["instance_id"])]
    if not tasks:
        raise ValueError("--filter matched no instances")
    results = []
    with experiment_lock(root):
        for task in tasks:
            # Alternate which variant goes first (by frozen position, so --filter
            # does not change it): API load and cache warmth drift over time.
            ordered = (
                variants if position[task["instance_id"]] % 2 == 0 else variants[::-1]
            )
            for variant in ordered:
                directory = root / "runs" / variant / task["instance_id"]
                existing = completed_result(
                    root, variant, task["instance_id"], manifest["fingerprint"]
                )
                if existing:
                    previous = sorted(archived_attempts(directory))
                    if retry_infra and existing["status"] == "infra_error":
                        if len(previous) >= MAX_INFRA_RETRIES:
                            print(
                                f"skip {variant} {task['instance_id']} "
                                f"(infra retries exhausted: {len(previous)})",
                                flush=True,
                            )
                            continue
                        # Infrastructure failures are not agent outcomes: archive
                        # the sealed attempt (kept on disk) and run a fresh one.
                        os.replace(
                            directory / "result.json",
                            directory / f"result.infra-{existing['attempt']}.json",
                        )
                        previous.append(existing["attempt"])
                        print(
                            f"retry {variant} {task['instance_id']} "
                            f"after infra error ({existing['stop_reason']})",
                            flush=True,
                        )
                    elif resume or retry_infra:
                        print(
                            f"skip {variant} {task['instance_id']} "
                            f"({existing['stop_reason']})",
                            flush=True,
                        )
                        continue
                    else:
                        raise ValueError(
                            "Existing result found; use --resume (one attempt per task)"
                        )
                else:
                    previous = sorted(archived_attempts(directory))
                    if directory.exists() and live_attempts(directory):
                        raise RuntimeError(
                            f"Interrupted controller attempt found at {directory}. "
                            "Use recover before --resume; model attempts are not "
                            "repeated."
                        )
                result = run_one(
                    root,
                    task,
                    variant,
                    manifest,
                    interpreters[variant],
                    pull=pull,
                    previous_attempts=previous,
                )
                results.append(result)
                print(
                    f"{variant} {task['instance_id']}: "
                    f"{result['status']}/{result['stop_reason']}",
                    flush=True,
                )
    return results


def export(root: Path, variant: str) -> Path:
    manifest = load_experiment(root, verify_source=False)
    predictions, coverage = [], []
    for task in read_json(root / "tasks.json"):
        instance = task["instance_id"]
        result = completed_result(root, variant, instance, manifest["fingerprint"])
        patch = ""
        if result and result.get("patch_sha256"):
            patch = (
                root / "runs" / variant / instance / result["attempt"] / "model.patch"
            ).read_text(encoding="utf-8")
        predictions.append(
            {
                "instance_id": instance,
                "model_name_or_path": f"mh-{variant}",
                "model_patch": patch,
            }
        )
        coverage.append(
            {
                "instance_id": instance,
                "status": result["status"] if result else "pending",
                "stop_reason": result["stop_reason"] if result else None,
                "patch_bytes": len(patch.encode()),
            }
        )
    path = root / "predictions" / f"{variant}.jsonl"
    atomic_write(
        path,
        b"".join(
            json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in predictions
        ),
    )
    write_json(path.with_suffix(".coverage.json"), coverage)
    return path
