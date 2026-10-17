from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
BACKEND = PACKAGE.parents[2]
VARIANTS = ("mini", "muharness")


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json_bytes(value))


def source_fingerprint() -> str:
    """Include dirty/untracked Python, rather than pretending HEAD describes it."""
    files = sorted((BACKEND / "app").rglob("*.py")) + sorted(PACKAGE.glob("*.py"))
    return digest(
        b"".join(
            str(p.relative_to(BACKEND)).replace("\\", "/").encode()
            + b"\0"
            + p.read_bytes()
            for p in files
        )
    )


def command(
    args: list[str], *, timeout: float = 60, **kwargs: Any
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, check=True, capture_output=True, timeout=timeout, **kwargs
    )


def python_executable(value: str) -> str:
    return str(Path(shutil.which(value) or value).expanduser().resolve())


def load_experiment(root: Path, *, verify_source: bool = True) -> dict:
    manifest = read_json(root / "manifest.json")
    fingerprint = manifest.pop("fingerprint")
    if digest(json_bytes(manifest)) != fingerprint:
        raise ValueError("manifest.json changed; prepare a new experiment")
    manifest["fingerprint"] = fingerprint
    for name, expected in manifest["files"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    if verify_source and manifest["source_sha256"] != source_fingerprint():
        raise ValueError("Evaluation/runtime source changed; prepare a new experiment")
    return manifest


def validate_config(config: dict) -> dict:
    defaults = {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0,
        "max_output_tokens": 8192,
        "extra_body": {},
        "model_timeout_seconds": 120,
        "model_calls": 100,
        "command_timeout_seconds": 120,
        "task_timeout_seconds": 1800,
        "hard_tokens": 3_000_000,
        "context": {},
        # Bounded, visible retries for transient provider errors (429/5xx/timeouts),
        # identical for both variants. Retried attempts are logged and refunded
        # from the call budget; the final failure still ends the run.
        "transient_retries": 2,
        "retry_backoff_seconds": 5,
        # "normal": one agent run. "mea": product long-task mode (plan turn,
        # then Manager/Executor/Auditor rounds) under the same call meter.
        "agent_mode": "normal",
        "mea_rounds": 12,
    }
    if set(config) - defaults.keys():
        raise ValueError(
            f"Unknown config keys: {sorted(set(config) - defaults.keys())}"
        )
    result = defaults | config
    if result["provider"] not in {"openai", "deepseek", "qwen"}:
        raise ValueError(
            "This comparison uses a shared OpenAI-compatible chat protocol"
        )
    for key in (
        "max_output_tokens",
        "model_calls",
        "command_timeout_seconds",
        "task_timeout_seconds",
        "hard_tokens",
        "model_timeout_seconds",
    ):
        if type(result[key]) not in (int, float) or result[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("max_output_tokens", "model_calls", "hard_tokens"):
        if type(result[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if type(result["transient_retries"]) is not int or not (
        0 <= result["transient_retries"] <= 5
    ):
        raise ValueError("transient_retries must be an integer between 0 and 5")
    if type(result["retry_backoff_seconds"]) not in (int, float) or not (
        0 <= result["retry_backoff_seconds"] <= 60
    ):
        raise ValueError("retry_backoff_seconds must be between 0 and 60")
    if result["agent_mode"] not in {"normal", "mea"}:
        raise ValueError("agent_mode must be 'normal' or 'mea'")
    if type(result["mea_rounds"]) is not int or not 1 <= result["mea_rounds"] <= 50:
        raise ValueError("mea_rounds must be an integer between 1 and 50")
    if result["command_timeout_seconds"] > 120 or result["hard_tokens"] < 3:
        raise ValueError("Command timeout must be <=120; hard_tokens must be >=3")
    if not isinstance(result["extra_body"], dict) or not isinstance(
        result["context"], dict
    ):
        raise ValueError("extra_body and context must be objects")
    from urllib.parse import urlsplit

    url = urlsplit(result["base_url"])
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
    ):
        raise ValueError("base_url must be an HTTP(S) endpoint without credentials")
    if url.query or url.fragment:
        raise ValueError("base_url must not contain query parameters or fragments")
    import re

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", result["api_key_env"]):
        raise ValueError(
            "api_key_env must name an environment variable (never put the key here)"
        )
    # extra_body is public and persisted; secrets belong only in api_key_env.
    forbidden = {
        "api_key",
        "authorization",
        "headers",
        "api_base",
        "base_url",
        "model",
        "messages",
        "tools",
        "max_tokens",
        "temperature",
        "stream",
    }
    if forbidden & {k.lower() for k in result["extra_body"]}:
        raise ValueError(
            "extra_body cannot override transport, credentials, or common parameters"
        )
    return result
