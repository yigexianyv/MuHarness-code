from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import now, write_json


class CallLimitExceeded(RuntimeError):
    pass


# Provider/SDK failures that say nothing about the agent: retried by the worker,
# and classified as infrastructure only when the run ends on one of them.
TRANSIENT_ERROR_TYPES = frozenset(
    {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ServiceUnavailableError",
        "Timeout",
    }
)
TRANSIENT_STATUS = frozenset({408, 409, 429})


def error_chain(exc: BaseException) -> list[BaseException]:
    # Explicit causes only (`raise X from Y`). SDKs raise their typed errors
    # `from None` inside an except block, so __context__ would point at
    # transport internals (httpx.HTTPStatusError) instead of RateLimitError.
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append(exc)
        exc = exc.__cause__
    return chain


def root_error_type(exc: BaseException) -> str:
    return type(error_chain(exc)[-1]).__name__


def is_transient(exc: BaseException) -> bool:
    for item in error_chain(exc):
        if type(item).__name__ in TRANSIENT_ERROR_TYPES:
            return True
        status = getattr(item, "status_code", None)
        if isinstance(status, int) and (status in TRANSIENT_STATUS or status >= 500):
            return True
    return False


def fatal_error_type(stats: dict) -> str | None:
    """The error type of the LAST physical call, if that call failed.

    Earlier failures that the harness recovered from (e.g. a failed context
    summary, or a retried 429) must not turn a finished run into infrastructure.
    """
    if stats.get("last_error_call") == stats.get("model_calls"):
        return stats.get("last_error_type")
    return None


class Meter:
    """Count physical API attempts, including failed calls and summarization.

    The budget limit applies to *effective* calls: transient provider failures
    that the worker retried are refunded, so a flaky API does not eat the agent's
    budget. Every physical attempt is still logged and counted in model_calls.
    """

    def __init__(self, directory: Path, limit: int, secret: str = ""):
        self.directory = directory
        self.limit = limit
        self.secret = secret
        self.stats = {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "usage_missing_calls": 0,
            "transient_failures": 0,
            "error_types": {},
        }

    def clean(self, value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return text.replace(self.secret, "[REDACTED]") if self.secret else text

    def event(self, kind: str, value: Any) -> None:
        with (self.directory / "trajectory.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                self.clean({"time": now(), "kind": kind, "value": value}) + "\n"
            )

    def progress(self, text: str) -> None:
        """One human-readable line, echoed live by the controller."""
        import time

        if self.secret:
            text = text.replace(self.secret, "[REDACTED]")
        stamp = time.strftime("%H:%M:%S")
        with (self.directory / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{stamp} {' '.join(text.splitlines())}\n")

    def save(self) -> None:
        write_json(self.directory / "usage.json", self.stats)

    @property
    def effective_calls(self) -> int:
        return self.stats["model_calls"] - self.stats["transient_failures"]

    def start(self, request: Any, *, retry: int = 0) -> None:
        if self.effective_calls >= self.limit:
            # Record it as the terminal "error" of this run: budget exhaustion is an
            # agent outcome and must override any earlier, recovered API failure.
            self.stats["last_error_type"] = CallLimitExceeded.__name__
            self.stats["last_error_call"] = self.stats["model_calls"]
            self.save()
            raise CallLimitExceeded(f"Physical model call limit reached: {self.limit}")
        self.stats["model_calls"] += 1
        if retry:
            self.progress(f"模型调用 #{self.stats['model_calls']}（第 {retry} 次重试）")
        self.stats["usage_missing_calls"] += 1
        self.save()
        self.event("model_request", {"retry": retry, "request": request})

    def finish(self, response: Any, usage: dict | None) -> None:
        if usage is not None:
            self.stats["usage_missing_calls"] -= 1
            self.stats["input_tokens"] += usage.get("input_tokens", 0)
            self.stats["output_tokens"] += usage.get("output_tokens", 0)
            self.stats["cached_input_tokens"] += usage.get("cached_input_tokens") or 0
        self.save()
        self.event("model_response", response)

    def error(self, exc: Exception, *, will_retry: bool = False) -> None:
        name = root_error_type(exc)
        self.stats["last_error_type"] = name
        self.stats["last_error_call"] = self.stats["model_calls"]
        counts = Counter(self.stats["error_types"])
        counts[name] += 1
        self.stats["error_types"] = dict(counts)
        if will_retry:
            self.stats["transient_failures"] += 1
        self.save()
        self.progress(
            f"模型调用 #{self.stats['model_calls']} 出错: {name} "
            f"{str(exc)[:160]}" + ("" if will_retry else " （不再重试）")
        )
        self.event(
            "model_error",
            {"type": name, "message": str(exc), "will_retry": will_retry},
        )
