"""测评专用接口重试：保留每次调用，原任务与验收标准不变。"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

import httpx
from openai import APIConnectionError

from app.models.adapter import ModelAdapter
from app.models.types import ModelRequest, ModelResponse

from .transport import NonStreamingAdapter


def error_kind(exc: Exception) -> tuple[int | None, bool]:
    """普通鉴权/权限错误不重试；403 仅放行已实测间歇性的分组错误。"""
    current: BaseException | None = exc
    status = None
    transient = False
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidate = getattr(current, "status_code", None)
        if isinstance(candidate, int):
            status = candidate
        transient |= isinstance(
            current,
            (TimeoutError, ConnectionError, httpx.TransportError, APIConnectionError),
        )
        current = current.__cause__ or current.__context__
    if status is None:
        match = re.search(r"\bError code:\s*(\d{3})\b", str(exc))
        status = int(match[1]) if match else None
    if status is not None:
        transient = status in {408, 429} or 500 <= status <= 599
        if status == 403:
            transient = (
                "no active subscription found for this group" in str(exc).lower()
            )
    return status, transient


class RetryingAdapter(ModelAdapter):
    def __init__(
        self,
        delegate: ModelAdapter,
        *,
        max_retries: int = 3,
        events: list[dict[str, Any]] | None = None,
        base_delay: float = 1,
        on_retry: Callable[[str], None] | None = None,
    ) -> None:
        if not 0 <= max_retries <= 6:
            raise ValueError("api_retries 必须在 0 到 6 之间")
        super().__init__(delegate.config)
        self.delegate = delegate
        self.max_retries = max_retries
        self.events = events if events is not None else []
        self.base_delay = base_delay
        self.on_retry = on_retry

    async def _invoke(
        self,
        request: ModelRequest,
        invocation: Callable[[], Awaitable[ModelResponse]],
        *,
        transport: str,
        visible: Callable[[], bool],
    ) -> ModelResponse:
        call_id = uuid4().hex
        request_digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        for index in range(self.max_retries + 1):
            started = time.perf_counter()
            row: dict[str, Any] = {
                "call_id": call_id,
                "attempt": index + 1,
                "provider": self.provider,
                "model": request.model or self.default_model,
                "request_sha256": request_digest,
                "transport": transport,
                "status": "pending",
            }
            self.events.append(row)
            delay = 0.0
            try:
                response = await invocation()
                row.update(status="ok", usage=response.usage.model_dump(mode="json"))
                return response
            except asyncio.CancelledError:
                row["status"] = "cancelled"
                raise
            except Exception as exc:
                status, transient = error_kind(exc)
                secret = self.config.api_key_value()
                rendered = (
                    str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
                )
                retry = transient and not visible() and index < self.max_retries
                delay = min(self.base_delay * 2**index, 8) if retry else 0
                row.update(
                    status="error",
                    http_status=status,
                    error_type=type(exc).__name__,
                    error=rendered,
                    retryable=transient,
                    visible_output=visible(),
                    will_retry=retry,
                    retry_delay_seconds=delay,
                )
                if not retry:
                    raise
                if self.on_retry is not None:
                    self.on_retry(
                        f"模型接口 {status or type(exc).__name__}；"
                        f"{delay:g} 秒后重试（{index + 1}/{self.max_retries}）"
                    )
            finally:
                row["duration_seconds"] = round(time.perf_counter() - started, 3)
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await self._invoke(
            request,
            lambda: self.delegate.complete(request),
            transport="non_stream",
            visible=lambda: False,
        )

    async def complete_stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> ModelResponse:
        if isinstance(self.delegate, NonStreamingAdapter):
            return await self.complete(request)
        emitted = False

        async def text_delta(text: str) -> None:
            nonlocal emitted
            emitted |= bool(text)
            await on_text_delta(text)

        async def reasoning_delta(text: str) -> None:
            nonlocal emitted
            emitted |= bool(text)
            if on_reasoning_delta is not None:
                await on_reasoning_delta(text)

        return await self._invoke(
            request,
            lambda: self.delegate.complete_stream(
                request,
                on_text_delta=text_delta,
                on_reasoning_delta=reasoning_delta,
            ),
            transport="stream",
            visible=lambda: emitted,
        )

    async def close(self) -> None:
        await self.delegate.close()
