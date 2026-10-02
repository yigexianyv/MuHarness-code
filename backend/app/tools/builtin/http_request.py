
from __future__ import annotations

import asyncio
import ipaddress
import socket
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.models.types import ToolDefinition, ToolPermission

from ..base import BaseTool

MAX_HTTP_TIMEOUT_SECONDS = 60.0
_ALLOWED_METHODS = {"GET", "POST", "HEAD"}


class HttpRequestTool(BaseTool):
    # 函数说明：HttpRequestTool.__init__
    # 用途：初始化 HttpRequestTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   allow_private：`allow_private`输入或配置值，类型 `bool`；默认 `False`。
    #   allowed_hosts：传给 `set` 的输入，类型 `tuple[str, ...]`；默认 `()`。
    #   client：模型、HTTP 或 MCP 客户端，类型 `httpx.AsyncClient | None`；默认 `None`。
    #   max_response_bytes：响应输入或配置值，类型 `int`；默认 `200000`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._allow_private`、`self._allowed_hosts`、`self._client`、
    # `self._max_response_bytes`。
    def __init__(
        self,
        *,
        allow_private: bool = False,
        allowed_hosts: tuple[str, ...] = (),
        client: httpx.AsyncClient | None = None,
        max_response_bytes: int = 200_000,
    ) -> None:
        self._allow_private = allow_private
        self._allowed_hosts = set(allowed_hosts)
        self._client = client
        self._max_response_bytes = max_response_bytes

    # 函数说明：HttpRequestTool.definition
    # 用途：提供 HttpRequestTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="http_request",
            description=(
                "向明确的 http(s) URL 发起 GET、POST 或 HEAD 请求，返回状态码、响应头和"
                "响应正文文本，正文可能截断。需要读取已知网页或调用已知 HTTP 接口时使用；"
                "寻找来源用 web_search，本地文件读取用 read_file，不用本工具探测无关地址"
                "或绕过访问限制。请求需要人工审批，默认拦截私有或内部地址。调用成功只表示"
                "收到了 HTTP 响应；须检查 status_code、text 和 truncated，不能据此宣称"
                "接口业务成功、正文完整或用户任务完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要实际请求的完整 http(s) URL，不是搜索词或本地路径。",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "HEAD"],
                        "default": "GET",
                        "description": "请求方法，默认 GET；HEAD 读取响应头，POST 提交已授权的请求正文。",
                    },
                    "headers": {
                        "type": "object",
                        "description": "可选请求头对象，值为字符串；仅填写目标接口确实需要的头。",
                    },
                    "body": {
                        "type": "string",
                        "description": "可选请求正文文本，通常用于 POST；需要 JSON 时提供已序列化的字符串并设置对应请求头。",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": (
                            "等待响应的正数秒数，默认 15；"
                            f"超过 {MAX_HTTP_TIMEOUT_SECONDS:g} 按上限处理，"
                            "超时不证明服务端没有执行请求。"
                        ),
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            strict=True,
            permission=ToolPermission.HUMAN_APPROVAL,
        )

    # 函数说明：HttpRequestTool.execute
    # 用途：执行HttpRequestTool，供内置工作区工具使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `method`、`url`、`headers`、`body`、`timeout_seconds`。
    # 返回：类型 `dict[str, Any]`；按分支返回
    # `await self._do_request(self._client, method, url, headers, body, timeout)`；
    # `await self._do_request(client, method, url, headers, body, timeout)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `str(arguments.get('method', 'GET')).upper` → `urlsplit` → `self._do_request` →
    # `self._assert_safe_url` → `httpx.AsyncClient`。
    # 分支与异常：
    #   当 `method not in _ALLOWED_METHODS` 时，抛出 `ValueError(…)`。
    #   当 `not isinstance(url, str) or not url` 时，抛出
    # `ValueError("'url' must be a non-empty string")`。
    #   当 `not isinstance(raw_headers, dict)` 时，抛出
    # `ValueError("'headers' must be an object")`。
    #   当 `body is not None and (not isinstance(body, str))` 时，抛出
    # `ValueError("'body' must be a string")`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        method = str(arguments.get("method", "GET")).upper()
        if method not in _ALLOWED_METHODS:
            raise ValueError(f"'method' must be one of {sorted(_ALLOWED_METHODS)}")

        url = arguments.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("'url' must be a non-empty string")

        headers: dict[str, str] | None = None
        raw_headers = arguments.get("headers")
        if raw_headers is not None:
            if not isinstance(raw_headers, dict):
                raise ValueError("'headers' must be an object")
            headers = {str(key): str(value) for key, value in raw_headers.items()}

        body = arguments.get("body")
        if body is not None and not isinstance(body, str):
            raise ValueError("'body' must be a string")

        timeout = arguments.get("timeout_seconds", 15.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("'timeout_seconds' must be a positive number")
        timeout = min(float(timeout), MAX_HTTP_TIMEOUT_SECONDS)

        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("'url' must be an http(s) URL")

        if self._client is not None:
            return await self._do_request(
                self._client, method, url, headers, body, timeout
            )

        await self._assert_safe_url(parsed.hostname)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await self._do_request(
                client, method, url, headers, body, timeout
            )

    # 函数说明：HttpRequestTool._do_request
    # 用途：在内置工作区工具中处理 `_do_request`，通过 `client.stream` 完成首个内部处理
    # 步骤。
    # 参数：
    #   client：模型、HTTP 或 MCP 客户端，类型 `httpx.AsyncClient`。
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   url：目标 HTTP 地址，类型 `str`。
    #   headers：`headers`输入或配置值，类型 `dict[str, str] | None`。
    #   body：请求正文或内容主体，类型 `str | None`。
    #   timeout：等待超时配置，类型 `float`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `method`、`url`、`status_code`、
    # `headers`、`text`、`truncated`、`elapsed_ms`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`perf_counter` → `bytearray` →
    # `client.stream` → `response.aiter_bytes` → `bytes(content).decode` → `bytes`；另有
    #  1 个调用点。
    # 资源/并发边界：
    # `client.stream(method, url, headers=headers, content=body, timeout=timeout)`，上下
    # 文退出时执行相应清理。
    # 分支与异常：
    #   当 `len(content) >= self._max_response_bytes` 时，结束当前循环。
    async def _do_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        body: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        content = bytearray()
        response_headers: dict[str, str] = {}
        status_code: int | None = None
        encoding = "utf-8"

        async with client.stream(
            method,
            url,
            headers=headers,
            content=body,
            timeout=timeout,
        ) as response:
            status_code = response.status_code
            response_headers = dict(response.headers.items())
            encoding = response.charset_encoding or "utf-8"
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) >= self._max_response_bytes:
                    break

        return {
            "method": method,
            "url": url,
            "status_code": status_code,
            "headers": response_headers,
            "text": bytes(content).decode(encoding, errors="replace"),
            "truncated": len(content) >= self._max_response_bytes,
            "elapsed_ms": round((perf_counter() - started_at) * 1000, 3),
        }

    # 函数说明：HttpRequestTool._assert_safe_url
    # 用途：在内置工作区工具中处理 `_assert_safe_url`，通过 `asyncio.to_thread` 完成首个
    # 内部处理步骤。
    # 参数：
    #   hostname：传给 `asyncio.to_thread` 的输入，类型 `str`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread` →
    # `ipaddress.ip_address` → `_is_blocked_address`。
    # 分支与异常：
    #   当 `self._allow_private or hostname in self._allowed_hosts` 时，返回 `None`。
    #   捕获 `socket.gaierror` 后，转换或抛出
    # `ValueError(f'URL host {hostname!r} could not be resolved (SSRF guard).')`。
    #   当 `_is_blocked_address(address)` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def _assert_safe_url(self, hostname: str) -> None:
        if self._allow_private or hostname in self._allowed_hosts:
            return

        try:
            address_records = await asyncio.to_thread(
                socket.getaddrinfo, hostname, None
            )
        except socket.gaierror:
            raise ValueError(
                f"URL host {hostname!r} could not be resolved (SSRF guard)."
            ) from None

        for address_record in address_records:
            address = ipaddress.ip_address(address_record[4][0])
            if _is_blocked_address(address):
                raise ValueError(
                    f"URL host {hostname!r} resolves to a private or internal "
                    "address and is blocked (SSRF guard)."
                )


# 函数说明：_is_blocked_address
# 用途：返回 `ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or
# ip.is_multicast…`，提供 内置工作区工具 的派生值。
# 参数：
#   ip：`ip`输入或配置值，类型 `ipaddress._BaseAddress`。
# 返回：类型 `bool`；返回 `ip.is_private or ip.is_loopback or ip.is_link_local or ip.
# is_reserved or ip.is_multicast…`。
def _is_blocked_address(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )
