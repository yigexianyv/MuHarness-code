"""沙箱出网代理：只为白名单域名建立 HTTPS 隧道。

运行在单独的联网容器里（``python3 -I -c <本文件源码> <socket 路径> <域名...>``），
只能使用标准库，也不能导入 app 包。Agent 的沙箱容器仍是 ``--network none``，
唯一的出口是共享卷里的 Unix socket，经由本代理按以下规则放行：

- 只接受 ``CONNECT host:443``，其他方法和端口一律拒绝；
- host 必须匹配白名单（``pypi.org`` 精确匹配，``*.example.com`` 只匹配子域名）；
- 域名解析出的地址必须是公网地址，并且直接连接这个已检查过的地址，
  防止借白名单域名访问内网或回环地址（包括 DNS 重绑定）。
- 宿主机开着 Clash 等 TUN 代理（fake-ip）时，域名会被解析到 198.18.0.0/15 这类保留地址，
  可以通过 ``MUHARNESS_EGRESS_ALLOW_CIDRS`` 额外放行这些网段；回环和链路本地地址
  （含云元数据 169.254.169.254）永远不能放行。
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
import sys
from collections.abc import Awaitable, Callable, Iterable

ALLOWED_PORT = 443
HEADER_LIMIT = 8192
HEADER_TIMEOUT_SECONDS = 10.0
CONNECT_TIMEOUT_SECONDS = 10.0

_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_DOMAIN_PATTERN = re.compile(rf"^(\*\.)?(?:{_LABEL}\.)+{_LABEL}$")

ALLOW_CIDRS_ENV = "MUHARNESS_EGRESS_ALLOW_CIDRS"
Network = ipaddress.IPv4Network | ipaddress.IPv6Network
_NEVER_ALLOWED = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
)

Resolver = Callable[[str, int], Awaitable[list[str]]]
Connector = Callable[
    [str, int], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]
]


class Denied(Exception):
    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def normalize_domain_patterns(values: Iterable[str]) -> tuple[str, ...]:
    """规范化白名单；非法条目（IP、裸 ``*``、带端口或路径）直接报错，不静默忽略。"""
    patterns: list[str] = []
    for raw in values:
        value = raw.strip().lower().rstrip(".")
        if not value:
            continue
        if not _DOMAIN_PATTERN.match(value) or _is_ip_literal(value.removeprefix("*.")):
            raise ValueError(f"无效的白名单域名：{raw!r}")
        if value not in patterns:
            patterns.append(value)
    return tuple(patterns)


def host_allowed(host: str, patterns: Iterable[str]) -> bool:
    host = host.lower().rstrip(".")
    for pattern in patterns:
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]):
                return True
        elif host == pattern:
            return True
    return False


def normalize_allow_networks(values: Iterable[str]) -> tuple[str, ...]:
    """额外放行的网段（如 fake-ip 的 198.18.0.0/15）；不能包含回环或链路本地地址。"""
    networks: list[str] = []
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        try:
            network = ipaddress.ip_network(value, strict=False)
        except ValueError:
            raise ValueError(f"无效的网段：{raw!r}") from None
        if any(network.overlaps(blocked) for blocked in _NEVER_ALLOWED
               if blocked.version == network.version):
            raise ValueError(f"不能放行回环或链路本地网段：{raw!r}")
        if str(network) not in networks:
            networks.append(str(network))
    return tuple(networks)


def is_public_address(address: str, extra: tuple[Network, ...] = ()) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    return ip.is_global or any(ip in network for network in extra)


def parse_connect(header: bytes) -> tuple[str, int]:
    """解析请求行，返回 (host, port)；不是合法的 CONNECT 请求时抛出 Denied。"""
    try:
        request_line = header.split(b"\r\n", 1)[0].decode("ascii")
        method, target, version = request_line.split(" ")
    except (UnicodeDecodeError, ValueError):
        raise Denied(400, "malformed request") from None
    if method != "CONNECT":
        raise Denied(405, "only CONNECT is supported")
    if not version.startswith("HTTP/1."):
        raise Denied(400, "unsupported HTTP version")
    host, separator, port_text = target.rpartition(":")
    if not separator or not host or not port_text.isdigit():
        raise Denied(400, "CONNECT target must be host:port")
    return host, int(port_text)


async def resolve_host(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    )
    return list(dict.fromkeys(info[4][0] for info in infos))


async def open_tcp(
    address: str, port: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection(address, port)


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    allowed: tuple[str, ...],
    extra_networks: tuple[Network, ...] = (),
    resolve: Resolver = resolve_host,
    connect: Connector = open_tcp,
    log: Callable[[str], None] = print,
) -> None:
    upstream_writer: asyncio.StreamWriter | None = None
    target = "-"
    try:
        try:
            async with asyncio.timeout(HEADER_TIMEOUT_SECONDS):
                header = await reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError as exc:
            if not exc.partial:
                return  # 只连上就断开：转发器的可用性探测
            raise Denied(400, "incomplete request header") from None
        except (asyncio.LimitOverrunError, TimeoutError):
            raise Denied(400, "oversized or slow request header") from None
        host, port = parse_connect(header)
        target = f"{host}:{port}"
        if port != ALLOWED_PORT:
            raise Denied(403, f"port {port} is not allowed")
        if not host_allowed(host, allowed):
            raise Denied(403, "host is not in the allowlist")
        try:
            async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS):
                addresses = await resolve(host, port)
        except (OSError, TimeoutError):
            raise Denied(502, "host could not be resolved") from None
        public = [
            address
            for address in addresses
            if is_public_address(address, extra_networks)
        ]
        if not public or len(public) != len(addresses):
            # 只要有一个解析结果是内网/回环地址就整体拒绝，不挑着连
            raise Denied(403, "host resolves to a non-public address")
        upstream_reader, upstream_writer = await _connect_first(public, port, connect)
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        log(f"ALLOW {target}")
        await asyncio.gather(
            _pipe(reader, upstream_writer),
            _pipe(upstream_reader, writer),
        )
    except Denied as denied:
        log(f"DENY {target} {denied.status} {denied.reason}")
        body = denied.reason.encode("ascii", errors="replace")
        writer.write(
            f"HTTP/1.1 {denied.status} Denied\r\n".encode("ascii")
            + f"X-MuHarness-Denied: {denied.reason}\r\n".encode("ascii", errors="replace")
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
            + body
        )
        try:
            await writer.drain()
        except OSError:
            pass
    except (OSError, asyncio.IncompleteReadError):
        pass
    finally:
        for stream in (writer, upstream_writer):
            if stream is not None:
                stream.close()


async def _connect_first(
    addresses: list[str], port: int, connect: Connector
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    for address in addresses:
        try:
            async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS):
                return await connect(address, port)
        except (OSError, TimeoutError):
            continue
    raise Denied(502, "upstream connection failed")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        try:
            if writer.can_write_eof():
                writer.write_eof()
        except OSError:
            pass


async def serve(
    socket_path: str,
    allowed: tuple[str, ...],
    extra_networks: tuple[Network, ...] = (),
) -> None:
    # 先绑定到临时名字，再原子改名：socket 一出现就已经可以连接
    temporary = f"{socket_path}.{os.getpid()}.tmp"
    for path in (socket_path, temporary):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    async def on_client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await handle_client(
            reader, writer, allowed=allowed, extra_networks=extra_networks
        )

    server = await asyncio.start_unix_server(
        on_client, path=temporary, limit=HEADER_LIMIT
    )
    os.replace(temporary, socket_path)
    print(f"READY {' '.join(allowed)}", flush=True)
    async with server:
        await server.serve_forever()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: egress_proxy SOCKET DOMAIN...", file=sys.stderr)
        return 2
    allowed = normalize_domain_patterns(argv[1:])
    if not allowed:
        print("allowlist is empty", file=sys.stderr)
        return 2
    extra_networks = tuple(
        ipaddress.ip_network(value)
        for value in normalize_allow_networks(
            os.environ.get(ALLOW_CIDRS_ENV, "").replace(",", " ").split()
        )
    )
    sys.stdout.reconfigure(line_buffering=True)
    asyncio.run(serve(argv[0], allowed, extra_networks))
    return 0


def _is_ip_literal(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
