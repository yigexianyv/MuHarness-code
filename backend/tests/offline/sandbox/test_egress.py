"""第二阶段：Shell 沙箱经白名单代理出网。

沙箱容器仍是 --network none，只多挂一个只读的共享卷，里面是代理的 Unix socket；
代理在单独的联网容器里，只放行白名单域名的 CONNECT :443，并拒绝解析到内网的地址。
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from app.safety.sandbox import (
    DockerSandboxBackend,
    SandboxBackend,
    SandboxConfig,
    SandboxLaunchSpec,
    SandboxNetworkMode,
    SandboxPolicy,
    SandboxPolicyError,
    SandboxSupervisor,
    SandboxUnavailableError,
    UnsupportedSandboxBackend,
)
from app.safety.sandbox import backends as backends_module
from app.safety.sandbox import egress_forwarder, egress_proxy
from app.safety.sandbox.egress_proxy import (
    handle_client,
    host_allowed,
    is_public_address,
    normalize_allow_networks,
    normalize_domain_patterns,
)
from app.tools.builtin.shell import ShellCommandTool

PUBLIC_IP = "93.184.216.34"
UNIX_SOCKETS = sys.platform != "win32" and hasattr(socket, "AF_UNIX")

# ---------------------------------------------------------------- 白名单规则


def test_normalize_domain_patterns_lowercases_and_deduplicates() -> None:
    assert normalize_domain_patterns(
        ["PyPI.org", "pypi.org.", " files.pythonhosted.org ", "*.Example.com", ""]
    ) == ("pypi.org", "files.pythonhosted.org", "*.example.com")


@pytest.mark.parametrize(
    "value",
    ["*", "*.com.", "1.2.3.4", "localhost", "pypi.org:443", "https://pypi.org", "a..b", "*.10.0.0.1"],
)
def test_normalize_domain_patterns_rejects_invalid_entries(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_domain_patterns([value])


def test_host_allowed_exact_and_subdomain_only() -> None:
    patterns = ("pypi.org", "*.example.com")
    assert host_allowed("pypi.org", patterns)
    assert host_allowed("PYPI.ORG.", patterns)
    assert not host_allowed("evil-pypi.org", patterns)
    assert not host_allowed("pypi.org.evil.com", patterns)
    assert host_allowed("a.b.example.com", patterns)
    assert not host_allowed("example.com", patterns)
    assert not host_allowed("badexample.com", patterns)


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("8.8.8.8", True),
        ("2606:4700:4700::1111", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.17.0.1", False),
        ("192.168.65.254", False),
        ("169.254.169.254", False),
        ("100.64.0.1", False),
        ("0.0.0.0", False),
        ("::1", False),
        ("fd00::1", False),
        ("fe80::1%eth0", False),
        ("not-an-ip", False),
    ],
)
def test_is_public_address(address: str, public: bool) -> None:
    assert is_public_address(address) is public


def test_extra_networks_admit_fake_ip_range_only() -> None:
    fake_ip = (ipaddress.ip_network("198.18.0.0/15"),)
    assert not is_public_address("198.18.0.7")
    assert is_public_address("198.18.0.7", fake_ip)
    assert not is_public_address("10.0.0.1", fake_ip)


def test_normalize_allow_networks() -> None:
    assert normalize_allow_networks(["198.18.0.1/15", " 198.18.0.0/15", ""]) == (
        "198.18.0.0/15",
    )


@pytest.mark.parametrize(
    "value", ["0.0.0.0/0", "127.0.0.0/8", "169.254.169.254/32", "::/0", "fe80::/64", "nope"]
)
def test_normalize_allow_networks_rejects_loopback_and_link_local(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_allow_networks([value])


# ---------------------------------------------------------------- 代理行为


class ProxyHarness:
    """在 TCP 上跑代理的处理逻辑：解析结果和上游连接都由测试控制。"""

    def __init__(
        self,
        allowed: tuple[str, ...],
        addresses: dict[str, list[str]],
        extra_networks: tuple[Any, ...] = (),
    ):
        self.allowed = allowed
        self.extra_networks = extra_networks
        self.addresses = addresses
        self.connected: list[tuple[str, int]] = []
        self.logs: list[str] = []
        self.echo_port = 0

    async def resolve(self, host: str, port: int) -> list[str]:
        del port
        if host not in self.addresses:
            raise OSError("no such host")
        return self.addresses[host]

    async def connect(
        self, address: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        self.connected.append((address, port))
        return await asyncio.open_connection("127.0.0.1", self.echo_port)

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await handle_client(
            reader,
            writer,
            allowed=self.allowed,
            extra_networks=self.extra_networks,
            resolve=self.resolve,
            connect=self.connect,
            log=self.logs.append,
        )


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(1024):
        writer.write(data)
        await writer.drain()
    writer.close()


async def _request(port: int, raw: bytes) -> tuple[bytes, bytes]:
    """发送请求头；成功建立隧道时再发 ping，返回 (响应头, 隧道里收到的数据)。"""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(raw)
    await writer.drain()
    head = await reader.readuntil(b"\r\n\r\n")
    tunneled = b""
    if head.startswith(b"HTTP/1.1 200"):
        writer.write(b"ping")
        await writer.drain()
        tunneled = await reader.readexactly(4)
    writer.close()
    return head, tunneled


async def _run_proxy_case(
    raw: bytes,
    *,
    allowed: tuple[str, ...] = ("pypi.org",),
    addresses: dict[str, list[str]] | None = None,
    extra_networks: tuple[Any, ...] = (),
) -> tuple[bytes, bytes, ProxyHarness]:
    if addresses is None:
        addresses = {"pypi.org": [PUBLIC_IP]}
    harness = ProxyHarness(allowed, addresses, extra_networks)
    echo = await asyncio.start_server(_echo, "127.0.0.1", 0)
    harness.echo_port = echo.sockets[0].getsockname()[1]
    proxy = await asyncio.start_server(harness.handle, "127.0.0.1", 0)
    port = proxy.sockets[0].getsockname()[1]
    try:
        head, tunneled = await asyncio.wait_for(_request(port, raw), 5)
    finally:
        proxy.close()
        echo.close()
    return head, tunneled, harness


def _connect(target: str) -> bytes:
    return f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode()


@pytest.mark.asyncio
async def test_proxy_tunnels_allowlisted_host_to_checked_address() -> None:
    head, tunneled, harness = await _run_proxy_case(_connect("pypi.org:443"))

    assert head.startswith(b"HTTP/1.1 200")
    assert tunneled == b"ping"
    # 连接的是检查过的地址，而不是再解析一次域名
    assert harness.connected == [(PUBLIC_IP, 443)]
    assert harness.logs == ["ALLOW pypi.org:443"]


@pytest.mark.asyncio
async def test_proxy_tunnels_fake_ip_only_when_configured() -> None:
    fake = {"pypi.org": ["198.18.0.7"]}
    head, _, harness = await _run_proxy_case(_connect("pypi.org:443"), addresses=fake)
    assert head.startswith(b"HTTP/1.1 403")
    assert harness.connected == []

    head, tunneled, harness = await _run_proxy_case(
        _connect("pypi.org:443"),
        addresses=fake,
        extra_networks=(ipaddress.ip_network("198.18.0.0/15"),),
    )
    assert head.startswith(b"HTTP/1.1 200")
    assert tunneled == b"ping"
    assert harness.connected == [("198.18.0.7", 443)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "addresses", "status", "reason"),
    [
        (_connect("example.com:443"), None, b"403", b"allowlist"),
        (_connect("pypi.org:80"), None, b"403", b"port 80"),
        (_connect("pypi.org:443"), {"pypi.org": ["10.0.0.5"]}, b"403", b"non-public"),
        # 有一个内网结果就整体拒绝，不挑着连公网那个
        (_connect("pypi.org:443"), {"pypi.org": [PUBLIC_IP, "127.0.0.1"]}, b"403", b"non-public"),
        (_connect("pypi.org:443"), {}, b"502", b"resolved"),
        (b"GET http://pypi.org/ HTTP/1.1\r\nHost: pypi.org\r\n\r\n", None, b"405", b"CONNECT"),
        (b"CONNECT pypi.org HTTP/1.1\r\n\r\n", None, b"400", b"host:port"),
    ],
)
async def test_proxy_denies_without_connecting(
    raw: bytes,
    addresses: dict[str, list[str]] | None,
    status: bytes,
    reason: bytes,
) -> None:
    head, tunneled, harness = await _run_proxy_case(raw, addresses=addresses)

    assert head.startswith(b"HTTP/1.1 " + status)
    assert reason in head
    assert tunneled == b""
    assert harness.connected == []
    assert harness.logs and harness.logs[0].startswith("DENY")


# ---------------------------------------------------------------- 沙箱启动参数


def _docker_backend(
    tmp_path: Path, egress_allow_cidrs: tuple[str, ...] = ()
) -> DockerSandboxBackend:
    return DockerSandboxBackend(
        tmp_path,
        docker_command=sys.executable,
        image="test-muharness-sandbox:latest",
        instance_id="abc123",
        egress_allow_cidrs=egress_allow_cidrs,
    )


def _prepare(
    backend: SandboxBackend,
    tmp_path: Path,
    *,
    domains: tuple[str, ...] = ("pypi.org",),
    network: SandboxNetworkMode = SandboxNetworkMode.DENIED,
) -> SandboxLaunchSpec:
    supervisor = SandboxSupervisor(tmp_path, native_backend=backend)
    return supervisor.prepare_launch(
        command="/bin/sh",
        args=("-c", "pip install six"),
        env={"PATH": os.environ.get("PATH", "")},
        cwd=None,
        config=SandboxConfig(network=network, allowed_domains=domains),
    )


def test_docker_allowlist_requires_ready_egress_proxy(tmp_path: Path) -> None:
    backend = _docker_backend(tmp_path)

    with pytest.raises(SandboxUnavailableError, match="出网代理"):
        _prepare(backend, tmp_path)

    backend._egress_domains = ("files.pythonhosted.org",)
    with pytest.raises(SandboxUnavailableError, match="出网代理"):
        _prepare(backend, tmp_path)


def test_docker_allowlist_only_with_denied_network(tmp_path: Path) -> None:
    backend = _docker_backend(tmp_path)
    backend._egress_domains = ("pypi.org",)

    with pytest.raises(SandboxPolicyError, match="断网"):
        _prepare(backend, tmp_path, network=SandboxNetworkMode.UNRESTRICTED)


def test_docker_allowlist_keeps_network_none_and_wraps_command(tmp_path: Path) -> None:
    backend = _docker_backend(tmp_path)
    backend._egress_domains = ("pypi.org",)

    args = _prepare(backend, tmp_path, domains=("PyPI.org",)).args

    assert args[args.index("--network") + 1] == "none"
    mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
    assert (
        "type=volume,source=muharness-egress-abc123,target=/egress,readonly" in mounts
    )
    image_index = args.index("test-muharness-sandbox:latest")
    assert args[image_index + 1 : image_index + 4] == ("python3", "-I", "-c")
    assert args[image_index + 4] == Path(egress_forwarder.__file__).read_text(
        encoding="utf-8"
    ).replace("\r\n", "\n")
    assert args[image_index + 5 :] == (
        "/egress/proxy.sock",
        "/bin/sh",
        "-c",
        "pip install six",
    )


def test_docker_without_allowlist_is_unchanged(tmp_path: Path) -> None:
    args = _prepare(_docker_backend(tmp_path), tmp_path, domains=()).args

    assert "python3" not in args
    assert not any("egress" in value for value in args)
    assert args[-3:] == ("/bin/sh", "-c", "pip install six")


def test_unsupported_backend_rejects_allowlist(tmp_path: Path) -> None:
    supervisor = SandboxSupervisor(
        tmp_path, native_backend=UnsupportedSandboxBackend("test")
    )
    with pytest.raises(SandboxUnavailableError, match="域名白名单"):
        asyncio.run(
            supervisor.ensure_egress(SandboxConfig(allowed_domains=("pypi.org",)))
        )


# ---------------------------------------------------------------- 代理容器生命周期


class FakeDocker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.running = False
        self.ready = True

    async def __call__(self, docker_command: str, *args: str, timeout: float = 30) -> str | None:
        del docker_command, timeout
        self.calls.append(args)
        match args[0]:
            case "run":
                self.running = self.ready
                return "container-id\n"
            case "inspect":
                return "true\n" if self.running else None
            case "logs":
                return "READY pypi.org\n" if self.ready else "Traceback: boom\n"
            case "rm":
                self.running = False
                return ""
            case _:
                return ""

    def commands(self) -> list[str]:
        return [" ".join(call[:2]) if call[0] == "volume" else call[0] for call in self.calls]


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(backends_module, "_run_docker", fake)
    return fake


@pytest.mark.asyncio
async def test_ensure_egress_starts_hardened_proxy_once(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    backend = _docker_backend(tmp_path)

    await backend.ensure_egress(("PyPI.org", "files.pythonhosted.org"))

    assert fake_docker.commands() == ["rm", "volume create", "run", "logs"]
    volume = fake_docker.calls[1]
    assert "type=tmpfs" in volume
    options = next(value for value in volume if value.startswith("o="))
    assert "mode=0700" in options and "uid=" in options
    run = fake_docker.calls[2]
    assert "--network" not in run  # 代理容器需要网络；沙箱容器不用它
    for flag in ("--read-only", "--cap-drop", "--security-opt", "--pids-limit", "--memory"):
        assert flag in run
    assert "type=volume,source=muharness-egress-abc123,target=/egress" in run
    assert run[-7:-4] == ("python3", "-I", "-c")
    assert run[-4] == Path(egress_proxy.__file__).read_text(
        encoding="utf-8"
    ).replace("\r\n", "\n")
    assert run[-3:] == ("/egress/proxy.sock", "pypi.org", "files.pythonhosted.org")
    assert "--env" not in run

    # 已在运行且白名单不变：只检查一次状态
    fake_docker.calls.clear()
    await backend.ensure_egress(("pypi.org", "files.pythonhosted.org"))
    assert fake_docker.commands() == ["inspect"]

    # 之后的沙箱带上代理卷
    args = _prepare(backend, tmp_path, domains=("pypi.org", "files.pythonhosted.org")).args
    assert "/egress/proxy.sock" in args


@pytest.mark.asyncio
async def test_ensure_egress_passes_extra_networks_to_proxy(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    backend = _docker_backend(tmp_path, ("198.18.0.1/15",))

    await backend.ensure_egress(("pypi.org",))

    run = fake_docker.calls[2]
    assert run[run.index("--env") + 1] == "MUHARNESS_EGRESS_ALLOW_CIDRS=198.18.0.0/15"


def test_backend_rejects_loopback_extra_network(tmp_path: Path) -> None:
    with pytest.raises(SandboxPolicyError, match="ALLOW_CIDRS"):
        _docker_backend(tmp_path, ("127.0.0.0/8",))


@pytest.mark.asyncio
async def test_ensure_egress_restarts_dead_or_changed_proxy(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    backend = _docker_backend(tmp_path)
    await backend.ensure_egress(("pypi.org",))

    fake_docker.running = False  # 比如 Docker 重启过
    fake_docker.calls.clear()
    await backend.ensure_egress(("pypi.org",))
    assert "run" in fake_docker.commands()

    fake_docker.calls.clear()
    await backend.ensure_egress(("registry.npmjs.org",))
    assert "run" in fake_docker.commands()
    with pytest.raises(SandboxUnavailableError):
        _prepare(backend, tmp_path, domains=("pypi.org",))


@pytest.mark.asyncio
async def test_ensure_egress_failure_leaves_sandbox_closed(
    tmp_path: Path, fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backends_module, "EGRESS_READY_TIMEOUT_SECONDS", 0.5)
    fake_docker.ready = False
    backend = _docker_backend(tmp_path)

    with pytest.raises(SandboxUnavailableError, match="boom"):
        await backend.ensure_egress(("pypi.org",))

    assert fake_docker.calls[-1][0] == "rm"
    with pytest.raises(SandboxUnavailableError):
        _prepare(backend, tmp_path)


@pytest.mark.asyncio
async def test_ensure_egress_rejects_invalid_allowlist(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    with pytest.raises(SandboxPolicyError):
        await _docker_backend(tmp_path).ensure_egress(("10.0.0.1",))
    assert fake_docker.calls == []


@pytest.mark.asyncio
async def test_close_removes_proxy_and_volume(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    backend = _docker_backend(tmp_path)
    await backend.close()
    assert fake_docker.calls == []  # 从没启动过就不调用 docker

    await backend.ensure_egress(("pypi.org",))
    fake_docker.calls.clear()
    await backend.close()

    assert fake_docker.calls == [
        ("rm", "--force", "muharness-egress-abc123"),
        ("volume", "rm", "--force", "muharness-egress-abc123"),
    ]
    with pytest.raises(SandboxUnavailableError):
        _prepare(backend, tmp_path)


# ---------------------------------------------------------------- Shell 工具


class RecordingEgressBackend(SandboxBackend):
    resolve_executable_on_host = False

    def __init__(self) -> None:
        self.events: list[str] = []
        self.policy: SandboxPolicy | None = None

    async def ensure_egress(self, domains: tuple[str, ...]) -> None:
        self.events.append("ensure:" + ",".join(domains))

    def prepare(
        self,
        *,
        command: str | Path,
        args: tuple[str, ...],
        env: dict[str, str],
        policy: SandboxPolicy,
    ) -> SandboxLaunchSpec:
        del command, args
        self.events.append("prepare")
        self.policy = policy
        return SandboxLaunchSpec(
            command=sys.executable,
            args=("-c", "print('ok')"),
            cwd=str(policy.working_directory),
            env=env,
            backend="recording",
            sandboxed=True,
        )


@pytest.mark.asyncio
async def test_shell_ensures_egress_before_launch(tmp_path: Path) -> None:
    backend = RecordingEgressBackend()
    tool = ShellCommandTool(
        tmp_path,
        sandbox_supervisor=SandboxSupervisor(tmp_path, native_backend=backend),
        allowed_domains=("pypi.org",),
    )

    result = await tool.execute({"command": "pip install six"})

    assert result["exit_code"] == 0
    assert backend.events == ["ensure:pypi.org", "prepare"]
    assert backend.policy is not None
    assert backend.policy.network is SandboxNetworkMode.DENIED
    assert backend.policy.allowed_domains == ("pypi.org",)
    assert "pypi.org" in tool.definition.description


@pytest.mark.asyncio
async def test_shell_without_allowlist_never_touches_egress(tmp_path: Path) -> None:
    backend = RecordingEgressBackend()
    tool = ShellCommandTool(
        tmp_path,
        sandbox_supervisor=SandboxSupervisor(tmp_path, native_backend=backend),
    )

    await tool.execute({"command": "true"})

    assert backend.events == ["prepare"]
    assert "沙箱网络禁用" in tool.definition.description


# ---------------------------------------------------------------- 转发器（需要 Unix socket）


async def _run_forwarder(socket_path: str, child: str) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        Path(egress_forwarder.__file__).read_text(encoding="utf-8"),
        socket_path,
        sys.executable,
        "-c",
        child,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
    assert process.returncode is not None
    return process.returncode, stdout.decode(), stderr.decode()


_CHILD_THROUGH_PROXY = r"""
import os, socket, urllib.parse
proxy = urllib.parse.urlparse(os.environ["HTTPS_PROXY"])
assert proxy.hostname == "127.0.0.1", proxy
assert os.environ["http_proxy"] == os.environ["HTTPS_PROXY"]
assert "NO_PROXY" not in os.environ
for target in ("pypi.org:443", "example.com:443"):
    s = socket.create_connection((proxy.hostname, proxy.port), timeout=5)
    s.sendall(f"CONNECT {target} HTTP/1.1\r\n\r\n".encode())
    stream = s.makefile("rb")
    status = stream.readline().split()[1].decode()
    while stream.readline() not in (b"\r\n", b""):
        pass
    if status == "200":
        s.sendall(b"ping")
        status += ":" + stream.read(4).decode()
    print(target, status)
    s.close()
raise SystemExit(7)
"""


@pytest.mark.skipif(not UNIX_SOCKETS, reason="需要 Unix socket（Linux/macOS）")
@pytest.mark.asyncio
async def test_forwarder_routes_child_through_proxy_socket(tmp_path: Path) -> None:
    harness = ProxyHarness(("pypi.org",), {"pypi.org": [PUBLIC_IP]})
    echo = await asyncio.start_server(_echo, "127.0.0.1", 0)
    harness.echo_port = echo.sockets[0].getsockname()[1]
    socket_path = str(tmp_path / "proxy.sock")
    proxy = await asyncio.start_unix_server(harness.handle, path=socket_path)
    try:
        code, stdout, stderr = await _run_forwarder(socket_path, _CHILD_THROUGH_PROXY)
    finally:
        proxy.close()
        echo.close()

    assert code == 7, stderr  # 子命令的退出码原样返回
    assert stdout.splitlines() == ["pypi.org:443 200:ping", "example.com:443 403"]


@pytest.mark.skipif(not UNIX_SOCKETS, reason="需要 Unix socket（Linux/macOS）")
@pytest.mark.asyncio
async def test_forwarder_fails_closed_without_proxy(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    code, _, stderr = await _run_forwarder(
        str(tmp_path / "missing.sock"),
        f"open({str(marker)!r}, 'w').close()",
    )

    assert code == egress_forwarder.PROXY_UNAVAILABLE_EXIT_CODE
    assert "出网代理不可用" in stderr
    assert not marker.exists()


# ---------------------------------------------------------------- 真实 Docker


_E2E_PROBE = r"""
import errno, os, socket, urllib.parse
proxy = urllib.parse.urlparse(os.environ["HTTPS_PROXY"])
def tunnel(target):
    s = socket.create_connection((proxy.hostname, proxy.port), timeout=30)
    s.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
    stream = s.makefile("rb")
    status = stream.readline().decode().split()[1]
    reason = ""
    while (header := stream.readline()) not in (b"\r\n", b""):
        if header.lower().startswith(b"x-muharness-denied:"):
            reason = header.decode().split(":", 1)[1].strip()
    s.close()
    return f"{status} {reason}".strip()
print("allowed", tunnel("pypi.org:443"))
print("not_listed", tunnel("example.com:443"))
print("internal", tunnel("host.docker.internal:443"))
try:
    socket.create_connection(("1.1.1.1", 443), timeout=5)
    print("direct connected")
except OSError as exc:
    print("direct", errno.errorcode.get(exc.errno, exc.errno))
"""


@pytest.mark.skipif(
    os.environ.get("MUHARNESS_RUN_DOCKER_E2E") != "1",
    reason="需要显式启用真实 Docker 集成测试",
)
@pytest.mark.asyncio
async def test_real_docker_shell_egress_allowlist(tmp_path: Path, docker_engine: Any) -> None:
    del docker_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "probe.py").write_text(_E2E_PROBE, encoding="utf-8")
    domains = ("pypi.org", "files.pythonhosted.org", "host.docker.internal")
    backend = DockerSandboxBackend(workspace, instance_id=f"e2e{os.getpid()}")
    supervisor = SandboxSupervisor(workspace, native_backend=backend)
    tool = ShellCommandTool(
        workspace, sandbox_supervisor=supervisor, allowed_domains=domains
    )
    try:
        probe = await tool.execute({"command": "python3 probe.py", "timeout_seconds": 60})
        pip = await tool.execute(
            {
                "command": (
                    "python3 -m pip download --no-deps --no-cache-dir --quiet "
                    "--dest /tmp/dl six==1.16.0 && ls /tmp/dl"
                ),
                "timeout_seconds": 120,
            }
        )
    finally:
        await supervisor.close()

    assert probe["exit_code"] == 0, probe
    # 每行形如 "allowed 403 host resolves to a non-public address"，失败时整段输出带上原因
    lines = dict(line.split(" ", 1) for line in probe["stdout"].splitlines())
    status = {key: value.split()[0] for key, value in lines.items()}
    assert status["allowed"] == "200", probe["stdout"]
    assert status["not_listed"] == "403", probe["stdout"]
    # 在白名单里但解析到宿主机/内网：Docker Desktop 上 403，原生 Linux 解析不到则 502
    assert status["internal"] in {"403", "502"}, probe["stdout"]
    assert status["direct"] in {"ENETUNREACH", "EHOSTUNREACH"}, probe["stdout"]
    assert pip["exit_code"] == 0, pip
    assert "six-1.16.0" in pip["stdout"]
