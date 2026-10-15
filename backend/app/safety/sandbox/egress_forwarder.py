"""沙箱内的出网转发器：把本机回环端口转发到出网代理的 Unix socket，再运行命令。

在 ``--network none`` 的沙箱容器里运行（``python3 -I -c <本文件源码> <socket> <命令...>``），
只能使用标准库。容器里没有任何网卡，只有回环；pip、npm、git 等通过
``HTTPS_PROXY=http://127.0.0.1:<端口>`` 找到这里，再经共享卷里的 socket 到达代理。

先确认代理可连接、本地端口已在监听，然后才启动命令；代理不可用时命令不会执行，
退出码 125，不会退回直连。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading

PROXY_UNAVAILABLE_EXIT_CODE = 125
PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def _connect(socket_path: str) -> socket.socket:
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.settimeout(5)
        upstream.connect(socket_path)
        upstream.settimeout(None)
    except OSError:
        upstream.close()
        raise
    return upstream


def _pump(source: socket.socket, target: socket.socket) -> None:
    try:
        while data := source.recv(65536):
            target.sendall(data)
    except OSError:
        pass
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _handle(client: socket.socket, socket_path: str) -> None:
    try:
        upstream = _connect(socket_path)
    except OSError:
        client.close()
        return
    reverse = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
    reverse.start()
    _pump(client, upstream)
    reverse.join()
    client.close()
    upstream.close()


def _accept_forever(listener: socket.socket, socket_path: str) -> None:
    while True:
        client, _ = listener.accept()
        threading.Thread(
            target=_handle, args=(client, socket_path), daemon=True
        ).start()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: egress_forwarder SOCKET COMMAND...", file=sys.stderr)
        return 2
    socket_path, command = argv[0], argv[1:]
    try:
        _connect(socket_path).close()
    except OSError as exc:
        print(
            f"muharness: 出网代理不可用（{exc}），命令未执行",
            file=sys.stderr,
            flush=True,
        )
        return PROXY_UNAVAILABLE_EXIT_CODE

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(32)
    port = listener.getsockname()[1]
    threading.Thread(
        target=_accept_forever, args=(listener, socket_path), daemon=True
    ).start()

    environment = dict(os.environ)
    proxy_url = f"http://127.0.0.1:{port}"
    for key in PROXY_ENV_KEYS:
        environment[key] = proxy_url
    environment.pop("NO_PROXY", None)
    environment.pop("no_proxy", None)
    code = subprocess.call(command, env=environment)
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
