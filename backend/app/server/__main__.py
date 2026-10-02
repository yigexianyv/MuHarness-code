

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any


# 函数说明：_parser
# 用途：在HTTP 服务与应用生命周期中处理 `_parser`，通过 `argparse.ArgumentParser` 完成首
# 个内部处理步骤。
# 返回：类型 `argparse.ArgumentParser`；返回 `parser`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`argparse.ArgumentParser` →
# `parser.add_argument`。
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MuHarness Host (FastAPI + JSON-RPC WebSocket)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--database",
        help="SQLite database path (default: backend/.muharness/muharness.db).",
    )
    parser.add_argument("--provider", help="Model provider (default: auto-select).")
    parser.add_argument("--model", help="Override the configured model name.")
    parser.add_argument(
        "--mcp-config",
        help="Path to the MCP Server JSON configuration file.",
    )
    return parser


# 函数说明：_application_kwargs
# 用途：处理HTTP 服务与应用生命周期中的 `_application_kwargs` 数据；结果及边界条件见下方
# 说明。
# 参数：
#   args：`args`输入或配置值，类型 `argparse.Namespace`。
# 返回：类型 `dict[str, object]`；返回 `application_kwargs`。
def _application_kwargs(args: argparse.Namespace) -> dict[str, object]:

    application_kwargs: dict[str, object] = {}
    if args.database:
        application_kwargs["database"] = args.database
    if args.mcp_config:
        application_kwargs["mcp_config"] = args.mcp_config

    return application_kwargs


# 函数说明：_serve
# 用途：在HTTP 服务与应用生命周期中处理 `_serve`，通过 `uvicorn.Config` 完成首个内部处理
# 步骤。
# 参数：
#   args：传给 `_application_kwargs` 的输入，类型 `argparse.Namespace`。
# 返回：类型 `int`；按分支返回 `2`；`0`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Application` → `_application_kwargs`
# → `create_app` → `uvicorn.Config` → `uvicorn.Server` → `server.serve`；另有 2 个调用点
# 。
# 分支与异常：
#   捕获 `ValueError` 后，返回 `2`。
#   当 `not restart_requested` 时，返回 `0`。
async def _serve(args: argparse.Namespace) -> int:

    import uvicorn

    from app.application import Application
    from app.server.app import create_app

    while True:
        restart_requested = False
        server: Any = None

        # 函数说明：_serve.request_restart
        # 用途：处理HTTP 服务与应用生命周期中的 `request_restart` 数据；结果及边界条件见
        # 下方说明。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        # 副作用与资源：
        #   更新对象字段：`server.should_exit`。
        # 闭包依赖：从外层读取 `server`。
        def request_restart() -> None:
            nonlocal restart_requested
            restart_requested = True
            if server is not None:
                server.should_exit = True

        try:
            application = Application(
                provider=args.provider,
                model=args.model,
                **_application_kwargs(args),
            )
        except ValueError as exc:
            print(f"启动失败：{exc}", file=sys.stderr)
            return 2

        app = create_app(application, restart_callback=request_restart)
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
        if not restart_requested:
            return 0
        logging.getLogger("muharness.server").info(
            "Restarting MuHarness Host with saved configuration"
        )


# 函数说明：main
# 用途：运行HTTP 服务与应用生命周期入口，按照当前参数装配依赖并驱动主流程。
# 返回：类型 `int`；按分支返回 `asyncio.run(_serve(args))`；`0`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_parser().parse_args` → `_parser` →
# `logging.basicConfig` → `asyncio.run` → `_serve`。
# 分支与异常：
#   捕获 `KeyboardInterrupt` 后，返回 `0`。
def main() -> int:
    args = _parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        return asyncio.run(_serve(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
