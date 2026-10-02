
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import FileResponse

from app.application import Application

from .rpc import (
    RpcBroadcastEventHandler,
    RpcConnection,
    RpcHub,
    build_dispatcher,
)
from .version import __version__

logger = logging.getLogger("muharness.server")

_ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


# 函数说明：artifact_content
# 用途：校验产物访问并返回本地文件响应。
# 参数：
#   artifact_id：交付物标识，类型 `str`。
#   request：待处理的请求对象，类型 `Request`。
# 返回：类型 `FileResponse`；返回 `FileResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_ARTIFACT_ID_RE.fullmatch` →
# `service.file_path` → `file_path.resolve` → `resolved.relative_to` →
# `resolved.is_file` → `FileResponse`。
# 分支与异常：
#   当 `client_host not in _ALLOWED_LOOPBACK_HOSTS` 时，抛出
# `HTTPException(status_code=403, detail='forbidden')`。
#   当 `not _ARTIFACT_ID_RE.fullmatch(artifact_id)` 时，抛出
# `HTTPException(status_code=404, detail='not found')`。
#   当 `service is None` 时，抛出 `HTTPException(status_code=404, detail='not found')`。
#   当 `file_path is None` 时，抛出 `HTTPException(status_code=404, detail='not found')`
# 。
#   捕获 `(OSError, ValueError)` 后，转换或抛出
# `HTTPException(status_code=404, detail='not found')`。
async def artifact_content(artifact_id: str, request: Request) -> FileResponse:

    """校验产物访问并返回本地文件响应。"""
    client_host = request.client.host if request.client is not None else ""
    if client_host not in _ALLOWED_LOOPBACK_HOSTS:
        raise HTTPException(status_code=403, detail="forbidden")

    if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise HTTPException(status_code=404, detail="not found")

    application = request.app.state.application
    service = application.artifact_service
    if service is None:
        raise HTTPException(status_code=404, detail="not found")

    file_path = await service.file_path(artifact_id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="not found")

    try:
        resolved = file_path.resolve()
        resolved.relative_to(service.managed_dir)
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="not found")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="not found")

    artifact = await service.store.get(artifact_id)
    mime = (
        artifact.mime_type
        if artifact and artifact.mime_type
        else "application/octet-stream"
    )
    filename = artifact.filename if artifact and artifact.filename else "artifact"

    return FileResponse(
        path=resolved,
        media_type=mime,
        filename=filename,
        content_disposition_type="attachment",
        headers={
            "Cache-Control": "no-store",
        },
    )


# 函数说明：create_app
# 用途：好评，创建 FastAPI 应用，绑定生命周期、健康检查、产物下载与 RPC 路由。
# 参数：
#   application：已装配的应用依赖，类型 `Application | None`；默认 `None`。
#   restart_callback：`restart_callback`输入或配置值，类型 `Callable[[], None] | None`；
# 默认 `None`。
# 返回：类型 `FastAPI`；返回 `app`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Application` → `RpcHub` →
# `RpcBroadcastEventHandler` → `build_dispatcher` → `FastAPI`。
# 副作用与资源：
#   更新对象字段：`application.web_approval`、`application.host_restart_callback`、
# `application.shared_event_handler`、`app.state.application`、`app.state.hub`、
# `app.state.dispatcher`。
def create_app(
    application: Application | None = None,
    *,
    restart_callback: Callable[[], None] | None = None,
) -> FastAPI:

    """好评，创建 FastAPI 应用，绑定生命周期、健康检查、产物下载与 RPC 路由。"""
    if application is None:
        application = Application(web_approval=True)
    else:
        application.web_approval = True
    application.host_restart_callback = restart_callback

    hub = RpcHub()
    broadcast = RpcBroadcastEventHandler(hub)
    application.shared_event_handler = broadcast

    dispatcher = build_dispatcher()

    # 函数说明：create_app.lifespan
    # 用途：在HTTP 服务与应用生命周期中处理 `lifespan`，通过 `application.start` 完成首
    # 个内部处理步骤。
    # 参数：
    #   app：应用实例，类型 `FastAPI`。
    # 返回：异步生成器，逐项产出 `None`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`application.start` →
    # `approval_gate.set_broadcaster` → `artifact_service.set_broadcaster` →
    # `application.mea_events.set_broadcaster` → `logger.info` → `application.close`。
    # 闭包依赖：从外层读取 `application`、`hub`。
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await application.start()

        approval_gate = application.web_approval_gate
        if approval_gate is not None:
            approval_gate.set_broadcaster(hub.broadcast)

        artifact_service = application.artifact_service
        if artifact_service is not None:
            artifact_service.set_broadcaster(hub.broadcast)
        application.mea_events.set_broadcaster(hub.broadcast)
        logger.info(
            "MuHarness Host started · provider=%s · model=%s",
            application.provider,
            application.model,
        )
        try:
            yield
        finally:
            await application.close()
            logger.info("MuHarness Host stopped")

    app = FastAPI(
        title="MuHarness Host",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.application = application
    app.state.hub = hub
    app.state.dispatcher = dispatcher


    # 函数说明：create_app.health
    # 用途：处理HTTP 服务与应用生命周期中的 `health` 数据；结果及边界条件见下方说明。
    # 参数：
    #   request：待处理的请求对象，类型 `Request`。
    # 返回：类型 `dict[str, object]`；字典，包含字段 `status`、`provider`、`model`、
    # `version`。
    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        current: Application = request.app.state.application
        return {
            "status": "ok",
            "provider": current.provider,
            "model": current.model,
            "version": __version__,
        }

    # 函数说明：create_app.artifact_content_route
    # 用途：返回 `await artifact_content(artifact_id, request)`，提供 HTTP 服务与应用生
    # 命周期 的派生值。
    # 参数：
    #   artifact_id：交付物标识，类型 `str`。
    #   request：待处理的请求对象，类型 `Request`。
    # 返回：类型 `Response`；返回 `await artifact_content(artifact_id, request)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`artifact_content`。
    @app.get("/artifacts/{artifact_id}/content")
    async def artifact_content_route(
        artifact_id: str, request: Request
    ) -> Response:
        return await artifact_content(artifact_id, request)

    # 函数说明：create_app.rpc_endpoint
    # 用途：在HTTP 服务与应用生命周期中处理 `rpc_endpoint`，通过 `websocket.accept` 完成
    # 首个内部处理步骤。
    # 参数：
    #   websocket：当前 WebSocket 连接，类型 `WebSocket`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`websocket.accept` →
    # `RpcConnection` → `hub.register` → `connection.run` → `hub.unregister`。
    # 闭包依赖：从外层读取 `application`、`dispatcher`、`hub`。
    @app.websocket("/rpc")
    async def rpc_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()

        connection = RpcConnection(websocket, dispatcher, application, hub)
        await hub.register(connection)
        try:
            await connection.run()
        finally:
            await hub.unregister(connection)
    return app
__all__ = ["__version__", "create_app"]
