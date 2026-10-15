"""工作区环境准备：由用户在界面上触发，模型无法调用这些方法。"""

from __future__ import annotations

from typing import Any

from app.runtime.workspace_setup import (
    DEFAULT_SETUP_SCRIPT,
    WorkspaceSetupBusy,
    WorkspaceSetupError,
    WorkspaceSetupService,
)

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import INVALID_STATE, JsonRpcError, RpcErrorCode


async def workspace_setup_start(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:
    service = _require_service(ctx)
    script_path = params.get("script_path", DEFAULT_SETUP_SCRIPT)
    verify_command = params.get("verify_command")
    if not isinstance(script_path, str) or not script_path.strip():
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS, "script_path must be a non-empty string"
        )
    if verify_command is not None and not isinstance(verify_command, str):
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS, "verify_command must be a string"
        )
    try:
        record = service.start(
            script_path.strip(),
            verify_command=verify_command,
            busy_reason=ctx.application.workspace_busy_reason(),
        )
    except WorkspaceSetupBusy as exc:
        raise JsonRpcError(INVALID_STATE, str(exc)) from exc
    except WorkspaceSetupError as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"setup": record}


async def workspace_setup_status(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:
    service = _require_service(ctx)
    return {"current": service.current, "last": service.last}


async def workspace_setup_cancel(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:
    service = _require_service(ctx)
    cancelled = await service.cancel()
    return {"cancelled": cancelled, "last": service.last}


def _require_service(ctx: RpcContext) -> WorkspaceSetupService:
    service = ctx.application.workspace_setup
    if service is None:
        raise JsonRpcError(INVALID_STATE, "workspace setup not available")
    return service


def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("workspace.setup.start", workspace_setup_start)
    dispatcher.register("workspace.setup.status", workspace_setup_status)
    dispatcher.register("workspace.setup.cancel", workspace_setup_cancel)
