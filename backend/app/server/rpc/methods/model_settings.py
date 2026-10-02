
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.model_settings import ModelSettingsUpdate, ProviderSettingsUpdate
from app.models.errors import ModelAdapterError

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import INVALID_STATE, JsonRpcError, RpcErrorCode


# 函数说明：_invalid_params
# 用途：返回 `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`，提供 JSON-RPC 请求处
# 理 的派生值。
# 参数：
#   exc：传给 `str` 的输入，类型 `ValueError`。
# 返回：类型 `JsonRpcError`；返回 `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`
# 。
def _invalid_params(exc: ValueError) -> JsonRpcError:
    return JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))


# 函数说明：_view
# 用途：在JSON-RPC 请求处理中处理 `_view`，通过
# `application.model_settings_service.view` 完成首个内部处理步骤。
# 参数：
#   application：已装配的应用依赖，类型 `Any`。
# 返回：类型 `dict[str, Any]`；返回 `result`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `application.model_settings_service.view` → `result.update`。
def _view(application: Any) -> dict[str, Any]:
    result = application.model_settings_service.view(
        active_provider=application.provider,
        active_model=application.model,
        active_roles=application.active_model_roles,
    )
    active_run_ids = application.run_manager.active_run_ids
    result.update(
        {
            "restart_supported": application.host_restart_callback is not None,
            "restart_blocked_by_run_ids": list(active_run_ids),
            "can_restart": (
                application.host_restart_callback is not None
                and not active_run_ids
            ),
        }
    )
    return result


# 函数说明：model_settings_get
# 用途：获取模型设置，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；返回 `_view(application)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_view`。
async def model_settings_get(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:
    application = ctx.application
    return _view(application)


# 函数说明：model_settings_update
# 用途：更新模型设置，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；返回 `result`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettingsUpdate.model_validate` →
#  `ctx.application.model_settings_service.save` → `_invalid_params` → `_view`。
# 分支与异常：
#   捕获 `(ValidationError, ValueError)` 后，转换或抛出 `_invalid_params(exc)`。
async def model_settings_update(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:
    try:
        update = ModelSettingsUpdate.model_validate(params)
        ctx.application.model_settings_service.save(update)
    except (ValidationError, ValueError) as exc:
        raise _invalid_params(exc) from exc
    result = _view(ctx.application)
    result["restart_required"] = True
    return result


# 函数说明：model_settings_test
# 用途：在JSON-RPC 请求处理中处理 `model_settings_test`，通过
# `ProviderSettingsUpdate.model_validate` 完成首个内部处理步骤。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；返回
# `await ctx.application.model_settings_service.test(provider)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `ProviderSettingsUpdate.model_validate` →
# `ctx.application.model_settings_service.test` → `_invalid_params`。
# 分支与异常：
#   捕获 `(ValidationError, ValueError)` 后，转换或抛出 `_invalid_params(exc)`。
#   捕获 `ModelAdapterError` 后，转换或抛出
# `JsonRpcError(INVALID_STATE, f'模型连接失败：{exc}')`。
async def model_settings_test(
    params: dict[str, Any], ctx: RpcContext
) -> dict[str, Any]:
    try:
        provider = ProviderSettingsUpdate.model_validate(params)
        return await ctx.application.model_settings_service.test(provider)
    except (ValidationError, ValueError) as exc:
        raise _invalid_params(exc) from exc
    except ModelAdapterError as exc:
        raise JsonRpcError(INVALID_STATE, f"模型连接失败：{exc}") from exc


# 函数说明：register
# 用途：注册JSON-RPC 请求处理，供JSON-RPC 请求处理使用。
# 参数：
#   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dispatcher.register`。
def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("model_settings.get", model_settings_get)
    dispatcher.register("model_settings.update", model_settings_update)
    dispatcher.register("model_settings.test", model_settings_test)
