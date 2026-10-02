
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.domain.skills import SkillScope
from app.integrations.extensions import (
    ExtensionImportError,
    apply_import_plan,
    parse_import_plan,
)
from app.integrations.mcp import MCPConfigurationError, MCPServerConfig

from ..dispatcher import RpcContext, RpcDispatcher
from ..protocol import JsonRpcError, RpcErrorCode


# 函数说明：extension_list
# 用途：列出`extension`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `skills`、`skill_diagnostics`、`mcp`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_store.managed_catalog` →
# `skill_store.diagnostics` → `config_store.load` → `manager.statuses` →
# `config_store.restart_required`。
# 分支与异常：
#   捕获 `MCPConfigurationError` 后，执行异常处理调用 `str`。
async def extension_list(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    del params
    skill_store = ctx.application.skill_store
    skills = (
        await skill_store.managed_catalog() if skill_store is not None else ()
    )
    diagnostics = skill_store.diagnostics() if skill_store is not None else ()

    config_store = ctx.application.mcp_config_store
    try:
        settings = await config_store.load()
        config_error = ctx.application.mcp_error
    except MCPConfigurationError as exc:
        settings = None
        config_error = str(exc)

    manager = ctx.application.mcp_manager
    statuses = {item.name: item for item in manager.statuses()} if manager else {}
    servers: list[dict[str, Any]] = []
    if settings is not None:
        for config in settings.servers:
            status = statuses.get(config.name)
            restart_required = config_store.restart_required(config.name)
            servers.append(
                {
                    "name": config.name,
                    "command": config.command,
                    "args": list(config.args),
                    "cwd": config.cwd,
                    "enabled": config.enabled,
                    "permission": config.permission.value,
                    "env_names": sorted(config.env),
                    "sandbox": config.sandbox.model_dump(mode="json"),
                    "sandboxed": status.sandboxed if status else None,
                    "sandbox_backend": status.sandbox_backend if status else None,
                    "state": (
                        "restart_required"
                        if restart_required or status is None
                        else status.state.value
                    ),
                    "tool_names": list(status.tool_names) if status else [],
                    "error": status.error if status else None,
                }
            )

    return {
        "skills": [
            {
                "name": item.metadata.name,
                "description": item.metadata.description,
                "scope": item.metadata.scope.value,
                "location": str(item.metadata.location),
                "enabled": item.enabled,
            }
            for item in skills
        ],
        "skill_diagnostics": [
            {
                "name": item.name,
                "scope": item.scope.value,
                "location": item.location,
                "reason": item.reason,
            }
            for item in diagnostics
        ],
        "mcp": {
            "config_path": str(config_store.path),
            "error": config_error,
            "restart_required": config_store.has_pending_changes,
            "servers": servers,
        },
    }


# 函数说明：extension_import_preview
# 用途：导入`preview`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `skill_scope`、`input`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `plan`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillScope` → `_import_permission` →
# `parse_import_plan` → `_require_str` → `plan.public_dict`。
# 分支与异常：
#   捕获 `(ValueError, ExtensionImportError)` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
async def extension_import_preview(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:

    del ctx
    try:
        scope = SkillScope(str(params.get("skill_scope", "project")))
        permission = _import_permission(params)
        plan = parse_import_plan(
            _require_str(params, "input"),
            skill_scope=scope,
            mcp_permission=permission,
        )
    except (ValueError, ExtensionImportError) as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"plan": plan.public_dict()}


# 函数说明：extension_import_apply
# 用途：导入`apply`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `confirmed`、`skill_scope`
# 、`input`、`fingerprint`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；返回 `result`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillScope` → `_import_permission` →
# `parse_import_plan` → `_require_str` → `apply_import_plan`。
# 分支与异常：
#   当 `params.get('confirmed') is not True` 时，抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, '必须先预览并明确确认导入')`。
#   当 `skill_store is None or mcp_store is None` 时，抛出 `JsonRpcError(…)`。
#   当 `plan.fingerprint != _require_str(params, 'fingerprint')` 时，抛出
# `ExtensionImportError('导入内容已变化，请重新生成预览')`。
#   捕获
# `(ExtensionImportError, MCPConfigurationError, OSError, RuntimeError, ValueError)` 后
# ，转换或抛出 `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
async def extension_import_apply(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:

    if params.get("confirmed") is not True:
        raise JsonRpcError(
            RpcErrorCode.INVALID_PARAMS,
            "必须先预览并明确确认导入",
        )
    skill_store = ctx.application.skill_store
    mcp_store = ctx.application.mcp_config_store
    if skill_store is None or mcp_store is None:
        raise JsonRpcError(RpcErrorCode.INTERNAL_ERROR, "Extension Store unavailable")
    try:
        scope = SkillScope(str(params.get("skill_scope", "project")))
        permission = _import_permission(params)
        plan = parse_import_plan(
            _require_str(params, "input"),
            skill_scope=scope,
            mcp_permission=permission,
        )
        if plan.fingerprint != _require_str(params, "fingerprint"):
            raise ExtensionImportError("导入内容已变化，请重新生成预览")
        result = await apply_import_plan(
            plan,
            skill_store=skill_store,
            mcp_store=mcp_store,
        )
    except (
        ExtensionImportError,
        MCPConfigurationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return result


# 函数说明：skill_install
# 用途：在JSON-RPC 请求处理中处理 `skill_install`，通过 `params.get` 完成首个内部处理步
# 骤。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `name`、`description`、
# `instructions`、`scope`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `skill`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` → `SkillScope` →
# `store.install`。
# 分支与异常：
#   当 `store is None` 时，抛出 `JsonRpcError(…)`。
#   捕获 `(ValueError, OSError)` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
async def skill_install(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    store = ctx.application.skill_store
    if store is None:
        raise JsonRpcError(RpcErrorCode.INTERNAL_ERROR, "Skill Store unavailable")
    name = _require_str(params, "name")
    description = _require_str(params, "description")
    instructions = _require_str(params, "instructions")
    try:
        scope = SkillScope(str(params.get("scope", SkillScope.PROJECT.value)))
        skill = await store.install(
            name=name,
            description=description,
            instructions=instructions,
            scope=scope,
        )
    except (ValueError, OSError) as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {
        "skill": {
            "name": skill.metadata.name,
            "description": skill.metadata.description,
            "scope": skill.metadata.scope.value,
            "location": str(skill.metadata.location),
            "enabled": True,
        }
    }


# 函数说明：skill_set_enabled
# 用途：设置`enabled`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `name`、`scope`、`enabled`
# 。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `skill`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.set_enabled` → `_require_str` →
#  `SkillScope` → `_require_bool`。
# 分支与异常：
#   当 `store is None` 时，抛出 `JsonRpcError(…)`。
#   捕获 `(KeyError, ValueError, OSError, RuntimeError)` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
async def skill_set_enabled(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    store = ctx.application.skill_store
    if store is None:
        raise JsonRpcError(RpcErrorCode.INTERNAL_ERROR, "Skill Store unavailable")
    try:
        entry = await store.set_enabled(
            name=_require_str(params, "name"),
            scope=SkillScope(_require_str(params, "scope")),
            enabled=_require_bool(params, "enabled"),
        )
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {
        "skill": {
            "name": entry.metadata.name,
            "description": entry.metadata.description,
            "scope": entry.metadata.scope.value,
            "location": str(entry.metadata.location),
            "enabled": entry.enabled,
        }
    }


# 函数说明：skill_delete
# 用途：删除技能，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `name`、`scope`、`enabled`
# 。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `deleted`、`name`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` → `store.delete` →
# `SkillScope` → `_require_bool`。
# 分支与异常：
#   当 `store is None` 时，抛出 `JsonRpcError(…)`。
#   捕获 `(KeyError, ValueError, OSError)` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
async def skill_delete(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    store = ctx.application.skill_store
    if store is None:
        raise JsonRpcError(RpcErrorCode.INTERNAL_ERROR, "Skill Store unavailable")
    name = _require_str(params, "name")
    try:
        await store.delete(
            name=name,
            scope=SkillScope(_require_str(params, "scope")),
            enabled=_require_bool(params, "enabled"),
        )
    except (KeyError, ValueError, OSError) as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"deleted": True, "name": name}


# 函数说明：mcp_add
# 用途：添加`mcp`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `name`、`command`、`args`、
# `env`、`cwd`、`enabled`、`startup_timeout_seconds`、`call_timeout_seconds`、
# `permission`、`sandbox`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `server`、`restart_required`、
# `config_path`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerConfig.model_validate` →
# `ctx.application.mcp_config_store.add`。
# 分支与异常：
#   捕获 `(ValidationError, ValueError, MCPConfigurationError, OSError)` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
async def mcp_add(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    try:
        server = MCPServerConfig.model_validate(
            {
                "name": params.get("name"),
                "transport": "stdio",
                "command": params.get("command"),
                "args": params.get("args", []),
                "env": params.get("env", {}),
                "cwd": params.get("cwd"),
                "enabled": params.get("enabled", True),
                "startup_timeout_seconds": params.get(
                    "startup_timeout_seconds", 15.0
                ),
                "call_timeout_seconds": params.get(
                    "call_timeout_seconds", 30.0
                ),
                "permission": params.get("permission", "human_approval"),
                "sandbox": params.get("sandbox", {}),
            }
        )
        await ctx.application.mcp_config_store.add(server)
    except (ValidationError, ValueError, MCPConfigurationError, OSError) as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {
        "server": {
            "name": server.name,
            "command": server.command,
            "args": list(server.args),
            "cwd": server.cwd,
            "enabled": server.enabled,
            "permission": server.permission.value,
            "env_names": sorted(server.env),
            "sandbox": server.sandbox.model_dump(mode="json"),
            "sandboxed": None,
            "sandbox_backend": None,
            "state": "restart_required",
            "tool_names": [],
            "error": None,
        },
        "restart_required": True,
        "config_path": str(ctx.application.mcp_config_store.path),
    }


# 函数说明：mcp_set_enabled
# 用途：设置`enabled`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `name`、`enabled`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `server`、`restart_required`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `ctx.application.mcp_config_store.set_enabled` → `_require_str` → `_require_bool` →
# `_server_dict`。
# 分支与异常：
#   捕获 `(KeyError, ValueError, MCPConfigurationError, OSError)` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
async def mcp_set_enabled(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    try:
        server = await ctx.application.mcp_config_store.set_enabled(
            _require_str(params, "name"),
            enabled=_require_bool(params, "enabled"),
        )
    except (KeyError, ValueError, MCPConfigurationError, OSError) as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {
        "server": _server_dict(server, state="restart_required"),
        "restart_required": True,
    }


# 函数说明：mcp_delete
# 用途：删除`mcp`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `name`。
#   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `RpcContext`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `deleted`、`name`、`restart_required`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_require_str` →
# `ctx.application.mcp_config_store.delete`。
# 分支与异常：
#   捕获 `(KeyError, ValueError, MCPConfigurationError, OSError)` 后，转换或抛出
# `JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc))`。
async def mcp_delete(
    params: dict[str, Any],
    ctx: RpcContext,
) -> dict[str, Any]:
    name = _require_str(params, "name")
    try:
        await ctx.application.mcp_config_store.delete(name)
    except (KeyError, ValueError, MCPConfigurationError, OSError) as exc:
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"deleted": True, "name": name, "restart_required": True}


# 函数说明：_require_str
# 用途：获取并校验必需的`str`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
# 返回：类型 `str`；返回 `value.strip()`。
# 分支与异常：
#   当 `not isinstance(value, str) or not value.strip()` 时，抛出 `JsonRpcError(…)`。
def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, f"{key} is required")
    return value.strip()


# 函数说明：_require_bool
# 用途：获取并校验必需的`bool`，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
#   key：字段名或查询键，类型 `str`。
# 返回：类型 `bool`；返回 `value`。
# 分支与异常：
#   当 `not isinstance(value, bool)` 时，抛出 `JsonRpcError(…)`。
def _require_bool(params: dict[str, Any], key: str) -> bool:
    value = params.get(key)
    if not isinstance(value, bool):
        raise JsonRpcError(RpcErrorCode.INVALID_PARAMS, f"{key} must be boolean")
    return value


# 函数说明：_import_permission
# 用途：导入权限，供JSON-RPC 请求处理使用。
# 参数：
#   params：JSON-RPC 方法参数，类型 `dict[str, Any]`；读取键 `mcp_permission`。
# 返回：类型 `str`；返回 `value`。
# 分支与异常：
#   当 `value not in {'allowed', 'human_approval', 'forbidden'}` 时，抛出
# `ValueError('mcp_permission is invalid')`。
def _import_permission(params: dict[str, Any]) -> str:
    value = str(params.get("mcp_permission", "human_approval"))
    if value not in {"allowed", "human_approval", "forbidden"}:
        raise ValueError("mcp_permission is invalid")
    return value


# 函数说明：_server_dict
# 用途：返回
# `{'name': server.name, 'command': server.command, 'args': list(server.args), 'cwd':…`
# ，提供 JSON-RPC 请求处理 的派生值。
# 参数：
#   server：服务输入或配置值，类型 `MCPServerConfig`。
#   state：当前状态快照，类型 `str`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `name`、`command`、`args`、`cwd`、
# `enabled`、`permission`、`env_names`、`sandbox`、`sandboxed`、`sandbox_backend`、
# `state`、`tool_names`、`error`。
def _server_dict(
    server: MCPServerConfig,
    *,
    state: str,
) -> dict[str, Any]:
    return {
        "name": server.name,
        "command": server.command,
        "args": list(server.args),
        "cwd": server.cwd,
        "enabled": server.enabled,
        "permission": server.permission.value,
        "env_names": sorted(server.env),
        "sandbox": server.sandbox.model_dump(mode="json"),
        "sandboxed": None,
        "sandbox_backend": None,
        "state": state,
        "tool_names": [],
        "error": None,
    }


# 函数说明：register
# 用途：注册JSON-RPC 请求处理，供JSON-RPC 请求处理使用。
# 参数：
#   dispatcher：JSON-RPC 方法分发器，类型 `RpcDispatcher`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dispatcher.register`。
def register(dispatcher: RpcDispatcher) -> None:
    dispatcher.register("extension.list", extension_list)
    dispatcher.register("extension.import.preview", extension_import_preview)
    dispatcher.register("extension.import.apply", extension_import_apply)
    dispatcher.register("skill.install", skill_install)
    dispatcher.register("skill.set_enabled", skill_set_enabled)
    dispatcher.register("skill.delete", skill_delete)
    dispatcher.register("mcp.add", mcp_add)
    dispatcher.register("mcp.set_enabled", mcp_set_enabled)
    dispatcher.register("mcp.delete", mcp_delete)


__all__ = [
    "extension_list",
    "extension_import_apply",
    "extension_import_preview",
    "mcp_add",
    "mcp_delete",
    "mcp_set_enabled",
    "register",
    "skill_delete",
    "skill_install",
    "skill_set_enabled",
]
