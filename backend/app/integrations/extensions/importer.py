
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import shlex
import stat
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import httpx

from app.domain.skills import SkillParseError, SkillScope, parse_skill_document
from app.integrations.mcp import MCPServerConfig

if TYPE_CHECKING:
    from app.domain.skills import SkillStore
    from app.integrations.mcp import MCPConfigurationStore

_GITHUB_REPOSITORY_RE = re.compile(
    r"^(?:https://github\.com/)?"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,38}))"
    r"/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_MCP_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_MAX_IMPORT_TEXT_CHARS = 200_000
_MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
_MAX_SKILL_PACKAGE_BYTES = 10 * 1024 * 1024
_RESOURCE_DIRS = frozenset({"scripts", "references", "assets"})


class ExtensionImportError(ValueError):
    pass


@dataclass(frozen=True)
class GitHubSkillSource:

    owner: str
    repository: str
    scope: SkillScope

    # 函数说明：GitHubSkillSource.slug
    # 用途：返回 `f'{self.owner}/{self.repository}'`，提供 GitHubSkillSource 的派生值。
    # 返回：类型 `str`；返回 `f'{self.owner}/{self.repository}'`。
    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repository}"

    # 函数说明：GitHubSkillSource.url
    # 用途：返回 `f'https://github.com/{self.slug}'`，提供 GitHubSkillSource 的派生值。
    # 返回：类型 `str`；返回 `f'https://github.com/{self.slug}'`。
    @property
    def url(self) -> str:
        return f"https://github.com/{self.slug}"


@dataclass(frozen=True)
class ExtensionImportPlan:

    fingerprint: str
    raw_input: str
    skill_sources: tuple[GitHubSkillSource, ...]
    mcp_servers: tuple[MCPServerConfig, ...]
    warnings: tuple[str, ...]

    # 函数说明：ExtensionImportPlan.public_dict
    # 用途：将当前记录转为字典载荷，具体公开字段及转换规则由返回表达式确定。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `fingerprint`、`items`、`actions`、
    # `warnings`、`requires_download`、`requires_restart`。
    def public_dict(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        actions: list[str] = []
        for source in self.skill_sources:
            items.append(
                {
                    "kind": "skill",
                    "name": source.repository,
                    "source": source.url,
                    "scope": source.scope.value,
                    "summary": "确认后下载仓库并安装其中通过校验的 Skill 包",
                }
            )
            actions.extend(
                (
                    f"下载 {source.url} 的静态仓库归档",
                    "检查 SKILL.md，并只复制 scripts / references / assets",
                    "不会执行 npx、git、安装脚本或仓库中的代码",
                )
            )
        for server in self.mcp_servers:
            command = shlex.join((server.command, *server.args))
            items.append(
                {
                    "kind": "mcp",
                    "name": server.name,
                    "source": "外部 MCP 配置",
                    "summary": f"写入 stdio Server；重启 Host 后运行 {command}",
                    "command": server.command,
                    "args": list(server.args),
                    "cwd": server.cwd,
                    "env_names": sorted(server.env),
                    "permission": server.permission.value,
                    "sandbox": server.sandbox.model_dump(mode="json"),
                }
            )
            actions.append(f"写入 MCP {server.name}：{command}")
        return {
            "fingerprint": self.fingerprint,
            "items": items,
            "actions": actions,
            "warnings": list(self.warnings),
            "requires_download": bool(self.skill_sources),
            "requires_restart": bool(self.mcp_servers),
        }


# 函数说明：parse_import_plan
# 用途：解析计划，供扩展配置导入使用。
# 参数：
#   raw_input：传给 `html.unescape` 的输入，类型 `str`。
#   skill_scope：传给 `_parse_skill_source_or_command` 的输入，类型 `SkillScope`；默认
# `SkillScope.PROJECT`。
#   mcp_permission：传给 `raw_server.get` 的输入，类型 `str`；默认 `'human_approval'`。
# 返回：类型 `ExtensionImportPlan`；返回 `ExtensionImportPlan(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`html.unescape` → `_try_json` →
# `_parse_skill_source_or_command` → `_external_servers` → `_command_and_args` →
# `_skill_add_source`；另有 9 个调用点。
# 分支与异常：
#   当 `not cleaned` 时，抛出
# `ExtensionImportError('请粘贴 GitHub 地址、owner/repo 或 MCP JSON')`。
#   当 `len(cleaned) > _MAX_IMPORT_TEXT_CHARS` 时，抛出
# `ExtensionImportError('导入内容过大')`。
#   `skill_slug is not None` 分支在完成前置处理后跳过当前循环项。
#   捕获 `(TypeError, ValueError)` 后，转换或抛出
# `ExtensionImportError(f'MCP {external_name!r} 配置无效：{exc}')`。
def parse_import_plan(
    raw_input: str,
    *,
    skill_scope: SkillScope = SkillScope.PROJECT,
    mcp_permission: str = "human_approval",
) -> ExtensionImportPlan:

    cleaned = html.unescape(raw_input).strip()
    if not cleaned:
        raise ExtensionImportError("请粘贴 GitHub 地址、owner/repo 或 MCP JSON")
    if len(cleaned) > _MAX_IMPORT_TEXT_CHARS:
        raise ExtensionImportError("导入内容过大")

    skill_sources: list[GitHubSkillSource] = []
    mcp_servers: list[MCPServerConfig] = []
    warnings: list[str] = []
    payload = _try_json(cleaned)
    if payload is None:
        source = _parse_skill_source_or_command(cleaned, skill_scope)
        skill_sources.append(source)
    else:
        raw_servers = _external_servers(payload)
        for external_name, raw_server in raw_servers:
            command, args = _command_and_args(raw_server)
            skill_slug = _skill_add_source(command, args)
            if skill_slug is not None:
                skill_sources.append(
                    _parse_github_source(skill_slug, skill_scope)
                )
                continue
            normalized_name = _normalize_mcp_name(external_name)
            if normalized_name != external_name:
                warnings.append(
                    f"MCP 名称 {external_name!r} 已转换为 {normalized_name!r}"
                )
            env = _string_mapping(raw_server.get("env", {}), "env")
            if any(not _is_env_reference(value) for value in env.values()):
                warnings.append(
                    f"MCP {normalized_name} 含直接环境变量值；"
                    "建议改为 ${ENV_NAME} 引用"
                )
            try:
                mcp_servers.append(
                    MCPServerConfig.model_validate(
                        {
                            "name": normalized_name,
                            "transport": "stdio",
                            "command": command,
                            "args": args,
                            "env": env,
                            "cwd": raw_server.get("cwd"),
                            "enabled": raw_server.get("enabled", True),
                            "startup_timeout_seconds": raw_server.get(
                                "startup_timeout_seconds", 15.0
                            ),
                            "call_timeout_seconds": raw_server.get(
                                "call_timeout_seconds", 30.0
                            ),
                            "permission": raw_server.get(
                                "permission", mcp_permission
                            ),
                            "sandbox": raw_server.get("sandbox", {}),
                        }
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ExtensionImportError(
                    f"MCP {external_name!r} 配置无效：{exc}"
                ) from exc

    _ensure_unique(skill_sources, mcp_servers)
    fingerprint = _fingerprint(cleaned, skill_scope, mcp_permission)
    return ExtensionImportPlan(
        fingerprint=fingerprint,
        raw_input=cleaned,
        skill_sources=tuple(skill_sources),
        mcp_servers=tuple(mcp_servers),
        warnings=tuple(dict.fromkeys(warnings)),
    )


# 函数说明：apply_import_plan
# 用途：应用计划，供扩展配置导入使用。
# 参数：
#   plan：计划输入或配置值，类型 `ExtensionImportPlan`。
#   skill_store：技能存储，类型 `SkillStore`。
#   mcp_store：`mcp`持久化存储依赖，类型 `MCPConfigurationStore`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `skills`、`mcp_servers`、
# `restart_required`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_download_github_archive` →
# `_skill_packages_from_archive` → `skill_store.install_package` → `mcp_store.add_many`
# 。
# 分支与异常：
#   当 `not discovered` 时，抛出 `ExtensionImportError(…)`。
async def apply_import_plan(
    plan: ExtensionImportPlan,
    *,
    skill_store: SkillStore,
    mcp_store: MCPConfigurationStore,
) -> dict[str, Any]:

    packages: list[tuple[GitHubSkillSource, dict[str, bytes]]] = []
    for source in plan.skill_sources:
        archive = await _download_github_archive(source)
        discovered = _skill_packages_from_archive(archive)
        if not discovered:
            raise ExtensionImportError(
                f"{source.slug} 中没有找到可由 MuHarness 加载的 SKILL.md"
            )
        packages.extend((source, package) for package in discovered)

    installed_skills: list[dict[str, str]] = []
    for source, package in packages:
        skill = await skill_store.install_package(
            files=package,
            scope=source.scope,
        )
        installed_skills.append(
            {
                "name": skill.metadata.name,
                "scope": skill.metadata.scope.value,
                "source": source.url,
            }
        )

    if plan.mcp_servers:
        await mcp_store.add_many(plan.mcp_servers)

    return {
        "skills": installed_skills,
        "mcp_servers": [server.name for server in plan.mcp_servers],
        "restart_required": bool(plan.mcp_servers),
    }


# 函数说明：_try_json
# 用途：在扩展配置导入中处理 `_try_json`，通过 `json.loads` 完成首个内部处理步骤。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `dict[str, Any] | None`；按分支返回 `None`；`payload`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads`。
# 分支与异常：
#   捕获 `json.JSONDecodeError` 后，返回 `None`。
#   当 `not isinstance(payload, dict)` 时，抛出
# `ExtensionImportError('MCP JSON 顶层必须是对象')`。
def _try_json(value: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        raise ExtensionImportError("MCP JSON 顶层必须是对象")
    return payload


# 函数说明：_external_servers
# 用途：在扩展配置导入中处理 `_external_servers`，通过 `servers.items` 完成首个内部处理
# 步骤。
# 参数：
#   payload：传输或持久化载荷，类型 `dict[str, Any]`；读取键 `mcpServers`、`servers`。
# 返回：类型 `list[tuple[str, dict[str, Any]]]`；返回 `result`。
# 分支与异常：
#   `'mcpServers' in payload` 分支在完成前置处理后返回 `result`。
#   当 `not isinstance(servers, dict) or not servers` 时，抛出
# `ExtensionImportError('mcpServers 必须是非空对象')`。
#   当 `not isinstance(name, str) or not isinstance(config, dict)` 时，抛出
# `ExtensionImportError('mcpServers 中的名称和配置必须有效')`。
#   `'servers' in payload` 分支在完成前置处理后返回 `result`。
def _external_servers(
    payload: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    if "mcpServers" in payload:
        servers = payload["mcpServers"]
        if not isinstance(servers, dict) or not servers:
            raise ExtensionImportError("mcpServers 必须是非空对象")
        result: list[tuple[str, dict[str, Any]]] = []
        for name, config in servers.items():
            if not isinstance(name, str) or not isinstance(config, dict):
                raise ExtensionImportError("mcpServers 中的名称和配置必须有效")
            result.append((name, config))
        return result
    if "servers" in payload:
        servers = payload["servers"]
        if not isinstance(servers, list) or not servers:
            raise ExtensionImportError("servers 必须是非空数组")
        result = []
        for config in servers:
            if not isinstance(config, dict):
                raise ExtensionImportError("servers 中的配置必须是对象")
            name = config.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ExtensionImportError("servers 中的每项都必须包含 name")
            result.append((name.strip(), config))
        return result
    raise ExtensionImportError("未找到 mcpServers 或 servers")


# 函数说明：_command_and_args
# 用途：在扩展配置导入中处理 `_command_and_args`，通过 `config.get` 完成首个内部处理步骤
# 。
# 参数：
#   config：运行配置，类型 `dict[str, Any]`；读取键 `command`、`args`。
# 返回：类型 `tuple[str, tuple[str, ...]]`；返回
# `(command.strip(), tuple((item for item in args if item)))`。
# 分支与异常：
#   当 `not isinstance(command, str) or not command.strip()` 时，抛出
# `ExtensionImportError('MCP command 必须是非空字符串')`。
#   当 `not isinstance(args, list) or not all((isinstance(item, str…` 时，抛出
# `ExtensionImportError('MCP args 必须是字符串数组')`。
def _command_and_args(config: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    command = config.get("command")
    args = config.get("args", [])
    if not isinstance(command, str) or not command.strip():
        raise ExtensionImportError("MCP command 必须是非空字符串")
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ExtensionImportError("MCP args 必须是字符串数组")
    return command.strip(), tuple(item for item in args if item)


# 函数说明：_parse_skill_source_or_command
# 用途：解析技能，供扩展配置导入使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
#   scope：记忆、规则或查询作用域，类型 `SkillScope`。
# 返回：类型 `GitHubSkillSource`；按分支返回 `_parse_github_source(source, scope)`；
# `_parse_github_source(value, scope)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_skill_add_source` →
# `_parse_github_source`。
# 分支与异常：
#   捕获 `ValueError` 后，转换或抛出 `ExtensionImportError(f'无法解析输入：{exc}')`。
#   当 `source is not None` 时，返回 `_parse_github_source(source, scope)`。
def _parse_skill_source_or_command(
    value: str,
    scope: SkillScope,
) -> GitHubSkillSource:
    try:
        parts = shlex.split(value)
    except ValueError as exc:
        raise ExtensionImportError(f"无法解析输入：{exc}") from exc
    if parts:
        source = _skill_add_source(parts[0], tuple(parts[1:]))
        if source is not None:
            return _parse_github_source(source, scope)
    return _parse_github_source(value, scope)


# 函数说明：_skill_add_source
# 用途：添加`source`，供扩展配置导入使用。
# 参数：
#   command：待执行的 Shell 命令，类型 `str`。
#   args：`args`输入或配置值，类型 `tuple[str, ...]`。
# 返回：类型 `str | None`；按分支返回 `None`；`args[index + 2]`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`PurePosixPath(command).name.lower` →
# `PurePosixPath` → `item.lower`。
# 分支与异常：
#   当 `executable not in {'npx', 'npm', 'pnpm', 'yarn', 'bunx'}` 时，返回 `None`。
#   当 `lowered[index] == 'skills' and lowered[index + 1] == 'add'` 时，返回
# `args[index + 2]`。
def _skill_add_source(command: str, args: tuple[str, ...]) -> str | None:
    executable = PurePosixPath(command).name.lower()
    if executable not in {"npx", "npm", "pnpm", "yarn", "bunx"}:
        return None
    lowered = [item.lower() for item in args]
    for index in range(len(lowered) - 2):
        if lowered[index] == "skills" and lowered[index + 1] == "add":
            return args[index + 2]
    return None


# 函数说明：_parse_github_source
# 用途：解析`github_source`，供扩展配置导入使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
#   scope：记忆、规则或查询作用域，类型 `SkillScope`。
# 返回：类型 `GitHubSkillSource`；返回 `GitHubSkillSource(owner=match.group('owner'),
# repository=match.group('repo'), scope=scope)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_GITHUB_REPOSITORY_RE.fullmatch` →
# `GitHubSkillSource` → `match.group`。
# 分支与异常：
#   当 `match is None` 时，抛出 `ExtensionImportError(…)`。
def _parse_github_source(value: str, scope: SkillScope) -> GitHubSkillSource:
    match = _GITHUB_REPOSITORY_RE.fullmatch(value.strip())
    if match is None:
        raise ExtensionImportError(
            "GitHub 来源必须是 https://github.com/owner/repo 或 owner/repo"
        )
    return GitHubSkillSource(
        owner=match.group("owner"),
        repository=match.group("repo"),
        scope=scope,
    )


# 函数说明：_normalize_mcp_name
# 用途：规范化名称，供扩展配置导入使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `str`；返回 `normalized`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`re.sub` → `_MCP_NAME_RE.fullmatch`。
# 分支与异常：
#   当 `not normalized or _MCP_NAME_RE.fullmatch(normalized) is None` 时，抛出
# `ExtensionImportError(f'无法把 MCP 名称 {value!r} 转成安全名称')`。
def _normalize_mcp_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized or _MCP_NAME_RE.fullmatch(normalized) is None:
        raise ExtensionImportError(f"无法把 MCP 名称 {value!r} 转成安全名称")
    return normalized


# 函数说明：_string_mapping
# 用途：在扩展配置导入中处理 `_string_mapping`，通过 `value.items` 完成首个内部处理步骤
# 。
# 参数：
#   value：待校验、规范化或转换的值，类型 `object`。
#   label：`label`输入或配置值，类型 `str`。
# 返回：类型 `dict[str, str]`；返回 `dict(value)`。
# 分支与异常：
#   当 `not isinstance(value, dict)` 时，抛出
# `ExtensionImportError(f'{label} 必须是对象')`。
#   当 `not all((isinstance(key, str) and isinstance(item, str) for…` 时，抛出
# `ExtensionImportError(f'{label} 的名称和值都必须是字符串')`。
def _string_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ExtensionImportError(f"{label} 必须是对象")
    if not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise ExtensionImportError(f"{label} 的名称和值都必须是字符串")
    return dict(value)


# 函数说明：_is_env_reference
# 用途：返回 `re.fullmatch('\\$\\{[A-Za-z_][A-Za-z0-9_]*\\}', value) is not None`，提供
# 扩展配置导入 的派生值。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `bool`；返回
# `re.fullmatch('\\$\\{[A-Za-z_][A-Za-z0-9_]*\\}', value) is not None`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`re.fullmatch`。
def _is_env_reference(value: str) -> bool:
    return re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", value) is not None


# 函数说明：_ensure_unique
# 用途：确保`unique`，供扩展配置导入使用。
# 参数：
#   skill_sources：技能输入或配置值，类型 `list[GitHubSkillSource]`。
#   mcp_servers：`mcp_servers`输入或配置值，类型 `list[MCPServerConfig]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`item.slug.lower`。
# 分支与异常：
#   当 `len(skill_slugs) != len(set(skill_slugs))` 时，抛出
# `ExtensionImportError('导入内容包含重复的 Skill 来源')`。
#   当 `len(server_names) != len(set(server_names))` 时，抛出
# `ExtensionImportError('名称转换后产生了重复 MCP Server')`。
#   当 `not skill_sources and (not mcp_servers)` 时，抛出
# `ExtensionImportError('没有识别到可导入的扩展')`。
def _ensure_unique(
    skill_sources: list[GitHubSkillSource],
    mcp_servers: list[MCPServerConfig],
) -> None:
    skill_slugs = [item.slug.lower() for item in skill_sources]
    server_names = [item.name for item in mcp_servers]
    if len(skill_slugs) != len(set(skill_slugs)):
        raise ExtensionImportError("导入内容包含重复的 Skill 来源")
    if len(server_names) != len(set(server_names)):
        raise ExtensionImportError("名称转换后产生了重复 MCP Server")
    if not skill_sources and not mcp_servers:
        raise ExtensionImportError("没有识别到可导入的扩展")


# 函数说明：_fingerprint
# 用途：在扩展配置导入中处理 `_fingerprint`，通过
# `f'v1\x00{scope.value}\x00{permission}\x00{raw}'.encode` 完成首个内部处理步骤。
# 参数：
#   raw：`raw`输入或配置值，类型 `str`。
#   scope：记忆、规则或查询作用域，类型 `SkillScope`。
#   permission：所需权限等级，类型 `str`。
# 返回：类型 `str`；返回 `hashlib.sha256(material).hexdigest()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `f'v1\x00{scope.value}\x00{permission}\x00{raw}'.encode` →
# `hashlib.sha256(material).hexdigest` → `hashlib.sha256`。
def _fingerprint(raw: str, scope: SkillScope, permission: str) -> str:
    material = f"v1\0{scope.value}\0{permission}\0{raw}".encode()
    return hashlib.sha256(material).hexdigest()


# 函数说明：_download_github_archive
# 用途：归档`download_github`，供扩展配置导入使用。
# 参数：
#   source：输入来源或原始数据，类型 `GitHubSkillSource`。
# 返回：类型 `bytes`；返回 `bytes(chunks)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`httpx.AsyncClient` → `httpx.Timeout`
# → `client.stream` → `response.raise_for_status` → `bytearray` → `response.aiter_bytes`
# ；另有 1 个调用点。
# 资源/并发边界：`httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(45.0))`
# ，上下文退出时执行相应清理。
# 分支与异常：
#   捕获 `ValueError` 后，执行异常分支中的状态更新；具体更新见实现。
#   当 `declared_bytes > _MAX_ARCHIVE_BYTES` 时，抛出 `ExtensionImportError(…)`。
#   当 `len(chunks) > _MAX_ARCHIVE_BYTES` 时，抛出 `ExtensionImportError(…)`。
#   捕获 `httpx.HTTPError` 后，转换或抛出
# `ExtensionImportError(f'下载 {source.slug} 失败：{type(exc).__name__}: {exc}')`。
async def _download_github_archive(source: GitHubSkillSource) -> bytes:
    url = f"https://api.github.com/repos/{source.slug}/zipball"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "MuHarness"}
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(45.0),
        ) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                declared_size = response.headers.get("content-length")
                try:
                    declared_bytes = int(declared_size) if declared_size else 0
                except ValueError:
                    declared_bytes = 0
                if declared_bytes > _MAX_ARCHIVE_BYTES:
                    raise ExtensionImportError(
                        f"{source.slug} 归档超过 "
                        f"{_MAX_ARCHIVE_BYTES // 1024 // 1024}MB 限制"
                    )
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > _MAX_ARCHIVE_BYTES:
                        raise ExtensionImportError(
                            f"{source.slug} 归档超过 "
                            f"{_MAX_ARCHIVE_BYTES // 1024 // 1024}MB 限制"
                        )
    except httpx.HTTPError as exc:
        raise ExtensionImportError(
            f"下载 {source.slug} 失败：{type(exc).__name__}: {exc}"
        ) from exc
    return bytes(chunks)


# 函数说明：_skill_packages_from_archive
# 用途：归档技能，供扩展配置导入使用。
# 参数：
#   archive：传给 `io.BytesIO` 的输入，类型 `bytes`。
# 返回：类型 `list[dict[str, bytes]]`；返回 `packages`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`zipfile.ZipFile` → `io.BytesIO` →
# `bundle.infolist` → `PurePosixPath` → `archive_entry.is_dir` →
# `archive_path.is_absolute`；另有 6 个调用点。
# 分支与异常：
#   捕获 `(OSError, zipfile.BadZipFile)` 后，转换或抛出
# `ExtensionImportError('GitHub 返回的不是有效 ZIP 归档')`。
#   当 `archive_entry.is_dir()` 时，跳过当前循环项。
#   当 `archive_path.is_absolute() or '..' in archive_path.parts` 时，抛出
# `ExtensionImportError('Skill 归档包含越界路径')`。
#   当 `stat.S_ISLNK(mode)` 时，跳过当前循环项。
#   捕获 `(UnicodeError, SkillParseError)` 后，跳过当前循环项，继续处理后续项。
#   捕获 `ValueError` 后，跳过当前循环项，继续处理后续项。
def _skill_packages_from_archive(archive: bytes) -> list[dict[str, bytes]]:
    try:
        bundle = zipfile.ZipFile(io.BytesIO(archive))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ExtensionImportError("GitHub 返回的不是有效 ZIP 归档") from exc

    archive_files: dict[PurePosixPath, bytes] = {}
    total_uncompressed_bytes = 0
    for archive_entry in bundle.infolist():
        archive_path = PurePosixPath(archive_entry.filename)
        if archive_entry.is_dir():
            continue
        if archive_path.is_absolute() or ".." in archive_path.parts:
            raise ExtensionImportError("Skill 归档包含越界路径")
        mode = archive_entry.external_attr >> 16
        if stat.S_ISLNK(mode):
            continue
        total_uncompressed_bytes += archive_entry.file_size
        if total_uncompressed_bytes > _MAX_SKILL_PACKAGE_BYTES:
            raise ExtensionImportError("Skill 文件超过 10MB 安全限制")
        archive_files[archive_path] = bundle.read(archive_entry)

    packages: list[dict[str, bytes]] = []
    for archive_path, content in archive_files.items():
        if archive_path.name != "SKILL.md":
            continue
        skill_name = archive_path.parent.name
        try:
            document = content.decode("utf-8")
            parse_skill_document(document, expected_name=skill_name)
        except (UnicodeError, SkillParseError):
            continue
        package: dict[str, bytes] = {"SKILL.md": content}
        for candidate, candidate_content in archive_files.items():
            try:
                relative = candidate.relative_to(archive_path.parent)
            except ValueError:
                continue
            if len(relative.parts) >= 2 and relative.parts[0] in _RESOURCE_DIRS:
                package[relative.as_posix()] = candidate_content
        packages.append(package)
    return packages


__all__ = [
    "ExtensionImportError",
    "ExtensionImportPlan",
    "apply_import_plan",
    "parse_import_plan",
]
