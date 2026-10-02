
from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from app.domain.skills import SkillScope, SkillStore
from app.integrations.extensions import (
    ExtensionImportError,
    apply_import_plan,
    importer,
    parse_import_plan,
)
from app.integrations.mcp import MCPConfigurationStore, MCPServerConfig
from app.server.rpc.dispatcher import RpcContext
from app.server.rpc.methods import extensions
from app.server.rpc.protocol import JsonRpcError


# 函数说明：test_skill_install_generates_valid_markdown_and_catalog
# 用途：回归验证回归测试与测试辅助中的
# `skill_install_generates_valid_markdown_and_catalog` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillStore` → `store.initialize` →
# `store.install` → `(tmp_path / 'project/release-check/SKILL.md').read_text` →
# `store.catalog`。
# 分支与异常：
#   验证条件：`installed.metadata.name == 'release-check'`。
#   验证条件：`installed.metadata.scope is SkillScope.PROJECT`。
#   验证条件：`'name: release-check' in text`。
#   验证条件：`'# Release Check' in text`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'project/release-check/SKILL.md').read_text`。
@pytest.mark.asyncio
async def test_skill_install_generates_valid_markdown_and_catalog(tmp_path) -> None:
    store = SkillStore(
        user_dir=tmp_path / "user",
        project_dir=tmp_path / "project",
    )
    await store.initialize()

    installed = await store.install(
        name="release-check",
        description="发布前检查：冒号也应被正确转义",
        instructions="# Release Check\n\n1. 运行测试。\n2. 检查差异。",
        scope=SkillScope.PROJECT,
    )

    assert installed.metadata.name == "release-check"
    assert installed.metadata.scope is SkillScope.PROJECT
    text = (tmp_path / "project/release-check/SKILL.md").read_text("utf-8")
    assert "name: release-check" in text
    assert "# Release Check" in text
    assert [item.name for item in await store.catalog()] == ["release-check"]


# 函数说明：test_skill_install_rejects_duplicate_without_changing_file
# 用途：回归验证回归测试与测试辅助中的
# `skill_install_rejects_duplicate_without_changing_file` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillStore` → `store.initialize` →
# `store.install` → `path.read_bytes` → `pytest.raises`。
# 分支与异常：
#   验证条件：`path.read_bytes() == before`。
#   预期异常：`pytest.raises(ValueError, match='already exists')`。
# 副作用与资源：
#   文件或资源访问：`path.read_bytes`。
@pytest.mark.asyncio
async def test_skill_install_rejects_duplicate_without_changing_file(tmp_path) -> None:
    store = SkillStore(
        user_dir=tmp_path / "user",
        project_dir=tmp_path / "project",
    )
    await store.initialize()
    await store.install(
        name="demo",
        description="原描述",
        instructions="原指令",
    )
    path = tmp_path / "project/demo/SKILL.md"
    before = path.read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        await store.install(
            name="demo",
            description="新描述",
            instructions="新指令",
        )

    assert path.read_bytes() == before


# 函数说明：test_skill_disable_enable_and_delete_are_scope_safe
# 用途：回归验证回归测试与测试辅助中的 `skill_disable_enable_and_delete_are_scope_safe`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillStore` → `store.initialize` →
# `store.install` → `store.set_enabled` → `store.catalog` →
# `(tmp_path / 'project/.disabled/demo/SKILL.md').is_file`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`disabled.enabled is False`。
#   验证条件：`await store.catalog() == ()`。
#   验证条件：`(tmp_path / 'project/.disabled/demo/SKILL.md').is_file()`。
#   验证条件：
# `[(item.metadata.name, item.enabled) for item in managed] == [('demo', False)]`。
@pytest.mark.asyncio
async def test_skill_disable_enable_and_delete_are_scope_safe(tmp_path) -> None:
    store = SkillStore(
        user_dir=tmp_path / "user",
        project_dir=tmp_path / "project",
    )
    await store.initialize()
    await store.install(
        name="demo",
        description="演示",
        instructions="演示步骤",
        scope=SkillScope.PROJECT,
    )

    disabled = await store.set_enabled(
        name="demo",
        scope=SkillScope.PROJECT,
        enabled=False,
    )
    assert disabled.enabled is False
    assert await store.catalog() == ()
    assert (tmp_path / "project/.disabled/demo/SKILL.md").is_file()
    managed = await store.managed_catalog()
    assert [(item.metadata.name, item.enabled) for item in managed] == [
        ("demo", False),
    ]
    assert store.diagnostics() == ()

    enabled = await store.set_enabled(
        name="demo",
        scope=SkillScope.PROJECT,
        enabled=True,
    )
    assert enabled.enabled is True
    assert [item.name for item in await store.catalog()] == ["demo"]

    await store.delete(
        name="demo",
        scope=SkillScope.PROJECT,
        enabled=True,
    )
    assert await store.managed_catalog() == ()


# 函数说明：test_mcp_configuration_store_writes_valid_json_and_preserves_entries
# 用途：回归验证回归测试与测试辅助中的
# `mcp_configuration_store_writes_valid_json_and_preserves_entries` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPConfigurationStore` → `store.add`
# → `MCPServerConfig` → `json.loads` → `path.read_text`。
# 分支与异常：
#   验证条件：
# `[item['name'] for item in payload['servers']] == ['filesystem', 'weather']`。
#   验证条件：`payload['servers'][1]['args'] == ['weather-mcp']`。
#   验证条件：
# `payload['servers'][1]['env'] == {'WEATHER_API_KEY': '${WEATHER_API_KEY}'}`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
@pytest.mark.asyncio
async def test_mcp_configuration_store_writes_valid_json_and_preserves_entries(
    tmp_path,
) -> None:
    path = tmp_path / "mcp.json"
    store = MCPConfigurationStore(path)
    await store.add(
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=("-y", "@modelcontextprotocol/server-filesystem", "/tmp/work"),
        )
    )
    await store.add(
        MCPServerConfig(
            name="weather",
            command="uvx",
            args=("weather-mcp",),
            env={"WEATHER_API_KEY": "${WEATHER_API_KEY}"},
            permission="allowed",
        )
    )

    payload = json.loads(path.read_text("utf-8"))
    assert [item["name"] for item in payload["servers"]] == [
        "filesystem",
        "weather",
    ]
    assert payload["servers"][1]["args"] == ["weather-mcp"]
    assert payload["servers"][1]["env"] == {
        "WEATHER_API_KEY": "${WEATHER_API_KEY}"
    }


# 函数说明：test_mcp_configuration_store_disable_enable_and_delete
# 用途：回归验证回归测试与测试辅助中的
# `mcp_configuration_store_disable_enable_and_delete` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPConfigurationStore` → `store.add`
# → `MCPServerConfig` → `store.set_enabled` → `store.load` → `store.restart_required`；
# 另有 1 个调用点。
# 分支与异常：
#   验证条件：`disabled.enabled is False`。
#   验证条件：`(await store.load()).servers[0].enabled is False`。
#   验证条件：`store.restart_required('demo') is True`。
#   验证条件：`enabled.enabled is True`。
@pytest.mark.asyncio
async def test_mcp_configuration_store_disable_enable_and_delete(tmp_path) -> None:
    store = MCPConfigurationStore(tmp_path / "mcp.json")
    await store.add(MCPServerConfig(name="demo", command="server"))

    disabled = await store.set_enabled("demo", enabled=False)
    assert disabled.enabled is False
    assert (await store.load()).servers[0].enabled is False
    assert store.restart_required("demo") is True

    enabled = await store.set_enabled("demo", enabled=True)
    assert enabled.enabled is True
    await store.delete("demo")
    assert (await store.load()).servers == ()


# 函数说明：test_extension_list_never_returns_mcp_secret_values
# 用途：回归验证回归测试与测试辅助中的 `extension_list_never_returns_mcp_secret_values`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillStore` →
# `skill_store.initialize` → `MCPConfigurationStore` → `config_store.add` →
# `MCPServerConfig` → `SimpleNamespace`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`server['env_names'] == ['API_KEY']`。
#   验证条件：`'real-secret' not in json.dumps(result)`。
#   验证条件：`server['state'] == 'restart_required'`。
@pytest.mark.asyncio
async def test_extension_list_never_returns_mcp_secret_values(tmp_path) -> None:
    skill_store = SkillStore(
        user_dir=tmp_path / "user",
        project_dir=tmp_path / "project",
    )
    await skill_store.initialize()
    config_store = MCPConfigurationStore(tmp_path / "mcp.json")
    await config_store.add(
        MCPServerConfig(
            name="private",
            command="server",
            env={"API_KEY": "real-secret"},
        )
    )
    app = SimpleNamespace(
        skill_store=skill_store,
        mcp_config_store=config_store,
        mcp_manager=None,
        mcp_error=None,
    )
    ctx = RpcContext(app, SimpleNamespace())

    result = await extensions.extension_list({}, ctx)

    server = result["mcp"]["servers"][0]
    assert server["env_names"] == ["API_KEY"]
    assert "real-secret" not in json.dumps(result)
    assert server["state"] == "restart_required"


# 函数说明：test_mcp_add_rejects_duplicate_and_keeps_json_unchanged
# 用途：回归验证回归测试与测试辅助中的
# `mcp_add_rejects_duplicate_and_keeps_json_unchanged` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPConfigurationStore` →
# `config_store.add` → `MCPServerConfig` → `config_store.path.read_bytes` →
# `SimpleNamespace` → `RpcContext`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`config_store.path.read_bytes() == before`。
#   预期异常：`pytest.raises(JsonRpcError)`。
# 副作用与资源：
#   文件或资源访问：`config_store.path.read_bytes`。
@pytest.mark.asyncio
async def test_mcp_add_rejects_duplicate_and_keeps_json_unchanged(tmp_path) -> None:
    config_store = MCPConfigurationStore(tmp_path / "mcp.json")
    await config_store.add(MCPServerConfig(name="demo", command="server"))
    before = config_store.path.read_bytes()
    app = SimpleNamespace(mcp_config_store=config_store)
    ctx = RpcContext(app, SimpleNamespace())

    with pytest.raises(JsonRpcError):
        await extensions.mcp_add(
            {"name": "demo", "command": "another-server"},
            ctx,
        )

    assert config_store.path.read_bytes() == before


# 函数说明：test_import_preview_recognizes_skill_installer_without_executing_it
# 用途：回归验证回归测试与测试辅助中的
# `import_preview_recognizes_skill_installer_without_executing_it` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_import_plan` → `json.dumps` →
# `plan.public_dict`。
# 分支与异常：
#   验证条件：
# `[item.slug for item in plan.skill_sources] == ['assafelovic/gpt-researcher']`。
#   验证条件：`plan.mcp_servers == ()`。
#   验证条件：`public['items'][0]['kind'] == 'skill'`。
#   验证条件：`any(('不会执行 npx' in action for action in public['actions']))`。
def test_import_preview_recognizes_skill_installer_without_executing_it() -> None:
    plan = parse_import_plan(
        json.dumps(
            {
                "mcpServers": {
                    "gpt-researcher": {
                        "command": "npx",
                        "args": [
                            "skills",
                            "add",
                            "assafelovic/gpt-researcher",
                        ],
                    }
                }
            }
        )
    )

    assert [item.slug for item in plan.skill_sources] == [
        "assafelovic/gpt-researcher"
    ]
    assert plan.mcp_servers == ()
    public = plan.public_dict()
    assert public["items"][0]["kind"] == "skill"
    assert any("不会执行 npx" in action for action in public["actions"])


# 函数说明：test_import_preview_converts_external_mcp_json
# 用途：回归验证回归测试与测试辅助中的 `import_preview_converts_external_mcp_json` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_import_plan` → `json.dumps` →
# `plan.public_dict`。
# 分支与异常：
#   验证条件：`server.name == 'weather_server'`。
#   验证条件：`server.command == 'uvx'`。
#   验证条件：`server.args == ('weather-mcp',)`。
#   验证条件：`plan.public_dict()['items'][0]['env_names'] == ['API_KEY']`。
def test_import_preview_converts_external_mcp_json() -> None:
    plan = parse_import_plan(
        json.dumps(
            {
                "mcpServers": {
                    "weather-server": {
                        "command": "uvx",
                        "args": ["weather-mcp"],
                        "env": {"API_KEY": "${API_KEY}"},
                    }
                }
            }
        )
    )

    server = plan.mcp_servers[0]
    assert server.name == "weather_server"
    assert server.command == "uvx"
    assert server.args == ("weather-mcp",)
    assert plan.public_dict()["items"][0]["env_names"] == ["API_KEY"]
    assert "${API_KEY}" not in json.dumps(plan.public_dict())


# 函数说明：test_import_apply_downloads_static_skill_without_running_command
# 用途：回归验证回归测试与测试辅助中的
# `import_apply_downloads_static_skill_without_running_command` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   monkeypatch：pytest 提供的临时替换依赖夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`io.BytesIO` → `zipfile.ZipFile` →
# `archive.writestr` → `monkeypatch.setattr` → `SkillStore` → `skill_store.initialize`；
# 另有 6 个调用点。
# 分支与异常：
#   验证条件：`result['skills'][0]['name'] == 'research'`。
#   验证条件：`(tmp_path / 'project/research/SKILL.md').is_file()`。
#   验证条件：`(tmp_path / 'project/research/references/checklist.md').read_text('utf-8'
# ) == 'verify sources'`。
#   验证条件：`not (tmp_path / 'project/research/package.json').exists()`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'project/research/references/checklist.md').read_text`
# 。
@pytest.mark.asyncio
async def test_import_apply_downloads_static_skill_without_running_command(
    tmp_path,
    monkeypatch,
) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(
            "owner-repo/skills/research/SKILL.md",
            "---\nname: research\ndescription: 研究流程\n---\n\n# Steps\n\n1. Search.",
        )
        archive.writestr(
            "owner-repo/skills/research/references/checklist.md",
            "verify sources",
        )
        archive.writestr("owner-repo/package.json", "{}")

    # 函数说明：
    # test_import_apply_downloads_static_skill_without_running_command.fake_download
    # 用途：返回 `archive_buffer.getvalue()`，提供 回归测试与测试辅助 的派生值。
    # 参数：
    #   _source：`source`输入或配置值。
    # 返回：类型 `bytes`；返回 `archive_buffer.getvalue()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`archive_buffer.getvalue`。
    # 闭包依赖：从外层读取 `archive_buffer`。
    async def fake_download(_source) -> bytes:
        return archive_buffer.getvalue()

    monkeypatch.setattr(importer, "_download_github_archive", fake_download)
    skill_store = SkillStore(
        user_dir=tmp_path / "user",
        project_dir=tmp_path / "project",
    )
    await skill_store.initialize()
    mcp_store = MCPConfigurationStore(tmp_path / "mcp.json")
    plan = parse_import_plan("owner/repo")

    result = await apply_import_plan(
        plan,
        skill_store=skill_store,
        mcp_store=mcp_store,
    )

    assert result["skills"][0]["name"] == "research"
    assert (tmp_path / "project/research/SKILL.md").is_file()
    assert (
        tmp_path / "project/research/references/checklist.md"
    ).read_text("utf-8") == "verify sources"
    assert not (tmp_path / "project/research/package.json").exists()


# 函数说明：test_import_rejects_zip_path_traversal
# 用途：回归验证回归测试与测试辅助中的 `import_rejects_zip_path_traversal` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`io.BytesIO` → `zipfile.ZipFile` →
# `archive.writestr` → `pytest.raises` → `importer._skill_packages_from_archive` →
# `archive_buffer.getvalue`。
# 分支与异常：
#   预期异常：`pytest.raises(ExtensionImportError, match='越界路径')`。
def test_import_rejects_zip_path_traversal() -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("owner-repo/../outside.txt", "unsafe")

    with pytest.raises(ExtensionImportError, match="越界路径"):
        importer._skill_packages_from_archive(archive_buffer.getvalue())


# 函数说明：test_import_apply_requires_confirmation_and_matching_preview
# 用途：回归验证回归测试与测试辅助中的
# `import_apply_requires_confirmation_and_matching_preview` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillStore` →
# `skill_store.initialize` → `SimpleNamespace` → `MCPConfigurationStore` → `RpcContext`
# → `json.dumps`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result['mcp_servers'] == ['demo']`。
#   预期异常：`pytest.raises(JsonRpcError, match='明确确认')`。
#   预期异常：`pytest.raises(JsonRpcError, match='重新生成预览')`。
@pytest.mark.asyncio
async def test_import_apply_requires_confirmation_and_matching_preview(
    tmp_path,
) -> None:
    skill_store = SkillStore(
        user_dir=tmp_path / "user",
        project_dir=tmp_path / "project",
    )
    await skill_store.initialize()
    app = SimpleNamespace(
        skill_store=skill_store,
        mcp_config_store=MCPConfigurationStore(tmp_path / "mcp.json"),
    )
    ctx = RpcContext(app, SimpleNamespace())
    raw = json.dumps(
        {"mcpServers": {"demo": {"command": "uvx", "args": ["demo"]}}}
    )
    preview = await extensions.extension_import_preview({"input": raw}, ctx)

    with pytest.raises(JsonRpcError, match="明确确认"):
        await extensions.extension_import_apply({"input": raw}, ctx)
    with pytest.raises(JsonRpcError, match="重新生成预览"):
        await extensions.extension_import_apply(
            {
                "input": raw + " ",
                "fingerprint": "wrong",
                "confirmed": True,
            },
            ctx,
        )

    result = await extensions.extension_import_apply(
        {
            "input": raw,
            "fingerprint": preview["plan"]["fingerprint"],
            "confirmed": True,
        },
        ctx,
    )
    assert result["mcp_servers"] == ["demo"]
