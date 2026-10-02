from __future__ import annotations

import json
from typing import cast

import pytest

from app.application import (
    _DEFERRED_TOOL_NAMES,
    DEFAULT_SYSTEM_PROMPT,
    _mark_deferred_tools,
)
from app.domain.conversation.store import SQLiteConversationStore
from app.domain.conversation.tools import register_history_tools
from app.domain.memory.manager import MemoryManager
from app.domain.memory.tools import register_memory_tools
from app.integrations.mcp.client import MCPClientProtocol
from app.integrations.mcp.manager import mcp_tool_name
from app.integrations.mcp.models import MCPRemoteTool
from app.integrations.mcp.tool import MCPToolAdapter
from app.models.types import ToolPermission
from app.records.evidence.store import SQLiteEvidenceStore
from app.records.evidence.tools import register_evidence_tools
from app.runtime.mea.runtimes import ROLE_SYSTEM_PROMPT
from app.tools.builtin.http_request import HttpRequestTool
from app.tools.catalog import ToolCatalog, ToolSearchTool
from app.tools.registry import ToolRegistry


class _UnavailableBackend:
    """Discovery reads definitions only; touching a real backend is a test error."""

    # 函数说明：_UnavailableBackend.__getattr__
    # 用途：处理回归测试与测试辅助中的 `__getattr__` 数据；结果及边界条件见下方说明。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `object`；不返回结果值（隐式 None）。
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"Capability discovery touched backend method {name!r}")


# 函数说明：test_mcp_common_constraints_survive_outside_search_index
# 用途：回归验证回归测试与测试辅助中的
# `mcp_common_constraints_survive_outside_search_index` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   prompt：本次调用使用的提示文本，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`boundary in prompt`。
@pytest.mark.parametrize(
    "prompt", [DEFAULT_SYSTEM_PROMPT, ROLE_SYSTEM_PROMPT], ids=["default", "mea-role"],
)
def test_mcp_common_constraints_survive_outside_search_index(prompt: str) -> None:
    for boundary in (
        "MCP 工具仅在当前任务需要该远端能力",
        "用途和参数含义明确",
        "不重复试探",
        "不凭服务名推测其他能力",
        "未提供用途说明时先确认用途",
        "调用仍遵守当前权限",
        "远端成功回执只代表本次报告",
        "须核实内容",
        "不等于结果已验证或用户目标已完成",
    ):
        assert boundary in prompt


# 函数说明：_remote_tool
# 用途：返回 `MCPRemoteTool(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   description：补充描述，类型 `str`。
#   parameter_name：名称输入或配置值，类型 `str`。
#   parameter_description：`parameter_description`输入或配置值，类型 `str`。
# 返回：类型 `MCPRemoteTool`；返回 `MCPRemoteTool(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPRemoteTool`。
def _remote_tool(
    name: str,
    description: str,
    parameter_name: str,
    parameter_description: str,
) -> MCPRemoteTool:
    return MCPRemoteTool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {
                parameter_name: {
                    "type": "string",
                    "description": parameter_description,
                },
            },
            "required": [parameter_name],
        },
    )


# 函数说明：mixed_registry
# 用途：在回归测试与测试辅助中处理 `mixed_registry`，通过 `registry.register` 完成首个内
# 部处理步骤。
# 返回：类型 `ToolRegistry`；返回 `registry`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `_UnavailableBackend`
#  → `register_memory_tools` → `cast` → `register_history_tools` →
# `register_evidence_tools`；另有 6 个调用点。
# 副作用与资源：
#   数据库操作：CREATE.ISSUE；连接与事务边界以 with/提交语句为准。
@pytest.fixture
def mixed_registry() -> ToolRegistry:
    """Real local schemas compete with realistic, namespaced remote schemas."""
    registry = ToolRegistry()
    backend = _UnavailableBackend()
    register_memory_tools(registry, cast(MemoryManager, backend))
    register_history_tools(registry, cast(SQLiteConversationStore, backend))
    register_evidence_tools(registry, cast(SQLiteEvidenceStore, backend))
    registry.register(HttpRequestTool())
    _mark_deferred_tools(registry, _DEFERRED_TOOL_NAMES)

    remote_tools = (
        (
            "db",
            _remote_tool(
                "query", "Execute SQL against an analytics database",
                "statement", "SQL statement to execute",
            ),
        ),
        (
            "fs",
            _remote_tool(
                "read.text", "Read a file from the local filesystem",
                "path", "Absolute filesystem pathname",
            ),
        ),
        (
            "github",
            _remote_tool(
                "create.issue", "Create an issue in a GitHub repository",
                "title", "Bug tracker title",
            ),
        ),
        (
            "github",
            _remote_tool(
                "list.prs", "List pull requests for a GitHub repository",
                "repository", "Repository owner and project",
            ),
        ),
        (
            "calendar",
            _remote_tool(
                "list.events", "List scheduled calendar appointments",
                "interval", "Start and end dates",
            ),
        ),
        (
            "mail",
            _remote_tool(
                "send", "Send an email message",
                "recipient", "Recipient mailbox",
            ),
        ),
        (
            "weather",
            _remote_tool(
                "forecast", "Forecast weather conditions by city",
                "city", "City and forecast date",
            ),
        ),
        (
            "images",
            _remote_tool(
                "resize", "Resize an image to target dimensions",
                "size", "Pixel dimensions",
            ),
        ),
        (
            "diagnostics",
            _remote_tool(
                "inspect", "", "probe", "Thermal calibration probe channel",
            ),
        ),
    )
    for server_name, remote_tool in remote_tools:
        registry.register(
            MCPToolAdapter(
                server_name=server_name,
                registered_name=mcp_tool_name(server_name, remote_tool.name),
                remote_tool=remote_tool,
                client=cast(MCPClientProtocol, backend),
                permission=ToolPermission.HUMAN_APPROVAL,
            ),
            deferred=True,
        )
    return registry


# 函数说明：_search_names
# 用途：检索`names`，供回归测试与测试辅助使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   query：检索查询文本，类型 `str`。
#   surface：`surface`输入或配置值，类型 `str`。
# 返回：类型 `list[str]`；按分支返回
# `[match.name for match in ToolCatalog(registry).search(query)]`；
# `[tool['name'] for tool in payload['tools']]`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolCatalog(registry).search` →
# `ToolCatalog` → `json.loads` → `ToolSearchTool(registry).execute` → `ToolSearchTool`。
# 分支与异常：
#   当 `surface == 'catalog'` 时，返回
# `[match.name for match in ToolCatalog(registry).search(query…`。
#   验证条件：`payload['query'] == query`。
#   验证条件：`payload['count'] == len(payload['tools'])`。
async def _search_names(
    registry: ToolRegistry, query: str, surface: str,
) -> list[str]:
    if surface == "catalog":
        return [match.name for match in ToolCatalog(registry).search(query)]
    payload = json.loads(await ToolSearchTool(registry).execute({"query": query}))
    assert payload["query"] == query
    assert payload["count"] == len(payload["tools"])
    return [tool["name"] for tool in payload["tools"]]


# 函数说明：test_local_capability_ranking_in_mixed_catalog
# 用途：回归验证回归测试与测试辅助中的 `local_capability_ranking_in_mixed_catalog` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   mixed_registry：传给 `_search_names` 的输入，类型 `ToolRegistry`。
#   query：检索查询文本，类型 `str`。
#   expected：`expected`输入或配置值，类型 `str`。
#   surface：传给 `_search_names` 的输入，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_search_names`。
# 分支与异常：
#   验证条件：`names`。
#   验证条件：`names[0] == expected`。
@pytest.mark.parametrize("surface", ["catalog", "tool_search"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("工具原文", "evidence_search", id="missing-tool-original"),
        pytest.param("evidence search", "evidence_search", id="evidence-search-en"),
        pytest.param("读取证据正文", "evidence_read", id="evidence-body"),
        pytest.param("evidence_id", "evidence_read", id="known-evidence-id"),
        pytest.param("evidence read", "evidence_read", id="evidence-read-en"),
        pytest.param("搜索历史用户原话", "history_search", id="find-user-statement"),
        pytest.param("history search", "history_search", id="history-search-en"),
        pytest.param("已知 sequence 读取消息窗口", "history_read", id="history-window"),
        pytest.param("history read sequence", "history_read", id="history-read-en"),
        pytest.param("core memory update", "core_memory_update", id="update-core"),
        pytest.param("core memory remove", "core_memory_remove", id="remove-core"),
        pytest.param("memory list", "memory_list", id="memory-directory"),
    ],
)
async def test_local_capability_ranking_in_mixed_catalog(
    mixed_registry: ToolRegistry, query: str, expected: str, surface: str,
) -> None:
    names = await _search_names(mixed_registry, query, surface)

    assert names, f"No capability found for {query!r}"
    assert names[0] == expected, names


# 函数说明：test_http_discovery_excludes_unrelated_remote_tools
# 用途：回归验证回归测试与测试辅助中的 `http_discovery_excludes_unrelated_remote_tools`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   mixed_registry：传给 `_search_names` 的输入，类型 `ToolRegistry`。
#   query：检索查询文本，类型 `str`。
#   surface：传给 `_search_names` 的输入，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_search_names` → `name.startswith`。
# 分支与异常：
#   验证条件：`names and names[0] == 'http_request'`。
#   验证条件：`not any((name.startswith('mcp__') for name in names))`。
@pytest.mark.parametrize("surface", ["catalog", "tool_search"])
@pytest.mark.parametrize("query", ["获取网页内容", "调用接口获取结果"])
async def test_http_discovery_excludes_unrelated_remote_tools(
    mixed_registry: ToolRegistry, query: str, surface: str,
) -> None:
    names = await _search_names(mixed_registry, query, surface)

    assert names and names[0] == "http_request", names
    assert not any(name.startswith("mcp__") for name in names), names


# 函数说明：test_remote_capabilities_remain_discoverable
# 用途：回归验证回归测试与测试辅助中的 `remote_capabilities_remain_discoverable` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   mixed_registry：传给 `_search_names` 的输入，类型 `ToolRegistry`。
#   query：检索查询文本，类型 `str`。
#   server：传给 `mcp_tool_name` 的输入，类型 `str`。
#   remote_name：传给 `mcp_tool_name` 的输入，类型 `str`。
#   surface：传给 `_search_names` 的输入，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_search_names` → `mcp_tool_name`。
# 分支与异常：
#   验证条件：`names and names[0] == mcp_tool_name(server, remote_name)`。
@pytest.mark.parametrize("surface", ["catalog", "tool_search"])
@pytest.mark.parametrize(
    ("query", "server", "remote_name"),
    [
        pytest.param("SQL analytics database", "db", "query", id="remote-description"),
        pytest.param(
            "pull requests repository", "github", "list.prs", id="remote-purpose",
        ),
        pytest.param("weather conditions", "weather", "forecast", id="remote-forecast"),
        pytest.param(
            "pixel dimensions", "images", "resize", id="parameter-description",
        ),
        pytest.param(
            "thermal calibration", "diagnostics", "inspect", id="no-description",
        ),
    ],
)
async def test_remote_capabilities_remain_discoverable(
    mixed_registry: ToolRegistry,
    query: str,
    server: str,
    remote_name: str,
    surface: str,
) -> None:
    names = await _search_names(mixed_registry, query, surface)

    assert names and names[0] == mcp_tool_name(server, remote_name), names
