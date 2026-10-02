from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.models.types import AgentMode, ToolCall, ToolDefinition
from app.tools import (
    BaseTool,
    CurrentTimeTool,
    ListFilesTool,
    ReadFileTool,
    ToolExecutor,
    ToolRegistry,
    WriteFileTool,
)
from app.tools.catalog import ToolCatalog, ToolSearchTool


class EchoTool(BaseTool):
    # 函数说明：EchoTool.definition
    # 用途：提供 EchoTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(name='echo')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="echo")

    # 函数说明：EchoTool.execute
    # 用途：执行EchoTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；返回 `arguments`。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        return arguments


class StubDefinitionTool(BaseTool):
    # 函数说明：StubDefinitionTool.__init__
    # 用途：初始化 StubDefinitionTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   definition：工具定义输入或配置值，类型 `ToolDefinition`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._definition`。
    def __init__(self, definition: ToolDefinition) -> None:
        self._definition = definition

    # 函数说明：StubDefinitionTool.definition
    # 用途：提供 StubDefinitionTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `self._definition`。
    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    # 函数说明：StubDefinitionTool.execute
    # 用途：执行StubDefinitionTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；返回 `arguments`。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        return arguments


class FailingTool(BaseTool):
    # 函数说明：FailingTool.definition
    # 用途：提供 FailingTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(name='failing')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="failing")

    # 函数说明：FailingTool.execute
    # 用途：执行FailingTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("boom")


class SlowTool(BaseTool):
    # 函数说明：SlowTool.definition
    # 用途：提供 SlowTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(name='slow')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="slow")

    # 函数说明：SlowTool.execute
    # 用途：执行SlowTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；返回 `'too late'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.sleep`。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        await asyncio.sleep(1)
        return "too late"


class LargeOutputTool(BaseTool):
    # 函数说明：LargeOutputTool.definition
    # 用途：提供 LargeOutputTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(name='large_output')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="large_output")

    # 函数说明：LargeOutputTool.execute
    # 用途：执行LargeOutputTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；返回 `'x' * 25000`。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        return "x" * 25_000


# 函数说明：test_tool_registration_and_definitions
# 用途：回归验证回归测试与测试辅助中的 `tool_registration_and_definitions` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `EchoTool` →
# `registry.register` → `registry.definitions`。
# 分支与异常：
#   验证条件：`registry.get('echo') is tool`。
#   验证条件：`[definition.name for definition in registry.definitions()] == ['echo']`。
def test_tool_registration_and_definitions() -> None:
    registry = ToolRegistry()
    tool = EchoTool()

    registry.register(tool)

    assert registry.get("echo") is tool
    assert [definition.name for definition in registry.definitions()] == ["echo"]


# 函数说明：test_duplicate_tool_registration_is_rejected
# 用途：回归验证回归测试与测试辅助中的 `duplicate_tool_registration_is_rejected` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `EchoTool` → `pytest.raises`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='already registered')`。
def test_duplicate_tool_registration_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())


# 函数说明：test_model_definitions_are_stable_across_registration_order
# 用途：回归验证回归测试与测试辅助中的
# `model_definitions_are_stable_across_registration_order` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `first.register` →
# `StubDefinitionTool` → `ToolDefinition` → `second.register` →
# `first.model_definitions`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`first.model_definitions() == second.model_definitions()`。
#   验证条件：`[item.name for item in first.model_definitions()] == ['alpha', 'zeta']`。
def test_model_definitions_are_stable_across_registration_order() -> None:
    first = ToolRegistry()
    first.register(StubDefinitionTool(ToolDefinition(name="zeta")))
    first.register(StubDefinitionTool(ToolDefinition(name="alpha")))
    second = ToolRegistry()
    second.register(StubDefinitionTool(ToolDefinition(name="alpha")))
    second.register(StubDefinitionTool(ToolDefinition(name="zeta")))

    assert first.model_definitions() == second.model_definitions()
    assert [item.name for item in first.model_definitions()] == ["alpha", "zeta"]


# 函数说明：test_closing_definitions_only_include_declared_delivery_tools
# 用途：回归验证回归测试与测试辅助中的
# `closing_definitions_only_include_declared_delivery_tools` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `EchoTool` → `StubDefinitionTool` → `ToolDefinition` →
# `registry.closing_definitions_for_mode`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`[definition.name for definition in definitions] == ['deliver']`。
#   验证条件：`registry.is_allowed_during_closing('deliver', AgentMode.NORMAL) is True`
# 。
#   验证条件：`registry.is_allowed_during_closing('echo', AgentMode.NORMAL) is False`。
def test_closing_definitions_only_include_declared_delivery_tools() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(
        StubDefinitionTool(
            ToolDefinition(name="deliver", closing_allowed=True)
        )
    )

    definitions = registry.closing_definitions_for_mode(AgentMode.NORMAL)

    assert [definition.name for definition in definitions] == ["deliver"]
    assert registry.is_allowed_during_closing("deliver", AgentMode.NORMAL) is True
    assert registry.is_allowed_during_closing("echo", AgentMode.NORMAL) is False


# 函数说明：test_tool_catalog_tracks_deferred_registry_changes_automatically
# 用途：回归验证回归测试与测试辅助中的
# `tool_catalog_tracks_deferred_registry_changes_automatically` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `StubDefinitionTool`
# → `ToolDefinition` → `registry.register` → `ToolCatalog` → `catalog.search`；另有 1 个
# 调用点。
# 分支与异常：
#   验证条件：`[match.name for match in catalog.search('weather city')] == ['
# mcp__weather__forecast']`。
#   验证条件：`catalog.search('weather city') == ()`。
def test_tool_catalog_tracks_deferred_registry_changes_automatically() -> None:
    registry = ToolRegistry()
    weather = StubDefinitionTool(
        ToolDefinition(
            name="mcp__weather__forecast",
            description="Get weather forecast by city and date",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        )
    )
    registry.register(weather, deferred=True)
    catalog = ToolCatalog(registry)

    assert [match.name for match in catalog.search("weather city")] == [
        "mcp__weather__forecast"
    ]
    registry.unregister("mcp__weather__forecast")
    assert catalog.search("weather city") == ()


# 函数说明：test_tool_search_returns_compact_matching_definitions
# 用途：回归验证回归测试与测试辅助中的
# `tool_search_returns_compact_matching_definitions` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `StubDefinitionTool` → `ToolDefinition` → `EchoTool` → `json.loads`；另有 3 个调用点
# 。
# 分支与异常：
#   验证条件：`payload['count'] == 1`。
#   验证条件：`payload['tools'][0]['name'] == 'mcp__maps__search_places'`。
#   验证条件：`[item.name for item in registry.model_definitions()] == ['echo']`。
#   验证条件：`{item.name for item in registry.model_definitions(activated_names={'
# mcp__maps__search_places'})} == {'echo', '…`。
@pytest.mark.asyncio
async def test_tool_search_returns_compact_matching_definitions() -> None:
    registry = ToolRegistry()
    registry.register(
        StubDefinitionTool(
            ToolDefinition(
                name="mcp__maps__search_places",
                description="Search places and points of interest on a map",
            )
        ),
        deferred=True,
    )
    registry.register(EchoTool())

    payload = json.loads(
        await ToolSearchTool(registry).execute({"query": "map places"})
    )

    assert payload["count"] == 1
    assert payload["tools"][0]["name"] == "mcp__maps__search_places"
    assert [item.name for item in registry.model_definitions()] == ["echo"]
    assert {
        item.name
        for item in registry.model_definitions(
            activated_names={"mcp__maps__search_places"}
        )
    } == {"echo", "mcp__maps__search_places"}


# 函数说明：test_current_time_tool_returns_local_time_on_demand
# 用途：回归验证回归测试与测试辅助中的 `current_time_tool_returns_local_time_on_demand`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`CurrentTimeTool().execute` →
# `CurrentTimeTool`。
# 分支与异常：
#   验证条件：`output['date'] in output['datetime']`。
#   验证条件：`output['time'] in output['datetime']`。
#   验证条件：`output['timezone']`。
#   验证条件：`output['utc_offset']`。
@pytest.mark.asyncio
async def test_current_time_tool_returns_local_time_on_demand() -> None:
    output = await CurrentTimeTool().execute({})

    assert output["date"] in output["datetime"]
    assert output["time"] in output["datetime"]
    assert output["timezone"]
    assert output["utc_offset"]
    assert isinstance(output["unix_timestamp"], int)


# 函数说明：test_current_time_tool_supports_iana_timezone
# 用途：回归验证回归测试与测试辅助中的 `current_time_tool_supports_iana_timezone` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`CurrentTimeTool().execute` →
# `CurrentTimeTool`。
# 分支与异常：
#   验证条件：`output['timezone'] == 'Asia/Shanghai'`。
#   验证条件：`output['utc_offset'] == '+08:00'`。
@pytest.mark.asyncio
async def test_current_time_tool_supports_iana_timezone() -> None:
    output = await CurrentTimeTool().execute({"timezone": "Asia/Shanghai"})

    assert output["timezone"] == "Asia/Shanghai"
    assert output["utc_offset"] == "+08:00"


# 函数说明：test_current_time_tool_rejects_unknown_timezone
# 用途：回归验证回归测试与测试辅助中的 `current_time_tool_rejects_unknown_timezone` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `CurrentTimeTool().execute` → `CurrentTimeTool`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='未知 IANA 时区')`。
@pytest.mark.asyncio
async def test_current_time_tool_rejects_unknown_timezone() -> None:
    with pytest.raises(ValueError, match="未知 IANA 时区"):
        await CurrentTimeTool().execute({"timezone": "Mars/Olympus"})


# 函数说明：test_write_file_creates_parent_and_returns_metadata
# 用途：回归验证回归测试与测试辅助中的 `write_file_creates_parent_and_returns_metadata`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `WriteFileTool` → `ToolExecutor` → `executor.execute` → `ToolCall`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`result.error is None`。
#   验证条件：
# `json.loads(result.output or '{}') == {'path': 'notes/你好.txt', 'characters': 4}`。
#   验证条件：`(tmp_path / 'notes/你好.txt').read_text(encoding='utf-8') == '本地工具'`
# 。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'notes/你好.txt').read_text`。
@pytest.mark.asyncio
async def test_write_file_creates_parent_and_returns_metadata(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    registry.register(WriteFileTool(tmp_path))
    executor = ToolExecutor(registry)

    result = await executor.execute(
        ToolCall(
            id="write-1",
            name="write_file",
            arguments={"path": "notes/你好.txt", "content": "本地工具"},
        )
    )

    assert result.success is True
    assert result.error is None
    assert json.loads(result.output or "{}") == {
        "path": "notes/你好.txt",
        "characters": 4,
    }
    assert (tmp_path / "notes/你好.txt").read_text(encoding="utf-8") == "本地工具"


# 函数说明：test_read_file_returns_utf8_text
# 用途：回归验证回归测试与测试辅助中的 `read_file_returns_utf8_text` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`target.write_text` → `ToolRegistry` →
#  `registry.register` → `ReadFileTool` → `ToolExecutor` → `executor.execute`；另有 1 个
# 调用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`result.output == '你好，MuHarness'`。
#   验证条件：`result.duration_ms >= 0`。
# 副作用与资源：
#   文件或资源访问：`target.write_text`。
@pytest.mark.asyncio
async def test_read_file_returns_utf8_text(tmp_path: Path) -> None:
    target = tmp_path / "文档.txt"
    target.write_text("你好，MuHarness", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ReadFileTool(tmp_path))
    executor = ToolExecutor(registry)

    result = await executor.execute(
        ToolCall(
            id="read-1",
            name="read_file",
            arguments='{"path":"文档.txt"}',
        )
    )

    assert result.success is True
    assert result.output == "你好，MuHarness"
    assert result.duration_ms >= 0


# 函数说明：test_path_traversal_is_rejected
# 用途：回归验证回归测试与测试辅助中的 `path_traversal_is_rejected` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   tool：目标工具实例，类型 `type[BaseTool]`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `(tmp_path / 'secret.txt').write_text` → `ToolRegistry` → `registry.register` → `tool`
#  → `ToolExecutor`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`result.output is None`。
#   验证条件：`result.error is not None`。
#   验证条件：`'escapes the workspace' in result.error`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(tmp_path / 'secret.txt').write_text`。
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (ReadFileTool, {"path": "../secret.txt"}),
        (
            WriteFileTool,
            {"path": "../escaped.txt", "content": "must not be written"},
        ),
    ],
)
async def test_path_traversal_is_rejected(
    tmp_path: Path,
    tool: type[BaseTool],
    arguments: dict[str, Any],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(tool(workspace))  
    executor = ToolExecutor(registry)

    result = await executor.execute(
        ToolCall(
            id="traversal-1", name=tool(workspace).definition.name, arguments=arguments
        )  
    )

    assert result.success is False
    assert result.output is None
    assert result.error is not None
    assert "escapes the workspace" in result.error
    assert not (tmp_path / "escaped.txt").exists()


# 函数说明：test_list_files_supports_subdirectory_and_limit
# 用途：回归验证回归测试与测试辅助中的 `list_files_supports_subdirectory_and_limit` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`subdirectory.mkdir` →
# `(subdirectory / f'{index:03}.txt').write_text` → `ToolRegistry` → `registry.register`
#  → `ListFilesTool` → `ToolExecutor`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`output['count'] == 200`。
#   验证条件：`len(output['files']) == 200`。
#   验证条件：`output['files'][0] == 'items/000.txt'`。
# 副作用与资源：
#   文件或资源访问：`subdirectory.mkdir`、
# `(subdirectory / f'{index:03}.txt').write_text`。
@pytest.mark.asyncio
async def test_list_files_supports_subdirectory_and_limit(tmp_path: Path) -> None:
    subdirectory = tmp_path / "items"
    subdirectory.mkdir()
    for index in range(205):
        (subdirectory / f"{index:03}.txt").write_text("x", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ListFilesTool(tmp_path))
    executor = ToolExecutor(registry)

    result = await executor.execute(
        ToolCall(
            id="list-1",
            name="list_files",
            arguments={"directory": "items"},
        )
    )

    output = json.loads(result.output or "{}")
    assert result.success is True
    assert output["count"] == 200
    assert len(output["files"]) == 200
    assert output["files"][0] == "items/000.txt"
    assert output["truncated"] is True


# 函数说明：test_unknown_tool_returns_failure
# 用途：回归验证回归测试与测试辅助中的 `unknown_tool_returns_failure` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolExecutor(ToolRegistry()).execute`
#  → `ToolExecutor` → `ToolRegistry` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`result.error == 'Tool not found: missing'`。
@pytest.mark.asyncio
async def test_unknown_tool_returns_failure() -> None:
    result = await ToolExecutor(ToolRegistry()).execute(
        ToolCall(id="missing-1", name="missing", arguments={})
    )

    assert result.success is False
    assert result.error == "Tool not found: missing"


# 函数说明：test_tool_exception_returns_failure
# 用途：回归验证回归测试与测试辅助中的 `tool_exception_returns_failure` 场景，下方断言说
# 明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `FailingTool` → `ToolExecutor(registry).execute` → `ToolExecutor` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`result.error == 'Tool execution failed: RuntimeError: boom'`。
#   验证条件：`result.duration_ms >= 0`。
@pytest.mark.asyncio
async def test_tool_exception_returns_failure() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool())

    result = await ToolExecutor(registry).execute(
        ToolCall(id="fail-1", name="failing", arguments={})
    )

    assert result.success is False
    assert result.error == "Tool execution failed: RuntimeError: boom"
    assert result.duration_ms >= 0


# 函数说明：test_tool_timeout_returns_failure
# 用途：回归验证回归测试与测试辅助中的 `tool_timeout_returns_failure` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `SlowTool` → `ToolExecutor(registry, timeout_seconds=0.01).execute` → `ToolExecutor`
# → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`result.error == 'Tool timed out after 0.01 seconds.'`。
#   验证条件：`result.duration_ms >= 10`。
@pytest.mark.asyncio
async def test_tool_timeout_returns_failure() -> None:
    registry = ToolRegistry()
    registry.register(SlowTool())

    result = await ToolExecutor(registry, timeout_seconds=0.01).execute(
        ToolCall(id="slow-1", name="slow", arguments={})
    )

    assert result.success is False
    assert result.error == "Tool timed out after 0.01 seconds."
    assert result.duration_ms >= 10


# 函数说明：test_invalid_json_arguments_return_failure
# 用途：回归验证回归测试与测试辅助中的 `invalid_json_arguments_return_failure` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `EchoTool` → `ToolExecutor(registry).execute` → `ToolExecutor` → `ToolCall`；另有 1
# 个调用点。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`result.error is not None`。
#   验证条件：`result.error.startswith('Invalid arguments:')`。
@pytest.mark.asyncio
async def test_invalid_json_arguments_return_failure() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    result = await ToolExecutor(registry).execute(
        ToolCall(id="echo-1", name="echo", arguments="{invalid")
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("Invalid arguments:")


# 函数说明：test_tool_output_is_limited_to_20000_characters
# 用途：回归验证回归测试与测试辅助中的 `tool_output_is_limited_to_20000_characters` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `LargeOutputTool` → `ToolExecutor(registry).execute` → `ToolExecutor` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`result.output is not None`。
#   验证条件：`len(result.output) == 20000`。
@pytest.mark.asyncio
async def test_tool_output_is_limited_to_20000_characters() -> None:
    registry = ToolRegistry()
    registry.register(LargeOutputTool())

    result = await ToolExecutor(registry).execute(
        ToolCall(id="large-1", name="large_output", arguments={})
    )

    assert result.success is True
    assert result.output is not None
    assert len(result.output) == 20_000


# 函数说明：test_tool_output_limit_cannot_exceed_20000_characters
# 用途：回归验证回归测试与测试辅助中的
# `tool_output_limit_cannot_exceed_20000_characters` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `ToolExecutor` →
# `ToolRegistry`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='cannot exceed 20000')`。
def test_tool_output_limit_cannot_exceed_20000_characters() -> None:
    with pytest.raises(ValueError, match="cannot exceed 20000"):
        ToolExecutor(ToolRegistry(), max_output_chars=20_001)
