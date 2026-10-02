from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent
from pydantic import SecretStr

from app.integrations.mcp import (
    MCPClientManager,
    MCPConfigurationError,
    MCPRemoteTool,
    MCPServerConfig,
    MCPServerState,
    MCPStatusTool,
    MCPToolCallError,
    StdioMCPClient,
    load_mcp_settings,
    mcp_tool_name,
    serialize_mcp_result,
)
from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolPermission,
)
from app.runtime.agent.runtime import AgentRuntime
from app.safety.sandbox import SandboxFilesystemMode
from app.tools import ToolExecutor, ToolRegistry


class FakeMCPClient:

    # 函数说明：FakeMCPClient.__init__
    # 用途：初始化 FakeMCPClient；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `MCPServerConfig`。
    #   fail_start：`fail_start`输入或配置值，类型 `bool`；默认 `False`。
    #   tools：可用工具定义或工具实例集合，类型 `tuple[MCPRemoteTool, ...] | None`；默认
    #  `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPRemoteTool`。
    # 副作用与资源：
    #   更新对象字段：`self.config`、`self.fail_start`、`self.tools`、`self.started`、
    # `self.closed`、`self.calls`。
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        fail_start: bool = False,
        tools: tuple[MCPRemoteTool, ...] | None = None,
    ) -> None:
        self.config = config
        self.fail_start = fail_start
        self.tools = tools or (
            MCPRemoteTool(
                name="echo.text",
                description="返回参数",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
        )
        self.started = False
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # 函数说明：FakeMCPClient.start
    # 用途：启动FakeMCPClient，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `self.fail_start` 时，抛出 `RuntimeError('无法启动')`。
    # 副作用与资源：
    #   更新对象字段：`self.started`。
    async def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("无法启动")
        self.started = True

    # 函数说明：FakeMCPClient.list_tools
    # 用途：列出工具集合，供回归测试与测试辅助使用。
    # 返回：类型 `tuple[MCPRemoteTool, ...]`；返回 `self.tools`。
    async def list_tools(self) -> tuple[MCPRemoteTool, ...]:
        return self.tools

    # 函数说明：FakeMCPClient.call_tool
    # 用途：在回归测试与测试辅助中处理 `call_tool`，通过 `self.calls.append` 完成首个内
    # 部处理步骤。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `str`；返回
    # `json.dumps({'remote_name': name, 'arguments': arguments}, ensure_ascii=False)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return json.dumps(
            {"remote_name": name, "arguments": arguments},
            ensure_ascii=False,
        )

    # 函数说明：FakeMCPClient.close
    # 用途：关闭FakeMCPClient，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.closed`。
    async def close(self) -> None:
        self.closed = True


class FakeModelAdapter(ModelAdapter):

    # 函数说明：FakeModelAdapter.__init__
    # 用途：初始化 FakeModelAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.requests`。
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.requests: list[ModelRequest] = []

    # 函数说明：FakeModelAdapter.complete
    # 用途：完成FakeModelAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolCall` →
    # `ModelResponse`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            message = Message(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="tool-search-1",
                        name="tool_search",
                        arguments={"query": "echo 返回参数"},
                    ),
                ),
            )
        elif len(self.requests) == 2:
            message = Message(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="mcp-runtime-1",
                        name="mcp__demo__echo_text",
                        arguments={"text": "runtime"},
                    ),
                ),
            )
        else:
            message = Message(role=MessageRole.ASSISTANT, content="MCP 调用完成")
        return ModelResponse(
            id=f"response-{len(self.requests)}",
            provider="fake",
            model="fake-model",
            message=message,
        )

    # 函数说明：FakeModelAdapter.close
    # 用途：关闭FakeModelAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class DirectDeferredCallAdapter(ModelAdapter):

    # 函数说明：DirectDeferredCallAdapter.__init__
    # 用途：初始化 DirectDeferredCallAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.calls`。
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.calls = 0

    # 函数说明：DirectDeferredCallAdapter.complete
    # 用途：完成DirectDeferredCallAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Message` → `ToolCall` →
    # `ModelResponse`。
    # 副作用与资源：
    #   更新对象字段：`self.calls`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            message = Message(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="guessed-mcp-call",
                        name="mcp__demo__echo_text",
                        arguments={"text": "bypass"},
                    ),
                ),
            )
        else:
            message = Message(role=MessageRole.ASSISTANT, content="已停止绕过")
        return ModelResponse(
            id=f"direct-{self.calls}",
            provider="fake",
            model="fake-model",
            message=message,
        )

    # 函数说明：DirectDeferredCallAdapter.close
    # 用途：关闭DirectDeferredCallAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：test_manager_registers_mcp_tool_into_existing_execution_chain
# 用途：回归验证回归测试与测试辅助中的
# `manager_registers_mcp_tool_into_existing_execution_chain` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerConfig` → `ToolRegistry` →
# `MCPClientManager` → `manager.start` → `ToolExecutor(registry).execute` →
# `ToolExecutor`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`statuses[0].state is MCPServerState.RUNNING`。
#   验证条件：`statuses[0].tool_names == ('mcp__demo__echo_text',)`。
#   验证条件：`registry.deferred_names() == ('mcp__demo__echo_text',)`。
#   验证条件：`registry.model_definitions() == ()`。
@pytest.mark.asyncio
async def test_manager_registers_mcp_tool_into_existing_execution_chain() -> None:
    config = MCPServerConfig(
        name="demo",
        command="unused",
        permission=ToolPermission.ALLOWED,
    )
    clients: list[FakeMCPClient] = []

    # 函数说明：test_manager_registers_mcp_tool_into_existing_execution_chain.factory
    # 用途：在回归测试与测试辅助中处理 `factory`，通过 `clients.append` 完成首个内部处理
    # 步骤。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `MCPServerConfig`。
    # 返回：类型 `FakeMCPClient`；返回 `client`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeMCPClient`。
    # 闭包依赖：从外层读取 `clients`。
    def factory(value: MCPServerConfig) -> FakeMCPClient:
        client = FakeMCPClient(value)
        clients.append(client)
        return client

    registry = ToolRegistry()
    manager = MCPClientManager((config,), client_factory=factory)

    statuses = await manager.start(registry)
    result = await ToolExecutor(registry).execute(
        ToolCall(
            id="mcp-1",
            name="mcp__demo__echo_text",
            arguments={"text": "你好"},
        )
    )

    assert statuses[0].state is MCPServerState.RUNNING
    assert statuses[0].tool_names == ("mcp__demo__echo_text",)
    assert registry.deferred_names() == ("mcp__demo__echo_text",)
    assert registry.model_definitions() == ()
    assert result.success is True
    assert json.loads(result.output or "{}") == {
        "remote_name": "echo.text",
        "arguments": {"text": "你好"},
    }
    await manager.close(registry)
    assert clients[0].closed is True
    assert registry.names() == ()


# 函数说明：test_agent_runtime_can_complete_an_mcp_tool_call
# 用途：回归验证回归测试与测试辅助中的 `agent_runtime_can_complete_an_mcp_tool_call` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerConfig` → `ToolRegistry` →
# `MCPClientManager` → `manager.start` → `ProviderConfig` → `SecretStr`；另有 7 个调用点
# 。
# 分支与异常：
#   验证条件：`result.content == 'MCP 调用完成'`。
#   验证条件：`len(result.tool_calls) == 2`。
#   验证条件：`all((record.result.success for record in result.tool_calls))`。
#   验证条件：`[tool.name for tool in adapter.requests[0].tools] == ['tool_search']`。
@pytest.mark.asyncio
async def test_agent_runtime_can_complete_an_mcp_tool_call() -> None:
    config = MCPServerConfig(
        name="demo",
        command="unused",
        permission=ToolPermission.ALLOWED,
    )
    tools = ToolRegistry()
    manager = MCPClientManager(
        (config,),
        client_factory=lambda value: FakeMCPClient(value),
    )
    await manager.start(tools)
    provider_config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = FakeModelAdapter(provider_config)
    models = ModelAdapterRegistry(ModelSettings(_env_file=None))
    models.register("fake", lambda _: adapter, config=provider_config)

    result = await AgentRuntime(models, tools, provider="fake").run("调用 MCP")

    assert result.content == "MCP 调用完成"
    assert len(result.tool_calls) == 2
    assert all(record.result.success for record in result.tool_calls)
    assert [tool.name for tool in adapter.requests[0].tools] == ["tool_search"]
    assert {tool.name for tool in adapter.requests[1].tools} == {
        "tool_search",
        "mcp__demo__echo_text",
    }
    tool_message = adapter.requests[2].messages[-1]
    assert tool_message.role is MessageRole.TOOL
    assert "runtime" in (tool_message.content or "")
    await manager.close(tools)


# 函数说明：test_runtime_rejects_deferred_tool_before_catalog_activation
# 用途：回归验证回归测试与测试辅助中的
# `runtime_rejects_deferred_tool_before_catalog_activation` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerConfig` → `ToolRegistry` →
# `FakeMCPClient` → `MCPClientManager` → `manager.start` → `ProviderConfig`；另有 8 个调
# 用点。
# 分支与异常：
#   验证条件：`result.tool_calls[0].result.success is False`。
#   验证条件：`'tool_search' in (result.tool_calls[0].result.error or '')`。
#   验证条件：`client.calls == []`。
@pytest.mark.asyncio
async def test_runtime_rejects_deferred_tool_before_catalog_activation() -> None:
    config = MCPServerConfig(
        name="demo",
        command="unused",
        permission=ToolPermission.ALLOWED,
    )
    tools = ToolRegistry()
    client = FakeMCPClient(config)
    manager = MCPClientManager((config,), client_factory=lambda value: client)
    await manager.start(tools)
    provider_config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = DirectDeferredCallAdapter(provider_config)
    models = ModelAdapterRegistry(ModelSettings(_env_file=None))
    models.register("fake", lambda _: adapter, config=provider_config)

    result = await AgentRuntime(models, tools, provider="fake").run("绕过目录")

    assert result.tool_calls[0].result.success is False
    assert "tool_search" in (result.tool_calls[0].result.error or "")
    assert client.calls == []
    await manager.close(tools)


# 函数说明：test_one_failed_server_does_not_block_other_servers
# 用途：回归验证回归测试与测试辅助中的 `one_failed_server_does_not_block_other_servers`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerConfig` → `ToolRegistry` →
# `MCPClientManager` → `manager.start` → `registry.names` → `manager.close`。
# 分支与异常：
#   验证条件：`statuses[0].state is MCPServerState.FAILED`。
#   验证条件：`'无法启动' in (statuses[0].error or '')`。
#   验证条件：`statuses[1].state is MCPServerState.RUNNING`。
#   验证条件：`registry.names() == ('mcp__healthy__echo_text',)`。
@pytest.mark.asyncio
async def test_one_failed_server_does_not_block_other_servers() -> None:
    configs = (
        MCPServerConfig(name="broken", command="unused"),
        MCPServerConfig(name="healthy", command="unused"),
    )

    # 函数说明：test_one_failed_server_does_not_block_other_servers.factory
    # 用途：返回 `FakeMCPClient(config, fail_start=config.name == 'broken')`，提供 回归
    # 测试与测试辅助 的派生值。
    # 参数：
    #   config：运行配置，类型 `MCPServerConfig`。
    # 返回：类型 `FakeMCPClient`；返回
    # `FakeMCPClient(config, fail_start=config.name == 'broken')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeMCPClient`。
    def factory(config: MCPServerConfig) -> FakeMCPClient:
        return FakeMCPClient(config, fail_start=config.name == "broken")

    registry = ToolRegistry()
    manager = MCPClientManager(configs, client_factory=factory)

    statuses = await manager.start(registry)

    assert statuses[0].state is MCPServerState.FAILED
    assert "无法启动" in (statuses[0].error or "")
    assert statuses[1].state is MCPServerState.RUNNING
    assert registry.names() == ("mcp__healthy__echo_text",)
    await manager.close(registry)


# 函数说明：test_mcp_status_tool_reads_live_manager_without_starting_servers
# 用途：回归验证回归测试与测试辅助中的
# `mcp_status_tool_reads_live_manager_without_starting_servers` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerConfig` → `ToolRegistry` →
# `MCPClientManager` → `manager.start` → `MCPStatusTool` → `tool.execute`；另有 1 个调用
# 点。
# 分支与异常：
#   验证条件：`result['server_count'] == 2`。
#   验证条件：`result['running_count'] == 1`。
#   验证条件：`result['servers'][0]['state'] == 'failed'`。
#   验证条件：`result['servers'][1]['tools'] == ['mcp__healthy__echo_text']`。
@pytest.mark.asyncio
async def test_mcp_status_tool_reads_live_manager_without_starting_servers() -> None:
    configs = (
        MCPServerConfig(name="broken", command="unused"),
        MCPServerConfig(name="healthy", command="unused"),
    )
    clients: list[FakeMCPClient] = []

    # 函数说明：test_mcp_status_tool_reads_live_manager_without_starting_servers.factory
    # 用途：在回归测试与测试辅助中处理 `factory`，通过 `clients.append` 完成首个内部处理
    # 步骤。
    # 参数：
    #   config：运行配置，类型 `MCPServerConfig`。
    # 返回：类型 `FakeMCPClient`；返回 `client`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeMCPClient`。
    # 闭包依赖：从外层读取 `clients`。
    def factory(config: MCPServerConfig) -> FakeMCPClient:
        client = FakeMCPClient(
            config,
            fail_start=config.name == "broken",
        )
        clients.append(client)
        return client

    registry = ToolRegistry()
    manager = MCPClientManager(configs, client_factory=factory)
    await manager.start(registry)
    starts_before = [client.started for client in clients]
    tool = MCPStatusTool(manager)

    result = await tool.execute({})

    assert result["server_count"] == 2
    assert result["running_count"] == 1
    assert result["servers"][0]["state"] == "failed"
    assert result["servers"][1]["tools"] == ["mcp__healthy__echo_text"]
    assert [client.started for client in clients] == starts_before
    assert tool.definition.permission is ToolPermission.ALLOWED
    await manager.close(registry)


# 函数说明：test_mcp_status_tool_filters_server_and_can_hide_tool_names
# 用途：回归验证回归测试与测试辅助中的
# `mcp_status_tool_filters_server_and_can_hide_tool_names` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerConfig` → `ToolRegistry` →
# `MCPClientManager` → `manager.start` → `MCPStatusTool(manager).execute` →
# `MCPStatusTool`；另有 2 个调用点。
# 分支与异常：
#   验证条件：
# `result['servers'] == [{'name': 'demo', 'state': 'running', 'tool_count': 1}]`。
#   预期异常：`pytest.raises(ValueError, match='不存在')`。
@pytest.mark.asyncio
async def test_mcp_status_tool_filters_server_and_can_hide_tool_names() -> None:
    config = MCPServerConfig(name="demo", command="unused")
    registry = ToolRegistry()
    manager = MCPClientManager(
        (config,),
        client_factory=lambda value: FakeMCPClient(value),
    )
    await manager.start(registry)

    result = await MCPStatusTool(manager).execute(
        {"server": "demo", "include_tools": False}
    )

    assert result["servers"] == [{"name": "demo", "state": "running", "tool_count": 1}]
    with pytest.raises(ValueError, match="不存在"):
        await MCPStatusTool(manager).execute({"server": "missing"})
    await manager.close(registry)


# 函数说明：test_registration_collision_rolls_back_server_tools
# 用途：回归验证回归测试与测试辅助中的 `registration_collision_rolls_back_server_tools`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPRemoteTool` → `MCPServerConfig` →
# `ToolRegistry` → `MCPClientManager` → `manager.start` → `registry.names`。
# 分支与异常：
#   验证条件：`statuses[0].state is MCPServerState.FAILED`。
#   验证条件：`'发生冲突' in (statuses[0].error or '')`。
#   验证条件：`registry.names() == ()`。
@pytest.mark.asyncio
async def test_registration_collision_rolls_back_server_tools() -> None:
    tools = (
        MCPRemoteTool(name="same.name"),
        MCPRemoteTool(name="same-name"),
    )
    config = MCPServerConfig(name="demo", command="unused")
    registry = ToolRegistry()
    manager = MCPClientManager(
        (config,),
        client_factory=lambda value: FakeMCPClient(value, tools=tools),
    )

    statuses = await manager.start(registry)

    assert statuses[0].state is MCPServerState.FAILED
    assert "发生冲突" in (statuses[0].error or "")
    assert registry.names() == ()


# 函数说明：test_mcp_tool_name_uses_stable_namespace
# 用途：回归验证回归测试与测试辅助中的 `mcp_tool_name_uses_stable_namespace` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`mcp_tool_name` → `pytest.raises`。
# 分支与异常：
#   验证条件：`mcp_tool_name('files', 'read.file-v2') == 'mcp__files__read_file_v2'`。
#   预期异常：`pytest.raises(MCPConfigurationError, match='无法注册')`。
def test_mcp_tool_name_uses_stable_namespace() -> None:
    assert mcp_tool_name("files", "read.file-v2") == "mcp__files__read_file_v2"
    with pytest.raises(MCPConfigurationError, match="无法注册"):
        mcp_tool_name("files", "---")


# 函数说明：test_load_settings_handles_missing_and_invalid_config
# 用途：回归验证回归测试与测试辅助中的
# `load_settings_handles_missing_and_invalid_config` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`load_mcp_settings` →
# `invalid.write_text` → `pytest.raises`。
# 分支与异常：
#   验证条件：`(await load_mcp_settings(tmp_path / 'missing.json')).servers == ()`。
#   预期异常：`pytest.raises(MCPConfigurationError, match='无法加载')`。
# 副作用与资源：
#   文件或资源访问：`invalid.write_text`。
@pytest.mark.asyncio
async def test_load_settings_handles_missing_and_invalid_config(tmp_path: Path) -> None:
    assert (await load_mcp_settings(tmp_path / "missing.json")).servers == ()
    invalid = tmp_path / "mcp.json"
    invalid.write_text('{"servers":[{"name":"bad name"}]}', encoding="utf-8")

    with pytest.raises(MCPConfigurationError, match="无法加载"):
        await load_mcp_settings(invalid)


# 函数说明：test_missing_environment_reference_fails_only_that_server
# 用途：回归验证回归测试与测试辅助中的
# `missing_environment_reference_fails_only_that_server` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具，类型 `pytest.MonkeyPatch`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`monkeypatch.delenv` →
# `MCPServerConfig` → `ToolRegistry` → `MCPClientManager` → `manager.start`。
# 分支与异常：
#   验证条件：`statuses[0].state is MCPServerState.FAILED`。
#   验证条件：`'MUHARNESS_MISSING_MCP_KEY' in (statuses[0].error or '')`。
@pytest.mark.asyncio
async def test_missing_environment_reference_fails_only_that_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MUHARNESS_MISSING_MCP_KEY", raising=False)
    config = MCPServerConfig(
        name="missing_env",
        command=sys.executable,
        args=("-c", "pass"),
        env={"API_KEY": "${MUHARNESS_MISSING_MCP_KEY}"},
    )
    registry = ToolRegistry()
    manager = MCPClientManager((config,))

    statuses = await manager.start(registry)

    assert statuses[0].state is MCPServerState.FAILED
    assert "MUHARNESS_MISSING_MCP_KEY" in (statuses[0].error or "")


# 函数说明：test_mcp_config_defaults_to_workspace_sandbox
# 用途：回归验证回归测试与测试辅助中的 `mcp_config_defaults_to_workspace_sandbox` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MCPServerConfig`。
# 分支与异常：
#   验证条件：`config.sandbox.filesystem is SandboxFilesystemMode.WORKSPACE_WRITE`。
def test_mcp_config_defaults_to_workspace_sandbox() -> None:
    config = MCPServerConfig(name="sandboxed", command="unused")

    assert config.sandbox.filesystem is SandboxFilesystemMode.WORKSPACE_WRITE


# 函数说明：test_mcp_environment_does_not_inherit_unlisted_host_secret
# 用途：回归验证回归测试与测试辅助中的
# `mcp_environment_does_not_inherit_unlisted_host_secret` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具，类型 `pytest.MonkeyPatch`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`monkeypatch.setenv` →
# `_resolve_environment` → `MCPServerConfig`。
# 分支与异常：
#   验证条件：`environment['PATH'] == '/usr/bin'`。
#   验证条件：`'MUHARNESS_HOST_SECRET' not in environment`。
def test_mcp_environment_does_not_inherit_unlisted_host_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.integrations.mcp.client import _resolve_environment

    monkeypatch.setenv("MUHARNESS_HOST_SECRET", "must-not-leak")
    monkeypatch.setenv("PATH", "/usr/bin")

    environment = _resolve_environment(
        MCPServerConfig(name="safe_env", command="unused")
    )

    assert environment["PATH"] == "/usr/bin"
    assert "MUHARNESS_HOST_SECRET" not in environment


# 函数说明：test_serialize_mcp_result_preserves_text_and_structured_content
# 用途：回归验证回归测试与测试辅助中的
# `serialize_mcp_result_preserves_text_and_structured_content` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`CallToolResult` → `TextContent` →
# `serialize_mcp_result` → `json.loads`。
# 分支与异常：
#   验证条件：`serialize_mcp_result(plain) == 'hello'`。
#   验证条件：`json.loads(serialize_mcp_result(structured)) == {'content': [{'type': '
# text', 'text': 'hello'}], 'structured_content': {'count':…`。
def test_serialize_mcp_result_preserves_text_and_structured_content() -> None:
    plain = CallToolResult(content=[TextContent(type="text", text="hello")])
    structured = CallToolResult(
        content=[TextContent(type="text", text="hello")],
        structuredContent={"count": 1},
    )

    assert serialize_mcp_result(plain) == "hello"
    assert json.loads(serialize_mcp_result(structured)) == {
        "content": [{"type": "text", "text": "hello"}],
        "structured_content": {"count": 1},
    }


# 函数说明：_stdio_config
# 用途：在回归测试与测试辅助中处理 `_stdio_config`，通过 `Path(__file__).resolve` 完成首
# 个内部处理步骤。
# 参数：
#   call_timeout：调用输入或配置值，类型 `float`；默认 `2.0`。
# 返回：类型 `MCPServerConfig`；返回 `MCPServerConfig(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(__file__).resolve` → `Path` →
# `MCPServerConfig`。
def _stdio_config(*, call_timeout: float = 2.0) -> MCPServerConfig:
    server_path = (
        Path(__file__).resolve().parents[2] / "fixtures" / "fake_mcp_server.py"
    )
    return MCPServerConfig(
        name="stdio_test",
        command=sys.executable,
        args=(str(server_path),),
        startup_timeout_seconds=5,
        call_timeout_seconds=call_timeout,
        permission=ToolPermission.ALLOWED,
    )


# 函数说明：test_stdio_client_discovers_and_calls_real_fake_server
# 用途：回归验证回归测试与测试辅助中的
# `stdio_client_discovers_and_calls_real_fake_server` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StdioMCPClient` → `_stdio_config` →
# `client.start` → `client.list_tools` → `json.loads` → `client.call_tool`；另有 2 个调
# 用点。
# 分支与异常：
#   验证条件：`{tool.name for tool in tools} == {'echo', 'fail', 'slow'}`。
#   验证条件：`output['content'] == [{'type': 'text', 'text': '你好 MCP'}]`。
#   验证条件：`output['structured_content'] == {'result': '你好 MCP'}`。
#   预期异常：`pytest.raises(MCPToolCallError, match='fake boom')`。
@pytest.mark.asyncio
async def test_stdio_client_discovers_and_calls_real_fake_server() -> None:
    client = StdioMCPClient(_stdio_config())
    await client.start()
    try:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {"echo", "fail", "slow"}
        output = json.loads(await client.call_tool("echo", {"text": "你好 MCP"}))
        assert output["content"] == [{"type": "text", "text": "你好 MCP"}]
        assert output["structured_content"] == {"result": "你好 MCP"}
        with pytest.raises(MCPToolCallError, match="fake boom"):
            await client.call_tool("fail", {})
    finally:
        await client.close()


# 函数说明：test_stdio_client_call_timeout
# 用途：回归验证回归测试与测试辅助中的 `stdio_client_call_timeout` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StdioMCPClient` → `_stdio_config` →
# `client.start` → `pytest.raises` → `client.call_tool` → `client.close`。
# 分支与异常：
#   预期异常：`pytest.raises(MCPToolCallError, match='TimeoutError')`。
@pytest.mark.asyncio
async def test_stdio_client_call_timeout() -> None:
    client = StdioMCPClient(_stdio_config(call_timeout=0.05))
    await client.start()
    try:
        with pytest.raises(MCPToolCallError, match="TimeoutError"):
            await client.call_tool("slow", {"delay": 0.5})
    finally:
        await client.close()
