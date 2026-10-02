
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.models.types import ToolCall, ToolPermission
from app.tools import (
    AutoApproveGate,
    HttpRequestTool,
    ShellCommandTool,
    ToolExecutor,
    ToolRegistry,
    WebSearchTool,
)


# 函数说明：executor_with
# 用途：在回归测试与测试辅助中处理 `executor_with`，通过 `registry.register` 完成首个内
# 部处理步骤。
# 参数：
#   gate：`gate`输入或配置值，类型 `AutoApproveGate | None`；默认 `None`。
#   *tools：额外位置参数，按实现向内部调用传递。
# 返回：类型 `ToolExecutor`；返回
# `ToolExecutor(registry, approval_gate=gate or AutoApproveGate())`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `ToolExecutor` → `AutoApproveGate`。
def executor_with(*tools: Any, gate: AutoApproveGate | None = None) -> ToolExecutor:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return ToolExecutor(registry, approval_gate=gate or AutoApproveGate())


# 函数说明：test_new_tool_permissions_match_operation_risk
# 用途：回归验证回归测试与测试辅助中的 `new_tool_permissions_match_operation_risk` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ShellCommandTool` → `HttpRequestTool`
#  → `WebSearchTool`。
# 分支与异常：
#   验证条件：
# `ShellCommandTool().definition.permission is ToolPermission.HUMAN_APPROVAL`。
#   验证条件：`HttpRequestTool().definition.permission is ToolPermission.HUMAN_APPROVAL`
# 。
#   验证条件：`WebSearchTool().definition.permission is ToolPermission.ALLOWED`。
def test_new_tool_permissions_match_operation_risk() -> None:
    assert ShellCommandTool().definition.permission is ToolPermission.HUMAN_APPROVAL
    assert HttpRequestTool().definition.permission is ToolPermission.HUMAN_APPROVAL
    assert WebSearchTool().definition.permission is ToolPermission.ALLOWED


# 函数说明：test_shell_blocked_without_approval_gate
# 用途：回归验证回归测试与测试辅助中的 `shell_blocked_without_approval_gate` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `ShellCommandTool` → `ToolExecutor(registry).execute` → `ToolExecutor` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`'human approval' in (result.error or '')`。
@pytest.mark.asyncio
async def test_shell_blocked_without_approval_gate(tmp_path) -> None:
    registry = ToolRegistry()
    registry.register(ShellCommandTool(tmp_path))

    result = await ToolExecutor(registry).execute(
        ToolCall(
            id="sh-0",
            name="run_shell_command",
            arguments={"command": "echo nope"},
        )
    )

    assert result.success is False
    assert "human approval" in (result.error or "")


# 函数说明：test_shell_command_runs_and_captures_output
# 用途：回归验证回归测试与测试辅助中的 `shell_command_runs_and_captures_output` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   docker_engine：可用 Docker 引擎的测试前置夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`executor_with` → `ShellCommandTool` →
#  `executor.execute` → `ToolCall` → `json.loads`。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`output['exit_code'] == 0`。
#   验证条件：`'hello muharness' in output['stdout']`。
@pytest.mark.asyncio
async def test_shell_command_runs_and_captures_output(tmp_path, docker_engine) -> None:
    executor = executor_with(ShellCommandTool(tmp_path))

    result = await executor.execute(
        ToolCall(
            id="sh-1",
            name="run_shell_command",
            arguments={"command": "echo hello muharness"},
        )
    )

    assert result.success is True
    output = json.loads(result.output or "{}")
    assert output["exit_code"] == 0
    assert "hello muharness" in output["stdout"]


# 函数说明：test_shell_command_reports_nonzero_exit
# 用途：回归验证回归测试与测试辅助中的 `shell_command_reports_nonzero_exit` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   docker_engine：可用 Docker 引擎的测试前置夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`executor_with` → `ShellCommandTool` →
#  `executor.execute` → `ToolCall` → `json.loads`。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`output['exit_code'] == 3`。
@pytest.mark.asyncio
async def test_shell_command_reports_nonzero_exit(tmp_path, docker_engine) -> None:
    executor = executor_with(ShellCommandTool(tmp_path))

    result = await executor.execute(
        ToolCall(
            id="sh-2",
            name="run_shell_command",
            arguments={"command": "exit 3"},
        )
    )

    assert result.success is True
    output = json.loads(result.output or "{}")
    assert output["exit_code"] == 3


# 函数说明：test_shell_command_timeout_terminates
# 用途：回归验证回归测试与测试辅助中的 `shell_command_timeout_terminates` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   docker_engine：可用 Docker 引擎的测试前置夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`executor_with` → `ShellCommandTool` →
#  `executor.execute` → `ToolCall` → `json.loads`。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`output['timed_out'] is True`。
@pytest.mark.asyncio
async def test_shell_command_timeout_terminates(tmp_path, docker_engine) -> None:
    executor = executor_with(ShellCommandTool(tmp_path))

    result = await executor.execute(
        ToolCall(
            id="sh-3",
            name="run_shell_command",
            arguments={"command": "sleep 30", "timeout_seconds": 0.2},
        )
    )

    assert result.success is True
    output = json.loads(result.output or "{}")
    assert output["timed_out"] is True


# 函数说明：test_shell_command_uses_working_directory
# 用途：回归验证回归测试与测试辅助中的 `shell_command_uses_working_directory` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   docker_engine：可用 Docker 引擎的测试前置夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`executor_with` → `ShellCommandTool` →
#  `executor.execute` → `ToolCall` → `json.loads` → `Path`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`output['exit_code'] == 0`。
#   验证条件：`Path(output['working_directory']) == tmp_path.resolve()`。
@pytest.mark.asyncio
async def test_shell_command_uses_working_directory(tmp_path, docker_engine) -> None:
    executor = executor_with(ShellCommandTool(tmp_path))

    result = await executor.execute(
        ToolCall(
            id="sh-4",
            name="run_shell_command",
            arguments={"command": "pwd", "working_directory": "."},
        )
    )

    output = json.loads(result.output or "{}")
    assert output["exit_code"] == 0
    assert Path(output["working_directory"]) == tmp_path.resolve()


# 函数说明：test_http_request_get_returns_body
# 用途：回归验证回归测试与测试辅助中的 `http_request_get_returns_body` 场景，下方断言说
# 明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`httpx.MockTransport` →
# `httpx.AsyncClient` → `executor_with` → `HttpRequestTool` → `executor.execute` →
# `ToolCall`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`output['status_code'] == 200`。
#   验证条件：`'muharness' in output['text']`。
@pytest.mark.asyncio
async def test_http_request_get_returns_body() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text="<html>muharness</html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )
    )
    client = httpx.AsyncClient(transport=transport)
    executor = executor_with(HttpRequestTool(client=client))

    result = await executor.execute(
        ToolCall(
            id="http-1",
            name="http_request",
            arguments={"url": "https://example.com/", "method": "GET"},
        )
    )

    assert result.success is True
    output = json.loads(result.output or "{}")
    assert output["status_code"] == 200
    assert "muharness" in output["text"]


# 函数说明：test_http_request_post_sends_body
# 用途：回归验证回归测试与测试辅助中的 `http_request_post_sends_body` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`httpx.AsyncClient` →
# `httpx.MockTransport` → `executor_with` → `HttpRequestTool` → `executor.execute` →
# `ToolCall`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`json.loads(result.output or '{}')['status_code'] == 201`。
#   验证条件：`captured['method'] == 'POST'`。
#   验证条件：`captured['content'] == '{"name":"x"}'`。
@pytest.mark.asyncio
async def test_http_request_post_sends_body() -> None:
    captured: dict[str, Any] = {}

    # 函数说明：test_http_request_post_sends_body.handler
    # 用途：在回归测试与测试辅助中处理 `handler`，通过 `request.content.decode` 完成首个
    # 内部处理步骤。
    # 参数：
    #   request：待处理的请求对象，类型 `httpx.Request`。
    # 返回：类型 `httpx.Response`；返回 `httpx.Response(201, text='created')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`request.content.decode` →
    # `httpx.Response`。
    # 闭包依赖：从外层读取 `captured`。
    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["content"] = request.content.decode()
        return httpx.Response(201, text="created")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    executor = executor_with(HttpRequestTool(client=client))

    result = await executor.execute(
        ToolCall(
            id="http-2",
            name="http_request",
            arguments={
                "url": "https://example.com/items",
                "method": "POST",
                "body": '{"name":"x"}',
                "headers": {"content-type": "application/json"},
            },
        )
    )

    assert result.success is True
    assert json.loads(result.output or "{}")["status_code"] == 201
    assert captured["method"] == "POST"
    assert captured["content"] == '{"name":"x"}'


# 函数说明：test_http_request_blocks_private_addresses
# 用途：回归验证回归测试与测试辅助中的 `http_request_blocks_private_addresses` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`executor_with` → `HttpRequestTool` →
# `executor.execute` → `ToolCall`。
# 分支与异常：
#   验证条件：`result.success is False`。
#   验证条件：`'SSRF' in (result.error or '')`。
@pytest.mark.asyncio
async def test_http_request_blocks_private_addresses() -> None:
    executor = executor_with(HttpRequestTool())

    result = await executor.execute(
        ToolCall(
            id="http-3",
            name="http_request",
            arguments={"url": "http://127.0.0.1:9999/"},
        )
    )

    assert result.success is False
    assert "SSRF" in (result.error or "")


# 函数说明：test_http_request_with_injected_client_bypasses_ssrf
# 用途：回归验证回归测试与测试辅助中的 `http_request_with_injected_client_bypasses_ssrf`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`httpx.AsyncClient` →
# `httpx.MockTransport` → `executor_with` → `HttpRequestTool` → `executor.execute` →
# `ToolCall`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`json.loads(result.output or '{}')['text'] == 'ok'`。
@pytest.mark.asyncio
async def test_http_request_with_injected_client_bypasses_ssrf() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))
    )
    executor = executor_with(HttpRequestTool(client=client))

    result = await executor.execute(
        ToolCall(
            id="http-4",
            name="http_request",
            arguments={"url": "http://127.0.0.1:9/"},
        )
    )

    assert result.success is True
    assert json.loads(result.output or "{}")["text"] == "ok"
