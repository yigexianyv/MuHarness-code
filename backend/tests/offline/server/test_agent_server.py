
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.application import DEFAULT_SYSTEM_PROMPT, Application
from app.domain.memory import (
    MemoryMaintenanceConfig,
    MemoryReflectionConfig,
)
from app.domain.skill_learning import SkillLearningSettings
from app.domain.task import TaskStep
from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    AgentMode,
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolDefinition,
    ToolPermission,
)
from app.server.app import create_app
from app.tools.base import BaseTool


# 函数说明：_model_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str`；默认 `'已完成'`。
#   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`；默认 `()`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def _model_response(
    content: str = "已完成",
    tool_calls: tuple[ToolCall, ...] = (),
) -> ModelResponse:
    return ModelResponse(
        id="fake-response",
        provider="fake",
        model="fake-model",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        ),
        usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


class RepeatingFakeAdapter(ModelAdapter):

    # 函数说明：RepeatingFakeAdapter.__init__
    # 用途：初始化 RepeatingFakeAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   response：模型、工具或服务返回的响应，类型 `ModelResponse`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.response`、`self.requests`。
    def __init__(self, config: ProviderConfig, response: ModelResponse) -> None:
        super().__init__(config)
        self.response = response
        self.requests: list[ModelRequest] = []

    # 函数说明：RepeatingFakeAdapter.complete
    # 用途：完成RepeatingFakeAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `self.response`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.response

    # 函数说明：RepeatingFakeAdapter.close
    # 用途：关闭RepeatingFakeAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class ScriptedFakeAdapter(ModelAdapter):

    # 函数说明：ScriptedFakeAdapter.__init__
    # 用途：初始化 ScriptedFakeAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   responses：预设的模型或服务响应序列，类型 `Sequence[ModelResponse | Exception]`
    # 。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.requests`。
    def __init__(
        self,
        config: ProviderConfig,
        responses: Sequence[ModelResponse | Exception],
    ) -> None:
        super().__init__(config)
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    # 函数说明：ScriptedFakeAdapter.complete
    # 用途：完成ScriptedFakeAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.responses.pop`。
    # 分支与异常：
    #   当 `isinstance(response, Exception)` 时，抛出 `response`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    # 函数说明：ScriptedFakeAdapter.close
    # 用途：关闭ScriptedFakeAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class ApprovalProbeTool(BaseTool):

    definition = ToolDefinition(
        name="approval_probe",
        description="需要审批的探测工具",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        permission=ToolPermission.HUMAN_APPROVAL,
    )

    # 函数说明：ApprovalProbeTool.__init__
    # 用途：初始化 ApprovalProbeTool；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    def __init__(self) -> None:
        self.executions = 0

    # 函数说明：ApprovalProbeTool.execute
    # 用途：执行ApprovalProbeTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, object]`；读取键
    # `value`。
    # 返回：类型 `str`；返回 `f"probe-{arguments.get('value')}"`。
    # 副作用与资源：
    #   更新对象字段：`self.executions`。
    async def execute(self, arguments: dict[str, object]) -> str:
        self.executions += 1
        return f"probe-{arguments.get('value')}"


# 函数说明：_drain_until
# 用途：排空`until`，供回归测试与测试辅助使用。
# 参数：
#   websocket：当前 WebSocket 连接，类型 `Any`。
#   predicate：`predicate`输入或配置值，类型 `Any`。
#   limit：本次返回或处理的数量上限，类型 `int`；默认 `500`。
# 返回：类型 `dict[str, Any]`；返回 `message`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` →
# `websocket.receive_text` → `predicate`。
# 分支与异常：
#   当 `predicate(message)` 时，返回 `message`。
def _drain_until(websocket: Any, predicate: Any, *, limit: int = 500) -> dict[str, Any]:

    for _ in range(limit):
        message = json.loads(websocket.receive_text())
        if predicate(message):
            return message
    raise AssertionError("未在限次内等到目标消息")


# 函数说明：_model_with_tool_call
# 用途：返回 `_model_response(…)`，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `ModelResponse`；返回 `_model_response(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_model_response` → `ToolCall`。
def _model_with_tool_call() -> ModelResponse:
    return _model_response(
        tool_calls=(
            ToolCall(
                id="ap-1",
                name="approval_probe",
                arguments={"value": 7},
            ),
        )
    )


class BlockingFakeAdapter(ModelAdapter):

    # 函数说明：BlockingFakeAdapter.__init__
    # 用途：初始化 BlockingFakeAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super` →
    # `asyncio.Event`。
    # 副作用与资源：
    #   更新对象字段：`self.started`、`self.cancelled`、`self.requests`。
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.started = asyncio.Event()
        self.cancelled = False
        self.requests: list[ModelRequest] = []

    # 函数说明：BlockingFakeAdapter.complete
    # 用途：完成BlockingFakeAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.started.set` →
    # `asyncio.Event().wait` → `asyncio.Event`。
    # 分支与异常：
    #   捕获 `asyncio.CancelledError` 后，重新抛出原异常。
    # 副作用与资源：
    #   更新对象字段：`self.cancelled`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("阻塞模型不应正常完成")

    # 函数说明：BlockingFakeAdapter.close
    # 用途：关闭BlockingFakeAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：make_app
# 用途：构造`app`，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：返回 `build`。
@pytest.fixture
def make_app(tmp_path):

    # 函数说明：make_app.build
    # 用途：构建回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   blocking：`blocking`输入或配置值，类型 `bool`；默认 `False`。
    #   response_text：传给 `_model_response` 的输入，类型 `str`；默认 `'已完成'`。
    #   responses：预设的模型或服务响应序列，类型
    # `Sequence[ModelResponse | Exception] | None`；默认 `None`。
    #   workspace_root：文件工具允许访问的工作区根目录，类型 `Path | None`；默认 `None`
    # 。
    # 返回：返回 `(app, application, adapter)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
    # `BlockingFakeAdapter` → `ScriptedFakeAdapter` → `RepeatingFakeAdapter` →
    # `_model_response`；另有 8 个调用点。
    # 闭包依赖：从外层读取 `tmp_path`。
    def build(
        *,
        blocking: bool = False,
        response_text: str = "已完成",
        responses: Sequence[ModelResponse | Exception] | None = None,
        workspace_root: Path | None = None,
    ):
        config = ProviderConfig(
            provider="fake",
            model="fake-model",
            api_key=SecretStr("offline-test-key"),
            api_style=ApiStyle.CHAT_COMPLETIONS,
        )
        if blocking:
            adapter: ModelAdapter = BlockingFakeAdapter(config)
        elif responses is not None:
            adapter = ScriptedFakeAdapter(config, responses)
        else:
            adapter = RepeatingFakeAdapter(config, _model_response(response_text))
        registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
        registry.register("fake", lambda _: adapter, config=config)

        application = Application(
            provider="fake",
            model="fake-model",
            database=tmp_path / "muharness.db",
            tasks_dir=tmp_path / "tasks",
            mcp_config=tmp_path / "mcp.json",
            memory_dir=tmp_path / "memory",
            skills_user_dir=tmp_path / "skills-user",
            skills_project_dir=tmp_path / "skills-project",
            workspace_root=workspace_root,
            registry=registry,
            memory_reflection_config=MemoryReflectionConfig(
                _env_file=None, enabled=False
            ),
            memory_maintenance_config=MemoryMaintenanceConfig(
                _env_file=None, enabled=False
            ),
            skill_learning_settings=SkillLearningSettings(
                _env_file=None,
                skill_learning_enabled=False,
                skill_learning_data_dir=tmp_path / "skill-learning",
            ),
        )
        app = create_app(application)
        return app, application, adapter

    return build


# 函数说明：_rpc_request
# 用途：返回 `json.dumps(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   request_id：请求标识，类型 `int`。
#   method：HTTP 或 RPC 方法名称，类型 `str`。
#   params：JSON-RPC 方法参数，类型 `dict[str, Any] | None`；默认 `None`。
# 返回：类型 `str`；返回 `json.dumps(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
def _rpc_request(
    request_id: int,
    method: str,
    params: dict[str, Any] | None = None,
) -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        },
        ensure_ascii=False,
    )


# 函数说明：_rpc_call
# 用途：在回归测试与测试辅助中处理 `_rpc_call`，通过 `websocket.send_text` 完成首个内部
# 处理步骤。
# 参数：
#   websocket：当前 WebSocket 连接，类型 `Any`。
#   request_id：请求标识，类型 `int`。
#   method：HTTP 或 RPC 方法名称，类型 `str`。
#   params：JSON-RPC 方法参数，类型 `dict[str, Any] | None`；默认 `None`。
# 返回：类型 `tuple[dict[str, Any], list[dict[str, Any]]]`；返回
# `(message, notifications)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`websocket.send_text` → `_rpc_request`
#  → `json.loads` → `websocket.receive_text`。
# 分支与异常：
#   当 `message.get('id') == request_id` 时，返回 `(message, notifications)`。
def _rpc_call(
    websocket: Any,
    request_id: int,
    method: str,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:

    websocket.send_text(_rpc_request(request_id, method, params))
    notifications: list[dict[str, Any]] = []
    while True:
        message = json.loads(websocket.receive_text())
        if message.get("id") == request_id:
            return message, notifications
        notifications.append(message)


# 函数说明：_require_result
# 用途：获取并校验必需的结果，供回归测试与测试辅助使用。
# 参数：
#   response：模型、工具或服务返回的响应，类型 `dict[str, Any]`；读取键 `error`、
# `result`。
# 返回：类型 `dict[str, Any]`；返回 `response['result']`。
# 分支与异常：
#   验证条件：`'error' not in response`。
def _require_result(response: dict[str, Any]) -> dict[str, Any]:
    assert "error" not in response, f"unexpected error: {response.get('error')}"
    return response["result"]




# 函数说明：test_parse_error
# 用途：回归验证回归测试与测试辅助中的 `parse_error` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `websocket.send_text` → `json.loads` →
# `websocket.receive_text`。
# 分支与异常：
#   验证条件：`message['id'] is None`。
#   验证条件：`message['error']['code'] == -32700`。
def test_parse_error(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            websocket.send_text("{not valid json")
            message = json.loads(websocket.receive_text())
            assert message["id"] is None
            assert message["error"]["code"] == -32700


# 函数说明：test_invalid_request_missing_method
# 用途：回归验证回归测试与测试辅助中的 `invalid_request_missing_method` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `websocket.send_text` → `json.dumps` → `json.loads`；另有
#  1 个调用点。
# 分支与异常：
#   验证条件：`message['id'] == 1`。
#   验证条件：`message['error']['code'] == -32600`。
def test_invalid_request_missing_method(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            websocket.send_text(json.dumps({"jsonrpc": "2.0", "id": 1}))
            message = json.loads(websocket.receive_text())
            assert message["id"] == 1
            assert message["error"]["code"] == -32600


# 函数说明：test_method_not_found
# 用途：回归验证回归测试与测试辅助中的 `method_not_found` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_rpc_call`。
# 分支与异常：
#   验证条件：`message['id'] == 1`。
#   验证条件：`message['error']['code'] == -32601`。
def test_method_not_found(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            message, _ = _rpc_call(websocket, 1, "no.such.method")
            assert message["id"] == 1
            assert message["error"]["code"] == -32601


# 函数说明：test_invalid_params
# 用途：回归验证回归测试与测试辅助中的 `invalid_params` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_rpc_call`。
# 分支与异常：
#   验证条件：`message['id'] == 1`。
#   验证条件：`message['error']['code'] == -32602`。
def test_invalid_params(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            message, _ = _rpc_call(
                websocket,
                1,
                "conversation.create",
                {"title": 123},
            )
            assert message["id"] == 1
            assert message["error"]["code"] == -32602


# 函数说明：test_internal_error_does_not_leak_traceback
# 用途：回归验证回归测试与测试辅助中的 `internal_error_does_not_leak_traceback` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `app.state.dispatcher.register` → `client.websocket_connect` → `_rpc_call` →
# `json.dumps`。
# 分支与异常：
#   验证条件：`message['error']['code'] == -32603`。
#   验证条件：`message['error']['message'] == 'Internal error'`。
#   验证条件：`'secret inner detail' not in json.dumps(message)`。
def test_internal_error_does_not_leak_traceback(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        # 函数说明：test_internal_error_does_not_leak_traceback.explode
        # 用途：处理回归测试与测试辅助中的 `explode` 数据；结果及边界条件见下方说明。
        # 参数：
        #   params：JSON-RPC 方法参数，类型 `dict[str, Any]`。
        #   ctx：RPC 上下文，提供当前应用和连接依赖，类型 `Any`。
        # 返回：类型 `None`；不返回结果值（隐式 None）。
        async def explode(params: dict[str, Any], ctx: Any) -> None:
            raise RuntimeError("secret inner detail")

        app.state.dispatcher.register("test.explode", explode)
        with client.websocket_connect("/rpc") as websocket:
            message, _ = _rpc_call(websocket, 1, "test.explode")
            assert message["error"]["code"] == -32603
            assert message["error"]["message"] == "Internal error"
            assert "secret inner detail" not in json.dumps(message)


# 函数说明：test_request_id_correlation
# 用途：回归验证回归测试与测试辅助中的 `request_id_correlation` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `websocket.send_text` → `_rpc_request` → `json.loads`；另
# 有 1 个调用点。
# 分支与异常：
#   验证条件：`{first.get('id'), second.get('id')} == {10, 20}`。
#   验证条件：`first['result']['provider'] == 'fake'`。
#   验证条件：`second['result']['provider'] == 'fake'`。
def test_request_id_correlation(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            websocket.send_text(_rpc_request(10, "system.info"))
            websocket.send_text(_rpc_request(20, "system.info"))
            first = json.loads(websocket.receive_text())
            second = json.loads(websocket.receive_text())
            assert {first.get("id"), second.get("id")} == {10, 20}
            assert first["result"]["provider"] == "fake"
            assert second["result"]["provider"] == "fake"


# 函数说明：test_notification_gets_no_response
# 用途：回归验证回归测试与测试辅助中的 `notification_gets_no_response` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `websocket.send_text` → `json.dumps` → `_rpc_call`。
# 分支与异常：
#   验证条件：`message['id'] == 7`。
#   验证条件：`'result' in message`。
def test_notification_gets_no_response(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            websocket.send_text(
                json.dumps({"jsonrpc": "2.0", "method": "conversation.list"})
            )
            message, _ = _rpc_call(websocket, 7, "system.info")
            assert message["id"] == 7
            assert "result" in message


# 函数说明：test_system_info
# 用途：回归验证回归测试与测试辅助中的 `system_info` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_rpc_call` → `_require_result`。
# 分支与异常：
#   验证条件：`result['status'] == 'ok'`。
#   验证条件：`result['provider'] == 'fake'`。
#   验证条件：`result['model'] == 'fake-model'`。
#   验证条件：`result['database']`。
def test_system_info(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            message, _ = _rpc_call(websocket, 1, "system.info")
            result = _require_result(message)
            assert result["status"] == "ok"
            assert result["provider"] == "fake"
            assert result["model"] == "fake-model"
            assert result["database"]




# 函数说明：test_conversation_create_list_get
# 用途：回归验证回归测试与测试辅助中的 `conversation_create_list_get` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`any((item['id'] == conversation_id for item in listed['conversations']))`
# 。
#   验证条件：`detail['conversation']['id'] == conversation_id`。
#   验证条件：`detail['messages'] == []`。
def test_conversation_create_list_get(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            created = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )
            conversation_id = created["conversation"]["id"]

            listed = _require_result(_rpc_call(websocket, 2, "conversation.list")[0])
            assert any(
                item["id"] == conversation_id for item in listed["conversations"]
            )

            detail = _require_result(
                _rpc_call(
                    websocket,
                    3,
                    "conversation.get",
                    {"conversation_id": conversation_id},
                )[0]
            )
            assert detail["conversation"]["id"] == conversation_id
            assert detail["messages"] == []


# 函数说明：test_conversation_rename_and_delete
# 用途：回归验证回归测试与测试辅助中的 `conversation_rename_and_delete` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`renamed['conversation']['title'] == '重命名后'`。
#   验证条件：`deleted['deleted'] is True`。
#   验证条件：`message['error']['code'] == -32000`。
def test_conversation_rename_and_delete(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]

            renamed = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "conversation.rename",
                    {"conversation_id": conversation_id, "title": "重命名后"},
                )[0]
            )
            assert renamed["conversation"]["title"] == "重命名后"

            deleted = _require_result(
                _rpc_call(
                    websocket,
                    3,
                    "conversation.delete",
                    {"conversation_id": conversation_id},
                )[0]
            )
            assert deleted["deleted"] is True

            message, _ = _rpc_call(
                websocket,
                4,
                "conversation.delete",
                {"conversation_id": conversation_id},
            )
            assert message["error"]["code"] == -32000


# 函数说明：test_conversation_send_goes_through_service_and_writes_back
# 用途：回归验证回归测试与测试辅助中的
# `conversation_send_goes_through_service_and_writes_back` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`result['conversation_id'] == conversation_id`。
#   验证条件：`result['run']['conversation_id'] == conversation_id`。
#   验证条件：`result['content'] == '已完成'`。
#   验证条件：`calls and calls[0]['conversation_id'] == conversation_id`。
# 副作用与资源：
#   更新对象字段：`application.conversation_service.dispatch`。
def test_conversation_send_goes_through_service_and_writes_back(make_app) -> None:
    app, application, adapter = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]

            original_dispatch = application.conversation_service.dispatch
            calls: list[dict[str, Any]] = []

            # 函数说明：
            # test_conversation_send_goes_through_service_and_writes_back.spy_dispatch
            # 用途：分发`spy`，供回归测试与测试辅助使用。
            # 参数：
            #   **kwargs：额外关键字参数，按实现处理或转交。
            # 返回：返回 `await original_dispatch(**kwargs)`。
            # 关键调用（按源码出现顺序，实际执行取决于分支）：`original_dispatch`。
            # 闭包依赖：从外层读取 `calls`、`original_dispatch`。
            async def spy_dispatch(**kwargs: Any):
                calls.append(kwargs)
                return await original_dispatch(**kwargs)

            application.conversation_service.dispatch = spy_dispatch  

            message, notifications = _rpc_call(
                websocket,
                2,
                "conversation.send",
                {"conversation_id": conversation_id, "content": "帮我总结进度"},
            )
            result = _require_result(message)
            assert result["conversation_id"] == conversation_id
            assert result["run"]["conversation_id"] == conversation_id
            assert result["content"] == "已完成"
            assert calls and calls[0]["conversation_id"] == conversation_id
            assert adapter.requests
            request = adapter.requests[0]
            assert request.messages[0].role is MessageRole.SYSTEM
            assert request.messages[0].content == DEFAULT_SYSTEM_PROMPT
            assert sum(
                message.content == DEFAULT_SYSTEM_PROMPT
                for message in request.messages
            ) == 1

            agent_types = {
                item["params"]["type"]
                for item in notifications
                if item.get("method") == "agent.event"
            }
            assert "agent_started" in agent_types
            assert "agent_completed" in agent_types

            detail = _require_result(
                _rpc_call(
                    websocket,
                    3,
                    "conversation.get",
                    {"conversation_id": conversation_id},
                )[0]
            )
            assert detail["conversation"]["title"] == "帮我总结进度"
            roles = [msg["role"] for msg in detail["messages"]]
            assert "user" in roles and "assistant" in roles
            assert "system" not in roles


# 函数说明：test_conversation_send_missing_conversation
# 用途：回归验证回归测试与测试辅助中的 `conversation_send_missing_conversation` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_rpc_call`。
# 分支与异常：
#   验证条件：`message['error']['code'] == -32000`。
def test_conversation_send_missing_conversation(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            message, _ = _rpc_call(
                websocket,
                1,
                "conversation.send",
                {"conversation_id": "nope", "content": "hi"},
            )
            assert message["error"]["code"] == -32000




# 函数说明：test_cancel_while_send_in_flight
# 用途：回归验证回归测试与测试辅助中的 `cancel_while_send_in_flight` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call` → `websocket.send_text`；
# 另有 5 个调用点。
# 分支与异常：
#   当 `adapter.started.is_set()` 时，结束当前循环。
#   `msg.get('method') == 'agent.event' and msg['params'].get('…` 分支在完成前置处理后结
# 束当前循环。
#   验证条件：`adapter.started.is_set()`。
#   验证条件：`run_id is not None`。
#   验证条件：`result['run']['status'] == 'cancelled'`。
def test_cancel_while_send_in_flight(make_app) -> None:
    app, application, adapter = make_app(blocking=True)
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]

            websocket.send_text(
                _rpc_request(
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "阻塞"},
                )
            )
            for _ in range(200):
                if adapter.started.is_set():
                    break
                time.sleep(0.01)
            assert adapter.started.is_set()

            run_id = None
            for _ in range(50):
                msg = json.loads(websocket.receive_text())
                if (
                    msg.get("method") == "agent.event"
                    and msg["params"].get("type") == "agent_started"
                ):
                    run_id = msg["params"]["run_id"]
                    break
            assert run_id is not None

            message, _ = _rpc_call(websocket, 3, "run.cancel", {"run_id": run_id})
            result = _require_result(message)
            assert result["run"]["status"] == "cancelled"


# 函数说明：test_delete_conversation_stops_in_flight_run
# 用途：回归验证回归测试与测试辅助中的 `delete_conversation_stops_in_flight_run` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call` → `websocket.send_text`；
# 另有 7 个调用点。
# 分支与异常：
#   当 `adapter.started.is_set()` 时，结束当前循环。
#   `message.get('method') == 'agent.event' and message['params'…` 分支在完成前置处理后
# 结束当前循环。
#   `message.get('id') == 3` 分支在完成前置处理后结束当前循环。
#   验证条件：`adapter.started.is_set()`。
#   验证条件：`run_id is not None`。
#   验证条件：`result['deleted'] is True`。
#   验证条件：`result['cancelled_runs'] == 1`。
def test_delete_conversation_stops_in_flight_run(make_app) -> None:

    app, application, adapter = make_app(blocking=True)
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            websocket.send_text(
                _rpc_request(
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "阻塞"},
                )
            )
            for _ in range(200):
                if adapter.started.is_set():
                    break
                time.sleep(0.01)
            assert adapter.started.is_set()

            run_id = None
            for _ in range(50):
                message = json.loads(websocket.receive_text())
                if (
                    message.get("method") == "agent.event"
                    and message["params"].get("type") == "agent_started"
                ):
                    run_id = message["params"]["run_id"]
                    break
            assert run_id is not None

            websocket.send_text(
                _rpc_request(
                    3,
                    "conversation.delete",
                    {"conversation_id": conversation_id},
                )
            )
            delete_response = None
            for _ in range(100):
                message = json.loads(websocket.receive_text())
                if message.get("id") == 3:
                    delete_response = message
                    break
            result = _require_result(delete_response)
            assert result["deleted"] is True
            assert result["cancelled_runs"] == 1
            assert result["deleted_runs"] == 1
            assert result["deleted_traces"] == 1
            assert adapter.cancelled is True

            assert client.portal.call(
                lambda: application.conversation_store.get(conversation_id)
            ) is None
            assert client.portal.call(
                lambda: application.run_store.get(run_id)
            ) is None
            assert client.portal.call(
                lambda: application.trace_store.get(run_id)
            ) is None
            assert application.run_manager.result(run_id) is None




# 函数说明：test_run_list_and_detail
# 用途：回归验证回归测试与测试辅助中的 `run_list_and_detail` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`any((item['id'] == run_id for item in listed['runs']))`。
#   验证条件：`any((item['id'] == run_id for item in filtered['runs']))`。
#   验证条件：`detail['status'] == 'completed'`。
#   验证条件：`detail['source'] == 'manual'`。
def test_run_list_and_detail(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            run_id = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "运行一次"},
                )[0]
            )["run"]["id"]

            listed = _require_result(_rpc_call(websocket, 3, "run.list")[0])
            assert any(item["id"] == run_id for item in listed["runs"])

            filtered = _require_result(
                _rpc_call(
                    websocket,
                    4,
                    "run.list",
                    {"conversation_id": conversation_id},
                )[0]
            )
            assert any(item["id"] == run_id for item in filtered["runs"])

            detail = _require_result(
                _rpc_call(websocket, 5, "run.get", {"run_id": run_id})[0]
            )["run"]
            assert detail["status"] == "completed"
            assert detail["source"] == "manual"
            assert detail["conversation_id"] == conversation_id


# 函数说明：test_run_cancel_terminal_conflict
# 用途：回归验证回归测试与测试辅助中的 `run_cancel_terminal_conflict` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`result['run']['status'] == 'completed'`。
def test_run_cancel_terminal_conflict(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            run_id = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "运行"},
                )[0]
            )["run"]["id"]
            message, _ = _rpc_call(websocket, 3, "run.cancel", {"run_id": run_id})
            result = _require_result(message)
            assert result["run"]["status"] == "completed"


# 函数说明：test_run_recover_keeps_old_interrupted
# 用途：回归验证回归测试与测试辅助中的 `run_recover_keeps_old_interrupted` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call` → `client.portal.call`。
# 分支与异常：
#   验证条件：`result['recovered_from_run_id'] == old_run_id`。
#   验证条件：`new_run_id != old_run_id`。
#   验证条件：`result['run']['status'] == 'completed'`。
#   验证条件：`old['status'] == 'interrupted'`。
def test_run_recover_keeps_old_interrupted(make_app) -> None:

    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]

            application = app.state.application
            old_run_id = client.portal.call(
                lambda: _make_interrupted_run(application, conversation_id)
            )

            message, _ = _rpc_call(
                websocket, 2, "run.recover", {"run_id": old_run_id}
            )
            result = _require_result(message)
            assert result["recovered_from_run_id"] == old_run_id
            new_run_id = result["run"]["id"]
            assert new_run_id != old_run_id
            assert result["run"]["status"] == "completed"

            old = _require_result(
                _rpc_call(websocket, 3, "run.get", {"run_id": old_run_id})[0]
            )["run"]
            assert old["status"] == "interrupted"

            new = _require_result(
                _rpc_call(websocket, 4, "run.get", {"run_id": new_run_id})[0]
            )["run"]
            assert new["recovered_from_run_id"] == old_run_id


# 函数说明：_make_interrupted_run
# 用途：构造运行，供回归测试与测试辅助使用。
# 参数：
#   application：已装配的应用依赖，类型 `Application`。
#   conversation_id：目标会话标识，类型 `str`。
# 返回：类型 `str`；返回 `run.id`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`application.run_store.create` →
# `application.run_store.mark_started` → `application.checkpoint_store.start` →
# `Message` → `application.checkpoint_store.interrupt` →
# `application.run_store.mark_interrupted`。
async def _make_interrupted_run(
    application: Application,
    conversation_id: str,
) -> str:

    run = await application.run_store.create(
        conversation_id=conversation_id,
        user_message="中断任务",
    )
    await application.run_store.mark_started(run.id)
    await application.checkpoint_store.start(
        run.id,
        conversation_id=conversation_id,
        user_message=Message(role=MessageRole.USER, content="中断任务"),
    )
    await application.checkpoint_store.interrupt(run.id, error="simulated stop")
    await application.run_store.mark_interrupted(run.id, error="simulated stop")
    return run.id


# 函数说明：test_trace_get
# 用途：回归验证回归测试与测试辅助中的 `trace_get` 场景，下方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`trace['run']['run_id'] == run_id`。
#   验证条件：`'agent_started' in event_types`。
#   验证条件：`'agent_completed' in event_types`。
def test_trace_get(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            run_id = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "记录轨迹"},
                )[0]
            )["run"]["id"]

            trace = _require_result(
                _rpc_call(websocket, 3, "trace.get", {"run_id": run_id})[0]
            )
            assert trace["run"]["run_id"] == run_id
            event_types = {event["type"] for event in trace["events"]}
            assert "agent_started" in event_types
            assert "agent_completed" in event_types




# 函数说明：test_automation_crud_and_control
# 用途：回归验证回归测试与测试辅助中的 `automation_crud_and_control` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`created['automation']['status'] == 'active'`。
#   验证条件：`any((item['id'] == automation_id for item in listed['automations']))`。
#   验证条件：`detail['automation']['prompt'] == '检查进度'`。
#   验证条件：`paused['automation']['status'] == 'paused'`。
def test_automation_crud_and_control(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]

            created = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "automation.create",
                    {
                        "title": "每小时检查",
                        "prompt": "检查进度",
                        "kind": "interval",
                        "interval_seconds": 3600,
                        "conversation_id": conversation_id,
                    },
                )[0]
            )
            automation_id = created["automation"]["id"]
            assert created["automation"]["status"] == "active"

            listed = _require_result(_rpc_call(websocket, 3, "automation.list")[0])
            assert any(item["id"] == automation_id for item in listed["automations"])

            detail = _require_result(
                _rpc_call(
                    websocket,
                    4,
                    "automation.get",
                    {"automation_id": automation_id},
                )[0]
            )
            assert detail["automation"]["prompt"] == "检查进度"

            paused = _require_result(
                _rpc_call(
                    websocket,
                    5,
                    "automation.pause",
                    {"automation_id": automation_id},
                )[0]
            )
            assert paused["automation"]["status"] == "paused"
            resumed = _require_result(
                _rpc_call(
                    websocket,
                    6,
                    "automation.resume",
                    {"automation_id": automation_id},
                )[0]
            )
            assert resumed["automation"]["status"] == "active"
            cancelled = _require_result(
                _rpc_call(
                    websocket,
                    7,
                    "automation.cancel",
                    {"automation_id": automation_id},
                )[0]
            )
            assert cancelled["automation"]["status"] == "cancelled"


# 函数说明：test_automation_create_validation
# 用途：回归验证回归测试与测试辅助中的 `automation_create_validation` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_rpc_call`。
# 分支与异常：
#   验证条件：`message['error']['code'] == -32602`。
def test_automation_create_validation(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            message, _ = _rpc_call(
                websocket,
                1,
                "automation.create",
                {"title": "t", "prompt": "p", "kind": "once"},
            )
            assert message["error"]["code"] == -32602


# 函数说明：test_automation_run_broadcasts_and_keeps_provenance
# 用途：回归验证回归测试与测试辅助中的 `automation_run_broadcasts_and_keeps_provenance`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call` →
# `(datetime.now(UTC) + timedelta(seconds=1)).isoformat`；另有 5 个调用点。
# 分支与异常：
#   当 `msg['params'].get('type') == 'agent_completed'` 时，结束当前循环。
#   验证条件：`saw_agent_event`。
#   验证条件：`automation_runs`。
#   验证条件：`automation_runs[0]['source_id'] == automation_id`。
#   验证条件：`'assistant' in roles`。
def test_automation_run_broadcasts_and_keeps_provenance(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            run_at = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
            created = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "automation.create",
                    {
                        "title": "稍后总结",
                        "prompt": "总结项目进度",
                        "kind": "once",
                        "run_at": run_at,
                        "conversation_id": conversation_id,
                    },
                )[0]
            )
            automation_id = created["automation"]["id"]

            time.sleep(2.5)

            saw_agent_event = False
            for _ in range(100):
                msg = json.loads(websocket.receive_text())
                if msg.get("method") == "agent.event":
                    saw_agent_event = True
                    if msg["params"].get("type") == "agent_completed":
                        break
            assert saw_agent_event, "automation Run 未广播 agent.event"

            runs = _require_result(
                _rpc_call(
                    websocket,
                    3,
                    "run.list",
                    {"conversation_id": conversation_id},
                )[0]
            )
            automation_runs = [
                run for run in runs["runs"] if run["source"] == "automation"
            ]
            assert automation_runs, "未找到 source=automation 的 Run"
            assert automation_runs[0]["source_id"] == automation_id

            detail = _require_result(
                _rpc_call(
                    websocket,
                    4,
                    "conversation.get",
                    {"conversation_id": conversation_id},
                )[0]
            )
            roles = [msg["role"] for msg in detail["messages"]]
            assert "assistant" in roles




# 函数说明：test_disconnect_does_not_cancel_running_run
# 用途：回归验证回归测试与测试辅助中的 `disconnect_does_not_cancel_running_run` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call` → `websocket.send_text`；
# 另有 6 个调用点。
# 分支与异常：
#   当 `adapter.started.is_set()` 时，结束当前循环。
#   `msg.get('method') == 'agent.event' and msg['params'].get('…` 分支在完成前置处理后结
# 束当前循环。
#   验证条件：`adapter.started.is_set()`。
#   验证条件：`run_id is not None`。
#   验证条件：`run is not None and run.status.value == 'running'`。
#   验证条件：`adapter.cancelled is False`。
def test_disconnect_does_not_cancel_running_run(make_app) -> None:
    app, application, adapter = make_app(blocking=True)
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            websocket.send_text(
                _rpc_request(
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "阻塞"},
                )
            )
            for _ in range(200):
                if adapter.started.is_set():
                    break
                time.sleep(0.01)
            assert adapter.started.is_set()
            run_id = None
            for _ in range(50):
                msg = json.loads(websocket.receive_text())
                if (
                    msg.get("method") == "agent.event"
                    and msg["params"].get("type") == "agent_started"
                ):
                    run_id = msg["params"]["run_id"]
                    break
            assert run_id is not None
        run = client.portal.call(lambda: application.run_manager.get_run(run_id))
        assert run is not None and run.status.value == "running"
        assert adapter.cancelled is False
        client.portal.call(lambda: application.run_manager.cancel(run_id))




# 函数说明：test_approval_approve_flow_via_websocket
# 用途：回归验证回归测试与测试辅助中的 `approval_approve_flow_via_websocket` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_with_tool_call` →
#  `_model_response` → `TestClient` → `client.websocket_connect` → `client.portal.call`
# ；另有 7 个调用点。
# 分支与异常：
#   验证条件：`approval['status'] == 'pending'`。
#   验证条件：`approval['tool_name'] == 'approval_probe'`。
#   验证条件：`approval['run_id'] is not None`。
#   验证条件：`approval['conversation_id'] == conversation_id`。
def test_approval_approve_flow_via_websocket(make_app) -> None:

    app, application, _ = make_app(
        responses=[
            _model_with_tool_call(),
            _model_response(content="审批通过，任务完成"),
        ]
    )
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            client.portal.call(
                lambda: application.tool_registry.register(ApprovalProbeTool())
            )
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]

            websocket.send_text(
                _rpc_request(
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "执行审批"},
                )
            )
            required = _drain_until(
                websocket,
                lambda msg: msg.get("method") == "approval.required",
            )
            approval = required["params"]["approval"]
            assert approval["status"] == "pending"
            assert approval["tool_name"] == "approval_probe"
            assert approval["run_id"] is not None
            assert approval["conversation_id"] == conversation_id
            run_id = approval["run_id"]

            websocket.send_text(
                _rpc_request(
                    3,
                    "approval.approve",
                    {"approval_id": approval["id"]},
                )
            )
            send_response: dict[str, Any] | None = None
            approve_response: dict[str, Any] | None = None
            resolved_seen = False
            while send_response is None or approve_response is None:
                msg = json.loads(websocket.receive_text())
                if msg.get("id") == 2:
                    send_response = msg
                elif msg.get("id") == 3:
                    approve_response = msg
                if msg.get("method") == "approval.resolved":
                    resolved_seen = True
            assert approve_response is not None
            assert approve_response["result"]["approval"]["status"] == "approved"
            assert resolved_seen, "应广播 approval.resolved"

            result = _require_result(send_response)
            assert result["run"]["id"] == run_id
            assert result["run"]["status"] == "completed"
            assert result["result"]["messages"]

            run = _require_result(
                _rpc_call(websocket, 4, "run.get", {"run_id": run_id})[0]
            )["run"]
            assert run["status"] == "completed"


# 函数说明：test_approval_deny_blocks_tool_execution
# 用途：回归验证回归测试与测试辅助中的 `approval_deny_blocks_tool_execution` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_with_tool_call` →
#  `_model_response` → `TestClient` → `client.websocket_connect` → `ApprovalProbeTool`；
# 另有 8 个调用点。
# 分支与异常：
#   验证条件：`deny_response['result']['approval']['status'] == 'denied'`。
#   验证条件：`result['run']['status'] == 'completed'`。
#   验证条件：`probe.executions == 0`。
def test_approval_deny_blocks_tool_execution(make_app) -> None:
    app, application, _ = make_app(
        responses=[
            _model_with_tool_call(),
            _model_response(content="审批被拒绝"),
        ]
    )
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            probe = ApprovalProbeTool()
            client.portal.call(lambda: application.tool_registry.register(probe))
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            websocket.send_text(
                _rpc_request(
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "执行审批"},
                )
            )
            required = _drain_until(
                websocket,
                lambda msg: msg.get("method") == "approval.required",
            )
            approval_id = required["params"]["approval"]["id"]

            websocket.send_text(
                _rpc_request(
                    3,
                    "approval.deny",
                    {"approval_id": approval_id},
                )
            )
            send_response: dict[str, Any] | None = None
            deny_response: dict[str, Any] | None = None
            while send_response is None or deny_response is None:
                msg = json.loads(websocket.receive_text())
                if msg.get("id") == 2:
                    send_response = msg
                elif msg.get("id") == 3:
                    deny_response = msg
            assert deny_response["result"]["approval"]["status"] == "denied"
            result = _require_result(send_response)
            assert result["run"]["status"] == "completed"
            assert probe.executions == 0


# 函数说明：test_approval_duplicate_resolve_rejected
# 用途：回归验证回归测试与测试辅助中的 `approval_duplicate_resolve_rejected` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_with_tool_call` →
#  `_model_response` → `TestClient` → `client.websocket_connect` → `client.portal.call`
# ；另有 5 个调用点。
# 分支与异常：
#   验证条件：`approved['result']['approval']['status'] == 'approved'`。
#   验证条件：`again['error']['code'] == -32001`。
#   验证条件：`denied['error']['code'] == -32001`。
def test_approval_duplicate_resolve_rejected(make_app) -> None:
    app, application, _ = make_app(
        responses=[
            _model_with_tool_call(),
            _model_response(content="审批通过"),
        ]
    )
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            client.portal.call(
                lambda: application.tool_registry.register(ApprovalProbeTool())
            )
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            websocket.send_text(
                _rpc_request(
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "执行审批"},
                )
            )
            required = _drain_until(
                websocket,
                lambda msg: msg.get("method") == "approval.required",
            )
            approval_id = required["params"]["approval"]["id"]

            approved, _ = _rpc_call(
                websocket, 3, "approval.approve", {"approval_id": approval_id}
            )
            assert approved["result"]["approval"]["status"] == "approved"

            again, _ = _rpc_call(
                websocket, 4, "approval.approve", {"approval_id": approval_id}
            )
            assert again["error"]["code"] == -32001
            denied, _ = _rpc_call(
                websocket, 5, "approval.deny", {"approval_id": approval_id}
            )
            assert denied["error"]["code"] == -32001


# 函数说明：test_approval_list_and_get
# 用途：回归验证回归测试与测试辅助中的 `approval_list_and_get` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_with_tool_call` →
#  `_model_response` → `TestClient` → `client.websocket_connect` → `client.portal.call`
# ；另有 5 个调用点。
# 分支与异常：
#   验证条件：`any((item['id'] == approval_id for item in listed['approvals']))`。
#   验证条件：`any((item['id'] == approval_id for item in pending['approvals']))`。
#   验证条件：`detail['id'] == approval_id`。
#   验证条件：`detail['arguments']['value'] == 7`。
def test_approval_list_and_get(make_app) -> None:
    app, application, _ = make_app(
        responses=[
            _model_with_tool_call(),
            _model_response(content="审批通过"),
        ]
    )
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            client.portal.call(
                lambda: application.tool_registry.register(ApprovalProbeTool())
            )
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            websocket.send_text(
                _rpc_request(
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "执行审批"},
                )
            )
            required = _drain_until(
                websocket,
                lambda msg: msg.get("method") == "approval.required",
            )
            approval_id = required["params"]["approval"]["id"]

            listed = _require_result(
                _rpc_call(websocket, 3, "approval.list")[0]
            )
            assert any(
                item["id"] == approval_id for item in listed["approvals"]
            )
            pending = _require_result(
                _rpc_call(
                    websocket, 4, "approval.list", {"status": "pending"}
                )[0]
            )
            assert any(item["id"] == approval_id for item in pending["approvals"])

            detail = _require_result(
                _rpc_call(
                    websocket, 5, "approval.get", {"approval_id": approval_id}
                )[0]
            )["approval"]
            assert detail["id"] == approval_id
            assert detail["arguments"]["value"] == 7


# 函数说明：test_approval_not_auto_resolved_on_disconnect
# 用途：回归验证回归测试与测试辅助中的 `approval_not_auto_resolved_on_disconnect` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_with_tool_call` →
#  `_model_response` → `TestClient` → `client.websocket_connect` → `client.portal.call`
# ；另有 5 个调用点。
# 分支与异常：
#   验证条件：`pending is not None`。
#   验证条件：`pending.status.value == 'pending'`。
#   验证条件：`approved.status.value == 'approved'`。
def test_approval_not_auto_resolved_on_disconnect(make_app) -> None:

    app, application, _ = make_app(
        responses=[
            _model_with_tool_call(),
            _model_response(content="审批通过"),
        ]
    )
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            client.portal.call(
                lambda: application.tool_registry.register(ApprovalProbeTool())
            )
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            websocket.send_text(
                _rpc_request(
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "执行审批"},
                )
            )
            required = _drain_until(
                websocket,
                lambda msg: msg.get("method") == "approval.required",
            )
            approval_id = required["params"]["approval"]["id"]
            run_id = required["params"]["approval"]["run_id"]
        pending = client.portal.call(
            lambda: application.approval_store.get(approval_id)
        )
        assert pending is not None
        assert pending.status.value == "pending"
        approved = client.portal.call(
            lambda: application.approval_gate.approve(approval_id)
        )
        assert approved.status.value == "approved"
        client.portal.call(lambda: application.run_manager.cancel(run_id))


# 函数说明：test_automation_run_can_produce_approval
# 用途：回归验证回归测试与测试辅助中的 `automation_run_can_produce_approval` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_with_tool_call` →
#  `_model_response` → `TestClient` → `client.websocket_connect` → `client.portal.call`
# ；另有 9 个调用点。
# 分支与异常：
#   当 `detail['status'] == 'completed'` 时，结束当前循环。
#   验证条件：`approval['status'] == 'pending'`。
#   验证条件：`approval['tool_name'] == 'approval_probe'`。
#   验证条件：`detail['status'] == 'completed'`。
#   验证条件：`detail['source'] == 'automation'`。
def test_automation_run_can_produce_approval(make_app) -> None:

    app, application, _ = make_app(
        responses=[
            _model_with_tool_call(),
            _model_response(content="自动化审批完成"),
        ]
    )
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            client.portal.call(
                lambda: application.tool_registry.register(ApprovalProbeTool())
            )
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            run_at = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
            created = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "automation.create",
                    {
                        "title": "自动审批任务",
                        "prompt": "执行审批",
                        "kind": "once",
                        "run_at": run_at,
                        "conversation_id": conversation_id,
                    },
                )[0]
            )
            automation_id = created["automation"]["id"]

            required = _drain_until(
                websocket,
                lambda msg: msg.get("method") == "approval.required",
            )
            approval = required["params"]["approval"]
            assert approval["status"] == "pending"
            assert approval["tool_name"] == "approval_probe"

            websocket.send_text(
                _rpc_request(
                    3,
                    "approval.approve",
                    {"approval_id": approval["id"]},
                )
            )
            completed = _drain_until(
                websocket,
                lambda msg: (
                    msg.get("method") == "agent.event"
                    and msg["params"].get("type") == "agent_completed"
                ),
            )
            run_id = completed["params"]["run_id"]

            detail: dict[str, Any] = {}
            for _ in range(200):
                detail = _require_result(
                    _rpc_call(websocket, 4, "run.get", {"run_id": run_id})[0]
                )["run"]
                if detail["status"] == "completed":
                    break
                time.sleep(0.02)
            assert detail["status"] == "completed"
            assert detail["source"] == "automation"
            assert detail["source_id"] == automation_id




# 函数说明：test_plan_mode_send_creates_pending_task_and_accept
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_send_creates_pending_task_and_accept`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_response` →
# `ToolCall` → `TestClient` → `client.websocket_connect` → `_require_result`；另有 1 个
# 调用点。
# 分支与异常：
#   验证条件：`result['run']['mode'] == 'plan'`。
#   验证条件：`result['plan_task_id']`。
#   验证条件：`detail['status'] == 'pending'`。
#   验证条件：`detail['goal'] == '实现 Evidence Runtime V1'`。
def test_plan_mode_send_creates_pending_task_and_accept(make_app) -> None:
    app, _, _ = make_app(
        responses=[
            _model_response(
                tool_calls=(
                    ToolCall(
                        id="plan-1",
                        name="task_create",
                        arguments={
                            "title": "实现 Evidence Runtime",
                            "goal": "实现 Evidence Runtime V1",
                            "steps": [
                                {"title": "定义 protocol"},
                                {"title": "实现 observe"},
                                {"title": "实现 click/type"},
                            ],
                        },
                    ),
                )
            ),
            _model_response(content="计划已形成"),
        ]
    )
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            message, _ = _rpc_call(
                websocket,
                2,
                "conversation.send",
                {
                    "conversation_id": conversation_id,
                    "content": "帮我实现 Evidence Runtime",
                    "mode": "plan",
                },
            )
            result = _require_result(message)
            assert result["run"]["mode"] == "plan"
            assert result["plan_task_id"]
            plan_task_id = result["plan_task_id"]

            detail = _require_result(
                _rpc_call(websocket, 3, "task.get", {"task_id": plan_task_id})[0]
            )["task"]
            assert detail["status"] == "pending"
            assert detail["goal"] == "实现 Evidence Runtime V1"
            assert len(detail["steps"]) == 3

            accepted = _require_result(
                _rpc_call(
                    websocket, 4, "task.plan_accept", {"task_id": plan_task_id}
                )[0]
            )["task"]
            assert accepted["status"] == "active"

            again = _rpc_call(
                websocket, 5, "task.plan_accept", {"task_id": plan_task_id}
            )[0]
            assert again["error"]["code"] == -32001
            reject = _rpc_call(
                websocket, 6, "task.plan_reject", {"task_id": plan_task_id}
            )[0]
            assert reject["error"]["code"] == -32001


# 函数说明：test_plan_mode_send_reject_task
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_send_reject_task` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_response` →
# `ToolCall` → `TestClient` → `client.websocket_connect` → `_require_result`；另有 1 个
# 调用点。
# 分支与异常：
#   验证条件：`rejected['status'] == 'cancelled'`。
def test_plan_mode_send_reject_task(make_app) -> None:
    app, _, _ = make_app(
        responses=[
            _model_response(
                tool_calls=(
                    ToolCall(
                        id="plan-1",
                        name="task_create",
                        arguments={
                            "title": "另一个计划",
                            "goal": "目标",
                            "steps": [{"title": "第一步"}],
                        },
                    ),
                )
            ),
            _model_response(content="计划已形成"),
        ]
    )
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            result = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "conversation.send",
                    {
                        "conversation_id": conversation_id,
                        "content": "规划",
                        "mode": "plan",
                    },
                )[0]
            )
            plan_task_id = result["plan_task_id"]
            rejected = _require_result(
                _rpc_call(
                    websocket, 3, "task.plan_reject", {"task_id": plan_task_id}
                )[0]
            )["task"]
            assert rejected["status"] == "cancelled"


# 函数说明：test_plan_mode_send_blocks_side_effect_tool
# 用途：回归验证回归测试与测试辅助中的 `plan_mode_send_blocks_side_effect_tool` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_response` →
# `ToolCall` → `TestClient` → `client.websocket_connect` → `_require_result`；另有 1 个
# 调用点。
# 分支与异常：
#   验证条件：`result['run']['mode'] == 'plan'`。
#   验证条件：`tool_result['success'] is False`。
#   验证条件：`'not allowed in plan mode' in (tool_result['error'] or '')`。
#   验证条件：`result['plan_task_id'] is None`。
def test_plan_mode_send_blocks_side_effect_tool(make_app) -> None:
    app, _, _ = make_app(
        responses=[
            _model_response(
                tool_calls=(
                    ToolCall(
                        id="bad-1",
                        name="write_file",
                        arguments={"path": "evil.txt", "content": "x"},
                    ),
                )
            ),
            _model_response(content="不应能写文件"),
        ]
    )
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            result = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "conversation.send",
                    {
                        "conversation_id": conversation_id,
                        "content": "写个文件",
                        "mode": "plan",
                    },
                )[0]
            )
            assert result["run"]["mode"] == "plan"
            tool_result = result["result"]["tool_calls"][0]["result"]
            assert tool_result["success"] is False
            assert "not allowed in plan mode" in (tool_result["error"] or "")
            assert result["plan_task_id"] is None


# 函数说明：test_conversation_send_default_mode_is_normal
# 用途：回归验证回归测试与测试辅助中的 `conversation_send_default_mode_is_normal` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`result['run']['mode'] == 'normal'`。
def test_conversation_send_default_mode_is_normal(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            result = _require_result(
                _rpc_call(
                    websocket,
                    2,
                    "conversation.send",
                    {"conversation_id": conversation_id, "content": "你好"},
                )[0]
            )
            assert result["run"]["mode"] == "normal"


# 函数说明：test_conversation_send_invalid_mode
# 用途：回归验证回归测试与测试辅助中的 `conversation_send_invalid_mode` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`message['error']['code'] == -32602`。
def test_conversation_send_invalid_mode(make_app) -> None:
    app, _, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            message, _ = _rpc_call(
                websocket,
                2,
                "conversation.send",
                {
                    "conversation_id": conversation_id,
                    "content": "hi",
                    "mode": "bogus",
                },
            )
            assert message["error"]["code"] == -32602




# 函数说明：test_server_shutdown_closes_resources
# 用途：回归验证回归测试与测试辅助中的 `server_shutdown_closes_resources` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `TestClient` →
# `client.websocket_connect` → `_require_result` → `_rpc_call`。
# 分支与异常：
#   验证条件：`application._started is True`。
#   验证条件：`application._started is False`。
#   验证条件：`application.automation_scheduler._running == set()`。
#   验证条件：`application.automation_scheduler._job_ids == {}`。
def test_server_shutdown_closes_resources(make_app) -> None:
    app, application, _ = make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            _require_result(_rpc_call(websocket, 1, "system.info")[0])
        assert application._started is True
    assert application._started is False
    assert application.automation_scheduler._running == set()
    assert application.automation_scheduler._job_ids == {}


# 函数说明：test_application_start_close_idempotent
# 用途：回归验证回归测试与测试辅助中的 `application_start_close_idempotent` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `RepeatingFakeAdapter` → `_model_response` → `ModelAdapterRegistry` → `ModelSettings`
# ；另有 7 个调用点。
# 分支与异常：
#   验证条件：`application._started is True`。
#   验证条件：`application._started is False`。
async def test_application_start_close_idempotent(tmp_path: Path) -> None:
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = RepeatingFakeAdapter(config, _model_response("ok"))
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)

    application = Application(
        provider="fake",
        model="fake-model",
        database=tmp_path / "muharness.db",
        tasks_dir=tmp_path / "tasks",
        mcp_config=tmp_path / "mcp.json",
        memory_dir=tmp_path / "memory",
        skills_user_dir=tmp_path / "skills-user",
        skills_project_dir=tmp_path / "skills-project",
        registry=registry,
        memory_reflection_config=MemoryReflectionConfig(
            _env_file=None, enabled=False
        ),
        memory_maintenance_config=MemoryMaintenanceConfig(
            _env_file=None, enabled=False
        ),
        skill_learning_settings=SkillLearningSettings(
            _env_file=None,
            skill_learning_enabled=False,
            skill_learning_data_dir=tmp_path / "skill-learning",
        ),
    )
    await application.start()
    await application.start()  
    assert application._started is True
    await application.close()
    await application.close()  
    assert application._started is False


# 函数说明：test_application_wires_artifact_publish_into_mea_role_runtimes
# 用途：回归验证回归测试与测试辅助中的
# `application_wires_artifact_publish_into_mea_role_runtimes` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `(workspace / 'deliverable.txt').write_text` → `ToolCall` → `make_app` →
# `_model_response` → `application.start`；另有 9 个调用点。
# 分支与异常：
#   验证条件：`len(adapter.requests) == 4`。
#   验证条件：`'artifact_publish' in executor_tools`。
#   验证条件：`'artifact_publish' not in manager_tools`。
#   验证条件：`'artifact_publish' not in auditor_tools`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / 'deliverable.txt').write_text`。
async def test_application_wires_artifact_publish_into_mea_role_runtimes(
    make_app,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "deliverable.txt").write_text("production wiring", encoding="utf-8")
    publish_call = ToolCall(
        id="publish-from-executor",
        name="artifact_publish",
        arguments={"path": "deliverable.txt"},
    )
    _, application, adapter = make_app(
        workspace_root=workspace,
        responses=[
            _model_response(tool_calls=(publish_call,)),
            _model_response("executor complete"),
            _model_response("manager complete"),
            _model_response("auditor complete"),
        ],
    )

    await application.start()
    try:
        task = await application.task_store.create(
            title="Publish deliverable",
            owner_conversation_id="conversation-1",
            steps=(TaskStep(id="publish", title="Publish the file"),),
        )
        await application.task_store.plan_accept(task.id)

        executor_result = await application.mea_runtimes[AgentMode.EXECUTE].run(
            "Publish deliverable.txt",
            conversation_id="conversation-1",
            run_id="executor-run",
            mode=AgentMode.EXECUTE,
        )
        await application.mea_runtimes[AgentMode.MANAGE].run(
            "Inspect task state",
            conversation_id="conversation-1",
            run_id="manager-run",
            mode=AgentMode.MANAGE,
        )
        await application.mea_runtimes[AgentMode.AUDIT].run(
            "Audit the result",
            conversation_id="conversation-1",
            run_id="auditor-run",
            mode=AgentMode.AUDIT,
        )
    finally:
        await application.close()

    assert len(adapter.requests) == 4
    executor_tools = {definition.name for definition in adapter.requests[0].tools}
    manager_tools = {definition.name for definition in adapter.requests[2].tools}
    auditor_tools = {definition.name for definition in adapter.requests[3].tools}
    assert "artifact_publish" in executor_tools
    assert "artifact_publish" not in manager_tools
    assert "artifact_publish" not in auditor_tools

    [publication] = executor_result.tool_calls
    assert publication.result.success, publication.result.error
    receipt = json.loads(publication.result.output)
    assert receipt["task_id"] == task.id
    assert receipt["run_id"] == "executor-run"
    [stored] = await application.artifact_service.store.list()
    assert stored.id == receipt["id"]
    assert stored.task_id == task.id


_MEA_REPORT = (
    "状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n"
    "审计事实: 已核验。\n\n验收约束反查:\n契约结论: aligned\n阻断约束: 无\n"
    "范围外约束: 无\n给任务管理器的状态更新: 见上"
)


# 函数说明：_mea_manager
# 用途：返回 `f'当前任务状态:\n- 已完成: 见审计\n\n任务契约:\n- 目标状态: 读到 CSV 表头\
# n\n步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推进\n…`，提供 回归测试与测试辅助
#  的派生值。
# 参数：
#   route：路由输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `f'当前任务状态:\n- 已完成: 见审计\n\n任务契约:\n- 目标状态: 读
# 到 CSV 表头\n\n步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推进\n…`。
def _mea_manager(route: str) -> str:
    return (
        "当前任务状态:\n- 已完成: 见审计\n\n任务契约:\n- 目标状态: 读到 CSV 表头\n\n"
        f"步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推进\n\n{route}"
    )


# 函数说明：test_mea_start_runs_to_completion_over_rpc
# 用途：回归验证回归测试与测试辅助中的 `mea_start_runs_to_completion_over_rpc` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   make_app：构造隔离测试应用的工厂夹具。
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`make_app` → `_model_response` →
# `ToolCall` → `_mea_manager` → `workspace.mkdir` → `workspace.resolve`；另有 6 个调用点
# 。
# 分支与异常：
#   验证条件：`'memory_create' not in tools`。
#   验证条件：`bad['error']['code'] == -32602`。
#   验证条件：`started['task']['status'] == 'active'`。
#   验证条件：`[r['kind'] for r in detail['rounds']] == ['normal', 'final_audit']`。
# 副作用与资源：
#   更新对象字段：`application.workspace_root`。
#   文件或资源访问：`workspace.mkdir`。
def test_mea_start_runs_to_completion_over_rpc(make_app, tmp_path) -> None:
    app, application, _ = make_app(
        responses=[
            # 计划模式：创建带验收标准的计划
            _model_response(
                tool_calls=(
                    ToolCall(
                        id="plan-1",
                        name="task_create",
                        arguments={
                            "title": "读取 CSV",
                            "goal": "读到 users.csv 的表头",
                            "steps": [{"title": "读取表头", "acceptance": "报告里有表头"}],
                        },
                    ),
                )
            ),
            _model_response(content="计划已形成"),
            # 长任务：Manager → Executor → Auditor，然后最终验收和最终回复
            _model_response(content=_mea_manager(
                "下一步: 执行任务\n步骤: s1\n任务: 读取表头\n验收标准: 见步骤验收\n"
                "相关审计报告: 无\n相关已审计状态: 无\n边界: 只读"
            )),
            _model_response(content="表头是 id,name"),
            _model_response(content=_MEA_REPORT),
            _model_response(content=_mea_manager("下一步: 最终验收\n验收重点: 表头")),
            _model_response(content=_MEA_REPORT),
            _model_response(content="长任务已完成：表头是 id,name。"),
        ]
    )
    workspace = tmp_path / "mea-workspace"
    workspace.mkdir()
    application.workspace_root = workspace.resolve()
    with TestClient(app) as client:
        with client.websocket_connect("/rpc") as websocket:
            conversation_id = _require_result(
                _rpc_call(websocket, 1, "conversation.create")[0]
            )["conversation"]["id"]
            plan = _require_result(_rpc_call(
                websocket, 2, "conversation.send",
                {"conversation_id": conversation_id, "content": "读一下 users.csv 的表头",
                 "mode": "plan"},
            )[0])
            task_id = plan["plan_task_id"]

            tools = _require_result(_rpc_call(websocket, 3, "mea.tools")[0])["tools"]
            assert "memory_create" not in tools
            bad = _rpc_call(websocket, 4, "mea.start", {
                "conversation_id": conversation_id, "task_id": task_id,
                "extra_tools": ["memory_create"],
            })[0]
            assert bad["error"]["code"] == -32602

            message, early = _rpc_call(websocket, 5, "mea.start", {
                "conversation_id": conversation_id, "task_id": task_id,
            })
            started = _require_result(message)
            mea_id = started["mea"]["id"]
            assert started["task"]["status"] == "active"

            # 函数说明：test_mea_start_runs_to_completion_over_rpc.completed
            # 用途：返回 `m.get('method') == 'mea.status' and m['params']['mea']['id'] =
            # = mea_id and (m['params']['…`，提供 回归测试与测试辅助 的派生值。
            # 参数：
            #   m：`m`输入或配置值，类型 `dict[str, Any]`；读取键 `method`、`params`。
            # 返回：类型 `bool`；返回 `m.get('method') == 'mea.status' and m['params']['
            # mea']['id'] == mea_id and (m['params']['…`。
            # 闭包依赖：从外层读取 `mea_id`。
            def completed(m: dict[str, Any]) -> bool:
                return (
                    m.get("method") == "mea.status"
                    and m["params"]["mea"]["id"] == mea_id
                    and m["params"]["mea"]["status"] == "completed"
                )

            if not any(completed(m) for m in early):
                _drain_until(websocket, completed, limit=5_000)

            detail = _require_result(_rpc_call(websocket, 6, "mea.get", {"mea_id": mea_id})[0])
            assert [r["kind"] for r in detail["rounds"]] == ["normal", "final_audit"]
            assert detail["task"]["status"] == "completed"
            assert detail["mea"]["final_response"] == "长任务已完成：表头是 id,name。"

            messages = _require_result(_rpc_call(
                websocket, 7, "conversation.get", {"conversation_id": conversation_id}
            )[0])["messages"]
            assert messages[-1]["content"] == "长任务已完成：表头是 id,name。"

            runs = _require_result(_rpc_call(
                websocket, 8, "run.list", {"conversation_id": conversation_id}
            )[0])["runs"]
            sources = {run["source"] for run in runs if run["source_id"] == mea_id}
            assert sources == {"mea:manager", "mea:executor", "mea:auditor"}

            late = _require_result(_rpc_call(
                websocket, 9, "mea.note", {"mea_id": mea_id, "text": "再加一列"}
            )[0])
            assert late["accepted"] is False and late["reason"] == "mea_finalized"
