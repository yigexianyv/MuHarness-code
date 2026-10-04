"""MeaRunner 接真实 RunManager + 三个角色 AgentRuntime 的集成测试。

模型是离线脚本；其余都是真实组件：runs 表、检查点、工具执行器、角色边界、Task 存储、
会话存储。验证子 Run 的 ID、来源和模式真的落在 runs 表里，auditor 的写调用在执行前被拒绝，
最终回复只追加一次。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import SecretStr

from app.domain.conversation.store import SQLiteConversationStore
from app.domain.task import FileTaskStore, TaskStatus, TaskStep, TaskStepStatus
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
)
from app.records.trace import SQLiteTraceEventHandler, SQLiteTraceStore
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.checkpoint import SQLiteCheckpointStore
from app.runtime.mea.events import MeaRunGateway
from app.runtime.mea.executor_evidence import ExecutorEvidenceProvider
from app.runtime.mea.models import MeaStatus, RoundKind, RoundPhase
from app.runtime.mea.prompts import FINAL_RESPONSE_INSTRUCTIONS
from app.runtime.mea.runner import ROLE_NAMES, MeaRunner
from app.runtime.mea.store import SQLiteMeaStore
from app.runtime.run import RunManager, RunStatus, SQLiteRunStore
from app.tools.base import BaseTool
from app.tools.builtin.read_file import ReadFileTool
from app.tools.builtin.write_file import WriteFileTool
from app.tools.registry import ToolRegistry


# 函数说明：manager
# 用途：返回 `f'当前任务状态:\n{state}\n\n任务契约:\n- 目标状态: users 表包含 CSV 全部数
# 据\n\n步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按…`，提供 回归测试与测试辅助 的派生
# 值。
# 参数：
#   route：路由输入或配置值，类型 `str`。
#   state：当前状态快照，类型 `str`；默认 `'- 已完成: 见审计'`。
# 返回：类型 `str`；返回 `f'当前任务状态:\n{state}\n\n任务契约:\n- 目标状态: users 表包
# 含 CSV 全部数据\n\n步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按…`。
def manager(route: str, *, state: str = "- 已完成: 见审计") -> str:
    return (
        f"当前任务状态:\n{state}\n\n任务契约:\n- 目标状态: users 表包含 CSV 全部数据\n\n"
        f"步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推进\n\n{route}"
    )


# 函数说明：execute
# 用途：执行回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   step：当前任务步骤，类型 `str`。
# 返回：类型 `str`；返回 `f'下一步: 执行任务\n步骤: {step}\n任务: 完成 {step}\n验收标准:
#  见步骤验收\n相关审计报告: 无\n相关已审计状态: 无\n边界: 不改表结构'`。
def execute(step: str) -> str:
    return (
        f"下一步: 执行任务\n步骤: {step}\n任务: 完成 {step}\n验收标准: 见步骤验收\n"
        "相关审计报告: 无\n相关已审计状态: 无\n边界: 不改表结构"
    )


FINAL = "下一步: 最终验收\n验收重点: 行数与编码"


# 函数说明：report
# 用途：返回 `'状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n审
# 计事实: 已独立核验。\n\n验收约束反查:\n契约结论:…`，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `str`；返回 `'状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收:
# satisfied\n审计事实: 已独立核验。\n\n验收约束反查:\n契约结论:…`。
def report() -> str:
    return (
        "状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n"
        "审计事实: 已独立核验。\n\n验收约束反查:\n"
        "契约结论: aligned\n阻断约束: 无\n范围外约束: 无\n"
        "给任务管理器的状态更新: 见上"
    )


class ScriptedAdapter(ModelAdapter):
    """按请求内容回复：reply(request, 第几次调用) -> ModelResponse。"""

    # 函数说明：ScriptedAdapter.__init__
    # 用途：初始化 ScriptedAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   reply：用户回复或工具响应，类型 `Callable[[ModelRequest], ModelResponse]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self._reply`、`self.requests`。
    def __init__(
        self,
        config: ProviderConfig,
        reply: Callable[[ModelRequest], ModelResponse],
    ) -> None:
        super().__init__(config)
        self._reply = reply
        self.requests: list[ModelRequest] = []

    # 函数说明：ScriptedAdapter.complete
    # 用途：完成ScriptedAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `self._reply(request)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._reply`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self._reply(request)

    # 函数说明：ScriptedAdapter.close
    # 用途：关闭ScriptedAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


class FakeWriteTool(BaseTool):
    """和内置 write_file 同名的替身：角色边界按名字判断，所以足够验证拒绝发生在执行之前。"""

    definition = ToolDefinition(
        name="write_file",
        description="Write a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    )

    # 函数说明：FakeWriteTool.__init__
    # 用途：初始化 FakeWriteTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   root：当前操作的根目录。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.root`、`self.calls`。
    def __init__(self, root) -> None:
        self.root = root
        self.calls: list[str] = []

    # 函数说明：FakeWriteTool.execute
    # 用途：执行FakeWriteTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `path`
    # 、`content`。
    # 返回：类型 `str`；返回 `'ok'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `(self.root / str(arguments['path'])).write_text`。
    # 副作用与资源：
    #   文件或资源访问：`(self.root / str(arguments['path'])).write_text`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        self.calls.append(str(arguments["path"]))
        (self.root / str(arguments["path"])).write_text(str(arguments["content"]), encoding="utf-8")
        return "ok"


# 函数说明：_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str | None`；默认 `None`。
#   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`；默认 `()`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def _response(content: str | None = None, *, tool_calls: tuple[ToolCall, ...] = ()) -> ModelResponse:
    return ModelResponse(
        id="fake",
        provider="fake",
        model="fake-model",
        message=Message(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls),
        usage=ModelUsage(),
    )


# 函数说明：_last_user
# 用途：处理回归测试与测试辅助中的 `_last_user` 数据；结果及边界条件见下方说明。
# 参数：
#   request：待处理的请求对象，类型 `ModelRequest`。
# 返回：类型 `str`；按分支返回 `message.content or ''`；`''`。
# 分支与异常：
#   当 `message.role is MessageRole.USER` 时，返回 `message.content or ''`。
def _last_user(request: ModelRequest) -> str:
    for message in reversed(request.messages):
        if message.role is MessageRole.USER:
            return message.content or ""
    return ""


# 函数说明：_after_tool
# 用途：返回 `bool(request.messages) and request.messages[-1].role is MessageRole.TOOL`
# ，提供 回归测试与测试辅助 的派生值。
# 参数：
#   request：待处理的请求对象，类型 `ModelRequest`。
# 返回：类型 `bool`；返回
# `bool(request.messages) and request.messages[-1].role is MessageRole.TOOL`。
def _after_tool(request: ModelRequest) -> bool:
    return bool(request.messages) and request.messages[-1].role is MessageRole.TOOL


# 函数说明：_registry
# 用途：在回归测试与测试辅助中处理 `_registry`，通过 `registry.register` 完成首个内部处
# 理步骤。
# 参数：
#   reply：用户回复或工具响应，类型 `Callable[[ModelRequest], ModelResponse]`。
# 返回：类型 `ModelAdapterRegistry`；返回 `registry`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `ModelAdapterRegistry` → `ModelSettings` → `registry.register`。
def _registry(reply: Callable[[ModelRequest], ModelResponse]) -> ModelAdapterRegistry:
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: ScriptedAdapter(config, reply), config=config)
    return registry


# 函数说明：test_full_mea_over_real_run_manager
# 用途：回归验证回归测试与测试辅助中的 `full_mea_over_real_run_manager` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `(workspace / 'users.csv').write_text` → `SQLiteRunStore` → `SQLiteCheckpointStore` →
# `SQLiteConversationStore` → `SQLiteMeaStore`；另有 21 个调用点。
# 分支与异常：
#   验证条件：`mea.status is MeaStatus.COMPLETED`。
#   验证条件：`final_task is not None and final_task.status is TaskStatus.COMPLETED`。
#   验证条件：`{step.id: step.status for step in final_task.steps} == {'s1':
# TaskStepStatus.DONE, 's2': TaskStepStatus.DONE}`。
#   验证条件：`write_tool.calls == ['s1.txt']`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / 'users.csv').write_text`。
async def test_full_mea_over_real_run_manager(tmp_path) -> None:
    database = tmp_path / "muharness.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "users.csv").write_text("id,name\n1,a\n", encoding="utf-8")

    run_store = SQLiteRunStore(database)
    checkpoints = SQLiteCheckpointStore(database)
    conversations = SQLiteConversationStore(database)
    mea_store = SQLiteMeaStore(database)
    for store in (run_store, checkpoints, conversations, mea_store):
        await store.initialize()
    tasks = FileTaskStore(tmp_path / "tasks")
    await tasks.initialize()
    conversation = await conversations.create(title="长任务")

    manager_outputs = [
        manager(execute("s1"), state="- 已完成: 无"),
        manager(execute("s2")),
        manager(FINAL),
    ]

    # 函数说明：test_full_mea_over_real_run_manager.manage
    # 用途：管理回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；按分支返回 `_response('长任务已完成：CSV 已导入。')`；
    # `_response(manager_outputs.pop(0))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_last_user(request).startswith` →
    #  `_last_user` → `_response` → `manager_outputs.pop`。
    # 分支与异常：
    #   当 `_last_user(request).startswith(FINAL_RESPONSE_INSTRUCTIONS)` 时，返回
    # `_response('长任务已完成：CSV 已导入。')`。
    # 闭包依赖：从外层读取 `manager_outputs`。
    def manage(request: ModelRequest) -> ModelResponse:
        if _last_user(request).startswith(FINAL_RESPONSE_INSTRUCTIONS):
            return _response("长任务已完成：CSV 已导入。")
        return _response(manager_outputs.pop(0))

    # 函数说明：test_full_mea_over_real_run_manager.run_executor
    # 用途：运行执行者，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；按分支返回
    # `_response('已写入 s1.txt，读到表头 id,name。')`；`_response(tool_calls=(call,))`
    # ；`_response('已导入 1 行。')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_after_tool` → `_response` →
    # `(workspace / 's1.txt').exists` → `ToolCall`。
    # 分支与异常：
    #   当 `_after_tool(request)` 时，返回
    # `_response('已写入 s1.txt，读到表头 id,name。')`。
    #   `not (workspace / 's1.txt').exists()` 分支在完成前置处理后返回
    # `_response(tool_calls=(call,))`。
    # 闭包依赖：从外层读取 `workspace`。
    def run_executor(request: ModelRequest) -> ModelResponse:
        if _after_tool(request):
            return _response("已写入 s1.txt，读到表头 id,name。")
        if not (workspace / "s1.txt").exists():
            call = ToolCall(id="exec-w", name="write_file", arguments={"path": "s1.txt", "content": "x"})
            return _response(tool_calls=(call,))
        return _response("已导入 1 行。")

    # 函数说明：test_full_mea_over_real_run_manager.run_auditor
    # 用途：运行审计者，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；按分支返回 `_response(report())`；
    # `_response(tool_calls=(call,))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_after_tool` → `_response` →
    # `report` → `ToolCall`。
    # 分支与异常：
    #   当 `_after_tool(request)` 时，返回 `_response(report())`。
    def run_auditor(request: ModelRequest) -> ModelResponse:
        if _after_tool(request):
            text = report()
            if "- 最终验收。" in _last_user(request):
                text = text.replace("步骤验收: satisfied", "步骤验收: not_applicable")
            return _response(text)
        call = ToolCall(id="audit-w", name="write_file", arguments={"path": "audit.txt", "content": "x"})
        return _response(tool_calls=(call,))

    tools = ToolRegistry()
    write_tool = FakeWriteTool(workspace)
    tools.register(write_tool)

    # 函数说明：test_full_mea_over_real_run_manager.role_runtime
    # 用途：返回 `AgentRuntime(_registry(reply), tools, provider='fake',
    # checkpoint_store=checkpoints)`，提供 回归测试与测试辅助 的派生值。
    # 参数：
    #   reply：用户回复或工具响应，类型 `Callable[[ModelRequest], ModelResponse]`。
    # 返回：类型 `AgentRuntime`；返回 `AgentRuntime(_registry(reply), tools, provider='
    # fake', checkpoint_store=checkpoints)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentRuntime` → `_registry`。
    # 闭包依赖：从外层读取 `checkpoints`、`tools`。
    def role_runtime(reply: Callable[[ModelRequest], ModelResponse]) -> AgentRuntime:
        return AgentRuntime(_registry(reply), tools, provider="fake", checkpoint_store=checkpoints)

    runtimes = {
        AgentMode.MANAGE: role_runtime(manage),
        AgentMode.EXECUTE: role_runtime(run_executor),
        AgentMode.AUDIT: role_runtime(run_auditor),
    }
    runs = RunManager(run_store, checkpoints, runtimes[AgentMode.MANAGE])

    # 函数说明：test_full_mea_over_real_run_manager.sink
    # 用途：在回归测试与测试辅助中处理 `sink`，通过 `conversations.append_messages` 完成
    # 首个内部处理步骤。
    # 参数：
    #   conversation_id：目标会话标识，类型 `str`。
    #   text：待处理的文本，类型 `str`。
    #   key：字段名或查询键，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`conversations.append_messages` →
    # `Message`。
    # 闭包依赖：从外层读取 `conversations`。
    async def sink(conversation_id: str, text: str, key: str) -> None:
        await conversations.append_messages(
            conversation_id,
            (Message(role=MessageRole.ASSISTANT, content=text),),
            idempotency_key=key,
        )

    runner = MeaRunner(
        store=mea_store,
        tasks=tasks,
        runs=runs,
        runtimes=runtimes,
        workspace_root=workspace,
        final_message_sink=sink,
    )

    task = await tasks.create(
        owner_conversation_id=conversation.id,
        title="导入用户",
        goal="users 表包含 CSV 全部数据",
        steps=(
            TaskStep(id="s1", title="读取 CSV", acceptance="读到表头和数据"),
            TaskStep(id="s2", title="导入数据库", acceptance="users 表行数正确"),
        ),
    )
    task = await tasks.set_status(task.id, TaskStatus.ACTIVE)
    mea = await runner.start(
        task_id=task.id,
        conversation_id=conversation.id,
        original_request="把 users.csv 导入数据库",
        spawn=False,
    )
    mea = await runner.run(mea.id)

    assert mea.status is MeaStatus.COMPLETED, mea.abort_reason
    final_task = await tasks.get(task.id)
    assert final_task is not None and final_task.status is TaskStatus.COMPLETED
    assert {step.id: step.status for step in final_task.steps} == {
        "s1": TaskStepStatus.DONE,
        "s2": TaskStepStatus.DONE,
    }

    # executor 的写入真的执行了；auditor 的写入在执行前就被角色边界拒绝
    assert write_tool.calls == ["s1.txt"]
    assert not (workspace / "audit.txt").exists()

    rounds = await mea_store.rounds(mea.id)
    assert [rnd.kind for rnd in rounds] == [RoundKind.NORMAL, RoundKind.NORMAL, RoundKind.FINAL_AUDIT]
    assert all(rnd.phase is RoundPhase.APPLIED for rnd in rounds)
    assert rounds[-1].executor_run_id is None

    # 预分配的子 Run ID 就是 runs 表的主键，来源和模式都记下了
    expected: dict[str, AgentMode] = {}
    for rnd in rounds:
        expected[rnd.manager_run_id] = AgentMode.MANAGE
        expected[rnd.auditor_run_id] = AgentMode.AUDIT
        if rnd.executor_run_id:
            expected[rnd.executor_run_id] = AgentMode.EXECUTE
    for run_id, mode in expected.items():
        stored = await run_store.get(run_id)
        assert stored is not None, run_id
        assert stored.status is RunStatus.COMPLETED
        assert stored.mode is mode
        assert stored.source == f"mea:{ROLE_NAMES[mode]}" and stored.source_id == mea.id
        assert stored.conversation_id == conversation.id

    messages = await conversations.load_messages(conversation.id)
    assert [m.content for m in messages] == ["长任务已完成：CSV 已导入。"]

    # 重放终结（同一个幂等键）不会重复追加
    await sink(conversation.id, "长任务已完成：CSV 已导入。", f"{mea.id}/final")
    assert len(await conversations.load_messages(conversation.id)) == 1


@pytest.mark.parametrize(
    "scenario",
    ["scope_error", "audit_recheck", "split_batch", "existing_b",
     "stale_plan_description", "user_plan_only"],
)
async def test_three_round_file_task_with_real_tools_and_persisted_audit_evidence(
    tmp_path, scenario
):
    database = tmp_path / "m.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    old_description = "当前为计划模式，仅保存执行计划，不修改文件。"
    has_old_description = scenario in ("stale_plan_description", "user_plan_only")
    if scenario == "existing_b":
        (workspace / "demo").mkdir()
        (workspace / "demo/b.txt").write_bytes(b"previous run")
    run_store = SQLiteRunStore(database)
    checkpoints = SQLiteCheckpointStore(database)
    traces = SQLiteTraceStore(database)
    mea_store = SQLiteMeaStore(database)
    for store in (run_store, checkpoints, traces, mea_store):
        await store.initialize()
    tasks = FileTaskStore(tmp_path / "tasks")
    await tasks.initialize()
    tools = ToolRegistry()
    tools.register(WriteFileTool(workspace))
    tools.register(ReadFileTool(workspace))
    routes = [execute("s1"), execute("s2")]
    if scenario == "split_batch":
        routes.append("下一步: 阻塞")
    else:
        routes.append(execute("s3"))
        if scenario == "audit_recheck":
            routes.append(
                "下一步: 仅审计\n步骤: s3\n核实重点: 核对原运行回复和调用清单"
            )
        routes.append(FINAL)
    repairs = []

    def manage(request):
        prompt = _last_user(request)
        if prompt.startswith(FINAL_RESPONSE_INSTRUCTIONS):
            return _response("demo/a.txt 的内容是 v2")
        if prompt.startswith("上一份 auditor"):
            repairs.append(prompt)
            return _response(report())
        assert "计划已被接受，当前是长任务执行阶段" in prompt
        if has_old_description:
            assert old_description in prompt
        if scenario == "user_plan_only":
            assert "[原始请求]\n只生成计划，不执行或修改文件。" in prompt
            assert "仍须遵守；有真实冲突时请示用户" in prompt
            return _response(manager("下一步: 请示用户\n问题: 是否授权执行？"))
        assert "不得因已有文件追加删除或清理前置条件" in prompt
        return _response(manager(routes.pop(0)))

    def run_executor(request):
        prompt = _last_user(request)
        assert "计划已被接受，当前是长任务执行阶段" in prompt
        if has_old_description:
            assert old_description in prompt
        step = re.search(r"所属步骤[^\n]*\n(s\d)", prompt).group(1)
        results = [m for m in request.messages if m.role is MessageRole.TOOL]
        if results:
            if step == "s2" and scenario == "split_batch" and len(results) == 1:
                return _response(
                    tool_calls=(
                        ToolCall(
                            id="b",
                            name="write_file",
                            arguments={"path": "demo/b.txt", "content": "hello"},
                        ),
                    )
                )
            return _response("v2" if step == "s3" else "写入成功")
        calls = {
            "s1": (
                ToolCall(
                    id="v1",
                    name="write_file",
                    arguments={"path": "demo/a.txt", "content": "v1"},
                ),
            ),
            "s2": (
                ToolCall(
                    id="v2",
                    name="write_file",
                    arguments={"path": "demo/a.txt", "content": "v2"},
                ),
                ToolCall(
                    id="b",
                    name="write_file",
                    arguments={"path": "demo/b.txt", "content": "hello"},
                ),
            ),
            "s3": (
                ToolCall(id="read", name="read_file", arguments={"path": "demo/a.txt"}),
            ),
        }[step]
        if step == "s2" and scenario == "split_batch":
            calls = calls[:1]
        return _response(tool_calls=calls)

    audited = []

    def run_auditor(request):
        prompt = _last_user(request)
        assert "计划已被接受，当前是长任务执行阶段" in prompt
        if has_old_description:
            assert old_description in prompt
        records = [
            json.loads(line)
            for line in prompt.splitlines()
            if line.startswith('{"run_id":')
        ]
        assert records and all(record["call_list_complete"] for record in records)
        if "- 最终验收。" in prompt:
            assert len(records) == 3
            return _response(
                report().replace("步骤验收: satisfied", "步骤验收: not_applicable")
            )
        step = re.search(r"所属步骤: (s\d)", prompt).group(1)
        record = records[-1]
        calls = record["calls"]
        audited.append((step, record))
        if step == "s1":
            assert "当前存在某文件，不能证明本轮提前创建了它" in prompt
            assert "明确要求初始不存在时仍须核验" in prompt
            assert len(calls) == 1
            assert calls[0]["arguments"] == {"path": "demo/a.txt", "content": "v1"}
            if scenario == "existing_b":
                assert (workspace / "demo/b.txt").read_bytes() == b"previous run"
        elif step == "s2":
            assert len(calls) == 2
            assert (workspace / "demo/b.txt").read_text() == "hello"
            if calls[0]["model_step"] != calls[1]["model_step"]:
                return _response(
                    report().replace("步骤验收: satisfied", "步骤验收: not_satisfied")
                )
        else:
            assert len(calls) == 1 and calls[0]["tool_name"] == "read_file"
            assert calls[0]["output_excerpt"] == record["final_reply"] == "v2"
            if scenario == "audit_recheck" and sum(s == "s3" for s, _ in audited) == 1:
                return _response(
                    report().replace("步骤验收: satisfied", "步骤验收: not_satisfied")
                )
            if scenario == "scope_error":
                return _response(
                    report().replace("步骤验收: satisfied", "步骤验收: not_applicable")
                )
        return _response(report())

    runtimes = {
        mode: AgentRuntime(
            _registry(reply), tools, provider="fake", checkpoint_store=checkpoints
        )
        for mode, reply in [
            (AgentMode.MANAGE, manage),
            (AgentMode.EXECUTE, run_executor),
            (AgentMode.AUDIT, run_auditor),
        ]
    }
    runs = RunManager(run_store, checkpoints, runtimes[AgentMode.MANAGE])
    runner = MeaRunner(
        store=mea_store,
        tasks=tasks,
        runs=MeaRunGateway(
            runs, trace_handler_factory=lambda: SQLiteTraceEventHandler(traces)
        ),
        runtimes=runtimes,
        workspace_root=workspace,
        executor_evidence=ExecutorEvidenceProvider(traces, run_store),
    )
    task = await tasks.create(
        owner_conversation_id="conv",
        title="三轮文件测试",
        goal="分三轮写入并读取",
        description=old_description if has_old_description else None,
        steps=tuple(
            TaskStep(id=f"s{i}", title=f"第{i}轮", acceptance=acceptance)
            for i, acceptance in enumerate(
                [
                    "write_file a=v1",
                    "同一次模型响应两次 write_file：a=v2，b=hello",
                    "read_file a 并回复 v2，不写文件",
                ],
                start=1,
            )
        ),
    )
    await tasks.plan_accept(task.id)
    mea = await runner.start(
        task_id=task.id,
        conversation_id="conv",
        spawn=False,
        original_request=(
            "只生成计划，不执行或修改文件。" if scenario == "user_plan_only"
            else "严格分三轮：写 a=v1；同轮写 a=v2 和 b=hello；读取 a 并回复"
        ),
    )
    mea = await runner.run(mea.id)
    rounds = await mea_store.rounds(mea.id)
    final_task = await tasks.get(task.id)
    if scenario == "user_plan_only":
        assert mea.status is MeaStatus.WAITING_USER
        assert not (workspace / "demo").exists()
        assert all(r.executor_run_id is None for r in rounds)
        return
    assert (workspace / "demo/a.txt").read_bytes() == b"v2"
    assert (workspace / "demo/b.txt").read_bytes() == b"hello"
    if scenario == "split_batch":
        assert mea.status is MeaStatus.BLOCKED
        assert final_task.steps[1].status is TaskStepStatus.TODO
        assert final_task.steps[2].status is TaskStepStatus.TODO
    else:
        assert mea.status is MeaStatus.COMPLETED, mea.abort_reason
        assert all(step.status is TaskStepStatus.DONE for step in final_task.steps)
        assert sum(bool(r.executor_run_id) for r in rounds) == 3
        assert len(repairs) == (1 if scenario == "scope_error" else 0)
        s3_records = [record for step, record in audited if step == "s3"]
        assert len({record["run_id"] for record in s3_records}) == 1
