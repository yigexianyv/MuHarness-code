"""进程在 Executor 运行中退出、重启后恢复的端到端测试（真实 RunManager、检查点、Trace）。

第一个“进程”里，fake Executor 先向 log.txt 追加一行，再发起一条永远不返回的 shell 调用，
这时进程“退出”（旧对象全部丢弃，不写任何状态）。第二个进程在同一个数据库上启动：
RunManager 对账把 Executor Run 标为 interrupted，长任务从 running 改为 paused；
用户点继续后不会再启动这个 Executor，而是进入核查审计，审计输入里有中断前的两条调用。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from app.runtime.mea import (
    MeaRunGateway,
    MeaRunner,
    RecoveryInfoProvider,
    SQLiteMeaStore,
)
from app.runtime.mea.models import MeaStatus, RoundKind
from app.runtime.mea.prompts import FINAL_RESPONSE_INSTRUCTIONS
from app.runtime.run import RunManager, RunStatus, SQLiteRunStore
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry


# 函数说明：_manager
# 用途：返回 `f'当前任务状态:\n- 已完成: 见审计\n\n任务契约:\n- 目标状态: log.txt 恰好一
# 行\n\n步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推…`，提供 回归测试与测试辅助
# 的派生值。
# 参数：
#   route：路由输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `f'当前任务状态:\n- 已完成: 见审计\n\n任务契约:\n- 目标状态:
# log.txt 恰好一行\n\n步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推…`。
def _manager(route: str) -> str:
    return (
        "当前任务状态:\n- 已完成: 见审计\n\n任务契约:\n- 目标状态: log.txt 恰好一行\n\n"
        f"步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推进\n\n{route}"
    )


# 函数说明：_execute
# 用途：执行回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   step：当前任务步骤，类型 `str`。
# 返回：类型 `str`；返回 `f'下一步: 执行任务\n步骤: {step}\n任务: 完成 {step}\n验收标准:
#  见步骤验收\n相关审计报告: 无\n相关已审计状态: 无\n边界: 只改…`。
def _execute(step: str) -> str:
    return (
        f"下一步: 执行任务\n步骤: {step}\n任务: 完成 {step}\n验收标准: 见步骤验收\n"
        "相关审计报告: 无\n相关已审计状态: 无\n边界: 只改 log.txt"
    )


REPORT = (
    "状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n"
    "审计事实: log.txt 只有一行。\n\n验收约束反查:\n契约结论: aligned\n阻断约束: 无\n"
    "范围外约束: 无\n给任务管理器的状态更新: 见上"
)


class ScriptedAdapter(ModelAdapter):
    # 函数说明：ScriptedAdapter.__init__
    # 用途：初始化 ScriptedAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   reply：用户回复或工具响应，类型 `Callable[[ModelRequest], ModelResponse]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self._reply`。
    def __init__(self, config: ProviderConfig, reply: Callable[[ModelRequest], ModelResponse]) -> None:
        super().__init__(config)
        self._reply = reply

    # 函数说明：ScriptedAdapter.complete
    # 用途：完成ScriptedAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `self._reply(request)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._reply`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self._reply(request)

    # 函数说明：ScriptedAdapter.close
    # 用途：关闭ScriptedAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str | None`；默认 `None`。
#   call：调用输入或配置值，类型 `ToolCall | None`；默认 `None`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def _response(content: str | None = None, *, call: ToolCall | None = None) -> ModelResponse:
    return ModelResponse(
        id="fake",
        provider="fake",
        model="fake-model",
        message=Message(
            role=MessageRole.ASSISTANT, content=content, tool_calls=(call,) if call else ()
        ),
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


class AppendTool(BaseTool):
    """替身 write_file：向文件追加一行，用来检查恢复后有没有重复执行。"""

    definition = ToolDefinition(
        name="write_file",
        description="append a line",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )

    # 函数说明：AppendTool.__init__
    # 用途：初始化 AppendTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   root：当前操作的根目录，类型 `Path`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.root`。
    def __init__(self, root: Path) -> None:
        self.root = root

    # 函数说明：AppendTool.execute
    # 用途：执行AppendTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `path`
    # 。
    # 返回：类型 `str`；返回 `'appended'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `(self.root / str(arguments['path'])).open` → `handle.write`。
    # 副作用与资源：
    #   文件或资源访问：`(self.root / str(arguments['path'])).open`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        with (self.root / str(arguments["path"])).open("a", encoding="utf-8") as handle:
            handle.write("line\n")
        return "appended"


class HangingShell(BaseTool):
    definition = ToolDefinition(
        name="run_shell_command",
        description="never returns",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
    )

    # 函数说明：HangingShell.__init__
    # 用途：初始化 HangingShell；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.Event`。
    # 副作用与资源：
    #   更新对象字段：`self.started`。
    def __init__(self) -> None:
        self.started = asyncio.Event()

    # 函数说明：HangingShell.execute
    # 用途：执行HangingShell，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `str`；返回 `'unreachable'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.started.set` →
    # `asyncio.Event().wait` → `asyncio.Event`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        self.started.set()
        await asyncio.Event().wait()
        return "unreachable"


@dataclass
class Models:
    """跨两个“进程”共享的模型脚本和调用记录。"""

    manager: list[str]
    executor_requests: list[ModelRequest] = field(default_factory=list)
    auditor_prompts: list[str] = field(default_factory=list)

    # 函数说明：Models.manage
    # 用途：管理Models，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；按分支返回 `_response('完成：log.txt 只有一行。')`；
    # `_response(self.manager.pop(0))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_last_user(request).startswith` →
    #  `_last_user` → `_response` → `self.manager.pop`。
    # 分支与异常：
    #   当 `_last_user(request).startswith(FINAL_RESPONSE_INSTRUCTIONS)` 时，返回
    # `_response('完成：log.txt 只有一行。')`。
    def manage(self, request: ModelRequest) -> ModelResponse:
        if _last_user(request).startswith(FINAL_RESPONSE_INSTRUCTIONS):
            return _response("完成：log.txt 只有一行。")
        return _response(self.manager.pop(0))

    # 函数说明：Models.execute
    # 用途：执行Models，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；按分支返回 `_response(call=ToolCall(id='append-1',
    # name='write_file', arguments={'path': 'log.txt'}))`；`_response(…)`；
    # `_response('s2 已完成。')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_response` → `ToolCall`。
    # 分支与异常：
    #   当 `count == 1` 时，返回 `_response(…)`。
    #   当 `count == 2` 时，返回 `_response(…)`。
    def execute(self, request: ModelRequest) -> ModelResponse:
        self.executor_requests.append(request)
        count = len(self.executor_requests)
        if count == 1:
            return _response(call=ToolCall(id="append-1", name="write_file",
                                           arguments={"path": "log.txt"}))
        if count == 2:
            return _response(call=ToolCall(id="shell-1", name="run_shell_command",
                                           arguments={"command": "python import.py"}))
        return _response("s2 已完成。")

    # 函数说明：Models.audit
    # 用途：审计Models，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `_response(REPORT)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_last_user` → `_response`。
    def audit(self, request: ModelRequest) -> ModelResponse:
        self.auditor_prompts.append(_last_user(request))
        text = REPORT
        if "- 最终验收。" in _last_user(request):
            text = text.replace("步骤验收: satisfied", "步骤验收: not_applicable")
        return _response(text)


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


@dataclass
class Process:
    runs: RunManager
    runner: MeaRunner
    mea_store: SQLiteMeaStore
    tasks: FileTaskStore
    conversations: SQLiteConversationStore
    run_store: SQLiteRunStore
    shell: HangingShell


# 函数说明：_boot
# 用途：在回归测试与测试辅助中处理 `_boot`，通过 `store.initialize` 完成首个内部处理步骤
# 。
# 参数：
#   root：当前操作的根目录，类型 `Path`。
#   workspace：目标工作区，类型 `Path`。
#   models：模型输入或配置值，类型 `Models`。
# 返回：类型 `Process`；返回
# `Process(runs, runner, mea_store, tasks, conversations, run_store, shell)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteRunStore` →
# `SQLiteCheckpointStore` → `SQLiteTraceStore` → `SQLiteConversationStore` →
# `SQLiteMeaStore` → `store.initialize`；另有 14 个调用点。
async def _boot(root: Path, workspace: Path, models: Models) -> Process:
    database = root / "muharness.db"
    run_store = SQLiteRunStore(database)
    checkpoints = SQLiteCheckpointStore(database)
    traces = SQLiteTraceStore(database)
    conversations = SQLiteConversationStore(database)
    mea_store = SQLiteMeaStore(database)
    for store in (checkpoints, traces, conversations, mea_store):
        await store.initialize()
    tasks = FileTaskStore(root / "tasks")
    await tasks.initialize()

    tools = ToolRegistry()
    tools.register(AppendTool(workspace))
    shell = HangingShell()
    tools.register(shell)

    # 函数说明：_boot.runtime
    # 用途：返回 `AgentRuntime(_registry(reply), tools, provider='fake',
    # checkpoint_store=checkpoints)`，提供 回归测试与测试辅助 的派生值。
    # 参数：
    #   reply：用户回复或工具响应，类型 `Callable[[ModelRequest], ModelResponse]`。
    # 返回：类型 `AgentRuntime`；返回 `AgentRuntime(_registry(reply), tools, provider='
    # fake', checkpoint_store=checkpoints)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentRuntime` → `_registry`。
    # 闭包依赖：从外层读取 `checkpoints`、`tools`。
    def runtime(reply: Callable[[ModelRequest], ModelResponse]) -> AgentRuntime:
        return AgentRuntime(_registry(reply), tools, provider="fake", checkpoint_store=checkpoints)

    runtimes = {
        AgentMode.MANAGE: runtime(models.manage),
        AgentMode.EXECUTE: runtime(models.execute),
        AgentMode.AUDIT: runtime(models.audit),
    }
    runs = RunManager(run_store, checkpoints, runtimes[AgentMode.MANAGE])
    await runs.initialize()  # 对账：上个进程留下的 running Run

    # 函数说明：_boot.sink
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
            conversation_id, (Message(role=MessageRole.ASSISTANT, content=text),),
            idempotency_key=key,
        )

    runner = MeaRunner(
        store=mea_store,
        tasks=tasks,
        runs=MeaRunGateway(runs, trace_handler_factory=lambda: SQLiteTraceEventHandler(traces)),
        runtimes=runtimes,
        workspace_root=workspace,
        final_message_sink=sink,
        recovery_info=RecoveryInfoProvider(checkpoints, traces),
    )
    await runner.reconcile()
    return Process(runs, runner, mea_store, tasks, conversations, run_store, shell)


# 函数说明：test_restart_during_executor_audits_instead_of_rerunning
# 用途：回归验证回归测试与测试辅助中的
# `restart_during_executor_audits_instead_of_rerunning` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` → `Models` →
# `_manager` → `_execute` → `_boot` → `first.conversations.create`；另有 16 个调用点。
# 分支与异常：
#   验证条件：`executor_run_id is not None`。
#   验证条件：`stored_run is not None and stored_run.status is RunStatus.INTERRUPTED`。
#   验证条件：`(await second.mea_store.require(mea.id)).status is MeaStatus.PAUSED`。
#   验证条件：`final.status is MeaStatus.COMPLETED`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / 'log.txt').read_text`。
async def test_restart_during_executor_audits_instead_of_rerunning(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    models = Models(manager=[_manager(_execute("s1")), _manager(_execute("s2")),
                             _manager("下一步: 最终验收\n验收重点: 行数")])

    first = await _boot(tmp_path, workspace, models)
    conversation = await first.conversations.create(title="长任务")
    task = await first.tasks.create(
        owner_conversation_id=conversation.id,
        title="写日志",
        goal="log.txt 恰好一行",
        steps=(
            TaskStep(id="s1", title="追加一行", acceptance="log.txt 恰好一行"),
            TaskStep(id="s2", title="收尾", acceptance="没有多余文件"),
        ),
    )
    task = await first.tasks.set_status(task.id, TaskStatus.ACTIVE)
    mea = await first.runner.start(
        task_id=task.id, conversation_id=conversation.id, original_request="往 log.txt 写一行"
    )
    try:
        await asyncio.wait_for(first.shell.started.wait(), 10)
        # —— 进程在这里退出：第一个进程的对象全部丢弃，不再写任何状态 ——
        executor_run_id = (await first.mea_store.rounds(mea.id))[0].executor_run_id
        assert executor_run_id is not None

        second = await _boot(tmp_path, workspace, models)
        stored_run = await second.run_store.get(executor_run_id)
        assert stored_run is not None and stored_run.status is RunStatus.INTERRUPTED
        assert (await second.mea_store.require(mea.id)).status is MeaStatus.PAUSED

        await second.runner.resume(mea.id)
        final = await asyncio.wait_for(second.runner.wait(mea.id), 20)
    finally:
        await first.runner.shutdown()
        for handle in list(first.runs._active_tasks.values()):
            handle.cancel()
        await asyncio.sleep(0)

    assert final.status is MeaStatus.COMPLETED, final.abort_reason
    assert (workspace / "log.txt").read_text(encoding="utf-8") == "line\n"
    assert len(models.executor_requests) == 3  # 第 1 轮的 Executor 没有被重新启动

    rounds = await second.mea_store.rounds(mea.id)
    assert rounds[0].interrupt_reason == "进程重启时 Executor 没有正常结束", [(r.index, r.kind, r.phase, r.interrupt_reason) for r in rounds]
    assert rounds[1].kind is RoundKind.RECOVERY_AUDIT
    recovery_prompt = models.auditor_prompts[0]
    assert "write_file" in recovery_prompt and "log.txt" in recovery_prompt
    assert "run_shell_command" in recovery_prompt and "python import.py" in recovery_prompt

    done = await second.tasks.get(task.id)
    assert done is not None and done.status is TaskStatus.COMPLETED
    assert [step.status for step in done.steps] == [TaskStepStatus.DONE, TaskStepStatus.DONE]
    messages = await second.conversations.load_messages(conversation.id)
    assert [m.content for m in messages] == ["完成：log.txt 只有一行。"]
