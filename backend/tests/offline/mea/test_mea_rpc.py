"""mea.* RPC 方法，以及长任务推进期间对普通对话和普通恢复的拦截。

RPC 层只做参数校验和调用 MeaRunner；这里用一个不启动循环的 MeaRunner，
推进逻辑由 test_mea_runner.py 覆盖。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.domain.conversation.store import SQLiteConversationStore
from app.domain.task import FileTaskStore, TaskStatus, TaskStep
from app.models.types import AgentMode, ToolDefinition
from app.runtime.mea import MeaRunner, MeaStatus, SQLiteMeaStore
from app.server.rpc.dispatcher import RpcContext
from app.server.rpc.methods import conversations as conversations_rpc
from app.server.rpc.methods import mea as mea_rpc
from app.server.rpc.methods import runs as runs_rpc
from app.server.rpc.protocol import (
    INVALID_STATE,
    RESOURCE_NOT_FOUND,
    JsonRpcError,
    RpcErrorCode,
)
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry


class NoLoopRunner(MeaRunner):
    """真实的 MeaRunner，只是 spawn 不启动循环。"""

    # 函数说明：NoLoopRunner.__init__
    # 用途：初始化 NoLoopRunner；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.spawned`。
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.spawned: list[str] = []

    # 函数说明：NoLoopRunner.spawn
    # 用途：启动NoLoopRunner，供回归测试与测试辅助使用。
    # 参数：
    #   mea_id：长任务协作记录标识，类型 `str`。
    # 返回：无结果值，显式返回 None。
    def spawn(self, mea_id: str):  # type: ignore[override]
        self.spawned.append(mea_id)
        return None


class NamedTool(BaseTool):
    # 函数说明：NamedTool.__init__
    # 用途：初始化 NamedTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    # 副作用与资源：
    #   更新对象字段：`self._definition`。
    def __init__(self, name: str) -> None:
        self._definition = ToolDefinition(
            name=name, description=name, parameters={"type": "object", "properties": {}}
        )

    # 函数说明：NamedTool.definition
    # 用途：提供 NamedTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `self._definition`。
    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    # 函数说明：NamedTool.execute
    # 用途：执行NamedTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `str`；返回 `'ok'`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        return "ok"


class FakeRunStore:
    # 函数说明：FakeRunStore.__init__
    # 用途：初始化 FakeRunStore；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.runs`。
    def __init__(self) -> None:
        self.runs: dict[str, Any] = {}

    # 函数说明：FakeRunStore.get
    # 用途：获取FakeRunStore，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Any`；返回 `self.runs.get(run_id)`。
    async def get(self, run_id: str) -> Any:
        return self.runs.get(run_id)


class FakeRunManager:
    # 函数说明：FakeRunManager.__init__
    # 用途：初始化 FakeRunManager；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   runs：运行集合输入或配置值，类型 `dict[str, Any]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._runs`。
    def __init__(self, runs: dict[str, Any]) -> None:
        self._runs = runs

    # 函数说明：FakeRunManager.get_run
    # 用途：获取运行，供回归测试与测试辅助使用。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `Any`；返回 `self._runs.get(run_id)`。
    async def get_run(self, run_id: str) -> Any:
        return self._runs.get(run_id)


# 函数说明：_app
# 用途：在回归测试与测试辅助中处理 `_app`，通过 `mea_store.initialize` 完成首个内部处理
# 步骤。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `SimpleNamespace`；返回 `SimpleNamespace(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteMeaStore` →
# `mea_store.initialize` → `SQLiteConversationStore` → `conversations.initialize` →
# `FileTaskStore` → `tasks.initialize`；另有 9 个调用点。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`。
async def _app(tmp_path) -> SimpleNamespace:
    database = tmp_path / "m.db"
    mea_store = SQLiteMeaStore(database)
    await mea_store.initialize()
    conversations = SQLiteConversationStore(database)
    await conversations.initialize()
    tasks = FileTaskStore(tmp_path / "tasks")
    await tasks.initialize()
    tools = ToolRegistry()
    for name in ("read_file", "web_search", "memory_create"):
        tools.register(NamedTool(name))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = NoLoopRunner(
        store=mea_store,
        tasks=tasks,
        runs=SimpleNamespace(active_run_ids=()),
        runtimes={AgentMode.MANAGE: object(), AgentMode.EXECUTE: object(), AgentMode.AUDIT: object()},
        workspace_root=workspace,
    )
    run_store = FakeRunStore()
    return SimpleNamespace(
        mea_store=mea_store,
        mea_runner=runner,
        task_store=tasks,
        tool_registry=tools,
        run_store=run_store,
        run_manager=FakeRunManager(run_store.runs),
        conversation_store=conversations,
        conversation_service=None,
    )


# 函数说明：_ctx
# 用途：返回 `RpcContext(app, SimpleNamespace())`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   app：应用实例，类型 `SimpleNamespace`。
# 返回：类型 `RpcContext`；返回 `RpcContext(app, SimpleNamespace())`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RpcContext` → `SimpleNamespace`。
def _ctx(app: SimpleNamespace) -> RpcContext:
    return RpcContext(app, SimpleNamespace())


# 函数说明：_plan
# 用途：在回归测试与测试辅助中处理 `_plan`，通过 `app.task_store.create` 完成首个内部处
# 理步骤。
# 参数：
#   app：应用实例，类型 `SimpleNamespace`。
#   conversation_id：目标会话标识，类型 `str`。
#   acceptance：`acceptance`输入或配置值，类型 `bool`；默认 `True`。
# 返回：返回 `task`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`app.task_store.create` → `TaskStep` →
#  `SimpleNamespace`。
async def _plan(app: SimpleNamespace, conversation_id: str, *, acceptance: bool = True):
    task = await app.task_store.create(
        owner_conversation_id=conversation_id,
        title="导入用户",
        goal="users 表包含 CSV 全部数据",
        steps=(TaskStep(id="s1", title="读取 CSV", acceptance="读到表头" if acceptance else None),),
        run_ids=("plan-run",),
    )
    app.run_store.runs["plan-run"] = SimpleNamespace(user_message="把 users.csv 导入数据库")
    return task  # 仍是 pending：计划待确认


# 函数说明：test_start_accepts_pending_plan_and_uses_plan_request
# 用途：回归验证回归测试与测试辅助中的
# `start_accepts_pending_plan_and_uses_plan_request` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` →
# `app.conversation_store.create` → `_plan` → `mea_rpc.mea_start` → `_ctx` →
# `app.mea_store.requirements`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`task.status is TaskStatus.PENDING`。
#   验证条件：`mea.auto_approve_sandbox is True`。
#   验证条件：`result['task'].status is TaskStatus.ACTIVE`。
#   验证条件：`mea.extra_tools == ('web_search',) and mea.round_budget == 25`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_start_accepts_pending_plan_and_uses_plan_request(tmp_path) -> None:
    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id)
    assert task.status is TaskStatus.PENDING

    result = await mea_rpc.mea_start(
        {"conversation_id": conversation.id, "task_id": task.id, "extra_tools": ["web_search"]},
        _ctx(app),
    )

    mea = result["mea"]
    assert mea.auto_approve_sandbox is True  # 默认自动批准沙箱内命令
    assert result["task"].status is TaskStatus.ACTIVE
    assert mea.extra_tools == ("web_search",) and mea.round_budget == 25
    assert app.mea_runner.spawned == [mea.id]
    requirements = await app.mea_store.requirements(mea.id)
    assert requirements.original_request == "把 users.csv 导入数据库"

    detail = await mea_rpc.mea_get({"mea_id": mea.id}, _ctx(app))
    assert detail["requirements"]["revision"] == 1 and detail["rounds"] == ()
    assert detail["task"].id == task.id
    listed = await mea_rpc.mea_list({"conversation_id": conversation.id}, _ctx(app))
    assert [item.id for item in listed["meas"]] == [mea.id]

    with pytest.raises(JsonRpcError) as again:
        await mea_rpc.mea_start({"conversation_id": conversation.id, "task_id": task.id}, _ctx(app))
    assert again.value.code == INVALID_STATE


# 函数说明：test_start_rejects_plan_without_acceptance_before_accepting_it
# 用途：回归验证回归测试与测试辅助中的
# `start_rejects_plan_without_acceptance_before_accepting_it` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` →
# `app.conversation_store.create` → `_plan` → `pytest.raises` → `mea_rpc.mea_start` →
# `_ctx`。
# 分支与异常：
#   验证条件：
# `rejected.value.code == INVALID_STATE and '验收标准' in rejected.value.message`。
#   验证条件：`(await app.task_store.get(task.id)).status is TaskStatus.PENDING`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_start_rejects_plan_without_acceptance_before_accepting_it(tmp_path) -> None:
    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id, acceptance=False)

    with pytest.raises(JsonRpcError) as rejected:
        await mea_rpc.mea_start({"conversation_id": conversation.id, "task_id": task.id}, _ctx(app))

    assert rejected.value.code == INVALID_STATE and "验收标准" in rejected.value.message
    assert (await app.task_store.get(task.id)).status is TaskStatus.PENDING


# 函数说明：test_start_validates_params
# 用途：回归验证回归测试与测试辅助中的 `start_validates_params` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` →
# `app.conversation_store.create` → `_plan` → `pytest.raises` → `mea_rpc.mea_start` →
# `_ctx`。
# 分支与异常：
#   验证条件：`error.value.code == code`。
#   验证条件：`(await app.task_store.get(task.id)).status is TaskStatus.PENDING`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_start_validates_params(tmp_path) -> None:
    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id)
    cases = [
        ({"conversation_id": conversation.id, "task_id": task.id, "extra_tools": ["memory_create"]},
         RpcErrorCode.INVALID_PARAMS),
        ({"conversation_id": conversation.id, "task_id": task.id, "round_budget": 0},
         RpcErrorCode.INVALID_PARAMS),
        ({"conversation_id": conversation.id, "task_id": task.id, "accept_plan": False},
         INVALID_STATE),
        ({"conversation_id": "other", "task_id": task.id}, RESOURCE_NOT_FOUND),
    ]
    for params, code in cases:
        with pytest.raises(JsonRpcError) as error:
            await mea_rpc.mea_start(params, _ctx(app))
        assert error.value.code == code, params
    assert (await app.task_store.get(task.id)).status is TaskStatus.PENDING


# 函数说明：test_note_answer_pause_resume_cancel
# 用途：回归验证回归测试与测试辅助中的 `note_answer_pause_resume_cancel` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` →
# `app.conversation_store.create` → `_plan` → `mea_rpc.mea_start` → `_ctx` →
# `mea_rpc.mea_note`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`persistent == {'accepted': True, 'reason': None, 'revision': 2, '
# amendment_id': 'A1'}`。
#   验证条件：`once['accepted'] and once['mea'].once_notes[0].text == '先看日志'`。
#   验证条件：`paused['mea'].pause_requested is True`。
#   验证条件：
# `resumed['mea'].status is MeaStatus.RUNNING and resumed['mea'].round_budget == 30`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_note_answer_pause_resume_cancel(tmp_path) -> None:
    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id)
    mea = (await mea_rpc.mea_start(
        {"conversation_id": conversation.id, "task_id": task.id}, _ctx(app)
    ))["mea"]
    ctx = _ctx(app)

    persistent = await mea_rpc.mea_note({"mea_id": mea.id, "text": "改用测试库"}, ctx)
    assert persistent == {"accepted": True, "reason": None, "revision": 2, "amendment_id": "A1"}
    once = await mea_rpc.mea_note({"mea_id": mea.id, "text": "先看日志", "kind": "once"}, ctx)
    assert once["accepted"] and once["mea"].once_notes[0].text == "先看日志"
    with pytest.raises(JsonRpcError):
        await mea_rpc.mea_note({"mea_id": mea.id, "text": "x", "kind": "forever"}, ctx)

    paused = await mea_rpc.mea_pause({"mea_id": mea.id}, ctx)
    assert paused["mea"].pause_requested is True
    await app.mea_store.save_run(
        (await app.mea_store.require(mea.id)).model_copy(update={"status": MeaStatus.PAUSED})
    )
    resumed = await mea_rpc.mea_resume({"mea_id": mea.id, "extra_rounds": 5}, ctx)
    assert resumed["mea"].status is MeaStatus.RUNNING and resumed["mea"].round_budget == 30

    cancelled = await mea_rpc.mea_cancel({"mea_id": mea.id}, ctx)
    assert cancelled["mea"].status is MeaStatus.CANCELLED
    late = await mea_rpc.mea_note({"mea_id": mea.id, "text": "还有一件事"}, ctx)
    assert late["accepted"] is False and late["reason"] == "mea_finalized"
    late_once = await mea_rpc.mea_note({"mea_id": mea.id, "text": "还有", "kind": "once"}, ctx)
    assert late_once == {"accepted": False, "reason": "mea_finalized"}
    with pytest.raises(JsonRpcError) as resume_error:
        await mea_rpc.mea_resume({"mea_id": mea.id}, ctx)
    assert resume_error.value.code == INVALID_STATE

    with pytest.raises(JsonRpcError) as missing:
        await mea_rpc.mea_get({"mea_id": "nope"}, ctx)
    assert missing.value.code == RESOURCE_NOT_FOUND


# 函数说明：test_finalizing_cannot_be_cancelled
# 用途：回归验证回归测试与测试辅助中的 `finalizing_cannot_be_cancelled` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` →
# `app.conversation_store.create` → `_plan` → `mea_rpc.mea_start` → `_ctx` →
# `app.mea_store.save_run`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`error.value.code == INVALID_STATE`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_finalizing_cannot_be_cancelled(tmp_path) -> None:
    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id)
    mea = (await mea_rpc.mea_start(
        {"conversation_id": conversation.id, "task_id": task.id}, _ctx(app)
    ))["mea"]
    await app.mea_store.save_run(mea.model_copy(update={"status": MeaStatus.FINALIZING}))

    with pytest.raises(JsonRpcError) as error:
        await mea_rpc.mea_cancel({"mea_id": mea.id}, _ctx(app))
    assert error.value.code == INVALID_STATE


# 函数说明：test_tools_lists_optional_executor_tools
# 用途：回归验证回归测试与测试辅助中的 `tools_lists_optional_executor_tools` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` → `mea_rpc.mea_tools` → `_ctx`
# 。
# 分支与异常：
#   验证条件：`(await mea_rpc.mea_tools({}, _ctx(app)))['tools'] == ('web_search',)`。
async def test_tools_lists_optional_executor_tools(tmp_path) -> None:
    app = await _app(tmp_path)
    assert (await mea_rpc.mea_tools({}, _ctx(app)))["tools"] == ("web_search",)


# 函数说明：test_conversation_send_is_rejected_while_mea_is_running
# 用途：回归验证回归测试与测试辅助中的
# `conversation_send_is_rejected_while_mea_is_running` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` →
# `app.conversation_store.create` → `_plan` → `mea_rpc.mea_start` → `_ctx` →
# `pytest.raises`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`error.value.code == INVALID_STATE`。
#   验证条件：
# `error.value.data == {'reason': 'mea_running', 'mea_id': mea.id, 'status': 'running'}`
# 。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_conversation_send_is_rejected_while_mea_is_running(tmp_path) -> None:
    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id)
    mea = (await mea_rpc.mea_start(
        {"conversation_id": conversation.id, "task_id": task.id}, _ctx(app)
    ))["mea"]

    with pytest.raises(JsonRpcError) as error:
        await conversations_rpc.conversation_send(
            {"conversation_id": conversation.id, "content": "顺便问一下"}, _ctx(app)
        )
    assert error.value.code == INVALID_STATE
    assert error.value.data == {"reason": "mea_running", "mea_id": mea.id, "status": "running"}


# 函数说明：test_run_recover_refuses_mea_child_runs
# 用途：回归验证回归测试与测试辅助中的 `run_recover_refuses_mea_child_runs` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` → `SimpleNamespace` →
# `pytest.raises` → `runs_rpc.run_recover` → `_ctx`。
# 分支与异常：
#   验证条件：`error.value.code == INVALID_STATE`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_run_recover_refuses_mea_child_runs(tmp_path) -> None:
    app = await _app(tmp_path)
    app.run_store.runs["exec-1"] = SimpleNamespace(id="exec-1", source="mea:executor")

    with pytest.raises(JsonRpcError) as error:
        await runs_rpc.run_recover({"run_id": "exec-1"}, _ctx(app))
    assert error.value.code == INVALID_STATE



# 函数说明：test_start_can_keep_manual_approval
# 用途：回归验证回归测试与测试辅助中的 `start_can_keep_manual_approval` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` →
# `app.conversation_store.create` → `_plan` → `mea_rpc.mea_start` → `_ctx` →
# `pytest.raises`。
# 分支与异常：
#   验证条件：`result['mea'].auto_approve_sandbox is False`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_start_can_keep_manual_approval(tmp_path) -> None:
    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id)

    result = await mea_rpc.mea_start(
        {"conversation_id": conversation.id, "task_id": task.id, "auto_approve_sandbox": False},
        _ctx(app),
    )

    assert result["mea"].auto_approve_sandbox is False
    with pytest.raises(JsonRpcError):
        await mea_rpc.mea_start(
            {"conversation_id": conversation.id, "task_id": task.id, "auto_approve_sandbox": "yes"},
            _ctx(app),
        )


# 函数说明：test_list_without_conversation_returns_all
# 用途：回归验证回归测试与测试辅助中的 `list_without_conversation_returns_all` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_app` →
# `app.conversation_store.create` → `_plan` → `mea_rpc.mea_start` → `_ctx` →
# `mea_rpc.mea_list`；另有 1 个调用点。
# 分支与异常：
#   验证条件：
# `{mea.conversation_id for mea in everything['meas']} == {first.id, second.id}`。
#   验证条件：`[mea.conversation_id for mea in only_first['meas']] == [first.id]`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_list_without_conversation_returns_all(tmp_path) -> None:
    app = await _app(tmp_path)
    first = await app.conversation_store.create()
    second = await app.conversation_store.create()
    for conversation in (first, second):
        task = await _plan(app, conversation.id)
        await mea_rpc.mea_start({"conversation_id": conversation.id, "task_id": task.id}, _ctx(app))

    everything = await mea_rpc.mea_list({}, _ctx(app))
    assert {mea.conversation_id for mea in everything["meas"]} == {first.id, second.id}
    only_first = await mea_rpc.mea_list({"conversation_id": first.id}, _ctx(app))
    assert [mea.conversation_id for mea in only_first["meas"]] == [first.id]
    with pytest.raises(JsonRpcError):
        await mea_rpc.mea_list({"conversation_id": ""}, _ctx(app))


@pytest.mark.parametrize("restart", [False, True])
async def test_explicit_execution_preserves_or_resets_progress(tmp_path, restart):
    from app.domain.task import TaskStepStatus

    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id)
    await app.task_store.replace_steps(task.id, (
        TaskStep(id="s1", title="read", acceptance="read file"),
        TaskStep(id="s2", title="write", acceptance="verify file"),
    ))
    await app.task_store.plan_accept(task.id)
    await app.task_store.set_step_status(
        task.id, "s1", TaskStepStatus.DONE, note="verified",
    )
    await app.task_store.set_step_status(
        task.id, "s2", TaskStepStatus.IN_PROGRESS, note="started",
    )
    await app.task_store.add_constraints(task.id, "only workspace")
    await app.task_store.update_state(task.id, "s1 already done")
    before = await app.task_store.get(task.id)
    result = await mea_rpc.mea_start({
        "conversation_id": conversation.id, "task_id": task.id, "restart": restart,
    }, _ctx(app))
    executed = result["task"]
    assert (executed.id != task.id) is restart
    assert result["mea"].task_id == executed.id
    assert executed.status is TaskStatus.ACTIVE
    assert executed.constraints == ("only workspace",)
    assert executed.run_ids == before.run_ids
    assert executed.steps[0].acceptance == "read file"
    assert executed.steps[1].acceptance == "verify file"
    assert await app.task_store.get(task.id) == before
    if restart:
        assert all(s.status is TaskStepStatus.TODO for s in executed.steps)
        assert all(s.note is None for s in executed.steps)
        assert executed.state == () and executed.applied_ops == {}
    else:
        assert executed.steps == before.steps
    requirements = await app.mea_store.requirements(result["mea"].id)
    assert requirements.original_request == "把 users.csv 导入数据库"


async def test_restart_does_not_duplicate_a_running_task(tmp_path):
    import asyncio

    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id)
    params = {"conversation_id": conversation.id, "task_id": task.id, "restart": True}
    results = await asyncio.gather(
        mea_rpc.mea_start(params, _ctx(app)),
        mea_rpc.mea_start(params, _ctx(app)),
        return_exceptions=True,
    )
    assert sum(isinstance(r, dict) for r in results) == 1
    errors = [r for r in results if isinstance(r, JsonRpcError)]
    assert len(errors) == 1 and errors[0].code == INVALID_STATE
    assert len(list(app.task_store.tasks_dir.glob("*.json"))) == 2


async def test_restart_rejects_other_conversation_and_invalid_options(tmp_path):
    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    other = await app.conversation_store.create()
    task = await _plan(app, conversation.id)
    for params, code in [
        (
            {"conversation_id": other.id, "task_id": task.id, "restart": True},
            RESOURCE_NOT_FOUND,
        ),
        (
            {"conversation_id": conversation.id, "task_id": task.id, "restart": "true"},
            RpcErrorCode.INVALID_PARAMS,
        ),
    ]:
        with pytest.raises(JsonRpcError) as error:
            await mea_rpc.mea_start(params, _ctx(app))
        assert error.value.code == code
    assert len(list(app.task_store.tasks_dir.glob("*.json"))) == 1


@pytest.mark.parametrize("status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
async def test_restart_closed_task_preserves_original(tmp_path, status):
    from app.domain.task import TaskStepStatus

    app = await _app(tmp_path)
    conversation = await app.conversation_store.create()
    task = await _plan(app, conversation.id)
    await app.task_store.plan_accept(task.id)
    await app.task_store.set_step_status(
        task.id, "s1", TaskStepStatus.DONE, note="verified",
    )
    before = await app.task_store.set_status(task.id, status)
    result = await mea_rpc.mea_start({
        "conversation_id": conversation.id, "task_id": task.id, "restart": True,
    }, _ctx(app))
    assert result["task"].id != task.id
    assert result["task"].steps[0].status is TaskStepStatus.TODO
    assert await app.task_store.get(task.id) == before
