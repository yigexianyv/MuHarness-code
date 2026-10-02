
from types import SimpleNamespace

import pytest

from app.domain.task import FileTaskStore
from app.server.rpc.dispatcher import RpcContext
from app.server.rpc.methods import tasks as tasks_rpc
from app.server.rpc.protocol import JsonRpcError, RpcErrorCode

pytestmark = pytest.mark.asyncio


# 函数说明：test_task_list_only_returns_current_conversation
# 用途：回归验证回归测试与测试辅助中的 `task_list_only_returns_current_conversation` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` → `store.initialize` →
#  `store.create` → `RpcContext` → `SimpleNamespace` → `tasks_rpc.task_list`。
# 分支与异常：
#   验证条件：`[task.id for task in result['tasks']] == [task_a.id]`。
#   验证条件：
# `all((task.owner_conversation_id == 'conv-a' for task in result['tasks']))`。
async def test_task_list_only_returns_current_conversation(tmp_path) -> None:
    store = FileTaskStore(tmp_path / "tasks")
    await store.initialize()
    task_a = await store.create(title="会话 A 任务", owner_conversation_id="conv-a")
    await store.create(title="会话 B 任务", owner_conversation_id="conv-b")
    ctx = RpcContext(SimpleNamespace(task_store=store), SimpleNamespace())

    result = await tasks_rpc.task_list({"conversation_id": "conv-a"}, ctx)

    assert [task.id for task in result["tasks"]] == [task_a.id]
    assert all(task.owner_conversation_id == "conv-a" for task in result["tasks"])


# 函数说明：test_task_list_requires_conversation_id
# 用途：回归验证回归测试与测试辅助中的 `task_list_requires_conversation_id` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` → `store.initialize` →
#  `RpcContext` → `SimpleNamespace` → `pytest.raises` → `tasks_rpc.task_list`。
# 分支与异常：
#   验证条件：`exc_info.value.code == RpcErrorCode.INVALID_PARAMS`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_task_list_requires_conversation_id(tmp_path) -> None:
    store = FileTaskStore(tmp_path / "tasks")
    await store.initialize()
    ctx = RpcContext(SimpleNamespace(task_store=store), SimpleNamespace())

    with pytest.raises(JsonRpcError) as exc_info:
        await tasks_rpc.task_list({}, ctx)

    assert exc_info.value.code == RpcErrorCode.INVALID_PARAMS
