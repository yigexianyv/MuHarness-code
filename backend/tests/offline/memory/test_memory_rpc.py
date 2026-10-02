
from types import SimpleNamespace

import pytest

from app.domain.memory import MemoryManager
from app.server.rpc.dispatcher import RpcContext
from app.server.rpc.methods import memories as memories_rpc

pytestmark = pytest.mark.asyncio


# 函数说明：test_memory_list_returns_core_active_and_archived
# 用途：回归验证回归测试与测试辅助中的 `memory_list_returns_core_active_and_archived` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryManager` → `manager.initialize`
#  → `manager.core.update` → `manager.create` → `manager.archive` → `RpcContext`；另有 2
#  个调用点。
# 分支与异常：
#   验证条件：`result['core'] == '用户偏好中文回答。'`。
#   验证条件：`result['active_count'] == 1`。
#   验证条件：`result['max_active'] == 5`。
#   验证条件：`result['active'][0]['id'] == active.id`。
async def test_memory_list_returns_core_active_and_archived(tmp_path) -> None:
    manager = MemoryManager(tmp_path / "memory", max_active=5)
    await manager.initialize()
    await manager.core.update("用户偏好中文回答。")
    active = await manager.create(
        title="项目决定",
        summary="记录项目的长期技术决定",
        content="使用 Markdown Memory。",
    )
    archived = await manager.create(
        title="旧决定",
        summary="已经过期的旧决定",
        content="旧的实现方案。",
    )
    await manager.archive(archived.id, reason="方案已替换")
    ctx = RpcContext(SimpleNamespace(memory_manager=manager), SimpleNamespace())

    result = await memories_rpc.memory_list({}, ctx)

    assert result["core"] == "用户偏好中文回答。"
    assert result["active_count"] == 1
    assert result["max_active"] == 5
    assert result["active"][0]["id"] == active.id
    assert result["active"][0]["content"] == "使用 Markdown Memory。"
    assert result["archived"][0]["id"] == archived.id
    assert result["archived"][0]["archive_reason"] == "方案已替换"


# 函数说明：test_memory_list_without_manager_returns_empty_view
# 用途：回归验证回归测试与测试辅助中的 `memory_list_without_manager_returns_empty_view`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RpcContext` → `SimpleNamespace` →
# `memories_rpc.memory_list`。
# 分支与异常：
#   验证条件：`result == {'core': '', 'active': [], 'archived': [], 'active_count': 0, '
# max_active': 0}`。
async def test_memory_list_without_manager_returns_empty_view() -> None:
    ctx = RpcContext(SimpleNamespace(memory_manager=None), SimpleNamespace())
    result = await memories_rpc.memory_list({}, ctx)
    assert result == {
        "core": "",
        "active": [],
        "archived": [],
        "active_count": 0,
        "max_active": 0,
    }
