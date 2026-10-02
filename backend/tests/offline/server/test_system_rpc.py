
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.server.rpc.methods import system
from app.server.rpc.protocol import INVALID_STATE, JsonRpcError


class ImmediateLoop:
    # 函数说明：ImmediateLoop.call_later
    # 用途：执行 `call_later` 测试辅助流程并检查预期结果。
    # 参数：
    #   delay：`delay`输入或配置值，类型 `float`。
    #   callback：处理结果或事件的回调。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`callback`。
    # 分支与异常：
    #   验证条件：`delay > 0`。
    def call_later(self, delay: float, callback) -> None:  
        assert delay > 0
        callback()


# 函数说明：test_restart_is_accepted_without_active_runs
# 用途：回归验证回归测试与测试辅助中的 `restart_is_accepted_without_active_runs` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` →
# `monkeypatch.setattr` → `system.system_restart`。
# 分支与异常：
#   验证条件：`result == {'accepted': True}`。
#   验证条件：`called == [True]`。
@pytest.mark.asyncio
async def test_restart_is_accepted_without_active_runs(monkeypatch) -> None:  
    called: list[bool] = []
    application = SimpleNamespace(
        run_manager=SimpleNamespace(active_run_ids=()),
        host_restart_callback=lambda: called.append(True),
    )
    monkeypatch.setattr(system.asyncio, "get_running_loop", ImmediateLoop)

    result = await system.system_restart(
        {},
        SimpleNamespace(application=application),
    )

    assert result == {"accepted": True}
    assert called == [True]


# 函数说明：test_restart_rejects_active_runs
# 用途：回归验证回归测试与测试辅助中的 `restart_rejects_active_runs` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` → `pytest.raises` →
# `system.system_restart`。
# 分支与异常：
#   验证条件：`caught.value.code == INVALID_STATE`。
#   验证条件：`caught.value.data == {'active_run_ids': ['run-1']}`。
#   预期异常：`pytest.raises(JsonRpcError)`。
@pytest.mark.asyncio
async def test_restart_rejects_active_runs() -> None:
    application = SimpleNamespace(
        run_manager=SimpleNamespace(active_run_ids=("run-1",)),
        host_restart_callback=lambda: None,
    )

    with pytest.raises(JsonRpcError) as caught:
        await system.system_restart(
            {},
            SimpleNamespace(application=application),
        )

    assert caught.value.code == INVALID_STATE
    assert caught.value.data == {"active_run_ids": ["run-1"]}


# 函数说明：test_restart_rejects_unsupported_entrypoint
# 用途：回归验证回归测试与测试辅助中的 `restart_rejects_unsupported_entrypoint` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` → `pytest.raises` →
# `system.system_restart`。
# 分支与异常：
#   验证条件：`caught.value.code == INVALID_STATE`。
#   预期异常：`pytest.raises(JsonRpcError)`。
@pytest.mark.asyncio
async def test_restart_rejects_unsupported_entrypoint() -> None:
    application = SimpleNamespace(
        run_manager=SimpleNamespace(active_run_ids=()),
        host_restart_callback=None,
    )

    with pytest.raises(JsonRpcError) as caught:
        await system.system_restart(
            {},
            SimpleNamespace(application=application),
        )

    assert caught.value.code == INVALID_STATE
