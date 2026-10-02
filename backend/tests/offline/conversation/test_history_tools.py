from __future__ import annotations

import pytest

from app.domain.conversation import SQLiteConversationStore
from app.domain.conversation.tools import HistoryReadTool, HistorySearchTool
from app.models.types import Message, MessageRole, ToolCall
from app.tools import ToolExecutionContext


# 函数说明：test_history_search_and_read_use_raw_current_conversation
# 用途：回归验证回归测试与测试辅助中的
# `history_search_and_read_use_raw_current_conversation` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `store.create` → `Message` → `ToolCall` → `ToolExecutionContext`
# ；另有 4 个调用点。
# 分支与异常：
#   验证条件：`searched['count'] == 1`。
#   验证条件：`'必须使用中文注释' in searched['results'][0]['content']`。
#   验证条件：`[item['sequence'] for item in window['messages']] == [0, 1]`。
@pytest.mark.asyncio
async def test_history_search_and_read_use_raw_current_conversation(tmp_path) -> None:
    store = SQLiteConversationStore(tmp_path / "muharness.db")
    await store.initialize()
    first = await store.create(
        messages=(
            Message(role=MessageRole.USER, content="项目必须使用中文注释"),
            Message(role=MessageRole.ASSISTANT, content="已记录这个约束"),
            Message(role=MessageRole.USER, content="继续开发"),
        )
    )
    await store.create(
        messages=(Message(role=MessageRole.USER, content="另一个会话也有中文"),)
    )
    call = ToolCall(id="history", name="history_search", arguments={})
    context = ToolExecutionContext(
        tool_call=call,
        conversation_id=first.id,
    )

    searched = await HistorySearchTool(store).execute_with_context(
        {"query": "中文"},
        context,
    )
    window = await HistoryReadTool(store).execute_with_context(
        {"sequence": searched["results"][0]["sequence"], "before": 0, "after": 1},
        context,
    )

    assert searched["count"] == 1
    assert "必须使用中文注释" in searched["results"][0]["content"]
    assert [item["sequence"] for item in window["messages"]] == [0, 1]


# 函数说明：test_history_tool_rejects_missing_conversation_context
# 用途：回归验证回归测试与测试辅助中的
# `history_tool_rejects_missing_conversation_context` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteConversationStore` →
# `store.initialize` → `ToolCall` → `pytest.raises` →
# `HistorySearchTool(store).execute_with_context` → `HistorySearchTool`；另有 1 个调用点
# 。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='conversation context')`。
@pytest.mark.asyncio
async def test_history_tool_rejects_missing_conversation_context(tmp_path) -> None:
    store = SQLiteConversationStore(tmp_path / "muharness.db")
    await store.initialize()
    call = ToolCall(id="history", name="history_search", arguments={})

    with pytest.raises(ValueError, match="conversation context"):
        await HistorySearchTool(store).execute_with_context(
            {"query": "anything"},
            ToolExecutionContext(tool_call=call),
        )
