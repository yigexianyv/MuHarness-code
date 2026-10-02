from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.domain.task import (
    FileTaskStore,
    TaskPatch,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
    TaskToolOutputAttributionResolver,
)
from app.models.types import ToolCall, ToolDefinition
from app.records.evidence import (
    EvidenceCapacityError,
    EvidenceReadTool,
    EvidenceRecorder,
    EvidenceSearchTool,
    SQLiteEvidenceStore,
)
from app.tools import BaseTool, ToolExecutionContext, ToolExecutor, ToolRegistry


class LargeOutputTool(BaseTool):
    # 函数说明：LargeOutputTool.definition
    # 用途：提供 LargeOutputTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(name='large_output')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="large_output")

    # 函数说明：LargeOutputTool.execute
    # 用途：执行LargeOutputTool，供回归测试与测试辅助使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `str`；返回 `'开头-' + 'x' * 24000 + '-隐藏结论'`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        return "开头-" + ("x" * 24_000) + "-隐藏结论"


class FailingRecorder:
    # 函数说明：FailingRecorder.record
    # 用途：记录FailingRecorder，供回归测试与测试辅助使用。
    # 参数：
    #   context：本次操作的上下文对象。
    #   content：内容正文。
    # 返回：不返回结果值（隐式 None）。
    async def record(self, context, content):
        raise OSError("disk unavailable")


class FailingAttributionResolver:
    # 函数说明：FailingAttributionResolver.resolve
    # 用途：解析或定位FailingAttributionResolver，供回归测试与测试辅助使用。
    # 参数：
    #   conversation_id：目标会话标识。
    # 返回：不返回结果值（隐式 None）。
    async def resolve(self, conversation_id):
        raise OSError("task store unavailable")


class UnrecordedOutputTool(LargeOutputTool):
    # 函数说明：UnrecordedOutputTool.definition
    # 用途：提供 UnrecordedOutputTool 的模型可见定义，包含名称、说明、参数结构及权限声明
    # 。
    # 返回：类型 `ToolDefinition`；返回
    # `ToolDefinition(name='unrecorded', record_output=False)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="unrecorded", record_output=False)


# 函数说明：test_executor_archives_raw_output_before_model_preview_is_truncated
# 用途：回归验证回归测试与测试辅助中的
# `executor_archives_raw_output_before_model_preview_is_truncated` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteEvidenceStore` →
# `store.initialize` → `ToolRegistry` → `registry.register` → `LargeOutputTool` →
# `ToolExecutor`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`result.output_truncated is True`。
#   验证条件：`len(result.output or '') == 20000`。
#   验证条件：`result.evidence_id is not None`。
@pytest.mark.asyncio
async def test_executor_archives_raw_output_before_model_preview_is_truncated(
    tmp_path,
) -> None:
    store = SQLiteEvidenceStore(tmp_path / "muharness.db")
    await store.initialize()
    registry = ToolRegistry()
    registry.register(LargeOutputTool())
    executor = ToolExecutor(registry, output_recorder=EvidenceRecorder(store))
    call = ToolCall(id="call-large", name="large_output", arguments={})

    result = await executor.execute(
        call,
        context=ToolExecutionContext(
            tool_call=call,
            run_id="run-a",
            conversation_id="conversation-a",
        ),
    )

    assert result.success is True
    assert result.output_truncated is True
    assert len(result.output or "") == 20_000
    assert result.evidence_id is not None
    document = await store.resolve(
        result.evidence_id,
        conversation_id="conversation-a",
    )
    assert document is not None
    assert document.content.endswith("-隐藏结论")
    assert document.record.content_chars == result.output_chars


# 函数说明：test_evidence_tools_are_conversation_private_and_support_pagination
# 用途：回归验证回归测试与测试辅助中的
# `evidence_tools_are_conversation_private_and_support_pagination` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteEvidenceStore` →
# `store.initialize` → `store.create` → `ToolCall` → `ToolExecutionContext` →
# `EvidenceSearchTool(store).execute_with_context`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`searched['results'][0]['evidence_id'] == record.id`。
#   验证条件：`first_page['content'] == 'alpha-关键'`。
#   验证条件：`first_page['next_offset'] == 8`。
#   验证条件：`hidden == {'found': False, 'evidence_id': record.id}`。
@pytest.mark.asyncio
async def test_evidence_tools_are_conversation_private_and_support_pagination(
    tmp_path,
) -> None:
    store = SQLiteEvidenceStore(tmp_path / "muharness.db")
    await store.initialize()
    record = await store.create(
        conversation_id="conversation-a",
        run_id="run-a",
        tool_call_id="call-a",
        tool_name="read_file",
        content="alpha-关键证据-omega",
        sha256="a" * 64,
    )
    call = ToolCall(id="lookup", name="evidence_search", arguments={})
    own_context = ToolExecutionContext(
        tool_call=call,
        conversation_id="conversation-a",
    )
    other_context = ToolExecutionContext(
        tool_call=call,
        conversation_id="conversation-b",
    )

    searched = await EvidenceSearchTool(store).execute_with_context(
        {"query": "关键证据"},
        own_context,
    )
    first_page = await EvidenceReadTool(store).execute_with_context(
        {"evidence_id": record.id[:8], "offset": 0, "limit": 8},
        own_context,
    )
    hidden = await EvidenceReadTool(store).execute_with_context(
        {"evidence_id": record.id},
        other_context,
    )

    assert searched["results"][0]["evidence_id"] == record.id
    assert first_page["content"] == "alpha-关键"
    assert first_page["next_offset"] == 8
    assert hidden == {"found": False, "evidence_id": record.id}


# 函数说明：test_evidence_create_is_idempotent_but_content_is_immutable
# 用途：回归验证回归测试与测试辅助中的
# `evidence_create_is_idempotent_but_content_is_immutable` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteEvidenceStore` →
# `store.initialize` → `store.create` → `pytest.raises`。
# 分支与异常：
#   验证条件：`first == second`。
#   预期异常：`pytest.raises(ValueError, match='evidence conflict')`。
@pytest.mark.asyncio
async def test_evidence_create_is_idempotent_but_content_is_immutable(
    tmp_path,
) -> None:
    store = SQLiteEvidenceStore(tmp_path / "muharness.db")
    await store.initialize()
    arguments = {
        "conversation_id": "conversation-a",
        "run_id": "run-a",
        "tool_call_id": "call-a",
        "tool_name": "read_file",
        "content": "原始结果",
        "sha256": "b" * 64,
    }

    first = await store.create(**arguments)
    second = await store.create(**arguments)

    assert first == second
    with pytest.raises(ValueError, match="evidence conflict"):
        await store.create(
            **{**arguments, "content": "被篡改", "sha256": "c" * 64}
        )


# 函数说明：test_concurrent_idempotent_create_returns_one_evidence
# 用途：回归验证回归测试与测试辅助中的
# `concurrent_idempotent_create_returns_one_evidence` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteEvidenceStore` →
# `store.initialize` → `asyncio.gather` → `store.create`。
# 分支与异常：
#   验证条件：`first.id == second.id`。
@pytest.mark.asyncio
async def test_concurrent_idempotent_create_returns_one_evidence(tmp_path) -> None:
    store = SQLiteEvidenceStore(tmp_path / "muharness.db")
    await store.initialize()
    arguments = {
        "conversation_id": "conversation-a",
        "run_id": "run-a",
        "tool_call_id": "call-a",
        "tool_name": "read_file",
        "content": "原始结果",
        "sha256": "d" * 64,
    }

    first, second = await asyncio.gather(
        store.create(**arguments),
        store.create(**arguments),
    )

    assert first.id == second.id


# 函数说明：test_evidence_capacity_rejects_new_content_without_changing_existing
# 用途：回归验证回归测试与测试辅助中的
# `evidence_capacity_rejects_new_content_without_changing_existing` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteEvidenceStore` →
# `store.initialize` → `store.create` → `pytest.raises` → `store.resolve` →
# `store.list_recent`。
# 分支与异常：
#   验证条件：`await store.resolve(first.id, conversation_id='conversation-a')`。
#   验证条件：`len(await store.list_recent(conversation_id='conversation-a')) == 1`。
#   预期异常：`pytest.raises(EvidenceCapacityError, match='max_total_bytes')`。
@pytest.mark.asyncio
async def test_evidence_capacity_rejects_new_content_without_changing_existing(
    tmp_path,
) -> None:
    store = SQLiteEvidenceStore(
        tmp_path / "muharness.db",
        max_item_bytes=4,
        max_total_bytes=6,
    )
    await store.initialize()
    first = await store.create(
        conversation_id="conversation-a",
        run_id="run-a",
        tool_call_id="call-a",
        tool_name="read_file",
        content="abc",
        sha256="e" * 64,
    )

    with pytest.raises(EvidenceCapacityError, match="max_total_bytes"):
        await store.create(
            conversation_id="conversation-a",
            run_id="run-b",
            tool_call_id="call-b",
            tool_name="read_file",
            content="defg",
            sha256="f" * 64,
        )

    assert await store.resolve(first.id, conversation_id="conversation-a")
    assert len(await store.list_recent(conversation_id="conversation-a")) == 1


# 函数说明：test_recorder_failure_does_not_turn_successful_tool_into_failure
# 用途：回归验证回归测试与测试辅助中的
# `recorder_failure_does_not_turn_successful_tool_into_failure` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `LargeOutputTool` → `ToolExecutor` → `FailingRecorder` → `ToolCall`；另有 2 个调用点
# 。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`result.evidence_id is None`。
#   验证条件：`result.evidence_error == 'OSError: disk unavailable'`。
@pytest.mark.asyncio
async def test_recorder_failure_does_not_turn_successful_tool_into_failure() -> None:
    registry = ToolRegistry()
    registry.register(LargeOutputTool())
    executor = ToolExecutor(registry, output_recorder=FailingRecorder())
    call = ToolCall(id="call-large", name="large_output", arguments={})

    result = await executor.execute(
        call,
        context=ToolExecutionContext(
            tool_call=call,
            run_id="run-a",
            conversation_id="conversation-a",
        ),
    )

    assert result.success is True
    assert result.evidence_id is None
    assert result.evidence_error == "OSError: disk unavailable"


# 函数说明：test_tool_definition_can_disable_output_recording
# 用途：回归验证回归测试与测试辅助中的 `tool_definition_can_disable_output_recording` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteEvidenceStore` →
# `store.initialize` → `ToolRegistry` → `registry.register` → `UnrecordedOutputTool` →
# `ToolExecutor`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.success is True`。
#   验证条件：`result.evidence_id is None`。
#   验证条件：`await store.list_recent(conversation_id='conversation-a') == ()`。
@pytest.mark.asyncio
async def test_tool_definition_can_disable_output_recording(tmp_path) -> None:
    store = SQLiteEvidenceStore(tmp_path / "muharness.db")
    await store.initialize()
    registry = ToolRegistry()
    registry.register(UnrecordedOutputTool())
    executor = ToolExecutor(registry, output_recorder=EvidenceRecorder(store))
    call = ToolCall(id="call-skip", name="unrecorded", arguments={})

    result = await executor.execute(
        call,
        context=ToolExecutionContext(
            tool_call=call,
            run_id="run-a",
            conversation_id="conversation-a",
        ),
    )

    assert result.success is True
    assert result.evidence_id is None
    assert await store.list_recent(conversation_id="conversation-a") == ()


# 函数说明：test_attribution_failure_still_archives_raw_output
# 用途：回归验证回归测试与测试辅助中的 `attribution_failure_still_archives_raw_output`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteEvidenceStore` →
# `store.initialize` → `EvidenceRecorder` → `FailingAttributionResolver` → `ToolCall` →
# `recorder.record`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`recorded is not None`。
#   验证条件：`document is not None`。
#   验证条件：`document.record.task_id is None`。
@pytest.mark.asyncio
async def test_attribution_failure_still_archives_raw_output(tmp_path) -> None:
    store = SQLiteEvidenceStore(tmp_path / "muharness.db")
    await store.initialize()
    recorder = EvidenceRecorder(
        store,
        attribution_resolver=FailingAttributionResolver(),
    )
    call = ToolCall(id="call-a", name="read_file", arguments={})

    recorded = await recorder.record(
        ToolExecutionContext(
            tool_call=call,
            run_id="run-a",
            conversation_id="conversation-a",
        ),
        "仍需保存",
    )

    assert recorded is not None
    document = await store.resolve(
        recorded.id,
        conversation_id="conversation-a",
    )
    assert document is not None
    assert document.record.task_id is None


# 函数说明：test_recorder_attributes_evidence_to_active_task_step
# 用途：回归验证回归测试与测试辅助中的
# `recorder_attributes_evidence_to_active_task_step` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` →
# `task_store.initialize` → `task_store.create` → `TaskStep` → `task_store.apply_patch`
# → `TaskPatch`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`recorded is not None`。
#   验证条件：`document is not None`。
#   验证条件：`document.record.task_id == task.id`。
#   验证条件：`document.record.task_step_id == 'research'`。
@pytest.mark.asyncio
async def test_recorder_attributes_evidence_to_active_task_step(tmp_path) -> None:
    task_store = FileTaskStore(tmp_path / "tasks")
    await task_store.initialize()
    task = await task_store.create(
        title="调查问题",
        owner_conversation_id="conversation-a",
        steps=(TaskStep(id="research", title="收集证据"),),
    )
    task = await task_store.apply_patch(
        task.id,
        TaskPatch(status=TaskStatus.ACTIVE),
    )
    await task_store.apply_patch(
        task.id,
        TaskPatch(
            step_id="research",
            step_status=TaskStepStatus.IN_PROGRESS,
        ),
    )
    store = SQLiteEvidenceStore(tmp_path / "muharness.db")
    await store.initialize()
    recorder = EvidenceRecorder(
        store,
        attribution_resolver=TaskToolOutputAttributionResolver(task_store),
    )
    call = ToolCall(id="call-a", name="web_search", arguments={})

    recorded = await recorder.record(
        ToolExecutionContext(
            tool_call=call,
            run_id="run-a",
            conversation_id="conversation-a",
        ),
        "可信结果",
    )

    assert recorded is not None
    document = await store.resolve(
        recorded.id,
        conversation_id="conversation-a",
    )
    assert document is not None
    assert document.record.task_id == task.id
    assert document.record.task_step_id == "research"
