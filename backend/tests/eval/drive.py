"""驱动一次交互，并把运行时里的结果整理成 ``Outcome``。

两种驱动方式：
- 对话：在一个新会话里依次发送 ``turns``，每轮等运行结束、后台反思结束再发下一轮；
- 长任务：直接创建一个已接受的计划并启动 MEA，等它离开 running。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

from app.domain.task import TaskStep
from app.models.types import AgentMode
from app.records.trace import summarize_run_usage
from app.runtime.agent.budget import chargeable_tokens
from app.runtime.agent.events import AgentEvent, AgentEventType
from app.runtime.agent.result import AgentStopReason
from app.runtime.mea.extra_tools import validate_extra_tools

from .outcome import (
    OUTPUT_KEEP_CHARS,
    ApprovalUse,
    ArtifactState,
    ContextStep,
    MeaState,
    MemoryState,
    Outcome,
    Reflection,
    RoundState,
    TaskState,
    ToolUse,
    TurnOutcome,
)
from .stage import Stage

POST_RUN_DRAIN_SECONDS = 180.0
MAX_SNAPSHOT_FILE_BYTES = 256_000


# 函数说明：drive
# 用途：在回归测试与测试辅助中处理 `drive`，通过 `time.perf_counter` 完成首个内部处理步
# 骤。
# 参数：
#   stage：传给 `runner` 的输入，类型 `Stage`。
#   attempt：`attempt`输入或配置值，类型 `int`。
# 返回：类型 `Outcome`；返回 `outcome`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Outcome` → `time.perf_counter` →
# `stage.app.conversation_store.create` → `asyncio.wait_for` → `runner` →
# `_collect_state`；另有 1 个调用点。
# 分支与异常：
#   捕获 `TimeoutError` 后，执行异常分支中的状态更新；具体更新见实现。
#   捕获 `Exception` 后，执行异常处理调用 `type`。
# 副作用与资源：
#   更新对象字段：`outcome.status`、`outcome.error`、`outcome.duration_seconds`。
async def drive(stage: Stage, *, attempt: int) -> Outcome:
    outcome = Outcome(case_id=stage.case.id, variant=stage.variant.name, attempt=attempt)
    outcome.initial_file_hashes = _snapshot_file_hashes(stage.paths.workspace)
    started = time.perf_counter()
    try:
        conversation = await stage.app.conversation_store.create(title=f"eval {stage.case.id}")
        outcome.conversations["default"] = conversation.id
        runner = _drive_mea if stage.case.mea is not None else _drive_chat
        await asyncio.wait_for(
            runner(stage, conversation.id, outcome),
            timeout=stage.case.timeout_seconds,
        )
        await _collect_state(stage, conversation.id, outcome)
    except TimeoutError:
        outcome.status = "timeout"
        outcome.error = f"超过 {stage.case.timeout_seconds:g} 秒"
    except Exception as exc:
        outcome.status = "error"
        outcome.error = f"{type(exc).__name__}: {exc}"
    outcome.duration_seconds = round(time.perf_counter() - started, 2)
    return outcome


# ---------------------------------------------------------------------------
# 对话
# ---------------------------------------------------------------------------


# 函数说明：_drive_chat
# 用途：在回归测试与测试辅助中处理 `_drive_chat`，通过
# `stage.app.conversation_service.dispatch` 完成首个内部处理步骤。
# 参数：
#   stage：传给 `_drain_post_run` 的输入，类型 `Stage`。
#   conversation_id：目标会话标识，类型 `str`。
#   outcome：执行或验收结果，类型 `Outcome`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `stage.app.conversation_service.dispatch` → `AgentMode` → `_drain_post_run` →
# `stage.app.trace_store.load_events` → `ToolUse` → `TurnOutcome`；另有 1 个调用点。
async def _drive_chat(stage: Stage, conversation_id: str, outcome: Outcome) -> None:
    for turn in stage.case.turns:
        if turn.session not in outcome.conversations:
            created = await stage.app.conversation_store.create(
                title=f"eval {stage.case.id} / {turn.session}"
            )
            outcome.conversations[turn.session] = created.id
        current_conversation_id = outcome.conversations[turn.session]
        user_sequence = len(await stage.app.conversation_store.load_messages(current_conversation_id))
        dispatched = await stage.app.conversation_service.dispatch(
            conversation_id=current_conversation_id,
            content=turn.say,
            mode=AgentMode(turn.mode),
        )
        await _drain_post_run(stage)
        events = await stage.app.trace_store.load_events(dispatched.run.id)
        result = dispatched.result
        tools = [
            ToolUse(
                name=record.tool_call.name,
                call_id=record.tool_call.id,
                round_index=record.round_index,
                arguments=record.tool_call.arguments,
                success=record.result.success,
                error=record.result.error,
                output=(record.result.output or "")[:OUTPUT_KEEP_CHARS],
                exit_code=_exit_code(record.result.output),
            )
            for record in result.tool_calls
        ]
        outcome.turns.append(
            TurnOutcome(
                say=turn.say,
                run_id=dispatched.run.id,
                conversation_id=current_conversation_id,
                user_sequence=user_sequence,
                answer=result.final_message.content or "",
                stop_reason=result.stop_reason.value,
                steps=result.steps,
                tools=tools,
                received_tool_outputs=(
                    await _received_tool_outputs(stage, dispatched.run.id, events, tools)
                    if any("grounded_answer" in check for check in stage.case.checks) else []
                ),
                **_from_events(events),
            )
        )
        if result.stop_reason in {AgentStopReason.MODEL_ERROR, AgentStopReason.CONTEXT_ERROR}:
            break


# 函数说明：_drain_post_run
# 用途：记忆反思在后台执行；等它结束，下一轮和最终检查才看得到它的结果。
# 参数：
#   stage：`stage`输入或配置值，类型 `Stage`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`time.monotonic` → `asyncio.sleep`。
# 分支与异常：
#   当 `time.monotonic() > deadline` 时，抛出 `TimeoutError('等待运行后反思超时')`。
async def _drain_post_run(stage: Stage) -> None:
    """记忆反思在后台执行；等它结束，下一轮和最终检查才看得到它的结果。"""

    deadline = time.monotonic() + POST_RUN_DRAIN_SECONDS
    while stage.app.post_run_processor.active_count:
        if time.monotonic() > deadline:
            raise TimeoutError("等待运行后反思超时")
        await asyncio.sleep(0.2)


# 函数说明：_from_events
# 用途：在回归测试与测试辅助中处理 `_from_events`，通过 `context.append` 完成首个内部处
# 理步骤。
# 参数：
#   events：事件集合，类型 `Sequence[AgentEvent]`。
# 返回：类型 `dict[str, object]`；字典，包含字段 `context`、`approvals`、`failed_events`
# 、`reflection`、`chargeable_tokens`、`model_calls`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Reflection` → `ContextStep` →
# `summarize_run_usage` → `chargeable_tokens`。
def _from_events(events: Sequence[AgentEvent]) -> dict[str, object]:
    context: list[ContextStep] = []
    approvals: list[str] = []
    approval_calls: list[ApprovalUse] = []
    failed: list[str] = []
    reflection = Reflection()
    for event in events:
        if event.type is AgentEventType.MODEL_STARTED:
            context.append(
                ContextStep(
                    stage=event.compaction_stage,
                    compacted_tool_results=event.compacted_tool_results or 0,
                    removed_tool_rounds=event.removed_tool_rounds or 0,
                    summary_updated=bool(event.summary_updated),
                    reached_target=event.reached_target,
                    prepared_input_tokens=event.prepared_input_tokens,
                    summary_covered_after=event.summary_covered_after,
                )
            )
        elif event.type is AgentEventType.TOOL_APPROVAL_COMPLETED and event.approval_decision:
            approvals.append(event.approval_decision.value)
            if event.tool_call is not None:
                approval_calls.append(ApprovalUse(
                    call_id=event.tool_call.id, decision=event.approval_decision.value,
                ))
        elif event.type is AgentEventType.AGENT_FAILED:
            failed.append(event.error.message if event.error else "agent_failed")
        elif event.type is AgentEventType.MEMORY_REFLECTION_COMPLETED:
            reflection = Reflection(
                status="completed",
                action=event.reflection_action,
                mutation_applied=event.reflection_mutation_applied,
            )
        elif event.type is AgentEventType.MEMORY_REFLECTION_SKIPPED:
            reflection = Reflection(status="skipped", skip_reason=event.reflection_skip_reason)
        elif event.type is AgentEventType.MEMORY_REFLECTION_FAILED:
            reflection = Reflection(status="failed", skip_reason=event.reflection_error)
    usage = summarize_run_usage(events).provider_total
    return {
        "context": context,
        "approvals": approvals,
        "approval_calls": approval_calls,
        "failed_events": failed,
        "reflection": reflection,
        "chargeable_tokens": chargeable_tokens(usage),
        "model_calls": usage.model_calls,
    }


# ---------------------------------------------------------------------------
# 长任务
# ---------------------------------------------------------------------------


# 函数说明：_drive_mea
# 用途：执行 `_drive_mea` 测试辅助流程并检查预期结果。
# 参数：
#   stage：`stage`输入或配置值，类型 `Stage`。
#   conversation_id：目标会话标识，类型 `str`。
#   outcome：执行或验收结果，类型 `Outcome`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`app.task_store.create` → `TaskStep` →
#  `app.task_store.plan_accept` → `app.mea_runner.start` → `validate_extra_tools` →
# `app.mea_runner.wait`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`plan is not None`。
# 副作用与资源：
#   更新对象字段：`outcome.mea`。
async def _drive_mea(stage: Stage, conversation_id: str, outcome: Outcome) -> None:
    plan = stage.case.mea
    assert plan is not None
    app = stage.app
    task = await app.task_store.create(
        title=plan.title,
        goal=plan.goal,
        steps=[TaskStep(id=step.id, title=step.title, acceptance=step.acceptance) for step in plan.steps],
        owner_conversation_id=conversation_id,
    )
    task = await app.task_store.plan_accept(task.id)
    mea = await app.mea_runner.start(
        task_id=task.id,
        conversation_id=conversation_id,
        original_request=plan.request,
        round_budget=plan.round_budget,
        extra_tools=validate_extra_tools(plan.extra_tools, app.tool_registry),
    )
    final = await app.mea_runner.wait(mea.id)
    rounds = await app.mea_store.rounds(mea.id)
    task_after = await app.task_store.get(task.id)
    outcome.mea = MeaState(
        status=final.status.value,
        abort_reason=final.abort_reason,
        pending_question=final.pending_question,
        rounds=[
            RoundState(
                index=rnd.index,
                kind=rnd.kind.value,
                phase=rnd.phase.value,
                step_id=rnd.step_id,
                audit_status=rnd.audit_status,
                integrity=rnd.integrity_status,
                step_acceptance=rnd.step_acceptance,
            )
            for rnd in rounds
        ],
        steps={step.id: step.status.value for step in (task_after.steps if task_after else ())},
        role_rejections=sum(sum(rnd.executor_rejections.values()) for rnd in rounds),
    )
    # 长任务的最终答复当作一轮回答，这样 answer_has 等检查同样适用
    outcome.turns.append(
        TurnOutcome(say=plan.request, answer=final.final_response or "", stop_reason=None)
    )


# ---------------------------------------------------------------------------
# 结束时的状态
# ---------------------------------------------------------------------------


# 函数说明：_collect_state
# 用途：收集状态，供回归测试与测试辅助使用。
# 参数：
#   stage：`stage`输入或配置值，类型 `Stage`。
#   conversation_id：目标会话标识，类型 `str`。
#   outcome：执行或验收结果，类型 `Outcome`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_snapshot_text_files` →
# `app.task_store.list_for_conversation` → `TaskState` → `app.artifact_store.list` →
# `ArtifactState` → `MemoryState`；另有 6 个调用点。
# 副作用与资源：
#   更新对象字段：`outcome.files`、`outcome.tasks`、`outcome.artifacts`、
# `outcome.memories`、`outcome.core_memory`、`outcome.chargeable_tokens`、
# `outcome.model_calls`。
async def _collect_state(stage: Stage, conversation_id: str, outcome: Outcome) -> None:
    app = stage.app
    outcome.files = _snapshot_text_files(stage.paths.workspace)
    outcome.file_hashes = _snapshot_file_hashes(stage.paths.workspace)
    tasks = await app.task_store.list_for_conversation(conversation_id)
    outcome.tasks = [
        TaskState(
            title=task.title,
            status=task.status.value,
            steps=[{"title": step.title, "status": step.status.value} for step in task.steps],
        )
        for task in tasks
    ]
    artifacts = await app.artifact_store.list(conversation_id=conversation_id, limit=200)
    outcome.artifacts = [
        ArtifactState(kind=item.kind.value, path=item.filename, url=item.source_url) for item in artifacts
    ]
    outcome.memories = [
        MemoryState(title=memory.title, summary=memory.summary, content=memory.content)
        for memory in await app.memory_manager.list()
    ]
    core_text, _ = await app.memory_manager.reflection_context()
    outcome.core_memory = core_text or ""

    # 成本按会话里所有运行汇总：普通轮次，以及长任务的管理者、执行者、审计者子运行
    total_tokens = 0
    total_calls = 0
    conversation_ids = set(outcome.conversations.values()) or {conversation_id}
    for identifier in conversation_ids:
        for run in await app.run_store.list_for_conversation(identifier):
            usage = summarize_run_usage(await app.trace_store.load_events(run.id)).provider_total
            total_tokens += chargeable_tokens(usage)
            total_calls += usage.model_calls
    outcome.chargeable_tokens = total_tokens
    outcome.model_calls = total_calls


# 函数说明：_snapshot_text_files
# 用途：在回归测试与测试辅助中处理 `_snapshot_text_files`，通过 `workspace.exists` 完成
# 首个内部处理步骤。
# 参数：
#   workspace：目标工作区，类型 `Path`。
# 返回：类型 `dict[str, str]`；返回 `files`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.exists` → `workspace.rglob`
#  → `path.is_file` → `path.is_symlink` → `path.stat` →
# `path.relative_to(workspace).as_posix`；另有 2 个调用点。
# 分支与异常：
#   当 `not workspace.exists()` 时，返回 `files`。
#   当 `not path.is_file() or path.is_symlink() or path.stat().…` 时，跳过当前循环项。
#   捕获 `UnicodeDecodeError` 后，跳过当前循环项，继续处理后续项。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
def _snapshot_text_files(workspace: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    if not workspace.exists():
        return files
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_SNAPSHOT_FILE_BYTES:
            continue
        try:
            files[path.relative_to(workspace).as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def _snapshot_file_hashes(workspace: Path) -> dict[str, str]:
    """Hash actual bytes, including binary files and Windows line endings."""
    return {
        path.relative_to(workspace).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in workspace.rglob("*")
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_SNAPSHOT_FILE_BYTES
    }


def _exit_code(output: str | None) -> int | None:
    try:
        value = json.loads(output or "")
    except ValueError:
        return None
    code = value.get("exit_code") if isinstance(value, dict) else None
    return code if type(code) is int else None


async def _received_tool_outputs(
    stage: Stage, run_id: str, events: Sequence[AgentEvent], tools: list[ToolUse],
) -> list[str]:
    """Read existing final request views, never the evidence or executor preview."""
    store = stage.app.run_step_store
    if store is None:
        return []
    successful = {tool.call_id for tool in tools if tool.success}
    outputs: dict[str, None] = {}
    for event in events:
        if event.type is not AgentEventType.MODEL_STARTED or event.step is None:
            continue
        for item in event.request_tool_views:
            identifier = item.get("tool_call_id")
            if identifier not in successful or not item.get("included"):
                continue
            view = await store.request_tool_view(run_id, event.step, identifier)
            if not view or not view.get("included") or not isinstance(view.get("content"), str):
                continue
            try:
                envelope = json.loads(view["content"])
            except ValueError:
                continue
            if isinstance(envelope, dict) and envelope.get("success") is True:
                output = envelope.get("output")
                if isinstance(output, str):
                    outputs[output] = None
    return list(outputs)


__all__ = ["drive"]
