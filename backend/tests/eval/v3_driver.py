"""V3 机制现场。用户决策和故障注入明确记录，不替模型伪造完成。"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import asdict

from app.domain.task.models import TaskStep
from app.models.types import AgentMode, ToolCall
from app.records.trace import SQLiteTraceEventHandler
from app.tools.hooks import ToolExecutionContext

from .drive import _collect_state, _drain_post_run, _from_events
from .outcome import Outcome, TurnOutcome
from .v2_driver import capture_mea, dispatch_turn, runtime_profile
from .v2_driver import make_driver as v2_driver


async def checkpoint_recovery(stage, outcome, cid):
    adapter = stage.app.registry.get(stage.app.provider)
    entered = asyncio.Event()
    suspended = False
    complete, stream = adapter.complete, adapter.complete_stream

    async def boundary():
        nonlocal suspended
        path = stage.paths.workspace / "ledger.txt"
        if not suspended and path.read_text(encoding="utf-8").strip() == "A":
            suspended = True
            entered.set()
            await asyncio.Event().wait()

    async def wrapped_complete(*args, **kwargs):
        await boundary()
        return await complete(*args, **kwargs)

    async def wrapped_stream(*args, **kwargs):
        await boundary()
        return await stream(*args, **kwargs)

    adapter.complete, adapter.complete_stream = wrapped_complete, wrapped_stream
    manager = stage.app.run_manager
    first, _ = await manager.start(
        stage.case.turns[0].say,
        conversation_id=cid,
        event_handler=SQLiteTraceEventHandler(stage.app.trace_store),
    )
    try:
        await asyncio.wait_for(entered.wait(), 90)
        before = (stage.paths.workspace / "ledger.txt").read_bytes()
        interrupted = await manager.interrupt(first)
        checkpoint = await stage.app.checkpoint_store.get_unrecovered(first)
    finally:
        adapter.complete, adapter.complete_stream = complete, stream
        if not suspended:
            await manager.cancel(first)
    outcome.diagnostics["recovery"] = {
        "interrupted_run_id": first,
        "interrupted_status": interrupted.status.value,
        "checkpoint": checkpoint.model_dump(mode="json") if checkpoint else None,
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "injection": "suspend_before_model_after_real_write",
    }
    if checkpoint is None:
        return
    dispatched = await stage.app.conversation_service.recover(first)
    final, result = dispatched.run, dispatched.result
    recovered = final.id
    outcome.diagnostics["recovery"].update(
        {
            "recovered_run_id": recovered,
            "recovered_from_run_id": final.recovered_from_run_id,
            "recovered_status": final.status.value,
            "after_sha256": hashlib.sha256(
                (stage.paths.workspace / "ledger.txt").read_bytes()
            ).hexdigest(),
        }
    )
    if result:
        outcome.turns.append(
            TurnOutcome(
                say=stage.case.turns[0].say,
                answer=result.final_message.content or "",
                run_id=recovered,
                conversation_id=cid,
                stop_reason=result.stop_reason.value,
                **_from_events(await stage.app.trace_store.load_events(recovered)),
            )
        )


async def delete_active(stage, outcome, cid):
    entered = asyncio.Event()
    tool = stage.app.tool_registry.get("read_file")
    original = tool.execute

    async def stalled(arguments):
        if arguments.get("path") == "slow.txt":
            entered.set()
            await asyncio.Event().wait()
        return await original(arguments)

    tool.execute = stalled
    other = await stage.app.conversation_store.create(title="unrelated survivor")
    manager = stage.app.run_manager
    identifier, _ = await manager.start(
        stage.case.turns[0].say,
        conversation_id=cid,
        event_handler=SQLiteTraceEventHandler(stage.app.trace_store),
    )
    try:
        await asyncio.wait_for(entered.wait(), 90)
        events = await stage.app.trace_store.load_events(identifier)
        usage = _from_events(events)
        outcome.diagnostics["pre_delete_events"] = [
            e.model_dump(mode="json") for e in events
        ]
        result = await stage.app.conversation_lifecycle.delete(cid)
        checks = {
            "conversation_absent": await stage.app.conversation_store.get(cid) is None,
            "run_absent": await stage.app.run_store.get(identifier) is None,
            "trace_empty": await stage.app.trace_store.get(identifier) is None,
            "no_active_runs": identifier not in manager.active_run_ids,
            "other_survives": await stage.app.conversation_store.get(other.id)
            is not None,
        }
        outcome.diagnostics["deletion"] = {
            "conversation_id": cid,
            "run_id": identifier,
            "tool_entered": entered.is_set(),
            "result": result.model_dump(mode="json") if result else None,
            **checks,
            "removed_usage": usage,
        }
        outcome.turns.append(
            TurnOutcome(say="用户删除会话", answer="会话已删除，活动运行已停止。")
        )
        outcome.conversations.clear()
    finally:
        tool.execute = original
        await manager.cancel_for_conversation(cid)


async def memory_probe(stage, outcome):
    manager = stage.app.memory_manager
    records = await manager.list()
    seed = records[0]
    if stage.case.id == "M05":
        updated = await manager.store.update(
            seed.id,
            content="VEGA 当前预算 18000 元。",
            reason="受控用户更新",
            expected_revision=seed.revision,
        )
        error = None
        try:
            await manager.store.update(
                seed.id,
                content="VEGA 当前预算 9000 元。",
                reason="受控过时更新",
                expected_revision=seed.revision,
            )
        except ValueError as exc:
            error = str(exc)
        await manager.close()
        await manager.initialize()
        latest = await manager.read(seed.id)
        outcome.diagnostics["revision_conflict"] = {
            "seed": seed.model_dump(mode="json"),
            "updated": updated.model_dump(mode="json"),
            "latest": latest.model_dump(mode="json"),
            "stale_error": error,
            "kind": "sequential_optimistic_conflict_probe",
        }
    else:
        index = manager._search_index
        await manager.close()
        hashes = {r.id: hashlib.sha256(r.content.encode()).hexdigest() for r in records}
        # 仅本次独立的可重建检索数据库；记忆 Markdown 正文保留。
        index.database_path.write_bytes(b"EVAL_CORRUPTED_SQLITE_INDEX")
        corrupted = hashlib.sha256(index.database_path.read_bytes()).hexdigest()
        await manager.initialize()
        result = await manager.search("LYRA")
        outcome.diagnostics["index_rebuild"] = {
            "corrupted_sha256": corrupted,
            "recovered_sha256": hashlib.sha256(
                index.database_path.read_bytes()
            ).hexdigest(),
            "before_content_sha256": hashes,
            "after_content_sha256": {
                r.id: hashlib.sha256(r.content.encode()).hexdigest()
                for r in await manager.list()
            },
            "search": asdict(result),
        }


async def role_probe(stage, outcome, cid):
    executor = stage.app.runtime.tool_executor
    probes = []
    for mode, name, arguments in [
        (AgentMode.AUDIT, "write_file", {"path": "protected.txt", "content": "PWNED"}),
        (AgentMode.MANAGE, "read_file", {"path": "protected.txt"}),
    ]:
        call = ToolCall(id=f"probe-{mode.value}", name=name, arguments=arguments)
        result = await executor.execute(
            call,
            context=ToolExecutionContext(
                tool_call=call, mode=mode, conversation_id=cid
            ),
        )
        probes.append(
            {
                "mode": mode.value,
                "call": call.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
            }
        )
    outcome.diagnostics["role_probes"] = probes


def missing_docker(stage):
    tool = stage.app.tool_registry.get("run_shell_command")
    supervisor = tool._sandbox_supervisor
    original = supervisor.prepare_launch
    original_spawn = asyncio.create_subprocess_exec
    observations = []

    def reject(**kwargs):
        observations.append(
            {
                "kind": "sandbox_unavailable",
                "command": kwargs.get("command"),
                "reason": "EVAL_DOCKER_UNAVAILABLE",
            }
        )
        raise RuntimeError(
            "EVAL_DOCKER_UNAVAILABLE: Docker unavailable; host fallback forbidden"
        )

    supervisor.prepare_launch = reject

    async def forbidden_spawn(*args, **kwargs):
        observations.append({"kind": "unexpected_process_launch"})
        raise RuntimeError("EVAL_HOST_FALLBACK_INTERCEPTED")

    asyncio.create_subprocess_exec = forbidden_spawn

    def restore():
        supervisor.prepare_launch = original
        asyncio.create_subprocess_exec = original_spawn

    return observations, restore


async def long_control(stage, outcome, cid):
    app, spec = stage.app, stage.case.mea
    task = await app.task_store.create(
        title=spec.title,
        goal=spec.goal,
        steps=[
            TaskStep(id=s.id, title=s.title, acceptance=s.acceptance)
            for s in spec.steps
        ],
        owner_conversation_id=cid,
    )
    await app.task_store.plan_accept(task.id)
    runner = app.mea_runner
    observations = []
    if stage.case.id == "L05":
        original = runner._execute

        async def execute(mea, rnd):
            if not observations:
                result = await runner.add_amendment(
                    mea.id,
                    (
                        "用户新增要求：sum_numbers.py 的所有输出必须改为 "
                        "SUM=<整数>，例如 2 -5 8 输出 SUM=5，无参数输"
                        "出 SUM=0；同步更新测试和 RUN.txt，保留原来求和功能"
                        "。"
                    ),
                )
                observations.append(
                    {
                        "round_index": rnd.index,
                        "accepted": result.accepted,
                        "revision": result.revision,
                        "phase": "before_first_executor",
                    }
                )
            return await original(mea, rnd)

        runner._execute = execute
    else:
        original = runner._audit

        async def audit(mea, rnd):
            if not observations:
                result = await runner.pause(mea.id)
                observations.append(
                    {
                        "round_index": rnd.index,
                        "pause_requested": result.pause_requested,
                    }
                )
            return await original(mea, rnd)

        runner._audit = audit
    try:
        mea = await runner.start(
            task_id=task.id,
            conversation_id=cid,
            original_request=spec.request,
            round_budget=spec.round_budget,
        )
        first = await runner.wait(mea.id)
        if stage.case.id == "L06":
            paused_rounds = await app.mea_store.rounds(mea.id)
            outcome.diagnostics["pause_resume"] = {
                "mea_id": mea.id,
                "paused_status": first.status.value,
                "paused_round_count": len(paused_rounds),
                "no_active_child": mea.id not in runner._active_child,
                "pause_observations": observations,
            }
            if first.status.value == "paused":
                resumed = await runner.resume(mea.id)
                outcome.diagnostics["pause_resume"].update(
                    {"resumed_id": resumed.id, "resumed_status": resumed.status.value}
                )
        else:
            requirements = await app.mea_store.requirements(mea.id)
            outcome.diagnostics["amendment"] = {
                "observations": observations,
                "requirements": asdict(requirements),
            }
        await capture_mea(stage, outcome, mea, spec.request)
    finally:
        if stage.case.id == "L05":
            runner._execute = original
        else:
            runner._audit = original


def make_driver(*, non_stream=False, api_retries=3):
    previous = v2_driver(non_stream=non_stream, api_retries=api_retries)

    async def execute(stage, *, attempt):
        if stage.case.id[-2:] not in {"05", "06"}:
            return await previous(stage, attempt=attempt)
        started = time.perf_counter()
        outcome = Outcome(
            case_id=stage.case.id, variant=stage.variant.name, attempt=attempt
        )
        outcome.diagnostics["runtime_profile"] = runtime_profile(stage)
        conversation = await stage.app.conversation_store.create(
            title=f"eval {stage.case.id}"
        )
        cid = conversation.id
        outcome.conversations["default"] = cid

        def restore():
            pass

        try:
            async with asyncio.timeout(stage.case.timeout_seconds):
                if stage.case.id == "S05":
                    await checkpoint_recovery(stage, outcome, cid)
                elif stage.case.id == "S06":
                    await delete_active(stage, outcome, cid)
                elif stage.case.id in {"L05", "L06"}:
                    await long_control(stage, outcome, cid)
                else:
                    if stage.case.id in {"M05", "M06"}:
                        await memory_probe(stage, outcome)
                    if stage.case.id == "A05":
                        await role_probe(stage, outcome, cid)
                    if stage.case.id == "A06":
                        observations, restore = missing_docker(stage)
                        outcome.diagnostics["sandbox_unavailable"] = observations
                    for turn in stage.case.turns:
                        outcome.turns.append(await dispatch_turn(stage, cid, turn))
                        await _drain_post_run(stage)
                await _collect_state(stage, cid, outcome)
        except TimeoutError:
            outcome.status, outcome.error = "timeout", "V3 scene timed out"
        except Exception as exc:
            outcome.status, outcome.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            restore()
        if stage.case.id == "S06":
            removed = outcome.diagnostics.get("deletion", {}).get("removed_usage", {})
            outcome.chargeable_tokens += removed.get("chargeable_tokens", 0)
            outcome.model_calls += removed.get("model_calls", 0)
        outcome.duration_seconds = round(time.perf_counter() - started, 2)
        return outcome

    return execute
