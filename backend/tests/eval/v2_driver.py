"""V2 的真实机制驱动；故障只注入本次隔离环境，并逐项记录。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from app.models.types import AgentMode
from app.runtime.context.config import ContextSettings
from app.tools.approval import ApprovalScope
from app.tools.permissions.rule_factory import build_safe_rule

from .drive import _collect_state, _drain_post_run, _from_events, drive
from .outcome import MeaState, Outcome, RoundState, ToolUse, TurnOutcome
from .stage import Stage, install_approval_policy
from .v1 import _secrets, write_json


def runtime_profile(stage: Stage) -> dict:
    app = stage.app
    config = app.registry.get(app.provider).config
    return {
        "provider": app.provider,
        "model": app.model,
        "api_style": config.api_style.value,
        "base_url_sha256": hashlib.sha256((config.base_url or "").encode()).hexdigest(),
        "timeout_seconds": config.timeout_seconds,
        "max_steps": app.max_steps,
        "max_tool_rounds": app.max_tool_rounds,
        "max_output_tokens": app.max_output_tokens,
        "system_prompt_sha256": hashlib.sha256(
            (app.system_prompt or "").encode()
        ).hexdigest(),
        "context": ContextSettings().model_dump(mode="json"),
        "run_budget": app.runtime._run_budget.config.model_dump(mode="json"),
        "memory_reflection_enabled": app.memory_reflection_enabled,
        "memory_reflection": app._memory_reflection_config.model_dump(mode="json")
        if app._memory_reflection_config is not None
        else None,
        "python_version": list(sys.version_info[:3]),
    }


async def dispatch_turn(stage: Stage, conversation_id: str, turn) -> TurnOutcome:
    dispatched = await stage.app.conversation_service.dispatch(
        conversation_id=conversation_id,
        content=turn.say,
        mode=AgentMode(turn.mode),
    )
    events = await stage.app.trace_store.load_events(dispatched.run.id)
    result = dispatched.result
    return TurnOutcome(
        say=turn.say,
        run_id=dispatched.run.id,
        conversation_id=conversation_id,
        answer=result.final_message.content or "",
        stop_reason=result.stop_reason.value,
        steps=result.steps,
        tools=[
            ToolUse(
                name=item.tool_call.name,
                arguments=item.tool_call.arguments,
                success=item.result.success,
                error=item.result.error,
                output=(item.result.output or "")[:4000],
            )
            for item in result.tool_calls
        ],
        **_from_events(events),
    )


def inject_read_failure(stage: Stage) -> list[dict]:
    tool = stage.app.tool_registry.get("read_file")
    original = tool.execute
    observations = []

    async def execute(arguments):
        if arguments.get("path") == "retry.txt" and not observations:
            observations.append(
                {"kind": "transient_read", "path": "retry.txt", "count": 1}
            )
            raise OSError("EVAL_TRANSIENT_READ_FAILURE: 临时读取失败，请重试")
        return await original(arguments)

    tool.execute = execute
    return observations


def install_scoped_approval(stage: Stage) -> dict:
    gate = stage.app.web_approval_gate
    observations = {"rule": None, "requests": []}

    async def decide(record):
        if not observations["requests"]:
            rule = build_safe_rule(
                tool_name=record.tool_name,
                arguments=record.arguments,
                scope=ApprovalScope.CONVERSATION,
                scope_id=record.conversation_id,
            )
            await stage.app.rule_store.add(rule)
            observations["rule"] = rule.model_dump(mode="json")
            decision = "approved"
            await gate.approve(record.id)
        else:
            decision = "denied"
            await gate.deny(record.id)
        observations["requests"].append(
            {
                "approval_id": record.id,
                "run_id": record.run_id,
                "conversation_id": record.conversation_id,
                "arguments": record.arguments,
                "decision": decision,
            }
        )

    async def broadcast(method, params):
        if method == "approval.required":
            asyncio.create_task(decide(params["approval"]))

    gate.set_broadcaster(broadcast)
    return observations


def inject_before_audit(stage: Stage) -> list[dict]:
    runner = stage.app.mea_runner
    original = runner._audit
    observations = []

    async def audit(mea, rnd):
        path = stage.paths.workspace / "sum_numbers.py"
        if not observations and path.is_file() and not path.is_symlink():
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            # 人工定义的受控故障发生在审计快照前，不伪造 Auditor 的判断。
            path.write_text("import sys\nprint(0)\n", encoding="utf-8")
            observations.append(
                {
                    "kind": "artifact_corruption",
                    "round_index": rnd.index,
                    "path": "sum_numbers.py",
                    "before_sha256": before,
                    "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "phase": "before_audit_snapshot",
                }
            )
        return await original(mea, rnd)

    runner._audit = audit
    return observations


async def capture_mea(stage: Stage, outcome: Outcome, mea, request: str) -> None:
    final = await stage.app.mea_runner.wait(mea.id)
    rounds = await stage.app.mea_store.rounds(mea.id)
    task = await stage.app.task_store.get(mea.task_id)
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
        steps={step.id: step.status.value for step in task.steps},
        role_rejections=sum(sum(rnd.executor_rejections.values()) for rnd in rounds),
    )
    outcome.turns.append(TurnOutcome(say=request, answer=final.final_response or ""))
    outcome.diagnostics["mea_rounds"] = [rnd.model_dump(mode="json") for rnd in rounds]


async def automatic_plan(stage: Stage, outcome: Outcome, conversation_id: str) -> None:
    turn = stage.case.turns[0]
    outcome.turns.append(await dispatch_turn(stage, conversation_id, turn))
    await _drain_post_run(stage)
    tasks = await stage.app.task_store.list_for_conversation(conversation_id)
    candidates = [task for task in tasks if task.steps and task.goal]
    outcome.diagnostics["generated_plans"] = [
        task.model_dump(mode="json") for task in candidates
    ]
    if len(candidates) != 1:
        return  # 缺少有效计划是被测结果，保留现场给验收器，而非补一个人工计划。
    task = await stage.app.task_store.plan_accept(candidates[0].id)
    execution_request = turn.say + "\n用户确认：已接受上述计划，现在执行、验证并交付。"
    outcome.diagnostics["plan_acceptance"] = {
        "task_id": task.id,
        "execution_request": execution_request,
    }
    mea = await stage.app.mea_runner.start(
        task_id=task.id,
        conversation_id=conversation_id,
        original_request=execution_request,
        round_budget=12,
    )
    await capture_mea(stage, outcome, mea, execution_request)


async def concurrent_session(
    stage: Stage, outcome: Outcome, conversation_id: str
) -> None:
    outcome.conversations["A"] = conversation_id
    requests = []

    async def send(turn):
        observation = {"say": turn.say, "submitted": time.monotonic()}
        requests.append(observation)
        result = await dispatch_turn(stage, conversation_id, turn)
        observation.update({"finished": time.monotonic(), "run_id": result.run_id})
        return result

    outcome.turns.extend(
        await asyncio.gather(*(send(turn) for turn in stage.case.turns))
    )
    await _drain_post_run(stage)
    outcome.diagnostics["concurrent_requests"] = requests


async def process_restart(
    stage: Stage,
    outcome: Outcome,
    conversation_id: str,
    *,
    non_stream: bool,
    api_retries: int,
) -> None:
    assert stage.factory is not None
    secrets = _secrets(stage.app)
    provider, model = stage.app.provider, stage.app.model
    await stage.app.close()
    workers = []
    outcome.diagnostics["restart_workers"] = workers
    try:
        for index, turn in enumerate(stage.case.turns):
            control = stage.paths.data / f"restart-control-{index}.json"
            result_path = stage.paths.data / f"restart-result-{index}.json"
            payload = {
                "root": str(stage.paths.root),
                "case": stage.case.model_dump(mode="json"),
                "variant": stage.variant.model_dump(mode="json"),
                "conversation_id": conversation_id,
                "turn": turn.model_dump(mode="json"),
                "result_path": str(result_path),
                "provider": provider,
                "model": model,
                "non_stream": non_stream,
                "api_retries": api_retries,
                "scripted": provider == "fake",
            }
            # 进程参数文件不保存模型凭据；凭据由子进程读取项目配置。
            if any(
                secret in json.dumps(payload, ensure_ascii=False) for secret in secrets
            ):
                raise ValueError("重启控制配置包含模型凭据，拒绝写入文件")
            write_json(control, payload)
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "tests.eval.restart_worker",
                str(control),
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await proc.communicate()
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.communicate()
            if proc.returncode:
                raise RuntimeError(
                    f"重启子进程失败 {proc.returncode}: "
                    + stderr.decode("utf-8", "replace")[-1500:]
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            workers.append({**result["worker"], "exit_code": proc.returncode})
            stage.retired_api_requests.extend(result["api_requests"])
            outcome.turns.append(TurnOutcome.model_validate(result["turn"]))
            if outcome.turns[-1].stop_reason in {"model_error", "context_error"}:
                break
    finally:
        # 即使子进程失败，也重新打开真实存储，以导出已经落盘的运行证据。
        stage.app = stage.factory(stage.paths, stage.variant)
        await stage.app.start()
        install_approval_policy(stage.app, stage.case.approvals)


def make_driver(*, non_stream: bool = False, api_retries: int = 3):
    async def execute(stage: Stage, *, attempt: int) -> Outcome:
        profile = runtime_profile(stage)
        injections = inject_read_failure(stage) if stage.case.id == "B04" else []
        scope = install_scoped_approval(stage) if stage.case.id == "A04" else None
        audit_injections = inject_before_audit(stage) if stage.case.id == "L04" else []
        if stage.case.id not in {"S03", "S04", "L03"}:
            outcome = await drive(stage, attempt=attempt)
        else:
            outcome = Outcome(
                case_id=stage.case.id, variant=stage.variant.name, attempt=attempt
            )
            conversation = await stage.app.conversation_store.create(
                title=f"eval {stage.case.id}"
            )
            outcome.conversations["default"] = conversation.id
            try:
                async with asyncio.timeout(stage.case.timeout_seconds):
                    if stage.case.id == "S03":
                        await process_restart(
                            stage,
                            outcome,
                            conversation.id,
                            non_stream=non_stream,
                            api_retries=api_retries,
                        )
                    elif stage.case.id == "S04":
                        await concurrent_session(stage, outcome, conversation.id)
                    else:
                        await automatic_plan(stage, outcome, conversation.id)
                    await _collect_state(stage, conversation.id, outcome)
            except TimeoutError:
                outcome.status = "timeout"
                outcome.error = f"超过 {stage.case.timeout_seconds:g} 秒"
            except Exception as exc:
                outcome.status = "error"
                outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.diagnostics.update(
            {
                "runtime_profile": profile,
                "injections": injections + audit_injections,
            }
        )
        if scope is not None:
            outcome.diagnostics["approval_scope"] = scope
        if outcome.mea is not None and "mea_rounds" not in outcome.diagnostics:
            runs = await stage.app.mea_store.list_runs(
                conversation_id=next(iter(outcome.conversations.values()))
            )
            if runs:
                outcome.diagnostics["mea_rounds"] = [
                    rnd.model_dump(mode="json")
                    for rnd in await stage.app.mea_store.rounds(runs[0].id)
                ]
        return outcome

    return execute
