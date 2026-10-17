"""Private worker entry point. Receives only the public task, never judge data."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

from .common import read_json, write_json
from .container import TaskContainer
from .meter import Meter, fatal_error_type, is_transient


def retry_delay(config: dict, attempt: int) -> float:
    # 5s, 20s with the default; identical schedule for both variants.
    # Capped so a long upstream overload is waited out without hour-long sleeps.
    return min(config["retry_backoff_seconds"] * 4**attempt, 300)


def recovered_errors(meter: Meter, fatal: str | None) -> int:
    total = sum(meter.stats.get("error_types", {}).values())
    return total - (1 if fatal and fatal != "CallLimitExceeded" else 0)


def credential(config: dict) -> str:
    # Allow the existing backend/.env, but do not let it change frozen settings.
    from dotenv import dotenv_values

    from .common import BACKEND

    key = os.environ.get(config["api_key_env"]) or dotenv_values(BACKEND / ".env").get(
        config["api_key_env"]
    )
    if not key:
        raise ValueError(f"Set {config['api_key_env']} before running the benchmark")
    return key


def measured_registry(config: dict, meter: Meter, key: str):
    """Model registry whose adapter counts, retries and logs every physical call."""
    from pydantic import SecretStr

    from app.models.adapter import ModelAdapter
    from app.models.config import ModelSettings, ProviderConfig
    from app.models.providers import OpenAICompatibleAdapter
    from app.models.registry import ModelAdapterRegistry
    from app.models.types import ApiStyle

    class MeasuredAdapter(ModelAdapter):
        def __init__(self, provider_config):
            super().__init__(provider_config)
            self.inner = OpenAICompatibleAdapter(provider_config)

        async def complete(self, model_request):
            model_request = model_request.model_copy(
                update={
                    "model": config["model"],
                    "temperature": config["temperature"],
                    "max_output_tokens": min(
                        model_request.max_output_tokens or config["max_output_tokens"],
                        config["max_output_tokens"],
                    ),
                    "extra_body": config["extra_body"] | model_request.extra_body,
                }
            )
            retries = config["transient_retries"]
            for attempt in range(retries + 1):
                meter.start(model_request.model_dump(mode="json"), retry=attempt)
                try:
                    response = await self.inner.complete(model_request)
                except Exception as exc:
                    retry = attempt < retries and is_transient(exc)
                    meter.error(exc, will_retry=retry)
                    if not retry:
                        raise
                    meter.progress(f"等待 {retry_delay(config, attempt):g} 秒后重试")
                    await asyncio.sleep(retry_delay(config, attempt))
                    continue
                usage = (
                    response.usage.model_dump()
                    if (response.raw or {}).get("usage") is not None
                    else None
                )
                meter.finish(response.model_dump(mode="json"), usage)
                return response
            raise AssertionError("unreachable")

        async def close(self):
            await self.inner.close()

    provider_config = ProviderConfig(
        provider=config["provider"],
        model=config["model"],
        api_key=SecretStr(key),
        api_style=ApiStyle.CHAT_COMPLETIONS,
        base_url=config["base_url"],
        timeout_seconds=config["model_timeout_seconds"],
        max_retries=0,
        default_max_output_tokens=config["max_output_tokens"],
    )
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register(
        config["provider"], MeasuredAdapter, config=provider_config, replace=True
    )
    return registry


def run_budget(config: dict):
    from app.runtime.agent.budget import RunBudgetConfig

    return RunBudgetConfig(
        _env_file=None,
        enabled=True,
        hard_tokens=config["hard_tokens"],
        warning_tokens=config["hard_tokens"] - 2,
        finalization_tokens=config["hard_tokens"] - 1,
        hard_model_calls=config["model_calls"],
        warning_model_calls=None,
        finalization_model_calls=None,
        finalization_max_output_tokens=min(1200, config["max_output_tokens"]),
    )


async def run_muharness(
    request: dict, container: TaskContainer, meter: Meter, key: str
) -> dict:
    from app.runtime.agent.events import AgentEventHandler
    from app.runtime.agent.runtime import AgentRuntime
    from app.runtime.context.config import ContextSettings
    from app.runtime.context.manager import ContextManager
    from app.runtime.context.reducers import ConversationReducer
    from app.runtime.context.summarizer import ModelContextSummarizer
    from app.tools.builtin import (
        EditFileTool,
        ListFilesTool,
        ReadFileTool,
        WriteFileTool,
    )
    from app.tools.registry import ToolRegistry

    from .exec_shell import ContainerShellTool

    config = request["config"]

    class EventLog(AgentEventHandler):
        async def emit(self, event):
            if event.delta is None and event.reasoning_delta is None:
                meter.event("runtime_event", event.model_dump(mode="json"))

    registry = measured_registry(config, meter, key)
    tools = ToolRegistry()
    for tool_type in (ListFilesTool, ReadFileTool, WriteFileTool, EditFileTool):
        tools.register(tool_type(container.workspace))
    tools.register(ContainerShellTool(container, meter))
    settings = ContextSettings(_env_file=None, **config["context"])
    summarizer = ModelContextSummarizer(
        registry,
        provider=config["provider"],
        model=config["model"],
        max_output_tokens=settings.context_summary_max_output_tokens,
    )
    reducer = ConversationReducer(
        summarizer,
        keep_recent_conversation_blocks=settings.context_keep_recent_conversation_blocks,
        keep_recent_tool_rounds=settings.context_keep_recent_tool_rounds,
        large_fold_span_tokens=settings.context_large_fold_span_tokens,
        large_fold_max_output_tokens=settings.context_summary_max_output_tokens_large_fold,
    )
    runtime = AgentRuntime(
        registry,
        tools,
        provider=config["provider"],
        model=config["model"],
        system_prompt=request["system_prompt"],
        max_steps=config["model_calls"] + 1,
        max_tool_rounds=config["model_calls"] + 1,
        max_output_tokens=config["max_output_tokens"],
        context_manager=ContextManager(
            context_settings=settings, conversation_reducer=reducer
        ),
        run_budget_config=run_budget(config),
    )
    try:
        result = await runtime.run(
            request["user_template"].format(
                problem_statement=request["task"]["problem_statement"]
            ),
            event_handler=EventLog(),
        )
        meter.event("agent_result", result.model_dump(mode="json"))
        # Only an error on the call that ended the run can be infrastructure.
        # Failures the runtime recovered from (e.g. a failed context summary)
        # are reported, but never discard an otherwise finished attempt.
        fatal = (
            fatal_error_type(meter.stats)
            if result.stop_reason.value == "model_error"
            else None
        )
        return {
            "stop_reason": result.stop_reason.value,
            "error": result.error.model_dump(mode="json") if result.error else None,
            "model_error_type": fatal,
            "recovered_errors": recovered_errors(meter, fatal),
        }
    finally:
        await registry.close()


MEA_ACCEPT = "\n\n用户确认：已接受上述计划。现在按计划执行、验证并交付。"
MEA_AUTO_ANSWER = (
    "评测模式下无人值守，无法回答。请根据问题描述和仓库现状自行做出最合理的判断并继续。"
)


ROUND_PHASES = {
    "managing": "管理者规划下一步",
    "planned": "已分派",
    "executing": "执行者工作中",
    "executed": "执行完成",
    "auditing": "审查者检查中",
    "audited": "审查完成",
    "applied": "本轮结果已记录",
    "interrupted": "被中断",
    "abandoned": "已放弃",
}


async def watch_rounds(app, mea_id: str, meter: Meter) -> None:
    seen: dict[int, str] = {}
    while True:
        try:
            for item in await app.mea_store.rounds(mea_id):
                phase = item.phase.value
                if seen.get(item.index) != phase:
                    seen[item.index] = phase
                    kind = (
                        "" if item.kind.value == "normal" else f"[{item.kind.value}] "
                    )
                    extra = (
                        f"：{item.subtask[:80]}"
                        if phase == "planned" and item.subtask
                        else ""
                    )
                    meter.progress(
                        f"第 {item.index} 轮 {kind}{ROUND_PHASES.get(phase, phase)}"
                        f"{extra}（累计 {meter.stats['model_calls']} 次调用）"
                    )
        except Exception:
            pass
        await asyncio.sleep(5)


async def run_muharness_mea(
    request: dict, container: TaskContainer, meter: Meter, key: str
) -> dict:
    """Product long-task path: plan turn -> accept -> Manager/Executor/Auditor rounds.

    Everything goes through MuHarness's own Application wiring; only the model
    adapter (metered), the shell (task container) and approvals (auto) differ.
    """
    import app.application as application_module
    from app.application import Application
    from app.domain.memory.maintenance_models import MemoryMaintenanceConfig
    from app.domain.memory.reflection_models import MemoryReflectionConfig
    from app.domain.skill_learning.config import SkillLearningSettings
    from app.models.types import AgentMode
    from app.runtime.mea.runner import MeaStartError

    from .exec_shell import ContainerShellTool

    config = request["config"]
    data = meter.directory / "app"
    data.mkdir(exist_ok=True)
    # The product shell is replaced below; keep its sandbox fully offline anyway.
    application_module._sandbox_allowed_domains = lambda: ()
    app = Application(
        provider=config["provider"],
        model=config["model"],
        system_prompt=request["system_prompt"],
        database=data / "muharness.db",
        tasks_dir=data / "tasks",
        mcp_config=data / "mcp.json",
        memory_dir=data / "memory",
        skills_user_dir=data / "skills-user",
        skills_project_dir=data / "skills-project",
        max_steps=config["model_calls"] + 1,
        max_tool_rounds=config["model_calls"] + 1,
        max_output_tokens=config["max_output_tokens"],
        run_budget_config=run_budget(config),
        registry=measured_registry(config, meter, key),
        memory_reflection_config=MemoryReflectionConfig(_env_file=None, enabled=False),
        memory_maintenance_config=MemoryMaintenanceConfig(
            _env_file=None, enabled=False
        ),
        skill_learning_settings=SkillLearningSettings(
            _env_file=None, skill_learning_enabled=False
        ),
        web_approval=True,
        workspace_root=container.workspace,
    )
    twin = None
    summary: dict = {"mode": "mea"}
    try:
        await app.start()
        gate = app.web_approval_gate
        pending: set = set()

        async def auto_approve(method, params):
            meter.event("approval", {"method": method, "params": params})
            if method == "approval.required":
                task = asyncio.create_task(gate.approve(params["approval"].id))
                pending.add(task)
                task.add_done_callback(pending.discard)

        gate.set_broadcaster(auto_approve)
        twin = await asyncio.to_thread(container.read_only_twin)
        tools = app.tool_registry
        deferred = tools.is_deferred("run_shell_command")
        tools.unregister("run_shell_command")
        tools.register(ContainerShellTool(container, meter, twin), deferred=deferred)
        for name in ("http_request", "web_search", "web_fetch"):
            with contextlib.suppress(Exception):
                tools.unregister(name)
        meter.event("tools", sorted(tools.names()))

        content = request["user_template"].format(
            problem_statement=request["task"]["problem_statement"]
        )
        meter.progress("规划中（plan 模式）…")
        # Task tools need a persisted conversation (as the web UI creates one).
        conversation = await app.conversation_store.create(
            title=request["task"].get("instance_id") or "SWE-bench"
        )
        conversation_id = conversation.id
        plan = await app.conversation_service.dispatch(
            conversation_id=conversation_id, content=content, mode=AgentMode.PLAN
        )
        meter.event("plan_result", plan.result.model_dump(mode="json"))
        summary["plan_stop_reason"] = plan.result.stop_reason.value
        candidates = [
            t
            for t in await app.task_store.list_for_conversation(conversation_id)
            if t.steps and t.goal and t.status.value == "pending"
        ]
        if not candidates:
            fatal = fatal_error_type(meter.stats)
            return summary | {
                "stop_reason": "plan_missing",
                "error": "Plan mode produced no task with goal and steps",
                "model_error_type": fatal,
                "recovered_errors": recovered_errors(meter, fatal),
            }
        task = max(candidates, key=lambda t: t.updated_at)
        task = await app.task_store.plan_accept(task.id)
        meter.event("plan_task", task.model_dump(mode="json"))
        meter.progress(f"计划已接受：{len(task.steps)} 步，开始长任务")
        for step in task.steps:
            meter.progress(f"  - {step.title}")
        summary["plan_steps"] = len(task.steps)
        try:
            mea = await app.mea_runner.start(
                task_id=task.id,
                conversation_id=conversation_id,
                original_request=content + MEA_ACCEPT,
                round_budget=config["mea_rounds"],
            )
        except MeaStartError as exc:
            # e.g. a step without acceptance criteria: a planning outcome.
            return summary | {
                "stop_reason": "mea_start_error",
                "error": str(exc),
                "model_error_type": None,
                "recovered_errors": recovered_errors(meter, None),
            }
        watcher = asyncio.create_task(watch_rounds(app, mea.id, meter))
        answers = 0
        deadline = config["task_timeout_seconds"] - 120
        loop = asyncio.get_running_loop()
        started = loop.time()
        while True:
            remaining = deadline - (loop.time() - started)
            try:
                final = await asyncio.wait_for(
                    app.mea_runner.wait(mea.id), max(remaining, 1)
                )
            except TimeoutError:
                summary["wall_timeout"] = True
                final = await app.mea_runner.cancel(mea.id)
                break
            if final.status.value == "waiting_user" and answers < 3:
                answers += 1
                meter.event("auto_answer", {"question": final.pending_question})
                await app.mea_runner.answer(mea.id, MEA_AUTO_ANSWER)
                continue
            break
        watcher.cancel()
        rounds = await app.mea_store.rounds(mea.id)
        meter.progress(
            f"长任务结束：{final.status.value}，{len(rounds)} 轮，"
            f"{meter.stats['model_calls']} 次调用"
        )
        meter.event("mea_run", final.model_dump(mode="json"))
        meter.event("mea_rounds", [r.model_dump(mode="json") for r in rounds])
        status = final.status.value
        fatal = fatal_error_type(meter.stats) if status == "failed" else None
        return summary | {
            "stop_reason": f"mea_{status}",
            "error": final.abort_reason,
            "model_error_type": fatal,
            "recovered_errors": recovered_errors(meter, fatal),
            "mea_rounds": len(rounds),
            "auto_answers": answers,
            "final_response": final.final_response,
        }
    finally:
        with contextlib.suppress(Exception):
            await app.close()
        if twin is not None:
            twin.close()


def run_mini(request: dict, container: TaskContainer, meter: Meter, key: str) -> dict:
    # mini v2 native-tool-call agent with its installed official SWE-bench prompt.
    # The adapter performs ONE request per query: neither LiteLLM nor tenacity
    # retries may silently consume uncounted calls.
    import time

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import FormatError, Submitted
    from minisweagent.models.litellm_model import LitellmModel

    config = request["config"]

    class Environment:
        def execute(self, action, cwd="", timeout=None):
            result = container.execute(
                action["command"], cwd=cwd or "/testbed", timeout=timeout
            )
            meter.event("shell", {"action": action, "result": result})
            if result["returncode"] == 0 and result["output"].splitlines()[:1] == [
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            ]:
                raise Submitted(
                    {
                        "role": "exit",
                        "content": result["output"],
                        "extra": {
                            "exit_status": "Submitted",
                            "submission": "\n".join(result["output"].splitlines()[1:]),
                        },
                    }
                )
            return result

        def get_template_vars(self):
            import platform

            return (
                request["mini_config"].get("environment", {})
                | platform.uname()._asdict()
            )

        def serialize(self):
            return {
                "info": {
                    "config": {
                        "environment": self.get_template_vars(),
                        "environment_type": "MuHarnessSharedOfflineContainer",
                    }
                }
            }

    class MeasuredMiniModel(LitellmModel):
        def _query(self, messages, **kwargs):
            return super()._query(messages, api_key=key, **kwargs)

        def query(self, messages, **kwargs):
            prepared = self._prepare_messages_for_api(messages)
            retries = config["transient_retries"]
            for attempt in range(retries + 1):
                meter.start(
                    {
                        "messages": prepared,
                        "model": config["model"],
                        "max_output_tokens": config["max_output_tokens"],
                    },
                    retry=attempt,
                )
                try:
                    response = self._query(prepared, **kwargs)
                    break
                except Exception as exc:
                    retry = attempt < retries and is_transient(exc)
                    meter.error(exc, will_retry=retry)
                    if not retry:
                        raise
                    meter.progress(f"等待 {retry_delay(config, attempt):g} 秒后重试")
                    time.sleep(retry_delay(config, attempt))
            usage = response.usage
            details = getattr(usage, "prompt_tokens_details", None)
            meter.finish(
                response.model_dump(),
                None
                if usage is None
                else {
                    "input_tokens": usage.prompt_tokens or 0,
                    "output_tokens": usage.completion_tokens or 0,
                    "cached_input_tokens": getattr(details, "cached_tokens", 0) or 0,
                },
            )
            try:
                actions = self._parse_actions(response)
            except FormatError as exc:
                exc.messages[0].setdefault("extra", {}).update(
                    {"cost": 0.0, "response": response.model_dump()}
                )
                raise
            return response.choices[0].message.model_dump() | {
                "extra": {
                    "actions": actions,
                    "response": response.model_dump(),
                    "cost": 0.0,
                    "timestamp": time.time(),
                }
            }

    model_config = dict(request["mini_config"].get("model", {}))
    for field in (
        "model_class",
        "model_name",
        "model_kwargs",
        "litellm_model_registry",
        "cost_tracking",
    ):
        model_config.pop(field, None)
    model = MeasuredMiniModel(
        **model_config,
        model_name="openai/" + config["model"],
        cost_tracking="ignore_errors",
        model_kwargs={
            "api_base": config["base_url"],
            "temperature": config["temperature"],
            "max_tokens": config["max_output_tokens"],
            "timeout": config["model_timeout_seconds"],
            "num_retries": 0,
            "extra_body": config["extra_body"],
            "parallel_tool_calls": True,
        },
    )
    agent_config = dict(request["mini_config"]["agent"]) | {
        "step_limit": config["model_calls"],
        "cost_limit": 0,
        "wall_time_limit_seconds": config["task_timeout_seconds"],
        "output_path": meter.directory / "mini-trajectory.json",
    }
    agent = DefaultAgent(model, Environment(), **agent_config)
    try:
        info = agent.run(request["task"]["problem_statement"])
    except Exception as exc:
        # DefaultAgent saved the exception and trajectory before raising.
        fatal = fatal_error_type(meter.stats)
        return {
            "stop_reason": type(exc).__name__,
            "error": str(exc),
            "model_error_type": fatal,
            "recovered_errors": recovered_errors(meter, fatal),
        }
    return {
        "stop_reason": info.get("exit_status", "unknown"),
        "model_error_type": None,
        "recovered_errors": recovered_errors(meter, None),
    }


def main() -> int:
    request = read_json(Path(sys.argv[1]))
    directory = Path(sys.argv[1]).parent
    config = request["config"]
    meter = Meter(directory, config["model_calls"])
    result = {"stop_reason": "worker_error"}
    try:
        key = credential(config)
        meter.secret = key
        container = TaskContainer(
            request["container"],
            Path(request["workspace"]),
            config["command_timeout_seconds"],
        )
        if request["variant"] == "mini":
            result = run_mini(request, container, meter, key)
        elif config["agent_mode"] == "mea":
            result = asyncio.run(run_muharness_mea(request, container, meter, key))
        else:
            result = asyncio.run(run_muharness(request, container, meter, key))
    except Exception as exc:
        result = {
            "stop_reason": "worker_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        meter.event("worker_error", result)
    finally:
        write_json(directory / "worker-result.json", json.loads(meter.clean(result)))
        meter.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
