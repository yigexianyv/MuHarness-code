
from datetime import UTC, datetime

from app.models.types import Message, MessageRole, ModelUsage
from app.records.trace import summarize_run_usage
from app.runtime.agent.events import AgentEvent, AgentEventType
from app.runtime.agent.result import AgentResult, AgentStopReason


# 函数说明：_event
# 用途：返回 `AgentEvent(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   sequence：事件或记录顺序号，类型 `int`。
#   event_type：事件输入或配置值，类型 `AgentEventType`。
#   usage：模型调用用量统计，类型 `ModelUsage | None`；默认 `None`。
# 返回：类型 `AgentEvent`；返回 `AgentEvent(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentEvent` → `datetime`。
def _event(
    sequence: int,
    event_type: AgentEventType,
    *,
    usage: ModelUsage | None = None,
) -> AgentEvent:
    return AgentEvent(
        run_id="run-usage",
        sequence=sequence,
        type=event_type,
        event_time=datetime(2026, 8, 22, tzinfo=UTC),
        step=sequence if event_type is AgentEventType.MODEL_COMPLETED else None,
        usage=usage,
    )


# 函数说明：test_summarizes_main_post_run_and_provider_total
# 用途：回归验证回归测试与测试辅助中的 `summarizes_main_post_run_and_provider_total` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelUsage` → `Message` →
# `AgentResult` → `AgentEvent` → `datetime` → `_event`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`summary.main_agent == main`。
#   验证条件：`summary.context_summary.total_tokens == 20`。
#   验证条件：`summary.context_summary_status == 'completed'`。
#   验证条件：`summary.context_summary_provider == 'qwen'`。
def test_summarizes_main_post_run_and_provider_total() -> None:
    first = ModelUsage(
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        cached_input_tokens=60,
        uncached_input_tokens=40,
        cache_read_input_tokens=60,
        model_calls=1,
    )
    second = ModelUsage(
        input_tokens=120,
        output_tokens=20,
        total_tokens=140,
        cached_input_tokens=80,
        uncached_input_tokens=40,
        cache_read_input_tokens=80,
        model_calls=1,
    )
    main = ModelUsage(
        input_tokens=220,
        output_tokens=30,
        total_tokens=250,
        cached_input_tokens=140,
        uncached_input_tokens=80,
        cache_read_input_tokens=140,
        model_calls=2,
    )
    final_message = Message(role=MessageRole.ASSISTANT, content="完成")
    result = AgentResult(
        run_id="run-usage",
        final_message=final_message,
        messages=(final_message,),
        steps=2,
        stop_reason=AgentStopReason.FINAL_ANSWER,
        usage=main,
    )
    events = (
        AgentEvent(
            run_id="run-usage",
            sequence=0,
            type=AgentEventType.MODEL_STARTED,
            event_time=datetime(2026, 8, 22, tzinfo=UTC),
            step=1,
            tool_schema_tokens=50,
            summary_usage=ModelUsage(
                input_tokens=15,
                output_tokens=5,
                total_tokens=20,
                model_calls=1,
            ),
            summary_updated=True,
            summary_provider="qwen",
            summary_model="qwen-summary",
            summary_duration_ms=125.0,
        ),
        _event(1, AgentEventType.MODEL_COMPLETED, usage=first),
        AgentEvent(
            run_id="run-usage",
            sequence=2,
            type=AgentEventType.MODEL_STARTED,
            event_time=datetime(2026, 8, 22, tzinfo=UTC),
            step=2,
            tool_schema_tokens=70,
            run_budget_status="warning",
            run_budget_reason="tokens",
            run_budget_chargeable_tokens=100,
            run_budget_model_calls=1,
            run_budget_warning_tokens=100,
            run_budget_finalization_tokens=200,
            run_budget_hard_tokens=300,
            run_budget_warning_model_calls=4,
            run_budget_finalization_model_calls=6,
            run_budget_hard_model_calls=8,
        ),
        _event(3, AgentEventType.MODEL_COMPLETED, usage=second),
        AgentEvent(
            run_id="run-usage",
            sequence=4,
            type=AgentEventType.AGENT_COMPLETED,
            event_time=datetime(2026, 8, 22, tzinfo=UTC),
            step=2,
            result=result,
            usage=main,
            stop_reason=AgentStopReason.FINAL_ANSWER,
        ),
        _event(
            5,
            AgentEventType.MEMORY_REFLECTION_COMPLETED,
            usage=ModelUsage(
                input_tokens=30,
                output_tokens=5,
                total_tokens=35,
                cached_input_tokens=10,
                uncached_input_tokens=20,
                cache_read_input_tokens=10,
                model_calls=1,
            ),
        ),
        _event(
            6,
            AgentEventType.MEMORY_MAINTENANCE_COMPLETED,
            usage=ModelUsage(input_tokens=15, output_tokens=2, total_tokens=17),
        ),
    )

    summary = summarize_run_usage(events)

    assert summary.main_agent == main
    assert summary.context_summary.total_tokens == 20
    assert summary.context_summary_status == "completed"
    assert summary.context_summary_provider == "qwen"
    assert summary.context_summary_model == "qwen-summary"
    assert summary.context_summary_duration_ms == 125.0
    assert summary.memory_reflection.total_tokens == 35
    assert summary.memory_maintenance.model_calls == 1
    assert summary.provider_total.total_tokens == 322
    assert summary.provider_total.model_calls == 5
    assert summary.provider_total.cached_input_tokens is None
    assert summary.tool_schema_tokens_estimated == 120
    assert summary.main_agent_chargeable_tokens == 130
    assert summary.run_budget_status == "warning"
    assert summary.run_budget_reason == "tokens"
    assert summary.run_budget_hard_tokens == 300


# 函数说明：test_reflection_skip_reason_is_visible_without_usage
# 用途：回归验证回归测试与测试辅助中的 `reflection_skip_reason_is_visible_without_usage`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentEvent` → `datetime` →
# `summarize_run_usage`。
# 分支与异常：
#   验证条件：`summary.memory_reflection_status == 'skipped'`。
#   验证条件：`summary.memory_reflection_skip_reason == 'gate:smalltalk'`。
#   验证条件：`summary.memory_reflection.total_tokens == 0`。
def test_reflection_skip_reason_is_visible_without_usage() -> None:
    skipped = AgentEvent(
        run_id="run-usage",
        sequence=1,
        type=AgentEventType.MEMORY_REFLECTION_SKIPPED,
        event_time=datetime(2026, 8, 22, tzinfo=UTC),
        reflection_triggered=False,
        reflection_skip_reason="gate:smalltalk",
    )

    summary = summarize_run_usage((skipped,))

    assert summary.memory_reflection_status == "skipped"
    assert summary.memory_reflection_skip_reason == "gate:smalltalk"
    assert summary.memory_reflection.total_tokens == 0


# 函数说明：test_old_trace_infers_calls_without_inventing_cache_breakdown
# 用途：回归验证回归测试与测试辅助中的
# `old_trace_infers_calls_without_inventing_cache_breakdown` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`summarize_run_usage` → `_event` →
# `ModelUsage`。
# 分支与异常：
#   验证条件：`summary.main_agent.model_calls == 1`。
#   验证条件：`summary.main_agent.cached_input_tokens is None`。
#   验证条件：`summary.provider_total.total_tokens == 15`。
def test_old_trace_infers_calls_without_inventing_cache_breakdown() -> None:
    summary = summarize_run_usage(
        (_event(1, AgentEventType.MODEL_COMPLETED, usage=ModelUsage(
            input_tokens=12,
            output_tokens=3,
            total_tokens=15,
        )),)
    )

    assert summary.main_agent.model_calls == 1
    assert summary.main_agent.cached_input_tokens is None
    assert summary.provider_total.total_tokens == 15


# 函数说明：test_chargeable_tokens_are_summed_per_call_when_cache_detail_is_mixed
# 用途：回归验证回归测试与测试辅助中的
# `chargeable_tokens_are_summed_per_call_when_cache_detail_is_mixed` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_event` → `ModelUsage` → `AgentEvent`
#  → `datetime` → `summarize_run_usage`。
# 分支与异常：
#   验证条件：`summary.main_agent.cached_input_tokens == 80`。
#   验证条件：`summary.context_summary.total_tokens == 12`。
#   验证条件：`summary.provider_total.total_tokens == 117`。
#   验证条件：`summary.main_agent_chargeable_tokens == 37`。
def test_chargeable_tokens_are_summed_per_call_when_cache_detail_is_mixed() -> None:
    events = (
        _event(
            1,
            AgentEventType.MODEL_COMPLETED,
            usage=ModelUsage(
                input_tokens=100,
                output_tokens=5,
                total_tokens=105,
                cached_input_tokens=80,
                uncached_input_tokens=20,
                model_calls=1,
            ),
        ),
        AgentEvent(
            run_id="run-usage",
            sequence=2,
            type=AgentEventType.MODEL_STARTED,
            event_time=datetime(2026, 8, 22, tzinfo=UTC),
            step=2,
            summary_usage=ModelUsage(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
                model_calls=1,
            ),
        ),
    )

    summary = summarize_run_usage(events)

    assert summary.main_agent.cached_input_tokens == 80
    assert summary.context_summary.total_tokens == 12
    assert summary.provider_total.total_tokens == 117
    assert summary.main_agent_chargeable_tokens == 37
