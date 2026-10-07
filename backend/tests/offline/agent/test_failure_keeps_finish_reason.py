"""失败结束的运行也要带出最后一次模型回复的结束原因，长任务据此识别输出截断。"""

from __future__ import annotations

import pytest

from app.models.types import AgentMode, ToolCall
from app.runtime.agent.result import AgentStopReason
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.context import ContextManager, ModelCapabilityRegistry
from app.runtime.mea.runner import SubRunResult
from app.tools.builtin.read_file import ReadFileTool
from app.tools.registry import ToolRegistry
from tests.offline.agent.test_agent_runtime import (
    CountingTool,
    DeterministicTokenEstimator,
    fake_registry,
    model_response,
)


def _truncated_empty():
    return model_response(content=None).model_copy(
        update={"finish_reason": "max_tokens"}
    )


def _runtime(registry, tools=None, **kwargs) -> AgentRuntime:
    return AgentRuntime(
        registry,
        tools or ToolRegistry(),
        provider="fake",
        context_manager=ContextManager(estimator=DeterministicTokenEstimator()),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retry_content", "retry_finish", "expected_ok", "unresolved"),
    [
        (None, "max_tokens", False, True),
        (None, "end_turn", False, True),
        ("full report", "end_turn", True, False),
        ("partial report", "max_tokens", True, True),
    ],
)
async def test_empty_reply_during_forced_finalization_keeps_truncation(
    tmp_path, retry_content, retry_finish, expected_ok, unresolved,
) -> None:
    # 结束原因与“报告是否补全”独立；空 end_turn 不能清除尚未补全的截断。
    (tmp_path / "proof.txt").write_text("verified evidence", encoding="utf-8")
    registry, adapter = fake_registry(
        [
            model_response(
                tool_calls=(
                    ToolCall(
                        id="read", name="read_file", arguments={"path": "proof.txt"}
                    ),
                )
            ),
            model_response(content="状态: complete\npartial").model_copy(
                update={"finish_reason": "max_tokens"}
            ),
            model_response(content=retry_content).model_copy(
                update={"finish_reason": retry_finish}
            ),
        ]
    )
    capabilities = ModelCapabilityRegistry()
    capabilities.register_override("fake", "fake-model", max_output_tokens=16384)
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))

    result = await AgentRuntime(
        registry,
        tools,
        provider="fake",
        max_steps=3,
        context_manager=ContextManager(
            registry=capabilities, estimator=DeterministicTokenEstimator()
        ),
    ).run("audit", mode=AgentMode.AUDIT)

    assert len(adapter.requests) == 3
    assert result.ok is expected_ok
    if not expected_ok:
        assert result.stop_reason is AgentStopReason.MODEL_ERROR
        assert "forced finalization" in str(result.error)
    assert result.model_finish_reason == retry_finish
    assert result.unresolved_output_truncation is unresolved
    sub = SubRunResult(
        "r",
        "",
        ok=result.ok,
        cancelled=False,
        model_finish_reason=result.model_finish_reason,
        unresolved_output_truncation=result.unresolved_output_truncation,
    )
    assert sub.truncated is unresolved


@pytest.mark.asyncio
async def test_repeated_truncated_empty_replies_keep_truncation() -> None:
    registry, _ = fake_registry(
        [_truncated_empty(), _truncated_empty(), _truncated_empty()]
    )
    result = await _runtime(registry).run("work")
    assert result.ok is False
    assert result.model_finish_reason == "max_tokens"


@pytest.mark.asyncio
async def test_model_error_does_not_reuse_previous_finish_reason() -> None:
    # 第二次调用抛异常：不能把上一次的结束原因误当成这次失败的原因
    registry, _ = fake_registry(
        [
            model_response(
                tool_calls=(ToolCall(id="c1", name="count", arguments={"value": 1}),)
            ).model_copy(update={"finish_reason": "max_tokens"}),
            RuntimeError("provider down"),
        ]
    )
    tools = ToolRegistry()
    tools.register(CountingTool())
    result = await _runtime(registry, tools).run("work")
    assert result.ok is False
    assert result.model_finish_reason is None
    assert result.unresolved_output_truncation is False
