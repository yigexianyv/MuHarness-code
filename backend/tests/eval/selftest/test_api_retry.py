"""接口故障恢复需可观察，不将任务失败重试成通过。"""

import json

import pytest
from pydantic import SecretStr

from app.models.config import ProviderConfig
from app.models.errors import ModelAdapterError
from app.models.types import ApiStyle, Message, MessageRole, ModelRequest
from tests.eval.retry import RetryingAdapter, error_kind
from tests.eval.selftest.scripted import ScriptedAdapter, answer, call, scripted_factory
from tests.eval.transport import NonStreamingAdapter
from tests.eval.v1 import run_trial
from tests.eval.v1_cases import v1_cases

GROUP_ERROR = "Error code: 403 - No active subscription found for this group"


def config():
    return ProviderConfig(
        provider="fake",
        model="fake",
        api_key=SecretStr("secret-123456789"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
        max_retries=0,
    )


def request():
    return ModelRequest(messages=(Message(role=MessageRole.USER, content="synthetic"),))


@pytest.mark.parametrize(
    "error,expected",
    [
        (ModelAdapterError(GROUP_ERROR), (403, True)),
        (ModelAdapterError("Error code: 403 - forbidden"), (403, False)),
        (ModelAdapterError("Error code: 401 - Invalid token"), (401, False)),
        (ModelAdapterError("Error code: 400 - invalid schema"), (400, False)),
        (ModelAdapterError("Error code: 429 - limit"), (429, True)),
        (ModelAdapterError("Error code: 503 - unavailable"), (503, True)),
        (TimeoutError("timeout"), (None, True)),
        (ValueError("invalid response"), (None, False)),
    ],
)
def test_only_supported_interface_errors_are_retried(error, expected):
    assert error_kind(error) == expected


def test_wrapped_timeout_is_retryable():
    error = ModelAdapterError("adapter failure")
    error.__cause__ = TimeoutError()
    assert error_kind(error) == (None, True)


async def test_identical_request_recovers_with_redacted_failure_record():
    count = 0

    def reply(req):
        nonlocal count
        count += 1
        if count < 3:
            raise ModelAdapterError(GROUP_ERROR + " secret-123456789")
        return answer("OK")

    delegate = ScriptedAdapter(config(), reply)
    adapter = RetryingAdapter(delegate, max_retries=3, base_delay=0)
    await adapter.complete(request())
    assert len(delegate.requests) == 3
    assert len({id(req) for req in delegate.requests}) == 1
    assert [row["status"] for row in adapter.events] == ["error", "error", "ok"]
    assert len({row["request_sha256"] for row in adapter.events}) == 1
    assert len({row["call_id"] for row in adapter.events}) == 1
    assert "secret-123456789" not in json.dumps(adapter.events)
    assert "[REDACTED]" in adapter.events[0]["error"]


async def test_explicit_nonstream_option_records_actual_transport():
    delegate = NonStreamingAdapter(ScriptedAdapter(config(), lambda _: answer("OK")))
    adapter = RetryingAdapter(delegate)

    async def unexpected_delta(text):
        raise AssertionError("nonstream has no deltas")

    await adapter.complete_stream(request(), on_text_delta=unexpected_delta)
    assert adapter.events[0]["transport"] == "non_stream"


@pytest.mark.parametrize(
    "message,count", [(GROUP_ERROR, 4), ("Error code: 401 - Invalid token", 1)]
)
async def test_retry_limit_and_stable_auth_failure(message, count):
    def reply(req):
        raise ModelAdapterError(message)

    delegate = ScriptedAdapter(config(), reply)
    adapter = RetryingAdapter(delegate, max_retries=3, base_delay=0)
    with pytest.raises(ModelAdapterError):
        await adapter.complete(request())
    assert len(delegate.requests) == count
    assert adapter.events[-1]["will_retry"] is False


@pytest.mark.parametrize("reasoning", [False, True])
async def test_visible_stream_is_not_replayed(reasoning):
    class PartialAdapter(ScriptedAdapter):
        async def complete_stream(self, req, *, on_text_delta, on_reasoning_delta=None):
            self.requests.append(req)
            callback = on_reasoning_delta if reasoning else on_text_delta
            await callback("partial")
            raise ModelAdapterError(GROUP_ERROR)

    delegate = PartialAdapter(config(), lambda _: answer("unused"))
    adapter = RetryingAdapter(delegate, max_retries=3, base_delay=0)
    deltas = []

    async def receive(text):
        deltas.append(text)

    with pytest.raises(ModelAdapterError):
        await adapter.complete_stream(
            request(), on_text_delta=receive, on_reasoning_delta=receive
        )
    assert len(delegate.requests) == 1
    assert deltas == ["partial"]
    assert adapter.events[0]["visible_output"] is True


def retrying_script(reply):
    base = scripted_factory(reply)

    def build(paths, variant):
        app = base(paths, variant)
        delegate = app.registry.get("fake")
        app.registry.register(
            "fake",
            lambda _: RetryingAdapter(delegate, base_delay=0),
            replace=True,
        )
        return app

    return build


async def test_recovered_case_counts_errors_and_executes_tool_once(tmp_path):
    count = 0

    def reply(req):
        nonlocal count
        count += 1
        if count <= 2:
            raise ModelAdapterError(GROUP_ERROR)
        if req.messages[-1].role is MessageRole.TOOL:
            return answer('{"owner":"林舟","version":7}')
        return call("read_file", path="project.txt")

    plan = next(plan for plan in v1_cases() if plan.case.id == "B01")
    row = await run_trial(
        plan, factory=retrying_script(reply), folder=tmp_path / "trial", attempt=1
    )
    assert row["status"] == "PASS"
    assert row["model_calls"] == row["api_requests"] == 4
    assert row["api_errors"] == row["api_retries"] == 2
    assert row["api_recovered_calls"] == 1
    assert row["usage_complete"] is False
    evidence = json.loads(
        (tmp_path / "trial/evidence.json").read_text(encoding="utf-8")
    )
    completed_tools = [
        event
        for run in evidence["runs"]
        for event in run["events"]
        if event["type"] == "tool_completed"
    ]
    assert len(completed_tools) == 1
    assert len(evidence["api_requests"]) == 4


async def test_exhausted_case_stays_blocked(tmp_path):
    def reply(req):
        raise ModelAdapterError(GROUP_ERROR)

    plan = next(plan for plan in v1_cases() if plan.case.id == "B01")
    row = await run_trial(
        plan, factory=retrying_script(reply), folder=tmp_path / "blocked", attempt=1
    )
    assert row["status"] == "BLOCKED"
    assert row["model_calls"] == row["api_requests"] == 4
    assert row["api_recovered_calls"] == 0


async def test_wrong_answer_is_not_sampled_again(tmp_path):
    plan = next(plan for plan in v1_cases() if plan.case.id == "B01")
    row = await run_trial(
        plan,
        factory=retrying_script(lambda _: answer('{"owner":"错误","version":0}')),
        folder=tmp_path / "wrong",
        attempt=1,
    )
    assert row["status"] == "FAIL"
    assert row["api_requests"] == 1
    assert row["api_retries"] == 0
