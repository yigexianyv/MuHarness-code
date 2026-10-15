"""Positive and false-pass cases for the v2 supplement. No model API calls."""

import hashlib
import json

import pytest

from app.models.types import ToolCall
from tests.eval.checks import evaluate, validate_spec
from tests.eval.drive import _snapshot_file_hashes
from tests.eval.outcome import ApprovalUse, ContextStep, Outcome, ToolUse, TurnOutcome
from tests.eval.selftest.scripted import (
    after_tool,
    answer,
    call,
    last_user,
    scripted_factory,
)
from tests.eval.session import run_cases
from tests.eval.spec import CURRENT, load_cases


def verdict(spec, *, tools=(), **fields):
    outcome = Outcome(
        case_id="supplement", variant="current", attempt=1,
        turns=[TurnOutcome(say="task", tools=list(tools))], **fields,
    )
    return evaluate(outcome, [spec])[0].passed


@pytest.mark.parametrize("order, expected", [
    (("read_file", "write_file"), True),
    (("read_file", "edit_file"), True),
    (("write_file", "read_file", "write_file"), False),
    (("read_file",), False),
])
def test_read_before_first_mutation(order, expected):
    tools = [ToolUse(name=name, arguments={"path": "src/calc.py"}, success=True, round_index=i)
             for i, name in enumerate(order, 1)]
    assert verdict({"read_before_write": {"path": "src/calc.py"}}, tools=tools) is expected


@pytest.mark.parametrize("read_path, read_success, read_round, expected", [
    ("src/./calc.py", True, 1, True),
    ("src/other.py", True, 1, False),
    ("src/calc.py", False, 1, False),
    ("src/calc.py", True, 2, False),  # Same response: model has not seen the result.
    ("src/calc.py", True, None, False),
])
def test_read_must_be_successful_same_file_and_previous_model_round(
    read_path, read_success, read_round, expected,
):
    tools = [
        ToolUse(name="read_file", arguments=json.dumps({"path": read_path}),
                success=read_success, round_index=read_round),
        ToolUse(name="write_file", arguments={"path": "src/calc.py"}, success=True, round_index=2),
    ]
    assert verdict({"read_before_write": {"path": "src/calc.py"}}, tools=tools) is expected


@pytest.mark.parametrize("after", [b"# tests removed\n", b"line\n", b"\xef\xbb\xbfline\r\n"])
def test_equals_original_compares_bytes_not_normalized_text(tmp_path, after):
    target = tmp_path / "test.py"
    target.write_bytes(b"line\r\n")
    initial = _snapshot_file_hashes(tmp_path)
    assert verdict({"file": {"path": "test.py", "equals_original": True}},
                   initial_file_hashes=initial, file_hashes=initial)
    target.write_bytes(after)
    assert not verdict({"file": {"path": "test.py", "equals_original": True}},
                       initial_file_hashes=initial, file_hashes=_snapshot_file_hashes(tmp_path))


def test_hash_checks_support_binary_and_fail_on_missing_evidence():
    digest = hashlib.sha256(b"\xff\x00").hexdigest()
    assert verdict({"file": {"path": "a.bin", "sha256": digest}}, file_hashes={"a.bin": digest})
    assert not verdict({"file": {"path": "a.bin", "equals_original": True}},
                       file_hashes={"a.bin": digest})
    assert not verdict({"file": {"path": "a.bin", "equals_original": True}},
                       initial_file_hashes={"a.bin": digest})
    with pytest.raises(ValueError, match="未知 file"):
        validate_spec({"file": {"path": "a.bin", "equals_origina": True}})


@pytest.mark.parametrize("count, updated, conversation, expected", [
    (8, True, "source", True),
    (7, True, "source", False),  # Covered count is an exclusive upper bound.
    (8, False, "source", False),
    (8, True, "other", False),
    (None, True, "source", False),
])
def test_summary_coverage_uses_message_position_and_conversation(count, updated, conversation, expected):
    outcome = Outcome(case_id="summary", variant="current", attempt=1, turns=[
        TurnOutcome(say="early fact", conversation_id="source", user_sequence=7),
        TurnOutcome(say="later", conversation_id=conversation, context=[
            ContextStep(summary_updated=updated, summary_covered_after=count),
        ]),
    ])
    assert evaluate(outcome, [{"summary_covers_turn": 1}])[0].passed is expected


def test_grounding_requires_actual_received_output_not_only_correct_answer():
    outcome = Outcome(case_id="facts", variant="current", attempt=1, turns=[
        TurnOutcome(say="read", answer="E4721 A7731", received_tool_outputs=["INFO only"]),
    ])
    spec = {"grounded_answer": {"all": ["E4721", "A7731"]}}
    assert not evaluate(outcome, [spec])[0].passed
    outcome.turns[0].received_tool_outputs = ["ERROR E4721 order=A7731"]
    assert evaluate(outcome, [spec])[0].passed
    outcome.turns[0].answer = "I do not know"
    assert not evaluate(outcome, [spec])[0].passed


@pytest.mark.parametrize("text, expected", [
    ("123 × 456 = **56,088**。", True),
    ("答案 56088", True),
    ("答案 56,088.00", True),
    ("答案 56089", False),
    ("答案 156088", False),
    ("答案 560880", False),
    ("答案 56088.1", False),
    ("答案 -56088", False),
    ("答案 56,08,8", False),
    ("答案 .56088", False),
])
def test_arithmetic_check_compares_whole_numbers(text, expected):
    outcome = Outcome(case_id="numbers", variant="current", attempt=1,
                      turns=[TurnOutcome(say="calculate", answer=text)])
    assert evaluate(outcome, [{"answer_number": 56088}])[0].passed is expected


@pytest.mark.parametrize("error, expected", [
    ("Tool execution failed: FileNotFoundError: missing", True),
    ("TimeoutError", False),
    (None, False),
])
def test_missing_file_requires_the_actual_missing_file_error(error, expected):
    assert verdict({"tool_used": {"name": "read_file", "success": False,
                                  "error_contains": ["FileNotFoundError"]}},
                   tools=[ToolUse(name="read_file", success=False, error=error)]) is expected


def test_read_in_other_conversation_does_not_justify_writing():
    outcome = Outcome(case_id="isolated", variant="current", attempt=1, turns=[
        TurnOutcome(say="read", conversation_id="other", tools=[ToolUse(
            name="read_file", arguments={"path": "a.txt"}, success=True, round_index=1,
        )]),
        TurnOutcome(say="write", conversation_id="current", tools=[ToolUse(
            name="write_file", arguments={"path": "a.txt"}, success=True, round_index=1,
        )]),
    ])
    assert not evaluate(outcome, [{"read_before_write": {"path": "a.txt", "turn": "all"}}])[0].passed
    outcome.turns[0].conversation_id = "current"
    assert evaluate(outcome, [{"read_before_write": {"path": "a.txt", "turn": "all"}}])[0].passed


@pytest.mark.parametrize("command, exit_code, decision_call, expected", [
    ("pytest tests/test_price.py", 1, "delete", True),
    ("pytest tests/other.py", 1, "delete", False),
    ("pytest tests/test_price.py", 0, "delete", False),
    ("pytest tests/test_price.py", None, "delete", False),
    ("pytest tests/test_price.py", 1, "unrelated", False),
])
def test_tool_check_links_arguments_exit_code_and_approval(command, exit_code, decision_call, expected):
    outcome = Outcome(case_id="shell", variant="current", attempt=1, turns=[TurnOutcome(
        say="test", tools=[ToolUse(name="run_shell_command", call_id="delete", success=True,
                                   arguments={"command": command}, exit_code=exit_code)],
        approval_calls=[ApprovalUse(call_id=decision_call, decision="denied")],
    )])
    spec = {"tool_used": {"name": "run_shell_command", "exit_code": 1,
                         "arguments_contains": {"command": ["tests/test_price.py"]},
                         "approval_decision": "denied"}}
    assert evaluate(outcome, [spec])[0].passed is expected


@pytest.mark.parametrize("fetch_middle", [False, True])
async def test_real_request_view_is_required_for_middle_fact(fetch_middle):
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return answer(calls=(ToolCall(id="whole", name="read_file", arguments={"path": "logs/app.log"}),))
        if calls == 2 and fetch_middle:
            return answer(calls=(ToolCall(id="page", name="read_file", arguments={
                "path": "logs/app.log", "start_line": 1501, "max_lines": 1,
            }),))
        return answer("错误码 E4721，订单 A7731。")

    case = load_cases(ids=["b20-middle-detail"])[0]
    report = await run_cases([case], variant=CURRENT, factory=scripted_factory(reply), keep_outcomes=True)
    assert report.attempts[0].passed is fetch_middle, report.attempts[0].verdicts


async def test_denied_delete_collects_matching_approval_and_original_hash():
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return answer(calls=(ToolCall(id="delete", name="run_shell_command", arguments={
                "command": "python -c \"from pathlib import Path; Path('tmp/protected.txt').unlink()\"",
            }),))
        return answer("审批被拒绝，未执行。")

    case = load_cases(ids=["b13-shell-denied"])[0]
    report = await run_cases([case], variant=CURRENT, factory=scripted_factory(reply))
    assert report.attempts[0].passed, report.attempts[0].verdicts


async def test_summary_case_actually_covers_first_user_message():
    summaries = 0

    def reply(request):
        nonlocal summaries
        if any((message.content or "").startswith("# 会话摘要规范") for message in request.messages):
            summaries += 1
            return answer(json.dumps({
                "current_objective": "回答最初的项目代号和交付日期",
                "user_constraints": [], "key_decisions": [], "completed_work": [],
                "current_state": [], "pending_work": [],
                "important_facts": ["项目青鸟-27，交付日期 11 月 18 日"],
            }, ensure_ascii=False))
        user = last_user(request)
        if "回到最开始" in user:
            return answer("青鸟-27，11月18日。")
        for name in ("a", "b", "c"):
            if f"docs/{name}.md" in user:
                return answer("已读取资料") if after_tool(request) else call("read_file", path=f"docs/{name}.md")
        return answer("收到")

    case = load_cases(ids=["c04-summary-keeps-facts"])[0]
    report = await run_cases([case], variant=CURRENT, factory=scripted_factory(reply))
    assert report.attempts[0].passed, report.attempts[0].verdicts
    assert summaries > 0
