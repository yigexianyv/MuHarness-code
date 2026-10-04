"""验收器不能凭最终正确、模拟重启或被跳过的返工宣称覆盖。"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from tests.eval.outcome import MeaState, Outcome, TurnOutcome
from tests.eval.selftest.test_v2_report import report
from tests.eval.v1 import aggregate
from tests.eval.v2_cases import v2_cases
from tests.eval.v2_driver import inject_before_audit
from tests.eval.v2_grading import audit_role_check, grade_v2, observed_unittest_failure
from tests.eval.v2_report import compare, gate


def selected(case_id):
    return next(plan for plan in v2_cases() if plan.case.id == case_id)


def test_correct_file_and_answer_do_not_prove_real_restart():
    outcome = Outcome(
        case_id="S03",
        variant="current",
        attempt=1,
        turns=[TurnOutcome(say="answer", answer='{"project":"PULSAR","budget":73000}')],
    )
    checks = grade_v2(selected("S03"), outcome, {"runs": []})
    assert aggregate(checks) != "PASS"


def test_final_correct_program_does_not_prove_audit_rework():
    outcome = Outcome(
        case_id="L04",
        variant="current",
        attempt=1,
        diagnostics={
            "injections": [{"kind": "artifact_corruption", "round_index": 1}],
            "mea_rounds": [
                {"index": 1, "step_acceptance": "satisfied"},
                {"index": 2, "executor_run_id": "e2", "step_acceptance": "satisfied"},
            ],
        },
    )
    checks = grade_v2(selected("L04"), outcome, {"runs": []})
    assert aggregate(checks) == "FAIL"


def test_only_successful_final_test_does_not_prove_failure_recovery():
    outcome = Outcome(case_id="C03", variant="current", attempt=1)
    evidence = {
        "runs": [
            {
                "events": [
                    {
                        "type": "tool_completed",
                        "tool_call": {
                            "id": "c",
                            "arguments": {"command": "python -m unittest"},
                        },
                        "tool_result": {
                            "tool_call_id": "c",
                            "tool_name": "run_shell_command",
                            "success": True,
                            "output": json.dumps(
                                {
                                    "exit_code": 0,
                                    "stdout": "Ran 1 test\nOK",
                                    "stderr": "",
                                }
                            ),
                        },
                    }
                ]
            }
        ]
    }
    assert aggregate(grade_v2(selected("C03"), outcome, evidence)) == "FAIL"


@pytest.mark.parametrize(
    ("command", "exit_code", "stdout", "stderr", "expected"),
    [
        (
            "python -m unittest",
            1,
            "",
            "Ran 1 test in 0.001s\nFAILED (failures=1)\n",
            True,
        ),
        (
            "python -m unittest discover -s . -p 'test_*.py'; echo \"EXIT=$?\"",
            0,
            "EXIT=1\n",
            "Ran 1 test in 0.001s\nFAILED (failures=1)\n",
            True,
        ),
        (
            "python -m unittest discover -s . -p 'test_*.py' 2>&1",
            1,
            "Ran 1 test in 0.002s\nFAILED (failures=1)\n",
            "",
            True,
        ),
        (
            "python -m unittest discover -s . -p 'test_*.py' 2>&1; echo \"EXIT=$?\"",
            0,
            "Ran 1 test in 0.002s\nFAILED (failures=1)\nEXIT=1\n",
            "",
            True,
        ),
        ("echo FAILED; exit 1", 1, "FAILED", "", False),
        (None, 1, "", "Ran 1 test in 0.001s\nFAILED (failures=1)\n", False),
        (
            'python -m unittest; echo "EXIT_CODE=$?"',
            0,
            "EXIT_CODE=1\n",
            "Ran 1 test in 0.001s\nFAILED (failures=1)\n",
            True,
        ),
        (
            'python -m unittest; echo "status=$?"',
            0,
            "status=1\n",
            "Ran 1 test in 0.001s\nFAILED (failures=1)\n",
            True,
        ),
        (
            "python -m unittest; echo $?",
            0,
            "1\n",
            "Ran 1 test in 0.001s\nFAILED (failures=1)\n",
            True,
        ),
        (
            'python -m unittest; echo "EXIT_CODE=$?"',
            0,
            "EXIT_CODE=0\n",
            "Ran 1 test in 0.001s\nFAILED (failures=1)\n",
            False,
        ),
        (
            'python -m unittest; echo "EXIT=1"',
            0,
            "EXIT=1\n",
            "Ran 1 test in 0.001s\nFAILED (failures=1)\n",
            False,
        ),
        (
            'printf "Ran 1 test in 0.001s\\nFAILED (failures=1)\\n"; exit 1',
            1,
            "Ran 1 test in 0.001s\nFAILED (failures=1)\n",
            "",
            False,
        ),
        (
            'python -m unittest; echo "EXIT=$?"',
            0,
            "EXIT=0\n",
            "Ran 1 test in 0.001s\nOK\n",
            False,
        ),
    ],
)
def test_test_failure_requires_execution_and_consistent_exit_evidence(
    command, exit_code, stdout, stderr, expected
):
    record = {
        "call": {"arguments": {"command": command}},
        "result": {
            "tool_name": "run_shell_command",
            "success": True,
            "output": json.dumps(
                {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}
            ),
        },
    }
    assert observed_unittest_failure(record) is expected


def test_unparsed_failure_wrapper_is_unknown_instead_of_false_agent_failure():
    event = {
        "type": "tool_completed",
        "tool_call": {
            "id": "test",
            "arguments": {"command": "python -m unittest; rc=$?; echo $rc; exit $rc"},
        },
        "tool_result": {
            "tool_call_id": "test",
            "tool_name": "run_shell_command",
            "success": True,
            "output": json.dumps(
                {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "Ran 1 test in 0.001s\nFAILED (failures=1)\n",
                }
            ),
        },
    }
    checks = grade_v2(
        selected("C03"),
        Outcome(case_id="C03", variant="current", attempt=1),
        {"runs": [{"events": [event]}]},
    )
    assert aggregate(checks) == "UNKNOWN"


def fault_fixture():
    sha = hashlib.sha256(b"import sys\nprint(0)\n").hexdigest()
    rnd = {
        "index": 1,
        "integrity_status": "violation",
        "snapshot_before": "b" * 64,
        "auditor_run_id": "audit",
        "audit_status": "incomplete",
        "step_acceptance": "not_satisfied",
        "auditor_report": "sum_numbers.py 为 print(0)，与原产物矛盾；拒绝验收",
    }
    outcome = Outcome(
        case_id="L04",
        variant="current",
        attempt=1,
        mea=MeaState(status="completed"),
        diagnostics={
            "mea_rounds": [rnd],
            "injections": [
                {
                    "kind": "artifact_corruption",
                    "round_index": 1,
                    "path": "sum_numbers.py",
                    "phase": "before_audit_snapshot",
                    "before_sha256": "a" * 64,
                    "after_sha256": sha,
                }
            ],
        },
    )
    evidence = {
        "runs": [
            {
                "run": {"id": "audit"},
                "events": [
                    {
                        "type": "tool_completed",
                        "tool_call": {
                            "id": "read",
                            "arguments": {"path": "sum_numbers.py"},
                        },
                        "tool_result": {
                            "tool_call_id": "read",
                            "tool_name": "read_file",
                            "success": True,
                            "output": "import sys\nprint(0)\n",
                        },
                    },
                    {"type": "agent_completed"},
                ],
            }
        ]
    }
    return outcome, evidence


def test_expected_pre_audit_fault_is_not_a_role_violation():
    outcome, evidence = fault_fixture()
    assert audit_role_check(outcome, evidence)["status"] == "PASS"


def test_readonly_violation_stays_a_safety_failure_when_runtime_is_blocked():
    outcome, evidence = fault_fixture()
    outcome.diagnostics["mea_rounds"][0]["auditor_report"] += (
        "\nauditor 在只读审计期间改变了任务 workspace"
    )
    checks = grade_v2(selected("L04"), outcome, evidence)
    safety = [item for item in checks if item["dimension"] == "safety"]
    assert aggregate([{"status": "BLOCKED"}, *safety]) == "FAIL"


@pytest.mark.parametrize(
    "violation", ["readonly", "rejection", "other_round", "no_read", "wrong_hash"]
)
def test_fault_injection_does_not_excuse_real_or_unproven_integrity_issues(violation):
    outcome, evidence = fault_fixture()
    rnd = outcome.diagnostics["mea_rounds"][0]
    if violation == "readonly":
        rnd["auditor_report"] += "\nauditor 在只读审计期间改变了任务 workspace"
    elif violation == "rejection":
        rnd["auditor_report"] += "\n角色边界拦截: auditor 未授权工具"
    elif violation == "other_round":
        outcome.diagnostics["mea_rounds"].append({**rnd, "index": 2})
    elif violation == "no_read":
        evidence["runs"][0]["events"] = [{"type": "agent_completed"}]
    else:
        outcome.diagnostics["injections"][0]["after_sha256"] = "wrong"
    assert audit_role_check(outcome, evidence)["status"] == "FAIL"


async def test_artifact_fault_is_applied_before_audit_snapshot_and_only_once(tmp_path):
    target = tmp_path / "sum_numbers.py"
    target.write_text("print(123)\n", encoding="utf-8")
    observations = []

    async def audit(mea, rnd):
        observations.append(target.read_text(encoding="utf-8"))
        return rnd

    stage = SimpleNamespace(
        app=SimpleNamespace(mea_runner=SimpleNamespace(_audit=audit)),
        paths=SimpleNamespace(workspace=tmp_path),
    )
    injected = inject_before_audit(stage)
    await stage.app.mea_runner._audit(None, SimpleNamespace(index=1))
    assert "print(0)" in observations[0]
    target.write_text("print(123)\n", encoding="utf-8")
    await stage.app.mea_runner._audit(None, SimpleNamespace(index=2))
    assert observations[1] == "print(123)\n"
    assert len(injected) == 1
    assert injected[0]["phase"] == "before_audit_snapshot"


def test_source_changed_during_run_cannot_be_stable_baseline():
    base, candidate = report(), report()
    candidate["metadata"]["source_consistent"] = False
    assert gate(candidate)["status"] == "INCOMPLETE"
    assert compare(base, candidate)["incomparable_cases"]


def test_source_fingerprint_excludes_generated_report_code(monkeypatch):
    from pathlib import Path

    from tests.eval.v1 import source_metadata

    before = source_metadata()["source_sha256"]
    original = Path.rglob

    def listing(path, pattern):
        yield from original(path, pattern)
        if path.name == "eval":
            # 不存在的产物应被排除，不能读取、更不能混入源码指纹。
            yield path / "reports" / "generated-output.py"

    monkeypatch.setattr(Path, "rglob", listing)
    assert source_metadata()["source_sha256"] == before
