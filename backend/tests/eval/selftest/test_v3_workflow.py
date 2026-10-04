from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from app.runtime.mea.prompts import (
    AUDITOR_INSTRUCTIONS,
    FINAL_RESPONSE_INSTRUCTIONS,
    MANAGER_INSTRUCTIONS,
)
from tests.eval.outcome import MeaState, Outcome, TurnOutcome
from tests.eval.selftest.scripted import (
    after_tool,
    answer,
    call,
    last_user,
    scripted_factory,
)
from tests.eval.selftest.test_v2_report import report as v2_report
from tests.eval.selftest.test_v3 import mechanism_reply, selected, verdicts
from tests.eval.v1 import run_trial, runtime_state, write_json
from tests.eval.v1_cases import Verification
from tests.eval.v3 import run_v3
from tests.eval.v3_driver import make_driver
from tests.eval.v3_experiment import analyze
from tests.eval.v3_grading import (
    grade_case_v3,
    grade_v3,
    recovered_audit_failures,
    role_check_v3,
    runtime_state_v3,
)
from tests.eval.v3_judge import LocalPreparationJudge, calibration_fixtures, validate
from tests.eval.v3_regrade import regrade, restore_verified_files
from tests.eval.v3_regression import capture
from tests.offline.mea.test_mea_integration import execute, manager
from tests.offline.mea.test_mea_integration import report as audit_report


def recovered_scene():
    rounds = [
        {"index": 1, "step_id": "s1", "auditor_run_id": "failed", "kind": "normal"},
        {"index": 2, "step_id": "s1", "auditor_run_id": "retry", "kind": "audit_only"},
        {"index": 3, "step_id": None, "auditor_run_id": "final", "kind": "final_audit"},
    ]
    for rnd in rounds:
        rnd.update(
            mea_run_id="m1",
            audit_requirements_revision=2,
            phase="applied",
            audit_status="complete",
            integrity_status="clean",
            contract_audit_status="aligned",
        )
    rounds[0]["audit_status"] = "blocked"
    rounds[0]["integrity_status"] = "suspect"
    outcome = Outcome(
        case_id="L05",
        attempt=1,
        variant="current",
        mea=MeaState(status="completed", steps={"s1": "done"}),
        turns=[TurnOutcome(say="task", answer="done")],
        diagnostics={"mea_rounds": rounds},
    )
    evidence = {
        "runs": [
            {
                "run": {"id": "failed", "mode": "audit", "source_id": "m1"},
                "events": [
                    {"type": "agent_failed", "error": {"type": "ModelInvocationError"}}
                ],
            },
            *[
                {"run": {"id": identifier}, "events": [{"type": "agent_completed"}]}
                for identifier in ["retry", "final"]
            ],
        ]
    }
    return outcome, evidence


def test_recovered_audit_does_not_block_a_completed_task():
    outcome, evidence = recovered_scene()
    assert runtime_state(outcome, evidence)[0] == "BLOCKED"
    assert recovered_audit_failures(outcome, evidence) == ["failed"]
    assert runtime_state_v3(outcome, evidence) is None
    assert role_check_v3(outcome, evidence)["status"] == "PASS"


@pytest.mark.parametrize(
    "violation",
    ["workspace_changed", "role_rejected", "tool_attempt", "integrity_violation"],
)
def test_recovered_model_error_never_excuses_role_or_workspace_violation(violation):
    outcome, evidence = recovered_scene()
    prior = outcome.diagnostics["mea_rounds"][0]
    if violation == "workspace_changed":
        prior["auditor_report"] = "auditor 在只读审计期间改变了任务 workspace"
    if violation == "role_rejected":
        outcome.mea.role_rejections = 1
    if violation == "tool_attempt":
        evidence["runs"][0]["events"].insert(0, {"type": "tool_started"})
    if violation == "integrity_violation":
        prior["integrity_status"] = "violation"
    assert role_check_v3(outcome, evidence)["status"] == "FAIL"


@pytest.mark.parametrize("status", ["done", "pending"])
def test_amendment_accepts_superseded_steps_but_requires_new_steps_done(status):
    outcome, evidence = recovered_scene()
    outcome.mea.steps = {"old": "superseded", "new": status}
    checks = grade_case_v3(selected("L05"), outcome, evidence)
    actual = next(c for c in checks if c["name"] == "mea_steps_done")
    assert actual["status"] == ("PASS" if status == "done" else "FAIL")


@pytest.mark.parametrize(
    "missing",
    [
        "completed",
        "final",
        "clean",
        "same_step",
        "same_revision",
        "model_error",
        "same_mea",
        "audit_role",
    ],
)
def test_recovery_requires_terminal_and_linked_audit_proof(missing):
    outcome, evidence = recovered_scene()
    rounds = outcome.diagnostics["mea_rounds"]
    if missing == "completed":
        outcome.mea.status = "failed"
    if missing == "final":
        evidence["runs"][2]["events"].clear()
    if missing == "clean":
        rounds[1]["integrity_status"] = "suspect"
    if missing == "same_step":
        rounds[1]["step_id"] = "s2"
    if missing == "same_revision":
        rounds[1]["audit_requirements_revision"] = 1
    if missing == "model_error":
        evidence["runs"][0]["events"][0]["error"]["type"] = "ToolFailure"
    if missing == "same_mea":
        evidence["runs"][0]["run"]["source_id"] = "other"
    if missing == "audit_role":
        evidence["runs"][0]["run"]["mode"] = "manage"
    assert not recovered_audit_failures(outcome, evidence)
    assert runtime_state_v3(outcome, evidence) is not None


@pytest.mark.parametrize("original", [b"print(1)\n", b"print(1)\r\n"])
def test_posthoc_verification_uses_original_bytes_only(tmp_path, original):
    evidence = {
        "outcome": {"files": {"main.py": "print(1)\n"}},
        "after": {"main.py": {"sha256": hashlib.sha256(original).hexdigest()}},
    }
    spec = Verification(("main.py",), "pass")
    restore_verified_files(evidence, spec, tmp_path)
    assert (tmp_path / "main.py").read_bytes() == original
    evidence["outcome"]["files"]["main.py"] = "print(2)\n"
    with pytest.raises(ValueError):
        restore_verified_files(evidence, spec, tmp_path)


def experiment_pair():
    off, on = v2_report(), v2_report()
    for payload, enabled in ((off, False), (on, True)):
        payload["metadata"].update(
            source_sha256="same",
            source_consistent=True,
            experiment_contract_sha256="same-input",
        )
        payload["metadata"]["case_ids"] = ["M03"]
        payload["attempts"] = []
        payload["repeat"] = 3
        for i in range(1, 4):
            payload["attempts"].append(
                {
                    "attempt": i,
                    "case_id": "M03",
                    "status": "PASS" if enabled else "FAIL",
                    "runtime_profile": {
                        "provider": "fake",
                        "model": "same",
                        "memory_reflection_enabled": enabled,
                        "memory_reflection": {"enabled": enabled, "max_attempts": 2},
                    },
                }
            )
    return off, on


async def test_v3_regrade_preserves_collection_and_never_calls_model(tmp_path):
    report, original = await run_v3(
        plans=[selected("A05")],
        factory=scripted_factory(mechanism_reply),
        repeat=1,
        skip_judge=True,
        out=tmp_path / "original",
        prerequisite={"available": False},
        progress=lambda _: None,
    )
    source = original / "report.json"
    evidence = original / report["attempts"][0]["evidence"]
    before = (source.read_bytes(), evidence.read_bytes())
    reviewed, folder = await regrade(source, out=tmp_path / "reviewed")
    assert reviewed["attempts"][0]["status"] == "PASS"
    assert (
        reviewed["attempts"][0]["model_calls"] == report["attempts"][0]["model_calls"]
    )
    assert reviewed["attempts"][0]["quality"]["state"] == "NOT_REASSESSED"
    assert (source.read_bytes(), evidence.read_bytes()) == before
    assert folder != original
    with pytest.raises(ValueError):
        await regrade(source, out=original / "bad")


def test_single_factor_analysis_checks_effective_configuration():
    off, on = experiment_pair()
    result = analyze(off, on)
    assert result["status"] == "COMPARABLE"
    assert result["delta_pp"] == 100
    changed = copy.deepcopy(on)
    changed["attempts"][0]["runtime_profile"]["model"] = "different"
    result = analyze(off, changed)
    assert result["status"] == "INCOMPARABLE"
    assert result["delta_pp"] is None


@pytest.mark.parametrize(
    "problem", ["source", "factor", "missing", "blocked", "contract"]
)
def test_bad_control_cannot_claim_improvement(problem):
    off, on = experiment_pair()
    if problem == "source":
        on["metadata"]["source_consistent"] = False
    if problem == "factor":
        on["attempts"][0]["runtime_profile"]["memory_reflection_enabled"] = False
    if problem == "missing":
        on["attempts"].pop()
    if problem == "blocked":
        on["attempts"][0]["status"] = "BLOCKED"
    if problem == "contract":
        on["metadata"]["experiment_contract_sha256"] = "changed"
    result = analyze(off, on)
    assert result["status"] == "INCOMPARABLE"
    assert result["delta_pp"] is None


def test_calibration_fake_success_is_explicitly_synthetic():
    fixtures = calibration_fixtures()
    assert len(fixtures) == 2
    for fixture in fixtures:
        assert fixture["origin"] == "calibration_fixture"
        assert fixture["packet"]["objective_status"] == "FAIL"
        assert "不是项目实际运行" in fixture["packet"]["fixture_notice"]
        result = verdicts("FAIL", ["fixture:observed"])
        assert validate(result, fixture["packet"])


async def test_local_preparation_never_claims_ai_assessment():
    judge = LocalPreparationJudge()
    result = await judge.assess(calibration_fixtures()[0]["packet"])
    assert result["state"] == "PENDING_JUDGE"
    assert all(r["status"] == "UNKNOWN" for r in result["rubrics"].values())
    assert result["api_requests"] == []
    assert result["usage"] is None
    assert judge.config["model"] is None


def test_abandoned_revision_does_not_crash_grader():
    outcome = Outcome(
        case_id="L05",
        attempt=1,
        variant="current",
        diagnostics={
            "amendment": {
                "observations": [{"accepted": True}],
                "requirements": {"amendments": ["changed"]},
            },
            "mea_rounds": [
                {
                    "exec_requirements_revision": None,
                    "audit_requirements_revision": None,
                },
                {"exec_requirements_revision": 2, "audit_requirements_revision": 2},
            ],
        },
    )
    checks = grade_v3(selected("L05"), outcome, {"runs": []})
    assert all(c["status"] == "PASS" for c in checks)


@pytest.mark.parametrize("identifier", ["L05", "L06"])
async def test_long_control_with_real_mea_runtime(identifier, tmp_path):
    """Scripted model, real task/round/requirements/pause stores and executor."""
    managing = 0

    def reply(request):
        nonlocal managing
        prompt = last_user(request)
        if prompt.startswith(FINAL_RESPONSE_INSTRUCTIONS):
            return answer("已完成文件并核验。")
        if prompt.startswith(MANAGER_INSTRUCTIONS):
            managing += 1
            execute_rounds = 2 if identifier == "L05" else 1
            route = (
                execute("s1")
                if managing <= execute_rounds
                else "下一步: 最终验收\n验收重点: 文件已创建"
            )
            return answer(manager(route))
        if prompt.startswith(AUDITOR_INSTRUCTIONS):
            return (
                answer(audit_report())
                if after_tool(request)
                else call("read_file", path="sum_numbers.py")
            )
        return (
            answer("已写入文件。")
            if after_tool(request)
            else call("write_file", path="sum_numbers.py", content="print('SUM=0')\n")
        )

    plan = selected(identifier)
    plan = replace(
        plan,
        case=plan.case.model_copy(
            update={
                "mea": plan.case.mea.model_copy(
                    update={"steps": plan.case.mea.steps[:1], "round_budget": 6}
                ),
                "timeout_seconds": 25,
                "checks": [{"no_run_errors": True}],
            }
        ),
        verification=None,
        observation=None,
        allowed_changes=("sum_numbers.py",),
    )
    row = await run_trial(
        plan,
        factory=scripted_factory(reply),
        folder=tmp_path / identifier,
        attempt=1,
        driver=make_driver(),
        observation_grader=grade_v3,
        case_evaluator=grade_case_v3,
    )
    assert row["status"] == "PASS", row["checks"]


def test_actual_failure_capture_is_immutable_and_replayable(tmp_path):
    plan = selected("A03")
    evidence = {
        "case_digest": plan.digest(),
        "outcome": {
            "turns": [
                {
                    "say": "only JSON",
                    "answer": '```json\n{"owner":"林舟","version":7}\n```\n解释',
                }
            ]
        },
    }
    write_json(tmp_path / "evidence.json", evidence)
    source = {
        "finished_at": "2026-10-03",
        "metadata": {"grader_version": "v3.1"},
        "attempts": [
            {
                "case_id": "A03",
                "attempt": 2,
                "status": "FAIL",
                "evidence": "evidence.json",
                "case_digest": plan.digest(),
            }
        ],
    }
    write_json(tmp_path / "report.json", source)
    original = (tmp_path / "report.json").read_bytes()
    bundle = capture(
        tmp_path / "report.json", case_id="A03", attempt=2, out=tmp_path / "saved"
    )
    assert bundle["minimal_replay"]["expected_verdict"] == "FAIL"
    assert (tmp_path / "report.json").read_bytes() == original
    assert (
        json.loads(
            (tmp_path / "saved" / "original-evidence.json").read_text(encoding="utf-8")
        )
        == evidence
    )
    with pytest.raises(ValueError):
        capture(
            tmp_path / "report.json", case_id="A03", attempt=2, out=tmp_path / "saved"
        )
    source["attempts"][0]["status"] = "UNKNOWN"
    write_json(tmp_path / "report.json", source)
    with pytest.raises(ValueError):
        capture(
            tmp_path / "report.json", case_id="A03", attempt=2, out=tmp_path / "unknown"
        )
