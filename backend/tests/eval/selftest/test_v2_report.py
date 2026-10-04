"""回归门槛的负例：缺证据、换条件、失败样本和不完整用量均不能被放行。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.eval.v2_report import SCHEMA, classify, compare, gate, summarize


def report(states=("PASS", "PASS", "PASS")):
    rows = [
        {
            "case_id": "B04",
            "attempt": i,
            "category": "Basic Agent",
            "status": state,
            "case_digest": "fixed-input",
            "runtime_profile": {"provider": "fake", "model": "fake-model"},
            "duration_seconds": 10,
            "chargeable_tokens": 100,
            "model_calls": 2,
            "tool_calls": 2,
            "usage_complete": True,
            "checks": [],
            "evidence": f"B04-{i:02d}/evidence.json",
        }
        for i, state in enumerate(states, 1)
    ]
    return {
        "schema_version": SCHEMA,
        "repeat": 3,
        "batch_id": "test",
        "finished_at": "now",
        "metadata": {
            "case_ids": ["B04"],
            "case_digests": {"B04": "fixed-input"},
            "case_info": {"B04": {"category": "Basic Agent"}},
            "grader_version": "v2.1",
            "model_transport": "stream",
            "api_retry_policy": {"max_retries": 3},
            "execution_kind": "injected_factory",
        },
        "attempts": rows,
    }


def test_stability_requires_three_real_passes():
    summary = summarize(report())
    assert summary["cases"][0]["stability"] == "stable_pass"
    assert summary["categories"][0]["coverage"] == 1
    assert gate(report())["status"] == "PASS"
    one = report(("PASS",))
    one["repeat"] = 1
    assert summarize(one)["cases"][0]["stability"] == "insufficient_repeats"
    assert gate(one)["status"] == "INCOMPLETE"


@pytest.mark.parametrize("state", ["UNKNOWN", "BLOCKED"])
def test_unknown_and_blocked_never_pass_or_enter_stability_baseline(state):
    current = report(("PASS", state, "PASS"))
    summary = summarize(current)
    assert summary["cases"][0]["stability"] == "incomplete"
    assert summary["categories"][0]["pass_rate"] == 1
    assert summary["categories"][0]["coverage"] == pytest.approx(2 / 3)
    assert not summary["baseline_eligible"]
    assert gate(current)["status"] == "INCOMPLETE"


def test_failures_produce_fluctuation_and_core_regression():
    base, candidate = report(), report(("PASS", "FAIL", "PASS"))
    assert summarize(candidate)["cases"][0]["stability"] == "fluctuating"
    comparison = compare(base, candidate)
    assert comparison["regressions"][0]["case_id"] == "B04"
    assert gate(candidate, comparison)["status"] == "REVIEW"


@pytest.mark.parametrize(
    "condition", ["digest", "model", "transport", "grader", "retries", "repeat"]
)
def test_changed_conditions_are_incomparable(condition):
    base, candidate = report(), report()
    if condition == "digest":
        candidate["attempts"][0]["case_digest"] = "different-input"
    elif condition == "model":
        candidate["attempts"][1]["runtime_profile"]["model"] = "other-model"
    elif condition == "repeat":
        candidate["repeat"] = 2
    else:
        key = {
            "transport": "model_transport",
            "grader": "grader_version",
            "retries": "api_retry_policy",
        }[condition]
        candidate["metadata"][key] = "different"
    comparison = compare(base, candidate)
    assert not comparison["comparable_cases"]
    assert comparison["incomparable_cases"]
    assert gate(candidate, comparison)["status"] == "INCOMPLETE"


def test_cost_comparison_uses_matched_successes_and_complete_usage_only():
    base, candidate = report(), report(("PASS", "FAIL", "PASS"))
    candidate["attempts"][0]["duration_seconds"] = 13
    candidate["attempts"][1]["duration_seconds"] = 1000
    candidate["attempts"][2]["duration_seconds"] = 13
    candidate["attempts"][2]["usage_complete"] = False
    candidate["attempts"][2]["chargeable_tokens"] = 0
    comparison = compare(base, candidate)
    duration = next(
        item
        for item in comparison["efficiency"]
        if item["metric"] == "duration_seconds"
    )
    tokens = next(
        item
        for item in comparison["efficiency"]
        if item["metric"] == "chargeable_tokens"
    )
    assert duration["matched_successes"] == 2
    assert duration["delta_percent"] == 30
    assert duration["warning"]
    assert tokens["matched_successes"] == 1
    assert tokens["delta_percent"] == 0


def test_incomplete_coverage_cannot_be_an_improvement():
    comparison = compare(
        report(("FAIL", "FAIL", "FAIL")), report(("PASS", "UNKNOWN", "PASS"))
    )
    assert not comparison["improvements"]
    assert comparison["warnings"]


def test_severe_safety_failure_wins_over_missing_coverage():
    current = report(("FAIL", "UNKNOWN", "PASS"))
    current["attempts"][0]["checks"] = [
        {"dimension": "safety", "status": "FAIL", "name": "泄漏", "detail": "leak"}
    ]
    assert gate(current)["status"] == "FAIL"


def test_recovered_tool_failure_and_correct_approval_denial_are_not_task_failures():
    row = report()["attempts"][0]
    evidence = {
        "runs": [
            {
                "events": [
                    {
                        "event_id": "e1",
                        "type": "tool_completed",
                        "tool_result": {"success": False},
                    }
                ]
            }
        ]
    }
    assert classify(row, evidence) == []
    row["checks"] = [
        {
            "dimension": "process",
            "name": "审批拒绝后停止",
            "status": "PASS",
            "detail": "denied",
        }
    ]
    assert classify(row, evidence) == []


def test_duplicate_or_missing_attempt_is_not_complete():
    current = report()
    current["attempts"][1] = deepcopy(current["attempts"][0])
    assert not summarize(current)["complete"]
    assert gate(current)["status"] == "INCOMPLETE"


def test_v1_review_cannot_be_used_as_v2_regression_baseline():
    base = report()
    base["schema_version"] = "muharness-eval-v1-review"
    with pytest.raises(ValueError, match="V2"):
        compare(base, report())
