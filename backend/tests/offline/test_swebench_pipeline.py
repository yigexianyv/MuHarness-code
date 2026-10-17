from __future__ import annotations

import pytest

from tests.eval.swebench import grading, pipeline, runner
from tests.eval.swebench.common import digest, read_json, write_json


@pytest.fixture
def harness(tmp_path, monkeypatch):
    state = {"events": [], "gold_pass": True, "model_attempts": 0}
    root = tmp_path / "oneclick"
    config = tmp_path / "config.json"
    write_json(config, {})
    rows = [
        {
            "instance_id": f"owner__repo-{i}",
            "repo": "owner/repo",
            "base_commit": "a" * 40,
            "problem_statement": "Fix it",
        }
        for i in (1, 2)
    ]
    package = tmp_path / "package"
    write_json(
        package / "lite30.json",
        {"instances": [{"instance_id": row["instance_id"]} for row in rows]},
    )
    environments = {"mini": {"config": {}}, "muharness": {}, "judge": {}}
    monkeypatch.setattr(pipeline, "PACKAGE", package)
    monkeypatch.setattr(pipeline, "credential", lambda config: "not-a-real-key")
    monkeypatch.setattr(pipeline, "probe", lambda python, kind: environments[kind])
    monkeypatch.setattr(
        pipeline,
        "fetch_rows",
        lambda split, revision: (rows, {"split": split, "revision": "frozen-sha"}),
    )

    def main(argv):
        state["events"].append(argv[0])
        if argv[0] == "prepare":
            runner.prepare(
                root,
                rows,
                None,
                {},
                {"split": "test", "revision": "frozen-sha"},
                environments,
            )
        if argv[0] == "run":
            assert "--retry-infra" in argv
            manifest = read_json(root / "manifest.json")
            for variant in ("mini", "muharness"):
                for row in rows:
                    directory = root / "runs" / variant / row["instance_id"]
                    if (directory / "result.json").exists():
                        continue
                    state["model_attempts"] += 1
                    (directory / "attempt-fixture").mkdir(parents=True)
                    (directory / "attempt-fixture/model.patch").write_bytes(b"")
                    write_json(
                        directory / "result.json",
                        {
                            "attempt": "attempt-fixture",
                            "fingerprint": manifest["fingerprint"],
                            "status": "completed",
                            "stop_reason": "final_answer",
                            "patch_sha256": digest(b""),
                            "duration_seconds": 1,
                            "usage": {},
                        },
                    )
        return 0

    def gold(*args):
        state["events"].append("gold")
        if not state["gold_pass"]:
            raise RuntimeError("Gold failed")
        return {"resolved": True}

    def grade(root, variant, python):
        state["events"].append("grade-" + variant)
        path = runner.export(root, variant)
        result = {
            "fingerprint": read_json(root / "manifest.json")["fingerprint"],
            "predictions_sha256": digest(path.read_bytes()),
            "missing_official": [],
            "official_returncode": 0,
            "outcomes": {
                row["instance_id"]: {"resolved": False, "source": "empty_prediction"}
                for row in rows
            },
        }
        write_json(root / "grades" / f"{variant}.json", result)
        return result

    import tests.eval.swebench.__main__ as cli

    monkeypatch.setattr(cli, "main", main)
    monkeypatch.setattr(grading, "selfcheck", gold)
    monkeypatch.setattr(grading, "grade", grade)
    state["execute"] = lambda **kwargs: pipeline.execute(
        root, config_path=config, mini_python="mini", judge_python="judge", **kwargs
    )
    state["root"] = root
    return state


def test_one_command_orders_gold_before_model_and_reuses_completed_results(harness):
    assert harness["execute"]() == 0
    assert harness["events"] == [
        "selfcheck",
        "prepare",
        "gold",
        "run",
        "grade-mini",
        "grade-muharness",
    ]
    assert (harness["root"] / "report.md").exists()
    assert harness["execute"]() == 0
    assert harness["model_attempts"] == 4
    assert harness["events"].count("gold") == 1
    assert harness["events"].count("grade-mini") == 1


def test_gold_failure_prevents_paid_model_calls(harness):
    harness["gold_pass"] = False
    with pytest.raises(RuntimeError, match="Gold failed"):
        harness["execute"]()
    assert "run" not in harness["events"]
    assert harness["model_attempts"] == 0


def test_check_only_does_not_prepare_or_call_models(harness):
    assert harness["execute"](check_only=True) == 0
    assert harness["events"] == ["selfcheck"]
    assert not harness["root"].exists()


def test_partial_cached_grading_is_recomputed_even_when_patches_stay_empty(harness):
    assert harness["execute"]() == 0
    path = harness["root"] / "grades/mini.json"
    result = read_json(path)
    result["outcomes"] = {}
    write_json(path, result)
    assert harness["execute"]() == 0
    assert harness["events"].count("grade-mini") == 2
