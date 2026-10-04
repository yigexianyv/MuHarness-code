"""真实存储、真实并发、真实子进程，模型使用离线脚本。"""

from __future__ import annotations

import json

import pytest

from tests.eval.selftest.scripted import after_tool, answer, call, scripted_factory
from tests.eval.v1 import run_trial
from tests.eval.v1_cases import CATEGORIES, v1_cases
from tests.eval.v2 import run_v2
from tests.eval.v2_cases import v2_cases
from tests.eval.v2_driver import make_driver
from tests.eval.v2_grading import grade_v2


def selected(case_id):
    return next(plan for plan in v2_cases() if plan.case.id == case_id)


def test_contracts_and_split_do_not_mutate_v1():
    before = {plan.case.id: plan.digest() for plan in v1_cases()}
    plans = v2_cases()
    assert len(plans) == 24
    assert all(
        sum(plan.category == category for plan in plans) == 4 for category in CATEGORIES
    )
    assert sum(plan.split == "dev" for plan in plans) == 18
    assert sum(plan.split == "holdout" for plan in plans) == 6
    assert len({plan.digest() for plan in plans}) == 24
    assert {plan.case.id: plan.digest() for plan in v1_cases()} == before
    assert (
        next(plan.case.turns[0].say for plan in plans if plan.case.id == "C01")
        != selected("C03").case.turns[0].say
    )


def recovering_reply(request):
    if not after_tool(request):
        return call("read_file", path="retry.txt")
    if "EVAL_TRANSIENT_READ_FAILURE" in (request.messages[-1].content or ""):
        return call("read_file", path="retry.txt")
    return answer('{"code":"ORION","count":9}')


async def test_transient_tool_failure_recovers_and_resets_each_trial(tmp_path):
    apps = []
    report, folder = await run_v2(
        plans=[selected("B04")],
        repeat=3,
        factory=scripted_factory(recovering_reply, sink=apps),
        out=tmp_path,
        prerequisite={"available": False, "reason": "offline"},
        progress=lambda _: None,
    )
    assert [row["status"] for row in report["attempts"]] == ["PASS"] * 3
    assert report["gate"]["status"] == "PASS"
    assert len({str(app.database) for app in apps}) == 3
    assert all(not app.database.exists() for app in apps)
    for i in range(1, 4):
        evidence = json.loads(
            (folder / f"B04-{i:02d}" / "evidence.json").read_text(encoding="utf-8")
        )
        assert len(evidence["outcome"]["diagnostics"]["injections"]) == 1


async def test_restart_uses_distinct_processes_and_persisted_history(tmp_path):
    row = await run_trial(
        selected("S03"),
        factory=scripted_factory(lambda _: answer("unused parent")),
        folder=tmp_path / "S03-01",
        attempt=1,
        driver=make_driver(),
        observation_grader=grade_v2,
    )
    assert row["status"] == "PASS", row["checks"]
    evidence = json.loads(
        (tmp_path / "S03-01" / "evidence.json").read_text(encoding="utf-8")
    )
    workers = evidence["outcome"]["diagnostics"]["restart_workers"]
    assert len({worker["pid"] for worker in workers}) == 2
    assert workers[0]["history_before_count"] == 0
    assert workers[1]["history_before_count"] >= 2
    assert len(evidence["runs"]) == 2


async def test_concurrent_requests_really_queue_and_preserve_both_updates(tmp_path):
    def reply(request):
        if not after_tool(request):
            return call("read_file", path="counter.txt")
        if request.messages[-1].tool_call_id == "call-read_file":
            feedback = json.loads(request.messages[-1].content or "{}")
            value = int(feedback["output"].strip())
            return call("write_file", path="counter.txt", content=f"{value + 1}\n")
        return answer("已加一")

    row = await run_trial(
        selected("S04"),
        factory=scripted_factory(reply),
        folder=tmp_path / "S04-01",
        attempt=1,
        driver=make_driver(),
        observation_grader=grade_v2,
    )
    assert row["status"] == "PASS", row["checks"]


async def test_missing_file_is_a_correct_outcome_not_tool_failure(tmp_path):
    def reply(request):
        return (
            answer('{"status":"missing"}')
            if after_tool(request)
            else call("read_file", path="absent.txt")
        )

    row = await run_trial(
        selected("B03"),
        factory=scripted_factory(reply),
        folder=tmp_path / "B03-01",
        attempt=1,
        driver=make_driver(),
        observation_grader=grade_v2,
    )
    assert row["status"] == "PASS", row["checks"]


@pytest.mark.parametrize("enabled,expected", [(True, "PASS"), (False, "FAIL")])
async def test_memory_correction_requires_real_post_run_write(
    enabled, expected, tmp_path
):
    from app.domain.memory import MemoryReflectionConfig
    from tests.eval.selftest.scripted import last_user

    def reply(request):
        user = last_user(request)
        try:
            payload = json.loads(user)
        except ValueError:
            payload = {}
        if isinstance(payload, dict) and "user_input" in payload:
            if "纠正" in payload["user_input"]:
                return answer(
                    json.dumps(
                        {
                            "action": "update",
                            "memory_id": "M001",
                            "title": "BETA 项目当前预算",
                            "summary": "BETA 预算 18000",
                            "content": "项目 BETA 的当前预算是 18000 元。",
                            "reason": "用户原话纠正当前预算",
                        },
                        ensure_ascii=False,
                    )
                )
            return answer('{"action":"none","reason":"只读取已有记忆"}')
        if not after_tool(request):
            return call("tool_search", query="memory_read")
        if request.messages[-1].tool_call_id == "call-tool_search":
            return call("memory_read", memory_id="M001")
        if "纠正" in user:
            return answer("已读取旧记录，更正交由运行后的反思更新原记忆。")
        feedback = json.loads(request.messages[-1].content)
        content = json.loads(feedback["output"])["content"]
        budget = 18000 if "18000" in content else 12000
        return answer(json.dumps({"project": "BETA", "budget": budget}))

    original_factory = scripted_factory(reply)

    def build(paths, variant):
        app = original_factory(paths, variant)
        app._memory_reflection_config = MemoryReflectionConfig(
            _env_file=None, enabled=enabled
        )
        return app

    row = await run_trial(
        selected("M03"),
        factory=build,
        folder=tmp_path / "M03-01",
        attempt=1,
        driver=make_driver(),
        observation_grader=grade_v2,
    )
    assert row["status"] == expected, row["checks"]


async def test_failure_fix_regression_loop_keeps_original_three_failures(tmp_path):
    def broken(request):
        if (
            not after_tool(request)
            or "EVAL_TRANSIENT_READ_FAILURE" in request.messages[-1].content
        ):
            return call("read_file", path="retry.txt")
        return answer('{"code":"WRONG","count":9}')

    base, base_folder = await run_v2(
        plans=[selected("B04")],
        repeat=3,
        factory=scripted_factory(broken),
        out=tmp_path / "base",
        prerequisite={"available": False, "reason": "offline"},
        progress=lambda _: None,
    )
    current, _ = await run_v2(
        plans=[selected("B04")],
        repeat=3,
        factory=scripted_factory(recovering_reply),
        out=tmp_path / "fixed",
        baseline=base_folder / "baseline.json",
        prerequisite={"available": False, "reason": "offline"},
        progress=lambda _: None,
    )
    assert [row["status"] for row in base["attempts"]] == ["FAIL"] * 3
    assert [row["status"] for row in current["attempts"]] == ["PASS"] * 3
    assert current["comparison"]["improvements"][0]["pass_rate_delta_pp"] == 100
    assert current["gate"]["status"] == "PASS"


async def test_wrong_answer_is_retained_and_not_task_resampled(tmp_path):
    def reply(request):
        return (
            answer('{"code":"WRONG","count":9}')
            if after_tool(request)
            else call("read_file", path="retry.txt")
        )

    report, _ = await run_v2(
        plans=[selected("B04")],
        repeat=3,
        factory=scripted_factory(reply),
        out=tmp_path,
        prerequisite={"available": False, "reason": "offline"},
        progress=lambda _: None,
    )
    assert [row["status"] for row in report["attempts"]] == ["FAIL"] * 3
    assert report["gate"]["status"] == "REVIEW"
    assert all(row["failures"] for row in report["attempts"])


def test_v2_cli_defaults_and_holdout_selection(capsys):
    from tests.eval.__main__ import build_parser, main

    assert build_parser().parse_args(["v2"]).repeat == 3
    assert main(["v2", "--list", "--split", "holdout"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6
    with pytest.raises(SystemExit, match="未知"):
        main(["v2", "--list", "--case", "TYPO"])


def test_old_compare_command_still_uses_legacy_renderer(monkeypatch, capsys):
    from tests.eval import __main__ as cli

    monkeypatch.setattr(cli.RunReport, "load", lambda _: object())
    monkeypatch.setattr(cli, "render_comparison", lambda *_: "legacy-comparison")
    assert cli.main(["compare", "unused-base.json", "unused-other.json"]) == 0
    assert "legacy-comparison" in capsys.readouterr().out
