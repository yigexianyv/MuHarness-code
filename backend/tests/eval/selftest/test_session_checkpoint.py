"""Interrupted batches retain completed attempts without claiming completion."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from tests.eval import session
from tests.eval.outcome import Outcome
from tests.eval.report import RunReport
from tests.eval.spec import CURRENT, Case


def case(name):
    return Case(id=name, suite="behavior", title=name, why="checkpoint", turns=["hello"],
                checks=[{"no_run_errors": True}])


@pytest.mark.parametrize("interrupted", [False, True])
async def test_batch_checkpoints_each_finished_attempt(monkeypatch, tmp_path, interrupted):
    @asynccontextmanager
    async def open_stage(case, *args, **kwargs):
        yield case

    async def drive(case, *, attempt):
        if interrupted and case.id == "b91-second":
            raise asyncio.CancelledError
        return Outcome(case_id=case.id, variant="current", attempt=attempt)

    monkeypatch.setattr(session, "open_stage", open_stage)
    monkeypatch.setattr(session, "drive", drive)
    run = session.run_cases(
        [case("b90-first"), case("b91-second")], variant=CURRENT, factory=lambda *_: None,
        keep_outcomes=True, checkpoint=lambda report: report.save(tmp_path),
    )
    if interrupted:
        with pytest.raises(asyncio.CancelledError):
            await run
    else:
        await run
    saved = RunReport.load(next(tmp_path.glob("*.json")))
    assert [item.case_id for item in saved.attempts] == (
        ["b90-first"] if interrupted else ["b90-first", "b91-second"]
    )
    assert saved.attempts[0].outcome is not None
    assert (saved.finished_at is None) is interrupted


async def test_skipped_attempts_are_also_checkpointed(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "docker_available", lambda: False)
    needs_docker = case("b90-docker").model_copy(update={"requires": ["docker"]})
    await session.run_cases(
        [needs_docker], variant=CURRENT, factory=lambda *_: None,
        checkpoint=lambda report: report.save(tmp_path),
    )
    saved = RunReport.load(next(tmp_path.glob("*.json")))
    assert saved.finished_at is not None
    assert saved.attempts[0].status == "skipped"


def test_failed_atomic_replace_preserves_previous_report(monkeypatch, tmp_path):
    report = RunReport(variant="current", repeat=1, started_at="2026-10-08T00:00:00+00:00")
    path, _ = report.save(tmp_path)
    report.finished_at = "2026-10-08T00:01:00+00:00"

    def fail(*args):
        raise PermissionError("file is in use")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(PermissionError):
        report.save(tmp_path)
    assert RunReport.load(path).finished_at is None
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".json", ".md"]
