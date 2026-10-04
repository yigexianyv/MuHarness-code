"""完整证据复核不能覆盖原报告、改采集条件或抹掉实际违规。"""

import json
from pathlib import Path

import pytest
import pytest_asyncio

from tests.eval.selftest.scripted import scripted_factory
from tests.eval.selftest.test_v2_driver import recovering_reply, selected
from tests.eval.v2 import run_v2
from tests.eval.v2_regrade import regrade


@pytest_asyncio.fixture
async def saved_batch(tmp_path):
    _, folder = await run_v2(
        plans=[selected("B04")],
        repeat=3,
        factory=scripted_factory(recovering_reply),
        out=tmp_path / "raw",
        prerequisite={"available": False},
        progress=lambda _: None,
    )
    return folder / "report.json", tmp_path / "reviewed"


def update(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def test_regrade_preserves_original_and_costs_and_can_be_compared(saved_batch):
    from tests.eval.v2_report import compare

    source, out = saved_batch
    raw = source.read_bytes()
    original = json.loads(raw)
    reviewed, folder = regrade(source, out=out)
    assert source.read_bytes() == raw
    assert reviewed["summary"]["status_counts"]["PASS"] == 3
    assert reviewed["gate"]["status"] == "PASS"
    assert reviewed["metadata"]["regrade"]["inputs_verified"]
    assert not compare(original, reviewed)["incomparable_cases"]
    for before, after in zip(original["attempts"], reviewed["attempts"], strict=True):
        for key in (
            "duration_seconds",
            "chargeable_tokens",
            "api_requests",
            "runtime_profile",
        ):
            assert before[key] == after[key]
        assert Path(after["evidence"]).is_absolute()
    assert (folder / "changes.json").is_file()
    assert "原始采集报告" in (folder / "report.md").read_text(encoding="utf-8")


async def test_regrade_keeps_unexpected_file_change_a_safety_failure(saved_batch):
    source, out = saved_batch
    path = source.parent / "B04-01/evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["after"]["unexpected.txt"] = {"sha256": "changed", "bytes": 1}
    update(path, evidence)
    reviewed, _ = regrade(source, out=out)
    assert reviewed["attempts"][0]["status"] == "FAIL"
    assert reviewed["gate"]["status"] == "FAIL"
    assert Path(reviewed["attempts"][0]["failures"][0]["evidence"]).is_absolute()


async def test_regrade_missing_evidence_cannot_pass(saved_batch):
    source, out = saved_batch
    (source.parent / "B04-01/evidence.json").unlink()
    reviewed, _ = regrade(source, out=out)
    assert reviewed["attempts"][0]["status"] == "UNKNOWN"
    assert not reviewed["baseline_eligible"]


@pytest.mark.parametrize("invalid", ["contract", "outside_evidence", "overwrite"])
async def test_regrade_rejects_changed_contract_and_unsafe_paths(saved_batch, invalid):
    source, out = saved_batch
    report = json.loads(source.read_text(encoding="utf-8"))
    if invalid == "contract":
        report["metadata"]["case_digests"]["B04"] = "changed"
        for row in report["attempts"]:
            row["case_digest"] = "changed"
    elif invalid == "outside_evidence":
        report["attempts"][0]["evidence"] = "../outside.json"
    else:
        out = source.parent
    update(source, report)
    raw = source.read_bytes()
    with pytest.raises(ValueError):
        regrade(source, out=out)
    assert source.read_bytes() == raw
