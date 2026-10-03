"""摘要用例必须真实施加 Token 压力，旧字段不能悄悄失效。"""

import json

from app.runtime.context.config import ContextSettings, ContextSummaryModelConfig
from tests.eval.selftest.scripted import (
    after_tool,
    answer,
    call,
    last_user,
    scripted_factory,
)
from tests.eval.v1 import run_trial
from tests.eval.v1_cases import v1_cases


def test_v1_context_variables_are_recognized_settings():
    names = set(ContextSettings.model_fields) | {
        f"context_summary_{key}" for key in ContextSummaryModelConfig.model_fields
    }
    for plan in v1_cases():
        for key in plan.case.env:
            if key.startswith("CONTEXT_"):
                assert key.lower() in names, key


async def test_summary_case_really_invokes_summarizer_and_preserves_constraint(
    tmp_path,
):
    summarized = 0

    def reply(request):
        nonlocal summarized
        if any(
            (message.content or "").startswith("# 会话摘要规范")
            for message in request.messages
        ):
            summarized += 1
            return answer(
                json.dumps(
                    {
                        "current_objective": "根据初始约束完成清单",
                        "user_constraints": ["仅苹果和梨，禁止香蕉；不写长期记忆"],
                        "key_decisions": [],
                        "completed_work": [],
                        "current_state": [],
                        "pending_work": ["写清单"],
                        "important_facts": [],
                    },
                    ensure_ascii=False,
                )
            )
        if after_tool(request):
            return answer("完成")
        if "在 result.txt 写最终清单" in last_user(request):
            return call("write_file", path="result.txt", content="苹果\n梨\n")
        return answer("收到")

    plan = next(plan for plan in v1_cases() if plan.case.id == "S02")
    row = await run_trial(
        plan, factory=scripted_factory(reply), folder=tmp_path / "S02", attempt=1
    )
    assert row["status"] == "PASS", row["checks"]
    assert summarized > 0
    evidence = json.loads((tmp_path / "S02/evidence.json").read_text(encoding="utf-8"))
    assert any(
        event.get("summary_updated")
        for run in evidence["runs"]
        for event in run["events"]
    )
