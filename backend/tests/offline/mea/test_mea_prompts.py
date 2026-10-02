
from __future__ import annotations

import pytest

from app.runtime.mea.parsing import parse_control_header
from app.runtime.mea.prompts import (
    MANAGER_REPORTS_TOTAL,
    AuditKind,
    RoundView,
    StepView,
    SupersededView,
    build_auditor_prompt,
    build_executor_prompt,
    build_final_response_prompt,
    build_format_repair_prompt,
    build_manager_prompt,
    format_audit_findings,
    format_audit_history,
    format_related_reports,
    format_step_list,
    invalid_final_audit_feedback,
    invalid_header_report,
    invalid_plan_feedback,
    no_report_placeholder,
    readonly_violation_report,
    recovery_record,
    requirements_updated_feedback,
    round_abandoned_feedback,
    supersede_rejected_feedback,
)
from app.runtime.mea.requirements import (
    AmendmentKind,
    add_amendment,
    build_requirements,
    render_requirements,
)

REQUIREMENTS = render_requirements(
    add_amendment(
        build_requirements(
            original_request="把 users.csv 导入数据库并导出 JSON",
            goal="users 表包含 CSV 全部数据",
            constraints=["不要修改数据库结构"],
        ),
        kind=AmendmentKind.NOTE,
        text="改为导出 CSV，不要 JSON",
    )
)

STEPS = (
    StepView("s1", "读取 CSV", "done", "表头和 120 行都读到", "round_002"),
    StepView("s2", "导入数据库", "todo", "users 表 120 行", "round_004", origin_revision=1),
    StepView("s3", "导出 JSON", "superseded", "生成 users.json", superseded_by="A1",
             superseded_reason="用户改为 CSV"),
)

STEP_REPORT = (
    "状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n"
    "审计事实: 已读取 120 行\n\n验收约束反查:\n阻断约束: 无"
)


# 函数说明：test_step_list_format
# 用途：回归验证回归测试与测试辅助中的 `step_list_format` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`format_step_list` → `text.splitlines`
# 。
# 分支与异常：
#   验证条件：`text.splitlines() == ['s1 [done 依据 round_002] 读取 CSV — 验收: 表头和 1
# 20 行都读到 — 产生于 v1', 's2 [todo 见 round_004] 导入数据库 — 验收: users 表 120…`。
def test_step_list_format() -> None:
    text = format_step_list(STEPS)
    assert text.splitlines() == [
        "s1 [done 依据 round_002] 读取 CSV — 验收: 表头和 120 行都读到 — 产生于 v1",
        "s2 [todo 见 round_004] 导入数据库 — 验收: users 表 120 行 — 产生于 v1",
        "s3 [superseded 依据 A1] 导出 JSON — 原验收: 生成 users.json — 原因: 用户改为 CSV",
    ]


# 函数说明：test_manager_prompt_carries_requirements_verbatim_and_budget
# 用途：回归验证回归测试与测试辅助中的
# `manager_prompt_carries_requirements_verbatim_and_budget` 场景，下方断言说明列出实际通
# 过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RoundView` → `build_manager_prompt`。
# 分支与异常：
#   验证条件：`REQUIREMENTS in prompt`。
#   验证条件：`'- 不要修改数据库结构' in prompt`。
#   验证条件：`'A1（生效于要求 v2' in prompt`。
#   验证条件：`'(还没有任务契约' in prompt`。
def test_manager_prompt_carries_requirements_verbatim_and_budget() -> None:
    rounds = [RoundView(2, AuditKind.NORMAL, "s1", "下一步: 执行任务\n步骤: s1", STEP_REPORT, "")]
    prompt = build_manager_prompt(
        requirements_text=REQUIREMENTS,
        steps=STEPS,
        contract=None,
        state=None,
        rounds=rounds,
        round_index=3,
        round_budget=25,
    )
    assert REQUIREMENTS in prompt
    assert "- 不要修改数据库结构" in prompt
    assert "A1（生效于要求 v2" in prompt
    assert "(还没有任务契约" in prompt
    assert "--- Round 2 auditor report（步骤审计）---" in prompt
    assert "round_id: round_002" in prompt
    assert "- 包含本轮在内的剩余轮次: 23" in prompt
    assert "本轮一次性指令" not in prompt
    assert "Executor 可直接调用 Host 工具 artifact_publish" in prompt
    assert "不是沙箱命令，不能通过 shell、pip 或 importlib 寻找" in prompt
    assert "发布不是每个任务的\n必选步骤" in prompt
    assert "Auditor 只读取 evidence_search/evidence_read\n中的真实回执核验，不能代为发布" in prompt


# 函数说明：test_manager_prompt_includes_once_notes_only_when_given
# 用途：回归验证回归测试与测试辅助中的
# `manager_prompt_includes_once_notes_only_when_given` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_manager_prompt`。
# 分支与异常：
#   验证条件：`'本轮一次性指令' in prompt`。
#   验证条件：`'- 这一轮先别碰 s2' in prompt`。
def test_manager_prompt_includes_once_notes_only_when_given() -> None:
    prompt = build_manager_prompt(
        requirements_text=REQUIREMENTS,
        steps=STEPS,
        contract="契约",
        state="状态",
        rounds=[],
        round_index=1,
        round_budget=25,
        once_notes=["这一轮先别碰 s2", "  "],
    )
    assert "本轮一次性指令" in prompt
    assert "- 这一轮先别碰 s2" in prompt


# 函数说明：test_audit_history_keeps_newest_whole_and_clips_oldest
# 用途：回归验证回归测试与测试辅助中的
# `audit_history_keeps_newest_whole_and_clips_oldest` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RoundView` → `format_audit_history`。
# 分支与异常：
#   验证条件：`len(text) <= MANAGER_REPORTS_TOTAL + 200`。
#   验证条件：`'--- Round 10 auditor report' in text`。
#   验证条件：`'--- Round 1 auditor report' not in text`。
#   验证条件：`'更早的' in text`。
def test_audit_history_keeps_newest_whole_and_clips_oldest() -> None:
    rounds = [
        RoundView(i, AuditKind.NORMAL, "s2", "子任务", STEP_REPORT + "\n" + "细节" * 1500, "")
        for i in range(1, 11)
    ]
    text = format_audit_history(rounds)
    assert len(text) <= MANAGER_REPORTS_TOTAL + 200
    assert "--- Round 10 auditor report" in text
    assert "--- Round 1 auditor report" not in text
    assert "更早的" in text


# 函数说明：test_related_reports_select_by_round_ref
# 用途：回归验证回归测试与测试辅助中的 `related_reports_select_by_round_ref` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RoundView` → `format_related_reports`
# 。
# 分支与异常：
#   验证条件：`'报告 2' in text`。
#   验证条件：`'报告 1' not in text and '报告 3' not in text`。
def test_related_reports_select_by_round_ref() -> None:
    rounds = [RoundView(i, AuditKind.NORMAL, "s1", "", f"报告 {i}", "") for i in (1, 2, 3)]
    text = format_related_reports(rounds, ["round_002"])
    assert "报告 2" in text
    assert "报告 1" not in text and "报告 3" not in text


# 函数说明：test_executor_prompt
# 用途：回归验证回归测试与测试辅助中的 `executor_prompt` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_executor_prompt`。
# 分支与异常：
#   验证条件：`REQUIREMENTS in prompt`。
#   验证条件：`'s2 导入数据库\n步骤验收标准: users 表 120 行' in prompt`。
#   验证条件：`'分配的子任务合同:\n下一步: 执行任务' in prompt`。
#   验证条件：`'(任务管理器没有引用相关报告。)' in prompt`。
def test_executor_prompt() -> None:
    prompt = build_executor_prompt(
        requirements_text=REQUIREMENTS,
        contract="契约 X",
        state="状态 Y",
        step=STEPS[1],
        subtask="下一步: 执行任务\n步骤: s2\n任务: 导入",
        related_reports="",
        workspace_root="/workspace",
    )
    assert REQUIREMENTS in prompt
    assert "s2 导入数据库\n步骤验收标准: users 表 120 行" in prompt
    assert "分配的子任务合同:\n下一步: 执行任务" in prompt
    assert "(任务管理器没有引用相关报告。)" in prompt
    assert "- Workspace 根目录: /workspace" in prompt
    assert "直接调用 artifact_publish（Host 工具，不是沙箱命令）" in prompt
    assert "文件使用 workspace 相对路径" in prompt
    assert "保留真实返回的产物 ID、size_bytes、sha256" in prompt
    assert "禁止伪造回执或在沙箱搜索同名入口" in prompt


# 函数说明：test_auditor_prompt_step_scope
# 用途：回归验证回归测试与测试辅助中的 `auditor_prompt_step_scope` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_auditor_prompt`。
# 分支与异常：
#   验证条件：`'- 步骤审计。所属步骤: s2 导入数据库' in prompt`。
#   验证条件：`'其他步骤: (无)' in prompt`。
#   验证条件：`'executor 自然语言输出:\nimported 120' in prompt`。
#   验证条件：`'本次审计依据要求版本 v2' in prompt`。
def test_auditor_prompt_step_scope() -> None:
    prompt = build_auditor_prompt(
        kind=AuditKind.NORMAL,
        requirements_text=REQUIREMENTS,
        contract="契约",
        state="状态",
        task_id="t" * 32,
        workspace_root="/workspace",
        audit_revision=2,
        step=STEPS[1],
        other_steps=STEPS,
        subtask="下一步: 执行任务\n步骤: s2",
        executor_output="imported 120",
    )
    assert "- 步骤审计。所属步骤: s2 导入数据库" in prompt
    assert "其他步骤: (无)" in prompt  # s1 已 done、s3 已取代
    assert "executor 自然语言输出:\nimported 120" in prompt
    assert "本次审计依据要求版本 v2" in prompt
    task_id = "t" * 32
    assert (
        f'evidence_search(query="{task_id}", task_id="{task_id}", '
        'tool_name="artifact_publish")'
        in prompt
    )
    assert "`范围外约束:`" in prompt
    assert "通过 evidence_search/evidence_read 读取 artifact_publish 的真实回执" in prompt
    assert "核对产物 ID、文件大小和 SHA-256" in prompt
    assert "文件哈希不能替代发布成功的证据" in prompt
    assert "不要在\n沙箱查找发布命令，也不要自行发布" in prompt
    assert "只读 artifact_list" in prompt
    assert "该记录不能证明原调用已经返回" in prompt
    assert "仍报告 unknown/blocked，不能弱化要求" in prompt


# 函数说明：test_auditor_prompt_final_scope_lists_superseded_steps
# 用途：回归验证回归测试与测试辅助中的
# `auditor_prompt_final_scope_lists_superseded_steps` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_auditor_prompt` →
# `SupersededView`。
# 分支与异常：
#   验证条件：`'- 最终验收。' in prompt`。
#   验证条件：
# `' - s3 — 原验收: 生成 users.json — 依据 A1: 改为导出 CSV，不要 JSON' in prompt`。
#   验证条件：`'任务管理器给出的核实重点:\nCSV 编码' in prompt`。
#   验证条件：`'executor 自然语言输出' not in prompt`。
def test_auditor_prompt_final_scope_lists_superseded_steps() -> None:
    prompt = build_auditor_prompt(
        kind=AuditKind.FINAL_AUDIT,
        requirements_text=REQUIREMENTS,
        contract="契约",
        state="状态",
        task_id="t" * 32,
        workspace_root="/workspace",
        audit_revision=2,
        superseded=[SupersededView("s3", "生成 users.json", "A1", "改为导出 CSV，不要 JSON", "用户改为 CSV")],
        focus="CSV 编码",
    )
    assert "- 最终验收。" in prompt
    assert "  - s3 — 原验收: 生成 users.json — 依据 A1: 改为导出 CSV，不要 JSON" in prompt
    assert "任务管理器给出的核实重点:\nCSV 编码" in prompt
    assert "executor 自然语言输出" not in prompt


# 函数说明：test_step_audits_require_a_step
# 用途：回归验证回归测试与测试辅助中的 `step_audits_require_a_step` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   kind：`kind`输入或配置值，类型 `AuditKind`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `build_auditor_prompt`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError)`。
@pytest.mark.parametrize("kind", [AuditKind.NORMAL, AuditKind.AUDIT_ONLY, AuditKind.RECOVERY_AUDIT])
def test_step_audits_require_a_step(kind: AuditKind) -> None:
    with pytest.raises(ValueError):
        build_auditor_prompt(
            kind=kind,
            requirements_text=REQUIREMENTS,
            contract=None,
            state=None,
            task_id="t",
            workspace_root="/w",
            audit_revision=1,
        )


# 函数说明：test_recovery_prompt_uses_record
# 用途：回归验证回归测试与测试辅助中的 `recovery_prompt_uses_record` 场景，下方断言说明
# 列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`recovery_record` →
# `build_auditor_prompt`。
# 分支与异常：
#   验证条件：`'- 中断核查。所属步骤: s2' in prompt`。
#   验证条件：`'round_004 的 executor 在运行中被打断（原因: 进程重启）' in prompt`。
#   验证条件：`'write_file data/out.csv' in prompt`。
def test_recovery_prompt_uses_record() -> None:
    record = recovery_record(
        round_index=4,
        reason="进程重启",
        subtask="下一步: 执行任务\n步骤: s2",
        completed_calls=["13:01 run_shell_command python import.py → ev_1234"],
        pending_calls=["write_file data/out.csv"],
    )
    prompt = build_auditor_prompt(
        kind=AuditKind.RECOVERY_AUDIT,
        requirements_text=REQUIREMENTS,
        contract=None,
        state=None,
        task_id="t",
        workspace_root="/w",
        audit_revision=2,
        step=STEPS[1],
        recovery_record=record,
    )
    assert "- 中断核查。所属步骤: s2" in prompt
    assert "round_004 的 executor 在运行中被打断（原因: 进程重启）" in prompt
    assert "write_file data/out.csv" in prompt


# 函数说明：test_synthetic_reports_have_valid_control_headers
# 用途：回归验证回归测试与测试辅助中的 `synthetic_reports_have_valid_control_headers` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`invalid_final_audit_feedback` →
# `invalid_plan_feedback` → `readonly_violation_report` → `no_report_placeholder` →
# `invalid_header_report` → `parse_control_header`。
# 分支与异常：
#   验证条件：`parse_control_header(report) is not None`。
def test_synthetic_reports_have_valid_control_headers() -> None:
    for report in (
        invalid_final_audit_feedback([("s2", "最近 round_005: not_satisfied")]),
        invalid_plan_feedback("没有有效路由"),
        readonly_violation_report(["+ tmp.txt"], "原始报告"),
        no_report_placeholder(),
        invalid_header_report("随便写的"),
    ):
        assert parse_control_header(report) is not None, report


# 函数说明：test_feedback_texts
# 用途：回归验证回归测试与测试辅助中的 `feedback_texts` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`invalid_final_audit_feedback` →
# `supersede_rejected_feedback` → `requirements_updated_feedback` →
# `round_abandoned_feedback`。
# 分支与异常：
#   验证条件：`'s2（最近 round_005: not_satisfied）' in invalid_final_audit_feedback([('
# s2', '最近 round_005: not_satisfied')])`。
#   验证条件：`'依据: A9。原因: 权威要求里不存在修订 A9' in supersede_rejected_feedback(
# 's1', 'A9', '权威要求里不存在修订 A9')`。
#   验证条件：
# `'round_006 的审计依据 v2，期间权威要求已更新到 v3（新增 A2: 用测试库）' in updated`。
#   验证条件：`'round_003 的计划已应用但 Executor 未启动' in applied`。
def test_feedback_texts() -> None:
    assert "s2（最近 round_005: not_satisfied）" in invalid_final_audit_feedback(
        [("s2", "最近 round_005: not_satisfied")]
    )
    assert "依据: A9。原因: 权威要求里不存在修订 A9" in supersede_rejected_feedback(
        "s1", "A9", "权威要求里不存在修订 A9"
    )
    updated = requirements_updated_feedback(
        round_index=6,
        old_revision=2,
        new_revision=3,
        new_amendments=[("A2", "用测试库")],
        step_ids=["s2"],
    )
    assert "round_006 的审计依据 v2，期间权威要求已更新到 v3（新增 A2: 用测试库）" in updated

    applied = round_abandoned_feedback(
        round_index=3, plan_applied=True, reason="权威要求已更新到 v3", added_step_ids=["s4"],
        amendment_ids=["A2"],
    )
    assert "round_003 的计划已应用但 Executor 未启动" in applied
    assert "新增的步骤 s4" in applied and "`取代:` 引用 A2" in applied
    not_applied = round_abandoned_feedback(
        round_index=3, plan_applied=False, reason="Task 在本轮期间被修改"
    )
    assert "尚未应用" in not_applied and "新增的步骤" not in not_applied


# 函数说明：test_format_repair_prompt
# 用途：回归验证回归测试与测试辅助中的 `format_repair_prompt` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_format_repair_prompt` →
# `prompt.rstrip().endswith` → `prompt.rstrip`。
# 分支与异常：
#   验证条件：
# `'步骤验收: satisfied | 步骤验收: not_satisfied | 步骤验收: not_applicable' in prompt`
# 。
#   验证条件：`prompt.rstrip().endswith('不输出 JSON。')`。
#   验证条件：`'上一份 auditor 报告:\n审计事实: 做完了' in prompt`。
def test_format_repair_prompt() -> None:
    prompt = build_format_repair_prompt("审计事实: 做完了")
    assert "步骤验收: satisfied | 步骤验收: not_satisfied | 步骤验收: not_applicable" in prompt
    assert prompt.rstrip().endswith("不输出 JSON。")
    assert "上一份 auditor 报告:\n审计事实: 做完了" in prompt


# 函数说明：test_final_response_prompt
# 用途：回归验证回归测试与测试辅助中的 `final_response_prompt` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RoundView` → `STEP_REPORT.replace` →
# `format_audit_findings` → `build_final_response_prompt`。
# 分支与异常：
#   验证条件：`'第 2 轮（步骤审计）\n审计事实: 已读取 120 行' in findings`。
#   验证条件：`'全部完成' * 500 in findings`。
#   验证条件：`'验收约束反查' not in findings and '状态: complete' not in findings`。
#   验证条件：`REQUIREMENTS in prompt`。
def test_final_response_prompt() -> None:
    rounds = [
        RoundView(2, AuditKind.NORMAL, "s1", "", STEP_REPORT, ""),
        RoundView(
            5, AuditKind.FINAL_AUDIT, None, "",
            STEP_REPORT.replace("已读取 120 行", "全部完成" * 500), "",
        ),
    ]
    findings = format_audit_findings(rounds, final_round_index=5)
    assert "第 2 轮（步骤审计）\n审计事实: 已读取 120 行" in findings
    assert "全部完成" * 500 in findings  # 最终验收报告完整保留
    assert "验收约束反查" not in findings and "状态: complete" not in findings

    prompt = build_final_response_prompt(
        requirements_text=REQUIREMENTS,
        outcome="complete",
        abort_reason=None,
        state="已完成",
        findings=findings,
        deliverables=["导出的 CSV 在 out/users.csv"],
    )
    assert REQUIREMENTS in prompt
    assert "运行结果: complete（结束原因: 无）" in prompt
    assert "导出的 CSV 在 out/users.csv" in prompt

    blocked = build_final_response_prompt(
        requirements_text=REQUIREMENTS, outcome="blocked", abort_reason="缺少权限", state=None,
        findings=findings, deliverables=["不应出现"],
    )
    assert "(没有独立的交付正文。)" in blocked and "不应出现" not in blocked
