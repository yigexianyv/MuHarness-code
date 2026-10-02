
from __future__ import annotations

import pytest

from app.runtime.mea.parsing import (
    COMPLETION_GUARD_NOTE,
    AuditStatus,
    ContractAudit,
    ControlHeader,
    Integrity,
    Route,
    StepAcceptance,
    apply_blocking_guard,
    audit_findings_body,
    clip_preserve,
    has_blocking_constraints,
    parse_control_header,
    parse_manager_output,
    parse_route,
    parse_step_updates,
    strip_control_header,
)

# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


# 函数说明：test_parse_route_single_line
# 用途：回归验证回归测试与测试辅助中的 `parse_route_single_line` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   line：传给 `parse_route` 的输入，类型 `str`。
#   expected：`expected`输入或配置值，类型 `Route`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_route`。
# 分支与异常：
#   验证条件：`parse_route(line) is expected`。
@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("下一步: 执行任务", Route.EXECUTE),
        ("下一步: 仅审计", Route.AUDIT_ONLY),
        ("下一步: 最终验收", Route.FINAL_AUDIT),
        ("下一步: 请示用户", Route.ASK),
        ("下一步: 阻塞", Route.BLOCKED),
        ("下一步: 完成", Route.INVALID),  # LHH 的“完成”路由已取消
        ("下一步：执行任务", Route.EXECUTE),
        ("**下一步: 执行任务**", Route.EXECUTE),
        ("`下一步: 仅审计`", Route.AUDIT_ONLY),
        ("- 下一步: 最终验收", Route.FINAL_AUDIT),
        ("下一步: 执行任务 — 先完成 s2 的导入", Route.EXECUTE),
        ("下一步: 执行任务（s2）", Route.EXECUTE),
        ("下一步: 请示用户 // 缺少数据库地址", Route.ASK),
        ("下一步: 执行任务吧", Route.INVALID),
        ("没有路由", Route.INVALID),
        ("", Route.INVALID),
    ],
)
def test_parse_route_single_line(line: str, expected: Route) -> None:
    assert parse_route(line) is expected


# 函数说明：test_parse_route_takes_last_valid_match
# 用途：回归验证回归测试与测试辅助中的 `parse_route_takes_last_valid_match` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_route`。
# 分支与异常：
#   验证条件：`parse_route(text) is Route.FINAL_AUDIT`。
def test_parse_route_takes_last_valid_match() -> None:
    text = (
        "依赖判断:\n- 上一轮我写了 `下一步: 执行任务`，现在改为验收。\n"
        "下一步: 执行任务\n"
        "（更正）\n"
        "下一步: 最终验收\n"
        "验收重点: CSV 导出\n"
        "下一步: 以后再说\n"  # 不合法的路由行不会覆盖前面的合法路由
    )
    assert parse_route(text) is Route.FINAL_AUDIT


# ---------------------------------------------------------------------------
# Manager 输出
# ---------------------------------------------------------------------------

MANAGER_OUTPUT = """我先整理一下。

当前任务状态:
- 已完成: s1 读取 CSV（round_002）
- 未完成: s2 导入数据库
- 阻塞/风险: 无
- 不可信/不可复用: 无

任务契约:
- 目标状态: users 表包含 CSV 中全部 120 行
- 验收约束: 不修改表结构

步骤更新:
- 新增: 校验导入行数 | 验收: users 表行数等于 CSV 行数 120
- 取代: s3 | 依据: A1 | 原因: 用户改为只导出 CSV，不再生成 JSON
- 把 s4 标成完成

依赖判断:
- 目标状态: 数据已导入
- 本轮路由理由: s1 已由 round_002 确认

下一步: 执行任务 — 导入
步骤: s2（导入数据库）
任务: 用 scripts/import.py 把 data/users.csv 导入 users 表
验收标准: 命令退出码为 0，输出 imported 120
相关审计报告: round_002（CSV 结构）、round_2 重复引用
相关已审计状态: round_001
边界: 不要修改 schema.sql
"""


# 函数说明：test_parse_manager_output_sections
# 用途：回归验证回归测试与测试辅助中的 `parse_manager_output_sections` 场景，下方断言说
# 明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_manager_output` →
# `output.plan_text.startswith` → `output.subtask.startswith` →
# `output.state.startswith`。
# 分支与异常：
#   验证条件：`output.route is Route.EXECUTE`。
#   验证条件：`output.plan_text.startswith('当前任务状态:')`。
#   验证条件：`output.subtask.startswith('下一步: 执行任务')`。
#   验证条件：`'边界: 不要修改 schema.sql' in output.subtask`。
def test_parse_manager_output_sections() -> None:
    output = parse_manager_output(MANAGER_OUTPUT)

    assert output.route is Route.EXECUTE
    assert output.plan_text.startswith("当前任务状态:")
    assert output.subtask.startswith("下一步: 执行任务")
    assert "边界: 不要修改 schema.sql" in output.subtask
    assert output.state is not None and output.state.startswith("- 已完成: s1")
    assert "任务契约" not in output.state
    assert output.contract is not None and "不修改表结构" in output.contract
    assert output.step_id == "s2"
    assert output.related_refs == ("round_002", "round_001")
    assert output.focus is None
    assert output.question is None

    updates = output.step_updates
    assert [(s.title, s.acceptance) for s in updates.added] == [
        ("校验导入行数", "users 表行数等于 CSV 行数 120")
    ]
    assert [(s.step_id, s.basis, s.reason) for s in updates.superseded] == [
        ("s3", "A1", "用户改为只导出 CSV，不再生成 JSON")
    ]
    assert updates.invalid_lines == ("- 把 s4 标成完成",)


# 函数说明：test_parse_manager_output_without_route
# 用途：回归验证回归测试与测试辅助中的 `parse_manager_output_without_route` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_manager_output`。
# 分支与异常：
#   验证条件：`output.route is Route.INVALID`。
#   验证条件：`output.subtask == ''`。
#   验证条件：`output.step_id is None`。
def test_parse_manager_output_without_route() -> None:
    output = parse_manager_output("当前任务状态:\n- 未完成: 全部\n")

    assert output.route is Route.INVALID
    assert output.subtask == ""
    assert output.step_id is None


# 函数说明：test_parse_manager_output_audit_only_and_final_focus
# 用途：回归验证回归测试与测试辅助中的 `parse_manager_output_audit_only_and_final_focus`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_manager_output`。
# 分支与异常：
#   验证条件：`audit_only.route is Route.AUDIT_ONLY`。
#   验证条件：`audit_only.step_id == 's2'`。
#   验证条件：`audit_only.focus == '数据库里是否已有 120 行'`。
#   验证条件：`final.route is Route.FINAL_AUDIT`。
def test_parse_manager_output_audit_only_and_final_focus() -> None:
    audit_only = parse_manager_output(
        "当前任务状态:\n- 未完成: s2\n\n下一步: 仅审计\n步骤: `s2`\n核实重点: 数据库里是否已有 120 行"
    )
    assert audit_only.route is Route.AUDIT_ONLY
    assert audit_only.step_id == "s2"
    assert audit_only.focus == "数据库里是否已有 120 行"

    final = parse_manager_output(
        "当前任务状态:\n- 已完成: 全部\n\n下一步: 最终验收\n验收重点:\n- CSV 编码\n- 不改表结构"
    )
    assert final.route is Route.FINAL_AUDIT
    assert final.step_id is None
    assert final.focus == "- CSV 编码\n- 不改表结构"


# 函数说明：test_parse_manager_output_question_and_choices
# 用途：回归验证回归测试与测试辅助中的 `parse_manager_output_question_and_choices` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_manager_output`。
# 分支与异常：
#   验证条件：`explicit.route is Route.ASK`。
#   验证条件：`explicit.question == '用哪个数据库？'`。
#   验证条件：`explicit.choices == ('测试库', '生产库')`。
#   验证条件：`yes_no.choices == ('是', '否')`。
def test_parse_manager_output_question_and_choices() -> None:
    explicit = parse_manager_output(
        "当前任务状态:\n- 阻塞: 缺连接串\n\n下一步: 请示用户\n问题: 用哪个数据库？\n选项: 测试库 | 生产库"
    )
    assert explicit.route is Route.ASK
    assert explicit.question == "用哪个数据库？"
    assert explicit.choices == ("测试库", "生产库")

    yes_no = parse_manager_output("下一步: 请示用户\n问题: 是否覆盖已有数据？")
    assert yes_no.choices == ("是", "否")


# 函数说明：test_parse_step_updates_none_and_bullets
# 用途：回归验证回归测试与测试辅助中的 `parse_step_updates_none_and_bullets` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_step_updates`。
# 分支与异常：
#   验证条件：`parse_step_updates('无').empty`。
#   验证条件：`parse_step_updates(None).empty`。
#   验证条件：`updates.added[0].title == '写 README'`。
#   验证条件：`updates.superseded[0].basis == 'A2'`。
def test_parse_step_updates_none_and_bullets() -> None:
    assert parse_step_updates("无").empty
    assert parse_step_updates(None).empty
    updates = parse_step_updates("1. 新增: 写 README | 验收: README 含安装步骤\n* 取代: s1｜依据: a2｜原因: 改需求")
    assert updates.added[0].title == "写 README"
    assert updates.superseded[0].basis == "A2"
    assert not updates.invalid_lines


# ---------------------------------------------------------------------------
# Auditor 控制头
# ---------------------------------------------------------------------------


# 函数说明：test_parse_control_header_valid
# 用途：回归验证回归测试与测试辅助中的 `parse_control_header_valid` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_control_header` →
# `ControlHeader`。
# 分支与异常：
#   验证条件：`header == ControlHeader(AuditStatus.COMPLETE, Integrity.CLEAN,
# ContractAudit.ALIGNED, StepAcceptance.SATISFIED)`。
def test_parse_control_header_valid() -> None:
    header = parse_control_header(
        "\n状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n审计事实: ..."
    )
    assert header == ControlHeader(
        AuditStatus.COMPLETE, Integrity.CLEAN, ContractAudit.ALIGNED, StepAcceptance.SATISFIED
    )


# 函数说明：test_parse_control_header_tolerates_markup_and_chinese_values
# 用途：回归验证回归测试与测试辅助中的
# `parse_control_header_tolerates_markup_and_chinese_values` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_control_header` →
# `ControlHeader`。
# 分支与异常：
#   验证条件：`header == ControlHeader(AuditStatus.INCOMPLETE, Integrity.SUSPECT,
# ContractAudit.NEEDS_REVISION, StepAcceptance.NOT_SATISFIED)`。
def test_parse_control_header_tolerates_markup_and_chinese_values() -> None:
    header = parse_control_header(
        "**状态: 未完成**\n完整性：`suspect`\n契约审计: needs-revision\n**步骤验收**: not satisfied"
    )
    assert header == ControlHeader(
        AuditStatus.INCOMPLETE,
        Integrity.SUSPECT,
        ContractAudit.NEEDS_REVISION,
        StepAcceptance.NOT_SATISFIED,
    )


# 函数说明：test_parse_control_header_invalid
# 用途：回归验证回归测试与测试辅助中的 `parse_control_header_invalid` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_control_header`。
# 分支与异常：
#   验证条件：`parse_control_header(text) is None`。
@pytest.mark.parametrize(
    "text",
    [
        "状态: complete\n完整性: clean\n契约审计: aligned",  # 只有三行（LHH 格式）
        "完整性: clean\n状态: complete\n契约审计: aligned\n步骤验收: satisfied",  # 顺序错
        "状态: done\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied",  # 值不合法
        "状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: mostly",
        "审计事实: 很好\n状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied",
        "",
    ],
)
def test_parse_control_header_invalid(text: str) -> None:
    assert parse_control_header(text) is None


# 函数说明：test_strip_control_header
# 用途：回归验证回归测试与测试辅助中的 `strip_control_header` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`strip_control_header`。
# 分支与异常：
#   验证条件：`strip_control_header(report) == '审计事实: 导入成功'`。
def test_strip_control_header() -> None:
    report = "状态: complete\n\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n审计事实: 导入成功"
    assert strip_control_header(report) == "审计事实: 导入成功"


# ---------------------------------------------------------------------------
# 阻断约束降级
# ---------------------------------------------------------------------------

_CLEAN = ControlHeader(
    AuditStatus.COMPLETE, Integrity.CLEAN, ContractAudit.ALIGNED, StepAcceptance.SATISFIED
)


# 函数说明：_report
# 用途：返回 `'状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n审
# 计事实: s1 已完成\n\n验收约束反查:\n契约结论:…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   backcheck：`backcheck`输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `'状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收:
# satisfied\n审计事实: s1 已完成\n\n验收约束反查:\n契约结论:…`。
def _report(backcheck: str) -> str:
    return (
        "状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n"
        "审计事实: s1 已完成\n\n验收约束反查:\n契约结论: aligned\n" + backcheck
        + "\n可能评分风险: 无\n给任务管理器的状态更新: s1 已满足"
    )


# 函数说明：test_blocking_guard_does_not_downgrade
# 用途：回归验证回归测试与测试辅助中的 `blocking_guard_does_not_downgrade` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   backcheck：传给 `_report` 的输入，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_report` → `has_blocking_constraints`
#  → `apply_blocking_guard`。
# 分支与异常：
#   验证条件：`not has_blocking_constraints(report)`。
#   验证条件：`not result.downgraded`。
#   验证条件：`result.header == _CLEAN`。
#   验证条件：`result.report == report`。
@pytest.mark.parametrize(
    "backcheck",
    [
        "阻断约束: 无",
        "阻断约束：无（本步骤范围内全部 verified）",
        "阻断约束:\n- 无",
        "阻断约束: none",
        # 只有范围外约束非空：属于其他步骤的待办，不降级
        "阻断约束: 无\n范围外约束:\n- s2 导出 CSV 尚未完成\n- s3 README 未写",
        "阻断约束:\n范围外约束:\n- s2 导出 CSV 尚未完成",
    ],
)
def test_blocking_guard_does_not_downgrade(backcheck: str) -> None:
    report = _report(backcheck)
    assert not has_blocking_constraints(report)
    result = apply_blocking_guard(_CLEAN, report)
    assert not result.downgraded
    assert result.header == _CLEAN
    assert result.report == report


# 函数说明：test_blocking_guard_downgrades
# 用途：回归验证回归测试与测试辅助中的 `blocking_guard_downgrades` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   backcheck：传给 `_report` 的输入，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_report` → `has_blocking_constraints`
#  → `apply_blocking_guard` → `ControlHeader` → `result.report.endswith`。
# 分支与异常：
#   验证条件：`has_blocking_constraints(report)`。
#   验证条件：`result.downgraded`。
#   验证条件：`result.header == ControlHeader(AuditStatus.INCOMPLETE, Integrity.CLEAN,
# ContractAudit.UNKNOWN, StepAcceptance.NOT_SATISFIED)`。
#   验证条件：`result.report.endswith(COMPLETION_GUARD_NOTE)`。
@pytest.mark.parametrize(
    "backcheck",
    [
        "阻断约束: users 表行数未验证（unknown）",
        "阻断约束:\n- 表结构是否保持不变: unknown\n范围外约束: 无",
        "**阻断约束**:\n- 编码: violated",
    ],
)
def test_blocking_guard_downgrades(backcheck: str) -> None:
    report = _report(backcheck)
    assert has_blocking_constraints(report)
    result = apply_blocking_guard(_CLEAN, report)
    assert result.downgraded
    assert result.header == ControlHeader(
        AuditStatus.INCOMPLETE,
        Integrity.CLEAN,
        ContractAudit.UNKNOWN,
        StepAcceptance.NOT_SATISFIED,
    )
    assert result.report.endswith(COMPLETION_GUARD_NOTE)


# 函数说明：test_blocking_guard_downgrades_satisfied_even_when_status_incomplete
# 用途：回归验证回归测试与测试辅助中的
# `blocking_guard_downgrades_satisfied_even_when_status_incomplete` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ControlHeader` →
# `apply_blocking_guard` → `_report`。
# 分支与异常：
#   验证条件：`result.header.step_acceptance is StepAcceptance.NOT_SATISFIED`。
#   验证条件：`result.header.contract_audit is ContractAudit.UNKNOWN`。
def test_blocking_guard_downgrades_satisfied_even_when_status_incomplete() -> None:
    header = ControlHeader(
        AuditStatus.INCOMPLETE, Integrity.CLEAN, ContractAudit.ALIGNED, StepAcceptance.SATISFIED
    )
    result = apply_blocking_guard(header, _report("阻断约束: 行数 unknown"))
    assert result.header.step_acceptance is StepAcceptance.NOT_SATISFIED
    assert result.header.contract_audit is ContractAudit.UNKNOWN


# 函数说明：test_blocking_guard_noop_when_already_failing
# 用途：回归验证回归测试与测试辅助中的 `blocking_guard_noop_when_already_failing` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ControlHeader` →
# `apply_blocking_guard` → `_report`。
# 分支与异常：
#   验证条件：`not result.downgraded`。
#   验证条件：`result.header == header`。
def test_blocking_guard_noop_when_already_failing() -> None:
    header = ControlHeader(
        AuditStatus.BLOCKED, Integrity.SUSPECT, ContractAudit.INVALID, StepAcceptance.NOT_SATISFIED
    )
    result = apply_blocking_guard(header, _report("阻断约束: 行数 unknown"))
    assert not result.downgraded
    assert result.header == header


# ---------------------------------------------------------------------------
# 其他
# ---------------------------------------------------------------------------


# 函数说明：test_audit_findings_body_drops_header_and_protocol
# 用途：回归验证回归测试与测试辅助中的 `audit_findings_body_drops_header_and_protocol`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`audit_findings_body` → `_report`。
# 分支与异常：
#   验证条件：`body == '审计事实: s1 已完成'`。
def test_audit_findings_body_drops_header_and_protocol() -> None:
    body = audit_findings_body(_report("阻断约束: 无"))
    assert body == "审计事实: s1 已完成"


# 函数说明：test_clip_preserve_keeps_head_and_tail
# 用途：回归验证回归测试与测试辅助中的 `clip_preserve_keeps_head_and_tail` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`clip_preserve` → `clipped.startswith`
#  → `clipped.endswith`。
# 分支与异常：
#   验证条件：`clipped.startswith('A' * 650)`。
#   验证条件：`clipped.endswith('C' * 350)`。
#   验证条件：`'已截断 700 字符' in clipped`。
#   验证条件：`clip_preserve('short', 1000) == 'short'`。
def test_clip_preserve_keeps_head_and_tail() -> None:
    text = "A" * 650 + "B" * 700 + "C" * 350
    clipped = clip_preserve(text, 1000)
    assert clipped.startswith("A" * 650)
    assert clipped.endswith("C" * 350)
    assert "已截断 700 字符" in clipped
    assert clip_preserve("short", 1000) == "short"
    assert clip_preserve("anything", 0) == "anything"


# 函数说明：test_bold_blocking_header_colon
# 用途：回归验证回归测试与测试辅助中的 `bold_blocking_header_colon` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   line：传给 `has_blocking_constraints` 的输入。
#   blocked：`blocked`输入或配置值。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`has_blocking_constraints`。
# 分支与异常：
#   验证条件：`has_blocking_constraints(line) is blocked`。
@pytest.mark.parametrize("line,blocked", [
    ("- **阻断约束:** 无（本次审计范围内所有 blocking 约束均 verified）。", False),
    ("- **阻断约束:** 数据丢失", True),
    ("**阻断约束**: 无", False),
])
def test_bold_blocking_header_colon(line, blocked):
    from app.runtime.mea.parsing import has_blocking_constraints
    assert has_blocking_constraints(line) is blocked


# 函数说明：test_route_first_separates_dispatch_from_state
# 用途：回归验证回归测试与测试辅助中的 `route_first_separates_dispatch_from_state` 场景
# ，下方断言说明列出实际通过条件。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_manager_output`。
# 分支与异常：
#   验证条件：`parsed.step_id == 's4'`。
#   验证条件：`parsed.focus == '测试通过'`。
#   验证条件：`parsed.state == 's4 待审计'`。
#   验证条件：`parsed.contract is None`。
def test_route_first_separates_dispatch_from_state():
    from app.runtime.mea.parsing import parse_manager_output
    parsed = parse_manager_output(
        "下一步: 仅审计\n步骤: s4\n核实重点: 测试通过\n"
        "当前任务状态: s4 待审计\n步骤更新: 无\n依赖判断: 文件已存在"
    )
    assert parsed.step_id == "s4"
    assert parsed.focus == "测试通过"
    assert parsed.state == "s4 待审计"
    assert parsed.contract is None
    assert "当前任务状态" not in parsed.subtask
