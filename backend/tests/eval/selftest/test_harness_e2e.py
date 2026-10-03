"""用脚本模型把整条评测链路跑通：搭现场 → 驱动 → 收集结果 → 检查 → 报告。

模型是脚本，其余都是真实组件（数据库、工具执行、审批、任务、长任务调度）。
"""

from __future__ import annotations

from app.models.types import ModelRequest
from app.runtime.mea.prompts import (
    AUDITOR_INSTRUCTIONS,
    EXECUTOR_INSTRUCTIONS,
    FINAL_RESPONSE_INSTRUCTIONS,
    MANAGER_INSTRUCTIONS,
)
from tests.eval.selftest.scripted import (
    after_tool,
    answer,
    call,
    last_user,
    scripted_factory,
)
from tests.eval.session import run_cases
from tests.eval.spec import CURRENT, Case


# 函数说明：_case
# 用途：返回
# `Case.model_validate({'suite': 'behavior', 'title': '自测', 'why': '自测', **fields})`
# ，提供 回归测试与测试辅助 的派生值。
# 参数：
#   **fields：额外关键字参数，按实现处理或转交。
# 返回：类型 `Case`；返回
# `Case.model_validate({'suite': 'behavior', 'title': '自测', 'why': '自测', **fields})`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Case.model_validate`。
def _case(**fields) -> Case:
    return Case.model_validate({"suite": "behavior", "title": "自测", "why": "自测", **fields})


# 函数说明：test_chat_case_runs_tools_and_passes_checks
# 用途：回归验证回归测试与测试辅助中的 `chat_case_runs_tools_and_passes_checks` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_case` → `run_cases` →
# `scripted_factory`。
# 分支与异常：
#   验证条件：`len(report.attempts) == 2`。
#   验证条件：`item.passed`。
#   验证条件：`item.model_calls == 2`。
#   验证条件：`item.chargeable_tokens == 240`。
async def test_chat_case_runs_tools_and_passes_checks() -> None:
    # 函数说明：test_chat_case_runs_tools_and_passes_checks.reply
    # 用途：回复回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：按分支返回 `answer('max_workers 是 17。')`；
    # `call('read_file', path='config/app.toml')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`after_tool` → `answer` → `call`。
    # 分支与异常：
    #   当 `after_tool(request)` 时，返回 `answer('max_workers 是 17。')`。
    def reply(request: ModelRequest):
        if after_tool(request):
            return answer("max_workers 是 17。")
        return call("read_file", path="config/app.toml")

    case = _case(
        id="b90-selftest-read",
        setup={"files": {"config/app.toml": "max_workers = 17\n"}},
        turns=["max_workers 是多少？"],
        checks=[
            {"no_run_errors": True},
            {"tool_used": {"name": "read_file", "success": True}},
            {"answer_has": ["17"]},
            {"task_count": 0},
            {"artifact": False},
        ],
    )
    report = await run_cases([case], variant=CURRENT, factory=scripted_factory(reply), repeat=2)
    assert len(report.attempts) == 2
    for item in report.attempts:
        assert item.passed, item.verdicts
        assert item.model_calls == 2
        assert item.chargeable_tokens == 240


# 函数说明：test_denied_approval_is_recorded_and_tool_does_not_run
# 用途：回归验证回归测试与测试辅助中的
# `denied_approval_is_recorded_and_tool_does_not_run` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_case` → `run_cases` →
# `scripted_factory`。
# 分支与异常：
#   验证条件：`item.passed`。
async def test_denied_approval_is_recorded_and_tool_does_not_run() -> None:
    # 函数说明：test_denied_approval_is_recorded_and_tool_does_not_run.reply
    # 用途：回复回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：按分支返回 `answer('命令被拒绝，未执行。')`；
    # `call('run_shell_command', command='echo hi')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`after_tool` → `answer` → `call`。
    # 分支与异常：
    #   当 `after_tool(request)` 时，返回 `answer('命令被拒绝，未执行。')`。
    def reply(request: ModelRequest):
        if after_tool(request):
            return answer("命令被拒绝，未执行。")
        return call("run_shell_command", command="echo hi")

    case = _case(
        id="b91-selftest-deny",
        approvals="deny",
        turns=["执行 echo hi"],
        checks=[
            {"approval": {"decision": "denied"}},
            {"tool_used": {"name": "run_shell_command", "success": False}},
            {"answer_has": ["未执行"]},
        ],
    )
    report = await run_cases([case], variant=CURRENT, factory=scripted_factory(reply))
    (item,) = report.attempts
    assert item.passed, item.verdicts


# 函数说明：test_failing_check_is_reported_with_detail
# 用途：回归验证回归测试与测试辅助中的 `failing_check_is_reported_with_detail` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_case` → `run_cases` →
# `scripted_factory`。
# 分支与异常：
#   验证条件：`not item.passed`。
#   验证条件：`'不会出现的词' in item.verdicts[0]['detail']`。
async def test_failing_check_is_reported_with_detail() -> None:
    case = _case(
        id="b92-selftest-fail",
        turns=["你好"],
        checks=[{"answer_has": ["不会出现的词"]}],
    )
    report = await run_cases([case], variant=CURRENT, factory=scripted_factory(lambda _: answer("你好！")))
    (item,) = report.attempts
    assert not item.passed
    assert "不会出现的词" in item.verdicts[0]["detail"]


# 函数说明：test_docker_requirement_skips_without_starting_app
# 用途：回归验证回归测试与测试辅助中的 `docker_requirement_skips_without_starting_app`
# 场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`session.docker_available.cache_clear`
#  → `_case` → `run_cases` → `scripted_factory`。
# 分支与异常：
#   验证条件：`item.status == 'skipped'`。
# 副作用与资源：
#   更新对象字段：`session.shutil.which`。
async def test_docker_requirement_skips_without_starting_app() -> None:
    import tests.eval.session as session

    session.docker_available.cache_clear()
    original = session.shutil.which
    session.shutil.which = lambda _: None
    try:
        case = _case(id="b93-selftest-docker", requires=["docker"], turns=["x"], checks=[{"no_tools": True}])
        report = await run_cases([case], variant=CURRENT, factory=scripted_factory(lambda _: answer("x")))
    finally:
        session.shutil.which = original
        session.docker_available.cache_clear()
    (item,) = report.attempts
    assert item.status == "skipped"


# 函数说明：_manager_plan
# 用途：返回 `f'当前任务状态:\n- 已完成: 无\n\n任务契约:\n- 目标状态: hello.py 可运行\n\
# n步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推进\…`，提供 回归测试与测试辅助 的
# 派生值。
# 参数：
#   route：路由输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `f'当前任务状态:\n- 已完成: 无\n\n任务契约:\n- 目标状态: hello.
# py 可运行\n\n步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推进\…`。
def _manager_plan(route: str) -> str:
    return (
        "当前任务状态:\n- 已完成: 无\n\n任务契约:\n- 目标状态: hello.py 可运行\n\n"
        f"步骤更新:\n无\n\n依赖判断:\n- 本轮路由理由: 按依赖推进\n\n{route}"
    )


EXECUTE_S1 = (
    "下一步: 执行任务\n步骤: s1\n任务: 创建 hello.py\n验收标准: 见步骤验收\n"
    "相关审计报告: 无\n相关已审计状态: 无\n边界: 只改 hello.py"
)
FINAL_AUDIT = "下一步: 最终验收\n验收重点: hello.py 内容"
AUDIT_REPORT = (
    "状态: complete\n完整性: clean\n契约审计: aligned\n步骤验收: satisfied\n"
    "审计事实: 已独立核验。\n\n验收约束反查:\n"
    "契约结论: aligned\n阻断约束: 无\n范围外约束: 无\n"
    "给任务管理器的状态更新: 见上"
)


# 函数说明：test_mea_case_collects_rounds_steps_and_final_answer
# 用途：回归验证回归测试与测试辅助中的 `mea_case_collects_rounds_steps_and_final_answer`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_manager_plan` →
# `Case.model_validate` → `run_cases` → `scripted_factory`。
# 分支与异常：
#   验证条件：`item.passed`。
#   验证条件：`item.model_calls >= 4`。
async def test_mea_case_collects_rounds_steps_and_final_answer() -> None:
    manager_turns = [_manager_plan(EXECUTE_S1), _manager_plan(FINAL_AUDIT)]

    # 函数说明：test_mea_case_collects_rounds_steps_and_final_answer.reply
    # 用途：回复回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：按分支返回 `answer('hello.py 已创建并通过审计。')`；
    # `answer(manager_turns.pop(0) if manager_turns else _manager_plan(FINAL_AUDIT))`；
    # `answer('已写入 hello.py。')`；
    # `call('write_file', path='hello.py', content='print("Hello MuHarness")\n')` 等 6
    # 种表达式。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`last_user` → `prompt.startswith`
    # → `answer` → `manager_turns.pop` → `_manager_plan` → `after_tool`；另有 1 个调用点
    # 。
    # 分支与异常：
    #   当 `prompt.startswith(FINAL_RESPONSE_INSTRUCTIONS[:30])` 时，返回
    # `answer('hello.py 已创建并通过审计。')`。
    #   当 `prompt.startswith(MANAGER_INSTRUCTIONS[:30])` 时，返回 `answer(…)`。
    #   `prompt.startswith(EXECUTOR_INSTRUCTIONS[:30])` 分支在完成前置处理后返回
    # `call(…)`。
    #   当 `after_tool(request)` 时，返回 `answer('已写入 hello.py。')`。
    # 闭包依赖：从外层读取 `manager_turns`。
    def reply(request: ModelRequest):
        prompt = last_user(request)
        if prompt.startswith(FINAL_RESPONSE_INSTRUCTIONS[:30]):
            return answer("hello.py 已创建并通过审计。")
        if prompt.startswith(MANAGER_INSTRUCTIONS[:30]):
            return answer(manager_turns.pop(0) if manager_turns else _manager_plan(FINAL_AUDIT))
        if prompt.startswith(EXECUTOR_INSTRUCTIONS[:30]):
            if after_tool(request):
                return answer("已写入 hello.py。")
            return call("write_file", path="hello.py", content='print("Hello MuHarness")\n')
        if prompt.startswith(AUDITOR_INSTRUCTIONS[:20]):
            return answer(AUDIT_REPORT)
        return answer("未识别的请求")

    case = Case.model_validate(
        {
            "id": "m90-selftest",
            "suite": "mea",
            "title": "自测",
            "why": "自测",
            "mea": {
                "request": "创建 hello.py",
                "title": "创建 hello.py",
                "goal": "hello.py 打印 Hello MuHarness",
                "round_budget": 5,
                "steps": [{"id": "s1", "title": "创建 hello.py", "acceptance": "hello.py 包含 print"}],
            },
            "checks": [
                {"mea_status": "completed"},
                {"mea_steps_done": "all"},
                {"file": {"path": "hello.py", "contains": ["Hello MuHarness"]}},
                {"answer_has": ["通过审计"]},
                {"mea_rounds_at_most": 3},
                {"mea_no_role_violation": True},
            ],
        }
    )
    report = await run_cases([case], variant=CURRENT, factory=scripted_factory(reply), keep_outcomes=True)
    (item,) = report.attempts
    assert item.passed, (item.error, item.verdicts)
    assert item.model_calls >= 4


# 函数说明：test_stage_start_failure_is_recorded_per_attempt
# 用途：回归验证回归测试与测试辅助中的 `stage_start_failure_is_recorded_per_attempt` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_case` → `run_cases`。
# 分支与异常：
#   验证条件：`[item.status for item in report.attempts] == ['error', 'error']`。
#   验证条件：`'没有配置模型服务商' in (report.attempts[0].error or '')`。
async def test_stage_start_failure_is_recorded_per_attempt() -> None:
    # 函数说明：test_stage_start_failure_is_recorded_per_attempt.broken_factory
    # 用途：处理回归测试与测试辅助中的 `broken_factory` 数据；结果及边界条件见下方说明。
    # 参数：
    #   paths：待处理的路径集合。
    #   variant：`variant`输入或配置值。
    # 返回：不返回结果值（隐式 None）。
    def broken_factory(paths, variant):
        raise RuntimeError("没有配置模型服务商")

    case = _case(id="b94-selftest-broken", turns=["x"], checks=[{"no_tools": True}])
    report = await run_cases([case], variant=CURRENT, factory=broken_factory, repeat=2)
    assert [item.status for item in report.attempts] == ["error", "error"]
    assert "没有配置模型服务商" in (report.attempts[0].error or "")
