"""按变体跑一批用例，得到一份 ``RunReport``。"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from functools import cache

from .checks import evaluate, validate_spec
from .drive import drive
from .outcome import Outcome
from .report import RunReport, now_iso, record
from .spec import Case, Variant
from .stage import AppFactory, open_stage

Progress = Callable[[str], None]


# 函数说明：docker_available
# 用途：在回归测试与测试辅助中处理 `docker_available`，通过 `shutil.which` 完成首个内部
# 处理步骤。
# 返回：类型 `bool`；按分支返回 `False`；
# `subprocess.run(['docker', 'info'], capture_output=True, timeout=15).returncode == 0`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`shutil.which` → `subprocess.run`。
# 分支与异常：
#   当 `shutil.which('docker') is None` 时，返回 `False`。
#   捕获 `(OSError, subprocess.TimeoutExpired)` 后，返回 `False`。
@cache
def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# 函数说明：preflight
# 用途：先把所有用例的检查项名字校验一遍，免得跑到一半才发现写错。
# 参数：
#   cases：`cases`输入或配置值，类型 `Sequence[Case]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_spec`。
# 分支与异常：
#   捕获 `ValueError` 后，转换或抛出 `ValueError(f'{case.id}（{case.source}）: {exc}')`
# 。
def preflight(cases: Sequence[Case]) -> None:
    """先把所有用例的检查项名字校验一遍，免得跑到一半才发现写错。"""

    for case in cases:
        for spec in case.checks:
            try:
                validate_spec(spec)
            except ValueError as exc:
                raise ValueError(f"{case.id}（{case.source}）: {exc}") from exc


# 函数说明：run_cases
# 用途：运行`cases`，供回归测试与测试辅助使用。
# 参数：
#   cases：传给 `preflight` 的输入，类型 `Sequence[Case]`。
#   variant：传给 `open_stage` 的输入，类型 `Variant`。
#   factory：构造目标依赖的工厂，类型 `AppFactory`。
#   repeat：`repeat`输入或配置值，类型 `int`；默认 `1`。
#   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
#   model：模型名称，类型 `str | None`；默认 `None`。
#   keep_outcomes：`keep_outcomes`输入或配置值，类型 `bool`；默认 `False`。
#   keep_stage：`keep_stage`输入或配置值，类型 `bool`；默认 `False`。
#   progress：`progress`输入或配置值，类型 `Progress | None`；默认 `None`。
# 返回：类型 `RunReport`；返回 `report`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`preflight` → `RunReport` → `now_iso`
# → `docker_available` → `Outcome` → `record`；另有 5 个调用点。
# 分支与异常：
#   `missing` 分支在完成前置处理后跳过当前循环项。
#   捕获 `Exception` 后，执行异常处理调用 `Outcome`、`type`。
# 副作用与资源：
#   更新对象字段：`report.finished_at`。
async def run_cases(
    cases: Sequence[Case],
    *,
    variant: Variant,
    factory: AppFactory,
    repeat: int = 1,
    provider: str | None = None,
    model: str | None = None,
    keep_outcomes: bool = False,
    keep_stage: bool = False,
    progress: Progress | None = None,
) -> RunReport:
    preflight(cases)
    report = RunReport(
        variant=variant.name,
        variant_description=variant.description,
        provider=provider,
        model=model,
        repeat=repeat,
        started_at=now_iso(),
    )
    say = progress or (lambda _: None)
    for case in cases:
        missing = [need for need in case.requires if need == "docker" and not docker_available()]
        for attempt in range(1, repeat + 1):
            if missing:
                outcome = Outcome(
                    case_id=case.id,
                    variant=variant.name,
                    attempt=attempt,
                    status="skipped",
                    error=f"缺少运行条件：{'、'.join(missing)}",
                )
                report.attempts.append(record(case, outcome, [], keep_outcome=False))
                continue
            try:
                async with open_stage(case, variant, factory, keep=keep_stage) as stage:
                    outcome = await drive(stage, attempt=attempt)
            except Exception as exc:  # 现场都没搭起来（配置错误、缺少密钥等），记为这次失败
                outcome = Outcome(
                    case_id=case.id,
                    variant=variant.name,
                    attempt=attempt,
                    status="error",
                    error=f"启动失败：{type(exc).__name__}: {exc}",
                )
            verdicts = evaluate(outcome, case.checks)
            item = record(case, outcome, verdicts, keep_outcome=keep_outcomes)
            report.attempts.append(item)
            mark = "✅" if item.passed else "❌"
            reason = "" if item.passed else " — " + (
                item.error
                or next((f"{v['check']}: {v['detail']}" for v in item.verdicts if not v["passed"]), "")
            )
            say(
                f"{mark} [{variant.name}] {case.id} #{attempt}  "
                f"{outcome.duration_seconds:g}s  {outcome.chargeable_tokens} tok{reason}"
            )
    report.finished_at = now_iso()
    return report


__all__ = ["docker_available", "preflight", "run_cases"]
