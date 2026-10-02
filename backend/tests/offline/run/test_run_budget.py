
import pytest
from pydantic import ValidationError

from app.models.types import ModelUsage
from app.runtime.agent.budget import (
    RunBudget,
    RunBudgetConfig,
    RunBudgetStatus,
    chargeable_tokens,
)


# 函数说明：_config
# 用途：处理回归测试与测试辅助中的 `_config` 数据；结果及边界条件见下方说明。
# 参数：
#   **overrides：额外关键字参数，按实现处理或转交。
# 返回：类型 `RunBudgetConfig`；返回 `RunBudgetConfig(_env_file=None, **values)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RunBudgetConfig`。
def _config(**overrides: object) -> RunBudgetConfig:
    values = {
        "warning_tokens": 100,
        "finalization_tokens": 200,
        "hard_tokens": 300,
        "warning_model_calls": 4,
        "finalization_model_calls": 6,
        "hard_model_calls": 8,
        **overrides,
    }
    return RunBudgetConfig(_env_file=None, **values)


# 函数说明：test_default_run_budget_thresholds
# 用途：回归验证回归测试与测试辅助中的 `default_run_budget_thresholds` 场景，下方断言说
# 明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RunBudgetConfig`。
# 分支与异常：
#   验证条件：`config.warning_tokens == 80000`。
#   验证条件：`config.finalization_tokens == 120000`。
#   验证条件：`config.hard_tokens == 160000`。
#   验证条件：`config.warning_model_calls is None`。
def test_default_run_budget_thresholds() -> None:
    config = RunBudgetConfig(_env_file=None)

    assert config.warning_tokens == 80_000
    assert config.finalization_tokens == 120_000
    assert config.hard_tokens == 160_000
    assert config.warning_model_calls is None
    assert config.finalization_model_calls is None
    assert config.hard_model_calls == 15


# 函数说明：test_default_model_call_budget_only_enforces_hard_limit
# 用途：回归验证回归测试与测试辅助中的
# `default_model_call_budget_only_enforces_hard_limit` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RunBudget` → `RunBudgetConfig` →
# `budget.evaluate` → `ModelUsage`。
# 分支与异常：
#   验证条件：`before_hard.status is RunBudgetStatus.ACTIVE`。
#   验证条件：`at_hard.status is RunBudgetStatus.EXCEEDED`。
#   验证条件：`at_hard.reason is not None`。
#   验证条件：`at_hard.reason.value == 'model_calls'`。
def test_default_model_call_budget_only_enforces_hard_limit() -> None:
    budget = RunBudget(RunBudgetConfig(_env_file=None))

    before_hard = budget.evaluate(
        ModelUsage(input_tokens=1, model_calls=14)
    )
    at_hard = budget.evaluate(ModelUsage(input_tokens=1, model_calls=15))

    assert before_hard.status is RunBudgetStatus.ACTIVE
    assert at_hard.status is RunBudgetStatus.EXCEEDED
    assert at_hard.reason is not None
    assert at_hard.reason.value == "model_calls"


# 函数说明：test_chargeable_tokens_prefers_uncached_input
# 用途：回归验证回归测试与测试辅助中的 `chargeable_tokens_prefers_uncached_input` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelUsage` → `chargeable_tokens`。
# 分支与异常：
#   验证条件：`chargeable_tokens(usage) == 120`。
def test_chargeable_tokens_prefers_uncached_input() -> None:
    usage = ModelUsage(
        input_tokens=1_000,
        output_tokens=20,
        total_tokens=1_020,
        cached_input_tokens=900,
        uncached_input_tokens=100,
    )

    assert chargeable_tokens(usage) == 120


# 函数说明：test_chargeable_tokens_falls_back_to_all_input_when_cache_is_unknown
# 用途：回归验证回归测试与测试辅助中的
# `chargeable_tokens_falls_back_to_all_input_when_cache_is_unknown` 场景，下方断言说明列
# 出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelUsage` → `chargeable_tokens`。
# 分支与异常：
#   验证条件：`chargeable_tokens(usage) == 1020`。
def test_chargeable_tokens_falls_back_to_all_input_when_cache_is_unknown() -> None:
    usage = ModelUsage(input_tokens=1_000, output_tokens=20, total_tokens=1_020)

    assert chargeable_tokens(usage) == 1_020


# 函数说明：test_budget_stages
# 用途：回归验证回归测试与测试辅助中的 `budget_stages` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   usage：模型调用用量统计，类型 `ModelUsage`。
#   status：目标状态，类型 `RunBudgetStatus`。
#   reason：状态变化、拒绝或降级原因，类型 `str | None`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RunBudget(_config()).evaluate` →
# `RunBudget` → `_config`。
# 分支与异常：
#   验证条件：`decision.status is status`。
#   验证条件：`(decision.reason.value if decision.reason else None) == reason`。
@pytest.mark.parametrize(
    ("usage", "status", "reason"),
    [
        (ModelUsage(input_tokens=10, model_calls=1), RunBudgetStatus.ACTIVE, None),
        (
            ModelUsage(input_tokens=100, model_calls=1),
            RunBudgetStatus.WARNING,
            "tokens",
        ),
        (
            ModelUsage(input_tokens=10, model_calls=6),
            RunBudgetStatus.FINALIZING,
            "model_calls",
        ),
        (
            ModelUsage(input_tokens=300, model_calls=1),
            RunBudgetStatus.EXCEEDED,
            "tokens",
        ),
    ],
)
def test_budget_stages(
    usage: ModelUsage,
    status: RunBudgetStatus,
    reason: str | None,
) -> None:
    decision = RunBudget(_config()).evaluate(usage)

    assert decision.status is status
    assert (decision.reason.value if decision.reason else None) == reason


# 函数说明：test_invalid_threshold_order_is_rejected
# 用途：回归验证回归测试与测试辅助中的 `invalid_threshold_order_is_rejected` 场景，下方
# 断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` → `_config`。
# 分支与异常：
#   预期异常：`pytest.raises(ValidationError, match='warning < finalization < hard')`。
def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValidationError, match="warning < finalization < hard"):
        _config(finalization_tokens=100)
