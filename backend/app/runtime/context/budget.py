
from __future__ import annotations

from dataclasses import dataclass

from .capabilities import ModelCapabilities
from .config import ContextSettings

DEFAULT_TRIGGER_RATIO = 0.80
DEFAULT_TARGET_RATIO = 0.60
DEFAULT_SAFETY_MARGIN_TOKENS = 4_096


@dataclass(frozen=True)
class ContextBudget:

    context_window: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    input_budget: int
    working_input_budget: int
    hard_trigger_tokens: int
    hard_target_tokens: int
    trigger_tokens: int
    target_tokens: int
    compact_ceiling_tokens: int
    forced_target_tokens: int
    tool_result_budget_tokens: int


class ContextBudgetPolicy:

    # 函数说明：ContextBudgetPolicy.__init__
    # 用途：初始化 ContextBudgetPolicy；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   trigger_ratio：`trigger_ratio`输入或配置值，类型 `float`；默认
    # `DEFAULT_TRIGGER_RATIO`。
    #   target_ratio：`target_ratio`输入或配置值，类型 `float`；默认
    # `DEFAULT_TARGET_RATIO`。
    #   safety_margin_tokens：Token 数量或 Token 预算，类型 `int`；默认
    # `DEFAULT_SAFETY_MARGIN_TOKENS`。
    #   preferred_input_tokens：Token 数量或 Token 预算，类型 `int`；默认 `64000`。
    #   working_trigger_ratio：`working_trigger_ratio`输入或配置值，类型 `float`；默认
    # `0.8`。
    #   working_target_ratio：`working_target_ratio`输入或配置值，类型 `float`；默认
    # `0.45`。
    #   compact_input_tokens：Token 数量或 Token 预算，类型 `int | None`；默认 `None`。
    #   tool_result_budget_ratio：工具结果预算输入或配置值，类型 `float`；默认 `0.35`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 分支与异常：
    #   当 `not 0.0 < trigger_ratio < 1.0` 时，抛出
    # `ValueError('trigger_ratio must be in (0, 1)')`。
    #   当 `not 0.0 < target_ratio < 1.0` 时，抛出
    # `ValueError('target_ratio must be in (0, 1)')`。
    #   当 `target_ratio >= trigger_ratio` 时，抛出
    # `ValueError('target_ratio must be lower than trigger_ratio')`。
    #   当 `safety_margin_tokens < 0` 时，抛出
    # `ValueError('safety_margin_tokens cannot be negative')`。
    # 副作用与资源：
    #   更新对象字段：`self._trigger_ratio`、`self._target_ratio`、
    # `self._safety_margin_tokens`、`self._preferred_input_tokens`、
    # `self._working_trigger_ratio`、`self._working_target_ratio`、
    # `self._compact_input_tokens`、`self._tool_result_budget_ratio`。
    def __init__(
        self,
        *,
        trigger_ratio: float = DEFAULT_TRIGGER_RATIO,
        target_ratio: float = DEFAULT_TARGET_RATIO,
        safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
        preferred_input_tokens: int = 64_000,
        working_trigger_ratio: float = 0.80,
        working_target_ratio: float = 0.45,
        compact_input_tokens: int | None = None,
        tool_result_budget_ratio: float = 0.35,
    ) -> None:
        if not 0.0 < trigger_ratio < 1.0:
            raise ValueError("trigger_ratio must be in (0, 1)")
        if not 0.0 < target_ratio < 1.0:
            raise ValueError("target_ratio must be in (0, 1)")
        if target_ratio >= trigger_ratio:
            raise ValueError("target_ratio must be lower than trigger_ratio")
        if safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens cannot be negative")
        if preferred_input_tokens <= 0:
            raise ValueError("preferred_input_tokens must be greater than zero")
        if not 0.0 < working_trigger_ratio <= 1.0:
            raise ValueError("working_trigger_ratio must be in (0, 1]")
        if not 0.0 < working_target_ratio < working_trigger_ratio:
            raise ValueError(
                "working_target_ratio must be lower than working_trigger_ratio"
            )
        if compact_input_tokens is not None and compact_input_tokens <= 0:
            raise ValueError("compact_input_tokens must be greater than zero")
        if not 0.0 < tool_result_budget_ratio < 1.0:
            raise ValueError("tool_result_budget_ratio must be in (0, 1)")
        self._trigger_ratio = trigger_ratio
        self._target_ratio = target_ratio
        self._safety_margin_tokens = safety_margin_tokens
        self._preferred_input_tokens = preferred_input_tokens
        self._working_trigger_ratio = working_trigger_ratio
        self._working_target_ratio = working_target_ratio
        self._compact_input_tokens = compact_input_tokens
        self._tool_result_budget_ratio = tool_result_budget_ratio

    # 函数说明：ContextBudgetPolicy.compute
    # 用途：结合模型窗口与输出预留量，计算本次请求可用的上下文预算。
    # 参数：
    #   capabilities：模型能力输入或配置值，类型 `ModelCapabilities`。
    #   max_output_tokens：模型输出 Token 上限，类型 `int | None`；默认 `None`。
    # 返回：类型 `ContextBudget`；返回 `ContextBudget(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextBudget`。
    # 分支与异常：
    #   当 `reserved_output <= 0` 时，抛出
    # `ValueError('max_output_tokens must be greater than zero')`。
    #   当 `reserved_output > capabilities.max_output_tokens` 时，抛出 `ValueError(…)`。
    #   当 `input_budget <= 0` 时，抛出 `ValueError(…)`。
    def compute(
        self,
        capabilities: ModelCapabilities,
        *,
        max_output_tokens: int | None = None,
    ) -> ContextBudget:

        """结合模型窗口与输出预留量，计算本次请求可用的上下文预算。"""
        context_window = capabilities.context_window
        reserved_output = (
            max_output_tokens
            if max_output_tokens is not None
            else capabilities.max_output_tokens
        )
        if reserved_output <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if reserved_output > capabilities.max_output_tokens:
            raise ValueError(
                f"max_output_tokens ({reserved_output}) exceed model maximum "
                f"({capabilities.max_output_tokens})"
            )

        input_budget = (
            context_window - reserved_output - self._safety_margin_tokens
        )
        if input_budget <= 0:
            raise ValueError(
                f"invalid context budget: window={context_window} "
                f"reserved_output={reserved_output} "
                f"safety_margin={self._safety_margin_tokens} "
                f"input_budget={input_budget} (must be > 0)"
            )
        working_input_budget = min(input_budget, self._preferred_input_tokens)
        hard_trigger_tokens = int(input_budget * self._trigger_ratio)
        hard_target_tokens = int(input_budget * self._target_ratio)
        trigger_tokens = min(
            hard_trigger_tokens,
            int(working_input_budget * self._working_trigger_ratio),
        )
        target_tokens = min(
            hard_target_tokens,
            int(working_input_budget * self._working_target_ratio),
        )
        resolved_compact_input_tokens = (
            self._compact_input_tokens
            if self._compact_input_tokens is not None
            else 2 * self._preferred_input_tokens
        )
        compact_ceiling_tokens = min(
            resolved_compact_input_tokens,
            hard_trigger_tokens,
        )
        forced_target_tokens = min(
            compact_ceiling_tokens,
            max(
                target_tokens,
                min(working_input_budget, compact_ceiling_tokens // 2),
            ),
        )
        return ContextBudget(
            context_window=context_window,
            reserved_output_tokens=reserved_output,
            safety_margin_tokens=self._safety_margin_tokens,
            input_budget=input_budget,
            working_input_budget=working_input_budget,
            hard_trigger_tokens=hard_trigger_tokens,
            hard_target_tokens=hard_target_tokens,
            trigger_tokens=trigger_tokens,
            target_tokens=target_tokens,
            compact_ceiling_tokens=compact_ceiling_tokens,
            forced_target_tokens=forced_target_tokens,
            tool_result_budget_tokens=int(
                target_tokens * self._tool_result_budget_ratio
            ),
        )


# 函数说明：build_budget_policy
# 用途：构建预算，供模型上下文与输入预算使用。
# 参数：
#   settings：业务或模型设置，类型 `ContextSettings | None`；默认 `None`。
# 返回：类型 `ContextBudgetPolicy`；返回 `ContextBudgetPolicy(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ContextSettings` →
# `ContextBudgetPolicy`。
def build_budget_policy(
    settings: ContextSettings | None = None,
) -> ContextBudgetPolicy:

    resolved = settings or ContextSettings()
    return ContextBudgetPolicy(
        trigger_ratio=resolved.context_trigger_ratio,
        target_ratio=resolved.context_target_ratio,
        safety_margin_tokens=resolved.context_safety_margin_tokens,
        preferred_input_tokens=resolved.context_preferred_input_tokens,
        working_trigger_ratio=resolved.context_working_trigger_ratio,
        working_target_ratio=resolved.context_working_target_ratio,
        compact_input_tokens=resolved.context_compact_input_tokens,
        tool_result_budget_ratio=resolved.context_tool_result_budget_ratio,
    )


__all__ = [
    "ContextBudget",
    "ContextBudgetPolicy",
    "DEFAULT_SAFETY_MARGIN_TOKENS",
    "DEFAULT_TARGET_RATIO",
    "DEFAULT_TRIGGER_RATIO",
    "build_budget_policy",
]
