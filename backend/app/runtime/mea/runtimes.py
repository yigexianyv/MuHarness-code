"""长任务三个角色的 AgentRuntime。

三个 Runtime 共享同一个 ToolRegistry、审批、权限、检查点和 EvidenceRecorder，
区别只在步数 / 模型调用上限，以及不带任务上下文、记忆召回和运行后反思。
授权了额外工具的长任务用一个共享工具的注册表视图单独构造 Executor Runtime，按工具集合缓存。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.types import AgentMode
from app.runtime.agent.budget import RunBudgetConfig
from app.tools.registry import ToolRegistry

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class MeaSettings(BaseSettings):
    """长任务角色的输出预算，可在 backend/.env 用 MEA_ 前缀覆盖。

    聊天默认的单次输出（MODEL_DEFAULT_MAX_OUTPUT_TOKENS，4096）和收尾上限（1200）是按对话回答设计的；
    推理模型的思考也算在输出里，角色的报告按这个预算写会被截断。这里给角色单独的、更大的预算，
    实际使用时再以模型能力上限封顶。
    """

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="MEA_",
        extra="ignore",
    )

    role_max_output_tokens: int = Field(default=32_768, ge=1024)  # Executor / Auditor 单次输出
    manager_max_output_tokens: int = Field(default=16_384, ge=1024)  # Manager / 最终回复 / 格式修复
    finalization_max_output_tokens: int = Field(default=8_192, ge=1024)  # 步数用完、被迫收尾时的那次输出
    # Executor 自己的 token 阈值，不沿用主对话的 80k/120k/160k（那是按聊天一轮回答设计的）。
    # Executor 主要靠 30 分钟时限和 40 次模型调用约束；token 线只防失控，正常写代码不该碰到。
    executor_warning_tokens: int = Field(default=300_000, ge=1)
    executor_finalization_tokens: int = Field(default=400_000, ge=1)
    executor_hard_tokens: int = Field(default=500_000, ge=1)


ROLE_SYSTEM_PROMPT = (
    "你是 MuHarness 长任务中的一个角色。严格按照用户消息开头的角色说明和输出格式工作，"
    "只做该角色的事，使用用户的语言。"
    "\nMCP 工具仅在当前任务需要该远端能力、用途和参数含义明确时调用；"
    "已有可用结果不重复试探，"
    "不凭服务名推测其他能力。未提供用途说明时先确认用途，调用仍遵守当前权限；"
    "远端成功回执只代表本次报告，须核实内容，不等于结果已验证或用户目标已完成。"
    "服务状态与能力发现只使用当前角色实际获准的工具，不调用角色之外的能力。"
)


@dataclass(frozen=True, slots=True)
class RoleLimits:

    max_steps: int
    max_tool_rounds: int | None
    hard_model_calls: int
    finalization_model_calls: int | None = None
    warning_model_calls: int | None = None
    timeout_seconds: float | None = None
    # None 表示沿用 Runtime 的默认单次输出上限
    max_output_tokens: int | None = None
    # 收尾那次输出通常就是角色的交付报告，不能沿用聊天的 1200
    finalization_max_output_tokens: int = 8_192
    # 角色自己的 token 阈值；为 None 时沿用主运行预算的对应阈值。
    warning_tokens: int | None = None
    finalization_tokens: int | None = None
    hard_tokens: int | None = None

    # 函数说明：RoleLimits.budget
    # 用途：在主运行预算上替换模型调用次数阈值，以及角色单独设置的 token 阈值。
    # 参数：
    #   base：`base`输入或配置值，类型 `RunBudgetConfig | None`。
    # 返回：类型 `RunBudgetConfig`；返回 `RunBudgetConfig(**values)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`RunBudgetConfig` →
    # `values.update`。
    def budget(self, base: RunBudgetConfig | None) -> RunBudgetConfig:
        """在主运行预算上替换模型调用次数阈值，以及角色单独设置的 token 阈值。"""

        values = (base or RunBudgetConfig()).model_dump()
        values.update(
            hard_model_calls=self.hard_model_calls,
            finalization_model_calls=self.finalization_model_calls,
            warning_model_calls=self.warning_model_calls,
            finalization_max_output_tokens=self.finalization_max_output_tokens,
        )
        for name in ("warning_tokens", "finalization_tokens", "hard_tokens"):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        return RunBudgetConfig(**values)


DEFAULT_ROLE_LIMITS: Mapping[AgentMode, RoleLimits] = {
    # Manager 没有工具，一次模型调用就结束；留一次余量给空回复重试
    AgentMode.MANAGE: RoleLimits(max_steps=2, max_tool_rounds=None, hard_model_calls=2,
                                 timeout_seconds=600),
    # Executor 主要靠 30 分钟时限约束；调用次数线放宽到 60/70/80，避免正常写代码就碰到收口。
    AgentMode.EXECUTE: RoleLimits(max_steps=80, max_tool_rounds=80, hard_model_calls=80,
                                  finalization_model_calls=70, warning_model_calls=60,
                                  timeout_seconds=30 * 60),
    AgentMode.AUDIT: RoleLimits(max_steps=20, max_tool_rounds=20, hard_model_calls=20,
                                finalization_model_calls=18, warning_model_calls=15,
                                timeout_seconds=5 * 60),
}


# 函数说明：role_limits
# 用途：按设置给每个角色填上输出预算，并用模型能力上限封顶（超过会被上下文预算拒绝）。
# 参数：
#   settings：业务或模型设置，类型 `MeaSettings | None`；默认 `None`。
#   model_max_output_tokens：Token 数量或 Token 预算，类型 `int | None`；默认 `None`。
#   base：`base`输入或配置值，类型 `Mapping[AgentMode, RoleLimits]`；默认
# `DEFAULT_ROLE_LIMITS`。
# 返回：类型 `dict[AgentMode, RoleLimits]`；返回 `limits`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`MeaSettings` → `cap` → `replace`。
def role_limits(
    settings: MeaSettings | None = None,
    *,
    model_max_output_tokens: int | None = None,
    base: Mapping[AgentMode, RoleLimits] = DEFAULT_ROLE_LIMITS,
) -> dict[AgentMode, RoleLimits]:
    """按设置给每个角色填上输出预算，并用模型能力上限封顶（超过会被上下文预算拒绝）。"""

    settings = settings or MeaSettings()

    # 函数说明：role_limits.cap
    # 用途：返回
    # `min(value, model_max_output_tokens) if model_max_output_tokens else value`，提供
    # 规划、执行、审计协作 的派生值。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `int`。
    # 返回：类型 `int`；返回
    # `min(value, model_max_output_tokens) if model_max_output_tokens else value`。
    # 闭包依赖：从外层读取 `model_max_output_tokens`。
    def cap(value: int) -> int:
        return min(value, model_max_output_tokens) if model_max_output_tokens else value

    finalization = settings.finalization_max_output_tokens
    limits: dict[AgentMode, RoleLimits] = {}
    for mode, limit in base.items():
        wanted = (
            settings.manager_max_output_tokens
            if mode is AgentMode.MANAGE
            else settings.role_max_output_tokens
        )
        output = cap(wanted)
        limits[mode] = replace(
            limit,
            max_output_tokens=output,
            finalization_max_output_tokens=min(cap(finalization), output),
        )
        if mode is AgentMode.EXECUTE:
            limits[mode] = replace(
                limits[mode],
                warning_tokens=settings.executor_warning_tokens,
                finalization_tokens=settings.executor_finalization_tokens,
                hard_tokens=settings.executor_hard_tokens,
            )
    return limits


# (mode, registry, limits, auto_approve_sandbox) -> AgentRuntime
RuntimeBuilder = Callable[[AgentMode, ToolRegistry, RoleLimits, bool], Any]


class RoleRuntimes(Mapping[AgentMode, Any]):
    """按角色取 Runtime；``executor_for(extra_tools)`` 取带额外工具的 Executor Runtime。"""

    # 函数说明：RoleRuntimes.__init__
    # 用途：初始化 RoleRuntimes；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
    #   build：`build`输入或配置值，类型 `RuntimeBuilder`。
    #   limits：传给 `dict` 的输入，类型 `Mapping[AgentMode, RoleLimits]`；默认
    # `DEFAULT_ROLE_LIMITS`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`build`。
    # 副作用与资源：
    #   更新对象字段：`self._registry`、`self._build`、`self.limits`、`self._runtimes`、
    # `self._executors`。
    def __init__(
        self,
        registry: ToolRegistry,
        build: RuntimeBuilder,
        *,
        limits: Mapping[AgentMode, RoleLimits] = DEFAULT_ROLE_LIMITS,
    ) -> None:
        self._registry = registry
        self._build = build
        self.limits = dict(limits)
        # Auditor 的 shell 只读挂载、禁网，并且有工作区快照兜底：总是自动批准。
        # Executor 默认需要审批；长任务启动时选了自动批准，就用 executor_for(..., True) 取另一个 Runtime。
        self._runtimes = {
            AgentMode.MANAGE: build(AgentMode.MANAGE, registry, self.limits[AgentMode.MANAGE], False),
            AgentMode.EXECUTE: build(AgentMode.EXECUTE, registry, self.limits[AgentMode.EXECUTE], False),
            AgentMode.AUDIT: build(AgentMode.AUDIT, registry, self.limits[AgentMode.AUDIT], True),
        }
        self._executors: dict[tuple[frozenset[str], bool], Any] = {}

    # 函数说明：RoleRuntimes.__getitem__
    # 用途：返回 `self._runtimes[mode]`，提供 RoleRuntimes 的派生值。
    # 参数：
    #   mode：Agent 执行模式或检索模式，类型 `AgentMode`。
    # 返回：类型 `Any`；返回 `self._runtimes[mode]`。
    def __getitem__(self, mode: AgentMode) -> Any:
        return self._runtimes[mode]

    # 函数说明：RoleRuntimes.__iter__
    # 用途：返回 `iter(self._runtimes)`，提供 RoleRuntimes 的派生值。
    # 返回：返回 `iter(self._runtimes)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`iter`。
    def __iter__(self):
        return iter(self._runtimes)

    # 函数说明：RoleRuntimes.__len__
    # 用途：返回 `len(self._runtimes)`，提供 RoleRuntimes 的派生值。
    # 返回：类型 `int`；返回 `len(self._runtimes)`。
    def __len__(self) -> int:
        return len(self._runtimes)

    # 函数说明：RoleRuntimes.executor_for
    # 用途：在规划、执行、审计协作中处理 `executor_for`，通过 `self._executors.get` 完成
    # 首个内部处理步骤。
    # 参数：
    #   extra_tools：传给 `frozenset` 的输入，类型 `tuple[str, ...]`；默认 `()`。
    #   auto_approve_sandbox：传给 `self._build` 的输入，类型 `bool`；默认 `False`。
    # 返回：类型 `Any`；按分支返回 `self._runtimes[AgentMode.EXECUTE]`；`runtime`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`frozenset` →
    # `self._registry.with_role_extras` → `self._build`。
    # 分支与异常：
    #   当 `not extra_tools and (not auto_approve_sandbox)` 时，返回
    # `self._runtimes[AgentMode.EXECUTE]`。
    def executor_for(
        self, extra_tools: tuple[str, ...] = (), auto_approve_sandbox: bool = False
    ) -> Any:
        if not extra_tools and not auto_approve_sandbox:
            return self._runtimes[AgentMode.EXECUTE]
        key = (frozenset(extra_tools), auto_approve_sandbox)
        runtime = self._executors.get(key)
        if runtime is None:
            view = (
                self._registry.with_role_extras({AgentMode.EXECUTE: key[0]})
                if extra_tools
                else self._registry
            )
            runtime = self._build(
                AgentMode.EXECUTE, view, self.limits[AgentMode.EXECUTE], auto_approve_sandbox
            )
            self._executors[key] = runtime
        return runtime

    # 函数说明：RoleRuntimes.timeouts
    # 用途：返回
    # `{mode: limits.timeout_seconds for mode, limits in self.limits.items()}`，提供
    # RoleRuntimes 的派生值。
    # 返回：类型 `dict[AgentMode, float | None]`；返回
    # `{mode: limits.timeout_seconds for mode, limits in self.limits.items()}`。
    def timeouts(self) -> dict[AgentMode, float | None]:
        return {mode: limits.timeout_seconds for mode, limits in self.limits.items()}


__all__ = [
    "DEFAULT_ROLE_LIMITS",
    "ROLE_SYSTEM_PROMPT",
    "MeaSettings",
    "RoleLimits",
    "RoleRuntimes",
    "role_limits",
]
