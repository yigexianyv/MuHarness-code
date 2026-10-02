
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from app.models.registry import ModelAdapterRegistry
from app.models.types import Message, MessageRole, ModelRequest, ModelUsage

from .config import SkillLearningSettings


@dataclass
class ModelCallResult:

    raw_output: str | None = None
    provider: str | None = None
    model: str | None = None
    duration_ms: float = 0.0
    usage: ModelUsage = field(default_factory=ModelUsage)
    error: str | None = None

    # 函数说明：ModelCallResult.ok
    # 用途：返回 `self.error is None and bool(self.raw_output)`，提供 ModelCallResult 的
    # 派生值。
    # 返回：类型 `bool`；返回 `self.error is None and bool(self.raw_output)`。
    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.raw_output)


_DISABLE_THINKING_BODY = {"thinking": {"type": "disabled"}}
_REASONING_DISABLE_PROVIDERS = frozenset({"deepseek"})


# 函数说明：_should_disable_thinking
# 用途：在技能候选提炼与审核中处理 `_should_disable_thinking`，通过
# `(provider or '').strip().lower` 完成首个内部处理步骤。
# 参数：
#   settings：业务或模型设置，类型 `SkillLearningSettings`。
#   provider：模型或搜索服务商，类型 `str | None`。
# 返回：类型 `bool`；按分支返回 `requested`；
# `normalized in _REASONING_DISABLE_PROVIDERS`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(provider or '').strip().lower`。
# 分支与异常：
#   当 `requested is not None` 时，返回 `requested`。
def _should_disable_thinking(
    settings: SkillLearningSettings,
    provider: str | None,
) -> bool:

    requested = settings.skill_learning_disable_thinking
    if requested is not None:
        return requested
    normalized = (provider or "").strip().lower()
    return normalized in _REASONING_DISABLE_PROVIDERS


# 函数说明：call_model
# 用途：在技能候选提炼与审核中处理 `call_model`，通过 `time.perf_counter` 完成首个内部处
# 理步骤。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ModelAdapterRegistry`。
#   system_prompt：系统提示文本，类型 `str`。
#   user_content：正文输入或配置值，类型 `str`。
#   settings：业务或模型设置，类型 `SkillLearningSettings`。
#   default_provider：未指定服务商时的默认值，类型 `str | None`。
#   default_model：未指定模型时的默认值，类型 `str | None`。
# 返回：类型 `ModelCallResult`；返回 `ModelCallResult(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`time.perf_counter` → `ModelUsage` →
# `ModelRequest` → `Message` → `_should_disable_thinking` → `asyncio.timeout`；另有 2 个
# 调用点。
# 资源/并发边界：`asyncio.timeout(settings.skill_learning_timeout_seconds)`，上下文退出
# 时执行相应清理。
# 分支与异常：
#   当 `not raw_output` 时，抛出
# `ValueError('skill learning model returned empty content')`。
#   捕获 `Exception` 后，返回 `ModelCallResult(…)`。
async def call_model(
    registry: ModelAdapterRegistry,
    *,
    system_prompt: str,
    user_content: str,
    settings: SkillLearningSettings,
    default_provider: str | None,
    default_model: str | None,
) -> ModelCallResult:

    started = time.perf_counter()
    usage = ModelUsage()
    provider = settings.skill_learning_provider or default_provider
    model = settings.skill_learning_model or default_model
    raw_output: str | None = None
    try:
        adapter = registry.get(provider)
        resolved_provider = adapter.provider
        if settings.skill_learning_model is not None:
            resolved_model = settings.skill_learning_model
        elif settings.skill_learning_provider is not None:
            resolved_model = adapter.default_model
        else:
            resolved_model = default_model or adapter.default_model
        request = ModelRequest(
            messages=(
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_content),
            ),
            model=resolved_model,
            temperature=settings.skill_learning_temperature,
            max_output_tokens=settings.skill_learning_max_output_tokens,
            extra_body=(
                _DISABLE_THINKING_BODY
                if _should_disable_thinking(settings, provider)
                else {}
            ),
        )
        async with asyncio.timeout(settings.skill_learning_timeout_seconds):
            response = await adapter.complete(request)
        usage = response.usage
        raw_output = response.message.content
        if not raw_output:
            raise ValueError("skill learning model returned empty content")
        return ModelCallResult(
            raw_output=raw_output,
            provider=resolved_provider,
            model=resolved_model,
            duration_ms=(time.perf_counter() - started) * 1000,
            usage=usage,
        )
    except Exception as exc:
        return ModelCallResult(
            provider=provider,
            model=model,
            duration_ms=(time.perf_counter() - started) * 1000,
            usage=usage,
            error=f"{type(exc).__name__}: {exc}",
        )


# 函数说明：parse_strict_json
# 用途：解析JSON 数据，供技能候选提炼与审核使用。
# 参数：
#   raw_output：输出输入或配置值，类型 `str`。
# 返回：类型 `dict | None`；按分支返回 `None`；`payload`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`stripped.startswith` →
# `stripped.endswith` → `stripped.splitlines` → `json.loads`。
# 分支与异常：
#   捕获 `(ValueError, TypeError)` 后，返回 `None`。
#   当 `not isinstance(payload, dict)` 时，返回 `None`。
def parse_strict_json(raw_output: str) -> dict | None:

    stripped = raw_output.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


__all__ = ["ModelCallResult", "call_model", "parse_strict_json"]
