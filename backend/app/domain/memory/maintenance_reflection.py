
from __future__ import annotations

import asyncio
import json
import time

from app.models.registry import ModelAdapterRegistry
from app.models.types import Message, MessageRole, ModelRequest, ModelUsage

from .maintenance_models import (
    MemoryMaintenanceConfig,
    MemoryMaintenanceDecision,
    MemoryMaintenanceInput,
    MemoryMaintenanceProposal,
)

_MAINTENANCE_PROMPT = """# 记忆容量维护
职责：活动记忆已满或超限，仅审查输入 candidates，选择至多一条可恢复归档建议。

决策规则：有证据表明候选已过时、重复、被取代或不再具有跨会话价值时，选择 archive；候选各有价值或依据不足时选择 defer。容量压力不是随意丢弃知识的理由。
边界：archive 将 Markdown 记录移入可恢复档案，不是删除。不能编造 ID、修改正文、合并记忆、调用工具或回答用户。

输出约定：只返回严格 JSON，无 Markdown 或额外字段。action 只能取 archive、defer：
{"action":"defer","memory_id":null,"reason":"..."}
archive：memory_id 必须是 candidates 中的一个 ID，reason 说明归档依据。
defer：memory_id 必须为 null，reason 说明为何保留或无法判断。"""


class MemoryMaintenanceReflector:

    # 函数说明：MemoryMaintenanceReflector.__init__
    # 用途：初始化 MemoryMaintenanceReflector；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ModelAdapterRegistry`。
    #   config：运行配置，类型 `MemoryMaintenanceConfig | None`；默认 `None`。
    #   default_provider：未指定服务商时的默认值，类型 `str | None`；默认 `None`。
    #   default_model：未指定模型时的默认值，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryMaintenanceConfig`。
    # 副作用与资源：
    #   更新对象字段：`self._registry`、`self.config`、`self._default_provider`、
    # `self._default_model`。
    def __init__(
        self,
        registry: ModelAdapterRegistry,
        *,
        config: MemoryMaintenanceConfig | None = None,
        default_provider: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self._registry = registry
        self.config = config or MemoryMaintenanceConfig()
        self._default_provider = default_provider
        self._default_model = default_model

    # 函数说明：MemoryMaintenanceReflector.enabled
    # 用途：返回 `self.config.enabled`，提供 MemoryMaintenanceReflector 的派生值。
    # 返回：类型 `bool`；返回 `self.config.enabled`。
    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # 函数说明：MemoryMaintenanceReflector.provider_hint
    # 用途：返回 `self.config.provider or self._default_provider`，提供
    # MemoryMaintenanceReflector 的派生值。
    # 返回：类型 `str | None`；返回 `self.config.provider or self._default_provider`。
    @property
    def provider_hint(self) -> str | None:
        return self.config.provider or self._default_provider

    # 函数说明：MemoryMaintenanceReflector.model_hint
    # 用途：在长期记忆管理与检索中处理 `model_hint`，通过 `self._registry.get` 完成首个
    # 内部处理步骤。
    # 返回：类型 `str | None`；按分支返回 `self.config.model`；`self._default_model`；
    # `self._registry.get(self.config.provider).default_model`；`None`。
    # 分支与异常：
    #   当 `self.config.model is not None` 时，返回 `self.config.model`。
    #   当 `self.config.provider is None` 时，返回 `self._default_model`。
    #   捕获 `Exception` 后，返回 `None`。
    @property
    def model_hint(self) -> str | None:
        if self.config.model is not None:
            return self.config.model
        if self.config.provider is None:
            return self._default_model
        try:
            return self._registry.get(self.config.provider).default_model
        except Exception:
            return None

    # 函数说明：MemoryMaintenanceReflector.decide
    # 用途：对容量维护候选项作出保留、更新或归档决策。
    # 参数：
    #   maintenance_input：维护输入或配置值，类型 `MemoryMaintenanceInput`。
    # 返回：类型 `MemoryMaintenanceProposal`；按分支返回 `MemoryMaintenanceProposal()`；
    # `MemoryMaintenanceProposal(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryMaintenanceProposal` →
    # `time.perf_counter` → `ModelUsage` → `ModelRequest` → `Message` → `json.dumps`；另
    # 有 4 个调用点。
    # 资源/并发边界：`asyncio.timeout(self.config.timeout_seconds)`，上下文退出时执行相
    # 应清理。
    # 分支与异常：
    #   当 `not self.config.enabled` 时，返回 `MemoryMaintenanceProposal()`。
    #   当 `not response.message.content` 时，抛出
    # `ValueError('maintenance model returned empty content')`。
    #   捕获 `Exception` 后，返回 `MemoryMaintenanceProposal(…)`。
    async def decide(
        self,
        maintenance_input: MemoryMaintenanceInput,
    ) -> MemoryMaintenanceProposal:

        """对容量维护候选项作出保留、更新或归档决策。"""
        if not self.config.enabled:
            return MemoryMaintenanceProposal()
        started = time.perf_counter()
        usage = ModelUsage()
        provider = self.provider_hint
        model = self.model_hint
        try:
            adapter = self._registry.get(provider)
            provider = adapter.provider
            if self.config.model is not None:
                model = self.config.model
            elif self.config.provider is not None:
                model = adapter.default_model
            else:
                model = self._default_model or adapter.default_model
            request = ModelRequest(
                messages=(
                    Message(role=MessageRole.SYSTEM, content=_MAINTENANCE_PROMPT),
                    Message(
                        role=MessageRole.USER,
                        content=json.dumps(
                            maintenance_input.model_dump(mode="json"),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                ),
                model=model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
            async with asyncio.timeout(self.config.timeout_seconds):
                response = await adapter.complete(request)
            usage = response.usage
            if not response.message.content:
                raise ValueError("maintenance model returned empty content")
            decision = MemoryMaintenanceDecision.model_validate_json(
                _strip_code_fence(response.message.content)
            )
            return MemoryMaintenanceProposal(
                decision=decision,
                provider=provider,
                model=model,
                duration_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
            )
        except Exception as exc:
            return MemoryMaintenanceProposal(
                provider=provider,
                model=model,
                duration_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
                error=f"{type(exc).__name__}: {exc}",
            )


# 函数说明：_strip_code_fence
# 用途：移除指定内容`code_fence`，供长期记忆管理与检索使用。
# 参数：
#   content：内容正文，类型 `str`。
# 返回：类型 `str`；按分支返回 `'\n'.join(lines[1:-1]).strip()`；`stripped`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`stripped.startswith` →
# `stripped.endswith` → `stripped.splitlines`。
# 分支与异常：
#   `stripped.startswith('```') and stripped.endswith('```')` 分支在完成前置处理后返回
# `'\n'.join(lines[1:-1]).strip()`。
def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped


__all__ = ["MemoryMaintenanceReflector"]
