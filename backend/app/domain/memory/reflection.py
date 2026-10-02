
from __future__ import annotations

import asyncio
import json
import time

from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    Message,
    MessageRole,
    ModelRequest,
    ModelUsage,
    add_model_usage,
)

from .reflection_models import (
    MemoryReflectionConfig,
    MemoryReflectionInput,
    MemoryReflectionProposal,
    ReflectionDecision,
)

_REFLECTION_PROMPT = """# 普通记忆反思
职责：主 Agent 已结束任务，只判断本轮是否产生一个值得保留的耐久普通记忆增量。不要回答用户、继续任务、调用工具、修改 Task/Core Memory 或创建 Skills。

## 事实与边界
- 普通记忆保持稀疏，只保存重要的项目历史决定、长期方向变化和后续可能需要的背景。不保存当前进度、待办、临时限制、原始工具输出、一次性事实或可复用流程。
- 稳定身份、全局长期偏好、全局安全/隐私约束属于 Core，此处返回 none。即使主 Agent 未调用 core_memory_update、未找到按需工具或保存失败，也不能用普通记忆 create/update 兜底或备份 Core 信息。
- 用户明确确认项目决定或规则已定稿、完成、纠正或扩展，就是耐久证据，不要求本轮修改代码或文件。提议、猜测和助理自述不能变成已确认决定。

## 决策规则
- 同一主题的新耐久规则优先 update，不因它只是补充、未否定旧规则就选择 create 或 none。
- update 只能使用 recalled_memory_ids 中的 ID，它们表示本轮已成功读取全文。只有索引线索、未读正文时返回 none，不能猜测替换。
- 更新须完整保留旧内容中仍有效的事实，同时保留证据中的否定约束、被否决方案、取代关系、数值限制和安全要求；title/summary/content 必须一致。
- 有独立且未覆盖的耐久增量才 create；没有耐久增量或依据不足时 none，不强行新增。

## 输出约定
只返回一个严格 JSON 对象，不含 Markdown、说明或工具调用。action 只能取 none、create、update：
{"action":"none","memory_id":null,"title":null,
"summary":null,"content":null,"reason":"..."}
none：memory_id、title、summary、content 全部为 null。
create：memory_id 为 null，title、summary、content 为完整非空内容。
update：memory_id 来自 recalled_memory_ids，并给出完整替换的 title、summary、content。
reason 简述当前证据及选择依据，不添加字段。"""


class PostRunMemoryReflector:

    # 函数说明：PostRunMemoryReflector.__init__
    # 用途：初始化 PostRunMemoryReflector；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   registry：工具、模型或能力注册表，类型 `ModelAdapterRegistry`。
    #   config：运行配置，类型 `MemoryReflectionConfig | None`；默认 `None`。
    #   default_provider：未指定服务商时的默认值，类型 `str | None`；默认 `None`。
    #   default_model：未指定模型时的默认值，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryReflectionConfig`。
    # 副作用与资源：
    #   更新对象字段：`self._registry`、`self.config`、`self._default_provider`、
    # `self._default_model`。
    def __init__(
        self,
        registry: ModelAdapterRegistry,
        *,
        config: MemoryReflectionConfig | None = None,
        default_provider: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self._registry = registry
        self.config = config or MemoryReflectionConfig()
        self._default_provider = default_provider
        self._default_model = default_model

    # 函数说明：PostRunMemoryReflector.enabled
    # 用途：返回 `self.config.enabled`，提供 PostRunMemoryReflector 的派生值。
    # 返回：类型 `bool`；返回 `self.config.enabled`。
    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # 函数说明：PostRunMemoryReflector.provider_hint
    # 用途：返回 `self.config.provider or self._default_provider`，提供
    # PostRunMemoryReflector 的派生值。
    # 返回：类型 `str | None`；返回 `self.config.provider or self._default_provider`。
    @property
    def provider_hint(self) -> str | None:
        return self.config.provider or self._default_provider

    # 函数说明：PostRunMemoryReflector.model_hint
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

    # 函数说明：PostRunMemoryReflector.decide
    # 用途：分析运行结果并输出结构化的记忆增删改建议。
    # 参数：
    #   reflection_input：反思输入或配置值，类型 `MemoryReflectionInput`。
    # 返回：类型 `MemoryReflectionProposal`；按分支返回 `MemoryReflectionProposal()`；
    # `MemoryReflectionProposal(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MemoryReflectionProposal` →
    # `time.perf_counter` → `ModelUsage` → `json.dumps` → `asyncio.timeout` →
    # `ModelRequest`；另有 5 个调用点。
    # 资源/并发边界：`asyncio.timeout(self.config.timeout_seconds)`，上下文退出时执行相
    # 应清理。
    # 分支与异常：
    #   当 `not self.config.enabled` 时，返回 `MemoryReflectionProposal()`。
    #   当 `not raw_output` 时，抛出
    # `ValueError('reflection model returned empty content')`。
    #   捕获 `(ValueError, TypeError)` 后，重新抛出原异常。
    #   当 `attempt < self.config.max_attempts` 时，跳过当前循环项。
    #   捕获 `Exception` 后，返回 `MemoryReflectionProposal(…)`。
    async def decide(
        self,
        reflection_input: MemoryReflectionInput,
    ) -> MemoryReflectionProposal:

        """分析运行结果并输出结构化的记忆增删改建议。"""
        if not self.config.enabled:
            return MemoryReflectionProposal()
        started = time.perf_counter()
        usage = ModelUsage()
        provider = self.provider_hint
        model = self.model_hint
        attempts = 0
        finish_reason: str | None = None
        input_json = json.dumps(
            reflection_input.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        raw_output: str | None = None
        try:
            adapter = self._registry.get(provider)
            provider = adapter.provider
            if self.config.model is not None:
                model = self.config.model
            elif self.config.provider is not None:
                model = adapter.default_model
            else:
                model = self._default_model or adapter.default_model
            last_validation_error: Exception | None = None
            async with asyncio.timeout(self.config.timeout_seconds):
                for attempt in range(1, self.config.max_attempts + 1):
                    attempts = attempt
                    user_content = input_json
                    if attempt > 1:
                        user_content = (
                            "The previous response was empty, invalid, or truncated. "
                            "Return one complete strict JSON object now.\n\n"
                            f"{input_json}"
                        )
                    request = ModelRequest(
                        messages=(
                            Message(
                                role=MessageRole.SYSTEM,
                                content=_REFLECTION_PROMPT,
                            ),
                            Message(
                                role=MessageRole.USER,
                                content=user_content,
                            ),
                        ),
                        model=model,
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                    )
                    response = await adapter.complete(request)
                    usage = add_model_usage(usage, response.usage)
                    raw_output = response.message.content
                    finish_reason = response.finish_reason
                    try:
                        if not raw_output:
                            raise ValueError(
                                "reflection model returned empty content"
                            )
                        decision = ReflectionDecision.model_validate_json(
                            _strip_code_fence(raw_output)
                        )
                    except (ValueError, TypeError) as exc:
                        last_validation_error = exc
                        if attempt < self.config.max_attempts:
                            continue
                        raise
                    return MemoryReflectionProposal(
                        decision=decision,
                        provider=provider,
                        model=model,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        usage=usage,
                        attempts=attempts,
                        finish_reason=finish_reason,
                        input_json=(
                            input_json if self.config.capture_raw_io else None
                        ),
                        raw_output=(
                            raw_output if self.config.capture_raw_io else None
                        ),
                    )
            if last_validation_error is not None:
                raise last_validation_error
            raise RuntimeError("reflection model did not produce a decision")
        except Exception as exc:
            return MemoryReflectionProposal(
                provider=provider,
                model=model,
                duration_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
                attempts=attempts,
                finish_reason=finish_reason,
                error=f"{type(exc).__name__}: {exc}",
                input_json=(input_json if self.config.capture_raw_io else None),
                raw_output=(raw_output if self.config.capture_raw_io else None),
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


__all__ = ["PostRunMemoryReflector"]
