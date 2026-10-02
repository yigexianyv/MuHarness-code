from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.models.types import (
    Message,
    MessageRole,
    ModelProvider,
    ModelUsage,
    ToolCall,
    ToolDefinition,
    add_model_usage,
)

from .budget import RunBudgetConfig, RunBudgetDecision, RunBudgetStatus
from .result import ToolCallRecord

_LEGACY_DATE_PATTERN = re.compile(r"当前日期是 \d{4}-\d{2}-\d{2}。")


@dataclass(frozen=True)
class RequestPrefixState:
    source_messages: tuple[Message, ...]
    context_messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    sent_messages: tuple[Message, ...]
    request_config: tuple[object, ...] = ()

    # 函数说明：RequestPrefixState.extend
    # 用途：处理模型与工具执行循环中的 `extend` 数据；结果及边界条件见下方说明。
    # 参数：
    #   source_messages：传给 `len` 的输入，类型 `tuple[Message, ...]`。
    #   context_messages：上下文消息序列输入或配置值，类型 `tuple[Message, ...]`。
    #   tools：可用工具定义或工具实例集合，类型 `tuple[ToolDefinition, ...]`。
    #   request_config：请求配置输入或配置值，类型 `tuple[object, ...]`；默认 `()`。
    # 返回：类型 `tuple[Message, ...] | None`；按分支返回 `None`；
    # `(*self.sent_messages, *source_messages[previous_count:], *updates)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
    # 分支与异常：
    #   当 `tools != self.tools or request_config != self.request_config` 时，返回
    # `None`。
    #   当 `stable_context != previous_stable` 时，返回 `None`。
    #   当 `len(source_messages) < previous_count` 时，返回 `None`。
    #   当 `source_messages[:previous_count] != self.source_messages` 时，返回 `None`。
    def extend(
        self,
        *,
        source_messages: tuple[Message, ...],
        context_messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...],
        request_config: tuple[object, ...] = (),
    ) -> tuple[Message, ...] | None:

        if tools != self.tools or request_config != self.request_config:
            return None
        # Task snapshots are backend-origin data. A revision is appended after
        # the new results instead of replacing an earlier prompt segment.
        stable_context = tuple(
            message
            for message in context_messages
            if message.name != "muharness_active_task"
        )
        previous_stable = tuple(
            message
            for message in self.context_messages
            if message.name != "muharness_active_task"
        )
        if stable_context != previous_stable:
            return None
        previous_count = len(self.source_messages)
        if len(source_messages) < previous_count:
            return None
        if source_messages[:previous_count] != self.source_messages:
            return None
        previous_tasks = tuple(
            message
            for message in self.context_messages
            if message.name == "muharness_active_task"
        )
        current_tasks = tuple(
            message
            for message in context_messages
            if message.name == "muharness_active_task"
        )
        updates = current_tasks if current_tasks != previous_tasks else ()
        if previous_tasks and not current_tasks:
            updates = (
                Message(
                    role=MessageRole.USER,
                    name="muharness_active_task",
                    content="运行状态更新：当前会话没有活动 Task。此状态不是额外授权。",
                ),
            )
        return (*self.sent_messages, *source_messages[previous_count:], *updates)


# 函数说明：looks_like_textual_tool_call
# 用途：判断模型文本是否具有工具调用标记，用于受控提示或重试。
# 参数：
#   content：内容正文，类型 `str | None`。
# 返回：类型 `bool`；按分支返回 `False`；`any(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`content.lower`。
# 分支与异常：
#   当 `not content` 时，返回 `False`。
def looks_like_textual_tool_call(content: str | None) -> bool:

    if not content:
        return False
    lowered = content.lower()
    return any(
        marker in lowered
        for marker in (
            "<tool_calls",
            "<｜｜dsml｜｜tool_calls",
            "<｜｜dsml｜｜invoke",
        )
    )


# 函数说明：skill_read_outcome
# 用途：读取`outcome`，供模型与工具执行循环使用。
# 参数：
#   output：工具、模型或转换步骤的输出，类型 `object`。
# 返回：类型 `tuple[str | None, bool]`；按分支返回 `(None, False)`；
# `(normalized_name, payload.get('found') is True)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads`。
# 分支与异常：
#   捕获 `(ValueError, TypeError)` 后，返回 `(None, False)`。
def skill_read_outcome(output: object) -> tuple[str | None, bool]:

    if isinstance(output, str):
        try:
            payload = json.loads(output)
        except (ValueError, TypeError):
            return None, False
    elif isinstance(output, dict):
        payload = output
    else:
        return None, False
    name = payload.get("name")
    normalized_name = name if isinstance(name, str) and name else None
    return normalized_name, payload.get("found") is True


# 函数说明：plan_failure_message
# 用途：在模型与工具执行循环中处理 `plan_failure_message`，通过 `message.model_copy` 完
# 成首个内部处理步骤。
# 参数：
#   message：单条消息或通知，类型 `Message`。
#   prefix：`prefix`输入或配置值，类型 `str`。
# 返回：类型 `Message`；返回
# `message.model_copy(update={'content': f'{prefix}\n\n{content}'})`。
def plan_failure_message(message: Message, prefix: str) -> Message:

    content = message.content or ""
    return message.model_copy(update={"content": f"{prefix}\n\n{content}"})


# 函数说明：plan_task_id_from_output
# 用途：在模型与工具执行循环中处理 `plan_task_id_from_output`，通过 `json.loads` 完成首
# 个内部处理步骤。
# 参数：
#   output：工具、模型或转换步骤的输出，类型 `object`。
# 返回：类型 `str | None`；按分支返回 `None`；
# `task_id if isinstance(task_id, str) and task_id else None`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads`。
# 分支与异常：
#   捕获 `(ValueError, TypeError)` 后，返回 `None`。
#   当 `not isinstance(payload, dict)` 时，返回 `None`。
def plan_task_id_from_output(output: object) -> str | None:

    if isinstance(output, str):
        try:
            payload = json.loads(output)
        except (ValueError, TypeError):
            return None
    elif isinstance(output, dict):
        payload = output
    else:
        return None
    if not isinstance(payload, dict):
        return None
    task_id = payload.get("id")
    return task_id if isinstance(task_id, str) and task_id else None


# 函数说明：add_usage
# 用途：添加用量，供模型与工具执行循环使用。
# 参数：
#   total：传给 `add_model_usage` 的输入，类型 `ModelUsage`。
#   current：当前值或状态，类型 `ModelUsage`。
# 返回：类型 `ModelUsage`；返回 `add_model_usage(total, current)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`add_model_usage`。
def add_usage(total: ModelUsage, current: ModelUsage) -> ModelUsage:

    return add_model_usage(total, current)


# 函数说明：usage_call_count
# 用途：统计用量调用，供模型与工具执行循环使用。
# 参数：
#   usage：模型调用用量统计，类型 `ModelUsage`。
# 返回：类型 `int`；按分支返回 `usage.model_calls`；`1`；`0`。
# 分支与异常：
#   当 `usage.model_calls > 0` 时，返回 `usage.model_calls`。
#   当 `usage.input_tokens or usage.output_tokens or…` 时，返回 `1`。
def usage_call_count(usage: ModelUsage) -> int:

    if usage.model_calls > 0:
        return usage.model_calls
    if usage.input_tokens or usage.output_tokens or usage.total_tokens:
        return 1
    return 0


# 函数说明：run_budget_detail
# 用途：运行预算，供模型与工具执行循环使用。
# 参数：
#   decision：权限、上下文或审计决策，类型 `RunBudgetDecision`。
# 返回：类型 `str`；返回
# `f'reason={reason}, chargeable_tokens={decision.chargeable_tokens}, model_calls={…`。
def run_budget_detail(decision: RunBudgetDecision) -> str:

    reason = decision.reason.value if decision.reason is not None else "unknown"
    return (
        f"reason={reason}, chargeable_tokens={decision.chargeable_tokens}, "
        f"model_calls={decision.model_calls}"
    )


# 函数说明：run_budget_event_fields
# 用途：运行预算事件，供模型与工具执行循环使用。
# 参数：
#   decision：权限、上下文或审计决策，类型 `RunBudgetDecision`。
#   config：运行配置，类型 `RunBudgetConfig`。
#   status：目标状态，类型 `RunBudgetStatus | None`；默认 `None`。
# 返回：类型 `dict[str, object]`；字典，包含字段 `run_budget_status`、
# `run_budget_reason`、`run_budget_chargeable_tokens`、`run_budget_model_calls`、
# `run_budget_warning_tokens`、`run_budget_finalization_tokens`、
# `run_budget_hard_tokens`、`run_budget_warning_model_calls`、
# `run_budget_finalization_model_calls`、`run_budget_hard_model_calls`。
def run_budget_event_fields(
    decision: RunBudgetDecision,
    config: RunBudgetConfig,
    *,
    status: RunBudgetStatus | None = None,
) -> dict[str, object]:

    return {
        "run_budget_status": (status or decision.status).value,
        "run_budget_reason": (
            decision.reason.value if decision.reason is not None else None
        ),
        "run_budget_chargeable_tokens": decision.chargeable_tokens,
        "run_budget_model_calls": decision.model_calls,
        "run_budget_warning_tokens": config.warning_tokens,
        "run_budget_finalization_tokens": config.finalization_tokens,
        "run_budget_hard_tokens": config.hard_tokens,
        "run_budget_warning_model_calls": config.warning_model_calls,
        "run_budget_finalization_model_calls": config.finalization_model_calls,
        "run_budget_hard_model_calls": config.hard_model_calls,
    }


# 函数说明：reflection_tool_context
# 用途：在模型与工具执行循环中处理 `reflection_tool_context`，通过 `json.dumps` 完成首个
# 内部处理步骤。
# 参数：
#   records：记录集合输入或配置值，类型 `Sequence[ToolCallRecord]`。
#   max_chars：保留的字符数上限，类型 `int`。
# 返回：类型 `tuple[str, ...]`；按分支返回 `()`；`tuple(items)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
# 分支与异常：
#   当 `max_chars <= 0` 时，返回 `()`。
#   当 `remaining <= 0` 时，结束当前循环。
def reflection_tool_context(
    records: Sequence[ToolCallRecord],
    *,
    max_chars: int,
) -> tuple[str, ...]:

    if max_chars <= 0:
        return ()
    remaining = max_chars
    items: list[str] = []
    for record in records:
        payload = json.dumps(
            {
                "tool": record.tool_call.name,
                "arguments": record.tool_call.arguments,
                "success": record.result.success,
                "output": record.result.output,
                "error": record.result.error,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        if len(payload) > remaining:
            payload = payload[:remaining]
        if payload:
            items.append(payload)
            remaining -= len(payload)
        if remaining <= 0:
            break
    return tuple(items)


# 函数说明：recalled_memory_revisions
# 用途：在模型与工具执行循环中处理 `recalled_memory_revisions`，通过 `json.loads` 完成首
# 个内部处理步骤。
# 参数：
#   records：记录集合输入或配置值，类型 `Sequence[ToolCallRecord]`。
# 返回：类型 `dict[str, int]`；返回 `recalled`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` →
# `memory_id.strip().upper`。
# 分支与异常：
#   当 `record.tool_call.name != 'memory_read' or not…` 时，跳过当前循环项。
#   捕获 `json.JSONDecodeError` 后，跳过当前循环项，继续处理后续项。
#   当 `not isinstance(arguments, dict)` 时，跳过当前循环项。
#   当 `not isinstance(memory_id, str)` 时，跳过当前循环项。
#   捕获 `(json.JSONDecodeError, TypeError)` 后，跳过当前循环项，继续处理后续项。
def recalled_memory_revisions(
    records: Sequence[ToolCallRecord],
) -> dict[str, int]:

    recalled: dict[str, int] = {}
    for record in records:
        if record.tool_call.name != "memory_read" or not record.result.success:
            continue
        arguments: Any = record.tool_call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        memory_id = arguments.get("memory_id")
        if not isinstance(memory_id, str):
            continue
        try:
            output = json.loads(record.result.output or "")
        except (json.JSONDecodeError, TypeError):
            continue
        normalized = memory_id.strip().upper()
        revision = output.get("revision") if isinstance(output, dict) else None
        if (
            isinstance(output, dict)
            and output.get("found") is True
            and output.get("id") == normalized
            and isinstance(revision, int)
            and revision > 0
        ):
            recalled[normalized] = revision
    return recalled


# 函数说明：without_legacy_fixed_date
# 用途：在模型与工具执行循环中处理 `without_legacy_fixed_date`，通过
# `_LEGACY_DATE_PATTERN.sub` 完成首个内部处理步骤。
# 参数：
#   message：单条消息或通知，类型 `Message`。
# 返回：类型 `Message`；按分支返回 `message`；
# `message.model_copy(update={'content': cleaned})`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_LEGACY_DATE_PATTERN.sub`。
# 分支与异常：
#   当 `message.role is not MessageRole.SYSTEM or not…` 时，返回 `message`。
#   当 `cleaned == message.content` 时，返回 `message`。
def without_legacy_fixed_date(message: Message) -> Message:

    if message.role is not MessageRole.SYSTEM or not message.content:
        return message
    cleaned = _LEGACY_DATE_PATTERN.sub("", message.content)
    if cleaned == message.content:
        return message
    return message.model_copy(update={"content": cleaned})


# 函数说明：provider_name
# 用途：处理模型与工具执行循环中的 `provider_name` 数据；结果及边界条件见下方说明。
# 参数：
#   provider：模型或搜索服务商，类型 `ModelProvider | str | None`。
# 返回：类型 `str | None`；按分支返回 `provider.value`；`provider`。
# 分支与异常：
#   当 `isinstance(provider, ModelProvider)` 时，返回 `provider.value`。
def provider_name(provider: ModelProvider | str | None) -> str | None:

    if isinstance(provider, ModelProvider):
        return provider.value
    return provider


# 函数说明：tool_call_signature
# 用途：生成用于识别工具调用的比较签名。
# 参数：
#   tool_call：单次结构化工具调用，类型 `ToolCall`。
# 返回：类型 `str`；按分支返回 `f'{tool_call.name}:{arguments}'`；
# `f'{tool_call.name}:{canonical_arguments}'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` → `json.dumps`。
# 分支与异常：
#   捕获 `json.JSONDecodeError` 后，返回 `f'{tool_call.name}:{arguments}'`。
def tool_call_signature(tool_call: ToolCall) -> str:

    arguments: Any = tool_call.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"{tool_call.name}:{arguments}"

    canonical_arguments = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{tool_call.name}:{canonical_arguments}"


__all__ = [
    "RequestPrefixState",
    "add_usage",
    "looks_like_textual_tool_call",
    "plan_failure_message",
    "plan_task_id_from_output",
    "provider_name",
    "recalled_memory_revisions",
    "reflection_tool_context",
    "run_budget_detail",
    "run_budget_event_fields",
    "skill_read_outcome",
    "tool_call_signature",
    "usage_call_count",
    "without_legacy_fixed_date",
]
