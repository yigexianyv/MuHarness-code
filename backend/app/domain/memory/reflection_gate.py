
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ReflectionGateReason(StrEnum):

    SMALLTALK = "smalltalk"
    CAPABILITY_QUERY = "capability_query"
    EPHEMERAL_LOOKUP = "ephemeral_lookup"
    DURABLE_SIGNAL = "durable_signal"
    RECALLED_MEMORY = "recalled_memory"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class ReflectionGateDecision:
    should_reflect: bool
    reason: ReflectionGateReason


_SMALLTALK = frozenset(
    {
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "谢谢",
        "谢谢你",
        "好的",
        "好",
        "收到",
        "明白了",
        "辛苦了",
        "再见",
        "hi",
        "hello",
        "hey",
        "thanks",
        "thankyou",
        "ok",
        "okay",
        "gotit",
        "bye",
    }
)
_DURABLE_MARKERS = (
    "以后",
    "从现在开始",
    "一直",
    "记住",
    "偏好",
    "我喜欢",
    "我不喜欢",
    "我住",
    "我的项目",
    "项目决定",
    "最终决定",
    "规则",
    "改为",
    "纠正",
    "长期",
    "remember",
    "fromnowon",
    "iprefer",
    "ialways",
    "inever",
    "ilive",
    "myproject",
    "wedecided",
    "projectdecision",
)
_CAPABILITY_NOUNS = ("mcp", "skill", "技能", "工具", "功能", "能力")
_CAPABILITY_QUESTIONS = (
    "有什么",
    "有哪些",
    "支持什么",
    "支持哪些",
    "能做什么",
    "可以做什么",
    "列出",
    "看看",
    "看一下",
    "看下",
    "what",
    "which",
    "list",
    "available",
    "support",
)
_EPHEMERAL_NOUNS = (
    "天气",
    "气温",
    "温度",
    "几点",
    "当前时间",
    "现在时间",
    "今天日期",
    "weather",
    "temperature",
    "forecast",
    "whattime",
    "currenttime",
    "todaydate",
)


# 函数说明：decide_reflection_gate
# 用途：作出决策反思，供长期记忆管理与检索使用。
# 参数：
#   user_input：本次用户输入，类型 `str`。
#   recalled_memory_ids：记忆输入或配置值，类型 `tuple[str, ...]`；默认 `()`。
# 返回：类型 `ReflectionGateDecision`；按分支返回
# `ReflectionGateDecision(True, ReflectionGateReason.RECALLED_MEMORY)`；
# `ReflectionGateDecision(True, ReflectionGateReason.DURABLE_SIGNAL)`；
# `ReflectionGateDecision(False, ReflectionGateReason.SMALLTALK)`；
# `ReflectionGateDecision(False, ReflectionGateReason.CAPABILITY_QUERY)` 等 6 种表达式。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_normalize` →
# `ReflectionGateDecision`。
# 分支与异常：
#   当 `recalled_memory_ids` 时，返回 `ReflectionGateDecision(…)`。
#   当 `any((marker in normalized for marker in _DURABLE_MARKERS))` 时，返回
# `ReflectionGateDecision(…)`。
#   当 `normalized in _SMALLTALK` 时，返回 `ReflectionGateDecision(…)`。
#   当 `any((noun in normalized for noun in _CAPABILITY_NOUNS)) and…` 时，返回
# `ReflectionGateDecision(…)`。
def decide_reflection_gate(
    user_input: str,
    *,
    recalled_memory_ids: tuple[str, ...] = (),
) -> ReflectionGateDecision:

    normalized = _normalize(user_input)
    if recalled_memory_ids:
        return ReflectionGateDecision(True, ReflectionGateReason.RECALLED_MEMORY)
    if any(marker in normalized for marker in _DURABLE_MARKERS):
        return ReflectionGateDecision(True, ReflectionGateReason.DURABLE_SIGNAL)
    if normalized in _SMALLTALK:
        return ReflectionGateDecision(False, ReflectionGateReason.SMALLTALK)
    if (
        any(noun in normalized for noun in _CAPABILITY_NOUNS)
        and any(question in normalized for question in _CAPABILITY_QUESTIONS)
    ):
        return ReflectionGateDecision(
            False,
            ReflectionGateReason.CAPABILITY_QUERY,
        )
    if any(noun in normalized for noun in _EPHEMERAL_NOUNS):
        return ReflectionGateDecision(
            False,
            ReflectionGateReason.EPHEMERAL_LOOKUP,
        )
    return ReflectionGateDecision(True, ReflectionGateReason.UNCERTAIN)


# 函数说明：_normalize
# 用途：规范化长期记忆管理与检索，供长期记忆管理与检索使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `str`；返回 `re.sub('[\\W_]+', '', value.casefold(), flags=re.UNICODE)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`re.sub` → `value.casefold`。
def _normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)


__all__ = [
    "ReflectionGateDecision",
    "ReflectionGateReason",
    "decide_reflection_gate",
]
