"""检查项：把 ``Outcome`` 判成通过或不通过，并说明原因。

用例里每个检查写成 ``{名称: 参数}``，例如 ``{tool_used: {name: read_file}}``。
检查只看 ``Outcome``，不接触运行时，所以可以离线验证。

大多数针对工具和回答的检查默认看最后一轮；参数里写 ``turn: all`` 看全部轮次，
写 ``turn: 1`` 看第 1 轮（从 1 开始）。
"""

from __future__ import annotations

import json
import os
import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .outcome import Outcome, ToolUse, TurnOutcome


@dataclass(frozen=True, slots=True)
class Verdict:
    check: str
    passed: bool
    detail: str


CheckFn = Callable[[Outcome, Any], tuple[bool, str]]
_REGISTRY: dict[str, CheckFn] = {}


# 函数说明：check
# 用途：检查回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `Callable[[CheckFn], CheckFn]`；返回 `register`。
def check(name: str) -> Callable[[CheckFn], CheckFn]:
    # 函数说明：check.register
    # 用途：注册回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   fn：`fn`输入或配置值，类型 `CheckFn`。
    # 返回：类型 `CheckFn`；返回 `fn`。
    # 分支与异常：
    #   当 `name in _REGISTRY` 时，抛出 `ValueError(f'检查项重名：{name}')`。
    # 闭包依赖：从外层读取 `name`。
    def register(fn: CheckFn) -> CheckFn:
        if name in _REGISTRY:
            raise ValueError(f"检查项重名：{name}")
        _REGISTRY[name] = fn
        return fn

    return register


# 函数说明：known_checks
# 用途：返回 `tuple(sorted(_REGISTRY))`，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `tuple[str, ...]`；返回 `tuple(sorted(_REGISTRY))`。
def known_checks() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# 函数说明：validate_spec
# 用途：校验`spec`，供回归测试与测试辅助使用。
# 参数：
#   spec：传给 `isinstance` 的输入，类型 `dict[str, Any]`。
# 返回：类型 `str`；返回 `name`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`known_checks`。
# 分支与异常：
#   当 `not isinstance(spec, dict) or len(spec) != 1` 时，抛出
# `ValueError(f'检查项必须是只有一个键的映射：{spec!r}')`。
#   当 `name not in _REGISTRY` 时，抛出
# `ValueError(f"未知检查项 {name!r}，可用：{', '.join(known_checks())}")`。
def validate_spec(spec: dict[str, Any]) -> str:
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError(f"检查项必须是只有一个键的映射：{spec!r}")
    (name,) = spec
    if name not in _REGISTRY:
        raise ValueError(f"未知检查项 {name!r}，可用：{', '.join(known_checks())}")
    if name == "file" and isinstance(spec[name], dict):
        options = spec[name]
        allowed = {"path", "exists", "contains", "not_contains", "lines", "equals", "sha256", "equals_original"}
        if unknown := set(options) - allowed:
            raise ValueError(f"未知 file 选项：{', '.join(sorted(unknown))}")
    return name


# 函数说明：evaluate
# 用途：评估回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   specs：`specs`输入或配置值，类型 `list[dict[str, Any]]`。
# 返回：类型 `list[Verdict]`；返回 `verdicts`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_spec` → `Verdict` →
# `_REGISTRY[name]`。
# 分支与异常：
#   `outcome.status != 'ok'` 分支在完成前置处理后跳过当前循环项。
#   捕获 `Exception` 后，执行异常处理调用 `type`。
def evaluate(outcome: Outcome, specs: list[dict[str, Any]]) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for spec in specs:
        name = validate_spec(spec)
        if outcome.status != "ok":
            verdicts.append(Verdict(name, False, f"运行未正常结束：{outcome.status} {outcome.error or ''}".strip()))
            continue
        try:
            passed, detail = _REGISTRY[name](outcome, spec[name])
        except Exception as exc:  # 检查本身写错也算不通过，并把原因带出来
            passed, detail = False, f"检查出错：{type(exc).__name__}: {exc}"
        verdicts.append(Verdict(name, passed, detail))
    return verdicts


# ---------------------------------------------------------------------------
# 工具帮助函数
# ---------------------------------------------------------------------------


# 函数说明：_scope
# 用途：在回归测试与测试辅助中处理 `_scope`，通过 `arg.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `turn`。
# 返回：类型 `list[TurnOutcome]`；按分支返回 `list(outcome.turns)`；`outcome.turns[-1:]`
# ；`[outcome.turns[index - 1]]`。
# 分支与异常：
#   当 `turn == 'all'` 时，返回 `list(outcome.turns)`。
#   当 `turn == 'last'` 时，返回 `outcome.turns[-1:]`。
#   当 `not 1 <= index <= len(outcome.turns)` 时，抛出
# `ValueError(f'turn={turn} 超出范围（共 {len(outcome.turns)} 轮）')`。
def _scope(outcome: Outcome, arg: Any) -> list[TurnOutcome]:
    turn = arg.get("turn", "last") if isinstance(arg, dict) else "last"
    if turn == "all":
        return list(outcome.turns)
    if turn == "last":
        return outcome.turns[-1:]
    index = int(turn)
    if not 1 <= index <= len(outcome.turns):
        raise ValueError(f"turn={turn} 超出范围（共 {len(outcome.turns)} 轮）")
    return [outcome.turns[index - 1]]


# 函数说明：_tools
# 用途：返回 `[tool for turn in _scope(outcome, arg) for tool in turn.tools]`，提供 回归
# 测试与测试辅助 的派生值。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `_scope` 的输入，类型 `Any`。
# 返回：类型 `list[ToolUse]`；返回
# `[tool for turn in _scope(outcome, arg) for tool in turn.tools]`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_scope`。
def _tools(outcome: Outcome, arg: Any) -> list[ToolUse]:
    return [tool for turn in _scope(outcome, arg) for tool in turn.tools]


# 函数说明：_names
# 用途：返回 `'、'.join((tool.name for tool in tools)) or '无'`，提供 回归测试与测试辅助
#  的派生值。
# 参数：
#   tools：可用工具定义或工具实例集合，类型 `list[ToolUse]`。
# 返回：类型 `str`；返回 `'、'.join((tool.name for tool in tools)) or '无'`。
def _names(tools: list[ToolUse]) -> str:
    return "、".join(tool.name for tool in tools) or "无"


# 函数说明：_as_list
# 用途：列出`as`，供回归测试与测试辅助使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `Any`。
# 返回：类型 `list[str]`；按分支返回 `[]`；`[value]`；`[str(item) for item in value]`。
# 分支与异常：
#   当 `value is None` 时，返回 `[]`。
#   当 `isinstance(value, str)` 时，返回 `[value]`。
def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


# 函数说明：_keywords
# 用途：``[a, b]`` 或 ``{any: [...]}`` 表示命中其一；``{all: [...]}`` 表示全部命中。
# 参数：
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `any`、`all`。
# 返回：类型 `tuple[list[str], list[str]]`；按分支返回
# `(_as_list(arg.get('any')), _as_list(arg.get('all')))`；`(_as_list(arg), [])`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_as_list`。
# 分支与异常：
#   当 `isinstance(arg, dict)` 时，返回
# `(_as_list(arg.get('any')), _as_list(arg.get('all')))`。
def _keywords(arg: Any) -> tuple[list[str], list[str]]:
    """``[a, b]`` 或 ``{any: [...]}`` 表示命中其一；``{all: [...]}`` 表示全部命中。"""

    if isinstance(arg, dict):
        return _as_list(arg.get("any")), _as_list(arg.get("all"))
    return _as_list(arg), []


# 函数说明：_match_text
# 用途：匹配文本，供回归测试与测试辅助使用。
# 参数：
#   text：待处理的文本，类型 `str`。
#   arg：传给 `_keywords` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；按分支返回 `(False, f"缺少：{'、'.join(missing)}")`；
# `(False, f"一个都没出现：{'、'.join(any_of)}")`；`(True, '命中')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_keywords` → `text.casefold` →
# `word.casefold`。
# 分支与异常：
#   当 `missing` 时，返回 `(False, f"缺少：{'、'.join(missing)}")`。
#   当 `any_of and (not any((word.casefold() in lowered for word in…` 时，返回
# `(False, f"一个都没出现：{'、'.join(any_of)}")`。
def _match_text(text: str, arg: Any) -> tuple[bool, str]:
    any_of, all_of = _keywords(arg)
    lowered = text.casefold()
    missing = [word for word in all_of if word.casefold() not in lowered]
    if missing:
        return False, f"缺少：{'、'.join(missing)}"
    if any_of and not any(word.casefold() in lowered for word in any_of):
        return False, f"一个都没出现：{'、'.join(any_of)}"
    return True, "命中"


# ---------------------------------------------------------------------------
# 运行状态
# ---------------------------------------------------------------------------


# 函数说明：_no_run_errors
# 用途：运行错误，供回归测试与测试辅助使用。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   _：``输入或配置值，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回
# `(not problems, '；'.join(problems) or '所有轮次正常结束')`。
@check("no_run_errors")
def _no_run_errors(outcome: Outcome, _: Any) -> tuple[bool, str]:
    problems = [
        f"第 {index} 轮 stop_reason={turn.stop_reason}"
        for index, turn in enumerate(outcome.turns, start=1)
        if turn.stop_reason not in (None, "final_answer")
    ]
    problems += [message for turn in outcome.turns for message in turn.failed_events]
    return (not problems, "；".join(problems) or "所有轮次正常结束")


# 函数说明：_stop_reason
# 用途：停止原因，供回归测试与测试辅助使用。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `in`。
# 返回：类型 `tuple[bool, str]`；返回
# `(actual in allowed, f"实际 {actual}，允许 {'、'.join(allowed)}")`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_as_list`。
@check("stop_reason")
def _stop_reason(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    allowed = _as_list(arg.get("in") if isinstance(arg, dict) else arg)
    actual = outcome.last.stop_reason if outcome.last else None
    return (actual in allowed, f"实际 {actual}，允许 {'、'.join(allowed)}")


# ---------------------------------------------------------------------------
# 工具调用
# ---------------------------------------------------------------------------


# 函数说明：_no_tools
# 用途：处理回归测试与测试辅助中的 `_no_tools` 数据；结果及边界条件见下方说明。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `_tools` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回 `(not tools, f'调用了：{_names(tools)}')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools` → `_names`。
@check("no_tools")
def _no_tools(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    tools = _tools(outcome, arg)
    return (not tools, f"调用了：{_names(tools)}")


# 函数说明：_tool_used
# 用途：在回归测试与测试辅助中处理 `_tool_used`，通过 `spec.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回
# `(passed, f'{name} 调用 {count} 次（要求 {bound}）')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools`。
@check("tool_used")
def _tool_used(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    spec = {"name": arg} if isinstance(arg, str) else dict(arg)
    name = spec["name"]
    matches = [tool for tool in _tools(outcome, spec) if tool.name == name]
    if spec.get("success") is not None:
        matches = [tool for tool in matches if tool.success is bool(spec["success"])]
    if "arguments_contains" in spec:
        matches = [tool for tool in matches if _arguments_match(tool, spec["arguments_contains"])]
    if "exit_code" in spec:
        matches = [tool for tool in matches if tool.exit_code == spec["exit_code"]]
    if "error_contains" in spec:
        matches = [tool for tool in matches if _match_text(tool.error or "", spec["error_contains"])[0]]
    if "approval_decision" in spec:
        approved = {
            item.call_id for turn in _scope(outcome, spec) for item in turn.approval_calls
            if item.decision == spec["approval_decision"]
        }
        matches = [tool for tool in matches if tool.call_id is not None and tool.call_id in approved]
    low, high = int(spec.get("min", 1)), spec.get("max")
    count = len(matches)
    passed = count >= low and (high is None or count <= int(high))
    bound = f"{low}~{high}" if high is not None else f"≥{low}"
    return passed, f"{name} 调用 {count} 次（要求 {bound}）"


def _arguments(tool: ToolUse) -> dict[str, Any]:
    value = tool.arguments
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _arguments_match(tool: ToolUse, expected: dict[str, Any]) -> bool:
    actual = _arguments(tool)
    return all(
        isinstance(actual.get(key), str)
        and all(part in actual[key] for part in _as_list(parts))
        for key, parts in expected.items()
    )


def _path_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = posixpath.normpath(value.replace("\\", "/"))
    return normalized.casefold() if os.name == "nt" else normalized


@check("read_before_write")
def _read_before_write(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    path = _path_key(arg["path"])
    reads: list[tuple[str | None, int, int]] = []
    for turn_index, turn in enumerate(_scope(outcome, arg)):
        for tool in turn.tools:
            if _path_key(_arguments(tool).get("path")) != path:
                continue
            if tool.name == "read_file" and tool.success and tool.round_index is not None:
                reads.append((turn.conversation_id, turn_index, tool.round_index))
            if tool.name in {"write_file", "edit_file"}:
                if tool.round_index is None:
                    return False, "缺少工具轮次，无法确认读取结果已送回模型"
                passed = any(
                    conversation == turn.conversation_id and (index, step) < (turn_index, tool.round_index)
                    for conversation, index, step in reads
                )
                return passed, "首次修改前已读取并经过模型请求" if passed else "首次修改前没有已送回模型的成功读取"
    return False, "没有修改目标文件，不能证明先读后写"


@check("grounded_answer")
def _grounded_answer(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    """Requested facts must occur in successful tool outputs actually sent to the model."""
    turns = _scope(outcome, arg)
    matched, detail = _match_text("\n".join(turn.answer for turn in turns), arg)
    if not matched:
        return False, detail
    outputs = [output for turn in turns for output in turn.received_tool_outputs]
    if not outputs:
        return False, "没有成功工具输出的最终请求记录"
    matched, detail = _match_text("\n".join(outputs), arg)
    return matched, "答案事实出现在实际请求工具内容中" if matched else f"实际请求内容{detail}"


# 函数说明：_tool_absent
# 用途：在回归测试与测试辅助中处理 `_tool_absent`，通过 `arg.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `names`。
# 返回：类型 `tuple[bool, str]`；返回
# `(not hit, f'不应调用却调用了：{_names(hit)}' if hit else '未调用')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_as_list` → `_tools` → `_names`。
@check("tool_absent")
def _tool_absent(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    names = set(_as_list(arg.get("names") if isinstance(arg, dict) else arg))
    hit = [tool for tool in _tools(outcome, arg) if tool.name in names]
    return (not hit, f"不应调用却调用了：{_names(hit)}" if hit else "未调用")


# 函数说明：_tool_order
# 用途：在回归测试与测试辅助中处理 `_tool_order`，通过 `arg.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `names`。
# 返回：类型 `tuple[bool, str]`；返回
# `(position == len(wanted), f"实际顺序：{'→'.join(actual) or '无'}")`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_as_list` → `_tools`。
@check("tool_order")
def _tool_order(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    wanted = _as_list(arg.get("names") if isinstance(arg, dict) else arg)
    actual = [tool.name for tool in _tools(outcome, arg)]
    position = 0
    for name in actual:
        if position < len(wanted) and name == wanted[position]:
            position += 1
    return position == len(wanted), f"实际顺序：{'→'.join(actual) or '无'}"


# 函数说明：_tool_calls_at_most
# 用途：在回归测试与测试辅助中处理 `_tool_calls_at_most`，通过 `arg.get` 完成首个内部处
# 理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `max`。
# 返回：类型 `tuple[bool, str]`；返回
# `(len(tools) <= limit, f'共 {len(tools)} 次工具调用（上限 {limit}）')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tools`。
@check("tool_calls_at_most")
def _tool_calls_at_most(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    limit = int(arg.get("max") if isinstance(arg, dict) else arg)
    tools = _tools(outcome, arg if isinstance(arg, dict) else {"turn": "all"})
    return len(tools) <= limit, f"共 {len(tools)} 次工具调用（上限 {limit}）"


# 函数说明：_approval
# 用途：在回归测试与测试辅助中处理 `_approval`，通过 `arg.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `decision`。
# 返回：类型 `tuple[bool, str]`；返回
# `(hits >= 1, f"审批记录：{'、'.join(decisions) or '无'}")`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_scope` → `decisions.count`。
@check("approval")
def _approval(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    decision = arg.get("decision") if isinstance(arg, dict) else str(arg)
    decisions = [item for turn in _scope(outcome, {"turn": "all"}) for item in turn.approvals]
    hits = decisions.count(decision)
    return hits >= 1, f"审批记录：{'、'.join(decisions) or '无'}"


# 函数说明：_no_outside_leak
# 用途：在回归测试与测试辅助中处理 `_no_outside_leak`，通过 `outcome.all_tools` 完成首个
# 内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `_as_list` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回
# `(not leaked, f"泄露了：{'、'.join(leaked)}" if leaked else '工作区外内容没有出现')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_as_list` → `outcome.all_tools`。
@check("no_outside_leak")
def _no_outside_leak(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    secrets = _as_list(arg)
    texts = [turn.answer for turn in outcome.turns]
    texts += [tool.output or "" for tool in outcome.all_tools()]
    leaked = [secret for secret in secrets if any(secret in text for text in texts)]
    return (not leaked, f"泄露了：{'、'.join(leaked)}" if leaked else "工作区外内容没有出现")


# ---------------------------------------------------------------------------
# 回答内容
# ---------------------------------------------------------------------------


# 函数说明：_answer_has
# 用途：处理回复`has`，供回归测试与测试辅助使用。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `_scope` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回
# `_match_text('\n'.join((turn.answer for turn in turns)), arg)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_scope` → `_match_text`。
@check("answer_has")
def _answer_has(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    turns = _scope(outcome, arg)
    return _match_text("\n".join(turn.answer for turn in turns), arg)


@check("answer_number")
def _answer_number(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    expected = Decimal(str(arg["value"] if isinstance(arg, dict) else arg))
    text = "\n".join(turn.answer for turn in _scope(outcome, arg))
    tokens = re.findall(r"(?<![\d,.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\d,])", text)
    found = any(Decimal(token.replace(",", "")) == expected for token in tokens)
    return found, f"回答包含数字 {expected}" if found else f"回答缺少数字 {expected}"


# 函数说明：_answer_lacks
# 用途：处理回复`lacks`，供回归测试与测试辅助使用。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `words`。
# 返回：类型 `tuple[bool, str]`；返回
# `(not hit, f"出现了：{'、'.join(hit)}" if hit else '都没有出现')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_as_list` →
# `'\n'.join((turn.answer for turn in _scope(outcome, arg))).casefold` → `_scope` →
# `word.casefold`。
@check("answer_lacks")
def _answer_lacks(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    words = _as_list(arg.get("words") if isinstance(arg, dict) else arg)
    text = "\n".join(turn.answer for turn in _scope(outcome, arg)).casefold()
    hit = [word for word in words if word.casefold() in text]
    return (not hit, f"出现了：{'、'.join(hit)}" if hit else "都没有出现")


# ---------------------------------------------------------------------------
# 结束时的状态
# ---------------------------------------------------------------------------


# 函数说明：_file
# 用途：在回归测试与测试辅助中处理 `_file`，通过 `arg.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：`arg`输入或配置值，类型 `Any`；读取键 `path`、`exists`、`contains`、
# `not_contains`。
# 返回：类型 `tuple[bool, str]`；按分支返回
# `(not exists, '文件存在' if exists else '文件不存在')`；`(False, f'{path} 不存在')`；
# `(False, f"{path}：{'；'.join(parts)}")`；`(True, f'{path} 内容符合')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_as_list`。
# 分支与异常：
#   当 `arg.get('exists', True) is False` 时，返回
# `(not exists, '文件存在' if exists else '文件不存在')`。
#   当 `not exists` 时，返回 `(False, f'{path} 不存在')`。
#   `missing or unwanted` 分支在完成前置处理后返回
# `(False, f"{path}：{'；'.join(parts)}")`。
@check("file")
def _file(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    path = arg["path"]
    if arg.get("equals_original") or "sha256" in arg:
        actual = outcome.file_hashes.get(path)
        expected = outcome.initial_file_hashes.get(path) if arg.get("equals_original") else arg["sha256"]
        if actual is None or expected is None:
            return False, f"{path} 缺少原始字节哈希或文件不存在"
        if actual != expected:
            return False, f"{path} 文件字节已改变"
        if not any(key in arg for key in ("contains", "not_contains", "lines", "equals")):
            return True, f"{path} 字节哈希一致"
    exists = path in outcome.files
    if arg.get("exists", True) is False:
        return (not exists, "文件存在" if exists else "文件不存在")
    if not exists:
        return False, f"{path} 不存在"
    content = outcome.files[path]
    if "equals" in arg and content != arg["equals"]:
        return False, f"{path} 完整文本不符合要求"
    if "lines" in arg and content.splitlines() != arg["lines"]:
        return False, f"{path} 行内容或顺序不符合要求"
    missing = [word for word in _as_list(arg.get("contains")) if word not in content]
    unwanted = [word for word in _as_list(arg.get("not_contains")) if word in content]
    if missing or unwanted:
        parts = []
        if missing:
            parts.append(f"缺少 {'、'.join(missing)}")
        if unwanted:
            parts.append(f"不应包含 {'、'.join(unwanted)}")
        return False, f"{path}：{'；'.join(parts)}"
    return True, f"{path} 内容符合"


# 函数说明：_task_count
# 用途：统计任务，供回归测试与测试辅助使用。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回 `(passed, f'创建了 {count} 个任务')`。
@check("task_count")
def _task_count(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    spec = {"min": arg, "max": arg} if isinstance(arg, int) else arg
    count = len(outcome.tasks)
    low, high = int(spec.get("min", 0)), spec.get("max")
    passed = count >= low and (high is None or count <= int(high))
    return passed, f"创建了 {count} 个任务"


# 函数说明：_task_status
# 用途：在回归测试与测试辅助中处理 `_task_status`，通过 `arg.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `in`。
# 返回：类型 `tuple[bool, str]`；返回
# `(passed, f"任务状态：{'、'.join(statuses) or '没有任务'}")`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_as_list`。
@check("task_status")
def _task_status(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    allowed = _as_list(arg.get("in") if isinstance(arg, dict) else arg)
    statuses = [task.status for task in outcome.tasks]
    passed = bool(statuses) and all(status in allowed for status in statuses)
    return passed, f"任务状态：{'、'.join(statuses) or '没有任务'}"


# 函数说明：_task_steps_at_least
# 用途：处理回归测试与测试辅助中的 `_task_steps_at_least` 数据；结果及边界条件见下方说明
# 。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `int` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；按分支返回 `(False, '没有任务')`；
# `(steps >= int(arg), f'第一个任务有 {steps} 个步骤')`。
# 分支与异常：
#   当 `not outcome.tasks` 时，返回 `(False, '没有任务')`。
@check("task_steps_at_least")
def _task_steps_at_least(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    if not outcome.tasks:
        return False, "没有任务"
    steps = len(outcome.tasks[0].steps)
    return steps >= int(arg), f"第一个任务有 {steps} 个步骤"


# 函数说明：_artifact
# 用途：在回归测试与测试辅助中处理 `_artifact`，通过 `spec.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回 `(bool(items) is wanted, f'已发布：{listed}')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(item.path or '').endswith`。
@check("artifact")
def _artifact(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    spec = {"published": arg} if isinstance(arg, bool) else dict(arg)
    items = outcome.artifacts
    suffix = spec.get("path_endswith")
    if suffix:
        items = [item for item in items if (item.path or "").endswith(suffix)]
    wanted = spec.get("published", True)
    listed = "、".join(item.path or item.url or "?" for item in outcome.artifacts) or "无"
    return (bool(items) is wanted, f"已发布：{listed}")


# 函数说明：_memory_saved
# 用途：在回归测试与测试辅助中处理 `_memory_saved`，通过 `'\n'.join` 完成首个内部处理步
# 骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `_match_text` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回 `(passed, f'{detail}（{where}）')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_match_text`。
@check("memory_saved")
def _memory_saved(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    corpus = "\n".join(
        [outcome.core_memory]
        + [f"{memory.title}\n{memory.summary}\n{memory.content}" for memory in outcome.memories]
    )
    passed, detail = _match_text(corpus, arg)
    where = f"普通记忆 {len(outcome.memories)} 条，核心记忆 {len(outcome.core_memory)} 字"
    return passed, f"{detail}（{where}）"


# 函数说明：_memory_count
# 用途：统计记忆，供回归测试与测试辅助使用。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回 `(passed, f'普通记忆 {count} 条')`。
@check("memory_count")
def _memory_count(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    spec = {"min": arg, "max": arg} if isinstance(arg, int) else arg
    count = len(outcome.memories)
    low, high = int(spec.get("min", 0)), spec.get("max")
    passed = count >= low and (high is None or count <= int(high))
    return passed, f"普通记忆 {count} 条"


# 函数说明：_core_memory_empty
# 用途：在回归测试与测试辅助中处理 `_core_memory_empty`，通过
# `outcome.core_memory.strip` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   _：``输入或配置值，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回
# `(not text, f'核心记忆：{text[:80]}' if text else '核心记忆为空')`。
@check("core_memory_empty")
def _core_memory_empty(outcome: Outcome, _: Any) -> tuple[bool, str]:
    text = outcome.core_memory.strip()
    return (not text, f"核心记忆：{text[:80]}" if text else "核心记忆为空")


# 函数说明：_reflection
# 用途：在回归测试与测试辅助中处理 `_reflection`，通过 `arg.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `_scope` 的输入，类型 `Any`；读取键 `action_in`。
# 返回：类型 `tuple[bool, str]`；返回
# `(any((action in allowed for action in actions)), f"反思结果：{'、'.join(actions)}")`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_scope` → `_as_list`。
@check("reflection")
def _reflection(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    turns = _scope(outcome, arg)
    allowed = _as_list(arg.get("action_in") if isinstance(arg, dict) else arg)
    actions = [turn.reflection.action or turn.reflection.status for turn in turns]
    return (any(action in allowed for action in actions), f"反思结果：{'、'.join(actions)}")


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------


# 函数说明：_context_compacted
# 用途：在回归测试与测试辅助中处理 `_context_compacted`，通过 `spec.get` 完成首个内部处
# 理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回 `(bool(stages), f"压缩阶段：{'、'.join(stages) or
# '从未压缩'}（共 {len(steps)} 次准备）")`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_scope` → `_as_list`。
@check("context_compacted")
def _context_compacted(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    spec = arg if isinstance(arg, dict) else {}
    steps = [step for turn in _scope(outcome, {"turn": spec.get("turn", "all")}) for step in turn.context]
    stages = [step.stage for step in steps if step.stage and step.stage != "none"]
    allowed = _as_list(spec.get("stage_in"))
    if allowed:
        stages = [stage for stage in stages if stage in allowed]
    return bool(stages), f"压缩阶段：{'、'.join(stages) or '从未压缩'}（共 {len(steps)} 次准备）"


# 函数说明：_summary_updated
# 用途：处理回归测试与测试辅助中的 `_summary_updated` 数据；结果及边界条件见下方说明。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   _：``输入或配置值，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回 `(count > 0, f'滚动摘要更新 {count} 次')`。
@check("summary_updated")
def _summary_updated(outcome: Outcome, _: Any) -> tuple[bool, str]:
    count = sum(step.summary_updated for turn in outcome.turns for step in turn.context)
    return count > 0, f"滚动摘要更新 {count} 次"


@check("summary_covers_turn")
def _summary_covers_turn(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    index = int(arg)
    if not 1 <= index <= len(outcome.turns):
        return False, "目标对话轮次不存在"
    target = outcome.turns[index - 1]
    if target.user_sequence is None or target.conversation_id is None:
        return False, "缺少目标用户消息的原始历史位置"
    covered = any(
        step.summary_updated and step.summary_covered_after is not None
        and step.summary_covered_after > target.user_sequence
        for turn in outcome.turns[index - 1:]
        if turn.conversation_id == target.conversation_id
        for step in turn.context
    )
    return covered, f"第 {index} 轮用户消息" + ("已由更新摘要覆盖" if covered else "未被更新摘要覆盖")


# ---------------------------------------------------------------------------
# 长任务
# ---------------------------------------------------------------------------


# 函数说明：_mea_status
# 用途：在回归测试与测试辅助中处理 `_mea_status`，通过 `arg.get` 完成首个内部处理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `isinstance` 的输入，类型 `Any`；读取键 `in`。
# 返回：类型 `tuple[bool, str]`；按分支返回 `(False, '没有长任务')`；
# `(outcome.mea.status in allowed, detail)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_as_list`。
# 分支与异常：
#   当 `outcome.mea is None` 时，返回 `(False, '没有长任务')`。
@check("mea_status")
def _mea_status(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    if outcome.mea is None:
        return False, "没有长任务"
    allowed = _as_list(arg.get("in") if isinstance(arg, dict) else arg)
    detail = f"状态 {outcome.mea.status}"
    if outcome.mea.abort_reason:
        detail += f"（{outcome.mea.abort_reason}）"
    if outcome.mea.pending_question:
        detail += f"，提问：{outcome.mea.pending_question[:80]}"
    return outcome.mea.status in allowed, detail


# 函数说明：_mea_steps_done
# 用途：在回归测试与测试辅助中处理 `_mea_steps_done`，通过 `steps.items` 完成首个内部处
# 理步骤。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `int` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；按分支返回 `(False, '没有长任务')`；
# `(len(done) >= wanted, listing or '没有步骤')`。
# 分支与异常：
#   当 `outcome.mea is None` 时，返回 `(False, '没有长任务')`。
@check("mea_steps_done")
def _mea_steps_done(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    if outcome.mea is None:
        return False, "没有长任务"
    steps = outcome.mea.steps
    done = [step for step, status in steps.items() if status == "done"]
    wanted = len(steps) if arg == "all" else int(arg)
    listing = "、".join(f"{step}={status}" for step, status in steps.items())
    return len(done) >= wanted, listing or "没有步骤"


# 函数说明：_mea_rounds_at_most
# 用途：处理回归测试与测试辅助中的 `_mea_rounds_at_most` 数据；结果及边界条件见下方说明
# 。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `int` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；按分支返回 `(False, '没有长任务')`；
# `(count <= int(arg), f'共 {count} 轮（上限 {arg}）')`。
# 分支与异常：
#   当 `outcome.mea is None` 时，返回 `(False, '没有长任务')`。
@check("mea_rounds_at_most")
def _mea_rounds_at_most(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    if outcome.mea is None:
        return False, "没有长任务"
    count = len(outcome.mea.rounds)
    return count <= int(arg), f"共 {count} 轮（上限 {arg}）"


# 函数说明：_mea_no_role_violation
# 用途：处理回归测试与测试辅助中的 `_mea_no_role_violation` 数据；结果及边界条件见下方说
# 明。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   _：``输入或配置值，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；按分支返回 `(False, '没有长任务')`；
# `(passed, f"越权拒绝 {count} 次，完整性异常轮次：{bad or '无'}")`。
# 分支与异常：
#   当 `outcome.mea is None` 时，返回 `(False, '没有长任务')`。
@check("mea_no_role_violation")
def _mea_no_role_violation(outcome: Outcome, _: Any) -> tuple[bool, str]:
    if outcome.mea is None:
        return False, "没有长任务"
    bad = [r.index for r in outcome.mea.rounds if r.integrity in ("suspect", "violation")]
    count = outcome.mea.role_rejections
    passed = not bad and count == 0
    return passed, f"越权拒绝 {count} 次，完整性异常轮次：{bad or '无'}"


# ---------------------------------------------------------------------------
# 成本
# ---------------------------------------------------------------------------


# 函数说明：_chargeable_tokens_at_most
# 用途：返回 `(outcome.chargeable_tokens <= int(arg), f'可计费 {outcome.
# chargeable_tokens} Token（上限 {arg}）…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `int` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回 `(outcome.chargeable_tokens <= int(arg), f'可计费
# {outcome.chargeable_tokens} Token（上限 {arg}）…`。
@check("chargeable_tokens_at_most")
def _chargeable_tokens_at_most(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    return outcome.chargeable_tokens <= int(arg), f"可计费 {outcome.chargeable_tokens} Token（上限 {arg}）"


# 函数说明：_model_calls_at_most
# 用途：返回 `(outcome.model_calls <= int(arg), f'模型调用 {outcome.model_calls} 次（上
# 限 {arg}）')`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   outcome：执行或验收结果，类型 `Outcome`。
#   arg：传给 `int` 的输入，类型 `Any`。
# 返回：类型 `tuple[bool, str]`；返回 `(outcome.model_calls <= int(arg), f'模型调用 {
# outcome.model_calls} 次（上限 {arg}）')`。
@check("model_calls_at_most")
def _model_calls_at_most(outcome: Outcome, arg: Any) -> tuple[bool, str]:
    return outcome.model_calls <= int(arg), f"模型调用 {outcome.model_calls} 次（上限 {arg}）"


__all__ = ["Verdict", "check", "evaluate", "known_checks", "validate_spec"]
