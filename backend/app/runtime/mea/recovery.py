"""恢复审计需要的中断记录：被打断的 Executor 做过哪些工具调用。

检查点说明哪些调用已经返回结果、哪些已发出还没有结果；调用参数取自 Trace 里的
tool_started 事件（检查点只保留待执行调用的参数）。两者都查不到时返回空列表，
恢复审计照常进行，只是少了这段记录。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.models.types import ToolCall, ToolResult
from app.runtime.agent.events import AgentEventType

logger = logging.getLogger("muharness.mea.recovery")

_ARGUMENT_CHARS = 300
_OUTPUT_CHARS = 200
_MAX_CALLS = 60


class RecoveryInfoProvider:

    # 函数说明：RecoveryInfoProvider.__init__
    # 用途：初始化 RecoveryInfoProvider；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   checkpoint_store：运行检查点存储，类型 `Any`。
    #   trace_store：执行轨迹存储，类型 `Any | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._checkpoints`、`self._traces`。
    def __init__(self, checkpoint_store: Any, trace_store: Any | None = None) -> None:
        self._checkpoints = checkpoint_store
        self._traces = trace_store

    # 函数说明：RecoveryInfoProvider.__call__
    # 用途：合并检查点与 Trace 中的工具调用，区分已返回结果和结果未知的调用，为恢复审计
    # 生成有限摘要。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `tuple[list[str], list[str]]`；返回 `(completed, pending)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._trace_calls` →
    # `_completed_line` → `completed.insert` → `_pending_line`。
    async def __call__(self, run_id: str) -> tuple[list[str], list[str]]:
        checkpoint = await self._checkpoints.get(run_id)
        started, finished = await self._trace_calls(run_id)
        if checkpoint is not None:
            completed_results = list(checkpoint.completed_tool_results)
            pending_calls = list(checkpoint.pending_tool_calls)
        else:
            completed_results = list(finished.values())
            pending_calls = [call for call_id, call in started.items() if call_id not in finished]
        known = {result.tool_call_id for result in completed_results}
        known |= {call.id for call in pending_calls}
        # Trace 里开始了、检查点却没记下的调用（例如检查点写入前退出）也算“结果未知”
        pending_calls += [call for call_id, call in started.items() if call_id not in known]

        completed = [
            _completed_line(result, started.get(result.tool_call_id))
            for result in completed_results[-_MAX_CALLS:]
        ]
        if len(completed_results) > _MAX_CALLS:
            completed.insert(0, f"- （更早的 {len(completed_results) - _MAX_CALLS} 次调用省略）")
        pending = [_pending_line(call) for call in pending_calls[:_MAX_CALLS]]
        return completed, pending

    # 函数说明：RecoveryInfoProvider._trace_calls
    # 用途：从工具开始和完成事件重建调用与结果映射；轨迹缺失或读取失败时返回空映射作为恢
    # 复降级。
    # 参数：
    #   run_id：目标运行标识，类型 `str`。
    # 返回：类型 `tuple[dict[str, ToolCall], dict[str, ToolResult]]`；返回
    # `(started, finished)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._traces.load_events` →
    # `logger.exception`。
    # 分支与异常：
    #   当 `self._traces is None` 时，返回 `(started, finished)`。
    #   捕获 `KeyError` 后，返回 `(started, finished)`。
    #   捕获 `Exception` 后，返回 `(started, finished)`。
    async def _trace_calls(
        self, run_id: str
    ) -> tuple[dict[str, ToolCall], dict[str, ToolResult]]:
        started: dict[str, ToolCall] = {}
        finished: dict[str, ToolResult] = {}
        if self._traces is None:
            return started, finished
        try:
            events = await self._traces.load_events(run_id)
        except KeyError:
            return started, finished
        except Exception:
            logger.exception("failed to load trace for %s", run_id)
            return started, finished
        for event in events:
            if event.type is AgentEventType.TOOL_STARTED and event.tool_call is not None:
                started[event.tool_call.id] = event.tool_call
            elif event.type is AgentEventType.TOOL_COMPLETED and event.tool_result is not None:
                finished[event.tool_result.tool_call_id] = event.tool_result
        return started, finished


# 函数说明：_arguments
# 用途：在规划、执行、审计协作中处理 `_arguments`，通过 `json.dumps` 完成首个内部处理步
# 骤。
# 参数：
#   call：调用输入或配置值，类型 `ToolCall | None`。
# 返回：类型 `str`；按分支返回 `'(参数未记录)'`；
# `text if len(text) <= _ARGUMENT_CHARS else text[:_ARGUMENT_CHARS] + '…'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
# 分支与异常：
#   当 `call is None` 时，返回 `'(参数未记录)'`。
def _arguments(call: ToolCall | None) -> str:
    if call is None:
        return "(参数未记录)"
    raw = call.arguments
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    return text if len(text) <= _ARGUMENT_CHARS else text[:_ARGUMENT_CHARS] + "…"


# 函数说明：_excerpt
# 用途：在规划、执行、审计协作中处理 `_excerpt`，通过 `' '.join` 完成首个内部处理步骤。
# 参数：
#   text：待处理的文本，类型 `str | None`。
# 返回：类型 `str`；返回
# `compact if len(compact) <= _OUTPUT_CHARS else compact[:_OUTPUT_CHARS] + '…'`。
def _excerpt(text: str | None) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= _OUTPUT_CHARS else compact[:_OUTPUT_CHARS] + "…"


# 函数说明：_completed_line
# 用途：处理规划、执行、审计协作中的 `_completed_line` 数据；结果及边界条件见下方说明。
# 参数：
#   result：上一步计算或执行得到的结果，类型 `ToolResult`。
#   call：传给 `_arguments` 的输入，类型 `ToolCall | None`。
# 返回：类型 `str`；返回
# `f'- {result.tool_name} {_arguments(call)} → {outcome}{evidence}'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_excerpt` → `_arguments`。
def _completed_line(result: ToolResult, call: ToolCall | None) -> str:
    outcome = "成功" if result.success else f"失败: {_excerpt(result.error)}"
    evidence = f"（evidence: {result.evidence_id}）" if result.evidence_id else ""
    return f"- {result.tool_name} {_arguments(call)} → {outcome}{evidence}"


# 函数说明：_pending_line
# 用途：返回 `f'- {call.name} {_arguments(call)}'`，提供 规划、执行、审计协作 的派生值。
# 参数：
#   call：传给 `_arguments` 的输入，类型 `ToolCall`。
# 返回：类型 `str`；返回 `f'- {call.name} {_arguments(call)}'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_arguments`。
def _pending_line(call: ToolCall) -> str:
    return f"- {call.name} {_arguments(call)}"


__all__ = ["RecoveryInfoProvider"]
