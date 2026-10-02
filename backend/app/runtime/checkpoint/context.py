
from __future__ import annotations

import json

from app.models.types import Message, MessageRole

from .models import RunCheckpoint

CHECKPOINT_CONTEXT_MESSAGE_NAME = "muharness_interrupted_run"
_MAX_ARGUMENT_CHARS = 4_000


# 函数说明：render_checkpoint_context
# 用途：生成展示文本检查点上下文，供运行检查点持久化使用。
# 参数：
#   checkpoint：检查点输入或配置值，类型 `RunCheckpoint`。
# 返回：类型 `Message`；返回 `Message(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_compact_arguments` →
# `checkpoint.updated_at.isoformat` → `json.dumps` → `Message`。
def render_checkpoint_context(checkpoint: RunCheckpoint) -> Message:

    payload = {
        "run_id": checkpoint.run_id,
        "original_user_message": checkpoint.user_message.content,
        "phase": checkpoint.phase.value,
        "step": checkpoint.step,
        "pending_tool_calls": [
            {
                "id": call.id,
                "name": call.name,
                "arguments": _compact_arguments(call.arguments),
                "recovery_semantics": (
                    "execution outcome is uncertain; verify before retry"
                ),
            }
            for call in checkpoint.pending_tool_calls
        ],
        "completed_tool_results": [
            {
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "success": result.success,
                "error": result.error,
            }
            for result in checkpoint.completed_tool_results
        ],
        "error": checkpoint.error,
        "updated_at": checkpoint.updated_at.isoformat(),
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return Message(
        role=MessageRole.SYSTEM,
        name=CHECKPOINT_CONTEXT_MESSAGE_NAME,
        content=(
            "恢复依据：上次 Run 在终态前中断，下方是持久化检查点，不是新的执行计划。"
            "pending 表示结果不确定，既不证明失败，也不证明未执行。先核对 Trace、"
            "工具证据和实际环境，再决定补记已确认进度、继续工作或询问用户。"
            "有副作用的未决操作禁止直接重试；证据不足时保留未知，不伪造完成记录。\n"
            f"<interrupted_run>{serialized}</interrupted_run>"
        ),
    )


# 函数说明：_compact_arguments
# 用途：压缩调用参数，供运行检查点持久化使用。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `object`。
# 返回：类型 `object`；按分支返回 `arguments`；
# `serialized[:_MAX_ARGUMENT_CHARS] + '…<truncated>'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
# 分支与异常：
#   当 `len(serialized) <= _MAX_ARGUMENT_CHARS` 时，返回 `arguments`。
def _compact_arguments(arguments: object) -> object:

    serialized = json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    if len(serialized) <= _MAX_ARGUMENT_CHARS:
        return arguments
    return serialized[:_MAX_ARGUMENT_CHARS] + "…<truncated>"


__all__ = ["CHECKPOINT_CONTEXT_MESSAGE_NAME", "render_checkpoint_context"]
