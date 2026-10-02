
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from app.models.types import Message, MessageRole

from .prompts import MEMORY_RECALL_HEADER
from .search_index import MemorySearchResult, SearchMode

if TYPE_CHECKING:
    from .manager import MemoryManager

MEMORY_RECALL_MESSAGE_NAME = "muharness_memory_recall"


@dataclass(frozen=True, slots=True)
class MemoryRecallQueryInputs:

    user_message: str
    recent_user_messages: tuple[str, ...] = ()
    summary_objective: str | None = None
    task_title: str | None = None
    task_active_steps: tuple[str, ...] = ()

    # 函数说明：MemoryRecallQueryInputs.with_task
    # 用途：返回
    # `replace(self, task_title=task_title, task_active_steps=task_active_steps)`，提供
    # MemoryRecallQueryInputs 的派生值。
    # 参数：
    #   task_title：任务输入或配置值，类型 `str | None`。
    #   task_active_steps：任务活跃项步骤集合输入或配置值，类型 `tuple[str, ...]`。
    # 返回：类型 `MemoryRecallQueryInputs`；返回
    # `replace(self, task_title=task_title, task_active_steps=task_active_steps)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`replace`。
    def with_task(
        self,
        task_title: str | None,
        task_active_steps: tuple[str, ...],
    ) -> MemoryRecallQueryInputs:

        return replace(self, task_title=task_title, task_active_steps=task_active_steps)

    # 函数说明：MemoryRecallQueryInputs.render
    # 用途：生成展示文本MemoryRecallQueryInputs，供长期记忆管理与检索使用。
    # 参数：
    #   max_chars：保留的字符数上限，类型 `int`；默认 `1600`。
    # 返回：类型 `str`；返回 `query[:max_chars]`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`filter`。
    def render(self, *, max_chars: int = 1_600) -> str:

        sections = [
            self.user_message.strip(),
            " | ".join(
                text.strip() for text in self.recent_user_messages if text.strip()
            ),
            (self.summary_objective or "").strip(),
        ]
        if self.task_title:
            task_line = self.task_title.strip()
            if self.task_active_steps:
                steps = "; ".join(
                    step.strip() for step in self.task_active_steps if step.strip()
                )
                task_line = f"{task_line} > {steps}"
            sections.append(task_line)
        query = "\n".join(filter(None, sections))
        return query[:max_chars]


# 函数说明：recent_user_message_texts
# 用途：在长期记忆管理与检索中处理 `recent_user_message_texts`，通过
# `(message.content or '').strip` 完成首个内部处理步骤。
# 参数：
#   history：原始会话历史，类型 `Sequence[Message]`。
#   limit：本次返回或处理的数量上限，类型 `int`；默认 `3`。
#   max_chars：保留的字符数上限，类型 `int`；默认 `300`。
# 返回：类型 `tuple[str, ...]`；返回 `tuple(reversed(texts))`。
# 分支与异常：
#   当 `message.role is not MessageRole.USER` 时，跳过当前循环项。
#   当 `len(texts) >= limit` 时，结束当前循环。
def recent_user_message_texts(
    history: Sequence[Message],
    *,
    limit: int = 3,
    max_chars: int = 300,
) -> tuple[str, ...]:

    texts: list[str] = []
    for message in reversed(history):
        if message.role is not MessageRole.USER:
            continue
        content = (message.content or "").strip()
        if content:
            texts.append(content[:max_chars])
        if len(texts) >= limit:
            break
    return tuple(reversed(texts))


@dataclass(frozen=True, slots=True)
class MemoryRecallCandidate:

    memory_id: str
    title: str
    summary: str
    revision: int
    snippet: str
    rrf_score: float
    matched_by_vector: bool
    matched_by_fts: bool


@dataclass(frozen=True, slots=True)
class MemoryRecallSnapshot:

    query: str
    mode: SearchMode
    candidates: tuple[MemoryRecallCandidate, ...]
    degrade_reason: str | None = None

    # 函数说明：MemoryRecallSnapshot.render_message
    # 用途：生成展示文本消息，供长期记忆管理与检索使用。
    # 参数：
    #   max_chars：保留的字符数上限，类型 `int`；默认 `2400`。
    # 返回：类型 `Message | None`；按分支返回 `None`；`Message(role=MessageRole.SYSTEM,
    # name=MEMORY_RECALL_MESSAGE_NAME, content=content)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`MEMORY_RECALL_HEADER.rstrip` →
    # `'\n\n'.join([header, *entries]).rstrip` → `Message`。
    # 分支与异常：
    #   当 `not self.candidates` 时，返回 `None`。
    #   当 `entry_cost > remaining` 时，结束当前循环。
    #   当 `not entries` 时，返回 `None`。
    def render_message(self, *, max_chars: int = 2_400) -> Message | None:

        if not self.candidates:
            return None
        header = MEMORY_RECALL_HEADER.rstrip()
        entries: list[str] = []
        # 最后一个换行也属于预算，每条记录前还有两个分隔换行。
        remaining = max_chars - len(header) - 1
        for candidate in self.candidates:
            entry_lines = [
                f"[{candidate.memory_id}] {candidate.title} "
                f"(revision {candidate.revision})",
                f"Summary: {candidate.summary}",
            ]
            if candidate.snippet:
                entry_lines.append(f"Snippet: {candidate.snippet}")
            entry = "\n".join(entry_lines)
            entry_cost = len(entry) + 2
            if entry_cost > remaining:
                break
            entries.append(entry)
            remaining -= entry_cost
        if not entries:
            return None
        content = "\n\n".join([header, *entries]).rstrip() + "\n"
        return Message(
            role=MessageRole.SYSTEM,
            name=MEMORY_RECALL_MESSAGE_NAME,
            content=content,
        )


class MemoryRecallService:

    # 函数说明：MemoryRecallService.__init__
    # 用途：初始化 MemoryRecallService；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MemoryManager`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`。
    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # 函数说明：MemoryRecallService.recall
    # 用途：组合检索线索并挑选可注入当前上下文的记忆。
    # 参数：
    #   inputs：`inputs`输入或配置值，类型 `MemoryRecallQueryInputs`。
    # 返回：类型 `MemoryRecallSnapshot`；返回 `MemoryRecallSnapshot(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`inputs.render` →
    # `MemoryRecallSnapshot` → `self._manager.search` → `MemoryRecallCandidate`。
    # 分支与异常：
    #   当 `not query` 时，返回 `MemoryRecallSnapshot(…)`。
    #   捕获 `Exception` 后，返回 `MemoryRecallSnapshot(…)`。
    async def recall(
        self,
        inputs: MemoryRecallQueryInputs,
    ) -> MemoryRecallSnapshot:

        """组合检索线索并挑选可注入当前上下文的记忆。"""
        query = inputs.render(
            max_chars=self._manager.search_settings.query_max_chars
        )
        if not query:
            return MemoryRecallSnapshot(
                query="",
                mode=SearchMode.UNAVAILABLE,
                candidates=(),
                degrade_reason="empty recall query",
            )
        try:
            result: MemorySearchResult = await self._manager.search(query)
        except Exception as exc:
            return MemoryRecallSnapshot(
                query=query,
                mode=SearchMode.UNAVAILABLE,
                candidates=(),
                degrade_reason=f"recall failed: {type(exc).__name__}: {exc}",
            )
        candidates = tuple(
            MemoryRecallCandidate(
                memory_id=item.memory_id,
                title=item.title,
                summary=item.summary,
                revision=item.revision,
                snippet=item.snippet,
                rrf_score=item.rrf_score,
                matched_by_vector=item.matched_by_vector,
                matched_by_fts=item.matched_by_fts,
            )
            for item in result.candidates
        )
        return MemoryRecallSnapshot(
            query=query,
            mode=result.mode,
            candidates=candidates,
            degrade_reason=result.degrade_reason,
        )


__all__ = [
    "MEMORY_RECALL_MESSAGE_NAME",
    "MemoryRecallCandidate",
    "MemoryRecallQueryInputs",
    "MemoryRecallService",
    "MemoryRecallSnapshot",
    "recent_user_message_texts",
]
