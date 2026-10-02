"""Freeze a tool result's model-visible representation without changing history."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.types import Message, MessageRole


class ToolResultView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sequence: int = Field(ge=0)
    raw_message_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    representation: Literal["full", "excerpt"]
    content: str | None = None

    # 函数说明：ToolResultView.validate_content
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `ToolResultView`；返回 `self`。
    # 分支与异常：
    #   当 `self.representation == 'excerpt' and self.content is None` 时，抛出
    # `ValueError(…)`。
    #   当 `self.representation == 'full' and self.content is not None` 时，抛出
    # `ValueError(…)`。
    @model_validator(mode="after")
    def validate_content(self) -> ToolResultView:
        if self.representation == "excerpt" and self.content is None:
            raise ValueError("an excerpt must contain its exact model-visible content")
        if self.representation == "full" and self.content is not None:
            raise ValueError("full views use the raw message and must not duplicate it")
        return self


# 函数说明：raw_message_sha256
# 用途：在模型上下文与输入预算中处理 `raw_message_sha256`，通过 `json.dumps` 完成首个内
# 部处理步骤。
# 参数：
#   message：单条消息或通知，类型 `Message`。
# 返回：类型 `str`；返回 `hashlib.sha256(canonical.encode('utf-8')).hexdigest()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps` →
# `hashlib.sha256(canonical.encode('utf-8')).hexdigest` → `hashlib.sha256` →
# `canonical.encode`。
def raw_message_sha256(message: Message) -> str:
    """Hash the canonical raw message, not provider/prompt transformations."""
    canonical = json.dumps(
        message.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ToolResultViewState:
    # 函数说明：ToolResultViewState.__init__
    # 用途：初始化 ToolResultViewState；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   history：原始会话历史，类型 `Sequence[Message]`。
    #   views：`views`输入或配置值，类型 `Sequence[ToolResultView]`；默认 `()`。
    #   max_output_chars：工具输出的字符数上限，类型 `int`；默认 `8000`。
    #   head_chars：字符数量或字符预算，类型 `int`；默认 `4000`。
    #   tail_chars：字符数量或字符预算，类型 `int`；默认 `2000`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`raw_message_sha256` →
    # `self._views.setdefault` → `self._full_view`。
    # 分支与异常：
    #   当 `min(max_output_chars, head_chars, tail_chars) < 0` 时，抛出
    # `ValueError('tool view limits cannot be negative')`。
    #   当 `sequence >= len(history) or history[sequence].role is not…` 时，跳过当前循环
    # 项。
    #   当 `key in self._views and self._views[key] != view` 时，抛出 `ValueError(…)`。
    # 副作用与资源：
    #   更新对象字段：`self._max_output_chars`、`self._head_chars`、`self._tail_chars`、
    # `self._views`。
    def __init__(
        self,
        history: Sequence[Message],
        views: Sequence[ToolResultView] = (),
        *,
        max_output_chars: int = 8000,
        head_chars: int = 4000,
        tail_chars: int = 2000,
    ) -> None:
        if min(max_output_chars, head_chars, tail_chars) < 0:
            raise ValueError("tool view limits cannot be negative")
        self._max_output_chars = max_output_chars
        self._head_chars = head_chars
        self._tail_chars = tail_chars
        self._views: dict[tuple[int, str], ToolResultView] = {}
        for view in views:
            sequence = view.source_sequence
            if (
                sequence >= len(history)
                or history[sequence].role is not MessageRole.TOOL
                or raw_message_sha256(history[sequence]) != view.raw_message_sha256
            ):
                continue
            key = (view.source_sequence, view.raw_message_sha256)
            if key in self._views and self._views[key] != view:
                raise ValueError(
                    "conflicting tool result views for the same raw message"
                )
            self._views[key] = view
        # Existing histories may already have been sent. Never retrospectively
        # shorten a legacy result when no matching first-seen decision exists.
        for sequence, message in enumerate(history):
            if message.role is MessageRole.TOOL:
                key = (sequence, raw_message_sha256(message))
                self._views.setdefault(key, self._full_view(sequence, key[1]))

    # 函数说明：ToolResultViewState.project
    # 用途：在模型上下文与输入预算中处理 `project`，通过 `projected.append` 完成首个内部
    # 处理步骤。
    # 参数：
    #   raw_messages：传给 `enumerate` 的输入，类型 `Sequence[Message]`。
    # 返回：类型 `tuple[Message, ...]`；返回 `tuple(projected)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`raw_message_sha256` →
    # `self._excerpt` → `self._full_view` → `ToolResultView`。
    # 分支与异常：
    #   `message.role is not MessageRole.TOOL` 分支在完成前置处理后跳过当前循环项。
    def project(self, raw_messages: Sequence[Message]) -> tuple[Message, ...]:
        projected: list[Message] = []
        for sequence, message in enumerate(raw_messages):
            if message.role is not MessageRole.TOOL:
                projected.append(message)
                continue
            digest = raw_message_sha256(message)
            key = (sequence, digest)
            view = self._views.get(key)
            if view is None:
                content = self._excerpt(message.content)
                view = (
                    self._full_view(sequence, digest)
                    if content is None
                    else ToolResultView(
                        source_sequence=sequence,
                        raw_message_sha256=digest,
                        representation="excerpt",
                        content=content,
                    )
                )
                self._views[key] = view
            projected.append(
                message.model_copy(update={"content": view.content})
                if view.representation == "excerpt"
                else message
            )
        return tuple(projected)

    # 函数说明：ToolResultViewState.snapshot
    # 用途：在模型上下文与输入预算中处理 `snapshot`，通过 `self._views.get` 完成首个内部
    # 处理步骤。
    # 参数：
    #   raw_messages：传给 `enumerate` 的输入，类型 `Sequence[Message]`。
    # 返回：类型 `tuple[ToolResultView, ...]`；返回 `tuple(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`raw_message_sha256`。
    def snapshot(self, raw_messages: Sequence[Message]) -> tuple[ToolResultView, ...]:
        """Return only decisions matching the supplied canonical raw history."""
        return tuple(
            view
            for sequence, message in enumerate(raw_messages)
            if message.role is MessageRole.TOOL
            if (view := self._views.get((sequence, raw_message_sha256(message))))
            is not None
        )

    # 函数说明：ToolResultViewState._full_view
    # 用途：返回 `ToolResultView(source_sequence=sequence, raw_message_sha256=digest,
    # representation='full')`，提供 ToolResultViewState 的派生值。
    # 参数：
    #   sequence：事件或记录顺序号，类型 `int`。
    #   digest：`digest`输入或配置值，类型 `str`。
    # 返回：类型 `ToolResultView`；返回 `ToolResultView(source_sequence=sequence,
    # raw_message_sha256=digest, representation='full')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolResultView`。
    @staticmethod
    def _full_view(sequence: int, digest: str) -> ToolResultView:
        return ToolResultView(
            source_sequence=sequence,
            raw_message_sha256=digest,
            representation="full",
        )

    # 函数说明：ToolResultViewState._excerpt
    # 用途：在模型上下文与输入预算中处理 `_excerpt`，通过 `json.loads` 完成首个内部处理
    # 步骤。
    # 参数：
    #   content：内容正文，类型 `str | None`。
    # 返回：类型 `str | None`；按分支返回 `None`；
    # `json.dumps(envelope, ensure_ascii=False, separators=(',', ':'))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` → `json.dumps`。
    # 分支与异常：
    #   当 `content is None` 时，返回 `None`。
    #   捕获 `(ValueError, TypeError)` 后，返回 `None`。
    #   当 `not isinstance(envelope, dict)` 时，返回 `None`。
    #   当 `not isinstance(output, str) or len(output) <=…` 时，返回 `None`。
    def _excerpt(self, content: str | None) -> str | None:
        if content is None:
            return None
        try:
            envelope = json.loads(content)
        except (ValueError, TypeError):
            return None
        if not isinstance(envelope, dict):
            return None
        output = envelope.get("output")
        if not isinstance(output, str) or len(output) <= self._max_output_chars:
            return None
        retained = self._head_chars + self._tail_chars
        if retained >= len(output):
            return None
        omitted = len(output) - retained
        excerpt = (
            output[: self._head_chars]
            + f"\n\n[tool output excerpt: {omitted} characters "
            + "omitted from the middle]\n\n"
            + (output[-self._tail_chars :] if self._tail_chars else "")
        )
        if len(excerpt) >= len(output):
            return None
        envelope["output"] = excerpt
        return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))


__all__ = ["ToolResultView", "ToolResultViewState", "raw_message_sha256"]
