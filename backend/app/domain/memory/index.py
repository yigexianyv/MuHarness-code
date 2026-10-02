
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from .models import MemoryRecord

logger = logging.getLogger("muharness.memory.index")

_INDEX_HEADER = (
    "# Long-term Memory Index\n\n"
    "The following long-term memories are available.\n"
    "These cues are discovery metadata, not authoritative memory content.\n"
    "When a memory may materially help the current task, use memory_read "
    "before relying on it in an answer, decision, or action.\n"
)


class MemoryIndex:

    # 函数说明：MemoryIndex.__init__
    # 用途：初始化 MemoryIndex；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   memory_dir：记忆文件目录，类型 `str | Path`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Path`。
    # 副作用与资源：
    #   更新对象字段：`self.path`。
    def __init__(self, memory_dir: str | Path) -> None:
        self.path = Path(memory_dir) / "INDEX.md"

    # 函数说明：MemoryIndex.render
    # 用途：生成展示文本MemoryIndex，供长期记忆管理与检索使用。
    # 参数：
    #   memories：记忆集合输入或配置值，类型 `Sequence[MemoryRecord]`。
    # 返回：类型 `str`；按分支返回 `_INDEX_HEADER + '\n(No long-term memories yet.)\n'`
    # ；`'\n'.join(lines).rstrip() + '\n'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_single_line` →
    # `'\n'.join(lines).rstrip`。
    # 分支与异常：
    #   当 `not memories` 时，返回 `_INDEX_HEADER + '\n(No long-term memories yet.)\n'`
    # 。
    def render(self, memories: Sequence[MemoryRecord]) -> str:

        if not memories:
            return _INDEX_HEADER + "\n(No long-term memories yet.)\n"
        lines = [_INDEX_HEADER]
        for record in memories:
            cue = _single_line(record.summary) or record.title
            lines.append(f"[{record.id}] {record.title}")
            lines.append(f"Cue: {cue}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    # 函数说明：MemoryIndex.rebuild
    # 用途：重建MemoryIndex，供长期记忆管理与检索使用。
    # 参数：
    #   memories：传给 `self.render` 的输入，类型 `Sequence[MemoryRecord]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.render` →
    # `asyncio.to_thread`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def rebuild(self, memories: Sequence[MemoryRecord]) -> None:

        content = self.render(memories)
        await asyncio.to_thread(self._write_atomic, content)

    # 函数说明：MemoryIndex.load
    # 用途：加载MemoryIndex，供长期记忆管理与检索使用。
    # 返回：类型 `str | None`；按分支返回 `None`；
    # `await asyncio.to_thread(self.path.read_text, encoding='utf-8')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.to_thread`。
    # 分支与异常：
    #   当 `not await asyncio.to_thread(self.path.is_file)` 时，返回 `None`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def load(self) -> str | None:

        if not await asyncio.to_thread(self.path.is_file):
            return None
        return await asyncio.to_thread(self.path.read_text, encoding="utf-8")

    # 函数说明：MemoryIndex._write_atomic
    # 用途：写入`atomic`，供长期记忆管理与检索使用。
    # 参数：
    #   content：内容正文，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.path.parent.mkdir` →
    # `self.path.with_name` → `os.getpid` → `temporary.write_text` → `os.replace`。
    # 副作用与资源：
    #   文件或资源访问：`self.path.parent.mkdir`、`temporary.write_text`、`os.replace`。
    def _write_atomic(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, self.path)


# 函数说明：_single_line
# 用途：返回 `' '.join(text.split()).strip()`，提供 长期记忆管理与检索 的派生值。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `str`；返回 `' '.join(text.split()).strip()`。
def _single_line(text: str) -> str:
    return " ".join(text.split()).strip()


__all__ = ["MemoryIndex"]
