"""按片段修改工作区文件：只发送要替换的旧文本和新文本，不重发整个文件。

修改已有文件时用 write_file 必须重发全文，几百行的文件一次就要输出几万字符，
很容易连同思考一起撞上单次输出上限、工具参数被截断。edit_file 只携带改动片段。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.models.types import ToolDefinition

from ..base import BaseTool
from ..role_boundary import READ_ONLY_MODES
from ._workspace import resolve_workspace_path, workspace_root_path

if TYPE_CHECKING:
    from ..hooks import ToolExecutionContext


class EditFileTool(BaseTool):
    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._workspace_root = workspace_root_path(workspace_root)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="edit_file",
            description=(
                "按精确文本替换修改工作区里已有的 UTF-8 文件，只提交要改的片段，"
                "不重发整个文件。修改已有文件时优先使用它，而不是用 write_file 重写全文。"
                "先用 read_file 看到原文，old_string 必须与文件内容逐字一致（含缩进），"
                "且在文件中唯一；不唯一时加上相邻几行使其唯一，或用 replace_all 替换全部。"
                "通常 old_string 取 2～6 行即可，不要把大段原文放进去。"
                "可多次调用逐步修改或在文件末尾追加：以文件最后几行作为 old_string，"
                "new_string 写成这几行加上新内容。回执只证明替换已写入，不证明内容正确。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "已有文件的工作区相对路径；文件不存在时先用 write_file 创建。",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "要被替换的原文片段，必须与文件内容逐字一致且非空。",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "替换后的文本；空字符串表示删除 old_string。",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "default": False,
                        "description": "为 true 时替换所有出现位置；默认 false，要求 old_string 唯一。",
                    },
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            },
            strict=False,
            closing_allowed=True,
        )

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if context.mode in READ_ONLY_MODES:
            mode = context.mode.value
            raise PermissionError(
                f"edit_file is not allowed in {mode} mode (read-only role)"
            )
        return await self.execute(arguments)

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative_path = arguments.get("path")
        old_string = arguments.get("old_string")
        new_string = arguments.get("new_string")
        replace_all = arguments.get("replace_all", False)
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("'path' must be a non-empty string")
        if not isinstance(old_string, str) or not old_string:
            raise ValueError("'old_string' must be a non-empty string")
        if not isinstance(new_string, str):
            raise ValueError("'new_string' must be a string")
        if not isinstance(replace_all, bool):
            raise ValueError("'replace_all' must be a boolean")
        if old_string == new_string:
            raise ValueError(
                "'old_string' and 'new_string' are identical; nothing to change"
            )

        target = resolve_workspace_path(self._workspace_root, relative_path)
        replacements, characters = await asyncio.to_thread(
            _edit_utf8_file, target, old_string, new_string, replace_all
        )
        return {
            "path": target.relative_to(self._workspace_root).as_posix(),
            "replacements": replacements,
            "characters": characters,
        }


def _edit_utf8_file(
    path: Path, old_string: str, new_string: str, replace_all: bool
) -> tuple[int, int]:
    if not path.is_file():
        raise ValueError("file does not exist; create it with write_file first")
    with path.open("r", encoding="utf-8", newline="") as handle:
        original = handle.read()

    count = original.count(old_string)
    if count == 0 and "\r\n" in original and "\r\n" not in old_string:
        # 文件是 CRLF 换行而模型按 LF 给出片段时，按文件的换行风格匹配和写回
        crlf_old = old_string.replace("\n", "\r\n")
        if original.count(crlf_old):
            old_string = crlf_old
            new_string = new_string.replace("\r\n", "\n").replace("\n", "\r\n")
            count = original.count(old_string)
    if count == 0:
        raise ValueError(
            "old_string was not found in the file; read_file the current content "
            "and copy the exact text (including indentation)"
        )
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string occurs {count} times; add surrounding lines to make it "
            "unique, or set replace_all=true"
        )

    updated = (
        original.replace(old_string, new_string)
        if replace_all
        else original.replace(old_string, new_string, 1)
    )
    temporary = path.with_name(f".{path.name}.edit.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(updated)
    temporary.replace(path)
    return (count if replace_all else 1), len(updated)
