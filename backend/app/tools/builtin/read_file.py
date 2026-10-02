
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.models.types import ToolDefinition

from ..base import BaseTool
from ._workspace import resolve_workspace_path, workspace_root_path

DEFAULT_READ_LINES = 200
MAX_READ_LINES = 2_000
MAX_RANGE_CHARS = 12_000


class ReadFileTool(BaseTool):
    # 函数说明：ReadFileTool.__init__
    # 用途：初始化 ReadFileTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path | None`；默认
    # `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace_root_path`。
    # 副作用与资源：
    #   更新对象字段：`self._workspace_root`。
    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._workspace_root = workspace_root_path(workspace_root)

    # 函数说明：ReadFileTool.definition
    # 用途：提供 ReadFileTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description=(
                "读工作区 UTF-8 文本，核实内容或修改前用。未知路径先 list_files；"
                "大文件用 start_line/max_lines 分段读取，返回行范围和下一行；"
                "分段文本最多 12000 字符，line_truncated 表示长行只返回了开头。"
                "next_line 跳到下一完整行，不能取回被省略的长行尾部。"
                "只填 path 保留完整读取。不处理二进制或执行。"
                "成功仅取得文本，非验收通过。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "工作区相对文件路径。",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "从第几行开始，行号从 1 起，默认 1。",
                    },
                    "max_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_READ_LINES,
                        "description": "最多返回的行数，分段读取默认 200，最多 2000。",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：ReadFileTool.execute
    # 用途：在工作区边界内读取 UTF-8 文件；仅传 path 时完整读取，指定行范围时返回有限文
    # 本及后续行信息。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `path`
    # 、`start_line`、`max_lines`。
    # 返回：类型 `str | dict[str, Any]`；按分支返回
    # `await asyncio.to_thread(_read_utf8_file, target)`；`await asyncio.to_thread(
    # _read_utf8_range, target, relative_path, start_line, max_lines)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`resolve_workspace_path` →
    # `asyncio.to_thread`。
    # 分支与异常：
    #   当 `not isinstance(relative_path, str) or not relative_path` 时，抛出
    # `ValueError("'path' must be a non-empty string")`。
    #   当 `'start_line' not in arguments and 'max_lines' not in…` 时，返回
    # `await asyncio.to_thread(_read_utf8_file, target)`。
    #   当 `type(start_line) is not int or start_line < 1` 时，抛出
    # `ValueError("'start_line' must be a positive integer")`。
    #   当 `type(max_lines) is not int or not 1 <= max_lines <=…` 时，抛出
    # `ValueError(…)`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def execute(self, arguments: dict[str, Any]) -> str | dict[str, Any]:
        relative_path = arguments.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("'path' must be a non-empty string")
        target = resolve_workspace_path(self._workspace_root, relative_path)
        if "start_line" not in arguments and "max_lines" not in arguments:
            return await asyncio.to_thread(_read_utf8_file, target)
        start_line = arguments.get("start_line", 1)
        max_lines = arguments.get("max_lines", DEFAULT_READ_LINES)
        if type(start_line) is not int or start_line < 1:
            raise ValueError("'start_line' must be a positive integer")
        if type(max_lines) is not int or not 1 <= max_lines <= MAX_READ_LINES:
            raise ValueError(
                f"'max_lines' must be an integer from 1 to {MAX_READ_LINES}"
            )
        return await asyncio.to_thread(
            _read_utf8_range, target, relative_path, start_line, max_lines,
        )


# 函数说明：_read_utf8_file
# 用途：读取文件，供内置工作区工具使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
# 返回：类型 `str`；返回 `path.read_text(encoding='utf-8')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.is_file` → `path.read_text`。
# 分支与异常：
#   当 `not path.is_file()` 时，抛出
# `FileNotFoundError(f'file does not exist: {path.name}')`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
def _read_utf8_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"file does not exist: {path.name}")
    return path.read_text(encoding="utf-8")


# 函数说明：_read_utf8_range
# 用途：按行数和字符上限读取文件，排空超长行尾部以保持真实行号，并标明截断与下一完整行。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
#   relative_path：相对于工作区的路径，类型 `str`。
#   start_line：读取起始行号，从 1 起，类型 `int`。
#   max_lines：读取的最大行数，类型 `int`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `path`、`start_line`、`end_line`、
# `content`、`next_line`、`truncated`、`line_truncated`。
# 设计约束：next_line 指向下一完整行，不能重新获取已省略的超长行尾部；line_truncated 单
# 独记录长行截断。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.is_file` → `path.open` →
# `stream.readline` → `line.endswith` → `tail.endswith` → `stream.read`。
# 分支与异常：
#   当 `not path.is_file()` 时，抛出
# `FileNotFoundError(f'file does not exist: {path.name}')`。
#   当 `not line` 时，结束当前循环。
#   当 `line_number < start_line` 时，跳过当前循环项。
# 副作用与资源：
#   文件或资源访问：`path.open`。
def _read_utf8_range(
    path: Path, relative_path: str, start_line: int, max_lines: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"file does not exist: {path.name}")
    parts: list[str] = []
    remaining = MAX_RANGE_CHARS
    line_number = 0
    truncated_line: int | None = None
    with path.open(encoding="utf-8") as stream:
        while len(parts) < max_lines and remaining > 0:
            line = stream.readline(MAX_RANGE_CHARS + 1)
            if not line:
                break
            line_number += 1
            long_line = len(line) > MAX_RANGE_CHARS and not line.endswith("\n")
            if long_line:
                # Drain oversized lines in bounded chunks before advancing a line.
                tail = line
                while tail and not tail.endswith("\n"):
                    tail = stream.readline(MAX_RANGE_CHARS + 1)
            if line_number < start_line:
                continue
            if long_line or len(line) > remaining:
                truncated_line = line_number
            parts.append(line[:remaining])
            remaining -= len(parts[-1])
        has_more = bool(stream.read(1))
    return {
        "path": relative_path,
        "start_line": start_line,
        "end_line": line_number if parts else None,
        "content": "".join(parts),
        "next_line": line_number + 1 if has_more else None,
        "truncated": has_more or truncated_line is not None,
        "line_truncated": truncated_line,
    }
