
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.models.types import ToolDefinition

from ..base import BaseTool
from ._workspace import resolve_workspace_path, workspace_root_path

MAX_LISTED_FILES = 200


class ListFilesTool(BaseTool):
    # 函数说明：ListFilesTool.__init__
    # 用途：初始化 ListFilesTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path | None`；默认
    # `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace_root_path`。
    # 副作用与资源：
    #   更新对象字段：`self._workspace_root`。
    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._workspace_root = workspace_root_path(workspace_root)

    # 函数说明：ListFilesTool.definition
    # 用途：提供 ListFilesTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_files",
            description=(
                "递归列出工作区指定目录下的普通文件，返回工作区相对路径，不读取内容，"
                "跳过符号链接。尚不清楚目标文件路径或需要查看目录内文件时使用；"
                "优先指定相关子目录，已有确切路径且需要内容时直接用 read_file，"
                "不要反复列举同一范围。最多返回 200 个文件，truncated 为真时列表不完整，"
                "count 是本次返回数量。成功只证明列出的路径被发现，不证明文件内容、"
                "修改结果或任务完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "工作区内已存在目录的相对路径，默认 .；尽量缩小到相关子目录，"
                            "不接受绝对路径或越出工作区的路径。"
                        ),
                        "default": ".",
                    }
                },
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：ListFilesTool.execute
    # 用途：执行ListFilesTool，供内置工作区工具使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `directory`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `directory`、`files`、`count`、
    # `truncated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`resolve_workspace_path` →
    # `asyncio.to_thread` → `target.relative_to(self._workspace_root).as_posix` →
    # `target.relative_to`。
    # 分支与异常：
    #   当 `not isinstance(directory, str)` 时，抛出
    # `ValueError("'directory' must be a string")`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        directory = arguments.get("directory", ".")
        if not isinstance(directory, str):
            raise ValueError("'directory' must be a string")

        target = resolve_workspace_path(
            self._workspace_root,
            directory,
            allow_root=True,
        )
        files = await asyncio.to_thread(
            _list_workspace_files,
            self._workspace_root,
            target,
        )
        return {
            "directory": (target.relative_to(self._workspace_root).as_posix() or "."),
            "files": files[:MAX_LISTED_FILES],
            "count": min(len(files), MAX_LISTED_FILES),
            "truncated": len(files) > MAX_LISTED_FILES,
        }


# 函数说明：_list_workspace_files
# 用途：列出工作区文件集合，供内置工作区工具使用。
# 参数：
#   workspace_root：文件工具允许访问的工作区根目录，类型 `Path`。
#   directory：目录输入或配置值，类型 `Path`。
# 返回：类型 `list[str]`；返回 `sorted(files)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`directory.is_dir` → `directory.rglob`
#  → `entry.is_symlink` → `entry.is_file` → `entry.resolve` → `resolved.relative_to`；另
# 有 1 个调用点。
# 分支与异常：
#   当 `not directory.is_dir()` 时，抛出
# `ValueError('directory does not exist or is not a directory')`。
#   当 `entry.is_symlink() or not entry.is_file()` 时，跳过当前循环项。
#   捕获 `ValueError` 后，跳过当前循环项，继续处理后续项。
def _list_workspace_files(workspace_root: Path, directory: Path) -> list[str]:
    if not directory.is_dir():
        raise ValueError("directory does not exist or is not a directory")

    files: list[str] = []
    for entry in directory.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        resolved = entry.resolve()
        try:
            relative = resolved.relative_to(workspace_root)
        except ValueError:
            continue
        files.append(relative.as_posix())
    return sorted(files)
