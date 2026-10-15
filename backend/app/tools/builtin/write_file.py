
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


class WriteFileTool(BaseTool):
    # 函数说明：WriteFileTool.__init__
    # 用途：初始化 WriteFileTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path | None`；默认
    # `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace_root_path`。
    # 副作用与资源：
    #   更新对象字段：`self._workspace_root`。
    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._workspace_root = workspace_root_path(workspace_root)

    # 函数说明：WriteFileTool.definition
    # 用途：提供 WriteFileTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description=(
                "把完整 UTF-8 文本写入工作区文件；不存在的父目录会创建，已有文件会整体覆盖。"
                "用于新建文件或确需整体重写时；修改已有文件优先用 edit_file 只提交改动片段。"
                "新文件首次写入控制在约 300 行内，其余部分之后用 edit_file 逐步补全，"
                "避免一次输出过长被截断。整体重写已有文件时先用 read_file 确认内容，"
                "不要把局部片段当完整内容提交。它不追加、不执行、不验证文件，也不用于读取"
                "或绕过只读角色限制。成功回执中的 path 和 characters 只证明文本已写入，"
                "不证明内容正确或任务完成；需要检查时再读取或运行相关验证，需要作为正式"
                "交付物时再用 artifact_publish 发布。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目标文件的工作区相对路径；不接受绝对路径、工作区根目录或越界路径，已有文件将被覆盖。",
                    },
                    "content": {
                        "type": "string",
                        "description": "文件写入后的完整文本，空字符串会清空文件；不是追加片段或补丁，保留需要保留的原有内容。",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            strict=True,
            closing_allowed=True,
        )

    # 函数说明：WriteFileTool.execute_with_context
    # 用途：执行上下文，供内置工作区工具使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；返回 `await self.execute(arguments)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.execute`。
    # 分支与异常：
    #   当 `context.mode in READ_ONLY_MODES` 时，抛出 `PermissionError(…)`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if context.mode in READ_ONLY_MODES:
            raise PermissionError(
                f"write_file is not allowed in {context.mode.value} mode (read-only role)"
            )
        return await self.execute(arguments)

    # 函数说明：WriteFileTool.execute
    # 用途：执行WriteFileTool，供内置工作区工具使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `path`
    # 、`content`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `path`、`characters`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`resolve_workspace_path` →
    # `asyncio.to_thread` → `target.relative_to(self._workspace_root).as_posix` →
    # `target.relative_to`。
    # 分支与异常：
    #   当 `not isinstance(relative_path, str) or not relative_path` 时，抛出
    # `ValueError("'path' must be a non-empty string")`。
    #   当 `not isinstance(content, str)` 时，抛出
    # `ValueError("'content' must be a string")`。
    # 副作用与资源：
    #   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative_path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("'path' must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("'content' must be a string")

        target = resolve_workspace_path(self._workspace_root, relative_path)
        await asyncio.to_thread(_write_utf8_file, target, content)
        return {
            "path": target.relative_to(self._workspace_root).as_posix(),
            "characters": len(content),
        }


# 函数说明：_write_utf8_file
# 用途：写入文件，供内置工作区工具使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
#   content：内容正文，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.parent.mkdir` →
# `path.write_text`。
# 副作用与资源：
#   文件或资源访问：`path.parent.mkdir`、`path.write_text`。
def _write_utf8_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
