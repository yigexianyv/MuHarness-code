
from pathlib import Path

from app.safety.sandbox import SandboxSupervisor

from ..registry import ToolRegistry
from ..search import SearchSettings
from .current_time import CurrentTimeTool
from .edit_file import EditFileTool
from .http_request import HttpRequestTool
from .list_files import ListFilesTool
from .read_file import ReadFileTool
from .shell import ShellCommandTool
from .web_search import WebSearchTool
from .write_file import WriteFileTool


# 函数说明：build_builtin_tool_registry
# 用途：构建工具，供内置工作区工具使用。
# 参数：
#   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path | None`；默认
# `None`。
#   search_settings：检索设置输入或配置值，类型 `SearchSettings | None`；默认 `None`。
#   sandbox_supervisor：`sandbox_supervisor`输入或配置值，类型
# `SandboxSupervisor | None`；默认 `None`。
# 返回：类型 `ToolRegistry`；返回 `registry`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolRegistry` → `registry.register` →
#  `CurrentTimeTool` → `ListFilesTool` → `ReadFileTool` → `WriteFileTool`；另有 3 个调用
# 点。
def build_builtin_tool_registry(
    workspace_root: str | Path | None = None,
    *,
    search_settings: SearchSettings | None = None,
    sandbox_supervisor: SandboxSupervisor | None = None,
) -> ToolRegistry:

    registry = ToolRegistry()
    registry.register(CurrentTimeTool())
    registry.register(ListFilesTool(workspace_root))
    registry.register(ReadFileTool(workspace_root))
    registry.register(WriteFileTool(workspace_root))
    registry.register(EditFileTool(workspace_root))
    registry.register(
        ShellCommandTool(
            workspace_root,
            sandbox_supervisor=sandbox_supervisor,
        )
    )
    registry.register(HttpRequestTool())
    registry.register(WebSearchTool(settings=search_settings))
    return registry

__all__ = [
    "CurrentTimeTool",
    "EditFileTool",
    "HttpRequestTool",
    "ListFilesTool",
    "ReadFileTool",
    "ShellCommandTool",
    "WebSearchTool",
    "WriteFileTool",
    "build_builtin_tool_registry",
]
