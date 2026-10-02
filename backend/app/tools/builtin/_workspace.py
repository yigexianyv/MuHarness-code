
from __future__ import annotations

from pathlib import Path


# 函数说明：workspace_root_path
# 用途：在内置工作区工具中处理 `workspace_root_path`，通过 `Path(__file__).resolve` 完成
# 首个内部处理步骤。
# 参数：
#   workspace_root：文件工具允许访问的工作区根目录，类型 `str | Path | None`。
# 返回：类型 `Path`；返回 `Path(workspace_root).resolve()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(__file__).resolve` → `Path` →
# `Path(workspace_root).resolve`。
def workspace_root_path(workspace_root: str | Path | None) -> Path:
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[4] / "workspace"
    return Path(workspace_root).resolve()


# 函数说明：resolve_workspace_path
# 用途：解析工作区相对路径并验证最终路径仍在工作区内，阻止路径越界访问。
# 参数：
#   workspace_root：文件工具允许访问的工作区根目录，类型 `Path`。
#   relative_path：相对于工作区的路径，类型 `str`。
#   allow_root：相关数据的根目录或保存目录，类型 `bool`；默认 `False`。
# 返回：类型 `Path`；返回 `resolved`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path` → `requested.is_absolute` →
# `(workspace_root / requested).resolve` → `resolved.relative_to`。
# 分支与异常：
#   当 `requested.is_absolute()` 时，抛出
# `ValueError('path must be relative to the workspace')`。
#   捕获 `ValueError` 后，转换或抛出 `ValueError('path escapes the workspace')`。
#   当 `not allow_root and resolved == workspace_root` 时，抛出 `ValueError(…)`。
def resolve_workspace_path(
    workspace_root: Path,
    relative_path: str,
    *,
    allow_root: bool = False,
) -> Path:
    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError("path must be relative to the workspace")

    resolved = (workspace_root / requested).resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        raise ValueError("path escapes the workspace") from None

    if not allow_root and resolved == workspace_root:
        raise ValueError("path must reference an item inside the workspace")
    return resolved
