
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


# 函数说明：runtime_data_path
# 用途：将相对路径定位到后端根目录下的运行数据目录。
# 参数：
#   relative：相对路径，类型 `str`。
#   backend_root：后端根目录；省略时从当前模块位置推导，类型 `Path | None`；默认 `None`
# 。
# 返回：类型 `Path`；返回 `root / '.muharness' / relative`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(__file__).resolve` → `Path`。
def runtime_data_path(relative: str, *, backend_root: Path | None = None) -> Path:

    root = backend_root or Path(__file__).resolve().parents[1]
    return root / ".muharness" / relative


# 函数说明：user_data_path
# 用途：将相对路径定位到用户主目录下的应用数据目录。
# 参数：
#   relative：相对路径，类型 `str`。
#   home：用于定位用户数据的主目录；省略时使用系统用户主目录，类型 `Path | None`；默认
# `None`。
# 返回：类型 `Path`；返回 `(home or Path.home()) / '.muharness' / relative`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path.home`。
def user_data_path(relative: str, *, home: Path | None = None) -> Path:

    return (home or Path.home()) / ".muharness" / relative


# 函数说明：default_database_path
# 用途：获取默认 SQLite 数据库的运行数据路径。
# 参数：
#   backend_root：后端根目录；省略时从当前模块位置推导，类型 `Path | None`；默认 `None`
# 。
# 返回：类型 `Path`；返回 `runtime_data_path('muharness.db', backend_root=backend_root)`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`runtime_data_path`。
def default_database_path(backend_root: Path | None = None) -> Path:
    return runtime_data_path("muharness.db", backend_root=backend_root)


# 函数说明：preferred_env
# 用途：按新环境变量、旧别名、环境文件和默认值的顺序选择非空配置。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   previous_name：兼容旧配置使用的环境变量名称，类型 `str`。
#   default：未提供有效值时使用的默认值，类型 `str`。
#   env_file：备用环境配置文件路径；省略时使用后端环境文件，类型 `Path | None`；默认
# `None`。
# 返回：类型 `str`；按分支返回 `candidate.strip()`；`default`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`dotenv_values`。
# 分支与异常：
#   当 `candidate and candidate.strip()` 时，返回 `candidate.strip()`。
def preferred_env(
    name: str,
    previous_name: str,
    default: str,
    *,
    env_file: Path | None = None,
) -> str:

    for candidate in (os.environ.get(name), os.environ.get(previous_name)):
        if candidate and candidate.strip():
            return candidate.strip()

    settings = dotenv_values(env_file or _BACKEND_ENV_FILE)
    for candidate in (settings.get(name), settings.get(previous_name)):
        if candidate and candidate.strip():
            return candidate.strip()
    return default


__all__ = [
    "default_database_path",
    "preferred_env",
    "runtime_data_path",
    "user_data_path",
]
