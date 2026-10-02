from __future__ import annotations

from app.paths import default_database_path
from app.server.__main__ import _application_kwargs, _parser


# 函数说明：test_database_help_describes_only_current_default
# 用途：回归验证回归测试与测试辅助中的 `database_help_describes_only_current_default` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_parser` → `next` →
# `parser.format_help`。
# 分支与异常：
#   验证条件：
# `action.help == 'SQLite database path (default: backend/.muharness/muharness.db).'`。
#   验证条件：`'backend/.muharness/muharness.db' in parser.format_help()`。
def test_database_help_describes_only_current_default() -> None:
    parser = _parser()
    action = next(action for action in parser._actions if action.dest == "database")
    assert action.help == (
        "SQLite database path (default: backend/.muharness/muharness.db)."
    )
    assert "backend/.muharness/muharness.db" in parser.format_help()


# 函数说明：test_default_database_is_resolved_by_application
# 用途：回归验证回归测试与测试辅助中的 `default_database_is_resolved_by_application` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_parser().parse_args` → `_parser` →
# `_application_kwargs` → `default_database_path`。
# 分支与异常：
#   验证条件：`'database' not in _application_kwargs(args)`。
#   验证条件：
# `default_database_path(tmp_path) == tmp_path / '.muharness' / 'muharness.db'`。
def test_default_database_is_resolved_by_application(tmp_path) -> None:
    args = _parser().parse_args([])
    assert "database" not in _application_kwargs(args)
    assert default_database_path(tmp_path) == tmp_path / ".muharness" / "muharness.db"


# 函数说明：test_explicit_database_path_is_not_replaced
# 用途：回归验证回归测试与测试辅助中的 `explicit_database_path_is_not_replaced` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_parser().parse_args` → `_parser` →
# `_application_kwargs`。
# 分支与异常：
#   验证条件：`_application_kwargs(args)['database'] == 'custom/application.db'`。
def test_explicit_database_path_is_not_replaced() -> None:
    args = _parser().parse_args(["--database", "custom/application.db"])
    assert _application_kwargs(args)["database"] == "custom/application.db"
