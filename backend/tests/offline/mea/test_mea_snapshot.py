
from __future__ import annotations

from pathlib import Path

from app.runtime.mea.snapshot import snapshot_diff, snapshot_digest, snapshot_workspace


# 函数说明：_workspace
# 用途：在回归测试与测试辅助中处理 `_workspace`，通过 `(root / 'src').mkdir` 完成首个内
# 部处理步骤。
# 参数：
#   root：当前操作的根目录，类型 `Path`。
# 返回：类型 `Path`；返回 `root`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(root / 'src').mkdir` →
# `(root / 'src' / 'app.py').write_text` → `(root / 'data.csv').write_text` →
# `(root / 'notes').mkdir`。
# 副作用与资源：
#   文件或资源访问：`(root / 'src').mkdir`、`(root / 'src' / 'app.py').write_text`、
# `(root / 'data.csv').write_text`、`(root / 'notes').mkdir`。
def _workspace(root: Path) -> Path:
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "data.csv").write_text("id,name\n1,a\n", encoding="utf-8")
    (root / "notes").mkdir()
    return root


# 函数说明：test_no_change_is_clean
# 用途：回归验证回归测试与测试辅助中的 `no_change_is_clean` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_workspace` → `snapshot_workspace` →
# `snapshot_diff` → `snapshot_digest`。
# 分支与异常：
#   验证条件：`not diff.mutated`。
#   验证条件：`snapshot_digest(before) == snapshot_digest(after)`。
def test_no_change_is_clean(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    before = snapshot_workspace(root)
    after = snapshot_workspace(root)

    diff = snapshot_diff(before, after)
    assert not diff.mutated
    assert snapshot_digest(before) == snapshot_digest(after)


# 函数说明：test_added_file_is_reported
# 用途：回归验证回归测试与测试辅助中的 `added_file_is_reported` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_workspace` → `snapshot_workspace` →
# `(root / 'src' / 'new.py').write_text` → `snapshot_diff`。
# 分支与异常：
#   验证条件：`diff.mutated`。
#   验证条件：`'src/new.py' in diff.added`。
# 副作用与资源：
#   文件或资源访问：`(root / 'src' / 'new.py').write_text`。
def test_added_file_is_reported(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    before = snapshot_workspace(root)
    (root / "src" / "new.py").write_text("x = 1\n", encoding="utf-8")

    diff = snapshot_diff(before, snapshot_workspace(root))
    assert diff.mutated
    assert "src/new.py" in diff.added


# 函数说明：test_deleted_file_is_reported
# 用途：回归验证回归测试与测试辅助中的 `deleted_file_is_reported` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_workspace` → `snapshot_workspace` →
# `(root / 'data.csv').unlink` → `snapshot_diff`。
# 分支与异常：
#   验证条件：`diff.deleted == ('data.csv',)`。
# 副作用与资源：
#   文件或资源访问：`(root / 'data.csv').unlink`。
def test_deleted_file_is_reported(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    before = snapshot_workspace(root)
    (root / "data.csv").unlink()

    diff = snapshot_diff(before, snapshot_workspace(root))
    assert diff.deleted == ("data.csv",)


# 函数说明：test_modified_content_with_same_size_and_mtime_is_reported
# 用途：回归验证回归测试与测试辅助中的
# `modified_content_with_same_size_and_mtime_is_reported` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_workspace` → `target.stat` →
# `snapshot_workspace` → `target.write_text` → `os.utime` → `snapshot_diff`；另有 1 个调
# 用点。
# 分支与异常：
#   验证条件：`diff.changed == ('data.csv',)`。
#   验证条件：`snapshot_digest(before) != snapshot_digest(snapshot_workspace(root))`。
# 副作用与资源：
#   文件或资源访问：`target.write_text`。
def test_modified_content_with_same_size_and_mtime_is_reported(tmp_path: Path) -> None:
    import os

    root = _workspace(tmp_path)
    target = root / "data.csv"
    stat = target.stat()
    before = snapshot_workspace(root)
    target.write_text("id,name\n1,b\n", encoding="utf-8")  # 同样长度
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # 连 mtime 也还原

    diff = snapshot_diff(before, snapshot_workspace(root))
    assert diff.changed == ("data.csv",)
    assert snapshot_digest(before) != snapshot_digest(snapshot_workspace(root))


# 函数说明：test_type_change_is_reported
# 用途：回归验证回归测试与测试辅助中的 `type_change_is_reported` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_workspace` → `snapshot_workspace` →
# `(root / 'notes').rmdir` → `(root / 'notes').write_text` → `snapshot_diff` →
# `diff.paths`。
# 分支与异常：
#   验证条件：`diff.type_changed == ('notes',)`。
#   验证条件：`'! notes' in diff.paths()`。
# 副作用与资源：
#   文件或资源访问：`(root / 'notes').rmdir`、`(root / 'notes').write_text`。
def test_type_change_is_reported(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    before = snapshot_workspace(root)
    (root / "notes").rmdir()
    (root / "notes").write_text("now a file", encoding="utf-8")

    diff = snapshot_diff(before, snapshot_workspace(root))
    assert diff.type_changed == ("notes",)
    assert "! notes" in diff.paths()


# 函数说明：test_excluded_paths_are_ignored
# 用途：回归验证回归测试与测试辅助中的 `excluded_paths_are_ignored` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_workspace` →
# `(root / '.muharness').mkdir` → `snapshot_workspace` →
# `(root / '.muharness' / 'log.txt').write_text` → `snapshot_diff`。
# 分支与异常：
#   验证条件：`not diff.mutated`。
# 副作用与资源：
#   文件或资源访问：`(root / '.muharness').mkdir`、
# `(root / '.muharness' / 'log.txt').write_text`。
def test_excluded_paths_are_ignored(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    (root / ".muharness").mkdir()
    before = snapshot_workspace(root, exclude=[".muharness"])
    (root / ".muharness" / "log.txt").write_text("harness log", encoding="utf-8")

    diff = snapshot_diff(before, snapshot_workspace(root, exclude=[root / ".muharness"]))
    assert not diff.mutated


# 函数说明：test_missing_workspace_records_error
# 用途：回归验证回归测试与测试辅助中的 `missing_workspace_records_error` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`snapshot_workspace`。
# 分支与异常：
#   验证条件：`snapshot.records == {}`。
#   验证条件：`snapshot.errors and 'does not exist' in snapshot.errors[0]`。
def test_missing_workspace_records_error(tmp_path: Path) -> None:
    snapshot = snapshot_workspace(tmp_path / "missing")
    assert snapshot.records == {}
    assert snapshot.errors and "does not exist" in snapshot.errors[0]


# 函数说明：test_repeated_snapshots_of_nested_tree_are_stable
# 用途：回归验证回归测试与测试辅助中的 `repeated_snapshots_of_nested_tree_are_stable` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_workspace` → `deep.mkdir` →
# `(deep / f'f{index}.txt').write_text` → `snapshot_workspace` → `snapshot_diff`。
# 分支与异常：
#   验证条件：`not snapshot_diff(first, snapshot_workspace(root)).mutated`。
# 副作用与资源：
#   文件或资源访问：`deep.mkdir`、`(deep / f'f{index}.txt').write_text`。
def test_repeated_snapshots_of_nested_tree_are_stable(tmp_path: Path) -> None:
    # Windows 上父目录索引里的目录时间戳会延迟回写，曾导致没有改动也报 changed
    root = _workspace(tmp_path)
    deep = root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    for index in range(20):
        (deep / f"f{index}.txt").write_text(str(index), encoding="utf-8")

    first = snapshot_workspace(root)
    for _ in range(3):
        assert not snapshot_diff(first, snapshot_workspace(root)).mutated


# 函数说明：test_new_nested_file_is_reported_as_the_file_only
# 用途：回归验证回归测试与测试辅助中的 `new_nested_file_is_reported_as_the_file_only` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_workspace` → `snapshot_workspace` →
# `(root / 'src' / 'extra.py').write_text` → `snapshot_diff`。
# 分支与异常：
#   验证条件：`diff.added == ('src/extra.py',)`。
#   验证条件：`diff.changed == ()`。
# 副作用与资源：
#   文件或资源访问：`(root / 'src' / 'extra.py').write_text`。
def test_new_nested_file_is_reported_as_the_file_only(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    before = snapshot_workspace(root)
    (root / "src" / "extra.py").write_text("y = 2\n", encoding="utf-8")

    diff = snapshot_diff(before, snapshot_workspace(root))
    assert diff.added == ("src/extra.py",)
    assert diff.changed == ()  # 目录不记 mtime，不会连带报 src
