
from __future__ import annotations

from app.paths import (
    default_database_path,
    preferred_env,
    runtime_data_path,
    user_data_path,
)


# 函数说明：test_runtime_resources_always_use_current_directory
# 用途：回归验证回归测试与测试辅助中的 `runtime_resources_always_use_current_directory`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`default_database_path` →
# `runtime_data_path` → `current.exists` → `current.mkdir` → `expected_database.touch` →
#  `(current / 'tasks').mkdir`。
# 分支与异常：
#   验证条件：`default_database_path(tmp_path) == expected_database`。
#   验证条件：`runtime_data_path(relative, backend_root=tmp_path) == current / relative`
# 。
#   验证条件：`not current.exists()`。
# 副作用与资源：
#   文件或资源访问：`current.mkdir`、`(current / 'tasks').mkdir`。
def test_runtime_resources_always_use_current_directory(tmp_path) -> None:
    current = tmp_path / ".muharness"
    expected_database = current / "muharness.db"
    assert default_database_path(tmp_path) == expected_database
    for relative in ("tasks", "memory", "settings/models.json", "mcp.json"):
        assert runtime_data_path(relative, backend_root=tmp_path) == current / relative
    assert not current.exists()

    current.mkdir()
    expected_database.touch()
    (current / "tasks").mkdir()
    assert default_database_path(tmp_path) == expected_database
    assert runtime_data_path("tasks", backend_root=tmp_path) == current / "tasks"


# 函数说明：test_other_resource_directories_are_not_selected
# 用途：回归验证回归测试与测试辅助中的 `other_resource_directories_are_not_selected` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`alternate.mkdir` →
# `(alternate / 'vesta.db').touch` → `(alternate / 'tasks').mkdir` →
# `(alternate / 'memory').mkdir` → `(alternate / 'settings').mkdir` →
# `(alternate / 'settings' / 'models.json').touch`；另有 3 个调用点。
# 分支与异常：
#   验证条件：
# `default_database_path(tmp_path) == tmp_path / '.muharness' / 'muharness.db'`。
#   验证条件：`runtime_data_path(relative, backend_root=tmp_path) == tmp_path / '.
# muharness' / relative`。
#   验证条件：`not (tmp_path / '.muharness').exists()`。
# 副作用与资源：
#   文件或资源访问：`alternate.mkdir`、`(alternate / 'tasks').mkdir`、
# `(alternate / 'memory').mkdir`、`(alternate / 'settings').mkdir`。
def test_other_resource_directories_are_not_selected(tmp_path) -> None:
    alternate = tmp_path / ".vesta"
    alternate.mkdir()
    (alternate / "vesta.db").touch()
    (alternate / "tasks").mkdir()
    (alternate / "memory").mkdir()
    (alternate / "settings").mkdir()
    (alternate / "settings" / "models.json").touch()
    assert default_database_path(tmp_path) == tmp_path / ".muharness" / "muharness.db"
    for relative in ("tasks", "memory", "settings/models.json"):
        assert runtime_data_path(relative, backend_root=tmp_path) == (
            tmp_path / ".muharness" / relative
        )
    assert not (tmp_path / ".muharness").exists()


# 函数说明：test_user_skills_always_use_current_directory
# 用途：回归验证回归测试与测试辅助中的 `user_skills_always_use_current_directory` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `(tmp_path / '.vesta' / 'skills').mkdir` → `user_data_path` → `current_skills.exists`
# → `current_skills.mkdir`。
# 分支与异常：
#   验证条件：`user_data_path('skills', home=tmp_path) == current_skills`。
#   验证条件：`not current_skills.exists()`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / '.vesta' / 'skills').mkdir`、`current_skills.mkdir`。
def test_user_skills_always_use_current_directory(tmp_path) -> None:
    (tmp_path / ".vesta" / "skills").mkdir(parents=True)
    current_skills = tmp_path / ".muharness" / "skills"
    assert user_data_path("skills", home=tmp_path) == current_skills
    assert not current_skills.exists()
    current_skills.mkdir(parents=True)
    assert user_data_path("skills", home=tmp_path) == current_skills


# 函数说明：test_environment_precedence_and_blank_fallback
# 用途：回归验证回归测试与测试辅助中的 `environment_precedence_and_blank_fallback` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具。
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`env_file.write_text` →
# `monkeypatch.delenv` → `preferred_env` → `monkeypatch.setenv`。
# 分支与异常：
#   验证条件：`preferred_env('MUHARNESS_SANDBOX_IMAGE', 'VESTA_SANDBOX_IMAGE', 'default-
# image', env_file=env_file) == 'file-new'`。
#   验证条件：`preferred_env('MUHARNESS_SANDBOX_IMAGE', 'VESTA_SANDBOX_IMAGE', 'default-
# image', env_file=env_file) == 'process-old'`。
#   验证条件：`preferred_env('MUHARNESS_SANDBOX_IMAGE', 'VESTA_SANDBOX_IMAGE', 'default-
# image', env_file=env_file) == 'process-new'`。
# 副作用与资源：
#   文件或资源访问：`env_file.write_text`。
def test_environment_precedence_and_blank_fallback(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MUHARNESS_SANDBOX_IMAGE=file-new\n"
        "VESTA_SANDBOX_IMAGE=file-old\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MUHARNESS_SANDBOX_IMAGE", raising=False)
    monkeypatch.delenv("VESTA_SANDBOX_IMAGE", raising=False)
    assert preferred_env(
        "MUHARNESS_SANDBOX_IMAGE",
        "VESTA_SANDBOX_IMAGE",
        "default-image",
        env_file=env_file,
    ) == "file-new"

    monkeypatch.setenv("VESTA_SANDBOX_IMAGE", "process-old")
    assert preferred_env(
        "MUHARNESS_SANDBOX_IMAGE",
        "VESTA_SANDBOX_IMAGE",
        "default-image",
        env_file=env_file,
    ) == "process-old"

    monkeypatch.setenv("MUHARNESS_SANDBOX_IMAGE", "process-new")
    assert preferred_env(
        "MUHARNESS_SANDBOX_IMAGE",
        "VESTA_SANDBOX_IMAGE",
        "default-image",
        env_file=env_file,
    ) == "process-new"

    monkeypatch.setenv("MUHARNESS_SANDBOX_IMAGE", "   ")
    assert preferred_env(
        "MUHARNESS_SANDBOX_IMAGE",
        "VESTA_SANDBOX_IMAGE",
        "default-image",
        env_file=env_file,
    ) == "process-old"

    monkeypatch.delenv("VESTA_SANDBOX_IMAGE")
    env_file.write_text(
        "MUHARNESS_SANDBOX_IMAGE= \nVESTA_SANDBOX_IMAGE=file-old\n",
        encoding="utf-8",
    )
    assert preferred_env(
        "MUHARNESS_SANDBOX_IMAGE",
        "VESTA_SANDBOX_IMAGE",
        "default-image",
        env_file=env_file,
    ) == "file-old"
