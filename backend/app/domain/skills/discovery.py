
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.paths import runtime_data_path, user_data_path

from .models import (
    SKILL_FILE_NAME,
    SkillMetadata,
    SkillScope,
    validate_skill_name,
)
from .parser import SkillParseError, parse_skill_document

logger = logging.getLogger("muharness.skills.discovery")

MAX_SKILL_FILE_BYTES = 512_000

DEFAULT_USER_SKILLS_DIR = user_data_path("skills")
DEFAULT_PROJECT_SKILLS_DIR = runtime_data_path("skills")


@dataclass(frozen=True)
class SkillDiagnostic:

    scope: SkillScope
    name: str
    location: str
    reason: str

    # 函数说明：SkillDiagnostic.render
    # 用途：生成展示文本SkillDiagnostic，供技能发现与激活使用。
    # 返回：类型 `str`；返回 `f'skill[{self.scope.value}] {self.name} at {self.location}
    #  skipped: {self.reason}'`。
    def render(self) -> str:
        return (
            f"skill[{self.scope.value}] {self.name} at {self.location} "
            f"skipped: {self.reason}"
        )


class SkillDiscovery:

    # 函数说明：SkillDiscovery.__init__
    # 用途：初始化 SkillDiscovery；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   user_dir：相关数据的根目录或保存目录，类型 `str | Path`；默认
    # `DEFAULT_USER_SKILLS_DIR`。
    #   project_dir：相关数据的根目录或保存目录，类型 `str | Path`；默认
    # `DEFAULT_PROJECT_SKILLS_DIR`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(user_dir).expanduser().resolve` → `Path(user_dir).expanduser` → `Path` →
    # `Path(project_dir).expanduser().resolve` → `Path(project_dir).expanduser`。
    # 副作用与资源：
    #   更新对象字段：`self.user_dir`、`self.project_dir`、`self._diagnostics`。
    def __init__(
        self,
        user_dir: str | Path = DEFAULT_USER_SKILLS_DIR,
        project_dir: str | Path = DEFAULT_PROJECT_SKILLS_DIR,
    ) -> None:
        self.user_dir = Path(user_dir).expanduser().resolve()
        self.project_dir = Path(project_dir).expanduser().resolve()
        self._diagnostics: list[SkillDiagnostic] = []

    # 函数说明：SkillDiscovery.diagnostics
    # 用途：返回 `tuple(self._diagnostics)`，提供 SkillDiscovery 的派生值。
    # 返回：类型 `tuple[SkillDiagnostic, ...]`；返回 `tuple(self._diagnostics)`。
    def diagnostics(self) -> tuple[SkillDiagnostic, ...]:
        return tuple(self._diagnostics)

    # 函数说明：SkillDiscovery.discover
    # 用途：发现SkillDiscovery，供技能发现与激活使用。
    # 返回：类型 `tuple[SkillMetadata, ...]`；返回
    # `tuple(sorted(merged.values(), key=lambda item: item.name))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._discover_scope` →
    # `merged.setdefault`。
    # 副作用与资源：
    #   更新对象字段：`self._diagnostics`。
    def discover(self) -> tuple[SkillMetadata, ...]:

        self._diagnostics = []
        merged: dict[str, SkillMetadata] = {}
        for metadata in self._discover_scope(
            self.project_dir, SkillScope.PROJECT
        ):
            merged[metadata.name] = metadata
        for metadata in self._discover_scope(self.user_dir, SkillScope.USER):
            merged.setdefault(metadata.name, metadata)
        return tuple(sorted(merged.values(), key=lambda item: item.name))

    # 函数说明：SkillDiscovery._discover_scope
    # 用途：发现作用域，供技能发现与激活使用。
    # 参数：
    #   root：当前操作的根目录，类型 `Path`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    # 返回：类型 `tuple[SkillMetadata, ...]`；按分支返回 `()`；`tuple(found)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`root.is_dir` → `root.iterdir` →
    # `child.is_dir` → `name.startswith` → `validate_skill_name` → `self._record`；另有
    # 2 个调用点。
    # 分支与异常：
    #   当 `not root.is_dir()` 时，返回 `()`。
    #   当 `not child.is_dir()` 时，跳过当前循环项。
    #   当 `name.startswith('.')` 时，跳过当前循环项。
    #   捕获 `ValueError` 后，跳过当前循环项，继续处理后续项。
    def _discover_scope(
        self,
        root: Path,
        scope: SkillScope,
    ) -> tuple[SkillMetadata, ...]:
        if not root.is_dir():
            return ()
        found: list[SkillMetadata] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("."):
                continue
            try:
                validate_skill_name(name)
            except ValueError as exc:
                self._record(scope, name, str(child), str(exc))
                continue
            skill_dir = safe_skill_dir(root, name)
            if skill_dir is None:
                self._record(
                    scope,
                    name,
                    str(child),
                    "skill directory escapes root or is a symlink",
                )
                continue
            metadata = self._read_metadata(skill_dir, scope)
            if metadata is not None:
                found.append(metadata)
        return tuple(found)

    # 函数说明：SkillDiscovery._read_metadata
    # 用途：读取`metadata`，供技能发现与激活使用。
    # 参数：
    #   skill_dir：技能文件目录，类型 `Path`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    # 返回：类型 `SkillMetadata | None`；按分支返回 `None`；`SkillMetadata(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`safe_skill_file` → `self._record`
    #  → `skill_file.stat` → `skill_file.read_text` → `parse_skill_document` →
    # `SkillMetadata`。
    # 分支与异常：
    #   `skill_file is None` 分支在完成前置处理后返回 `None`。
    #   当 `skill_file.stat().st_size > MAX_SKILL_FILE_BYTES` 时，抛出
    # `SkillParseError(…)`。
    #   捕获 `(OSError, UnicodeError)` 后，返回 `None`。
    #   捕获 `SkillParseError` 后，返回 `None`。
    # 副作用与资源：
    #   文件或资源访问：`skill_file.read_text`。
    def _read_metadata(
        self,
        skill_dir: Path,
        scope: SkillScope,
    ) -> SkillMetadata | None:
        name = skill_dir.name
        skill_file = safe_skill_file(skill_dir)
        if skill_file is None:
            self._record(
                scope,
                name,
                str(skill_dir),
                "SKILL.md is missing, a symlink, or escapes the skill root",
            )
            return None
        try:
            if skill_file.stat().st_size > MAX_SKILL_FILE_BYTES:
                raise SkillParseError(
                    f"SKILL.md exceeds {MAX_SKILL_FILE_BYTES} bytes"
                )
            text = skill_file.read_text(encoding="utf-8")
            parsed = parse_skill_document(text, expected_name=name)
        except (OSError, UnicodeError) as exc:
            self._record(scope, name, str(skill_dir), f"read error: {exc}")
            return None
        except SkillParseError as exc:
            self._record(scope, name, str(skill_dir), str(exc))
            return None
        return SkillMetadata(
            name=parsed.name,
            description=parsed.description,
            scope=scope,
            location=skill_file,
            license=parsed.license,
            compatibility=parsed.compatibility,
            metadata=parsed.metadata,
            allowed_tools=parsed.allowed_tools,
        )

    # 函数说明：SkillDiscovery._record
    # 用途：记录SkillDiscovery，供技能发现与激活使用。
    # 参数：
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   location：`location`输入或配置值，类型 `str`。
    #   reason：状态变化、拒绝或降级原因，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillDiagnostic` →
    # `logger.warning` → `diagnostic.render`。
    def _record(
        self,
        scope: SkillScope,
        name: str,
        location: str,
        reason: str,
    ) -> None:
        diagnostic = SkillDiagnostic(
            scope=scope,
            name=name,
            location=location,
            reason=reason,
        )
        self._diagnostics.append(diagnostic)
        logger.warning(diagnostic.render())


# 函数说明：safe_skill_dir
# 用途：在技能发现与激活中处理 `safe_skill_dir`，通过 `Path(root).expanduser().resolve`
# 完成首个内部处理步骤。
# 参数：
#   root：当前操作的根目录，类型 `str | Path`。
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `Path | None`；按分支返回 `None`；`resolved`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_skill_name` →
# `Path(root).expanduser().resolve` → `Path(root).expanduser` → `Path` →
# `candidate.is_symlink` → `candidate.resolve`；另有 2 个调用点。
# 分支与异常：
#   捕获 `ValueError` 后，返回 `None`。
#   当 `candidate.is_symlink()` 时，返回 `None`。
#   当 `not _is_within(root_path, resolved)` 时，返回 `None`。
#   当 `not resolved.is_dir()` 时，返回 `None`。
def safe_skill_dir(root: str | Path, name: str) -> Path | None:

    try:
        validate_skill_name(name)
    except ValueError:
        return None
    root_path = Path(root).expanduser().resolve()
    candidate = root_path / name
    if candidate.is_symlink():
        return None
    resolved = candidate.resolve()
    if not _is_within(root_path, resolved):
        return None
    if not resolved.is_dir():
        return None
    return resolved


# 函数说明：safe_skill_file
# 用途：在技能发现与激活中处理 `safe_skill_file`，通过 `skill_file.is_symlink` 完成首个
# 内部处理步骤。
# 参数：
#   skill_dir：技能文件目录，类型 `Path`。
# 返回：类型 `Path | None`；按分支返回 `None`；`skill_file`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_file.is_symlink` →
# `skill_file.is_file` → `skill_file.resolve` → `_is_within` → `skill_dir.resolve`。
# 分支与异常：
#   当 `skill_file.is_symlink()` 时，返回 `None`。
#   当 `not skill_file.is_file()` 时，返回 `None`。
#   当 `not _is_within(skill_dir.resolve(), resolved)` 时，返回 `None`。
def safe_skill_file(skill_dir: Path) -> Path | None:

    skill_file = skill_dir / SKILL_FILE_NAME
    if skill_file.is_symlink():
        return None
    if not skill_file.is_file():
        return None
    resolved = skill_file.resolve()
    if not _is_within(skill_dir.resolve(), resolved):
        return None
    return skill_file


# 函数说明：safe_skill_resource
# 用途：在技能发现与激活中处理 `safe_skill_resource`，通过 `relative.startswith` 完成首
# 个内部处理步骤。
# 参数：
#   skill_dir：技能文件目录，类型 `Path`。
#   relative：相对路径，类型 `str`。
# 返回：类型 `Path | None`；按分支返回 `None`；`target`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`relative.startswith` → `Path` →
# `skill_dir.resolve` → `raw_target.is_symlink` → `raw_target.resolve` → `_is_within`；
# 另有 1 个调用点。
# 分支与异常：
#   当 `not relative or relative.startswith(('/', '\\'))` 时，返回 `None`。
#   当 `any((part in ('..', '.', '') for part in parts))` 时，返回 `None`。
#   当 `raw_target.is_symlink()` 时，返回 `None`。
#   当 `not _is_within(root, target)` 时，返回 `None`。
def safe_skill_resource(skill_dir: Path, relative: str) -> Path | None:

    if not relative or relative.startswith(("/", "\\")):
        return None
    parts = Path(relative).parts
    if any(part in ("..", ".", "") for part in parts):
        return None
    root = skill_dir.resolve()
    raw_target = root / Path(*parts)
    if raw_target.is_symlink():
        return None
    target = raw_target.resolve()
    if not _is_within(root, target):
        return None
    if not target.is_file():
        return None
    return target


# 函数说明：_is_within
# 用途：在技能发现与激活中处理 `_is_within`，通过 `target.relative_to` 完成首个内部处理
# 步骤。
# 参数：
#   root：当前操作的根目录，类型 `Path`。
#   target：`target`输入或配置值，类型 `Path`。
# 返回：类型 `bool`；按分支返回 `True`；`False`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`target.relative_to`。
# 分支与异常：
#   捕获 `ValueError` 后，返回 `False`。
def _is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = [
    "DEFAULT_PROJECT_SKILLS_DIR",
    "DEFAULT_USER_SKILLS_DIR",
    "MAX_SKILL_FILE_BYTES",
    "SkillDiagnostic",
    "SkillDiscovery",
    "safe_skill_dir",
    "safe_skill_file",
    "safe_skill_resource",
]
