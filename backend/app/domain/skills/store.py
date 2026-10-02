
from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

import yaml

from .config import SkillSettings
from .discovery import (
    DEFAULT_PROJECT_SKILLS_DIR,
    DEFAULT_USER_SKILLS_DIR,
    SkillDiagnostic,
    SkillDiscovery,
    safe_skill_dir,
    safe_skill_file,
)
from .models import (
    Skill,
    SkillMetadata,
    SkillResources,
    SkillScope,
    validate_skill_name,
)
from .parser import SkillParseError, parse_skill_document

_DISABLED_DIR_NAME = ".disabled"


@dataclass(frozen=True)
class ManagedSkillEntry:

    metadata: SkillMetadata
    enabled: bool


class SkillStore:

    # 函数说明：SkillStore.__init__
    # 用途：初始化 SkillStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   user_dir：相关数据的根目录或保存目录，类型 `str | Path`；默认
    # `DEFAULT_USER_SKILLS_DIR`。
    #   project_dir：相关数据的根目录或保存目录，类型 `str | Path`；默认
    # `DEFAULT_PROJECT_SKILLS_DIR`。
    #   settings：业务或模型设置，类型 `SkillSettings | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(user_dir).expanduser().resolve` → `Path(user_dir).expanduser` → `Path` →
    # `Path(project_dir).expanduser().resolve` → `Path(project_dir).expanduser` →
    # `SkillSettings`；另有 1 个调用点。
    # 副作用与资源：
    #   更新对象字段：`self.user_dir`、`self.project_dir`、`self.settings`、
    # `self.discovery`。
    def __init__(
        self,
        user_dir: str | Path = DEFAULT_USER_SKILLS_DIR,
        project_dir: str | Path = DEFAULT_PROJECT_SKILLS_DIR,
        *,
        settings: SkillSettings | None = None,
    ) -> None:
        self.user_dir = Path(user_dir).expanduser().resolve()
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.settings = settings or SkillSettings()
        self.discovery = SkillDiscovery(
            user_dir=self.user_dir,
            project_dir=self.project_dir,
        )

    # 函数说明：SkillStore.initialize
    # 用途：初始化SkillStore，供技能发现与激活使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.project_dir.mkdir`。
    # 副作用与资源：
    #   文件或资源访问：`self.project_dir.mkdir`。
    async def initialize(self) -> None:

        self.project_dir.mkdir(parents=True, exist_ok=True)

    # 函数说明：SkillStore.catalog
    # 用途：返回 `self.discovery.discover()`，提供 SkillStore 的派生值。
    # 返回：类型 `tuple[SkillMetadata, ...]`；返回 `self.discovery.discover()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.discovery.discover`。
    async def catalog(self) -> tuple[SkillMetadata, ...]:

        return self.discovery.discover()

    # 函数说明：SkillStore.diagnostics
    # 用途：返回 `self.discovery.diagnostics()`，提供 SkillStore 的派生值。
    # 返回：类型 `tuple[SkillDiagnostic, ...]`；返回 `self.discovery.diagnostics()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.discovery.diagnostics`。
    def diagnostics(self) -> tuple[SkillDiagnostic, ...]:
        return self.discovery.diagnostics()

    # 函数说明：SkillStore.load
    # 用途：加载SkillStore，供技能发现与激活使用。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    # 返回：类型 `Skill | None`；按分支返回 `self._load_metadata(metadata)`；`None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.catalog` →
    # `self._load_metadata`。
    # 分支与异常：
    #   当 `metadata.name == name` 时，返回 `self._load_metadata(metadata)`。
    async def load(self, name: str) -> Skill | None:

        for metadata in await self.catalog():
            if metadata.name == name:
                return self._load_metadata(metadata)
        return None

    # 函数说明：SkillStore.managed_catalog
    # 用途：在技能发现与激活中处理 `managed_catalog`，通过 `self.discovery.discover` 完
    # 成首个内部处理步骤。
    # 返回：类型 `tuple[ManagedSkillEntry, ...]`；返回 `tuple(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.discovery.discover` →
    # `ManagedSkillEntry` → `self._scope_catalog`。
    async def managed_catalog(self) -> tuple[ManagedSkillEntry, ...]:

        self.discovery.discover()
        active = [
            ManagedSkillEntry(metadata=item, enabled=True)
            for scope, root in (
                (SkillScope.PROJECT, self.project_dir),
                (SkillScope.USER, self.user_dir),
            )
            for item in self._scope_catalog(root, scope)
        ]
        disabled = [
            ManagedSkillEntry(metadata=item, enabled=False)
            for scope, root in (
                (SkillScope.PROJECT, self.project_dir),
                (SkillScope.USER, self.user_dir),
            )
            for item in self._scope_catalog(root / _DISABLED_DIR_NAME, scope)
        ]
        return tuple(
            sorted(
                (*active, *disabled),
                key=lambda item: (
                    item.metadata.name,
                    item.metadata.scope.value,
                    not item.enabled,
                ),
            )
        )

    # 函数说明：SkillStore.install
    # 用途：校验技能文档并写入指定作用域的托管目录。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   description：补充描述，类型 `str`。
    #   instructions：`instructions`输入或配置值，类型 `str`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`；默认 `SkillScope.PROJECT`。
    # 返回：类型 `Skill`；返回 `installed`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_skill_name` →
    # `target.exists` → `disabled_target.exists` → `_render_skill_document` →
    # `parse_skill_document` → `root.mkdir`；另有 11 个调用点。
    # 分支与异常：
    #   当 `target.exists() or disabled_target.exists()` 时，抛出
    # `ValueError(f"skill '{normalized_name}' already exists")`。
    #   当 `installed is None` 时，抛出
    # `RuntimeError('installed skill could not be loaded')`。
    # 副作用与资源：
    #   文件或资源访问：`root.mkdir`、`temporary.mkdir`、`skill_file.open`、`os.replace`
    # 。
    async def install(
        self,
        *,
        name: str,
        description: str,
        instructions: str,
        scope: SkillScope = SkillScope.PROJECT,
    ) -> Skill:

        """校验技能文档并写入指定作用域的托管目录。"""
        normalized_name = validate_skill_name(name.strip())
        normalized_description = description.strip()
        normalized_instructions = instructions.strip()
        root = self.project_dir if scope is SkillScope.PROJECT else self.user_dir
        target = root / normalized_name
        disabled_target = root / _DISABLED_DIR_NAME / normalized_name
        if target.exists() or disabled_target.exists():
            raise ValueError(f"skill '{normalized_name}' already exists")

        markdown = _render_skill_document(
            normalized_name,
            normalized_description,
            normalized_instructions,
        )
        parse_skill_document(markdown, expected_name=normalized_name)

        root.mkdir(parents=True, exist_ok=True)
        temporary = root / f".{normalized_name}.{uuid4().hex}.tmp"
        try:
            temporary.mkdir()
            skill_file = temporary / "SKILL.md"
            with skill_file.open("w", encoding="utf-8") as handle:
                handle.write(markdown)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        installed = await self.load(normalized_name)
        if installed is None:
            raise RuntimeError("installed skill could not be loaded")
        return installed

    # 函数说明：SkillStore.update
    # 用途：更新已托管技能，同时保留其作用域和资源约束。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   description：补充描述，类型 `str`。
    #   instructions：`instructions`输入或配置值，类型 `str`。
    # 返回：类型 `Skill`；返回 `updated`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_skill_name` →
    # `self.load` → `_render_skill_document` → `parse_skill_document` →
    # `target.with_name` → `uuid4`；另有 8 个调用点。
    # 分支与异常：
    #   当 `existing is None` 时，抛出
    # `ValueError(f"skill '{normalized_name}' not found")`。
    #   当 `target.name != 'SKILL.md' or target.parent.name !=…` 时，抛出
    # `ValueError(…)`。
    #   当 `updated is None` 时，抛出
    # `RuntimeError('updated skill could not be loaded')`。
    # 副作用与资源：
    #   文件或资源访问：`temporary.open`、`os.replace`、`temporary.unlink`。
    async def update(
        self,
        *,
        name: str,
        description: str,
        instructions: str,
    ) -> Skill:

        """更新已托管技能，同时保留其作用域和资源约束。"""
        normalized_name = validate_skill_name(name.strip())
        existing = await self.load(normalized_name)
        if existing is None:
            raise ValueError(f"skill '{normalized_name}' not found")
        target = existing.metadata.location
        if target.name != "SKILL.md" or target.parent.name != normalized_name:
            raise ValueError(f"refusing to update unexpected skill path: {target}")

        markdown = _render_skill_document(
            normalized_name,
            description.strip(),
            instructions.strip(),
        )
        parse_skill_document(markdown, expected_name=normalized_name)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(markdown)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

        updated = await self.load(normalized_name)
        if updated is None:
            raise RuntimeError("updated skill could not be loaded")
        return updated

    # 函数说明：SkillStore.install_package
    # 用途：校验整个技能包后安装文档与附属资源。
    # 参数：
    #   files：文件集合输入或配置值，类型 `Mapping[str, bytes]`；读取键 `SKILL.md`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`；默认 `SkillScope.PROJECT`。
    # 返回：类型 `Skill`；返回 `installed`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_document.decode` →
    # `text.splitlines` → `next` → `yaml.safe_load` → `validate_skill_name` →
    # `parse_skill_document`；另有 20 个调用点。
    # 分支与异常：
    #   当 `not isinstance(skill_document, bytes)` 时，抛出
    # `ValueError('Skill package is missing SKILL.md')`。
    #   捕获 `UnicodeError` 后，转换或抛出 `ValueError('SKILL.md must be UTF-8')`。
    #   当 `not isinstance(front, dict) or not isinstance(front.get('…` 时，抛出
    # `ValueError('Skill package has invalid front matter')`。
    #   当 `target.exists() or disabled_target.exists()` 时，抛出
    # `ValueError(f"skill '{name}' already exists")`。
    # 副作用与资源：
    #   文件或资源访问：`root.mkdir`、`temporary.mkdir`、`destination.parent.mkdir`、
    # `destination.open`、`os.replace`。
    async def install_package(
        self,
        *,
        files: Mapping[str, bytes],
        scope: SkillScope = SkillScope.PROJECT,
    ) -> Skill:

        """校验整个技能包后安装文档与附属资源。"""
        skill_document = files.get("SKILL.md")
        if not isinstance(skill_document, bytes):
            raise ValueError("Skill package is missing SKILL.md")
        try:
            text = skill_document.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("SKILL.md must be UTF-8") from exc

        lines = text.splitlines()
        closing = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            ),
            None,
        )
        front = (
            yaml.safe_load("\n".join(lines[1:closing]))
            if lines and lines[0].strip() == "---" and closing is not None
            else None
        )
        if not isinstance(front, dict) or not isinstance(front.get("name"), str):
            raise ValueError("Skill package has invalid front matter")
        name = validate_skill_name(front["name"])
        parse_skill_document(text, expected_name=name)

        root = self._root(scope)
        target = root / name
        disabled_target = root / _DISABLED_DIR_NAME / name
        if target.exists() or disabled_target.exists():
            raise ValueError(f"skill '{name}' already exists")

        normalized_files: dict[PurePosixPath, bytes] = {}
        total_bytes = 0
        for relative, content in files.items():
            if not isinstance(relative, str) or not isinstance(content, bytes):
                raise ValueError("Skill package files must be byte mappings")
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "." in path.parts:
                raise ValueError("Skill package contains an unsafe path")
            if path.as_posix() != "SKILL.md" and (
                len(path.parts) < 2
                or path.parts[0] not in {"scripts", "references", "assets"}
            ):
                raise ValueError("Skill package contains an unsupported file")
            total_bytes += len(content)
            if total_bytes > 10 * 1024 * 1024:
                raise ValueError("Skill package exceeds 10MB")
            normalized_files[path] = content

        root.mkdir(parents=True, exist_ok=True)
        temporary = root / f".{name}.{uuid4().hex}.tmp"
        try:
            temporary.mkdir()
            for relative, content in normalized_files.items():
                destination = temporary.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        installed = await self.load(name)
        if installed is None:
            if target.exists():
                shutil.rmtree(target)
            raise RuntimeError("installed skill package could not be loaded")
        return installed

    # 函数说明：SkillStore.set_enabled
    # 用途：设置`enabled`，供技能发现与激活使用。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    #   enabled：`enabled`输入或配置值，类型 `bool`。
    # 返回：类型 `ManagedSkillEntry`；返回
    # `ManagedSkillEntry(metadata=metadata, enabled=enabled)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_skill_name` →
    # `self._root` → `safe_skill_dir` → `target.exists` → `target_root.mkdir` →
    # `os.replace`；另有 2 个调用点。
    # 分支与异常：
    #   当 `source is None` 时，抛出 `KeyError(f"skill '{normalized_name}' not found")`
    # 。
    #   当 `target.exists()` 时，抛出 `ValueError(…)`。
    #   `metadata is None` 分支在完成前置处理后抛出
    # `RuntimeError('moved skill could not be loaded')`。
    # 副作用与资源：
    #   文件或资源访问：`target_root.mkdir`、`os.replace`。
    async def set_enabled(
        self,
        *,
        name: str,
        scope: SkillScope,
        enabled: bool,
    ) -> ManagedSkillEntry:

        normalized_name = validate_skill_name(name)
        root = self._root(scope)
        source_root = root / _DISABLED_DIR_NAME if enabled else root
        target_root = root if enabled else root / _DISABLED_DIR_NAME
        source = safe_skill_dir(source_root, normalized_name)
        if source is None:
            raise KeyError(f"skill '{normalized_name}' not found")
        target = target_root / normalized_name
        if target.exists():
            raise ValueError(f"skill '{normalized_name}' target already exists")
        target_root.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        metadata = self._read_metadata_at(target, scope)
        if metadata is None:
            os.replace(target, source)
            raise RuntimeError("moved skill could not be loaded")
        return ManagedSkillEntry(metadata=metadata, enabled=enabled)

    # 函数说明：SkillStore.delete
    # 用途：删除SkillStore，供技能发现与激活使用。
    # 参数：
    #   name：目标对象、工具或配置项名称，类型 `str`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    #   enabled：`enabled`输入或配置值，类型 `bool`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_skill_name` →
    # `self._root` → `safe_skill_dir` → `shutil.rmtree`。
    # 分支与异常：
    #   当 `source is None` 时，抛出 `KeyError(f"skill '{normalized_name}' not found")`
    # 。
    async def delete(
        self,
        *,
        name: str,
        scope: SkillScope,
        enabled: bool,
    ) -> None:

        normalized_name = validate_skill_name(name)
        root = self._root(scope)
        source_root = root if enabled else root / _DISABLED_DIR_NAME
        source = safe_skill_dir(source_root, normalized_name)
        if source is None:
            raise KeyError(f"skill '{normalized_name}' not found")
        shutil.rmtree(source)

    # 函数说明：SkillStore._root
    # 用途：返回 `self.project_dir if scope is SkillScope.PROJECT else self.user_dir`，
    # 提供 SkillStore 的派生值。
    # 参数：
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    # 返回：类型 `Path`；返回
    # `self.project_dir if scope is SkillScope.PROJECT else self.user_dir`。
    def _root(self, scope: SkillScope) -> Path:
        return self.project_dir if scope is SkillScope.PROJECT else self.user_dir

    # 函数说明：SkillStore._scope_catalog
    # 用途：在技能发现与激活中处理 `_scope_catalog`，通过 `discovery.discover` 完成首个
    # 内部处理步骤。
    # 参数：
    #   root：当前操作的根目录，类型 `Path`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    # 返回：类型 `tuple[SkillMetadata, ...]`；返回 `discovery.discover()`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillDiscovery` →
    # `discovery.discover`。
    def _scope_catalog(
        self,
        root: Path,
        scope: SkillScope,
    ) -> tuple[SkillMetadata, ...]:
        empty = root.parent / f".management-empty-{scope.value}"
        discovery = SkillDiscovery(
            project_dir=root if scope is SkillScope.PROJECT else empty,
            user_dir=root if scope is SkillScope.USER else empty,
        )
        return discovery.discover()

    # 函数说明：SkillStore._read_metadata_at
    # 用途：读取`metadata_at`，供技能发现与激活使用。
    # 参数：
    #   skill_dir：技能文件目录，类型 `Path`。
    #   scope：记忆、规则或查询作用域，类型 `SkillScope`。
    # 返回：类型 `SkillMetadata | None`；返回 `next(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._scope_catalog` → `next`。
    def _read_metadata_at(
        self,
        skill_dir: Path,
        scope: SkillScope,
    ) -> SkillMetadata | None:
        catalog = self._scope_catalog(skill_dir.parent, scope)
        return next(
            (
                item
                for item in catalog
                if item.name == skill_dir.name and item.scope is scope
            ),
            None,
        )

    # 函数说明：SkillStore._load_metadata
    # 用途：加载`metadata`，供技能发现与激活使用。
    # 参数：
    #   metadata：关联元数据，类型 `SkillMetadata`。
    # 返回：类型 `Skill | None`；按分支返回 `None`；`Skill(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`safe_skill_file` →
    # `skill_file.read_text` → `parse_skill_document` → `Skill` →
    # `self._discover_resources`。
    # 分支与异常：
    #   当 `skill_file is None` 时，返回 `None`。
    #   捕获 `(OSError, UnicodeError, SkillParseError)` 后，返回 `None`。
    # 副作用与资源：
    #   文件或资源访问：`skill_file.read_text`。
    def _load_metadata(self, metadata: SkillMetadata) -> Skill | None:
        skill_dir = metadata.location.parent
        skill_file = safe_skill_file(skill_dir)
        if skill_file is None:
            return None
        try:
            text = skill_file.read_text(encoding="utf-8")
            parsed = parse_skill_document(text, expected_name=metadata.name)
        except (OSError, UnicodeError, SkillParseError):
            return None
        return Skill(
            metadata=metadata,
            content=parsed.body,
            root=skill_dir,
            resources=self._discover_resources(skill_dir),
        )

    # 函数说明：SkillStore._discover_resources
    # 用途：发现`resources`，供技能发现与激活使用。
    # 参数：
    #   skill_dir：技能文件目录，类型 `Path`。
    # 返回：类型 `SkillResources`；返回 `SkillResources(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillResources` →
    # `_list_resource_dir`。
    def _discover_resources(self, skill_dir: Path) -> SkillResources:
        return SkillResources(
            scripts=_list_resource_dir(skill_dir, "scripts"),
            references=_list_resource_dir(skill_dir, "references"),
            assets=_list_resource_dir(skill_dir, "assets"),
        )


# 函数说明：_render_skill_document
# 用途：生成展示文本技能，供技能发现与激活使用。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   description：补充描述，类型 `str`。
#   instructions：`instructions`输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `f'---\n{front_matter}\n---\n\n{instructions}\n'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `yaml.safe_dump({'name': name, 'description': description}, allow_unicode=…` →
# `yaml.safe_dump`。
def _render_skill_document(
    name: str,
    description: str,
    instructions: str,
) -> str:

    front_matter = yaml.safe_dump(
        {"name": name, "description": description},
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    return f"---\n{front_matter}\n---\n\n{instructions}\n"


# 函数说明：_list_resource_dir
# 用途：列出`resource_dir`，供技能发现与激活使用。
# 参数：
#   skill_dir：技能文件目录，类型 `Path`。
#   subdir：`subdir`输入或配置值，类型 `str`。
# 返回：类型 `tuple[str, ...]`；按分支返回 `()`；`tuple(entries)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`directory.is_dir` →
# `directory.is_symlink` → `directory.rglob` → `path.is_file` → `path.is_symlink` →
# `path.relative_to(skill_dir).as_posix`；另有 1 个调用点。
# 分支与异常：
#   当 `not directory.is_dir() or directory.is_symlink()` 时，返回 `()`。
def _list_resource_dir(skill_dir: Path, subdir: str) -> tuple[str, ...]:
    directory = skill_dir / subdir
    if not directory.is_dir() or directory.is_symlink():
        return ()
    entries: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            entries.append(path.relative_to(skill_dir).as_posix())
    return tuple(entries)


__all__ = ["ManagedSkillEntry", "SkillStore"]
