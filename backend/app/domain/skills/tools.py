
from __future__ import annotations

from typing import Any

from app.models.types import ToolDefinition
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry

from .discovery import safe_skill_resource
from .store import SkillStore

SKILL_READ_TOOL_NAME = "skill_read"
SKILL_RESOURCE_READ_TOOL_NAME = "skill_resource_read"

_MAX_RESOURCE_BYTES = 64_000


class SkillReadTool(BaseTool):

    # 函数说明：SkillReadTool.__init__
    # 用途：初始化 SkillReadTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SkillStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: SkillStore) -> None:
        self._store = store

    # 函数说明：SkillReadTool.definition
    # 用途：提供 SkillReadTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="skill_read",
            record_output=False,
            description=(
                "仅加载本阶段相关 Skill，依目录选名，不全量或重复加载。资源用 "
                "skill_resource_read。found=true 仅找到，预算许可才注入正文；"
                "不执行、不扩权。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Available Skills 中精确 name。"
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：SkillReadTool.execute
    # 用途：执行SkillReadTool，供技能发现与激活使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `name`
    # 。
    # 返回：类型 `dict[str, Any]`；按分支返回 `{'found': False, 'name': name}`；`{'found
    # ': True, 'name': skill.metadata.name, 'description': skill.metadata.description, '
    # …`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.load` →
    # `skill.resources.as_dict`。
    # 分支与异常：
    #   当 `not isinstance(name, str) or not name.strip()` 时，抛出
    # `ValueError("'name' must be a non-empty string")`。
    #   当 `skill is None` 时，返回 `{'found': False, 'name': name}`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string")
        skill = await self._store.load(name)
        if skill is None:
            return {"found": False, "name": name}
        return {
            "found": True,
            "name": skill.metadata.name,
            "description": skill.metadata.description,
            "scope": skill.metadata.scope.value,
            "resources": skill.resources.as_dict(),
        }


class SkillResourceReadTool(BaseTool):

    # 函数说明：SkillResourceReadTool.__init__
    # 用途：初始化 SkillResourceReadTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `SkillStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: SkillStore) -> None:
        self._store = store

    # 函数说明：SkillResourceReadTool.definition
    # 用途：提供 SkillResourceReadTool 的模型可见定义，包含名称、说明、参数结构及权限声
    # 明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="skill_resource_read",
            record_output=False,
            description=(
                "读取本 Run 已激活 Skill 内一个需要使用的配套文本资源。仅在"
                "Active Skill 流程需要 references/scripts/assets 中的文件时"
                "按需读取，不全量扫资源；尚未激活先 skill_read。它不同于加载"
                "Skill 主指令，也不读取任意工作区文件。路径须留在 Skill 目录"
                "内，禁止绝对路径和 ../；只支持不超过 64000 字节的 UTF-8 文件。"
                "found=true 表示正文已读取，不表示 scripts 已执行、资源已应用"
                "或任务已完成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "本 Run 已由 skill_read 激活、且当前仍相关的精确 Skill 名称。",
                    },
                    "path": {
                        "type": "string",
                        "description": "Active Skill 列出的资源相对路径，例如 references/guide.md；不能使用绝对路径、../ 或目录。",
                    },
                },
                "required": ["name", "path"],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：SkillResourceReadTool.execute
    # 用途：执行SkillResourceReadTool，供技能发现与激活使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `dict[str, Any]`；返回 `await self._read_resource(arguments)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._read_resource`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:

        return await self._read_resource(arguments)

    # 函数说明：SkillResourceReadTool.execute_with_context
    # 用途：执行上下文，供技能发现与激活使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `name`
    # 。
    #   context：本次操作的上下文对象，类型 `Any`。
    # 返回：类型 `dict[str, Any]`；按分支返回 `{'found': False, 'name': name, 'error': '
    # skill is not active in the current run; call…`；
    # `await self._read_resource(arguments)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._read_resource`。
    # 分支与异常：
    #   当 `not isinstance(name, str) or not name.strip()` 时，抛出
    # `ValueError("'name' must be a non-empty string")`。
    #   当 `name not in active_names` 时，返回
    # `{'found': False, 'name': name, 'error': 'skill is not…`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:

        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string")
        active_names = context.metadata.get("active_skill_names", ())
        if name not in active_names:
            return {
                "found": False,
                "name": name,
                "error": (
                    "skill is not active in the current run; call skill_read first"
                ),
            }
        return await self._read_resource(arguments)

    # 函数说明：SkillResourceReadTool._read_resource
    # 用途：读取`resource`，供技能发现与激活使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键 `name`
    # 、`path`。
    # 返回：类型 `dict[str, Any]`；按分支返回 `{'found': False, 'name': name}`；`{'found
    # ': False, 'name': name, 'path': resource_path, 'error': 'resource path escapes…`；
    # `{'found': False, 'name': name, 'path': resource_path, 'error': f'resource exceeds
    #  {…`；`{'found': False, 'name': name, 'path': resource_path, 'error': 'resource
    # read failed'}` 等 5 种表达式。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.load` →
    # `safe_skill_resource` → `target.stat` → `target.read_text`。
    # 分支与异常：
    #   当 `not isinstance(name, str) or not name.strip()` 时，抛出
    # `ValueError("'name' must be a non-empty string")`。
    #   当 `not isinstance(resource_path, str) or not…` 时，抛出
    # `ValueError("'path' must be a non-empty string")`。
    #   当 `skill is None` 时，返回 `{'found': False, 'name': name}`。
    #   当 `target is None` 时，返回
    # `{'found': False, 'name': name, 'path': resource_path, '…`。
    #   捕获 `(OSError, UnicodeError)` 后，返回 `{'found': False, 'name': name, 'path':
    # resource_path, 'error': 'resource read failed'}`。
    # 副作用与资源：
    #   文件或资源访问：`target.read_text`。
    async def _read_resource(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("name")
        resource_path = arguments.get("path")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string")
        if not isinstance(resource_path, str) or not resource_path.strip():
            raise ValueError("'path' must be a non-empty string")
        skill = await self._store.load(name)
        if skill is None:
            return {"found": False, "name": name}
        target = safe_skill_resource(skill.root, resource_path)
        if target is None:
            return {
                "found": False,
                "name": name,
                "path": resource_path,
                "error": "resource path escapes skill root or is not a file",
            }
        try:
            if target.stat().st_size > _MAX_RESOURCE_BYTES:
                return {
                    "found": False,
                    "name": name,
                    "path": resource_path,
                    "error": f"resource exceeds {_MAX_RESOURCE_BYTES} bytes",
                }
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return {
                "found": False,
                "name": name,
                "path": resource_path,
                "error": "resource read failed",
            }
        return {
            "found": True,
            "name": name,
            "path": resource_path,
            "content": content,
        }


# 函数说明：register_skill_tools
# 用途：注册技能工具集合，供技能发现与激活使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   store：持久化存储依赖，类型 `SkillStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.register` → `SkillReadTool`
# → `SkillResourceReadTool`。
def register_skill_tools(
    registry: ToolRegistry,
    store: SkillStore,
) -> None:

    registry.register(SkillReadTool(store))
    registry.register(SkillResourceReadTool(store))


__all__ = [
    "SKILL_READ_TOOL_NAME",
    "SKILL_RESOURCE_READ_TOOL_NAME",
    "SkillReadTool",
    "SkillResourceReadTool",
    "register_skill_tools",
]
