
from __future__ import annotations

from collections.abc import Sequence

from app.models.types import Message, MessageRole
from app.runtime.context.tokens import default_token_estimator

from .models import Skill, SkillMetadata

SKILL_CATALOG_MESSAGE_NAME = "muharness_skill_catalog"
ACTIVE_SKILL_MESSAGE_NAME = "muharness_active_skill"

_CATALOG_HEADER = (
    "# Available Skills\n\n"
    "匹配各阶段：先 skill_read，再操作或答复；目录非正文，无关不读，不扩权。\n"
)

_DEFAULT_CATALOG_MAX_TOKENS = 2_048


class SkillContextProvider:

    # 函数说明：SkillContextProvider.__init__
    # 用途：初始化 SkillContextProvider；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   max_tokens：Token 数量或 Token 预算，类型 `int`。
    #   max_active：允许同时活跃的数量上限，类型 `int`。
    #   catalog_max_tokens：Token 数量或 Token 预算，类型 `int`；默认
    # `_DEFAULT_CATALOG_MAX_TOKENS`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`default_token_estimator`。
    # 副作用与资源：
    #   更新对象字段：`self.max_tokens`、`self.max_active`、`self.catalog_max_tokens`、
    # `self._estimator`。
    def __init__(
        self,
        *,
        max_tokens: int,
        max_active: int,
        catalog_max_tokens: int = _DEFAULT_CATALOG_MAX_TOKENS,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_active = max_active
        self.catalog_max_tokens = catalog_max_tokens
        self._estimator = default_token_estimator()


    # 函数说明：SkillContextProvider.render_catalog
    # 用途：生成展示文本`catalog`，供技能发现与激活使用。
    # 参数：
    #   metadata：关联元数据，类型 `Sequence[SkillMetadata]`。
    # 返回：类型 `str`；按分支返回 `_CATALOG_HEADER + '(No skills available.)\n'`；
    # `'\n'.join(lines).rstrip() + '\n'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_CATALOG_HEADER.rstrip` →
    # `self._estimator.estimate_text` → `'\n'.join(lines).rstrip`。
    # 分支与异常：
    #   当 `not metadata` 时，返回 `_CATALOG_HEADER + '(No skills available.)\n'`。
    #   当 `self._estimator.estimate_text('\n'.join(candidate)) >…` 时，结束当前循环。
    def render_catalog(self, metadata: Sequence[SkillMetadata]) -> str:

        if not metadata:
            return _CATALOG_HEADER + "(No skills available.)\n"
        lines = [_CATALOG_HEADER.rstrip()]
        shown = 0
        for item in metadata:
            candidate = lines + [f"[{item.name}] {item.description}"]
            if (
                self._estimator.estimate_text("\n".join(candidate))
                > self.catalog_max_tokens
            ):
                break
            lines = candidate
            shown += 1
        hidden = len(metadata) - shown
        if hidden > 0:
            lines.append(f"... {hidden} additional skills are not shown.")
        return "\n".join(lines).rstrip() + "\n"

    # 函数说明：SkillContextProvider.catalog_message
    # 用途：在技能发现与激活中处理 `catalog_message`，通过 `self.render_catalog` 完成首
    # 个内部处理步骤。
    # 参数：
    #   metadata：关联元数据，类型 `Sequence[SkillMetadata]`。
    # 返回：类型 `Message | None`；返回
    # `Message(role=MessageRole.SYSTEM, name=SKILL_CATALOG_MESSAGE_NAME, content=text)`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.render_catalog` → `Message`
    # 。
    def catalog_message(
        self,
        metadata: Sequence[SkillMetadata],
    ) -> Message | None:
        text = self.render_catalog(metadata)
        return Message(
            role=MessageRole.SYSTEM,
            name=SKILL_CATALOG_MESSAGE_NAME,
            content=text,
        )

    # 函数说明：SkillContextProvider.catalog_tokens
    # 用途：返回 `self._estimator.estimate_text(self.render_catalog(metadata))`，提供
    # SkillContextProvider 的派生值。
    # 参数：
    #   metadata：关联元数据，类型 `Sequence[SkillMetadata]`。
    # 返回：类型 `int`；返回
    # `self._estimator.estimate_text(self.render_catalog(metadata))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._estimator.estimate_text` →
    # `self.render_catalog`。
    def catalog_tokens(self, metadata: Sequence[SkillMetadata]) -> int:
        return self._estimator.estimate_text(self.render_catalog(metadata))


    # 函数说明：SkillContextProvider.active_messages
    # 用途：在技能发现与激活中处理 `active_messages`，通过 `seen.add` 完成首个内部处理步
    # 骤。
    # 参数：
    #   skills：技能集合输入或配置值，类型 `Sequence[Skill]`。
    # 返回：类型 `tuple[Message, ...]`；返回 `tuple(messages)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`seen.add` → `Message` →
    # `skill.render_instructions`。
    # 分支与异常：
    #   当 `skill.metadata.name in seen` 时，跳过当前循环项。
    def active_messages(
        self,
        skills: Sequence[Skill],
    ) -> tuple[Message, ...]:

        seen: set[str] = set()
        messages: list[Message] = []
        for skill in skills:
            if skill.metadata.name in seen:
                continue
            seen.add(skill.metadata.name)
            messages.append(
                Message(
                    role=MessageRole.SYSTEM,
                    name=ACTIVE_SKILL_MESSAGE_NAME,
                    content=skill.render_instructions(),
                )
            )
        return tuple(messages)

    # 函数说明：SkillContextProvider.active_tokens
    # 用途：在技能发现与激活中处理 `active_tokens`，通过 `self.active_messages` 完成首个
    # 内部处理步骤。
    # 参数：
    #   skills：传给 `self.active_messages` 的输入，类型 `Sequence[Skill]`。
    # 返回：类型 `int`；返回 `total`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.active_messages` →
    # `self._estimator.estimate_text`。
    def active_tokens(self, skills: Sequence[Skill]) -> int:
        total = 0
        for message in self.active_messages(skills):
            total += self._estimator.estimate_text(message.content or "")
        return total

    # 函数说明：SkillContextProvider.would_exceed_budget
    # 用途：在技能发现与激活中处理 `would_exceed_budget`，通过 `current_skills.append`
    # 完成首个内部处理步骤。
    # 参数：
    #   current：当前值或状态，类型 `Sequence[Skill]`。
    #   candidate：候选记录，类型 `Skill`。
    # 返回：类型 `bool`；按分支返回 `True`；`False`；
    # `self.active_tokens(current_skills) > self.max_tokens`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.active_tokens`。
    # 分支与异常：
    #   当 `len(current) >= self.max_active` 时，返回 `True`。
    #   当 `any(…)` 时，返回 `False`。
    def would_exceed_budget(
        self,
        current: Sequence[Skill],
        candidate: Skill,
    ) -> bool:

        if len(current) >= self.max_active:
            return True
        current_skills = [skill for skill in current]
        if any(
            skill.metadata.name == candidate.metadata.name for skill in current_skills
        ):
            return False
        current_skills.append(candidate)
        return self.active_tokens(current_skills) > self.max_tokens


__all__ = [
    "ACTIVE_SKILL_MESSAGE_NAME",
    "SKILL_CATALOG_MESSAGE_NAME",
    "SkillContextProvider",
]
