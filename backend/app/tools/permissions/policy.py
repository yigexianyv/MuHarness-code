
from __future__ import annotations

from collections.abc import Sequence

from .matchers import build_matcher
from .models import ApprovalScope, PermissionEffect, PermissionRule, PermissionVerdict
from .store import PermissionRuleStore


class PermissionPolicyEngine:

    # 函数说明：PermissionPolicyEngine.__init__
    # 用途：初始化 PermissionPolicyEngine；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   store：持久化存储依赖，类型 `PermissionRuleStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._store`。
    def __init__(self, store: PermissionRuleStore) -> None:
        self._store = store

    # 函数说明：PermissionPolicyEngine.evaluate
    # 用途：按规则优先级评估调用，返回匹配的允许或拒绝决策。
    # 参数：
    #   tool_name：工具名称，类型 `str`。
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
    #   scope_ids：传给 `tuple` 的输入，类型 `Sequence[str]`。
    # 返回：类型 `PermissionVerdict`；按分支返回
    # `PermissionVerdict(effect=selected.effect, rule_id=selected.id, rule=selected)`；
    # `PermissionVerdict(effect=PermissionEffect.ASK)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._store.list` →
    # `build_matcher` → `matcher.matches` → `PermissionVerdict`。
    # 分支与异常：
    #   当 `rule.tool_name != tool_name` 时，跳过当前循环项。
    #   捕获 `ValueError` 后，跳过当前循环项，继续处理后续项。
    #   捕获 `Exception` 后，跳过当前循环项，继续处理后续项。
    #   `matched_rules` 分支在完成前置处理后返回 `PermissionVerdict(…)`。
    async def evaluate(
        self,
        *,
        tool_name: str,
        arguments: dict,
        scope_ids: Sequence[str],
    ) -> PermissionVerdict:

        """按规则优先级评估调用，返回匹配的允许或拒绝决策。"""
        matched_rules: list[PermissionRule] = []
        rules = await self._store.list(scope_ids=tuple(scope_ids))
        for rule in rules:
            if rule.tool_name != tool_name:
                continue
            try:
                matcher = build_matcher(rule.matcher_type, rule.matcher)
            except ValueError:
                continue
            try:
                matched = matcher.matches(arguments)
            except Exception:
                continue
            if matched:
                matched_rules.append(rule)
        if matched_rules:
            selected = max(matched_rules, key=_rule_priority)
            return PermissionVerdict(
                effect=selected.effect,
                rule_id=selected.id,
                rule=selected,
            )
        return PermissionVerdict(effect=PermissionEffect.ASK)

    # 函数说明：PermissionPolicyEngine.store
    # 用途：保存PermissionPolicyEngine，供工具权限规则匹配使用。
    # 返回：类型 `PermissionRuleStore`；返回 `self._store`。
    @property
    def store(self) -> PermissionRuleStore:

        return self._store


# 函数说明：_rule_priority
# 用途：在工具权限规则匹配中处理 `_rule_priority`，通过 `rule.created_at.timestamp` 完成
# 首个内部处理步骤。
# 参数：
#   rule：待匹配的权限规则，类型 `PermissionRule`。
# 返回：类型 `tuple[int, int, float]`；返回
# `(effect_priority, scope_priority, rule.created_at.timestamp())`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`rule.created_at.timestamp`。
def _rule_priority(rule: PermissionRule) -> tuple[int, int, float]:

    effect_priority = {
        PermissionEffect.ALLOW: 1,
        PermissionEffect.ASK: 2,
        PermissionEffect.DENY: 3,
    }[rule.effect]
    scope_priority = 2 if rule.scope is ApprovalScope.RUN else 1
    return effect_priority, scope_priority, rule.created_at.timestamp()


__all__ = ["PermissionPolicyEngine"]
