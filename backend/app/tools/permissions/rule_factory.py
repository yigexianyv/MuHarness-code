
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..approval import ApprovalRequest
from .models import ApprovalScope, PermissionEffect, PermissionRule


# 函数说明：build_safe_rule
# 用途：构建规则，供工具权限规则匹配使用。
# 参数：
#   tool_name：工具名称，类型 `str`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
#   scope：记忆、规则或查询作用域，类型 `ApprovalScope`。
#   scope_id：作用域标识，类型 `str`。
#   effect：`effect`输入或配置值，类型 `PermissionEffect`；默认 `PermissionEffect.ALLOW`
# 。
# 返回：类型 `PermissionRule`；返回 `PermissionRule(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`PermissionRule` → `uuid4` →
# `datetime.now`。
def build_safe_rule(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    scope: ApprovalScope,
    scope_id: str,
    effect: PermissionEffect = PermissionEffect.ALLOW,
) -> PermissionRule:

    return PermissionRule(
        id=uuid4().hex,
        tool_name=tool_name,
        scope=scope,
        scope_id=scope_id,
        effect=effect,
        matcher_type="exact_arguments",
        matcher={"arguments": arguments},
        description=f"允许调用 {tool_name}（完整参数相同）",
        created_at=datetime.now(UTC),
    )


# 函数说明：describe_safe_rule
# 用途：返回 `'当前会话记住该操作（仅完整参数相同时自动通过）'`，提供 工具权限规则匹配
# 的派生值。
# 参数：
#   request：待处理的请求对象，类型 `ApprovalRequest`。
# 返回：类型 `str`；返回 `'当前会话记住该操作（仅完整参数相同时自动通过）'`。
def describe_safe_rule(request: ApprovalRequest) -> str:

    return "当前会话记住该操作（仅完整参数相同时自动通过）"


__all__ = ["build_safe_rule", "describe_safe_rule"]
