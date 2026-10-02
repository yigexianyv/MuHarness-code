
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from app.domain.conversation import DEFAULT_DATABASE_PATH

from .models import ApprovalScope, PermissionRule

_SCHEMA = """
CREATE TABLE IF NOT EXISTS permission_rules (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    effect TEXT NOT NULL,
    matcher_type TEXT NOT NULL,
    matcher_json TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_permission_rules_scope
ON permission_rules(scope_id);
"""


class PermissionRuleStore(ABC):

    # 函数说明：PermissionRuleStore.add
    # 用途：添加PermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   rule：待匹配的权限规则，类型 `PermissionRule`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    @abstractmethod
    async def add(self, rule: PermissionRule) -> None:
        pass

    # 函数说明：PermissionRuleStore.list
    # 用途：列出PermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   scope_ids：作用域输入或配置值，类型 `tuple[str, ...] | None`；默认 `None`。
    # 返回：类型 `tuple[PermissionRule, ...]`；不返回结果值（隐式 None）。
    @abstractmethod
    async def list(
        self,
        *,
        scope_ids: tuple[str, ...] | None = None,
    ) -> tuple[PermissionRule, ...]:
        pass

    # 函数说明：PermissionRuleStore.get
    # 用途：获取PermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   rule_id：权限规则标识，类型 `str`。
    # 返回：类型 `PermissionRule | None`；不返回结果值（隐式 None）。
    @abstractmethod
    async def get(self, rule_id: str) -> PermissionRule | None:
        pass

    # 函数说明：PermissionRuleStore.remove
    # 用途：移除PermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   rule_id：权限规则标识，类型 `str`。
    # 返回：类型 `bool`；不返回结果值（隐式 None）。
    @abstractmethod
    async def remove(self, rule_id: str) -> bool:
        pass

    # 函数说明：PermissionRuleStore.remove_scope
    # 用途：移除作用域，供工具权限规则匹配使用。
    # 参数：
    #   scope：记忆、规则或查询作用域，类型 `ApprovalScope`。
    #   scope_id：作用域标识，类型 `str`。
    # 返回：类型 `int`；不返回结果值（隐式 None）。
    @abstractmethod
    async def remove_scope(
        self,
        scope: ApprovalScope,
        scope_id: str,
    ) -> int:
        pass


class InMemoryPermissionRuleStore(PermissionRuleStore):

    # 函数说明：InMemoryPermissionRuleStore.__init__
    # 用途：初始化 InMemoryPermissionRuleStore；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._rules`。
    def __init__(self) -> None:
        self._rules: list[PermissionRule] = []

    # 函数说明：InMemoryPermissionRuleStore.add
    # 用途：添加InMemoryPermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   rule：待匹配的权限规则，类型 `PermissionRule`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def add(self, rule: PermissionRule) -> None:
        self._rules.append(rule)

    # 函数说明：InMemoryPermissionRuleStore.list
    # 用途：列出InMemoryPermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   scope_ids：传给 `set` 的输入，类型 `tuple[str, ...] | None`；默认 `None`。
    # 返回：类型 `tuple[PermissionRule, ...]`；按分支返回 `tuple(self._rules)`；
    # `tuple((rule for rule in reversed(self._rules) if rule.scope_id in allowed))`。
    # 分支与异常：
    #   当 `scope_ids is None` 时，返回 `tuple(self._rules)`。
    async def list(
        self,
        *,
        scope_ids: tuple[str, ...] | None = None,
    ) -> tuple[PermissionRule, ...]:
        if scope_ids is None:
            return tuple(self._rules)
        allowed = set(scope_ids)
        return tuple(
            rule
            for rule in reversed(self._rules)
            if rule.scope_id in allowed
        )

    # 函数说明：InMemoryPermissionRuleStore.get
    # 用途：获取InMemoryPermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   rule_id：权限规则标识，类型 `str`。
    # 返回：类型 `PermissionRule | None`；按分支返回 `rule`；`None`。
    # 分支与异常：
    #   当 `rule.id == rule_id` 时，返回 `rule`。
    async def get(self, rule_id: str) -> PermissionRule | None:
        for rule in self._rules:
            if rule.id == rule_id:
                return rule
        return None

    # 函数说明：InMemoryPermissionRuleStore.remove
    # 用途：移除InMemoryPermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   rule_id：权限规则标识，类型 `str`。
    # 返回：类型 `bool`；按分支返回 `True`；`False`。
    # 分支与异常：
    #   `rule.id == rule_id` 分支在完成前置处理后返回 `True`。
    async def remove(self, rule_id: str) -> bool:
        for index, rule in enumerate(self._rules):
            if rule.id == rule_id:
                del self._rules[index]
                return True
        return False

    # 函数说明：InMemoryPermissionRuleStore.remove_scope
    # 用途：移除作用域，供工具权限规则匹配使用。
    # 参数：
    #   scope：记忆、规则或查询作用域，类型 `ApprovalScope`。
    #   scope_id：作用域标识，类型 `str`。
    # 返回：类型 `int`；返回 `removed`。
    # 副作用与资源：
    #   更新对象字段：`self._rules`。
    async def remove_scope(self, scope: ApprovalScope, scope_id: str) -> int:
        retained = [
            rule
            for rule in self._rules
            if not (rule.scope is scope and rule.scope_id == scope_id)
        ]
        removed = len(self._rules) - len(retained)
        self._rules = retained
        return removed


class SQLitePermissionRuleStore(PermissionRuleStore):

    # 函数说明：SQLitePermissionRuleStore.__init__
    # 用途：初始化 SQLitePermissionRuleStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   database_path：SQLite 数据库路径，类型 `str | Path`；默认
    # `DEFAULT_DATABASE_PATH`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `Path(database_path).expanduser().resolve` → `Path(database_path).expanduser` →
    # `Path`。
    # 副作用与资源：
    #   更新对象字段：`self.database_path`。
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    # 函数说明：SQLitePermissionRuleStore.initialize
    # 用途：初始化SQLitePermissionRuleStore，供工具权限规则匹配使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.database_path.parent.mkdir`
    # → `self._connect` → `database.executescript` → `database.execute` →
    # `database.commit`。
    # 副作用与资源：
    #   数据库操作：DELETE permission_rules、UPDATE permission_rules；连接与事务边界以
    # with/提交语句为准。
    #   文件或资源访问：`self.database_path.parent.mkdir`。
    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.execute(
                "DELETE FROM permission_rules "
                "WHERE matcher_type IN ('command_prefix', 'command_contains', "
                "'host_exact')"
            )
            await database.execute(
                "UPDATE permission_rules SET scope = 'conversation' "
                "WHERE scope = 'project'"
            )
            await database.commit()

    # 函数说明：SQLitePermissionRuleStore.add
    # 用途：添加SQLitePermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   rule：待匹配的权限规则，类型 `PermissionRule`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `json.dumps` → `rule.created_at.isoformat` →
    # `database.commit`。
    # 副作用与资源：
    #   数据库操作：INSERT permission_rules；连接与事务边界以 with/提交语句为准。
    async def add(self, rule: PermissionRule) -> None:
        async with self._connect() as database:
            await database.execute(
                """
                INSERT INTO permission_rules (
                    id, tool_name, scope, scope_id, effect,
                    matcher_type, matcher_json, description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.id,
                    rule.tool_name,
                    rule.scope.value,
                    rule.scope_id,
                    rule.effect.value,
                    rule.matcher_type,
                    json.dumps(
                        rule.matcher,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    rule.description,
                    rule.created_at.isoformat(),
                ),
            )
            await database.commit()

    # 函数说明：SQLitePermissionRuleStore.list
    # 用途：列出SQLitePermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   scope_ids：作用域输入或配置值，类型 `tuple[str, ...] | None`；默认 `None`。
    # 返回：类型 `tuple[PermissionRule, ...]`；按分支返回 `()`；
    # `tuple((_rule_from_row(row) for row in rows))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchall` → `_rule_from_row`。
    # 分支与异常：
    #   当 `not scope_ids` 时，返回 `()`。
    async def list(
        self,
        *,
        scope_ids: tuple[str, ...] | None = None,
    ) -> tuple[PermissionRule, ...]:
        if scope_ids is None:
            query = "SELECT * FROM permission_rules"
            parameters: tuple[Any, ...] = ()
        elif not scope_ids:
            return ()
        else:
            placeholders = ", ".join("?" for _ in scope_ids)
            query = (
                "SELECT * FROM permission_rules "
                f"WHERE scope_id IN ({placeholders})"
            )
            parameters = scope_ids
        query += " ORDER BY created_at DESC, id DESC"
        async with self._connect() as database:
            cursor = await database.execute(query, parameters)
            rows = await cursor.fetchall()
        return tuple(_rule_from_row(row) for row in rows)

    # 函数说明：SQLitePermissionRuleStore.get
    # 用途：获取SQLitePermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   rule_id：权限规则标识，类型 `str`。
    # 返回：类型 `PermissionRule | None`；返回
    # `_rule_from_row(row) if row is not None else None`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `cursor.fetchone` → `_rule_from_row`。
    # 副作用与资源：
    #   数据库操作：SELECT permission_rules；连接与事务边界以 with/提交语句为准。
    async def get(self, rule_id: str) -> PermissionRule | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM permission_rules WHERE id = ?",
                (rule_id,),
            )
            row = await cursor.fetchone()
        return _rule_from_row(row) if row is not None else None

    # 函数说明：SQLitePermissionRuleStore.remove
    # 用途：移除SQLitePermissionRuleStore，供工具权限规则匹配使用。
    # 参数：
    #   rule_id：权限规则标识，类型 `str`。
    # 返回：类型 `bool`；返回 `cursor.rowcount > 0`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：DELETE permission_rules；连接与事务边界以 with/提交语句为准。
    async def remove(self, rule_id: str) -> bool:
        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM permission_rules WHERE id = ?",
                (rule_id,),
            )
            await database.commit()
        return cursor.rowcount > 0

    # 函数说明：SQLitePermissionRuleStore.remove_scope
    # 用途：移除作用域，供工具权限规则匹配使用。
    # 参数：
    #   scope：记忆、规则或查询作用域，类型 `ApprovalScope`。
    #   scope_id：作用域标识，类型 `str`。
    # 返回：类型 `int`；返回 `max(cursor.rowcount, 0)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._connect` →
    # `database.execute` → `database.commit`。
    # 副作用与资源：
    #   数据库操作：DELETE permission_rules；连接与事务边界以 with/提交语句为准。
    async def remove_scope(self, scope: ApprovalScope, scope_id: str) -> int:
        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM permission_rules WHERE scope = ? AND scope_id = ?",
                (scope.value, scope_id),
            )
            await database.commit()
        return max(cursor.rowcount, 0)

    # 函数说明：SQLitePermissionRuleStore._connect
    # 用途：连接SQLitePermissionRuleStore，供工具权限规则匹配使用。
    # 返回：异步生成器，逐项产出 `database`；资源与结束处理遵循生成器流程。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`aiosqlite.connect` →
    # `database.close`。
    # 副作用与资源：
    #   更新对象字段：`database.row_factory`。
    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        try:
            yield database
        finally:
            await database.close()


# 函数说明：_rule_from_row
# 用途：将数据库行解析为工具权限规则。
# 参数：
#   row：SQLite 查询返回的一行数据，类型 `aiosqlite.Row`；读取键 `id`、`tool_name`、
# `scope`、`scope_id`、`effect`、`matcher_type`、`matcher_json`、`description`、
# `created_at`。
# 返回：类型 `PermissionRule`；返回 `PermissionRule(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`PermissionRule` → `json.loads` →
# `datetime.fromisoformat`。
def _rule_from_row(row: aiosqlite.Row) -> PermissionRule:
    return PermissionRule(
        id=row["id"],
        tool_name=row["tool_name"],
        scope=row["scope"],
        scope_id=row["scope_id"],
        effect=row["effect"],
        matcher_type=row["matcher_type"],
        matcher=json.loads(row["matcher_json"]),
        description=row["description"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


__all__ = [
    "InMemoryPermissionRuleStore",
    "PermissionRuleStore",
    "SQLitePermissionRuleStore",
]
