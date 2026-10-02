
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class PermissionMatcher(ABC):

    # 函数说明：PermissionMatcher.matches
    # 用途：匹配PermissionMatcher，供工具权限规则匹配使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `bool`；不返回结果值（隐式 None）。
    @abstractmethod
    def matches(self, arguments: dict[str, Any]) -> bool:
        pass


# 函数说明：_canonical
# 用途：返回 `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'
# ), default=str)`，提供 工具权限规则匹配 的派生值。
# 参数：
#   value：待校验、规范化或转换的值，类型 `Any`。
# 返回：类型 `str`；返回 `json.dumps(value, ensure_ascii=False, sort_keys=True,
# separators=(',', ':'), default=str)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class ExactArgumentsMatcher(PermissionMatcher):

    # 函数说明：ExactArgumentsMatcher.__init__
    # 用途：初始化 ExactArgumentsMatcher；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_canonical`。
    # 副作用与资源：
    #   更新对象字段：`self._expected`。
    def __init__(self, arguments: dict[str, Any]) -> None:
        self._expected = _canonical(arguments or {})

    # 函数说明：ExactArgumentsMatcher.matches
    # 用途：匹配ExactArgumentsMatcher，供工具权限规则匹配使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `bool`；返回 `self._expected == _canonical(arguments or {})`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_canonical`。
    def matches(self, arguments: dict[str, Any]) -> bool:
        return self._expected == _canonical(arguments or {})


_MATCHERS: dict[str, type[PermissionMatcher]] = {
    "exact_arguments": ExactArgumentsMatcher,
}


# 函数说明：build_matcher
# 用途：构建`matcher`，供工具权限规则匹配使用。
# 参数：
#   matcher_type：传给 `_MATCHERS.get` 的输入，类型 `str`。
#   payload：传输或持久化载荷，类型 `dict[str, Any]`。
# 返回：类型 `PermissionMatcher`；返回 `matcher_cls(**payload)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`matcher_cls`。
# 分支与异常：
#   当 `matcher_cls is None` 时，抛出
# `ValueError(f'Unknown matcher type: {matcher_type}')`。
def build_matcher(
    matcher_type: str,
    payload: dict[str, Any],
) -> PermissionMatcher:

    matcher_cls = _MATCHERS.get(matcher_type)
    if matcher_cls is None:
        raise ValueError(f"Unknown matcher type: {matcher_type}")
    return matcher_cls(**payload)


__all__ = [
    "ExactArgumentsMatcher",
    "PermissionMatcher",
    "build_matcher",
]
