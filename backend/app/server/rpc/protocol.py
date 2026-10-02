



from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

JSONRPC_VERSION = "2.0"


class RpcErrorCode(IntEnum):

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


RESOURCE_NOT_FOUND = -32000
INVALID_STATE = -32001


class JsonRpcError(Exception):

    # 函数说明：JsonRpcError.__init__
    # 用途：初始化 JsonRpcError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   code：传给 `int` 的输入，类型 `int`。
    #   message：单条消息或通知，类型 `str`。
    #   data：待解析或写入的数据，类型 `Any`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.code`、`self.message`、`self.data`。
    def __init__(
        self,
        code: int,
        message: str,
        data: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = message
        self.data = data

    # 函数说明：JsonRpcError.to_body
    # 用途：处理JSON-RPC 连接与消息分发中的 `to_body` 数据；结果及边界条件见下方说明。
    # 返回：类型 `dict[str, Any]`；返回 `body`。
    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            body["data"] = self.data
        return body


@dataclass
class RpcRequest:

    id: str | int
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RpcNotification:

    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedMessage:

    request: RpcRequest | None = None
    notification: RpcNotification | None = None
    error: JsonRpcError | None = None
    id: str | int | None = None


# 函数说明：parse_message
# 用途：校验 JSON-RPC 消息结构，区分请求、响应与通知。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `ParsedMessage`；按分支返回
# `ParsedMessage(error=JsonRpcError(RpcErrorCode.PARSE_ERROR, 'Parse error'))`；
# `ParsedMessage(…)`；
# `ParsedMessage(notification=RpcNotification(method=method, params=params))`；`
# ParsedMessage(request=RpcRequest(id=raw_id, method=method, params=params), id=raw_id)`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.loads` → `ParsedMessage` →
# `_id_or_none` → `RpcNotification` → `RpcRequest`。
# 分支与异常：
#   捕获 `json.JSONDecodeError` 后，返回
# `ParsedMessage(error=JsonRpcError(RpcErrorCode.PARSE_ERROR, 'Parse error'))`。
#   当 `not isinstance(payload, dict)` 时，返回 `ParsedMessage(…)`。
#   当 `isinstance(raw_id, bool) or (raw_id is not None and (not…` 时，返回
# `ParsedMessage(…)`。
#   当 `payload.get('jsonrpc') != JSONRPC_VERSION` 时，返回 `ParsedMessage(…)`。
def parse_message(text: str) -> ParsedMessage:

    """校验 JSON-RPC 消息结构，区分请求、响应与通知。"""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ParsedMessage(
            error=JsonRpcError(RpcErrorCode.PARSE_ERROR, "Parse error")
        )

    if not isinstance(payload, dict):
        return ParsedMessage(
            error=JsonRpcError(
                RpcErrorCode.INVALID_REQUEST,
                "Invalid Request: message must be an object",
            )
        )

    raw_id = payload.get("id")
    if isinstance(raw_id, bool) or (
        raw_id is not None and not isinstance(raw_id, (str, int))
    ):
        return ParsedMessage(
            error=JsonRpcError(
                RpcErrorCode.INVALID_REQUEST,
                "Invalid Request: id must be a string or integer",
            )
        )
    request_id = _id_or_none(raw_id)

    if payload.get("jsonrpc") != JSONRPC_VERSION:
        return ParsedMessage(
            error=JsonRpcError(
                RpcErrorCode.INVALID_REQUEST,
                "Invalid Request: jsonrpc must be '2.0'",
            ),
            id=request_id,
        )

    method = payload.get("method")
    if not isinstance(method, str) or not method:
        return ParsedMessage(
            error=JsonRpcError(
                RpcErrorCode.INVALID_REQUEST,
                "Invalid Request: method is required",
            ),
            id=request_id,
        )

    params = payload.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return ParsedMessage(
            error=JsonRpcError(
                RpcErrorCode.INVALID_REQUEST,
                "Invalid Request: params must be an object",
            ),
            id=request_id,
        )

    if raw_id is None:
        return ParsedMessage(
            notification=RpcNotification(method=method, params=params)
        )
    return ParsedMessage(
        request=RpcRequest(id=raw_id, method=method, params=params),
        id=raw_id,
    )


# 函数说明：_id_or_none
# 用途：处理JSON-RPC 连接与消息分发中的 `_id_or_none` 数据；结果及边界条件见下方说明。
# 参数：
#   value：待校验、规范化或转换的值，类型 `object`。
# 返回：类型 `str | int | None`；按分支返回 `None`；`value`。
# 分支与异常：
#   当 `isinstance(value, bool)` 时，返回 `None`。
#   当 `isinstance(value, (str, int))` 时，返回 `value`。
def _id_or_none(value: object) -> str | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        return value
    return None


__all__ = [
    "INVALID_STATE",
    "JSONRPC_VERSION",
    "RESOURCE_NOT_FOUND",
    "JsonRpcError",
    "ParsedMessage",
    "RpcErrorCode",
    "RpcNotification",
    "RpcRequest",
    "parse_message",
]
