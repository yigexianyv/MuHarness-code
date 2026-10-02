
from __future__ import annotations

from typing import Any

from app.models.types import ToolDefinition, ToolPermission
from app.tools.base import BaseTool

from .client import MCPClientProtocol
from .models import MCPRemoteTool


class MCPToolAdapter(BaseTool):

    # 函数说明：MCPToolAdapter.__init__
    # 用途：初始化 MCPToolAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   server_name：服务名称输入或配置值，类型 `str`。
    #   registered_name：名称输入或配置值，类型 `str`。
    #   remote_tool：工具输入或配置值，类型 `MCPRemoteTool`。
    #   client：模型、HTTP 或 MCP 客户端，类型 `MCPClientProtocol`。
    #   permission：所需权限等级，类型 `ToolPermission`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    # 副作用与资源：
    #   更新对象字段：`self.server_name`、`self.remote_name`、`self._client`、
    # `self._definition`。
    def __init__(
        self,
        *,
        server_name: str,
        registered_name: str,
        remote_tool: MCPRemoteTool,
        client: MCPClientProtocol,
        permission: ToolPermission,
    ) -> None:
        self.server_name = server_name
        self.remote_name = remote_tool.name
        self._client = client
        self._definition = ToolDefinition(
            name=registered_name,
            description=(
                f"[MCP: {server_name}] {remote_tool.description}"
                if remote_tool.description
                else f"[MCP: {server_name}] 未提供用途说明。"
            ),
            parameters=remote_tool.input_schema,
            permission=permission,
        )

    # 函数说明：MCPToolAdapter.definition
    # 用途：提供 MCPToolAdapter 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `self._definition`。
    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    # 函数说明：MCPToolAdapter.execute
    # 用途：执行MCPToolAdapter，供MCP 连接与外部工具适配使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `str`；返回 `await self._client.call_tool(self.remote_name, arguments)`
    # 。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._client.call_tool`。
    async def execute(self, arguments: dict[str, Any]) -> str:
        return await self._client.call_tool(self.remote_name, arguments)


__all__ = ["MCPToolAdapter"]
