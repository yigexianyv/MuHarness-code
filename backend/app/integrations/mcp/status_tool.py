
from __future__ import annotations

from typing import Any

from app.models.types import ToolDefinition, ToolPermission
from app.tools.base import BaseTool

from .manager import MCPClientManager

MCP_STATUS_TOOL_NAME = "mcp_status"


class MCPStatusTool(BaseTool):

    definition = ToolDefinition(
        name=MCP_STATUS_TOOL_NAME,
        record_output=False,
        description=(
            "只读查看已配置 MCP 服务的当前连接状态、错误及已注册工具名，无需审批。"
            "需要确认服务是否运行或排查接入故障时使用；不启动、重连或调用远端工具。"
            "发现尚未可见的具体能力用 tool_search，普通任务已有可用工具时不必先查状态。"
            "成功只表示读到了本地管理器的状态记录，不保证后续远端调用成功或任务完成。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "可选，已配置服务的精确名称；省略时查看全部服务，不填远端工具名。",
                },
                "include_tools": {
                    "type": "boolean",
                    "description": (
                        "是否列出已注册的工具名，默认 true；列出名称不会激活或执行这些工具。"
                    ),
                    "default": True,
                },
            },
            "additionalProperties": False,
        },
        permission=ToolPermission.ALLOWED,
    )

    # 函数说明：MCPStatusTool.__init__
    # 用途：初始化 MCPStatusTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   manager：当前业务管理器，类型 `MCPClientManager | None`。
    #   configuration_error：错误输入或配置值，类型 `str | None`；默认 `None`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._manager`、`self._configuration_error`。
    def __init__(
        self,
        manager: MCPClientManager | None,
        *,
        configuration_error: str | None = None,
    ) -> None:
        self._manager = manager
        self._configuration_error = configuration_error

    # 函数说明：MCPStatusTool.execute
    # 用途：执行MCPStatusTool，供MCP 连接与外部工具适配使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `server`、`include_tools`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `server_count`、`running_count`、
    # `configuration_error`、`servers`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._manager.statuses`。
    # 分支与异常：
    #   当 `server is not None and (not isinstance(server, str) or not…` 时，抛出
    # `TypeError('server 必须是非空字符串')`。
    #   当 `not isinstance(include_tools, bool)` 时，抛出
    # `TypeError('include_tools 必须是布尔值')`。
    #   当 `not statuses` 时，抛出 `ValueError(f"MCP Server '{normalized}' 不存在")`。
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        server = arguments.get("server")
        if server is not None and (
            not isinstance(server, str) or not server.strip()
        ):
            raise TypeError("server 必须是非空字符串")
        include_tools = arguments.get("include_tools", True)
        if not isinstance(include_tools, bool):
            raise TypeError("include_tools 必须是布尔值")

        statuses = self._manager.statuses() if self._manager is not None else ()
        if server is not None:
            normalized = server.strip()
            statuses = tuple(item for item in statuses if item.name == normalized)
            if not statuses:
                raise ValueError(f"MCP Server '{normalized}' 不存在")

        entries: list[dict[str, Any]] = []
        for status in statuses:
            item: dict[str, Any] = {
                "name": status.name,
                "state": status.state.value,
                "tool_count": len(status.tool_names),
            }
            if include_tools:
                item["tools"] = list(status.tool_names)
            if status.error:
                item["error"] = status.error
            entries.append(item)

        return {
            "server_count": len(entries),
            "running_count": sum(
                1 for item in entries if item["state"] == "running"
            ),
            "configuration_error": self._configuration_error,
            "servers": entries,
        }


__all__ = ["MCP_STATUS_TOOL_NAME", "MCPStatusTool"]
