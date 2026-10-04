
from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

server = FastMCP("muharness-test")


# 函数说明：echo
# 用途：返回 `text`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `str`；返回 `text`。
@server.tool()
def echo(text: str) -> str:

    return text


# 函数说明：slow
# 用途：在回归测试与测试辅助中处理 `slow`，通过 `asyncio.sleep` 完成首个内部处理步骤。
# 参数：
#   delay：传给 `asyncio.sleep` 的输入，类型 `float`。
# 返回：类型 `str`；返回 `'done'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`asyncio.sleep`。
@server.tool()
async def slow(delay: float) -> str:

    await asyncio.sleep(delay)
    return "done"


# 函数说明：fail
# 用途：记录失败回归测试与测试辅助，供回归测试与测试辅助使用。
# 返回：类型 `str`；不返回结果值（隐式 None）。
@server.tool()
def fail() -> str:

    raise RuntimeError("fake boom")


if __name__ == "__main__":
    server.run(transport="stdio")
