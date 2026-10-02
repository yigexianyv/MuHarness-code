from __future__ import annotations

from typing import Any

import pytest

from app.cli_ui import print_banner, print_startup_status, run_setup
from app.model_settings import ModelSettingsUpdate, ProviderSettingsUpdate
from app.models.config import ModelSettings
from app.models.types import ModelProvider


class StubModelSettingsService:
    # 函数说明：StubModelSettingsService.__init__
    # 用途：初始化 StubModelSettingsService；参数及实际保存的实例字段见下方说明。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.saved`、`self.tested`。
    def __init__(self) -> None:
        self.saved: ModelSettingsUpdate | None = None
        self.tested: ProviderSettingsUpdate | None = None

    # 函数说明：StubModelSettingsService.view
    # 用途：在回归测试与测试辅助中处理 `view`，通过 `providers.append` 完成首个内部处理
    # 步骤。
    # 参数：
    #   **_：额外关键字参数，按实现处理或转交。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `default_provider`、`providers`、
    # `reflection`、`maintenance`、`summary`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettings`。
    def view(self, **_: str) -> dict[str, Any]:
        settings = ModelSettings()
        providers = []
        for provider in ModelProvider:
            prefix = provider.value
            providers.append(
                {
                    "provider": provider.value,
                    "model": getattr(settings, f"{prefix}_model"),
                    "base_url": getattr(settings, f"{prefix}_base_url"),
                    "api_style": (
                        "anthropic_messages"
                        if provider is ModelProvider.ANTHROPIC
                        else getattr(settings, f"{prefix}_api_style").value
                    ),
                    "configured": False,
                }
            )
        role = {
            "enabled": True,
            "inherit_main": True,
            "provider": None,
            "model": None,
        }
        return {
            "default_provider": "openai",
            "providers": providers,
            "reflection": role,
            "maintenance": role,
            "summary": role,
        }

    # 函数说明：StubModelSettingsService.save
    # 用途：保存StubModelSettingsService，供回归测试与测试辅助使用。
    # 参数：
    #   update：`update`输入或配置值，类型 `ModelSettingsUpdate`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self.saved`。
    def save(self, update: ModelSettingsUpdate) -> None:
        self.saved = update

    # 函数说明：StubModelSettingsService.test
    # 用途：处理回归测试与测试辅助中的 `test` 数据；结果及边界条件见下方说明。
    # 参数：
    #   item：当前集合元素，类型 `ProviderSettingsUpdate`。
    # 返回：类型 `dict[str, Any]`；字典，包含字段 `provider`、`model`、`duration_ms`。
    # 副作用与资源：
    #   更新对象字段：`self.tested`。
    async def test(self, item: ProviderSettingsUpdate) -> dict[str, Any]:
        self.tested = item
        return {
            "provider": item.provider.value,
            "model": item.model,
            "duration_ms": 12.0,
        }


# 函数说明：test_cli_banner_and_status_are_compact
# 用途：回归验证回归测试与测试辅助中的 `cli_banner_and_status_are_compact` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   capsys：pytest 提供的标准输出捕获夹具，类型 `pytest.CaptureFixture[str]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`print_banner` →
# `print_startup_status` → `capsys.readouterr`。
# 分支与异常：
#   验证条件：`'MuHarness CLI' in output`。
#   验证条件：`'deepseek/deepseek-chat' in output`。
#   验证条件：`'存在 MCP 启动失败' in output`。
#   验证条件：`'/help 查看命令' in output`。
def test_cli_banner_and_status_are_compact(capsys: pytest.CaptureFixture[str]) -> None:
    print_banner()
    print_startup_status(
        (("主模型", "deepseek/deepseek-chat"), ("会话", "已创建 abc123")),
        notices=("存在 MCP 启动失败",),
    )

    output = capsys.readouterr().out
    assert "MuHarness CLI" in output
    assert "deepseek/deepseek-chat" in output
    assert "存在 MCP 启动失败" in output
    assert "/help 查看命令" in output


# 函数说明：test_setup_saves_selected_provider_without_echoing_secret
# 用途：回归验证回归测试与测试辅助中的
# `setup_saves_selected_provider_without_echoing_secret` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   capsys：pytest 提供的标准输出捕获夹具，类型 `pytest.CaptureFixture[str]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StubModelSettingsService` → `iter` →
# `run_setup` → `next` → `capsys.readouterr`。
# 分支与异常：
#   验证条件：`should_start is False`。
#   验证条件：`service.saved is not None`。
#   验证条件：`service.saved.default_provider is ModelProvider.DEEPSEEK`。
#   验证条件：`selected.api_key == 'sk-private-test'`。
@pytest.mark.asyncio
async def test_setup_saves_selected_provider_without_echoing_secret(
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = StubModelSettingsService()
    answers = iter(
        (
            "3",  
            "deepseek-chat",
            "",
            "",  
            "n",  
            "n",  
        )
    )
    messages: list[str] = []

    should_start = await run_setup(
        service=service,  
        input_fn=lambda _: next(answers),
        secret_fn=lambda _: "sk-private-test",
        output_fn=messages.append,
    )

    assert should_start is False
    assert service.saved is not None
    assert service.saved.default_provider is ModelProvider.DEEPSEEK
    selected = next(
        item
        for item in service.saved.providers
        if item.provider is ModelProvider.DEEPSEEK
    )
    assert selected.api_key == "sk-private-test"
    assert "sk-private-test" not in "\n".join(messages)
    assert "sk-private-test" not in capsys.readouterr().out
