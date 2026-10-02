
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.model_settings import (
    ModelSettingsService,
    ModelSettingsStore,
    ModelSettingsUpdate,
    ProviderSettingsUpdate,
    load_effective_model_configuration,
)
from app.models.config import ModelSettings
from app.models.types import ApiStyle, ModelProvider


@dataclass
class FakeSecrets:
    values: dict[str, str] = field(default_factory=dict)
    reads: list[str] = field(default_factory=list)

    # 函数说明：FakeSecrets.get
    # 用途：获取FakeSecrets，供回归测试与测试辅助使用。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    # 返回：类型 `str | None`；返回 `self.values.get(provider)`。
    def get(self, provider: str) -> str | None:
        self.reads.append(provider)
        return self.values.get(provider)

    # 函数说明：FakeSecrets.set
    # 用途：设置FakeSecrets，供回归测试与测试辅助使用。
    # 参数：
    #   provider：模型或搜索服务商，类型 `str`。
    #   value：待校验、规范化或转换的值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    def set(self, provider: str, value: str) -> None:
        self.values[provider] = value


# 函数说明：_base_settings
# 用途：返回 `ModelSettings(…)`，提供 回归测试与测试辅助 的派生值。
# 返回：类型 `ModelSettings`；返回 `ModelSettings(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettings`。
def _base_settings() -> ModelSettings:
    return ModelSettings(
        _env_file=None,
        model_default_provider=ModelProvider.OPENAI,
        openai_api_key=None,
        qwen_api_key=None,
        deepseek_api_key=None,
        anthropic_api_key=None,
    )


# 函数说明：_update
# 用途：更新回归测试与测试辅助，供回归测试与测试辅助使用。
# 返回：类型 `ModelSettingsUpdate`；返回 `ModelSettingsUpdate(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettingsUpdate` →
# `ProviderSettingsUpdate`。
def _update() -> ModelSettingsUpdate:
    return ModelSettingsUpdate(
        default_provider=ModelProvider.QWEN,
        providers=(
            ProviderSettingsUpdate(
                provider=ModelProvider.OPENAI,
                model="gpt-test",
                api_style=ApiStyle.RESPONSES,
            ),
            ProviderSettingsUpdate(
                provider=ModelProvider.QWEN,
                model="qwen-test",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_style=ApiStyle.CHAT_COMPLETIONS,
                api_key="secret-qwen",
            ),
            ProviderSettingsUpdate(
                provider=ModelProvider.DEEPSEEK,
                model="deepseek-test",
                base_url="https://api.deepseek.com",
                api_style=ApiStyle.CHAT_COMPLETIONS,
            ),
            ProviderSettingsUpdate(
                provider=ModelProvider.ANTHROPIC,
                model="claude-test",
                api_style=ApiStyle.ANTHROPIC_MESSAGES,
            ),
        ),
        reflection={
            "enabled": True,
            "inherit_main": False,
            "provider": "qwen",
            "model": "qwen-small",
        },
        maintenance={
            "enabled": False,
            "inherit_main": False,
            "provider": "openai",
            "model": "gpt-small",
        },
        summary={
            "enabled": True,
            "inherit_main": False,
            "provider": "qwen",
            "model": "qwen-summary",
        },
    )


# 函数说明：test_save_keeps_api_key_out_of_json
# 用途：回归验证回归测试与测试辅助中的 `save_keeps_api_key_out_of_json` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeSecrets` → `ModelSettingsService`
#  → `ModelSettingsStore` → `_base_settings` → `service.save` → `_update`；另有 2 个调用
# 点。
# 分支与异常：
#   验证条件：`'secret-qwen' not in raw`。
#   验证条件：`'api_key' not in raw`。
#   验证条件：`secrets.values == {'qwen': 'secret-qwen'}`。
#   验证条件：`json.loads(raw)['default_provider'] == 'qwen'`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
def test_save_keeps_api_key_out_of_json(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    secrets = FakeSecrets()
    service = ModelSettingsService(
        store=ModelSettingsStore(path),
        secrets=secrets,
        base_settings=_base_settings(),
    )

    service.save(_update())

    raw = path.read_text(encoding="utf-8")
    assert "secret-qwen" not in raw
    assert "api_key" not in raw
    assert secrets.values == {"qwen": "secret-qwen"}
    assert json.loads(raw)["default_provider"] == "qwen"


# 函数说明：test_effective_configuration_merges_roles_and_secret_store
# 用途：回归验证回归测试与测试辅助中的
# `effective_configuration_merges_roles_and_secret_store` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettingsStore` → `FakeSecrets` →
#  `ModelSettingsService` → `_base_settings` → `service.save` → `_update`；另有 2 个调用
# 点。
# 分支与异常：
#   验证条件：`effective.settings.model_default_provider is ModelProvider.QWEN`。
#   验证条件：`effective.settings.qwen_model == 'qwen-test'`。
#   验证条件：`effective.settings.qwen_api_key == SecretStr('secret-qwen')`。
#   验证条件：`effective.reflection is not None`。
def test_effective_configuration_merges_roles_and_secret_store(tmp_path: Path) -> None:
    store = ModelSettingsStore(tmp_path / "models.json")
    secrets = FakeSecrets()
    service = ModelSettingsService(
        store=store,
        secrets=secrets,
        base_settings=_base_settings(),
    )
    service.save(_update())

    effective = load_effective_model_configuration(
        store=store,
        secrets=secrets,
        base_settings=_base_settings(),
    )

    assert effective.settings.model_default_provider is ModelProvider.QWEN
    assert effective.settings.qwen_model == "qwen-test"
    assert effective.settings.qwen_api_key == SecretStr("secret-qwen")
    assert effective.reflection is not None
    assert effective.reflection.provider == "qwen"
    assert effective.reflection.model == "qwen-small"
    assert effective.maintenance is not None
    assert effective.maintenance.enabled is False
    assert effective.summary is not None
    assert effective.summary.enabled is True
    assert effective.summary.provider == "qwen"
    assert effective.summary.model == "qwen-summary"


# 函数说明：test_view_compares_saved_and_current_model_roles
# 用途：回归验证回归测试与测试辅助中的 `view_compares_saved_and_current_model_roles` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettingsService` →
# `ModelSettingsStore` → `FakeSecrets` → `_base_settings` → `service.save` → `_update`；
# 另有 1 个调用点。
# 分支与异常：
#   验证条件：`current['restart_required'] is False`。
#   验证条件：`changed['restart_required'] is True`。
#   验证条件：`current['active_roles'] == active_roles`。
def test_view_compares_saved_and_current_model_roles(tmp_path: Path) -> None:
    service = ModelSettingsService(
        store=ModelSettingsStore(tmp_path / "models.json"),
        secrets=FakeSecrets(),
        base_settings=_base_settings(),
    )
    service.save(_update())
    active_roles = {
        "main": {"enabled": True, "provider": "qwen", "model": "qwen-test"},
        "summary": {
            "enabled": True,
            "provider": "qwen",
            "model": "qwen-summary",
        },
        "reflection": {
            "enabled": True,
            "provider": "qwen",
            "model": "qwen-small",
        },
        "maintenance": {
            "enabled": False,
            "provider": "openai",
            "model": "gpt-small",
        },
    }

    current = service.view(
        active_provider="qwen",
        active_model="qwen-test",
        active_roles=active_roles,
    )
    changed = service.view(
        active_provider="qwen",
        active_model="qwen-test",
        active_roles={
            **active_roles,
            "summary": {
                "enabled": True,
                "provider": "qwen",
                "model": "other-summary",
            },
        },
    )

    assert current["restart_required"] is False
    assert changed["restart_required"] is True
    assert current["active_roles"] == active_roles


# 函数说明：test_old_settings_without_summary_use_inherited_enabled_default
# 用途：回归验证回归测试与测试辅助中的
# `old_settings_without_summary_use_inherited_enabled_default` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `ModelSettingsService(store=ModelSettingsStore(path), secrets=FakeSecrets()…` →
# `ModelSettingsService` → `ModelSettingsStore` → `FakeSecrets` → `_base_settings` →
# `_update`；另有 6 个调用点。
# 分支与异常：
#   验证条件：`stored is not None`。
#   验证条件：`stored.summary.enabled is True`。
#   验证条件：`stored.summary.inherit_main is True`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`、`path.write_text`。
def test_old_settings_without_summary_use_inherited_enabled_default(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.json"
    ModelSettingsService(
        store=ModelSettingsStore(path),
        secrets=FakeSecrets(),
        base_settings=_base_settings(),
    ).save(_update())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("summary")
    path.write_text(json.dumps(payload), encoding="utf-8")

    stored = ModelSettingsStore(path).load()

    assert stored is not None
    assert stored.summary.enabled is True
    assert stored.summary.inherit_main is True


# 函数说明：test_connection_test_rejects_custom_endpoint_before_reading_key
# 用途：回归验证回归测试与测试辅助中的
# `connection_test_rejects_custom_endpoint_before_reading_key` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FakeSecrets` → `ModelSettingsService`
#  → `ModelSettingsStore` → `_base_settings` → `ProviderSettingsUpdate` →
# `pytest.raises`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`secrets.reads == []`。
#   预期异常：`pytest.raises(ValueError, match='official HTTPS endpoint')`。
@pytest.mark.asyncio
async def test_connection_test_rejects_custom_endpoint_before_reading_key(
    tmp_path: Path,
) -> None:
    secrets = FakeSecrets(values={"openai": "must-not-be-read"})
    service = ModelSettingsService(
        store=ModelSettingsStore(tmp_path / "models.json"),
        secrets=secrets,
        base_settings=_base_settings(),
    )
    provider = ProviderSettingsUpdate(
        provider=ModelProvider.OPENAI,
        model="proxy-model",
        base_url="https://example.invalid/v1",
        api_style=ApiStyle.RESPONSES,
    )

    with pytest.raises(ValueError, match="official HTTPS endpoint"):
        await service.test(provider)

    assert secrets.reads == []


# 函数说明：test_default_provider_without_any_key_is_rejected
# 用途：回归验证回归测试与测试辅助中的 `default_provider_without_any_key_is_rejected` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_update` → `ModelSettingsService` →
# `ModelSettingsStore` → `FakeSecrets` → `_base_settings` → `pytest.raises`；另有 1 个调
# 用点。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='default provider requires an API key')`
# 。
def test_default_provider_without_any_key_is_rejected(tmp_path: Path) -> None:
    update = _update().model_copy(
        update={
            "providers": tuple(
                item.model_copy(update={"api_key": None})
                for item in _update().providers
            )
        }
    )
    service = ModelSettingsService(
        store=ModelSettingsStore(tmp_path / "models.json"),
        secrets=FakeSecrets(),
        base_settings=_base_settings(),
    )

    with pytest.raises(ValueError, match="default provider requires an API key"):
        service.save(update)
