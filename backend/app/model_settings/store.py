
from __future__ import annotations

import json
import os
from pathlib import Path

from app.paths import runtime_data_path

from .models import StoredModelSettings

DEFAULT_MODEL_SETTINGS_PATH = runtime_data_path("settings/models.json")


class ModelSettingsStore:

    # 函数说明：ModelSettingsStore.__init__
    # 用途：初始化 ModelSettingsStore；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   path：目标文件或目录路径，类型 `str | Path`；默认 `DEFAULT_MODEL_SETTINGS_PATH`
    # 。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Path(path).expanduser().resolve`
    # → `Path(path).expanduser` → `Path`。
    # 副作用与资源：
    #   更新对象字段：`self.path`。
    def __init__(self, path: str | Path = DEFAULT_MODEL_SETTINGS_PATH) -> None:
        self.path = Path(path).expanduser().resolve()

    # 函数说明：ModelSettingsStore.load
    # 用途：加载ModelSettingsStore，供模型配置与密钥管理使用。
    # 返回：类型 `StoredModelSettings | None`；按分支返回 `None`；
    # `StoredModelSettings.model_validate(payload)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.path.is_file` →
    # `self.path.is_symlink` → `json.loads` → `self.path.read_text` →
    # `StoredModelSettings.model_validate`。
    # 分支与异常：
    #   当 `not self.path.is_file()` 时，返回 `None`。
    #   当 `self.path.is_symlink()` 时，抛出
    # `ValueError('model settings file cannot be a symbolic link')`。
    # 副作用与资源：
    #   文件或资源访问：`self.path.read_text`。
    def load(self) -> StoredModelSettings | None:
        if not self.path.is_file():
            return None
        if self.path.is_symlink():
            raise ValueError("model settings file cannot be a symbolic link")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return StoredModelSettings.model_validate(payload)

    # 函数说明：ModelSettingsStore.save
    # 用途：保存ModelSettingsStore，供模型配置与密钥管理使用。
    # 参数：
    #   settings：业务或模型设置，类型 `StoredModelSettings`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.path.parent.mkdir` →
    # `self.path.with_name` → `os.getpid` → `json.dumps` → `temporary.write_text` →
    # `os.replace`。
    # 副作用与资源：
    #   文件或资源访问：`self.path.parent.mkdir`、`temporary.write_text`、`os.replace`。
    def save(self, settings: StoredModelSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        serialized = json.dumps(
            settings.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        temporary.write_text(serialized + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


__all__ = ["DEFAULT_MODEL_SETTINGS_PATH", "ModelSettingsStore"]
