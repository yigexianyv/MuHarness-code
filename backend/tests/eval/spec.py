"""评测用例的描述格式与加载。

一个用例 = 准备好的环境（文件、记忆）+ 一段交互（多轮对话或一次长任务）+ 一组检查。
用例写在 ``cases/<suite>/*.yaml``，检查项的含义见 ``checks.py``。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CASES_DIR = Path(__file__).resolve().parent / "cases"
VARIANTS_DIR = Path(__file__).resolve().parent / "variants"

Suite = Literal["behavior", "context", "mea"]


class GeneratedFile(BaseModel):
    """按模板生成的大文件，用来制造上下文压力。``{i}`` 会替换成行号。"""

    model_config = ConfigDict(extra="forbid")

    line: str
    lines: int = Field(gt=0, le=20_000)
    head: str | None = None  # 放在文件开头的关键内容
    tail: str | None = None  # 放在文件结尾的关键内容
    insert: dict[int, str] = Field(default_factory=dict)  # 在第 N 行之后插入一行，用来放中段内容

    # 函数说明：GeneratedFile.render
    # 用途：生成展示文本GeneratedFile，供回归测试与测试辅助使用。
    # 返回：类型 `str`；返回 `'\n'.join(parts) + '\n'`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.line.replace`。
    def render(self) -> str:
        rows: list[str] = []
        for index in range(1, self.lines + 1):
            rows.append(self.line.replace("{i}", str(index)))
            if index in self.insert:
                rows.append(self.insert[index])
        body = "\n".join(rows)
        parts = [part for part in (self.head, body, self.tail) if part]
        return "\n".join(parts) + "\n"


class SeedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    content: str


class Setup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 工作区内的文件：路径 → 文本，或生成规则
    files: dict[str, str | GeneratedFile] = Field(default_factory=dict)
    # 工作区外、同一临时目录下的文件，用来检查越界访问
    outside_files: dict[str, str] = Field(default_factory=dict)
    memories: list[SeedMemory] = Field(default_factory=list)


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    say: str
    mode: Literal["normal", "plan"] = "normal"
    session: str = "default"


class MeaStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    acceptance: str


class MeaPlan(BaseModel):
    """直接以一个已接受的计划启动长任务，跳过规划模式，让结果只取决于 MEA 本身。"""

    model_config = ConfigDict(extra="forbid")

    request: str
    title: str
    goal: str
    steps: list[MeaStep] = Field(min_length=1)
    round_budget: int = Field(default=12, ge=1, le=60)
    extra_tools: list[str] = Field(default_factory=list)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    suite: Suite
    title: str
    why: str = ""  # 这个用例防的是哪种问题
    tags: list[str] = Field(default_factory=list)
    requires: list[Literal["docker"]] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    approvals: Literal["approve", "deny"] = "approve"
    timeout_seconds: float = Field(default=300, gt=0)
    setup: Setup = Field(default_factory=Setup)
    turns: list[Turn] = Field(default_factory=list)
    mea: MeaPlan | None = None
    checks: list[dict[str, Any]] = Field(min_length=1)
    source: str | None = None  # 加载时填入 YAML 路径

    # 函数说明：Case._plain_strings
    # 用途：校验并规范化模型字段 'turns'。
    # 参数：
    #   value：待校验、规范化或转换的值，类型 `object`。
    # 返回：类型 `object`；按分支返回
    # `[{'say': item} if isinstance(item, str) else item for item in value]`；`value`。
    # 分支与异常：
    #   当 `isinstance(value, list)` 时，返回
    # `[{'say': item} if isinstance(item, str) else item for item…`。
    @field_validator("turns", mode="before")
    @classmethod
    def _plain_strings(cls, value: object) -> object:
        if isinstance(value, list):
            return [{"say": item} if isinstance(item, str) else item for item in value]
        return value

    # 函数说明：Case._one_driver
    # 用途：校验并规范化模型字段 及字段之间的约束。
    # 返回：类型 `Case`；返回 `self`。
    # 分支与异常：
    #   当 `bool(self.turns) == (self.mea is not None)` 时，抛出
    # `ValueError(f'{self.id}: 用例需要 turns 或 mea 中的恰好一个')`。
    #   当 `self.suite == 'mea' and self.mea is None` 时，抛出
    # `ValueError(f'{self.id}: mea 套件的用例必须提供 mea 计划')`。
    @model_validator(mode="after")
    def _one_driver(self) -> Case:
        if bool(self.turns) == (self.mea is not None):
            raise ValueError(f"{self.id}: 用例需要 turns 或 mea 中的恰好一个")
        if self.suite == "mea" and self.mea is None:
            raise ValueError(f"{self.id}: mea 套件的用例必须提供 mea 计划")
        return self

    # 函数说明：Case.digest
    # 用途：在回归测试与测试辅助中处理 `digest`，通过 `self.model_dump` 完成首个内部处理
    # 步骤。
    # 返回：类型 `str`；返回 `hashlib.sha256(json.dumps(payload, sort_keys=True,
    # ensure_ascii=False).encode()).…`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).…` →
    # `hashlib.sha256` →
    # `json.dumps(payload, sort_keys=True, ensure_ascii=False).encode` → `json.dumps`。
    def digest(self) -> str:
        payload = self.model_dump(mode="json", exclude={"source"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


class Variant(BaseModel):
    """一次对照里的一侧：同一批用例，只换提示词或配置。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    system_prompt_file: str | None = None  # 相对 variants/ 目录
    env: dict[str, str] = Field(default_factory=dict)

    # 函数说明：Variant.system_prompt
    # 用途：在回归测试与测试辅助中处理 `system_prompt`，通过
    # `(VARIANTS_DIR / self.system_prompt_file).read_text(encoding='utf-8').rstrip` 完成
    # 首个内部处理步骤。
    # 返回：类型 `str | None`；按分支返回 `None`；`(VARIANTS_DIR / self.
    # system_prompt_file).read_text(encoding='utf-8').rstrip('\n')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `(VARIANTS_DIR / self.system_prompt_file).read_text(encoding='utf-8').rstrip` →
    # `(VARIANTS_DIR / self.system_prompt_file).read_text`。
    # 分支与异常：
    #   当 `self.system_prompt_file is None` 时，返回 `None`。
    # 副作用与资源：
    #   文件或资源访问：`(VARIANTS_DIR / self.system_prompt_file).read_text`。
    def system_prompt(self) -> str | None:
        if self.system_prompt_file is None:
            return None
        return (VARIANTS_DIR / self.system_prompt_file).read_text(encoding="utf-8").rstrip("\n")


CURRENT = Variant(name="current", description="仓库当前代码与默认提示词")


# 函数说明：load_cases
# 用途：加载`cases`，供回归测试与测试辅助使用。
# 参数：
#   suites：`suites`输入或配置值，类型 `Iterable[str] | None`；默认 `None`。
#   ids：`ids`输入或配置值，类型 `Iterable[str] | None`；默认 `None`。
#   tags：`tags`输入或配置值，类型 `Iterable[str] | None`；默认 `None`。
#   root：当前操作的根目录，类型 `Path`；默认 `CASES_DIR`。
# 返回：类型 `list[Case]`；返回 `cases`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`root.glob` → `yaml.safe_load` →
# `path.read_text` → `Case.model_validate` → `path.relative_to(root).as_posix` →
# `path.relative_to`；另有 1 个调用点。
# 分支与异常：
#   当 `case.id in seen` 时，抛出
# `ValueError(f'用例 id 重复：{case.id}（{seen[case.id]} 与 {path}）')`。
#   当 `path.parent.name != case.suite` 时，抛出 `ValueError(…)`。
#   当 `wanted_suites and case.suite not in wanted_suites` 时，跳过当前循环项。
#   当 `wanted_ids and case.id not in wanted_ids` 时，跳过当前循环项。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
def load_cases(
    *,
    suites: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
    root: Path = CASES_DIR,
) -> list[Case]:
    wanted_suites = set(suites or ())
    wanted_ids = set(ids or ())
    wanted_tags = set(tags or ())
    cases: list[Case] = []
    seen: dict[str, Path] = {}
    for path in sorted(root.glob("*/*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        case = Case.model_validate({**data, "source": path.relative_to(root).as_posix()})
        if case.id in seen:
            raise ValueError(f"用例 id 重复：{case.id}（{seen[case.id]} 与 {path}）")
        if path.parent.name != case.suite:
            raise ValueError(f"{path}: 目录名 {path.parent.name} 与 suite {case.suite} 不一致")
        seen[case.id] = path
        if wanted_suites and case.suite not in wanted_suites:
            continue
        if wanted_ids and case.id not in wanted_ids:
            continue
        if wanted_tags and not wanted_tags.intersection(case.tags):
            continue
        cases.append(case)
    return cases


# 函数说明：load_variant
# 用途：加载`variant`，供回归测试与测试辅助使用。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   root：当前操作的根目录，类型 `Path`；默认 `VARIANTS_DIR`。
# 返回：类型 `Variant`；按分支返回 `CURRENT`；`variant`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.is_file` → `root.glob` →
# `Variant.model_validate` → `yaml.safe_load` → `path.read_text`。
# 分支与异常：
#   当 `name == CURRENT.name` 时，返回 `CURRENT`。
#   `not path.is_file()` 分支在完成前置处理后抛出
# `ValueError(f'找不到变体 {name!r}，可用：{available}')`。
#   当 `variant.name != name` 时，抛出 `ValueError(f'{path}: name 字段应为 {name!r}')`。
# 副作用与资源：
#   文件或资源访问：`path.read_text`。
def load_variant(name: str, *, root: Path = VARIANTS_DIR) -> Variant:
    if name == CURRENT.name:
        return CURRENT
    path = root / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join([CURRENT.name, *(p.stem for p in sorted(root.glob("*.yaml")))])
        raise ValueError(f"找不到变体 {name!r}，可用：{available}")
    variant = Variant.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if variant.name != name:
        raise ValueError(f"{path}: name 字段应为 {name!r}")
    return variant


__all__ = [
    "CASES_DIR",
    "CURRENT",
    "Case",
    "GeneratedFile",
    "MeaPlan",
    "MeaStep",
    "SeedMemory",
    "Setup",
    "Turn",
    "VARIANTS_DIR",
    "Variant",
    "load_cases",
    "load_variant",
]
