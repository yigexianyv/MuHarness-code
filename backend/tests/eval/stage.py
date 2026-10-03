"""为单个用例搭一个隔离的运行现场。

每次执行都在新的临时目录里启动一个完整的 ``Application``：独立的数据库、任务、记忆、
技能目录和工作区。用例写的环境变量只在这次执行期间生效。审批改成程序自动裁决，
不会卡在等待人工点击上。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values

from app.application import Application

from .spec import Case, GeneratedFile, Variant


@dataclass(frozen=True, slots=True)
class StagePaths:
    root: Path
    workspace: Path
    data: Path

    # 函数说明：StagePaths.under
    # 用途：返回 `cls(root=root, workspace=root / 'workspace', data=root / 'data')`，提
    # 供 StagePaths 的派生值。
    # 参数：
    #   root：当前操作的根目录，类型 `Path`。
    # 返回：类型 `StagePaths`；返回
    # `cls(root=root, workspace=root / 'workspace', data=root / 'data')`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`cls`。
    @classmethod
    def under(cls, root: Path) -> StagePaths:
        return cls(root=root, workspace=root / "workspace", data=root / "data")


class AppFactory(Protocol):
    # 函数说明：AppFactory.__call__
    # 用途：以可调用对象接口处理 AppFactory 的输入并交付结果。
    # 参数：
    #   paths：待处理的路径集合，类型 `StagePaths`。
    #   variant：`variant`输入或配置值，类型 `Variant`。
    # 返回：类型 `Application`；不返回结果值（隐式 None）。
    def __call__(self, paths: StagePaths, variant: Variant) -> Application: ...


# 函数说明：live_app_factory
# 用途：使用 backend/.env 里配置的真实模型服务商。
# 参数：
#   provider：模型或搜索服务商，类型 `str | None`；默认 `None`。
#   model：模型名称，类型 `str | None`；默认 `None`。
# 返回：类型 `AppFactory`；返回 `build`。
def live_app_factory(*, provider: str | None = None, model: str | None = None) -> AppFactory:
    """使用 backend/.env 里配置的真实模型服务商。"""

    # 函数说明：live_app_factory.build
    # 用途：构建回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   paths：待处理的路径集合，类型 `StagePaths`。
    #   variant：`variant`输入或配置值，类型 `Variant`。
    # 返回：类型 `Application`；返回 `Application(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`Application` →
    # `variant.system_prompt` → `isolated_storage`。
    # 闭包依赖：从外层读取 `model`、`provider`。
    def build(paths: StagePaths, variant: Variant) -> Application:
        with project_model_env(variant):
            return Application(
                provider=provider,
                model=model,
                system_prompt=variant.system_prompt(),
                web_approval=True,
                workspace_root=paths.workspace,
                **isolated_storage(paths),
            )

    return build


# 函数说明：isolated_storage
# 用途：返回 `{'database': paths.data / 'muharness.db', 'tasks_dir': paths.data / 'tasks
# ', 'mcp_config'…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   paths：待处理的路径集合，类型 `StagePaths`。
# 返回：类型 `dict[str, Path]`；字典，包含字段 `database`、`tasks_dir`、`mcp_config`、
# `memory_dir`、`skills_user_dir`、`skills_project_dir`。
def isolated_storage(paths: StagePaths) -> dict[str, Path]:
    return {
        "database": paths.data / "muharness.db",
        "tasks_dir": paths.data / "tasks",
        "mcp_config": paths.data / "mcp.json",  # 不存在：评测不加载任何 MCP 服务
        "memory_dir": paths.data / "memory",
        "skills_user_dir": paths.data / "skills-user",
        "skills_project_dir": paths.data / "skills-project",
    }


# 函数说明：patched_env
# 用途：在回归测试与测试辅助中处理 `patched_env`，通过 `os.environ.get` 完成首个内部处理
# 步骤。
# 参数：
#   values：待处理的值集合，类型 `Mapping[str, str]`。
# 返回：生成器，逐项产出 `None`；资源与结束处理遵循生成器流程。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`os.environ.update` → `os.environ.pop`
# 。
@contextmanager
def patched_env(values: Mapping[str, str]) -> Any:
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def project_model_env(
    variant: Variant, *, env_file: Path | None = None
) -> Any:
    """本地测评优先使用项目模型配置，避免桌面进程的同名旧变量覆盖密钥。"""
    path = env_file or Path(__file__).resolve().parents[2] / ".env"
    prefixes = ("OPENAI_", "ANTHROPIC_", "DEEPSEEK_", "QWEN_", "DASHSCOPE_", "MODEL_")
    values = {
        key: value for key, value in dotenv_values(path, encoding="utf-8").items()
        if value is not None and key.startswith(prefixes)
    } if path.is_file() else {}
    # 明确的测评变体仍可以覆盖项目模型配置。
    values.update({key: value for key, value in variant.env.items()
                   if key.startswith(prefixes)})
    with patched_env(values):
        yield


# 函数说明：write_setup
# 用途：写入`setup`，供回归测试与测试辅助使用。
# 参数：
#   case：`case`输入或配置值，类型 `Case`。
#   paths：待处理的路径集合，类型 `StagePaths`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`paths.workspace.mkdir` →
# `paths.data.mkdir` → `_inside` → `target.parent.mkdir` → `content.render` →
# `target.write_text`；另有 1 个调用点。
# 分支与异常：
#   当 `target.is_relative_to(paths.workspace)` 时，抛出
# `ValueError(f'{case.id}: outside_files 不能写进工作区：{relative}')`。
# 副作用与资源：
#   文件或资源访问：`paths.workspace.mkdir`、`paths.data.mkdir`、`target.parent.mkdir`、
# `target.write_text`。
def write_setup(case: Case, paths: StagePaths) -> None:
    paths.workspace.mkdir(parents=True, exist_ok=True)
    paths.data.mkdir(parents=True, exist_ok=True)
    for relative, content in case.setup.files.items():
        target = _inside(paths.workspace, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = content.render() if isinstance(content, GeneratedFile) else content
        target.write_text(text, encoding="utf-8")
    for relative, content in case.setup.outside_files.items():
        target = _inside(paths.root, relative)
        if target.is_relative_to(paths.workspace):
            raise ValueError(f"{case.id}: outside_files 不能写进工作区：{relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


# 函数说明：_inside
# 用途：在回归测试与测试辅助中处理 `_inside`，通过 `(base / relative).resolve` 完成首个
# 内部处理步骤。
# 参数：
#   base：`base`输入或配置值，类型 `Path`。
#   relative：相对路径，类型 `str`。
# 返回：类型 `Path`；返回 `target`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(base / relative).resolve` →
# `target.is_relative_to` → `base.resolve`。
# 分支与异常：
#   当 `not target.is_relative_to(base.resolve())` 时，抛出
# `ValueError(f'路径越界：{relative}')`。
def _inside(base: Path, relative: str) -> Path:
    target = (base / relative).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError(f"路径越界：{relative}")
    return target


# 函数说明：install_approval_policy
# 用途：审批请求一出现就按用例规定批准或拒绝。
# 参数：
#   application：已装配的应用依赖，类型 `Application`。
#   decision：权限、上下文或审计决策，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`gate.set_broadcaster`。
# 分支与异常：
#   当 `gate is None` 时，抛出
# `RuntimeError('评测需要 web_approval=True，才能由程序裁决审批')`。
def install_approval_policy(application: Application, decision: str) -> None:
    """审批请求一出现就按用例规定批准或拒绝。"""

    gate = application.web_approval_gate
    if gate is None:
        raise RuntimeError("评测需要 web_approval=True，才能由程序裁决审批")

    # 函数说明：install_approval_policy.broadcast
    # 用途：广播回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `Any`；读取键 `approval`。
    # 返回：类型 `None`；无结果值，显式返回 None。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：
    # `asyncio.get_running_loop().call_soon` → `asyncio.get_running_loop`。
    # 分支与异常：
    #   当 `method != 'approval.required'` 时，返回 `None`。
    # 闭包依赖：从外层读取 `decision`、`gate`。
    async def broadcast(method: str, params: Any) -> None:
        if method != "approval.required":
            return
        approval_id = params["approval"].id
        resolve = gate.approve if decision == "approve" else gate.deny
        # 放到下一轮事件循环：此刻 Future 已登记，裁决不会和登记抢先后
        asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(resolve(approval_id)))

    gate.set_broadcaster(broadcast)


@dataclass(slots=True)
class Stage:
    case: Case
    variant: Variant
    paths: StagePaths
    app: Application


# 函数说明：open_stage
# 用途：打开`stage`，供回归测试与测试辅助使用。
# 参数：
#   case：传给 `write_setup` 的输入，类型 `Case`。
#   variant：传给 `factory` 的输入，类型 `Variant`。
#   factory：构造目标依赖的工厂，类型 `AppFactory`。
#   keep：`keep`输入或配置值，类型 `bool`；默认 `False`。
#   on_ready：`on_ready`输入或配置值，类型 `Callable[[Stage], None] | None`；默认 `None`
# 。
# 返回：异步生成器，逐项产出 `stage`；资源与结束处理遵循生成器流程。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Path` → `tempfile.mkdtemp` →
# `StagePaths.under` → `patched_env` → `write_setup` → `factory`；另有 7 个调用点。
@asynccontextmanager
async def open_stage(
    case: Case,
    variant: Variant,
    factory: AppFactory,
    *,
    keep: bool = False,
    on_ready: Callable[[Stage], None] | None = None,
) -> AsyncIterator[Stage]:
    root = Path(tempfile.mkdtemp(prefix=f"muharness-eval-{case.id}-"))
    paths = StagePaths.under(root)
    env = {**variant.env, **case.env}
    with patched_env(env):
        write_setup(case, paths)
        application = factory(paths, variant)
        stage = Stage(case=case, variant=variant, paths=paths, app=application)
        try:
            await application.start()
            install_approval_policy(application, case.approvals)
            for memory in case.setup.memories:
                await application.memory_manager.create(
                    title=memory.title, summary=memory.summary, content=memory.content
                )
            if on_ready is not None:
                on_ready(stage)
            yield stage
        finally:
            try:
                await application.close()
            finally:
                if not keep:
                    shutil.rmtree(root, ignore_errors=True)


__all__ = [
    "AppFactory",
    "Stage",
    "StagePaths",
    "install_approval_policy",
    "isolated_storage",
    "live_app_factory",
    "open_stage",
    "patched_env",
    "write_setup",
]
