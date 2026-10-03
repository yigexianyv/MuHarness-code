"""离线自测用的脚本模型：按请求内容决定回复，不访问网络。"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import SecretStr

from app.application import Application
from app.domain.memory import MemoryMaintenanceConfig, MemoryReflectionConfig
from app.domain.skill_learning import SkillLearningSettings
from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    ApiStyle,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from tests.eval.spec import Variant
from tests.eval.stage import StagePaths, isolated_storage

Reply = Callable[[ModelRequest], ModelResponse]


# 函数说明：answer
# 用途：处理回复回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   text：待处理的文本，类型 `str | None`；默认 `None`。
#   calls：调用集合输入或配置值，类型 `tuple[ToolCall, ...]`；默认 `()`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def answer(text: str | None = None, *, calls: tuple[ToolCall, ...] = ()) -> ModelResponse:
    return ModelResponse(
        id="scripted",
        provider="fake",
        model="fake-model",
        message=Message(role=MessageRole.ASSISTANT, content=text, tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop",
        usage=ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120, model_calls=1),
    )


# 函数说明：call
# 用途：返回
# `answer(calls=(ToolCall(id=f'call-{name}', name=name, arguments=arguments),))`，提供
# 回归测试与测试辅助 的派生值。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   **arguments：额外关键字参数，按实现处理或转交。
# 返回：类型 `ModelResponse`；返回
# `answer(calls=(ToolCall(id=f'call-{name}', name=name, arguments=arguments),))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`answer` → `ToolCall`。
def call(name: str, **arguments: object) -> ModelResponse:
    return answer(calls=(ToolCall(id=f"call-{name}", name=name, arguments=arguments),))


# 函数说明：last_user
# 用途：处理回归测试与测试辅助中的 `last_user` 数据；结果及边界条件见下方说明。
# 参数：
#   request：待处理的请求对象，类型 `ModelRequest`。
# 返回：类型 `str`；按分支返回 `message.content or ''`；`''`。
# 分支与异常：
#   当 `message.role is MessageRole.USER` 时，返回 `message.content or ''`。
def last_user(request: ModelRequest) -> str:
    for message in reversed(request.messages):
        if message.role is MessageRole.USER:
            return message.content or ""
    return ""


# 函数说明：after_tool
# 用途：在 `tool` 前后执行 回归测试与测试辅助 的生命周期钩子。
# 参数：
#   request：待处理的请求对象，类型 `ModelRequest`。
# 返回：类型 `bool`；返回 `request.messages[-1].role is MessageRole.TOOL`。
def after_tool(request: ModelRequest) -> bool:
    return request.messages[-1].role is MessageRole.TOOL


class ScriptedAdapter(ModelAdapter):
    # 函数说明：ScriptedAdapter.__init__
    # 用途：初始化 ScriptedAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   reply：用户回复或工具响应，类型 `Reply`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self._reply`、`self.requests`。
    def __init__(self, config: ProviderConfig, reply: Reply) -> None:
        super().__init__(config)
        self._reply = reply
        self.requests: list[ModelRequest] = []

    # 函数说明：ScriptedAdapter.complete
    # 用途：完成ScriptedAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象，类型 `ModelRequest`。
    # 返回：类型 `ModelResponse`；返回 `self._reply(request)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self._reply`。
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self._reply(request)

    # 函数说明：ScriptedAdapter.close
    # 用途：关闭ScriptedAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：scripted_factory
# 用途：返回一个 AppFactory：真实的 Application，只把模型换成脚本。
# 参数：
#   reply：用户回复或工具响应，类型 `Reply`。
#   sink：`sink`输入或配置值，类型 `list[Application] | None`；默认 `None`。
# 返回：返回 `build`。
def scripted_factory(reply: Reply, *, sink: list[Application] | None = None):
    """返回一个 AppFactory：真实的 Application，只把模型换成脚本。"""

    # 函数说明：scripted_factory.build
    # 用途：构建回归测试与测试辅助，供回归测试与测试辅助使用。
    # 参数：
    #   paths：待处理的路径集合，类型 `StagePaths`。
    #   variant：`variant`输入或配置值，类型 `Variant`。
    # 返回：类型 `Application`；返回 `application`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
    # `ModelAdapterRegistry` → `ModelSettings` → `registry.register` → `Application`；另
    # 有 5 个调用点。
    # 闭包依赖：从外层读取 `sink`。
    def build(paths: StagePaths, variant: Variant) -> Application:
        config = ProviderConfig(
            provider="fake",
            model="fake-model",
            api_key=SecretStr("offline"),
            api_style=ApiStyle.CHAT_COMPLETIONS,
        )
        registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
        registry.register("fake", lambda _: ScriptedAdapter(config, reply), config=config)
        application = Application(
            provider="fake",
            model="fake-model",
            registry=registry,
            system_prompt=variant.system_prompt(),
            web_approval=True,
            workspace_root=paths.workspace,
            memory_reflection_config=MemoryReflectionConfig(_env_file=None, enabled=False),
            memory_maintenance_config=MemoryMaintenanceConfig(_env_file=None, enabled=False),
            skill_learning_settings=SkillLearningSettings(
                _env_file=None,
                skill_learning_enabled=False,
                skill_learning_data_dir=paths.data / "skill-learning",
            ),
            **isolated_storage(paths),
        )
        if sink is not None:
            sink.append(application)
        return application

    return build
