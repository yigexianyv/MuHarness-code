"""避免桌面环境中的旧密钥覆盖项目配置；测试不访问真实接口。"""

from app.models.config import ModelSettings
from tests.eval.spec import CURRENT, Variant
from tests.eval.stage import project_model_env


def test_project_key_overrides_inherited_key_and_is_restored(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY=project-key\nOPENAI_MODEL=project-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "stale-desktop-key")
    with project_model_env(CURRENT, env_file=path):
        settings = ModelSettings(_env_file=None)
        assert settings.openai_api_key.get_secret_value() == "project-key"
        assert settings.openai_model == "project-model"
    assert (
        ModelSettings(_env_file=None).openai_api_key.get_secret_value()
        == "stale-desktop-key"
    )


def test_explicit_variant_and_unrelated_environment_are_preserved(
    tmp_path, monkeypatch
):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_MODEL=project-model\nTAVILY_API_KEY=project-search\n", encoding="utf-8"
    )
    monkeypatch.setenv("TAVILY_API_KEY", "existing-search")
    variant = Variant(name="override", env={"OPENAI_MODEL": "variant-model"})
    with project_model_env(variant, env_file=path):
        assert ModelSettings(_env_file=None).openai_model == "variant-model"
        import os

        assert os.environ["TAVILY_API_KEY"] == "existing-search"


async def test_non_streaming_option_uses_complete_and_preserves_response():
    from pydantic import SecretStr

    from app.models.config import ProviderConfig
    from app.models.types import ApiStyle, Message, MessageRole, ModelRequest
    from tests.eval.selftest.scripted import ScriptedAdapter, answer
    from tests.eval.transport import NonStreamingAdapter

    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("test"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    delegate = ScriptedAdapter(config, lambda _: answer("OK"))
    wrapper = NonStreamingAdapter(delegate)

    async def ignored_delta(text):
        raise AssertionError("没有流式输出")

    request = ModelRequest(messages=(Message(role=MessageRole.USER, content="hello"),))
    response = await wrapper.complete_stream(request, on_text_delta=ignored_delta)
    assert response.message.content == "OK"
    assert len(delegate.requests) == 1
    assert response.usage.model_calls == 1
    await wrapper.close()
