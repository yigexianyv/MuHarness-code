"""显式非流式测评，使用相同适配器的 complete 路径。"""

from app.models.adapter import ModelAdapter
from app.models.config import ProviderConfig
from app.models.providers import AnthropicAdapter, OpenAICompatibleAdapter
from app.models.types import ApiStyle, ModelRequest, ModelResponse


class NonStreamingAdapter(ModelAdapter):
    def __init__(self, delegate: ModelAdapter) -> None:
        super().__init__(delegate.config)
        self.delegate = delegate

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await self.delegate.complete(request)

    async def close(self) -> None:
        await self.delegate.close()


def non_streaming_factory(config: ProviderConfig) -> NonStreamingAdapter:
    adapter_type = (
        AnthropicAdapter
        if config.api_style is ApiStyle.ANTHROPIC_MESSAGES
        else OpenAICompatibleAdapter
    )
    return NonStreamingAdapter(adapter_type(config))
