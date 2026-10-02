
from __future__ import annotations

import argparse
import asyncio
import sys
from time import perf_counter

from .config import ModelSettings
from .registry import ModelAdapterRegistry
from .types import Message, MessageRole, ModelProvider, ModelRequest


# 函数说明：_test_provider
# 用途：在模型请求与响应处理中处理 `_test_provider`，通过 `registry.get` 完成首个内部处
# 理步骤。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ModelAdapterRegistry`。
#   provider：模型或搜索服务商，类型 `ModelProvider`。
#   prompt：本次调用使用的提示文本，类型 `str`。
#   max_output_tokens：模型输出 Token 上限，类型 `int`。
# 返回：类型 `bool`；按分支返回 `False`；`True`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`perf_counter` → `adapter.complete` →
# `ModelRequest` → `Message`。
# 分支与异常：
#   捕获 `Exception` 后，返回 `False`。
async def _test_provider(
    registry: ModelAdapterRegistry,
    provider: ModelProvider,
    *,
    prompt: str,
    max_output_tokens: int,
) -> bool:
    started_at = perf_counter()
    try:
        adapter = registry.get(provider)
        response = await adapter.complete(
            ModelRequest(
                messages=(Message(role=MessageRole.USER, content=prompt),),
                max_output_tokens=max_output_tokens,
            )
        )
    except Exception as exc:
        elapsed = perf_counter() - started_at
        print(
            f"FAIL {provider.value} ({elapsed:.2f}s): {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False

    elapsed = perf_counter() - started_at
    content = response.message.content or "<empty text response>"
    print(
        f"PASS {provider.value} model={response.model} "
        f"latency={elapsed:.2f}s tokens={response.usage.total_tokens}"
    )
    print(f"  {content.strip()}")
    return True


# 函数说明：_run
# 用途：运行模型请求与响应处理，供模型请求与响应处理使用。
# 参数：
#   args：`args`输入或配置值，类型 `argparse.Namespace`。
# 返回：类型 `int`；按分支返回 `2`；`0 if all(results) else 1`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelSettings` →
# `settings.configured_providers` → `ModelProvider` → `ModelAdapterRegistry` →
# `_test_provider` → `registry.close`。
# 分支与异常：
#   `requested not in configured` 分支在完成前置处理后返回 `2`。
#   `not providers` 分支在完成前置处理后返回 `2`。
async def _run(args: argparse.Namespace) -> int:
    settings = ModelSettings()
    configured = settings.configured_providers()

    if args.provider == "all":
        providers = configured
    else:
        requested = ModelProvider(args.provider)
        if requested not in configured:
            print(
                f"Provider '{requested.value}' is not configured in backend/.env.",
                file=sys.stderr,
            )
            return 2
        providers = (requested,)

    if not providers:
        print(
            "No model provider is configured in backend/.env.",
            file=sys.stderr,
        )
        return 2

    print("Testing providers:", ", ".join(provider.value for provider in providers))
    registry = ModelAdapterRegistry(settings)
    try:
        results = [
            await _test_provider(
                registry,
                provider,
                prompt=args.prompt,
                max_output_tokens=args.max_output_tokens,
            )
            for provider in providers
        ]
    finally:
        await registry.close()

    return 0 if all(results) else 1


# 函数说明：_parse_args
# 用途：解析`args`，供模型请求与响应处理使用。
# 返回：类型 `argparse.Namespace`；返回 `args`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`argparse.ArgumentParser` →
# `parser.add_argument` → `parser.parse_args` → `parser.error`。
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a minimal live request to configured model providers."
    )
    parser.add_argument(
        "--provider",
        choices=["all", *(provider.value for provider in ModelProvider)],
        default="all",
        help="Provider to test; defaults to every configured provider.",
    )
    parser.add_argument(
        "--prompt",
        default="只回复：连接成功",
        help="Minimal prompt sent to the provider.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=32,
        help="Maximum output tokens per request.",
    )
    args = parser.parse_args()
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be greater than zero")
    return args


# 函数说明：main
# 用途：运行模型请求与响应处理入口，按照当前参数装配依赖并驱动主流程。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SystemExit` → `asyncio.run` → `_run`
# → `_parse_args`。
def main() -> None:
    raise SystemExit(asyncio.run(_run(_parse_args())))


if __name__ == "__main__":
    main()
