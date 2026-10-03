"""命令行入口（在 backend 目录下运行）。

    python -m tests.eval list
    python -m tests.eval run --suite behavior --repeat 3
    python -m tests.eval ab --variants current,prompt-v1 --suite behavior --repeat 3
    python -m tests.eval compare reports/A.json reports/B.json

run 和 ab 会调用真实模型，费用取决于用例数 × 重复次数。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .checks import known_checks
from .report import RunReport, render_comparison
from .session import run_cases
from .spec import load_cases, load_variant
from .stage import live_app_factory

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


# 函数说明：_selection
# 用途：在回归测试与测试辅助中处理 `_selection`，通过 `parser.add_argument` 完成首个内部
# 处理步骤。
# 参数：
#   parser：`parser`输入或配置值，类型 `argparse.ArgumentParser`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parser.add_argument`。
def _selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--suite", action="append", choices=["behavior", "context", "mea"],
        help="可重复",
    )
    parser.add_argument("--case", action="append", help="用例 id，可重复")
    parser.add_argument("--tag", action="append", help="只跑带这个标签的用例，可重复")


# 函数说明：_execution
# 用途：在回归测试与测试辅助中处理 `_execution`，通过 `parser.add_argument` 完成首个内部
# 处理步骤。
# 参数：
#   parser：`parser`输入或配置值，类型 `argparse.ArgumentParser`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parser.add_argument`。
def _execution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repeat", type=int, default=1, help="每个用例重复次数（默认 1）"
    )
    parser.add_argument("--provider", help="覆盖 .env 里的默认服务商")
    parser.add_argument("--model", help="覆盖默认模型")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR, help="报告目录")
    parser.add_argument(
        "--keep-outcomes", action="store_true", help="报告里保存每次执行的完整结果"
    )
    parser.add_argument(
        "--keep-stage", action="store_true", help="保留临时运行目录，便于排查"
    )


# 函数说明：build_parser
# 用途：构建`parser`，供回归测试与测试辅助使用。
# 返回：类型 `argparse.ArgumentParser`；返回 `parser`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`argparse.ArgumentParser` →
# `parser.add_subparsers` → `commands.add_parser` → `_selection` →
# `listing.add_argument` → `_execution`；另有 3 个调用点。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.eval", description="MuHarness 行为评测"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    from .v1_cases import CATEGORIES

    v1 = commands.add_parser("v1", help="六分类第一版：12 项、完整证据与独立验收")
    v1.add_argument("--list", action="store_true", help="只列出用例，不调用模型")
    v1.add_argument("--category", action="append", choices=CATEGORIES)
    v1.add_argument("--case", action="append", help="例如 B01，可重复")
    v1.add_argument("--repeat", type=int, default=1)
    v1.add_argument("--provider")
    v1.add_argument("--model")
    v1.add_argument("--no-stream", action="store_true", help="显式使用非流式模型请求")
    v1.add_argument("--api-retries", type=int, choices=range(7), default=3,
                    help="每次模型请求的接口错误追加重试次数（默认 3，可设 0）")
    v1.add_argument("--out", type=Path)

    listing = commands.add_parser("list", help="列出用例")
    _selection(listing)
    listing.add_argument("--checks", action="store_true", help="同时列出可用的检查项")

    run = commands.add_parser("run", help="用一个变体跑用例")
    _selection(run)
    _execution(run)
    run.add_argument("--variant", default="current")

    ab = commands.add_parser("ab", help="两个变体跑同一批用例并生成对照")
    _selection(ab)
    _execution(ab)
    ab.add_argument("--variants", required=True, help="逗号分隔的两个变体，前者为基准")

    compare = commands.add_parser("compare", help="对照两份已有报告")
    compare.add_argument("base", type=Path)
    compare.add_argument("other", type=Path)
    compare.add_argument("--out", type=Path, help="写入文件，默认打印")
    return parser


# 函数说明：_cases
# 用途：处理回归测试与测试辅助中的 `_cases` 数据；结果及边界条件见下方说明。
# 参数：
#   args：`args`输入或配置值，类型 `argparse.Namespace`。
# 返回：返回 `cases`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`load_cases` → `SystemExit`。
# 分支与异常：
#   当 `not cases` 时，抛出 `SystemExit('没有匹配的用例')`。
def _cases(args: argparse.Namespace):
    cases = load_cases(suites=args.suite, ids=args.case, tags=args.tag)
    if not cases:
        raise SystemExit("没有匹配的用例")
    return cases


# 函数说明：_run_variant
# 用途：运行`variant`，供回归测试与测试辅助使用。
# 参数：
#   args：传给 `_cases` 的输入，类型 `argparse.Namespace`。
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `RunReport`；返回 `report`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`load_variant` → `live_app_factory` →
# `run_cases` → `_cases` → `report.save`。
async def _run_variant(args: argparse.Namespace, name: str) -> RunReport:
    variant = load_variant(name)
    factory = live_app_factory(provider=args.provider, model=args.model)
    report = await run_cases(
        _cases(args),
        variant=variant,
        factory=factory,
        repeat=args.repeat,
        provider=args.provider,
        model=args.model,
        keep_outcomes=args.keep_outcomes,
        keep_stage=args.keep_stage,
        progress=lambda line: print(line, flush=True),
    )
    json_path, md_path = report.save(args.out)
    print(f"报告：{md_path}\n数据：{json_path}")
    return report


# 函数说明：main
# 用途：运行回归测试与测试辅助入口，按照当前参数装配依赖并驱动主流程。
# 参数：
#   argv：传给 `build_parser().parse_args` 的输入，类型 `list[str] | None`；默认 `None`
# 。
# 返回：类型 `int`；按分支返回 `0`；`0 if all((item.passed or item.status == 'skipped'
# for item in report.attempts)) else 1`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`build_parser().parse_args` →
# `build_parser` → `_cases` → `known_checks` → `render_comparison` → `RunReport.load`；
# 另有 8 个调用点。
# 分支与异常：
#   `args.command == 'list'` 分支在完成前置处理后返回 `0`。
#   `args.command == 'compare'` 分支在完成前置处理后返回 `0`。
#   当 `args.repeat < 1` 时，抛出 `SystemExit('--repeat 至少为 1')`。
#   `args.command == 'run'` 分支在完成前置处理后返回
# `0 if all((item.passed or item.status == 'skipped' for item…`。
# 副作用与资源：
#   文件或资源访问：`args.out.write_text`、`path.write_text`。
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "v1":
        from .v1 import REPORTS_DIR as V1_REPORTS_DIR
        from .v1 import exit_code, run_v1
        from .v1_cases import v1_cases

        plans = [plan for plan in v1_cases()
                 if (not args.category or plan.category in args.category)
                 and (not args.case or plan.case.id in args.case)]
        if not plans:
            raise SystemExit("没有匹配的 V1 用例")
        if args.list:
            for plan in plans:
                needs = "（需要 Docker）" if plan.case.requires else ""
                print(f"{plan.case.id}  {plan.category:12s}  {plan.case.title}{needs}")
            return 0
        if args.repeat < 1:
            raise SystemExit("--repeat 至少为 1")
        report, folder = asyncio.run(run_v1(
            plans=plans, provider=args.provider, model=args.model,
            repeat=args.repeat, non_stream=args.no_stream, api_retries=args.api_retries,
            out=args.out or V1_REPORTS_DIR,
            progress=lambda line: print(line, flush=True),
        ))
        print(f"中文报告：{folder / 'report.md'}\n结构化报告：{folder / 'report.json'}")
        return exit_code(report)
    if args.command == "list":
        for case in _cases(args):
            needs = f"  需要 {','.join(case.requires)}" if case.requires else ""
            print(f"{case.suite:9s} {case.id:34s} {case.title}{needs}")
        if args.checks:
            print("\n检查项：" + "、".join(known_checks()))
        return 0
    if args.command == "compare":
        text = render_comparison(RunReport.load(args.base), RunReport.load(args.other))
        if args.out:
            args.out.write_text(text, encoding="utf-8")
            print(f"对照：{args.out}")
        else:
            print(text)
        return 0
    if args.repeat < 1:
        raise SystemExit("--repeat 至少为 1")
    if args.command == "run":
        report = asyncio.run(_run_variant(args, args.variant))
        passed = all(
            item.passed or item.status == "skipped" for item in report.attempts
        )
        return 0 if passed else 1

    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    if len(names) != 2:
        raise SystemExit("--variants 需要恰好两个变体，例如 current,prompt-v1")

    # 函数说明：main.both
    # 用途：返回
    # `(await _run_variant(args, names[0]), await _run_variant(args, names[1]))`，提供
    # 回归测试与测试辅助 的派生值。
    # 返回：类型 `tuple[RunReport, RunReport]`；返回
    # `(await _run_variant(args, names[0]), await _run_variant(args, names[1]))`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_run_variant`。
    # 闭包依赖：从外层读取 `args`、`names`。
    async def both() -> tuple[RunReport, RunReport]:
        return await _run_variant(args, names[0]), await _run_variant(args, names[1])

    base, other = asyncio.run(both())
    path = args.out / (
        f"{base.started_at.replace(':', '').replace('-', '')[:15]}"
        f"-{names[0]}-vs-{names[1]}.md"
    )
    path.write_text(render_comparison(base, other), encoding="utf-8")
    print(f"对照：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
