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

    v3 = commands.add_parser("v3", help="第三阶段：36项、客观验收与独立回答质量")
    v3.add_argument("--list", action="store_true")
    v3.add_argument("--category", action="append", choices=CATEGORIES)
    v3.add_argument("--case", action="append")
    v3.add_argument("--split", choices=["all", "dev", "holdout"], default="all")
    v3.add_argument("--repeat", type=int, choices=range(1, 21), default=3)
    v3.add_argument("--provider")
    v3.add_argument("--model")
    v3.add_argument("--no-stream", action="store_true")
    v3.add_argument("--api-retries", type=int, choices=range(7), default=3)
    v3.add_argument("--variant", default="current")
    v3.add_argument("--baseline", type=Path)
    v3.add_argument("--out", type=Path)
    v3.add_argument("--skip-judge", action="store_true")
    v3.add_argument("--judge-provider")
    v3.add_argument("--judge-model")

    review = commands.add_parser("v3-review", help="仅对既有完整证据调用质量裁判，不重跑Agent")
    review.add_argument("source", type=Path)
    review.add_argument("--out", type=Path, required=True)
    review.add_argument("--limit", type=int, choices=range(1, 1001))
    review.add_argument("--judge-provider")
    review.add_argument("--judge-model")
    review.add_argument("--prepare-only", action="store_true",
                        help="只准备本地评审材料，不加载密钥、不调用外部接口")
    review.add_argument("--api-retries", type=int, choices=range(7), default=3)
    review.add_argument("--include-calibration-fixtures", action="store_true",
                        help="加入两个明确标注的假成功负例，仅用于人工校准")

    regrade_v3 = commands.add_parser("v3-regrade", help="复核V3原始证据，保留历史错误")
    regrade_v3.add_argument("source", type=Path)
    regrade_v3.add_argument("--out", type=Path, required=True)
    regrade_v3.add_argument("--verify-missing", action="store_true",
                            help="原先跳过独立验收时，从哈希匹配的最终文件补跑Docker验收")

    human = commands.add_parser("v3-human-export", help="导出空白真人标注，不显示AI预评分")
    human.add_argument("source", type=Path)
    human.add_argument("--out", type=Path, required=True)
    calibration = commands.add_parser("v3-calibrate", help="核对真人标注和AI评分，输出分歧")
    calibration.add_argument("source", type=Path)
    calibration.add_argument("annotations", type=Path)
    calibration.add_argument("--out", type=Path, required=True)

    experiment = commands.add_parser("v3-experiment", help="M03反思开关单因素真实对照")
    experiment.add_argument("--repeat", type=int, choices=range(3, 21), default=3)
    experiment.add_argument("--provider")
    experiment.add_argument("--model")
    experiment.add_argument("--api-retries", type=int, choices=range(7), default=3)
    experiment.add_argument("--no-stream", action="store_true")
    experiment.add_argument("--out", type=Path, required=True)

    regression = commands.add_parser("v3-regression-capture", help="把真实失败留档为可复现回归合同")
    regression.add_argument("source", type=Path)
    regression.add_argument("--case", required=True)
    regression.add_argument("--attempt", type=int, required=True)
    regression.add_argument("--out", type=Path, required=True)

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

    v2 = commands.add_parser("v2", help="第二阶段：24 项、默认各 3 次、稳定性与回归")
    v2.add_argument("--list", action="store_true")
    v2.add_argument("--category", action="append", choices=CATEGORIES)
    v2.add_argument("--case", action="append")
    v2.add_argument("--split", choices=["all", "dev", "holdout"], default="all")
    v2.add_argument("--repeat", type=int, choices=range(1, 21), default=3)
    v2.add_argument("--provider")
    v2.add_argument("--model")
    v2.add_argument("--no-stream", action="store_true")
    v2.add_argument("--api-retries", type=int, choices=range(7), default=3)
    v2.add_argument("--variant", default="current")
    v2.add_argument("--baseline", type=Path, help="显式指定完成的 V2 基线 JSON")
    v2.add_argument("--out", type=Path)

    v2_compare = commands.add_parser(
        "v2-compare", help="仅对比两份 V2 报告，不调用模型"
    )
    v2_compare.add_argument("base", type=Path)
    v2_compare.add_argument("candidate", type=Path)
    v2_compare.add_argument("--out", type=Path, help="对比输出目录")

    v2_regrade = commands.add_parser(
        "v2-regrade", help="根据完整证据重新评分，不调用模型"
    )
    v2_regrade.add_argument("source", type=Path)
    v2_regrade.add_argument("--out", type=Path, required=True)

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
        checkpoint=lambda report: report.save(args.out),
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
    if args.command.startswith("v3"):
        return _main_v3(args)
    if args.command == "v2-regrade":
        from .v2 import exit_code as v2_exit_code
        from .v2_regrade import regrade

        report, folder = regrade(args.source, out=args.out)
        print(f"复核报告：{folder / 'report.md'}\n回归门槛：{report['gate']['status']}")
        return v2_exit_code(report)
    if args.command == "v2-compare":
        from .v1 import write_json
        from .v2 import load_report
        from .v2_report import compare, gate
        from .v2_report import render_comparison as render_v2_comparison

        base, candidate = load_report(args.base), load_report(args.candidate)
        comparison = compare(base, candidate)
        print(render_v2_comparison(comparison))
        decision = gate(candidate, comparison)
        print("回归门槛：" + decision["status"])
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            write_json(args.out / "comparison.json", {**comparison, "gate": decision})
            (args.out / "comparison.md").write_text(
                render_v2_comparison(comparison), encoding="utf-8"
            )
        if decision["status"] == "PASS":
            return 0
        return 2 if decision["status"] == "INCOMPLETE" else 1
    if args.command == "v2":
        from .v2 import REPORTS_DIR as V2_REPORTS_DIR
        from .v2 import exit_code as v2_exit_code
        from .v2 import run_v2
        from .v2_cases import v2_cases

        known = v2_cases()
        unknown = set(args.case or ()) - {plan.case.id for plan in known}
        if unknown:
            raise SystemExit("未知的 V2 用例：" + ",".join(sorted(unknown)))
        plans = [plan for plan in known
                 if (not args.category or plan.category in args.category)
                 and (not args.case or plan.case.id in args.case)
                 and (args.split == "all" or plan.split == args.split)]
        if not plans:
            raise SystemExit("没有匹配的 V2 用例")
        if args.list:
            for plan in plans:
                needs = "（需要 Docker）" if plan.case.requires else ""
                print(
                    f"{plan.case.id}  {plan.category:12s}  {plan.split:7s}  "
                    f"{plan.case.title}{needs}"
                )
            return 0
        try:
            report, folder = asyncio.run(run_v2(
                plans=plans, provider=args.provider, model=args.model,
                repeat=args.repeat,
                non_stream=args.no_stream, api_retries=args.api_retries,
                variant=load_variant(args.variant), baseline=args.baseline,
                out=args.out or V2_REPORTS_DIR,
                progress=lambda line: print(line, flush=True),
            ))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"中文报告：{folder / 'report.md'}\n结构化报告：{folder / 'report.json'}")
        print("回归门槛：" + report["gate"]["status"])
        return v2_exit_code(report)
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


def _main_v3(args) -> int:
    from .v1 import write_json
    from .v3 import REPORTS_DIR as V3_REPORTS_DIR
    from .v3 import exit_code as v3_exit_code
    from .v3 import run_v3

    if args.command == "v3-human-export":
        from .v3_calibration import export_annotations
        data = export_annotations(args.source, args.out)
        print(f"空白真人标注：{args.out}，{len(data['samples'])}条")
        return 0
    if args.command == "v3-regrade":
        from .v3_regrade import regrade
        report, folder = asyncio.run(regrade(args.source, out=args.out, verify_missing=args.verify_missing))
        print(f"V3复核：{folder / 'report.md'}，{report['summary']['status_counts']}")
        return v3_exit_code(report)
    if args.command == "v3-calibrate":
        from .v3_calibration import calibrate
        result = calibrate(args.source, args.annotations)
        write_json(args.out, result)
        print(f"校准状态：{result['status']}，真人样本：{result['labeled_samples']}")
        return 0 if result["status"] == "CALIBRATED_SMALL_SAMPLE" else 2
    if args.command == "v3-regression-capture":
        from .v3_regression import capture
        capture(args.source, case_id=args.case, attempt=args.attempt, out=args.out)
        print(f"原始失败及最小回归合同：{args.out}")
        return 0
    if args.command == "v3-experiment":
        from .v3_experiment import run_experiment
        result, folder = asyncio.run(run_experiment(out=args.out, repeat=args.repeat,
            provider=args.provider, model=args.model, api_retries=args.api_retries, non_stream=args.no_stream))
        print(f"对照结果：{folder / 'experiment.json'}，{result['status']}")
        return 0 if result["status"] == "COMPARABLE" else 2
    if args.command == "v3-review":
        from .v3_judge import LocalPreparationJudge, QualityJudge, review_report
        async def review_existing():
            judge = (LocalPreparationJudge() if args.prepare_only else
                     QualityJudge(provider=args.judge_provider, model=args.judge_model, api_retries=args.api_retries))
            try:
                return await review_report(args.source, out=args.out, judge=judge, limit=args.limit,
                                           include_calibration_fixtures=args.include_calibration_fixtures)
            finally:
                await judge.close()
        asyncio.run(review_existing())
        print(f"质量评审：{args.out / 'quality-report.json'}；未人工校准")
        return 2
    from .v3_cases import v3_cases
    known = v3_cases()
    unknown = set(args.case or ()) - {p.case.id for p in known}
    if unknown:
        raise SystemExit("未知V3用例：" + ",".join(sorted(unknown)))
    plans = [p for p in known if (not args.case or p.case.id in args.case)
             and (not args.category or p.category in args.category)
             and (args.split == "all" or p.split == args.split)]
    if not plans:
        raise SystemExit("没有匹配V3用例")
    if args.list:
        for plan in plans:
            print(f"{plan.case.id}  {plan.category:12s}  {plan.split:7s}  {plan.case.title}")
        return 0
    report, folder = asyncio.run(run_v3(plans=plans, provider=args.provider, model=args.model,
        repeat=args.repeat, non_stream=args.no_stream, api_retries=args.api_retries,
        variant=load_variant(args.variant), baseline=args.baseline, out=args.out or V3_REPORTS_DIR,
        skip_judge=args.skip_judge, judge_provider=args.judge_provider, judge_model=args.judge_model))
    print(f"V3报告：{folder / 'report.md'}\n客观门槛：{report['objective_gate']['status']}\n整体：{report['gate']['status']}")
    return v3_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
