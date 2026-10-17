from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import PACKAGE, VARIANTS, command, read_json
from .dataset import DATASET, fetch_rows, load_rows, local_source
from .runner import experiment_lock, export, prepare, probe, run


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="MuHarness normal vs mini-SWE-agent: frozen SWE-bench Lite"
    )
    sub = result.add_subparsers(dest="command", required=True)
    pipeline = sub.add_parser(
        "all", help="One command: Lite30, dev gold, both agents, grade, report"
    )
    pipeline.add_argument(
        "--experiment", type=Path, default=PACKAGE.parents[3] / "output/swe-lite30"
    )
    pipeline.add_argument(
        "--config", type=Path, default=PACKAGE / "config.example.json"
    )
    pipeline.add_argument("--mini-python", default=sys.executable)
    pipeline.add_argument("--judge-python", default=sys.executable)
    pipeline.add_argument(
        "--check-only",
        action="store_true",
        help="Dependency/Docker checks only; no model calls",
    )
    prep = sub.add_parser(
        "prepare",
        help="Validate selection, separate trusted data, freeze configuration",
    )
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--config", type=Path, default=PACKAGE / "config.example.json")
    prep.add_argument(
        "--selection",
        default=str(PACKAGE / "lite30.json"),
        help="Selection JSON, or 'all' for the entire chosen split",
    )
    prep.add_argument("--split", choices=("dev", "test"), default="test")
    prep.add_argument(
        "--dataset-file",
        type=Path,
        help="Frozen official JSON/JSONL snapshot (offline)",
    )
    prep.add_argument(
        "--revision", help="Hugging Face revision; resolved to a commit before download"
    )
    prep.add_argument("--mini-python", default=sys.executable)
    prep.add_argument("--judge-python", default=sys.executable)
    check = sub.add_parser(
        "selfcheck",
        help="Check dependencies/Docker; optionally run dev gold round-trip",
    )
    check.add_argument("--mini-python", default=sys.executable)
    check.add_argument("--judge-python", default=sys.executable)
    check.add_argument("--experiment", type=Path)
    check.add_argument(
        "--instance-id",
        help="Dev gold round-trip through container export and official judge",
    )
    check.add_argument("--no-pull", action="store_true")
    check.add_argument("--timeout", type=int, default=1800)
    execute = sub.add_parser(
        "run", help="Run isolated workers, one attempt per instance and variant"
    )
    execute.add_argument("--experiment", required=True, type=Path)
    execute.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    execute.add_argument("--mini-python", default=sys.executable)
    execute.add_argument(
        "--filter",
        default=".*",
        help="Regex matched against instance_id; denominator unchanged",
    )
    execute.add_argument(
        "--resume",
        action="store_true",
        help="Skip recorded attempts, including infrastructure errors",
    )
    execute.add_argument(
        "--retry-infra",
        action="store_true",
        help="Implies --resume; re-run only attempts sealed as infrastructure "
        "errors (API/Docker/setup), at most twice; old attempts are archived",
    )
    execute.add_argument("--no-pull", action="store_true")
    recovery = sub.add_parser(
        "recover", help="Export/seal interrupted controller attempts; no model calls"
    )
    recovery.add_argument("--experiment", required=True, type=Path)
    for name in ("export", "grade", "report"):
        item = sub.add_parser(name)
        item.add_argument("--experiment", required=True, type=Path)
        if name != "report":
            item.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
        if name == "grade":
            item.add_argument("--judge-python", default=sys.executable)
            item.add_argument("--timeout", type=int, default=1800)
            item.add_argument("--max-workers", type=int, default=1)
        if name == "report":
            item.add_argument(
                "--prices",
                type=float,
                nargs=3,
                metavar=("INPUT", "CACHED_INPUT", "OUTPUT"),
                help="Optional cost per million tokens, in one chosen currency",
            )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "all":
            from .pipeline import execute

            return execute(
                args.experiment,
                config_path=args.config,
                mini_python=args.mini_python,
                judge_python=args.judge_python,
                check_only=args.check_only,
            )
        elif args.command == "prepare":
            selection = (
                None if args.selection == "all" else read_json(Path(args.selection))
            )
            if selection and (
                selection.get("split") != args.split
                or selection.get("dataset") != DATASET
            ):
                raise ValueError(
                    "Selection dataset/split does not match the experiment"
                )
            if args.dataset_file:
                if args.revision:
                    raise ValueError("Use --dataset-file or --revision, not both")
                rows, source = (
                    load_rows(args.dataset_file),
                    local_source(args.dataset_file, args.split),
                )
            else:
                rows, source = fetch_rows(args.split, args.revision)
            environments = {
                "mini": probe(args.mini_python, "mini"),
                "judge": probe(args.judge_python, "judge"),
                "muharness": probe(sys.executable, "muharness"),
            }
            result = prepare(
                args.out.resolve(),
                rows,
                selection,
                read_json(args.config),
                source,
                environments,
            )
            print(
                json.dumps(
                    {
                        "fingerprint": result["fingerprint"],
                        "selection": result["selection"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "selfcheck":
            environments = {
                "mini": probe(args.mini_python, "mini"),
                "judge": probe(args.judge_python, "judge"),
                "muharness": probe(sys.executable, "muharness"),
            }
            docker = command(["docker", "info", "--format", "{{.OSType}}"], timeout=30)
            if docker.stdout.decode().strip() != "linux":
                raise ValueError("Docker must use Linux containers")
            print(
                json.dumps(
                    {
                        "docker": "linux",
                        "versions": {
                            key: {k: v for k, v in value.items() if k != "config"}
                            for key, value in environments.items()
                        },
                    },
                    indent=2,
                )
            )
            if args.instance_id:
                if not args.experiment:
                    raise ValueError("Gold selfcheck needs --experiment (dev)")
                from .grading import selfcheck

                print(
                    json.dumps(
                        selfcheck(
                            args.experiment.resolve(),
                            args.instance_id,
                            args.judge_python,
                            pull=not args.no_pull,
                            timeout=args.timeout,
                        ),
                        indent=2,
                    )
                )
        elif args.command == "run":
            variants = list(VARIANTS) if args.variant == "both" else [args.variant]
            run(
                args.experiment.resolve(),
                variants,
                mini_python=args.mini_python,
                filter_pattern=args.filter,
                resume=args.resume,
                pull=not args.no_pull,
                retry_infra=args.retry_infra,
            )
        elif args.command == "recover":
            from .recovery import recover

            print(
                json.dumps(
                    recover(args.experiment.resolve()), ensure_ascii=False, indent=2
                )
            )
        elif args.command in ("export", "grade"):
            variants = list(VARIANTS) if args.variant == "both" else [args.variant]
            for variant in variants:
                if args.command == "export":
                    with experiment_lock(args.experiment.resolve()):
                        print(export(args.experiment.resolve(), variant))
                else:
                    if args.timeout <= 0 or args.max_workers <= 0:
                        raise ValueError("timeout and max-workers must be positive")
                    from .grading import grade

                    result = grade(
                        args.experiment.resolve(),
                        variant,
                        args.judge_python,
                        timeout=args.timeout,
                        max_workers=args.max_workers,
                    )
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    if result["missing_official"] or result["official_returncode"]:
                        return 1
        elif args.command == "report":
            from .report import build_report

            if args.prices and any(price < 0 for price in args.prices):
                raise ValueError("Token prices cannot be negative")
            with experiment_lock(args.experiment.resolve()):
                result = build_report(
                    args.experiment.resolve(),
                    prices=tuple(args.prices) if args.prices else None,
                )
            print(
                json.dumps(
                    {
                        "selected": result["selected"],
                        "paired": result["paired"],
                        "report": str(args.experiment.resolve() / "report.md"),
                    },
                    indent=2,
                )
            )
        return 0
    except subprocess.CalledProcessError as exc:
        # Include dependency/CLI diagnostics, never dump the environment.
        detail = (
            (exc.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr
        )
        print(f"{type(exc).__name__}: {detail or exc}", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
