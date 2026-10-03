"""Command line entry point: `mam bench`, `mam analyze`, `mam ask`."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from . import __version__
from .architectures import Result, run_mav, run_proposer_judge, run_single
from .data import load_split, sample_items
from .llm import LlamaCppBackend
from .metrics import enrich, summarize
from .scoring import is_correct

MODES = ("single", "mav", "judge_loose", "judge_strict")


def _version(dist: str) -> str | None:
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _runs(args):
    """Yield (label, callable(backend, question, budget) -> Result) for the requested modes."""
    for mode in args.modes:
        if mode == "single":
            yield "single", run_single
        elif mode == "mav":
            yield "mav", run_mav
        else:
            strict = mode == "judge_strict"
            for m in args.judge_m:
                yield (
                    f"prop_judge_M{m}_{'strict' if strict else 'loose'}",
                    lambda b, q, budget, m=m, strict=strict: run_proposer_judge(
                        b, q, budget, m, strict
                    ),
                )


def cmd_bench(args) -> int:
    out = Path(args.out or f"runs/{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    items = sample_items(load_split(args.split, args.parquet), args.n, args.seed)
    backend = LlamaCppBackend(args.model, seed=args.gen_seed)

    config = {
        "package_version": __version__,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_file": Path(args.model).name,
        "split": args.split,
        "n": args.n,
        "sample_seed": args.seed,
        "generation_seed": args.gen_seed,
        "budgets": args.budgets,
        "modes": args.modes,
        "judge_m": args.judge_m,
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "llama_cpp_python": _version("llama-cpp-python"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # Rows are written as they finish, so an interrupted multi-hour sweep keeps its results.
    fields = [
        "mode",
        "total_budget",
        "item_index",
        "gold",
        "pred",
        "acc",
        "tokens",
        "latency_s",
        "calls",
    ]
    with (
        open(out / "items.csv", "w", newline="") as items_file,
        open(out / "transcripts.jsonl", "w") as transcripts,
    ):
        writer = csv.DictWriter(items_file, fieldnames=fields)
        writer.writeheader()
        for budget in args.budgets:
            for label, run in _runs(args):
                for k, item in enumerate(items, 1):
                    res: Result = run(backend, item.question, budget)
                    acc = int(is_correct(res.pred, item.gold))
                    row = {
                        "mode": label,
                        "total_budget": budget,
                        "item_index": item.index,
                        "gold": item.gold,
                        "pred": res.pred,
                        "acc": acc,
                        "tokens": res.tokens,
                        "latency_s": res.latency_s,
                        "calls": len(res.calls),
                    }
                    writer.writerow(row)
                    items_file.flush()
                    transcripts.write(
                        json.dumps(
                            {
                                **row,
                                "question": item.question,
                                "calls": [asdict(c) for c in res.calls],
                            }
                        )
                        + "\n"
                    )
                    transcripts.flush()
                    print(
                        f"[{label} cap={budget}] {k}/{len(items)} pred={res.pred} gold={item.gold}",
                        file=sys.stderr,
                    )

    items_df = pd.read_csv(out / "items.csv")
    summary = summarize(items_df)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {out}/items.csv, summary.csv, transcripts.jsonl, config.json")
    return 0


def cmd_analyze(args) -> int:
    summary = pd.read_csv(args.summary)
    df = enrich(summary, n=args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary_enriched.csv", index=False)

    from .plots import accuracy_vs_budget, cost_vs_error_reduction

    paths = [accuracy_vs_budget(df, out), cost_vs_error_reduction(df, out)]
    cols = [
        "mode",
        "total_budget",
        "accuracy",
        "ci95_low",
        "ci95_high",
        "error_reduction_pct",
        "tokens_per_correct",
        "secs_per_correct",
        "tokens_multiplier",
        "secs_multiplier",
    ]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(df[cols].to_string(index=False))
    print("\nWrote", ", ".join(str(p) for p in [out / "summary_enriched.csv", *paths]))
    return 0


def cmd_ask(args) -> int:
    backend = LlamaCppBackend(args.model, seed=args.gen_seed)
    if args.arch == "single":
        res = run_single(backend, args.question, args.budget)
    elif args.arch == "mav":
        res = run_mav(backend, args.question, args.budget)
    else:
        res = run_proposer_judge(backend, args.question, args.budget, args.m, strict=False)
    for c in res.calls:
        print(f"\n===== {c.role} ({c.tokens} tokens, {c.latency_s:.1f}s) =====\n{c.output}")
    print(f"\nFinal answer: {res.pred}   total: {res.tokens} tokens, {res.latency_s:.1f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mam", description="Multi-agent math reasoning benchmark")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run the GSM8K architecture sweep with a local GGUF model")
    b.add_argument("--model", required=True, help="path to the GGUF file (first shard if split)")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--seed", type=int, default=42, help="question sampling seed")
    b.add_argument(
        "--gen-seed",
        type=int,
        default=None,
        help="generation seed (default: random, as in the original run)",
    )
    b.add_argument("--split", default="train", choices=["train", "test"])
    b.add_argument(
        "--parquet", default=None, help="read GSM8K from a local parquet file instead of the Hub"
    )
    b.add_argument("--budgets", type=int, nargs="+", default=[256, 750, 1536])
    b.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    b.add_argument("--judge-m", type=int, nargs="+", default=[3, 5])
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_bench)

    a = sub.add_parser(
        "analyze", help="compute comparisons and figures from a summary CSV (no model needed)"
    )
    a.add_argument("summary")
    a.add_argument(
        "--n", type=int, default=None, help="items per configuration if the CSV has no 'n' column"
    )
    a.add_argument("--out", default="figures")
    a.set_defaults(func=cmd_analyze)

    q = sub.add_parser("ask", help="run one question through an architecture and print every call")
    q.add_argument("question")
    q.add_argument("--model", required=True)
    q.add_argument("--arch", choices=["single", "mav", "judge"], default="mav")
    q.add_argument("--budget", type=int, default=750)
    q.add_argument("--m", type=int, default=3)
    q.add_argument("--gen-seed", type=int, default=None)
    q.set_defaults(func=cmd_ask)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
