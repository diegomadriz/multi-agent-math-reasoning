"""Aggregate per-item results into the comparison metrics used in the report.

For N items and C correct answers at a given cap:
  accuracy             = C / N
  tokens_per_correct   = total generated tokens / C
  secs_per_correct     = total elapsed seconds / C
  error_reduction_pct  = 100 * (single error - mode error) / single error
  95% interval         = p +/- 1.96 * sqrt(p (1 - p) / N)   (pointwise Wald)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(items: pd.DataFrame) -> pd.DataFrame:
    """Per-item rows (mode, total_budget, acc, tokens, latency_s) -> one row per config."""
    rows = []
    for (mode, budget), g in items.groupby(["mode", "total_budget"]):
        correct = g["acc"].sum()
        total_lat = g["latency_s"].sum()
        total_tok = g["tokens"].sum()
        rows.append(
            {
                "mode": mode,
                "total_budget": budget,
                "n": len(g),
                "accuracy": g["acc"].mean(),
                "avg_latency_s": g["latency_s"].mean(),
                "avg_tokens": g["tokens"].mean(),
                "total_latency_s": total_lat,
                "total_tokens": total_tok,
                "secs_per_correct": total_lat / correct if correct else float("inf"),
                "tokens_per_correct": total_tok / correct if correct else float("inf"),
            }
        )
    return pd.DataFrame(rows).sort_values(["total_budget", "mode"]).reset_index(drop=True)


def enrich(summary: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """Add intervals and comparisons against `single` at the same cap."""
    df = summary.copy()
    df["mode"] = df["mode"].astype(str).str.strip()
    size = df["n"] if "n" in df and n is None else n
    if size is None:
        raise ValueError("Sample size unknown: pass n or include an 'n' column.")

    df["error_rate"] = 1.0 - df["accuracy"]
    se = np.sqrt(df["accuracy"] * (1 - df["accuracy"]) / size)
    df["ci95_low"] = (df["accuracy"] - 1.96 * se).clip(0.0, 1.0)
    df["ci95_high"] = (df["accuracy"] + 1.96 * se).clip(0.0, 1.0)

    base = df[df["mode"] == "single"][
        ["total_budget", "accuracy", "error_rate", "tokens_per_correct", "secs_per_correct"]
    ].rename(
        columns={
            "accuracy": "acc_single",
            "error_rate": "err_single",
            "tokens_per_correct": "tpc_single",
            "secs_per_correct": "spc_single",
        }
    )
    if base.empty:
        raise ValueError("No 'single' rows to compare against.")

    df = df.merge(base, on="total_budget", how="left", validate="many_to_one")
    df["delta_acc_pp"] = 100.0 * (df["accuracy"] - df["acc_single"])
    df["error_reduction_pct"] = 100.0 * (df["err_single"] - df["error_rate"]) / df["err_single"]
    df["tokens_multiplier"] = df["tokens_per_correct"] / df["tpc_single"]
    df["secs_multiplier"] = df["secs_per_correct"] / df["spc_single"]
    return df.sort_values(["total_budget", "mode"]).reset_index(drop=True)
