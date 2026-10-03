"""Figures for the README and report, drawn from an enriched summary table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABELS = {
    "single": "Single",
    "mav": "MAV (Proposer → Critic → Refiner)",
    "prop_judge_M3_loose": "Judge ×3 (loose)",
    "prop_judge_M5_loose": "Judge ×5 (loose)",
    "prop_judge_M3_strict": "Judge ×3 (strict)",
    "prop_judge_M5_strict": "Judge ×5 (strict)",
}
MAIN_MODES = ["single", "mav", "prop_judge_M3_loose", "prop_judge_M5_loose"]
COLORS = {
    "single": "#4c72b0",
    "mav": "#dd8452",
    "prop_judge_M3_loose": "#55a868",
    "prop_judge_M5_loose": "#c44e52",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def accuracy_vs_budget(df: pd.DataFrame, out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for mode in [m for m in MAIN_MODES if m in set(df["mode"])]:
        g = df[df["mode"] == mode].sort_values("total_budget")
        x = g["total_budget"].astype(str)
        color = COLORS.get(mode)
        ax.plot(x, 100 * g["accuracy"], marker="o", color=color, label=LABELS.get(mode, mode))
        ax.fill_between(x, 100 * g["ci95_low"], 100 * g["ci95_high"], color=color, alpha=0.08)
    ax.set_xlabel("Output cap (max_tokens)")
    ax.set_ylabel("Accuracy on GSM8K sample (%)")
    ax.set_title("Accuracy by architecture and output cap (95% Wald bands)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = out / "accuracy_vs_budget.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def cost_vs_error_reduction(df: pd.DataFrame, out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for mode in [m for m in MAIN_MODES if m in set(df["mode"]) and m != "single"]:
        g = df[df["mode"] == mode].sort_values("total_budget")
        ax.plot(
            g["tokens_multiplier"],
            g["error_reduction_pct"],
            marker="o",
            color=COLORS.get(mode),
            label=LABELS[mode],
        )
        for _, r in g.iterrows():
            ax.annotate(
                str(int(r["total_budget"])),
                (r["tokens_multiplier"], r["error_reduction_pct"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Generated tokens per correct answer, × Single")
    ax.set_ylabel("Error reduction vs Single (%)")
    ax.set_title("What the extra accuracy costs (labels = output cap)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "cost_vs_error_reduction.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
