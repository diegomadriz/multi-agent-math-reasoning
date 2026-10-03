import pandas as pd
import pytest

from multiagent_math.metrics import enrich, summarize


@pytest.fixture
def published(root):
    return enrich(pd.read_csv(root / "results/benchmark_token_budget_summary.csv"), n=200)


def row(df, mode, budget):
    return df[(df["mode"] == mode) & (df["total_budget"] == budget)].iloc[0]


def test_reproduces_headline_numbers(published):
    mav, single = row(published, "mav", 1536), row(published, "single", 1536)
    assert mav["accuracy"] == pytest.approx(0.92)
    assert single["accuracy"] == pytest.approx(0.795)
    assert mav["error_reduction_pct"] == pytest.approx(60.976, abs=1e-3)
    assert mav["tokens_multiplier"] == pytest.approx(3.98, abs=5e-3)
    assert mav["secs_multiplier"] == pytest.approx(4.46, abs=5e-3)


def test_matches_report_latex_table(published):
    # Values from results/table_main_results.tex
    assert row(published, "single", 256)["ci95_low"] == pytest.approx(0.701, abs=1e-3)
    assert row(published, "single", 256)["ci95_high"] == pytest.approx(0.819, abs=1e-3)
    assert row(published, "prop_judge_M3_loose", 750)["error_reduction_pct"] == pytest.approx(
        33.333, abs=1e-3
    )
    assert row(published, "mav", 256)["error_reduction_pct"] == pytest.approx(58.333, abs=1e-3)


def test_strict_judge_is_the_negative_result(published):
    assert row(published, "prop_judge_M5_strict", 256)["accuracy"] == pytest.approx(0.35)
    assert row(published, "prop_judge_M5_strict", 256)["error_reduction_pct"] < 0


def test_summarize_per_correct_costs():
    items = pd.DataFrame(
        {
            "mode": ["single"] * 4,
            "total_budget": [256] * 4,
            "acc": [1, 1, 0, 0],
            "tokens": [10, 20, 30, 40],
            "latency_s": [1.0, 1.0, 1.0, 1.0],
        }
    )
    s = summarize(items).iloc[0]
    assert s["accuracy"] == 0.5 and s["n"] == 4
    assert s["tokens_per_correct"] == 50 and s["secs_per_correct"] == 2.0
