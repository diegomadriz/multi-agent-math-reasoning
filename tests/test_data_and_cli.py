from pathlib import Path

import pandas as pd
import pytest

from multiagent_math.cli import main
from multiagent_math.data import sample_items

CACHED_TRAIN = next(
    Path.home().glob(
        ".cache/huggingface/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
    ),
    None,
)


def test_sampling_is_deterministic():
    df = pd.DataFrame(
        {"question": [f"q{i}" for i in range(50)], "answer": [f"#### {i}" for i in range(50)]}
    )
    a, b = sample_items(df, 10, seed=42), sample_items(df, 10, seed=42)
    assert [i.question for i in a] == [i.question for i in b]
    assert a[0].gold == a[0].answer.split()[-1]


@pytest.mark.skipif(CACHED_TRAIN is None, reason="GSM8K train parquet not cached locally")
def test_reproduces_the_pilot_question_set(root):
    pilot = pd.read_csv(root / "results/pilot_n25/gsm8k_single_n25.csv")
    items = sample_items(pd.read_parquet(CACHED_TRAIN), 25, seed=42)
    assert [i.question for i in items] == list(pilot["question"])
    assert [float(i.gold) for i in items] == list(pilot["gold"].astype(float))


def test_analyze_command_writes_figures(root, tmp_path):
    rc = main(
        [
            "analyze",
            str(root / "results/benchmark_token_budget_summary.csv"),
            "--n",
            "200",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    for name in ("summary_enriched.csv", "accuracy_vs_budget.png", "cost_vs_error_reduction.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_bench_command_end_to_end_with_fake_model(tmp_path, monkeypatch):
    from conftest import FakeBackend

    import multiagent_math.cli as cli

    data = tmp_path / "gsm8k.parquet"
    pd.DataFrame(
        {"question": [f"q{i}" for i in range(6)], "answer": ["#### 8", "#### 9"] * 3}
    ).to_parquet(data)
    monkeypatch.setattr(cli, "LlamaCppBackend", lambda *a, **k: FakeBackend())

    out = tmp_path / "run"
    rc = main(
        [
            "bench",
            "--model",
            "fake.gguf",
            "--parquet",
            str(data),
            "--n",
            "4",
            "--budgets",
            "256",
            "--judge-m",
            "3",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    items = pd.read_csv(out / "items.csv")
    assert set(items["mode"]) == {"single", "mav", "prop_judge_M3_loose", "prop_judge_M3_strict"}
    assert len(items) == 16 and items["acc"].between(0, 1).all()
    summary = pd.read_csv(out / "summary.csv")
    assert (summary["n"] == 4).all()
    assert len((out / "transcripts.jsonl").read_text().splitlines()) == 16
