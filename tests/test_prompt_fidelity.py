"""The package prompts must be byte-identical to the ones that produced the results."""

import json

import pytest

from multiagent_math import architectures, prompts


@pytest.fixture(scope="module")
def notebook_ns(root):
    nb = json.loads((root / "notebooks/original_experiment.ipynb").read_text())
    wanted = (
        "def math_proposer_agent",
        "MATH_JUDGE_MAX_TOKENS =",
        "def _join_numbered",
        "def build_math_judge_prompt",
    )
    captured = {}
    ns = {"generate_response": lambda prompt, **kw: captured.setdefault("last", (prompt, kw))}
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and any(w in src for w in wanted):
            exec(src, ns)
    ns["_captured"] = captured
    return ns


def _prompt_from(ns, fn, *args):
    ns["_captured"].clear()
    ns[fn](*args)
    return ns["_captured"]["last"][0]


def test_proposer(notebook_ns):
    assert _prompt_from(notebook_ns, "math_proposer_agent", "Q") == prompts.proposer("Q")


def test_critic(notebook_ns):
    assert _prompt_from(notebook_ns, "math_critic_agent", "Q", "P") == prompts.critic("Q", "P")


def test_refiner(notebook_ns):
    assert _prompt_from(notebook_ns, "math_refiner_agent", "Q", "P", "C") == prompts.refiner(
        "Q", "P", "C"
    )


def test_judge(notebook_ns):
    assert notebook_ns["build_math_judge_prompt"]("Q", ["a", "b"]) == prompts.math_judge(
        "Q", ["a", "b"]
    )


def test_judge_decoding(notebook_ns):
    assert notebook_ns["MATH_JUDGE_MAX_TOKENS"] == architectures.JUDGE_MAX_TOKENS
    assert notebook_ns["MATH_JUDGE_TEMPERATURE"] == architectures.JUDGE_TEMPERATURE
    assert notebook_ns["MATH_JUDGE_TOP_P"] == architectures.JUDGE_TOP_P


def test_published_csv_is_what_the_notebook_printed(root):
    """The accuracy matrix printed by the notebook's analysis cell matches results/."""
    import pandas as pd

    nb = json.loads((root / "notebooks/original_experiment.ipynb").read_text())
    printed = "".join(
        "".join(o.get("text", []))
        for c in nb["cells"]
        if c["cell_type"] == "code"
        for o in c["outputs"]
        if o["output_type"] == "stream"
    )
    block = printed.split("=== accuracy (rows=mode, cols=budget) ===")[1].split("===")[0]
    matrix = {}
    for line in block.strip().splitlines()[2:]:
        mode, *vals = line.split()
        matrix[mode] = [float(v) for v in vals]

    csv = pd.read_csv(root / "results/benchmark_token_budget_summary.csv")
    assert len(matrix) == 6
    for mode, vals in matrix.items():
        got = csv[csv["mode"] == mode].sort_values("total_budget")["accuracy"].round(3).tolist()
        assert got == vals, mode
