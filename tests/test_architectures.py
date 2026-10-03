import pytest

from multiagent_math.architectures import (
    JUDGE_MAX_TOKENS,
    judge_budgets,
    run_mav,
    run_proposer_judge,
    run_single,
)


def test_single_is_one_proposer_call(fake):
    res = run_single(fake, "Q?", 256)
    assert [c.role for c in res.calls] == ["proposer"]
    assert fake.calls[0][1].max_tokens == 256
    assert res.pred == "8" and res.mode == "single"


def test_mav_chains_outputs_and_caps_each_call(fake):
    fake.answers = ["PROPOSAL #### 7", "CRITIQUE", "REFINED #### 8"]
    res = run_mav(fake, "Q?", 750)
    assert [c.role for c in res.calls] == ["proposer", "critic", "refiner"]
    assert all(s.max_tokens == 750 for _, s in fake.calls)
    assert "PROPOSAL #### 7" in fake.calls[1][0]
    assert "PROPOSAL #### 7" in fake.calls[2][0] and "CRITIQUE" in fake.calls[2][0]
    # The final answer comes from the Refiner, not the Proposer.
    assert res.pred == "8"
    assert res.tokens == sum(c.tokens for c in res.calls)


@pytest.mark.parametrize(
    "budget,m,strict,expected",
    [
        (256, 3, True, (64, 64)),
        (256, 5, True, (42, 42)),
        (1536, 3, False, (1536, JUDGE_MAX_TOKENS)),
        (750, 5, False, (750, 64)),
    ],
)
def test_judge_budget_allocation(budget, m, strict, expected):
    assert judge_budgets(budget, m, strict) == expected


def test_judge_sees_all_candidates_and_uses_its_own_sampling(fake):
    fake.answers = ["A #### 1", "B #### 2", "C #### 3", "Chosen: 2\nFinal: #### 2"]
    res = run_proposer_judge(fake, "Q?", 256, m=3, strict=False)
    judge_prompt, judge_sampling = fake.calls[-1]
    assert all(f"Option {i}:" in judge_prompt for i in (1, 2, 3))
    assert judge_sampling.max_tokens == 64 and judge_sampling.temperature == 0.2
    assert res.mode == "prop_judge_M3_loose" and res.pred == "2"
    assert len(res.calls) == 4
