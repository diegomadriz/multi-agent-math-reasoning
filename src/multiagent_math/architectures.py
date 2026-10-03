"""The orchestration strategies compared in the experiment.

All of them use one backend; they differ in which roles are called, in what
order, and how a nominal output cap ("budget") is allocated across calls:

- single: one Proposer call with the full cap.
- mav: Proposer -> Critic -> Refiner, each call capped at the full budget.
- judge (loose): M Proposers each capped at the full budget, then a Judge
  capped at 64 tokens.
- judge (strict): the M + 1 calls share the budget, floor(budget / (M + 1)) each.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts
from .llm import Backend, Call, Sampling, timed_call
from .scoring import extract_final_number

JUDGE_MAX_TOKENS = 64
JUDGE_TEMPERATURE = 0.2
JUDGE_TOP_P = 0.95


@dataclass
class Result:
    mode: str
    total_budget: int
    pred: str | None
    calls: list[Call] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(c.tokens for c in self.calls)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)


def run_single(backend: Backend, question: str, budget: int) -> Result:
    call = timed_call(backend, "proposer", prompts.proposer(question), Sampling(budget))
    return Result("single", budget, extract_final_number(call.output), [call])


def run_mav(backend: Backend, question: str, budget: int) -> Result:
    proposal = timed_call(backend, "proposer", prompts.proposer(question), Sampling(budget))
    critique = timed_call(
        backend, "critic", prompts.critic(question, proposal.output), Sampling(budget)
    )
    final = timed_call(
        backend,
        "refiner",
        prompts.refiner(question, proposal.output, critique.output),
        Sampling(budget),
    )
    return Result("mav", budget, extract_final_number(final.output), [proposal, critique, final])


def judge_budgets(total_budget: int, m: int, strict: bool) -> tuple[int, int]:
    """Return (per-proposer cap, judge cap) for a Proposers + Judge run."""
    if strict:
        per_call = max(1, total_budget // (m + 1))
        return per_call, per_call
    return total_budget, JUDGE_MAX_TOKENS


def run_proposer_judge(
    backend: Backend, question: str, total_budget: int, m: int, strict: bool
) -> Result:
    prop_budget, judge_budget = judge_budgets(total_budget, m, strict)
    calls = [
        timed_call(backend, f"proposer_{k + 1}", prompts.proposer(question), Sampling(prop_budget))
        for k in range(m)
    ]
    judge = timed_call(
        backend,
        "judge",
        prompts.math_judge(question, [c.output for c in calls]),
        Sampling(judge_budget, temperature=JUDGE_TEMPERATURE, top_p=JUDGE_TOP_P),
    )
    mode = f"prop_judge_M{m}_" + ("strict" if strict else "loose")
    return Result(mode, total_budget, extract_final_number(judge.output), [*calls, judge])
