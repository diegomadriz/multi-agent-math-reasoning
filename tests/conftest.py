from pathlib import Path

import pytest

from multiagent_math.llm import Sampling

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    """Deterministic stand-in for the model: records every prompt and sampling setting."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls: list[tuple[str, Sampling]] = []

    def complete(self, prompt: str, sampling: Sampling) -> str:
        self.calls.append((prompt, sampling))
        return self.answers.pop(0) if self.answers else "working...\n#### 8"

    def count_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def fake():
    return FakeBackend()


@pytest.fixture(scope="session")
def root():
    return ROOT
