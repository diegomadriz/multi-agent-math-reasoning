"""GSM8K sampling.

The reported results use 200 items from the *train* split sampled with
`pandas.DataFrame.sample(random_state=42)`. Using the same split, size and seed
reproduces the same question set. For new studies prefer `--split test`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .scoring import extract_final_number


@dataclass(frozen=True)
class Item:
    index: int
    question: str
    answer: str

    @property
    def gold(self) -> str | None:
        return extract_final_number(self.answer)


def load_split(split: str = "train", parquet_path: str | None = None) -> pd.DataFrame:
    if parquet_path:
        return pd.read_parquet(parquet_path)
    from datasets import load_dataset  # optional dependency: pip install .[data]

    return load_dataset("openai/gsm8k", "main", split=split).to_pandas()


def sample_items(df: pd.DataFrame, n: int, seed: int = 42) -> list[Item]:
    sample = df.sample(n=n, random_state=seed)
    return [
        Item(int(i), q, a)
        for i, q, a in zip(sample.index, sample["question"], sample["answer"], strict=True)
    ]
