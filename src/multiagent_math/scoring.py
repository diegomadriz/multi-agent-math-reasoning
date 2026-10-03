"""Answer parsing and exact-match scoring (same rules as the experiment)."""

from __future__ import annotations

import re

# First `#### <number>` marker; signed integers or decimals only.
# Commas, fractions and scientific notation are not parsed (a known limitation).
_ANS_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def extract_final_number(text: str | None) -> str | None:
    if not text:
        return None
    m = _ANS_RE.search(text)
    return m.group(1) if m else None


def is_correct(pred: str | None, gold: str | None) -> bool:
    return pred is not None and gold is not None and float(pred) == float(gold)
