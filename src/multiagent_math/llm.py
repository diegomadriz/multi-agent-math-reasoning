"""Model backends.

Every agent role calls the same backend; only the prompt changes. The llama.cpp
backend reproduces the settings used in the experiment (Qwen2.5-7B-Instruct,
Q4_K_M GGUF, 4,096-token context, batch 512, all layers offloaded).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Sampling:
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = 50


@dataclass(frozen=True)
class Call:
    """One model call as recorded in a transcript."""

    role: str
    prompt: str
    output: str
    tokens: int
    latency_s: float
    max_tokens: int


class Backend(Protocol):
    def complete(self, prompt: str, sampling: Sampling) -> str: ...

    def count_tokens(self, text: str) -> int: ...


class LlamaCppBackend:
    """In-process llama.cpp model, loaded on first use."""

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_gpu_layers: int = 999,
        seed: int | None = None,
    ) -> None:
        self.model_path = model_path
        self._kwargs = dict(
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            chat_format="qwen",
            verbose=False,
        )
        if seed is not None:
            self._kwargs["seed"] = seed
        self._llm = None

    def _model(self):
        if self._llm is None:
            from llama_cpp import Llama  # optional dependency: pip install .[llama]

            self._llm = Llama(model_path=self.model_path, **self._kwargs)
        return self._llm

    def complete(self, prompt: str, sampling: Sampling) -> str:
        args = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p),
            "max_tokens": int(sampling.max_tokens),
            "stream": False,
        }
        if sampling.top_k is not None:
            args["top_k"] = int(sampling.top_k)
        out = self._model().create_chat_completion(**args)
        return out["choices"][0]["message"]["content"].strip()

    def count_tokens(self, text: str) -> int:
        # Same counting as the experiment: the model tokenizer on the generated text.
        return len(self._model().tokenize(text.encode("utf-8")))


def timed_call(backend: Backend, role: str, prompt: str, sampling: Sampling) -> Call:
    t0 = time.perf_counter()
    output = backend.complete(prompt, sampling)
    latency = time.perf_counter() - t0
    return Call(
        role=role,
        prompt=prompt,
        output=output,
        tokens=backend.count_tokens(output),
        latency_s=latency,
        max_tokens=sampling.max_tokens,
    )
