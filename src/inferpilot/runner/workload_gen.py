"""Workload generation (prompt construction) — a concern of its own.

Turns a declarative :class:`WorkloadSpec` into concrete prompt strings. Kept
separate from the client (which only sends and measures) and from generation
parameters. Generation is deterministic in ``seed`` so runs are reproducible.

Token counts are approximate: without loading the model tokenizer we cannot hit
an exact prompt length, and the workload only asks for *approximately* N input
tokens. The client records the server's actual counted tokens, so the aggregate
numbers use real counts regardless of this approximation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..workload import WorkloadSpec

# A small fixed vocabulary keeps prompts word-like and roughly one token/word.
_VOCAB = (
    "the quick brown fox jumps over a lazy dog while system latency and "
    "throughput are measured under a controlled inference workload today"
).split()


def _make_prompt(rng: random.Random, approx_tokens: int) -> str:
    words = [rng.choice(_VOCAB) for _ in range(approx_tokens)]
    return " ".join(words)


@dataclass(frozen=True)
class GeneratedWorkload:
    """Concrete prompts split into warm-up and measured sets."""

    warmup_prompts: list[str]
    measured_prompts: list[str]


def generate_workload(workload: WorkloadSpec) -> GeneratedWorkload:
    """Deterministically build warm-up and measured prompts for ``workload``."""
    rng = random.Random(workload.effective_prompt_seed)
    warmup = [_make_prompt(rng, workload.prompt_tokens) for _ in range(workload.warmup_requests)]
    measured = [_make_prompt(rng, workload.prompt_tokens) for _ in range(workload.num_requests)]
    return GeneratedWorkload(warmup_prompts=warmup, measured_prompts=measured)
