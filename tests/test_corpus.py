"""Held-out corpus preregistration contract and fail-closed invariants."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from inferpilot.corpus import HeldOutCorpusSpec


def _study(shape: str, partition: str, seeds: tuple[int, ...]) -> dict:
    return {
        "study_id": f"m2d-{shape}-{partition}",
        "objective": "tpot_p95_ms",
        "slo": {"ttft_p95_ms": 250, "tpot_p95_ms": 10},
        "varied_engine_fields": ["max_num_seqs", "max_num_batched_tokens"],
        "blocks": [
            {
                "seed": seed,
                "experiment_ids": [
                    f"m2d-{shape}-{partition}-s{seed}-{candidate}"
                    for candidate in ("s1t2k", "s1t4k", "s4t2k", "s4t4k")
                ],
            }
            for seed in seeds
        ],
    }


def _raw() -> dict:
    tasks = []
    for shape in ("prefill", "decode", "burst"):
        tasks.append(
            {
                "task_id": f"{shape}-dev",
                "shape_id": shape,
                "partition": "development",
                "study": _study(shape, "development", (0, 1, 2)),
            }
        )
        tasks.append(
            {
                "task_id": f"{shape}-heldout",
                "shape_id": shape,
                "partition": "heldout",
                "study": _study(shape, "heldout", (5, 6, 7)),
            }
        )
    return {
        "corpus_id": "m2d",
        "varied_engine_fields": ["max_num_seqs", "max_num_batched_tokens"],
        "candidates": [
            {"candidate_id": "s1t2k", "engine_values": {"max_num_seqs": 1, "max_num_batched_tokens": 2048}},
            {"candidate_id": "s1t4k", "engine_values": {"max_num_seqs": 1, "max_num_batched_tokens": 4096}},
            {"candidate_id": "s4t2k", "engine_values": {"max_num_seqs": 4, "max_num_batched_tokens": 2048}},
            {"candidate_id": "s4t4k", "engine_values": {"max_num_seqs": 4, "max_num_batched_tokens": 4096}},
        ],
        "tasks": tasks,
        "candidate_budgets": [1, 2, 4],
        "hypotheses": ["scheduler width and token budget interact across workload shapes"],
    }


def test_corpus_roundtrip() -> None:
    spec = HeldOutCorpusSpec.model_validate(_raw())
    assert HeldOutCorpusSpec.model_validate_json(spec.model_dump_json()) == spec
    assert spec.spec_version == "0.1.0"
    assert len(spec.tasks) == 6


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(spec_version="0.2.0"), "unsupported corpus"),
        (lambda r: r["candidates"][0]["engine_values"].pop("max_num_seqs"), "exactly the varied"),
        (lambda r: r["candidates"].__setitem__(1, deepcopy(r["candidates"][0])), "candidate ids"),
        (lambda r: r["tasks"].__setitem__(1, {**r["tasks"][1], "shape_id": "other"}), "same shape ids"),
        (lambda r: r["tasks"][1]["study"]["blocks"][0].update(seed=0), "disjoint"),
        (lambda r: r["tasks"][0]["study"].update(varied_engine_fields=["max_num_seqs"]), "corpus varied"),
        (lambda r: r["tasks"][0]["study"]["blocks"][0]["experiment_ids"].pop(), "same number of candidates"),
        (lambda r: r.update(candidate_budgets=[1, 4, 2]), "distinct and increasing"),
        (lambda r: r.update(candidate_budgets=[1, 5]), "within the candidate grid"),
    ],
)
def test_invalid_corpus_is_rejected(mutate, message) -> None:
    raw = _raw()
    mutate(raw)
    with pytest.raises(ValidationError, match=message):
        HeldOutCorpusSpec.model_validate(raw)


def test_duplicate_engine_point_rejected_independently() -> None:
    raw = _raw()
    raw["candidates"][1]["engine_values"] = deepcopy(raw["candidates"][0]["engine_values"])
    with pytest.raises(ValidationError, match="engine-value points"):
        HeldOutCorpusSpec.model_validate(raw)


def test_global_experiment_id_reuse_rejected() -> None:
    raw = _raw()
    raw["tasks"][1]["study"]["blocks"][0]["experiment_ids"][0] = \
        raw["tasks"][0]["study"]["blocks"][0]["experiment_ids"][0]
    with pytest.raises(ValidationError, match="globally unique"):
        HeldOutCorpusSpec.model_validate(raw)
