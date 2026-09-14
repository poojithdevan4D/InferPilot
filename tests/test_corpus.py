"""Held-out corpus preregistration contract and fail-closed invariants."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from inferpilot import EngineConfig, ExperimentConfig, WorkloadSpec
from inferpilot.corpus import HeldOutCorpusSpec, validate_corpus_configs


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


def _configs(spec: HeldOutCorpusSpec) -> dict[str, ExperimentConfig]:
    configs: dict[str, ExperimentConfig] = {}
    prompt_seed_by_partition = {"development": 101, "heldout": 202}
    shape_lengths = {"prefill": (1024, 16), "decode": (64, 256), "burst": (128, 32)}
    for task in spec.tasks:
        prompt_tokens, output_tokens = shape_lengths[task.shape_id]
        for block in task.study.blocks:
            for position, experiment_id in enumerate(block.experiment_ids):
                candidate = spec.candidates[position].engine_values
                configs[experiment_id] = ExperimentConfig(
                    schema_version="0.5.0",
                    experiment_id=experiment_id,
                    name=experiment_id,
                    engine=EngineConfig(
                        model="org/model", revision="abc", max_model_len=2048,
                        max_num_seqs=candidate["max_num_seqs"],
                        max_num_batched_tokens=candidate["max_num_batched_tokens"],
                        gpu_memory_utilization=0.85, enable_prefix_caching=False,
                        sampler_backend="pytorch",
                    ),
                    workload=WorkloadSpec(
                        name=task.shape_id, num_requests=64, warmup_requests=4,
                        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                        request_rate_qps=6.0, seed=0,
                        prompt_seed=prompt_seed_by_partition[task.partition],
                        arrival_seed=block.seed, ignore_eos=True,
                    ),
                )
    return configs


def test_generated_configs_bind_to_corpus() -> None:
    spec = HeldOutCorpusSpec.model_validate(_raw())
    validate_corpus_configs(spec, _configs(spec))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c, _s: c.pop(next(iter(c))), "config ids mismatch"),
        (lambda c, _s: setattr(next(iter(c.values())).workload, "arrival_seed", 999), "arrival_seed"),
        (lambda c, _s: setattr(next(iter(c.values())).workload, "prompt_seed", None), "set prompt_seed"),
        (lambda c, _s: setattr(next(iter(c.values())).engine, "max_num_seqs", 99), "candidate position"),
        (lambda c, _s: setattr(next(iter(c.values())).workload, "output_tokens", 17), "context must match"),
    ],
)
def test_config_binding_rejects_drift(mutate, message) -> None:
    spec = HeldOutCorpusSpec.model_validate(_raw())
    configs = _configs(spec)
    mutate(configs, spec)
    with pytest.raises(ValueError, match=message):
        validate_corpus_configs(spec, configs)
