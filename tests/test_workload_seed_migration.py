"""Backward-compatible prompt/arrival seed separation (schema 0.3.0 -> 0.4.0)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import (
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    FailureRecord,
    WorkloadSpec,
)
from inferpilot.comparison.fingerprint import (
    comparison_fingerprint,
    cross_block_fingerprint,
    exact_fingerprint,
)

# Frozen 0.3.0 fingerprints captured at HEAD before the migration. If any of
# these change, a supposedly-unchanged 0.3.0 identity was rewritten -> STOP.
LEGACY_EXACT = "bd03686ee34f8cd42ebafdfbc8dbbeff1c086d63eea0b7c1baf98459b6430a19"
LEGACY_COMPARE = "6223ee0d490e238dcdd99fdc6b16134ab756519db520fd372d15de7b1385287b"
LEGACY_CROSSBLK = "ac46d28329162c35e3b14fe04f2ddcd63f95de7a10057fff6493e1737082bad5"


def _result(workload: WorkloadSpec, *, schema="0.4.0", experiment_id="e") -> ExperimentResult:
    cfg = ExperimentConfig(
        schema_version=schema, experiment_id=experiment_id, name=experiment_id,
        engine=EngineConfig(model="org/m", revision="abc", max_model_len=2048, max_num_seqs=1,
                            gpu_memory_utilization=0.85, enable_prefix_caching=False),
        workload=workload,
    )
    return ExperimentResult(
        schema_version=schema, config=cfg, environment=EnvironmentMetadata(hostname="h"),
        status=ExperimentStatus.OOM,
        failure=FailureRecord(status=ExperimentStatus.OOM, error_type="X", message="m"),
    )


def _wl(**kw) -> WorkloadSpec:
    base = dict(name="w", num_requests=32, warmup_requests=4, prompt_tokens=128,
                output_tokens=32, request_rate_qps=8.0, seed=0)
    base.update(kw)
    return WorkloadSpec(**base)


# --- 0.3.0 read support + frozen-fingerprint regression --------------------- #


def test_legacy_example_loads_unchanged() -> None:
    cfg = ExperimentConfig.model_validate_json(open("examples/example_experiment.json").read())
    assert cfg.schema_version == "0.3.0"
    assert cfg.workload.prompt_seed is None and cfg.workload.arrival_seed is None
    # legacy equivalence: both effective seeds fall back to the single seed
    assert cfg.workload.effective_prompt_seed == cfg.workload.seed
    assert cfg.workload.effective_arrival_seed == cfg.workload.seed


def test_legacy_030_fingerprints_are_byte_stable() -> None:
    cfg = ExperimentConfig.model_validate_json(open("examples/example_experiment.json").read())
    r = ExperimentResult(config=cfg, environment=EnvironmentMetadata(hostname="h"),
                         status=ExperimentStatus.OOM,
                         failure=FailureRecord(status=ExperimentStatus.OOM,
                                               error_type="X", message="m"))
    assert exact_fingerprint(r) == LEGACY_EXACT
    assert comparison_fingerprint(r, ["max_num_seqs"]) == LEGACY_COMPARE
    assert cross_block_fingerprint(r, ["max_num_seqs"]) == LEGACY_CROSSBLK


def test_030_rejects_split_seeds() -> None:
    with pytest.raises(ValidationError, match="require schema_version 0.4.0"):
        _result(_wl(prompt_seed=1), schema="0.3.0")
    with pytest.raises(ValidationError, match="require schema_version 0.4.0"):
        _result(_wl(arrival_seed=2), schema="0.3.0")


def test_unknown_schema_version_fails_loudly() -> None:
    raw = _result(_wl()).model_dump(mode="json")
    raw["schema_version"] = "0.9.9"
    with pytest.raises(ValidationError, match="unsupported schema_version"):
        ExperimentResult.model_validate(raw)


# --- 0.4.0 semantics + round-trip ------------------------------------------- #


def test_040_independent_seeds_roundtrip() -> None:
    wl = _wl(prompt_seed=11, arrival_seed=22)
    assert wl.effective_prompt_seed == 11 and wl.effective_arrival_seed == 22
    r = _result(wl)
    assert ExperimentResult.model_validate_json(r.model_dump_json()) == r


def test_040_partial_fallbacks() -> None:
    assert _wl(prompt_seed=5, seed=3).effective_arrival_seed == 3   # arrival falls back
    assert _wl(arrival_seed=7, seed=3).effective_prompt_seed == 3   # prompt falls back


# --- fingerprint compatibility-significance (req 8) ------------------------- #


def test_same_prompt_diff_arrival_seed_is_significant() -> None:
    a = _result(_wl(prompt_seed=1, arrival_seed=1), experiment_id="a")
    b = _result(_wl(prompt_seed=1, arrival_seed=2), experiment_id="b")
    assert exact_fingerprint(a) != exact_fingerprint(b)
    assert comparison_fingerprint(a, ["max_num_seqs"]) != comparison_fingerprint(b, ["max_num_seqs"])
    # ...but a blocked study strips the arrival seed -> equal across blocks.
    assert cross_block_fingerprint(a, ["max_num_seqs"]) == cross_block_fingerprint(b, ["max_num_seqs"])


def test_diff_prompt_same_arrival_seed_is_significant_even_cross_block() -> None:
    a = _result(_wl(prompt_seed=1, arrival_seed=9), experiment_id="a")
    b = _result(_wl(prompt_seed=2, arrival_seed=9), experiment_id="b")
    assert exact_fingerprint(a) != exact_fingerprint(b)
    # prompt content must stay fixed within a study's blocks -> still significant.
    assert cross_block_fingerprint(a, ["max_num_seqs"]) != cross_block_fingerprint(b, ["max_num_seqs"])


def test_burst_pattern_is_compatibility_significant() -> None:
    plain = _result(_wl(prompt_seed=1, arrival_seed=9), schema="0.5.0", experiment_id="p")
    burst = _result(
        _wl(prompt_seed=1, arrival_seed=9, arrival_pattern="batched-poisson-v1", burst_size=4),
        schema="0.5.0", experiment_id="b",
    )
    assert exact_fingerprint(plain) != exact_fingerprint(burst)
    assert comparison_fingerprint(plain, ["max_num_seqs"]) != comparison_fingerprint(
        burst, ["max_num_seqs"]
    )
