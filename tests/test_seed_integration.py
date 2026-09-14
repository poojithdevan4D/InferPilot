"""Runner + feature wiring: prompt seed drives prompts, arrival seed drives schedule."""

from __future__ import annotations

import pytest

from inferpilot import (
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    FailureRecord,
    WorkloadSpec,
)
from inferpilot.features.arrival import extract_arrival_features
from inferpilot.features.models import ArrivalEvidence
from inferpilot.runner.schedule import generate_poisson_offsets
from inferpilot.runner.workload_gen import generate_workload


def _wl(**kw) -> WorkloadSpec:
    base = dict(name="w", num_requests=3, warmup_requests=0, prompt_tokens=64,
                output_tokens=16, request_rate_qps=8.0, seed=0)
    base.update(kw)
    return WorkloadSpec(**base)


def _offsets(wl: WorkloadSpec) -> list[float]:
    return generate_poisson_offsets(wl.num_requests, wl.request_rate_qps, wl.effective_arrival_seed)


def test_prompt_seed_drives_prompts_arrival_seed_drives_schedule() -> None:
    # same prompt seed, different arrival seed -> identical prompts, different offsets
    a = _wl(prompt_seed=1, arrival_seed=5)
    b = _wl(prompt_seed=1, arrival_seed=9)
    assert generate_workload(a).measured_prompts == generate_workload(b).measured_prompts
    assert _offsets(a) != _offsets(b)

    # different prompt seed, same arrival seed -> different prompts, identical offsets
    c = _wl(prompt_seed=2, arrival_seed=5)
    assert generate_workload(a).measured_prompts != generate_workload(c).measured_prompts
    assert _offsets(a) == _offsets(c)


def test_legacy_seed_drives_both() -> None:
    wl = _wl(seed=3)  # both split seeds omitted
    assert wl.effective_prompt_seed == 3 and wl.effective_arrival_seed == 3
    # a 0.4.0 workload relying only on `seed` reproduces the legacy single-seed run
    same = _wl(prompt_seed=3, arrival_seed=3)
    assert generate_workload(wl).measured_prompts == generate_workload(same).measured_prompts
    assert _offsets(wl) == _offsets(same)


# --- arrival-feature provenance binding ------------------------------------- #


def _open_result(schema: str, **wlkw) -> ExperimentResult:
    cfg = ExperimentConfig(
        schema_version=schema, experiment_id="e", name="e",
        engine=EngineConfig(model="org/m", revision="abc", max_model_len=2048, max_num_seqs=1,
                            gpu_memory_utilization=0.85, enable_prefix_caching=False),
        workload=_wl(**wlkw),
    )
    return ExperimentResult(
        schema_version=schema, config=cfg, environment=EnvironmentMetadata(hostname="h"),
        status=ExperimentStatus.OOM,
        failure=FailureRecord(status=ExperimentStatus.OOM, error_type="X", message="m"),
    )


def _evidence(seed: int, arrival_seed=None) -> ArrivalEvidence:
    return ArrivalEvidence(
        algorithm="poisson-v1", request_rate_qps=8.0, seed=seed, arrival_seed=arrival_seed,
        scheduled_offsets_s=[0.0, 0.1, 0.2], actual_dispatch_offsets_s=[0.0, 0.1, 0.2],
    )


def test_features_accept_matching_arrival_seed() -> None:
    result = _open_result("0.4.0", prompt_seed=1, arrival_seed=22)
    report = extract_arrival_features(
        result, _evidence(seed=22, arrival_seed=22), observation_cutoff_s=1.0
    )
    assert report.num_arrivals == 3


def test_features_reject_wrong_arrival_seed() -> None:
    result = _open_result("0.4.0", prompt_seed=1, arrival_seed=22)
    with pytest.raises(ValueError, match="arrival-seed does not match"):
        extract_arrival_features(
            result, _evidence(seed=99, arrival_seed=99), observation_cutoff_s=1.0
        )


def test_arrival_evidence_rejects_conflicting_seed_provenance() -> None:
    with pytest.raises(ValueError, match="seed and arrival_seed must agree"):
        _evidence(seed=22, arrival_seed=99)


def test_features_accept_legacy_seed_only_artifact() -> None:
    # 0.3.0-style artifact (no explicit arrival_seed); binds via its `seed`.
    result = _open_result("0.3.0", seed=7)  # no split seeds under 0.3.0
    report = extract_arrival_features(
        result, _evidence(seed=7, arrival_seed=None), observation_cutoff_s=1.0
    )
    assert report.num_arrivals == 3
