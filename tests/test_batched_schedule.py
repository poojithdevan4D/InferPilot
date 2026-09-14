"""Contract and deterministic schedule tests for batched-poisson-v1."""

import pytest
from pydantic import ValidationError

from inferpilot import EngineConfig, ExperimentConfig, WorkloadSpec
from inferpilot.runner.schedule import generate_batched_poisson_offsets


def _workload(**updates) -> WorkloadSpec:
    values = dict(
        name="burst", num_requests=10, prompt_tokens=128, output_tokens=32,
        request_rate_qps=8.0, arrival_pattern="batched-poisson-v1", burst_size=4,
        prompt_seed=1, arrival_seed=2,
    )
    values.update(updates)
    return WorkloadSpec(**values)


def test_batched_schedule_is_deterministic_and_seeded() -> None:
    a = generate_batched_poisson_offsets(10, 8.0, 2, 4)
    assert a == generate_batched_poisson_offsets(10, 8.0, 2, 4)
    assert a != generate_batched_poisson_offsets(10, 8.0, 3, 4)


def test_simultaneous_batches_and_truncated_final_batch() -> None:
    offsets = generate_batched_poisson_offsets(10, 8.0, 2, 4)
    assert offsets[:4] == [0.0] * 4
    assert len(offsets) == 10
    assert len(set(offsets[4:8])) == len(set(offsets[8:])) == 1
    assert offsets[4] > 0 and offsets[8] > offsets[4]


def test_invalid_arrival_parameter_combinations_fail() -> None:
    with pytest.raises(ValidationError, match="must not define burst_size"):
        _workload(arrival_pattern="poisson-v1")
    with pytest.raises(ValidationError, match="requires burst_size"):
        _workload(burst_size=None)
    with pytest.raises(ValidationError):
        _workload(burst_size=1)


def test_legacy_schema_rejects_burst_semantics() -> None:
    with pytest.raises(ValidationError, match="require schema_version 0.5.0"):
        ExperimentConfig(
            schema_version="0.4.0", experiment_id="x", name="x",
            engine=EngineConfig(model="org/model"), workload=_workload(),
        )
