"""Online-safe workload arrival feature extraction."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from inferpilot import (
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    WorkloadSpec,
)
from inferpilot.features import (
    ArrivalEvidence,
    ArrivalFeatureReport,
    extract_arrival_features,
)


def _result(*, seed: int = 7, rate: float | None = 5.0, count: int = 4) -> ExperimentResult:
    workload_args = {
        "name": "features",
        "num_requests": count,
        "prompt_tokens": 128,
        "output_tokens": 32,
        "seed": seed,
    }
    if rate is None:
        workload_args["max_concurrency"] = 1
    else:
        workload_args["request_rate_qps"] = rate
    return ExperimentResult(
        config=ExperimentConfig(
            experiment_id="feature-source",
            name="feature-source",
            engine=EngineConfig(model="org/model"),
            workload=WorkloadSpec(**workload_args),
        ),
        environment=EnvironmentMetadata(),
        status=ExperimentStatus.PENDING,
    )


def _arrivals(actual=(0.0, 0.1, 0.4, 1.2), *, seed=7, rate=5.0):
    return {
        "algorithm": "poisson-v1",
        "request_rate_qps": rate,
        "seed": seed,
        "scheduled_offsets_s": list(actual),
        "actual_dispatch_offsets_s": list(actual),
    }


def test_prefix_features_have_explicit_window_semantics() -> None:
    report = extract_arrival_features(
        _result(), _arrivals(), observation_cutoff_s=0.5
    )
    assert report.observed_dispatch_offsets_s == [0.0, 0.1, 0.4]
    assert report.num_arrivals == 3
    assert report.arrival_count_rate_qps == 6.0
    assert report.interarrival_mean_s == pytest.approx(0.2)
    assert report.interarrival_p50_s == pytest.approx(0.2)
    assert report.interarrival_p95_s == pytest.approx(0.29)
    assert report.interarrival_cv == pytest.approx(0.5)
    assert report.interarrival_rate_qps == pytest.approx(5.0)
    assert report.max_arrivals_by_window_s == {
        "0.25": 2,
        "0.5": 3,
        "1.0": 3,
        "2.0": 3,
    }


def test_arrival_exactly_at_cutoff_is_observable() -> None:
    report = extract_arrival_features(
        _result(count=2),
        _arrivals(actual=(0.0, 0.5)),
        observation_cutoff_s=0.5,
    )
    assert report.observed_dispatch_offsets_s == [0.0, 0.5]


def test_future_arrivals_cannot_change_prefix_features() -> None:
    result = _result()
    a = extract_arrival_features(
        result, _arrivals(actual=(0.0, 0.1, 1.0, 2.0)), observation_cutoff_s=0.5
    )
    b = extract_arrival_features(
        result, _arrivals(actual=(0.0, 0.1, 10.0, 20.0)), observation_cutoff_s=0.5
    )
    assert a == b
    serialized = a.model_dump(mode="json")
    assert "ttft" not in str(serialized).lower()
    assert "tpot" not in str(serialized).lower()
    assert "latency" not in str(serialized).lower()


def test_zero_or_one_observed_arrival_has_no_interarrival_estimate() -> None:
    none = extract_arrival_features(
        _result(), _arrivals(actual=(1.0, 2.0, 3.0, 4.0)), observation_cutoff_s=0.5
    )
    assert none.num_arrivals == 0
    assert none.interarrival_mean_s is None
    assert set(none.max_arrivals_by_window_s.values()) == {0}

    one = extract_arrival_features(
        _result(), _arrivals(actual=(0.1, 2.0, 3.0, 4.0)), observation_cutoff_s=0.5
    )
    assert one.num_arrivals == 1
    assert one.interarrival_rate_qps is None


def test_simultaneous_arrivals_have_defined_zero_intervals() -> None:
    report = extract_arrival_features(
        _result(count=2), _arrivals(actual=(0.0, 0.0)), observation_cutoff_s=0.5
    )
    assert report.interarrival_mean_s == 0
    assert report.interarrival_p95_s == 0
    assert report.interarrival_cv is None
    assert report.interarrival_rate_qps is None


@pytest.mark.parametrize(
    "arrivals,message",
    [
        (_arrivals(actual=(0.0, 0.2, 0.1, 1.0)), "arrival offsets must be monotonic"),
        (_arrivals(actual=(0.0, float("nan"), 0.2, 1.0)), "finite and non-negative"),
        ({**_arrivals(), "actual_dispatch_offsets_s": [0.0]}, "equal length"),
        ({**_arrivals(), "seed": 99}, "seed does not match"),
        ({**_arrivals(), "request_rate_qps": 9.0}, "rate does not match"),
    ],
)
def test_invalid_or_mismatched_arrival_evidence_is_refused(arrivals, message) -> None:
    with pytest.raises((ValidationError, ValueError), match=message):
        extract_arrival_features(_result(), arrivals, observation_cutoff_s=0.5)


def test_closed_loop_result_is_refused() -> None:
    with pytest.raises(ValueError, match="open-loop"):
        extract_arrival_features(
            _result(rate=None), _arrivals(), observation_cutoff_s=0.5
        )


@pytest.mark.parametrize(
    "mutator,message",
    [
        (lambda raw: raw.__setitem__("report_version", "9.9.9"), "unsupported arrival"),
        (lambda raw: raw.__setitem__("num_arrivals", 99), "num_arrivals"),
        (lambda raw: raw.__setitem__("arrival_count_rate_qps", 99), "count_rate"),
        (lambda raw: raw.__setitem__("interarrival_mean_s", 99), "inter-arrival"),
        (lambda raw: raw["max_arrivals_by_window_s"].__setitem__("0.25", 99), "burst"),
    ],
)
def test_feature_report_rejects_derived_field_tampering(mutator, message) -> None:
    report = extract_arrival_features(
        _result(), _arrivals(), observation_cutoff_s=0.5
    )
    raw = deepcopy(report.model_dump(mode="json"))
    mutator(raw)
    with pytest.raises(ValidationError, match=message):
        ArrivalFeatureReport.model_validate(raw)


def test_arrival_evidence_roundtrips() -> None:
    evidence = ArrivalEvidence.model_validate(_arrivals())
    assert ArrivalEvidence.model_validate_json(evidence.model_dump_json()) == evidence
