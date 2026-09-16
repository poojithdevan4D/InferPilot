"""Phase 2: workload-observation + derived profile contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import WorkloadObservation, WorkloadProfile, build_workload_profile
from inferpilot.workload_profile import nearest_rank


def _obs(t, p=128, o=32, success=True, c=None):
    return WorkloadObservation(arrival_offset_s=t, prompt_tokens=p, output_tokens=o,
                               success=success, completion_offset_s=c)


# --- nearest-rank convention ------------------------------------------------ #


def test_nearest_rank_semantics() -> None:
    v = [10, 20, 30, 40]
    assert nearest_rank(v, 50) == 20    # ceil(2)-1 = index 1
    assert nearest_rank(v, 95) == 40    # ceil(3.8)-1 = index 3 (max)
    assert nearest_rank(v, 0) == 10
    assert nearest_rank([7], 95) == 7


# --- positive + fixed-length exactness -------------------------------------- #


def test_fixed_length_uniform_arrivals_profile() -> None:
    obs = [_obs(i * 0.5) for i in range(5)]  # 0,0.5,1.0,1.5,2.0 -> exact 2.0 qps
    prof = build_workload_profile(obs)
    assert prof.observation_count == 5 and prof.successful_count == 5
    assert prof.window_duration_s == 2.0
    assert prof.realized_request_rate_qps == 2.0        # (5-1)/2.0
    assert prof.prompt_tokens_p50 == 128 and prof.prompt_tokens_p95 == 128
    assert prof.output_tokens_p50 == 32 and prof.output_tokens_p95 == 32
    assert prof.interarrival_p50_s == 0.5 and prof.interarrival_p95_s == 0.5
    assert prof.interarrival_cv == 0.0                  # all gaps equal -> zero spread
    assert prof.simultaneous_arrival_fraction == 0.0


def test_roundtrip_and_determinism() -> None:
    obs = [_obs(i * 0.5, p=100 + i, o=16) for i in range(6)]
    a = build_workload_profile(obs)
    b = build_workload_profile(list(obs))
    assert a.model_dump_json() == b.model_dump_json()          # byte-stable
    assert WorkloadProfile.model_validate_json(a.model_dump_json()) == a
    assert a.provenance_sha256 == b.provenance_sha256


# --- boundary / insufficient-vs-zero --------------------------------------- #


def test_empty_and_single_are_insufficient_not_zero() -> None:
    empty = build_workload_profile([])
    assert empty.observation_count == 0
    assert empty.realized_request_rate_qps is None        # insufficient, not 0.0
    assert empty.prompt_tokens_p50 is None
    assert empty.interarrival_p50_s is None and empty.interarrival_cv is None
    assert empty.simultaneous_arrival_fraction is None

    one = build_workload_profile([_obs(0.0)])
    assert one.window_duration_s == 0.0
    assert one.realized_request_rate_qps is None          # <2 obs -> undefined
    assert one.prompt_tokens_p50 == 128                   # a percentile of 1 obs is defined
    assert one.simultaneous_arrival_fraction is None


def test_simultaneous_arrivals_no_div0_and_fraction() -> None:
    # duplicate arrival offsets (batch): zero-duration gaps, no division by zero
    obs = [_obs(0.0), _obs(0.0), _obs(0.0), _obs(1.0)]
    prof = build_workload_profile(obs)
    assert prof.window_duration_s == 1.0
    assert prof.realized_request_rate_qps == 3.0          # (4-1)/1.0
    assert prof.simultaneous_arrival_fraction == pytest.approx(2 / 3)  # 2 of 3 gaps are 0
    assert prof.interarrival_cv is not None               # one positive gap -> cv 0.0
    assert prof.interarrival_cv == 0.0


def test_all_simultaneous_zero_window() -> None:
    obs = [_obs(0.0), _obs(0.0), _obs(0.0)]
    prof = build_workload_profile(obs)
    assert prof.window_duration_s == 0.0
    assert prof.realized_request_rate_qps is None         # zero-duration window -> undefined
    assert prof.simultaneous_arrival_fraction == 1.0      # all gaps zero (valid 1.0)
    assert prof.interarrival_cv is None                   # no positive gaps -> insufficient


# --- rejection -------------------------------------------------------------- #


def test_non_monotonic_offsets_rejected() -> None:
    # rejected both at build time (_derive) and on load (model validator).
    with pytest.raises((ValueError, ValidationError), match="non-decreasing"):
        build_workload_profile([_obs(1.0), _obs(0.5)])
    good = build_workload_profile([_obs(0.0), _obs(0.5)]).model_dump(mode="json")
    good["observations"] = list(reversed(good["observations"]))
    with pytest.raises(ValidationError, match="non-decreasing"):
        WorkloadProfile.model_validate(good)


def test_impossible_completion_offset_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot precede"):
        WorkloadObservation(arrival_offset_s=1.0, prompt_tokens=1, output_tokens=1,
                            success=True, completion_offset_s=0.5)


def test_negative_tokens_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkloadObservation(arrival_offset_s=0.0, prompt_tokens=-1, output_tokens=1, success=True)


def test_unsupported_version_rejected() -> None:
    raw = build_workload_profile([_obs(0.0), _obs(0.5)]).model_dump(mode="json")
    raw["profile_version"] = "9.9.9"
    with pytest.raises(ValidationError, match="unsupported workload profile_version"):
        WorkloadProfile.model_validate(raw)


_DERIVED_TAMPERS = [
    ("observation_count", 99),
    ("successful_count", 0),
    ("window_duration_s", 5.0),
    ("realized_request_rate_qps", 6.0),
    ("prompt_tokens_p95", 999),
    ("output_tokens_p50", 1),
    ("interarrival_p50_s", 9.9),
    ("interarrival_cv", 0.5),
    ("simultaneous_arrival_fraction", 1.0),
    ("provenance_sha256", "0" * 64),
]


@pytest.mark.parametrize("field,bad", _DERIVED_TAMPERS, ids=[t[0] for t in _DERIVED_TAMPERS])
def test_single_derived_field_tamper_rejected(field, bad) -> None:
    raw = build_workload_profile([_obs(i * 0.5, p=100 + i) for i in range(5)]).model_dump(mode="json")
    raw[field] = bad
    with pytest.raises(ValidationError, match="inconsistent with observations"):
        WorkloadProfile.model_validate(raw)


def test_tampered_observation_breaks_digest() -> None:
    raw = build_workload_profile([_obs(i * 0.5) for i in range(5)]).model_dump(mode="json")
    raw["observations"][2]["prompt_tokens"] = 999  # digest + percentiles no longer match
    with pytest.raises(ValidationError, match="inconsistent with observations"):
        WorkloadProfile.model_validate(raw)
