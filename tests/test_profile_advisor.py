"""Phase 4: profile -> advisor boundary integration tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import WorkloadObservation, WorkloadProfile, build_workload_profile
from inferpilot.advisor import (
    AdvisorPolicy, ProfileAdvisorDecision, ProfileContext, advise_from_profile,
)


def _policy() -> AdvisorPolicy:
    return AdvisorPolicy(
        policy_id="m3", model="q", revision="abc", gpu_name="g",
        prompt_tokens=128, output_tokens=32, arrival_pattern="poisson-v1",
        fixed_engine_fields={"max_num_batched_tokens": 512},
        regimes=[{"request_rate_qps": 2, "engine_overrides": {"max_num_seqs": 2}},
                 {"request_rate_qps": 6, "engine_overrides": {"max_num_seqs": 4}}],
        evidence_report_sha256="a" * 64,
    )


def _ctx(**kw) -> ProfileContext:
    base = dict(model="q", revision="abc", gpu_name="g",
                arrival_pattern="poisson-v1", declared_fixed_length=True)
    base.update(kw)
    return ProfileContext(**base)


def _obs(t, p=128, o=32):
    return WorkloadObservation(arrival_offset_s=t, prompt_tokens=p, output_tokens=o, success=True)


def _profile_rate2() -> WorkloadProfile:
    # 5 arrivals at 0,0.5,1.0,1.5,2.0 -> realized rate exactly 2.0, fixed 128/32.
    return build_workload_profile([_obs(i * 0.5) for i in range(5)])


def test_exact_fixed_length_supported_recommends() -> None:
    d = advise_from_profile(_policy(), _profile_rate2(), _ctx())
    assert d.status == "recommended"
    assert d.engine_overrides == {"max_num_batched_tokens": 512, "max_num_seqs": 2}
    assert d.profile_provenance_sha256 == _profile_rate2().provenance_sha256
    assert d.decision is not None and d.decision.request.request_rate_qps == 2.0


def test_nearby_rate_abstains_no_mapping() -> None:
    # arrivals giving realized rate ~2.14 (not exactly 2 or 6) -> abstain, never mapped.
    prof = build_workload_profile([_obs(0.0), _obs(0.5), _obs(1.0), _obs(1.4)])
    assert prof.realized_request_rate_qps != 2.0
    d = advise_from_profile(_policy(), prof, _ctx())
    assert d.status == "abstain" and "unsupported_request_rate_qps" in d.reasons
    assert d.engine_overrides is None


def test_mixed_prompt_lengths_abstain() -> None:
    prof = build_workload_profile([_obs(0.0, p=128), _obs(0.5, p=256), _obs(1.0, p=128),
                                   _obs(1.5, p=128), _obs(2.0, p=128)])
    d = advise_from_profile(_policy(), prof, _ctx())
    assert d.status == "abstain" and d.reasons == ["mixed_prompt_lengths"] and d.decision is None


def test_mixed_output_lengths_abstain() -> None:
    prof = build_workload_profile([_obs(i * 0.5, o=32 if i else 33) for i in range(5)])
    d = advise_from_profile(_policy(), prof, _ctx())
    assert d.status == "abstain" and d.reasons == ["mixed_output_lengths"]


def test_not_declared_fixed_length_abstains() -> None:
    d = advise_from_profile(_policy(), _profile_rate2(), _ctx(declared_fixed_length=False))
    assert d.status == "abstain" and d.reasons == ["workload_not_declared_fixed_length"]


@pytest.mark.parametrize("change,reason", [
    ({"gpu_name": "other"}, "unsupported_gpu_name"),
    ({"model": "other"}, "unsupported_model"),
    ({"revision": "other"}, "unsupported_revision"),
    ({"arrival_pattern": "batched-poisson-v1"}, "unsupported_arrival_pattern"),
])
def test_context_mismatch_abstains(change, reason) -> None:
    d = advise_from_profile(_policy(), _profile_rate2(), _ctx(**change))
    assert d.status == "abstain" and reason in d.reasons


def test_roundtrip_and_tampered_decision_rejected() -> None:
    d = advise_from_profile(_policy(), _profile_rate2(), _ctx())
    assert ProfileAdvisorDecision.model_validate_json(d.model_dump_json()) == d
    raw = d.model_dump(mode="json")
    raw["engine_overrides"]["max_num_seqs"] = 4  # disagree with embedded decision
    with pytest.raises(ValidationError, match="inconsistent"):
        ProfileAdvisorDecision.model_validate(raw)


def test_tampered_profile_evidence_rejected() -> None:
    raw = _profile_rate2().model_dump(mode="json")
    raw["observations"][2]["prompt_tokens"] = 999  # breaks digest + percentiles
    with pytest.raises(ValidationError, match="inconsistent with observations"):
        WorkloadProfile.model_validate(raw)
