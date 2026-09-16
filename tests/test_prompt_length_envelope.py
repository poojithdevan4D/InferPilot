"""Validated prompt-length envelope: within-evidence tolerance, still fail-closed.

Anchored to the finding in docs/experiments/2026-09-16-prompt-length-tolerance-finding.md:
the M3 policy's evidence spanned prompts of 128 AND 129 tokens (deterministic tokenizer
variance), so a policy may declare that envelope and apply inside it, while remaining
fail-closed for prompt lengths outside it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import WorkloadObservation, build_workload_profile
from inferpilot.advisor import (
    AdvisorPolicy,
    AdvisorRequest,
    ProfileContext,
    advise_from_profile,
    recommend,
)


def _policy(*, envelope: bool) -> AdvisorPolicy:
    extra = {"prompt_tokens_min": 128, "prompt_tokens_max": 129} if envelope else {}
    return AdvisorPolicy(
        policy_version="0.2.0", policy_id="m3", model="q", revision="abc",
        gpu_name="g", prompt_tokens=128, output_tokens=32,
        arrival_pattern="poisson-v1",
        fixed_engine_fields={"max_num_batched_tokens": 512},
        regimes=[
            {"request_rate_qps": 6, "min_rate_qps": 5.5, "max_rate_qps": 6.5,
             "engine_overrides": {"max_num_seqs": 4}},
        ],
        evidence_report_sha256="a" * 64, applicability_report_sha256="b" * 64,
        **extra,
    )


def _ctx() -> ProfileContext:
    return ProfileContext(model="q", revision="abc", gpu_name="g",
                          arrival_pattern="poisson-v1", declared_fixed_length=True)


def _profile(prompt_lengths):
    # arrivals at ~6 qps; prompt_lengths cycles over the given values
    n = len(prompt_lengths)
    return build_workload_profile([
        WorkloadObservation(
            arrival_offset_s=i / 6, prompt_tokens=prompt_lengths[i % n],
            output_tokens=32, success=True,
        )
        for i in range(30)
    ])


def test_envelope_admits_the_validated_128_129_mix() -> None:
    decision = advise_from_profile(_policy(envelope=True), _profile([128, 129]), _ctx())
    assert decision.status == "recommended"
    assert decision.engine_overrides == {"max_num_batched_tokens": 512, "max_num_seqs": 4}


def test_exact_policy_still_fails_closed_on_the_same_mix() -> None:
    decision = advise_from_profile(_policy(envelope=False), _profile([128, 129]), _ctx())
    assert decision.status == "abstain"
    assert decision.reasons == ["mixed_prompt_lengths"]


def test_envelope_still_fails_closed_outside_the_band() -> None:
    decision = advise_from_profile(_policy(envelope=True), _profile([128, 130]), _ctx())
    assert decision.status == "abstain"
    assert decision.reasons == ["prompt_lengths_outside_validated_envelope"]


def test_recommend_matches_inside_band_and_rejects_outside() -> None:
    policy = _policy(envelope=True)
    inside = AdvisorRequest(model="q", revision="abc", gpu_name="g",
                            arrival_pattern="poisson-v1", prompt_tokens=129,
                            output_tokens=32, request_rate_qps=6)
    assert recommend(policy, inside).status == "recommended"
    outside = inside.model_copy(update={"prompt_tokens": 130})
    out = recommend(policy, outside)
    assert out.status == "abstain" and "unsupported_prompt_tokens" in out.reasons


def test_envelope_requires_both_bounds_and_contains_nominal() -> None:
    with pytest.raises(ValidationError, match="requires both min and max"):
        AdvisorPolicy(
            policy_version="0.2.0", policy_id="m3", model="q", revision="abc",
            gpu_name="g", prompt_tokens=128, output_tokens=32,
            arrival_pattern="poisson-v1",
            fixed_engine_fields={"max_num_batched_tokens": 512},
            regimes=[{"request_rate_qps": 6, "min_rate_qps": 5.5, "max_rate_qps": 6.5,
                      "engine_overrides": {"max_num_seqs": 4}}],
            evidence_report_sha256="a" * 64, applicability_report_sha256="b" * 64,
            prompt_tokens_min=128,
        )
    with pytest.raises(ValidationError, match="must lie inside"):
        AdvisorPolicy(
            policy_version="0.2.0", policy_id="m3", model="q", revision="abc",
            gpu_name="g", prompt_tokens=128, output_tokens=32,
            arrival_pattern="poisson-v1",
            fixed_engine_fields={"max_num_batched_tokens": 512},
            regimes=[{"request_rate_qps": 6, "min_rate_qps": 5.5, "max_rate_qps": 6.5,
                      "engine_overrides": {"max_num_seqs": 4}}],
            evidence_report_sha256="a" * 64, applicability_report_sha256="b" * 64,
            prompt_tokens_min=130, prompt_tokens_max=140,
        )
