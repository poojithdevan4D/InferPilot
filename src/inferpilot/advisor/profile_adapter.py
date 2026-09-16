"""Bind a passive WorkloadProfile to the evidence-bound advisor (M5 phase 4).

This adapter only constructs an :class:`AdvisorRequest` when the profile and
caller-supplied context provide every field the policy requires. It never rounds,
maps to a nearest regime, bridges an unvalidated gap, or extrapolates: the observed
realized rate must satisfy the policy's evidence boundary or the adapter abstains
with the advisor's existing unsupported-rate reason. Prompt and
output token summaries are never treated as fixed request lengths unless the
caller declares a fixed-length workload AND every observation matches. The
resulting decision is bound to the profile's provenance digest.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, ValidationError, model_validator

from .._base import SchemaModel
from ..workload_profile import WorkloadProfile
from .models import AdvisorDecision, AdvisorPolicy, AdvisorRequest
from .recommend import recommend

PROFILE_DECISION_VERSION = "0.1.1"


class ProfileContext(SchemaModel):
    """Exact fields a passive profile cannot observe; supplied by the caller."""

    model: str
    revision: str
    gpu_name: str
    arrival_pattern: Literal["poisson-v1", "batched-poisson-v1"]
    declared_fixed_length: bool


class ProfileAdvisorDecision(SchemaModel):
    """Self-validating, profile-bound advisor decision."""

    profile_decision_version: Literal["0.1.1"] = "0.1.1"
    policy: AdvisorPolicy
    profile: WorkloadProfile
    profile_provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context: ProfileContext
    status: Literal["recommended", "abstain"]
    engine_overrides: Optional[dict[str, Any]] = None
    reasons: list[str]
    decision: Optional[AdvisorDecision] = None

    @model_validator(mode="after")
    def _check(self) -> "ProfileAdvisorDecision":
        if self.profile_provenance_sha256 != self.profile.provenance_sha256:
            raise ValueError("profile provenance digest is inconsistent with embedded profile")
        expected = _decision_fields(self.policy, self.profile, self.context)
        if (self.status, self.engine_overrides, self.reasons, self.decision) != expected:
            raise ValueError("profile advisor decision is inconsistent with embedded evidence")
        return self


def _decision_fields(policy: AdvisorPolicy, profile: WorkloadProfile, context: ProfileContext):
    """Pure derivation used by construction and serialized-report validation."""
    def _early(reasons: list[str]):
        return "abstain", None, reasons, None

    if not context.declared_fixed_length:
        return _early(["workload_not_declared_fixed_length"])
    prompts = {o.prompt_tokens for o in profile.observations}
    outputs = {o.output_tokens for o in profile.observations}
    mixed = []
    if policy.prompt_tokens_min is not None:
        # Policy declares a validated prompt-length envelope: every observed prompt
        # must fall inside it, but need not be a single value.
        lo, hi = policy.prompt_tokens_min, policy.prompt_tokens_max
        if any(not (lo <= p <= hi) for p in prompts):
            mixed.append("prompt_lengths_outside_validated_envelope")
        prompt_value = policy.prompt_tokens
    elif len(prompts) != 1:
        mixed.append("mixed_prompt_lengths")
        prompt_value = None
    else:
        prompt_value = next(iter(prompts))
    if len(outputs) != 1:
        mixed.append("mixed_output_lengths")
    if mixed:
        return _early(mixed)
    rate = profile.realized_request_rate_qps
    if rate is None:
        return _early(["unsupported_request_rate_qps"])
    try:
        request = AdvisorRequest(
            model=context.model, revision=context.revision, gpu_name=context.gpu_name,
            arrival_pattern=context.arrival_pattern,
            prompt_tokens=prompt_value, output_tokens=next(iter(outputs)),
            request_rate_qps=rate,
        )
    except ValidationError:
        return _early(["unsupported_workload_shape"])
    decision = recommend(policy, request)
    return decision.status, decision.engine_overrides, decision.reasons, decision


def advise_from_profile(
    policy: AdvisorPolicy, profile: WorkloadProfile, context: ProfileContext
) -> ProfileAdvisorDecision:
    """Recommend only for an evidence-supported fixed-length profile; else abstain."""
    status, overrides, reasons, decision = _decision_fields(policy, profile, context)
    return ProfileAdvisorDecision(
        policy=policy, profile=profile, profile_provenance_sha256=profile.provenance_sha256,
        context=context, status=status, engine_overrides=overrides, reasons=reasons, decision=decision,
    )
