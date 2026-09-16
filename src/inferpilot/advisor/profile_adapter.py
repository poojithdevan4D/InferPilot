"""Bind a passive WorkloadProfile to the evidence-bound advisor (M5 phase 4).

This adapter only constructs an :class:`AdvisorRequest` when the profile and
caller-supplied context provide EVERY exact field the policy requires. It never
rounds, buckets, interpolates, extrapolates, or maps a measured rate to a policy
regime: the observed realized rate must exactly match a validated regime or the
adapter abstains with the advisor's existing unsupported-rate reason. Prompt and
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

PROFILE_DECISION_VERSION = "0.1.0"


class ProfileContext(SchemaModel):
    """Exact fields a passive profile cannot observe; supplied by the caller."""

    model: str
    revision: str
    gpu_name: str
    arrival_pattern: Literal["poisson-v1", "batched-poisson-v1"]
    declared_fixed_length: bool


class ProfileAdvisorDecision(SchemaModel):
    """Self-validating, profile-bound advisor decision."""

    profile_decision_version: Literal["0.1.0"] = "0.1.0"
    profile_provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context: ProfileContext
    status: Literal["recommended", "abstain"]
    engine_overrides: Optional[dict[str, Any]] = None
    reasons: list[str]
    decision: Optional[AdvisorDecision] = None

    @model_validator(mode="after")
    def _check(self) -> "ProfileAdvisorDecision":
        if self.decision is not None:
            # Mirror the embedded (itself self-validating) advisor decision exactly.
            if (self.status, self.engine_overrides, self.reasons) != (
                self.decision.status, self.decision.engine_overrides, self.decision.reasons
            ):
                raise ValueError("profile decision is inconsistent with the embedded advisor decision")
        else:
            # Early abstention (no request formed): must be a reasoned abstain.
            if self.status != "abstain" or self.engine_overrides is not None or not self.reasons:
                raise ValueError("an early profile abstention must carry reasons and no overrides")
        return self


def advise_from_profile(
    policy: AdvisorPolicy, profile: WorkloadProfile, context: ProfileContext
) -> ProfileAdvisorDecision:
    """Recommend only for an exact, evidence-supported fixed-length profile; else abstain."""
    digest = profile.provenance_sha256

    def _early(reasons: list[str]) -> ProfileAdvisorDecision:
        return ProfileAdvisorDecision(
            profile_provenance_sha256=digest, context=context, status="abstain",
            engine_overrides=None, reasons=reasons, decision=None,
        )

    if not context.declared_fixed_length:
        return _early(["workload_not_declared_fixed_length"])

    prompts = {o.prompt_tokens for o in profile.observations}
    outputs = {o.output_tokens for o in profile.observations}
    mixed: list[str] = []
    if len(prompts) != 1:
        mixed.append("mixed_prompt_lengths")
    if len(outputs) != 1:
        mixed.append("mixed_output_lengths")
    if mixed:
        return _early(mixed)

    rate = profile.realized_request_rate_qps
    if rate is None:
        return _early(["unsupported_request_rate_qps"])  # insufficient evidence: no exact rate

    try:
        request = AdvisorRequest(
            model=context.model, revision=context.revision, gpu_name=context.gpu_name,
            arrival_pattern=context.arrival_pattern,
            prompt_tokens=next(iter(prompts)), output_tokens=next(iter(outputs)),
            request_rate_qps=rate,  # exact observed rate; never mapped to a regime
        )
    except ValidationError:
        return _early(["unsupported_workload_shape"])

    decision = recommend(policy, request)
    return ProfileAdvisorDecision(
        profile_provenance_sha256=digest, context=context, status=decision.status,
        engine_overrides=decision.engine_overrides, reasons=decision.reasons, decision=decision,
    )
