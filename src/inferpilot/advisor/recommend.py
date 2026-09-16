from __future__ import annotations

import hashlib
from pathlib import Path

from inferpilot.search.policy_benchmark import PolicyBenchmarkReport

from .models import AdvisorDecision, AdvisorPolicy, AdvisorRequest
from .rate_band_evidence import RateBandEvidenceReport


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_verified_policy(
    policy_path: Path,
    evidence_path: Path,
    applicability_path: Path | None = None,
) -> AdvisorPolicy:
    policy = AdvisorPolicy.model_validate_json(policy_path.read_text())
    evidence_payload = evidence_path.read_bytes()
    if _sha256(evidence_payload) != policy.evidence_report_sha256:
        raise ValueError("optimization evidence report digest mismatch")
    PolicyBenchmarkReport.model_validate_json(evidence_payload)
    if policy.policy_version == "0.1.0":
        if applicability_path is not None:
            raise ValueError("policy 0.1.0 does not accept applicability evidence")
        return policy
    if applicability_path is None:
        raise ValueError("policy 0.2.0 requires an applicability evidence report")
    applicability_payload = applicability_path.read_bytes()
    if _sha256(applicability_payload) != policy.applicability_report_sha256:
        raise ValueError("applicability evidence report digest mismatch")
    report = RateBandEvidenceReport.model_validate_json(applicability_payload)
    if report.policy_id != policy.policy_id:
        raise ValueError("applicability evidence policy id mismatch")
    context_fields = (
        "model", "revision", "gpu_name", "prompt_tokens", "output_tokens",
        "arrival_pattern", "fixed_engine_fields",
    )
    if any(getattr(report, field) != getattr(policy, field) for field in context_fields):
        raise ValueError("policy context does not match applicability evidence")
    expected = [
        (regime.min_rate_qps, regime.max_rate_qps, regime.engine_overrides.get("max_num_seqs"))
        for regime in policy.regimes
    ]
    observed = [
        (band.min_rate_qps, band.max_rate_qps, band.max_num_seqs)
        for band in report.bands
    ]
    if expected != observed:
        raise ValueError("policy rate bands do not match applicability evidence")
    return policy


def _derive(policy: AdvisorPolicy, request: AdvisorRequest):
    reasons = []
    for field in ("model", "revision", "gpu_name", "output_tokens", "arrival_pattern"):
        if getattr(request, field) != getattr(policy, field):
            reasons.append(f"unsupported_{field}")
    # prompt_tokens matches inside the validated envelope when declared, else exactly.
    prompt_lo = policy.prompt_tokens if policy.prompt_tokens_min is None else policy.prompt_tokens_min
    prompt_hi = policy.prompt_tokens if policy.prompt_tokens_max is None else policy.prompt_tokens_max
    if not prompt_lo <= request.prompt_tokens <= prompt_hi:
        reasons.append("unsupported_prompt_tokens")
    if policy.policy_version == "0.1.0":
        matches = [
            regime for regime in policy.regimes
            if regime.request_rate_qps == request.request_rate_qps
        ]
    else:
        matches = [
            regime for regime in policy.regimes
            if regime.min_rate_qps <= request.request_rate_qps <= regime.max_rate_qps
        ]
    if not matches:
        reasons.append("unsupported_request_rate_qps")
    if reasons:
        return "abstain", None, reasons
    return "recommended", {**policy.fixed_engine_fields, **matches[0].engine_overrides}, []


def recommend(policy: AdvisorPolicy, request: AdvisorRequest) -> AdvisorDecision:
    status, overrides, reasons = _derive(policy, request)
    return AdvisorDecision(
        policy=policy,
        request=request,
        status=status,
        engine_overrides=overrides,
        reasons=reasons,
    )
