"""One call, one answer: diagnose -> plan -> (size hardware) -> advise, with a
mechanistic, human-readable action log. This is the operator-facing entry point.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import model_validator

from .._base import SchemaModel
from ..config import SLO
from ..deployment import ModelFootprint, ScaleRecommendation, recommend_scale
from ..diagnosis import BottleneckDiagnosis
from ..results import ExperimentResult
from .capacity_advisory import CapacityAdvisory, OperatorEconomics, advise_capacity


def _action_log(
    diag: BottleneckDiagnosis, adv: CapacityAdvisory, scale: Optional[ScaleRecommendation]
) -> str:
    lines = [
        f"DIAGNOSIS: {diag.regime} (gpu_mean {diag.gpu_utilization_mean_pct:.0f}%, "
        f"kv_peak {diag.kv_cache_usage_peak_perc:.2f}, saturated={diag.saturated}).",
        f"PREDICTION: {diag.predicted_effect}.",
        f"GOODPUT: {adv.goodput_qps:.2f} qps under SLO"
        + (f", ~${adv.cost_per_million_output_tokens_usd:.2f}/1M output tokens."
           if adv.cost_per_million_output_tokens_usd is not None else "."),
        f"RECOMMENDATION: {adv.recommendation}",
    ]
    if adv.action == "tune" and diag.lever_preconditions:
        lines.append(f"VERIFY FIRST: {diag.lever_preconditions}.")
    # Precision-reducing levers (fp8/quantized KV) can silently change outputs -> apply-block
    # until a QualityGate passes. Never report such a win without a quality check.
    if adv.action == "tune" and "fp8" in diag.recommended_lever:
        lines.append("QUALITY GATE: fp8 KV can silently change outputs — apply BLOCKED until a "
                     "greedy-token-agreement QualityGate passes (evaluate_quality); mandatory on "
                     "long-context work.")
    if scale is not None and scale.cheapest is not None:
        f = scale.cheapest.fit
        lines.append(
            f"STRUCTURAL: to hold ~{scale.required_concurrency} concurrent "
            f"{scale.context_tokens}-tok requests, cheapest fit = {f.gpu_name} tp={f.tensor_parallel} "
            f"kv={f.kv_dtype} @ ${scale.cheapest.hourly_usd:.2f}/hr (holds {f.max_concurrent_requests})."
        )
    elif scale is not None:
        lines.append("STRUCTURAL: no candidate GPU/TP/precision holds the required concurrency — "
                     "reduce context/concurrency, use a smaller/quantized model, or a larger GPU.")
    lines.append("VERIFICATION: apply only after a fail-closed Pareto canary (compare_configs).")
    return "\n".join(lines)


class InferPilotRecommendation(SchemaModel):
    """Self-validating end-to-end recommendation with a mechanistic action log."""

    recommendation_version: Literal["0.1.0"] = "0.1.0"
    advisory: CapacityAdvisory
    scale: Optional[ScaleRecommendation] = None
    action_log: str

    @model_validator(mode="after")
    def _check(self) -> "InferPilotRecommendation":
        if self.action_log != _action_log(self.advisory.diagnosis, self.advisory, self.scale):
            raise ValueError("action log is inconsistent with the advisory/scale evidence")
        return self


def recommend_plan(
    result: ExperimentResult, slo: SLO, economics: OperatorEconomics,
    *, footprint: Optional[ModelFootprint] = None,
    candidate_gpus: Optional[list[str]] = None,
) -> InferPilotRecommendation:
    """Run the whole chain on a measured baseline and emit one actionable recommendation."""
    advisory = advise_capacity(result, slo, economics)
    scale = None
    if advisory.action == "scale" and footprint is not None:
        target = economics.target_qps or result.config.workload.request_rate_qps
        scale = recommend_scale(result, footprint, target, candidate_gpus=candidate_gpus)
    return InferPilotRecommendation(
        advisory=advisory, scale=scale,
        action_log=_action_log(advisory.diagnosis, advisory, scale),
    )
