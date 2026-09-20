"""Operator-facing optimization evidence card.

This is the product boundary between InferPilot's strict evidence machinery and
an engineer deciding what to do next.  It never converts a workload shape into
a bottleneck guess: the embedded ExperimentResult must contain aligned load
evidence before a tuning experiment can be recommended.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .advisor.capacity_advisory import OperatorEconomics
from .advisor.recommendation import InferPilotRecommendation, recommend_plan
from .config import SLO
from .mechanism import MechanismEvidence
from .results import ExperimentResult

CardStatus = Literal["experiment_recommended", "keep_current", "abstain"]
EvidenceStrength = Literal["insufficient", "limited", "aligned", "mechanism_observed"]
Transferability = Literal["test_before_use", "observed_window_only", "unsupported"]


def _candidate(recommendation: InferPilotRecommendation) -> tuple[dict[str, Any] | None, str]:
    diagnosis = recommendation.advisory.diagnosis
    if recommendation.advisory.action != "tune":
        return None, "none"
    if diagnosis.recommended_lever == "kv_cache_dtype=fp8":
        return {"kv_cache_dtype": "fp8"}, "reduce_kv_bytes_and_test_whether_preemption_waste_falls"
    # Unknown future levers must not be translated into executable overrides by guessing.
    return None, "unmapped_validated_lever"


def _criteria(slo: SLO, candidate: dict[str, Any] | None) -> tuple[str, ...]:
    criteria = ["all_requests_successful", "effective_config_verified", "aligned_load_evidence_complete"]
    for field, label in (
        (slo.ttft_p95_ms, "ttft_p95_ms"),
        (slo.tpot_p95_ms, "tpot_p95_ms"),
        (slo.e2e_p95_ms, "e2e_p95_ms"),
    ):
        if field is not None:
            criteria.append(f"{label}<={field:g}")
    if slo.min_throughput_tokens_per_s is not None:
        criteria.append(f"throughput_tokens_per_s>={slo.min_throughput_tokens_per_s:g}")
    if candidate and candidate.get("kv_cache_dtype") == "fp8":
        criteria.extend((
            "representative_task_quality_gate_passes",
            "long_context_accuracy_gate_passes",
            "candidate_is_pareto_noninferior_on_registered_latency_metrics",
        ))
    return tuple(criteria)


def _derive(
    result: ExperimentResult,
    slo: SLO,
    economics: OperatorEconomics,
    total_occupancy_s: Optional[float],
    mechanism_evidence: Optional[MechanismEvidence],
) -> dict:
    recommendation = recommend_plan(result, slo, economics)
    diagnosis = recommendation.advisory.diagnosis
    candidate, mechanism = _candidate(recommendation)

    status: CardStatus
    if not result.is_baseline_eligible:
        candidate = None
        mechanism = "baseline_ineligible"
        status = "abstain"
    elif recommendation.advisory.action == "tune" and candidate is not None:
        status = "experiment_recommended"
    elif recommendation.advisory.action == "adequate":
        status = "keep_current"
    else:
        status = "abstain"

    if not result.is_baseline_eligible or diagnosis.load_state == "indeterminate":
        strength: EvidenceStrength = "insufficient"
    elif diagnosis.load_state == "near_capacity":
        strength = "limited"
    elif (
        mechanism_evidence is not None
        and mechanism_evidence.coverage_complete
        and mechanism_evidence.experiment_id == result.config.experiment_id
    ):
        strength = "mechanism_observed"
    else:
        strength = "aligned"

    transferability: Transferability = (
        "test_before_use" if status == "experiment_recommended"
        else "observed_window_only" if status == "keep_current"
        else "unsupported"
    )

    aggregate = result.aggregates
    successful_tokens = 0 if aggregate is None else aggregate.total_output_tokens
    measured_s = 0.0 if aggregate is None else aggregate.duration_s
    baseline_cost = (
        economics.gpu_cost_per_hour_usd
        * economics.gpu_count
        * measured_s
        / 3600
        * 1_000_000
        / successful_tokens
        if successful_tokens
        else None
    )
    occupancy = total_occupancy_s if total_occupancy_s is not None else measured_s
    experiment_seconds = 2 * occupancy if status == "experiment_recommended" and occupancy > 0 else None
    experiment_cost = (
        economics.gpu_cost_per_hour_usd * economics.gpu_count * experiment_seconds / 3600
        if experiment_seconds is not None
        else None
    )
    cost_basis = (
        "paired_fresh_server_observed_total_occupancy"
        if experiment_seconds is not None and total_occupancy_s is not None
        else "paired_measured_window_lower_bound"
        if experiment_seconds is not None
        else "not_applicable"
    )

    reasons = [
        f"load_state={diagnosis.load_state}",
        f"diagnosis={diagnosis.regime}",
        f"advisor_action={recommendation.advisory.action}",
    ]
    reasons.extend(recommendation.advisory.diagnosis.load_assessment.reasons)
    if status == "experiment_recommended":
        reasons.append("candidate_must_be_validated_on_this_exact_model_hardware_engine_and_workload")
    elif status == "abstain":
        reasons.append("no_executable_candidate_is_supported_by_current_evidence")

    return {
        "recommendation": recommendation,
        "status": status,
        "evidence_strength": strength,
        "transferability": transferability,
        "candidate_engine_overrides": candidate,
        "mechanism": mechanism,
        "observed_gpu_cost_per_million_output_tokens_usd": baseline_cost,
        "estimated_paired_experiment_gpu_seconds": experiment_seconds,
        "estimated_paired_experiment_cost_usd": experiment_cost,
        "experiment_cost_basis": cost_basis,
        "acceptance_criteria": _criteria(slo, candidate),
        "reasons": tuple(reasons),
    }


class OptimizationEvidenceCard(SchemaModel):
    """Self-validating recommendation and its exact evidence boundary."""

    card_version: Literal["0.1.0"] = "0.1.0"
    baseline_result: ExperimentResult
    slo: SLO
    economics: OperatorEconomics
    total_occupancy_s: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    mechanism_evidence: Optional[MechanismEvidence] = None
    metadata_only: Literal[True] = True

    recommendation: InferPilotRecommendation
    status: CardStatus
    evidence_strength: EvidenceStrength
    transferability: Transferability
    candidate_engine_overrides: Optional[dict[str, Any]] = None
    mechanism: str
    observed_gpu_cost_per_million_output_tokens_usd: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    estimated_paired_experiment_gpu_seconds: Optional[float] = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    estimated_paired_experiment_cost_usd: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    experiment_cost_basis: Literal[
        "paired_fresh_server_observed_total_occupancy",
        "paired_measured_window_lower_bound",
        "not_applicable",
    ]
    acceptance_criteria: tuple[str, ...]
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> "OptimizationEvidenceCard":
        if self.mechanism_evidence is not None and (
            self.mechanism_evidence.experiment_id != self.baseline_result.config.experiment_id
        ):
            raise ValueError("mechanism evidence experiment ID does not match baseline")
        expected = _derive(
            self.baseline_result,
            self.slo,
            self.economics,
            self.total_occupancy_s,
            self.mechanism_evidence,
        )
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"evidence-card field {field!r} is inconsistent with inputs")
        return self


def build_evidence_card(
    result: ExperimentResult,
    slo: SLO,
    economics: OperatorEconomics,
    *,
    total_occupancy_s: Optional[float] = None,
    mechanism_evidence: Optional[MechanismEvidence] = None,
) -> OptimizationEvidenceCard:
    derived = _derive(result, slo, economics, total_occupancy_s, mechanism_evidence)
    return OptimizationEvidenceCard(
        baseline_result=result,
        slo=slo,
        economics=economics,
        total_occupancy_s=total_occupancy_s,
        mechanism_evidence=mechanism_evidence,
        **derived,
    )
