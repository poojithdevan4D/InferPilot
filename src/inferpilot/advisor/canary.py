"""Measured, self-validating canary evidence for controller decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import SchemaModel
from ..results import ExperimentResult
from ..runner.aggregate import compute_aggregates
from .profile_adapter import ProfileAdvisorDecision


class CanarySpec(SchemaModel):
    canary_version: Literal["0.1.0"] = "0.1.0"
    ttft_p95_limit_ms: float = Field(gt=0)
    tpot_p95_limit_ms: float = Field(gt=0)


class CanaryEvaluation(SchemaModel):
    evaluation_version: Literal["0.1.0"] = "0.1.0"
    spec: CanarySpec
    decision: ProfileAdvisorDecision
    result: ExperimentResult
    passed: bool
    reasons: list[str]

    @model_validator(mode="after")
    def check(self):
        expected = _derive(self.spec, self.decision, self.result)
        if (self.passed, self.reasons) != expected:
            raise ValueError("canary evaluation is inconsistent with evidence")
        return self


def _derive(
    spec: CanarySpec,
    decision: ProfileAdvisorDecision,
    result: ExperimentResult,
) -> tuple[bool, list[str]]:
    if decision.status != "recommended" or decision.decision is None:
        raise ValueError("canary evaluation requires a recommended advisor decision")
    reasons = []
    request = decision.decision.request
    config = result.config
    workload = config.workload
    if not result.is_baseline_eligible:
        reasons.append("result_not_baseline_eligible")
    if config.engine.model != request.model:
        reasons.append("model_mismatch")
    if config.engine.revision != request.revision:
        reasons.append("revision_mismatch")
    if result.environment.hardware.gpu_name != request.gpu_name:
        reasons.append("gpu_mismatch")
    if workload.prompt_tokens != request.prompt_tokens:
        reasons.append("prompt_tokens_mismatch")
    if workload.output_tokens != request.output_tokens:
        reasons.append("output_tokens_mismatch")
    if workload.arrival_pattern != request.arrival_pattern:
        reasons.append("arrival_pattern_mismatch")
    if workload.request_rate_qps != request.request_rate_qps:
        reasons.append("request_rate_mismatch")
    effective = result.effective_config
    for field, value in decision.engine_overrides.items():
        if not hasattr(config.engine, field) or getattr(config.engine, field) != value:
            reasons.append(f"requested_{field}_mismatch")
        if effective is None or not hasattr(effective, field) or getattr(effective, field) != value:
            reasons.append(f"effective_{field}_mismatch")
    aggregates = result.aggregates
    if aggregates is None:
        reasons.append("aggregates_missing")
    else:
        recomputed = compute_aggregates(result.measurements, aggregates.duration_s)
        expected = recomputed.model_dump(exclude={"gpu_memory_peak_mb"})
        observed = aggregates.model_dump(exclude={"gpu_memory_peak_mb"})
        if observed != expected:
            reasons.append("aggregates_inconsistent_with_measurements")
        if aggregates.ttft_p95_ms is None or aggregates.ttft_p95_ms > spec.ttft_p95_limit_ms:
            reasons.append("ttft_p95_slo_failed")
        if aggregates.tpot_p95_ms is None or aggregates.tpot_p95_ms > spec.tpot_p95_limit_ms:
            reasons.append("tpot_p95_slo_failed")
    return not reasons, reasons


def evaluate_canary(
    spec: CanarySpec,
    decision: ProfileAdvisorDecision,
    result: ExperimentResult,
) -> CanaryEvaluation:
    passed, reasons = _derive(spec, decision, result)
    return CanaryEvaluation(
        spec=spec,
        decision=decision,
        result=result,
        passed=passed,
        reasons=reasons,
    )
