"""Fail-closed observed-window advice; unknown capacity is not zero capacity."""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import Field, model_validator

from .._base import SchemaModel
from ..config import SLO
from ..diagnosis import BottleneckDiagnosis, diagnose
from ..results import ExperimentResult
from ..runner.aggregate import compute_aggregates
from ..saturation import LoadEvidence

Action = Literal["adequate", "tune", "scale", "abstain"]


class OperatorEconomics(SchemaModel):
    gpu_cost_per_hour_usd: float = Field(gt=0, allow_inf_nan=False)
    gpu_count: int = Field(default=1, ge=1)
    target_qps: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)


def _derive_action(diagnosis, econ, met_slo, goodput_qps, offered_qps):
    if diagnosis.load_state in ("indeterminate", "near_capacity"):
        return "abstain", None, "ABSTAIN: load evidence is inconclusive; extend the measured window and run a rate sweep."
    if met_slo is True and (econ.target_qps is None or goodput_qps >= econ.target_qps):
        return "adequate", None, "ADEQUATE for this observed workload/window only; no unmeasured headroom is claimed."
    if diagnosis.recommended_lever != "none":
        return "tune", None, f"TUNE via a controlled {diagnosis.recommended_lever} canary only: {diagnosis.predicted_effect}"
    return "abstain", None, "ABSTAIN: no validated tuning or scaling remedy; a rate sweep and bottleneck evidence are required before sizing."


def _metrics(result, diagnosis, slo, economics):
    if diagnosis.load_state != "healthy" or not result.is_baseline_eligible:
        return None, None, None
    original = result.aggregates
    actual = compute_aggregates(result.measurements, original.duration_s)
    if actual.model_dump(exclude={"gpu_memory_peak_mb"}) != original.model_dump(exclude={"gpu_memory_peak_mb"}):
        raise ValueError("aggregate evidence is inconsistent with measurements")
    e = diagnosis.load_assessment.evidence
    rows = [m for m in result.measurements if e.boundaries_s[0] <= m.start_time_s < e.boundaries_s[-1]]
    duration = e.boundaries_s[-1] - e.boundaries_s[0]
    a = compute_aggregates(rows, duration)
    constraints = [("ttft_p95_ms", "ttft_ms", slo.ttft_p95_ms),
                   ("tpot_p95_ms", "tpot_ms", slo.tpot_p95_ms),
                   ("e2e_p95_ms", "e2e_latency_ms", slo.e2e_p95_ms)]
    limits = [limit for _, _, limit in constraints if limit is not None]
    if slo.min_throughput_tokens_per_s is not None:
        limits.append(slo.min_throughput_tokens_per_s)
    if any(not math.isfinite(limit) for limit in limits):
        raise ValueError("SLO thresholds must be finite")
    if not limits:
        return None, None, None
    met = all(limit is None or (getattr(a, name) is not None and getattr(a, name) <= limit)
              for name, _, limit in constraints)
    if slo.min_throughput_tokens_per_s is not None:
        met = met and sum(e.delivered_output_tokens) / duration >= slo.min_throughput_tokens_per_s
    passing = [m for m in rows if m.success and all(
        limit is None or (getattr(m, name) is not None and getattr(m, name) <= limit)
        for _, name, limit in constraints)]
    # Observed arrival-cohort goodput, including outcomes after the window closes.
    # It is NOT a goodput ceiling and NOT the configured request rate.
    goodput = len(passing) / duration
    tokens = sum(m.output_tokens for m in passing)
    cost = economics.gpu_cost_per_hour_usd * economics.gpu_count * duration / 3600 * 1e6 / tokens if met and tokens else None
    return met, goodput, cost


class CapacityAdvisory(SchemaModel):
    advisory_version: Literal["0.2.0"] = "0.2.0"
    result: ExperimentResult
    diagnosis: BottleneckDiagnosis
    slo: SLO
    economics: OperatorEconomics
    met_slo: Optional[bool]
    offered_qps: float = Field(ge=0, allow_inf_nan=False, description="Configured rate, not measured capacity.")
    goodput_qps: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    cost_per_million_output_tokens_usd: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False,
        description="Observed cohort cost per SLO-passing output token; not quality-adjusted or a capacity prediction.")
    action: Action
    gpus_needed_for_target: Optional[int] = None
    recommendation: str

    @model_validator(mode="after")
    def _check(self) -> "CapacityAdvisory":
        e = self.diagnosis.load_assessment.evidence if self.diagnosis.load_assessment else None
        d = diagnose(self.result, load_evidence=e)
        met, goodput, cost = _metrics(self.result, d, self.slo, self.economics)
        offered = self.result.config.workload.request_rate_qps
        action, gpus, rec = _derive_action(d, self.economics, met, goodput, offered)
        if (self.diagnosis, self.met_slo, self.goodput_qps, self.cost_per_million_output_tokens_usd,
            self.offered_qps, self.action, self.gpus_needed_for_target, self.recommendation) != (
                d, met, goodput, cost, offered, action, gpus, rec):
            raise ValueError("capacity advisory is inconsistent with its diagnosis and evidence")
        return self


def advise_capacity(result: ExperimentResult, slo: SLO, economics: OperatorEconomics, *,
                    load_evidence: Optional[LoadEvidence] = None) -> CapacityAdvisory:
    diagnosis = diagnose(result, load_evidence=load_evidence)
    met, goodput, cost = _metrics(result, diagnosis, slo, economics)
    offered = result.config.workload.request_rate_qps
    action, gpus, rec = _derive_action(diagnosis, economics, met, goodput, offered)
    return CapacityAdvisory(result=result, diagnosis=diagnosis, slo=slo, economics=economics,
                            met_slo=met, offered_qps=offered, goodput_qps=goodput,
                            cost_per_million_output_tokens_usd=cost, action=action,
                            gpus_needed_for_target=gpus, recommendation=rec)
