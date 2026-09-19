"""Operator-facing advisory: tune, scale, or accept — with a dollar-denominated why.

Repositioning (2026-09-18, from external strategy reviews): InferPilot's value is not
"beat vLLM defaults via config tuning" — in most (compute-bound) regimes there is no
config win, and that is the *correct, valuable* diagnosis. The operator's real question
is "should I tune, add hardware, or change topology, and what does it cost?" This turns
a BottleneckDiagnosis + a measured baseline + an SLO + the operator's GPU economics into
that answer, framed as goodput-under-SLO and cost-per-token.

Goodput = sustained QPS that meets the SLO (not raw throughput). Config tuning is one
sub-case, surfaced ONLY when the diagnosis names a winnable lever; otherwise the honest
output is "scale" (with the GPU/cost delta) or "adequate".
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import Field, model_validator

from .._base import SchemaModel
from ..config import SLO
from ..diagnosis import BottleneckDiagnosis, diagnose
from ..results import ExperimentResult

Action = Literal["adequate", "tune", "scale"]


class OperatorEconomics(SchemaModel):
    gpu_cost_per_hour_usd: float = Field(gt=0)
    gpu_count: int = Field(default=1, ge=1)
    target_qps: Optional[float] = Field(default=None, gt=0, description="Desired sustained QPS.")


def _derive_action(
    diagnosis: BottleneckDiagnosis, econ: OperatorEconomics, met_slo: bool, goodput_qps: float,
    offered_qps: float,
) -> tuple[Action, Optional[int], str]:
    meets_target = econ.target_qps is None or goodput_qps >= econ.target_qps
    # 1. Already healthy -> do NOTHING, even if a lever exists. Tuning a passing
    #    deployment is needless operational risk.
    if met_slo and meets_target:
        return (
            "adequate", None,
            f"ADEQUATE ({diagnosis.regime}): meets SLO at {goodput_qps:.2f} qps (>= target); "
            f"keep the current config.",
        )
    # 2. A winnable config lever -> tune (verify + canary).
    if diagnosis.recommended_lever != "none":
        return (
            "tune", None,
            f"TUNE {diagnosis.recommended_lever} ({diagnosis.regime}): {diagnosis.predicted_effect}. "
            f"Verify preconditions {diagnosis.lever_preconditions} and confirm with a canary before apply.",
        )
    # 3. No lever and not adequate -> scale. Honest SLO wording; gpus only when goodput known.
    status = (f"meets SLO at {goodput_qps:.2f} qps but below the {econ.target_qps:.2f} qps target"
              if met_slo else f"does NOT meet SLO at the offered {offered_qps:.2f} qps")
    gpus_needed = None
    if econ.target_qps is not None and goodput_qps > 0:
        gpus_needed = max(econ.gpu_count + 1, math.ceil(econ.gpu_count * econ.target_qps / goodput_qps))
        how = (f"~{gpus_needed} GPU(s) (first-order linear estimate; real scaling is sublinear — "
               f"routing/TP/replication overhead), a faster SKU, or a smaller/quantized model")
    else:
        how = ("a rate sweep is needed to size the SLO-compliant capacity; then add GPUs, a faster "
               "SKU, or a smaller/quantized model")
    return (
        "scale", gpus_needed,
        f"SCALE ({diagnosis.regime}): no config lever helps — {status} on {econ.gpu_count} "
        f"GPU(s); to reach target: {how}.",
    )


class CapacityAdvisory(SchemaModel):
    """Self-validating tune/scale/accept recommendation with cost framing."""

    advisory_version: Literal["0.1.0"] = "0.1.0"
    diagnosis: BottleneckDiagnosis
    slo: SLO
    economics: OperatorEconomics
    met_slo: bool
    offered_qps: float = Field(ge=0)
    goodput_qps: float = Field(ge=0, description="SLO-compliant sustained qps (0 if SLO not met).")
    cost_per_million_output_tokens_usd: Optional[float] = Field(
        default=None, ge=0, description="$/1M output tokens; only set when the run meets SLO.")
    action: Action
    gpus_needed_for_target: Optional[int] = None
    recommendation: str

    @model_validator(mode="after")
    def _check(self) -> "CapacityAdvisory":
        action, gpus, rec = _derive_action(
            self.diagnosis, self.economics, self.met_slo, self.goodput_qps, self.offered_qps
        )
        if (self.action, self.gpus_needed_for_target, self.recommendation) != (action, gpus, rec):
            raise ValueError("capacity advisory is inconsistent with its diagnosis and economics")
        return self


def advise_capacity(result: ExperimentResult, slo: SLO, economics: OperatorEconomics) -> CapacityAdvisory:
    """Diagnose the baseline and turn it into a tune/scale/accept + cost recommendation."""
    a = result.aggregates
    diagnosis = diagnose(result)
    met = not diagnosis.saturated
    if slo.ttft_p95_ms is not None and (a.ttft_p95_ms is None or a.ttft_p95_ms > slo.ttft_p95_ms):
        met = False
    if slo.tpot_p95_ms is not None and (a.tpot_p95_ms is None or a.tpot_p95_ms > slo.tpot_p95_ms):
        met = False
    offered = result.config.workload.request_rate_qps
    # Goodput = SLO-compliant sustained qps. If the run did NOT meet SLO, its completion
    # throughput is NOT goodput (those completions violated the SLO); the true goodput is
    # below the offered rate and requires a rate sweep to locate, so report 0 here.
    goodput = offered if met else 0.0
    # Cost per token is only meaningful for an SLO-compliant run.
    cost = None
    if met and a.throughput_tokens_per_s:
        cost = (
            economics.gpu_cost_per_hour_usd * economics.gpu_count
            / (a.throughput_tokens_per_s * 3600.0) * 1_000_000.0
        )
    action, gpus, rec = _derive_action(diagnosis, economics, met, goodput, offered)
    return CapacityAdvisory(
        diagnosis=diagnosis, slo=slo, economics=economics, met_slo=met, offered_qps=offered,
        goodput_qps=goodput, cost_per_million_output_tokens_usd=cost, action=action,
        gpus_needed_for_target=gpus, recommendation=rec,
    )
