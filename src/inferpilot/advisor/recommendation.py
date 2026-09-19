"""Operator-facing report; missing load evidence propagates as abstention."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import model_validator

from .._base import SchemaModel
from ..config import SLO
from ..deployment import ModelFootprint, ScaleRecommendation, recommend_scale
from ..diagnosis import BottleneckDiagnosis
from ..results import ExperimentResult
from ..saturation import LoadEvidence
from .capacity_advisory import CapacityAdvisory, OperatorEconomics, advise_capacity


def _action_log(diag: BottleneckDiagnosis, adv: CapacityAdvisory, scale: Optional[ScaleRecommendation]) -> str:
    gpu = "unknown" if diag.gpu_utilization_mean_pct is None else f"{diag.gpu_utilization_mean_pct:.0f}%"
    kv = "unknown" if diag.kv_cache_usage_peak_perc is None else f"{diag.kv_cache_usage_peak_perc:.2f}"
    goodput = "unknown" if adv.goodput_qps is None else f"{adv.goodput_qps:.2f} qps (observed cohort, not capacity)"
    lines = [f"DIAGNOSIS: {diag.regime} (load={diag.load_state}, gpu_mean={gpu}, kv_peak={kv}).",
             f"PREDICTION: {diag.predicted_effect}", f"GOODPUT: {goodput}.",
             f"RECOMMENDATION: {adv.recommendation}"]
    if adv.cost_per_million_output_tokens_usd is not None:
        lines.append(f"OBSERVED COST: ${adv.cost_per_million_output_tokens_usd:.2f}/1M SLO-passing output tokens; not quality-adjusted.")
    if adv.action == "tune":
        lines.append(f"VERIFY FIRST: {diag.lever_preconditions}.")
        lines.append("QUALITY GATE: precision changes require representative task-quality evidence; token agreement alone does not establish quality.")
    if scale is not None:
        lines.append("STRUCTURAL: memory-fit calculation only, not a throughput or capacity prediction.")
    lines.append("VERIFICATION: compare_configs is a load/Pareto check, not an absolute SLO or quality certificate. No deployment is changed.")
    return "\n".join(lines)


class InferPilotRecommendation(SchemaModel):
    recommendation_version: Literal["0.2.0"] = "0.2.0"
    advisory: CapacityAdvisory
    scale: Optional[ScaleRecommendation] = None
    action_log: str

    @model_validator(mode="after")
    def _check(self) -> "InferPilotRecommendation":
        if self.action_log != _action_log(self.advisory.diagnosis, self.advisory, self.scale):
            raise ValueError("action log is inconsistent with the advisory/scale evidence")
        return self


def recommend_plan(result: ExperimentResult, slo: SLO, economics: OperatorEconomics, *,
                   footprint: Optional[ModelFootprint] = None, candidate_gpus: Optional[list[str]] = None,
                   load_evidence: Optional[LoadEvidence] = None) -> InferPilotRecommendation:
    advisory = advise_capacity(result, slo, economics, load_evidence=load_evidence)
    scale = None
    if advisory.action == "scale" and footprint is not None:
        target = economics.target_qps or result.config.workload.request_rate_qps
        scale = recommend_scale(result, footprint, target, candidate_gpus=candidate_gpus)
    return InferPilotRecommendation(advisory=advisory, scale=scale,
                                    action_log=_action_log(advisory.diagnosis, advisory, scale))
