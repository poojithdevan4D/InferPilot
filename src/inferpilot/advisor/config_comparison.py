"""Load-gated Pareto comparison. Not an absolute SLO or output-quality gate.

Old comparison artifacts require recomputation: stable TTFT no longer establishes
feasibility. Retained legacy spec fields do not override the new evidence gate.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from .._base import SchemaModel
from ..results import ExperimentResult
from ..runner.aggregate import compute_aggregates
from ..saturation import LoadEvidence, assess_load_state

_GUARDED = ("ttft_p95_ms", "tpot_p95_ms")
Verdict = Literal["candidate_dominates", "incumbent_kept", "inconclusive_tradeoff",
                  "candidate_infeasible", "inconclusive_evidence"]


class ComparisonSpec(SchemaModel):
    comparison_version: Literal["0.3.0"] = "0.3.0"
    keepup_fraction: float = Field(default=0.95, gt=0, le=1, allow_inf_nan=False,
                                   description="Legacy field; no drain-contaminated throughput fallback.")
    rel_tolerance: float = Field(default=0.02, ge=0, lt=1, allow_inf_nan=False)
    saturation_threshold: float = Field(default=1.5, gt=1, allow_inf_nan=False,
                                        description="Legacy TTFT field; not a feasibility criterion.")
    min_saturation_samples: int = Field(default=8, ge=2)


def _sound(label: str, result: ExperimentResult) -> list[str]:
    problems = []
    if not result.is_baseline_eligible:
        problems.append(f"{label}_not_baseline_eligible")
    agg = result.aggregates
    if agg is None:
        return problems + [f"{label}_aggregates_missing"]
    actual = compute_aggregates(result.measurements, agg.duration_s)
    if actual.model_dump(exclude={"gpu_memory_peak_mb"}) != agg.model_dump(exclude={"gpu_memory_peak_mb"}):
        problems.append(f"{label}_aggregates_inconsistent_with_measurements")
    return problems


def _same_context(incumbent: ExperimentResult, candidate: ExperimentResult) -> None:
    iw, cw = incumbent.config.workload, candidate.config.workload
    ie, ce = incumbent.config.engine, candidate.config.engine
    if not (ie.model == ce.model and ie.revision == ce.revision
            and incumbent.environment.hardware == candidate.environment.hardware
            and iw.prompt_tokens == cw.prompt_tokens and iw.output_tokens == cw.output_tokens
            and iw.arrival_pattern == cw.arrival_pattern and iw.request_rate_qps == cw.request_rate_qps):
        raise ValueError("config comparison requires the same model/hardware/workload context")


def _load(result, evidence):
    if evidence is not None and evidence.experiment_id != result.config.experiment_id:
        raise ValueError("load evidence experiment ID mismatch")
    return assess_load_state(result.measurements, evidence=evidence)


def _feasible(result: ExperimentResult, rate: float, spec: ComparisonSpec, *,
              load_evidence: Optional[LoadEvidence] = None) -> bool:
    """Legacy boolean helper: False includes unknown; never use False as proof of overload."""
    return _load(result, load_evidence).state == "healthy"


def _derive(spec, incumbent, candidate, incumbent_load_evidence=None, candidate_load_evidence=None):
    problems = _sound("incumbent", incumbent) + _sound("candidate", candidate)
    if problems:
        raise ValueError("config comparison evidence is unsound: " + ", ".join(problems))
    _same_context(incumbent, candidate)
    il, cl = _load(incumbent, incumbent_load_evidence), _load(candidate, candidate_load_evidence)
    if il.evidence is not None and cl.evidence is not None:
        if il.evidence.boundaries_s != cl.evidence.boundaries_s:
            raise ValueError("comparison requires aligned replay windows")
    if cl.state == "overloaded":
        return "candidate_infeasible", ["candidate_overloaded", *cl.reasons]
    if cl.state != "healthy" or il.state in ("indeterminate", "near_capacity"):
        return "inconclusive_evidence", [f"candidate_load_{cl.state}", f"incumbent_load_{il.state}"]
    if il.evidence.replay_sha256 is None or cl.evidence.replay_sha256 is None:
        return "inconclusive_evidence", ["missing_intended_replay_identity"]
    if il.evidence.replay_sha256 != cl.evidence.replay_sha256:
        raise ValueError("comparison requires the same intended replay")
    # Same planned trace can have small observed admission-time jitter at bin edges.
    ia, ca = il.evidence.offered_output_tokens, cl.evidence.offered_output_tokens
    if ia is None or ca is None or any(abs(a - b) > 0.05 * max(a, b) for a, b in zip(ia, ca)):
        return "inconclusive_evidence", ["offered_work_not_comparable_within_five_percent"]
    reasons = ["incumbent_overloaded"] if il.state == "overloaded" else []
    # Compare the same arrival cohort, not warmup/drain-mixed run percentiles.
    lo, hi = cl.evidence.boundaries_s[0], cl.evidence.boundaries_s[-1]
    aggregates = [compute_aggregates([m for m in r.measurements if lo <= m.start_time_s < hi], hi - lo)
                  for r in (incumbent, candidate)]
    not_worse, better = True, False
    for name in _GUARDED:
        iv, cv = getattr(aggregates[0], name), getattr(aggregates[1], name)
        if iv is None or cv is None:
            return "inconclusive_evidence", ["missing_cohort_latency_metric"]
        if cv < iv and cv <= iv * (1 - spec.rel_tolerance):
            better = True
            reasons.append(f"{name}_better({cv:.1f}<={iv:.1f})")
        elif cv <= iv * (1 + spec.rel_tolerance):
            reasons.append(f"{name}_tied({cv:.1f}~{iv:.1f})")
        else:
            not_worse = False
            reasons.append(f"{name}_worse({cv:.1f}>{iv:.1f})")
    if not_worse and better:
        return "candidate_dominates", reasons
    return ("incumbent_kept" if not_worse else "inconclusive_tradeoff"), reasons


class ConfigComparison(SchemaModel):
    spec: ComparisonSpec
    incumbent: ExperimentResult
    candidate: ExperimentResult
    incumbent_load_evidence: Optional[LoadEvidence] = None
    candidate_load_evidence: Optional[LoadEvidence] = None
    verdict: Verdict
    should_switch: bool
    reasons: list[str]

    @model_validator(mode="after")
    def _check(self) -> "ConfigComparison":
        verdict, reasons = _derive(self.spec, self.incumbent, self.candidate,
                                   self.incumbent_load_evidence, self.candidate_load_evidence)
        if (self.verdict, self.should_switch, self.reasons) != (verdict, verdict == "candidate_dominates", reasons):
            raise ValueError("config comparison is inconsistent with the measured evidence")
        return self


def compare_configs(spec: ComparisonSpec, incumbent: ExperimentResult, candidate: ExperimentResult, *,
                    incumbent_load_evidence: Optional[LoadEvidence] = None,
                    candidate_load_evidence: Optional[LoadEvidence] = None) -> ConfigComparison:
    verdict, reasons = _derive(spec, incumbent, candidate, incumbent_load_evidence, candidate_load_evidence)
    return ConfigComparison(spec=spec, incumbent=incumbent, candidate=candidate,
                            incumbent_load_evidence=incumbent_load_evidence,
                            candidate_load_evidence=candidate_load_evidence,
                            verdict=verdict, should_switch=verdict == "candidate_dominates", reasons=reasons)
