"""Measured, fail-closed comparison of a candidate config against an incumbent.

Born from the 2026-09-18 7B campaign, where a single-metric objective (minimise
tpot_p95) declared "winners" that were in fact 5-15x worse on ttft_p95 — an illusory
win driven by concurrency queueing. This contract makes that mistake impossible:

  * Feasibility first. A config that cannot keep up with the offered request rate
    (throughput < keepup_fraction * rate) is overloaded and can never "win".
  * Pareto, not single-metric. Among feasible configs, a candidate only beats the
    incumbent if it is no worse on EVERY guarded latency metric (ttft_p95, tpot_p95)
    and strictly better on at least one — both judged with a relative tolerance so
    measurement noise can't flip the verdict.
  * Prefer the incumbent. Ties and trade-offs keep the incumbent (e.g. the vLLM
    default). InferPilot only recommends switching away from a default it has
    measured to be dominated.

The comparison recomputes its verdict from the two embedded results on load, and
re-derives each result's aggregates from raw measurements, so tampering is rejected.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import SchemaModel
from ..results import ExperimentResult
from ..runner.aggregate import compute_aggregates

# lower-is-better latency metrics guarded by the Pareto test
_GUARDED = ("ttft_p95_ms", "tpot_p95_ms")

Verdict = Literal[
    "candidate_dominates",   # feasible and Pareto-better -> switch
    "incumbent_kept",        # tie within tolerance -> keep incumbent
    "inconclusive_tradeoff", # better on some, worse on others -> keep incumbent
    "candidate_infeasible",  # candidate cannot keep up with the offered rate
]


class ComparisonSpec(SchemaModel):
    comparison_version: Literal["0.1.0"] = "0.1.0"
    keepup_fraction: float = Field(default=0.95, gt=0, le=1)
    rel_tolerance: float = Field(default=0.02, ge=0, description="Relative noise band.")


class ConfigComparison(SchemaModel):
    """Self-validating candidate-vs-incumbent verdict over two measured results."""

    spec: ComparisonSpec
    incumbent: ExperimentResult
    candidate: ExperimentResult
    verdict: Verdict
    should_switch: bool
    reasons: list[str]

    @model_validator(mode="after")
    def _check(self) -> "ConfigComparison":
        verdict, reasons = _derive(self.spec, self.incumbent, self.candidate)
        if (self.verdict, self.should_switch, self.reasons) != (
            verdict, verdict == "candidate_dominates", reasons
        ):
            raise ValueError("config comparison is inconsistent with the measured evidence")
        return self


def _sound(label: str, result: ExperimentResult) -> list[str]:
    """Integrity + eligibility problems for one result (tamper-evident)."""
    problems: list[str] = []
    if not result.is_baseline_eligible:
        problems.append(f"{label}_not_baseline_eligible")
    agg = result.aggregates
    if agg is None:
        problems.append(f"{label}_aggregates_missing")
        return problems
    recomputed = compute_aggregates(result.measurements, agg.duration_s)
    if agg.model_dump(exclude={"gpu_memory_peak_mb"}) != recomputed.model_dump(
        exclude={"gpu_memory_peak_mb"}
    ):
        problems.append(f"{label}_aggregates_inconsistent_with_measurements")
    if agg.ttft_p95_ms is None or agg.tpot_p95_ms is None or agg.throughput_requests_per_s is None:
        problems.append(f"{label}_missing_guarded_metric")
    return problems


def _same_context(incumbent: ExperimentResult, candidate: ExperimentResult) -> None:
    iw, cw = incumbent.config.workload, candidate.config.workload
    ie, ce = incumbent.config.engine, candidate.config.engine
    same = (
        ie.model == ce.model and ie.revision == ce.revision
        and incumbent.environment.hardware.gpu_name == candidate.environment.hardware.gpu_name
        and iw.prompt_tokens == cw.prompt_tokens and iw.output_tokens == cw.output_tokens
        and iw.arrival_pattern == cw.arrival_pattern
        and iw.request_rate_qps == cw.request_rate_qps
    )
    if not same:
        raise ValueError("config comparison requires the same model/hardware/workload context")


def _feasible(result: ExperimentResult, rate: float, keepup: float) -> bool:
    return result.aggregates.throughput_requests_per_s >= keepup * rate


def _derive(
    spec: ComparisonSpec, incumbent: ExperimentResult, candidate: ExperimentResult
) -> tuple[Verdict, list[str]]:
    problems = _sound("incumbent", incumbent) + _sound("candidate", candidate)
    if problems:
        raise ValueError("config comparison evidence is unsound: " + ", ".join(problems))
    _same_context(incumbent, candidate)

    rate = candidate.config.workload.request_rate_qps
    reasons: list[str] = []
    if not _feasible(candidate, rate, spec.keepup_fraction):
        reasons.append("candidate_overloaded")
        return "candidate_infeasible", reasons
    if not _feasible(incumbent, rate, spec.keepup_fraction):
        # Feasibility trumps: a candidate that keeps up beats an overloaded incumbent
        # outright — the incumbent's per-token metric is not a valid competitor.
        reasons.append("incumbent_overloaded")
        return "candidate_dominates", reasons

    ia, ca = incumbent.aggregates, candidate.aggregates
    tol = spec.rel_tolerance
    not_worse, strictly_better = True, False
    for m in _GUARDED:
        iv, cv = getattr(ia, m), getattr(ca, m)
        if cv <= iv * (1 - tol):
            strictly_better = True
            reasons.append(f"{m}_better({cv:.1f}<={iv:.1f})")
        elif cv <= iv * (1 + tol):
            reasons.append(f"{m}_tied({cv:.1f}~{iv:.1f})")
        else:
            not_worse = False
            reasons.append(f"{m}_worse({cv:.1f}>{iv:.1f})")

    if not_worse and strictly_better:
        return "candidate_dominates", reasons
    if not_worse:
        return "incumbent_kept", reasons
    return "inconclusive_tradeoff", reasons


def compare_configs(
    spec: ComparisonSpec, incumbent: ExperimentResult, candidate: ExperimentResult
) -> ConfigComparison:
    """Fail-closed Pareto verdict; switch away from the incumbent only if dominated."""
    verdict, reasons = _derive(spec, incumbent, candidate)
    return ConfigComparison(
        spec=spec, incumbent=incumbent, candidate=candidate,
        verdict=verdict, should_switch=verdict == "candidate_dominates", reasons=reasons,
    )
