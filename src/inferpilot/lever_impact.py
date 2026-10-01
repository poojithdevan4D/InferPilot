"""What a config lever actually did to your capacity ceiling and cost.

An inference engineer does not want "fp8 is faster" — they want "turning on fp8 KV
moved my SLO-capacity ceiling from 4.2 to 5.6 QPS and cut cost-per-token by 18%, on
my model, my GPU, my SLO." That is a *measured* claim: run a rate sweep on the
incumbent config, run the same sweep with the lever applied, and compare the two
honest capacity frontiers.

This module does only that comparison, and it inherits the frontier's honesty:

  * It states a numeric ceiling shift only when **both** sweeps pinned a bracketed
    ceiling. If the candidate never hit its SLO wall within the measured rates, the
    truthful statement is "the ceiling moved past <top measured rate>", not a number.
  * If either sweep was inconclusive (too few points, already over SLO, a too-wide or
    non-monotone bracket), the impact is ``inconclusive`` and says why.
  * It refuses to compare sweeps run against different SLOs.

The report recomputes its verdict from the two embedded frontiers on load, so a
tampered impact claim is rejected.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .capacity_frontier import CapacityFrontier

ImpactVerdict = Literal[
    "lever_raises_ceiling",         # both bracketed; candidate ceiling higher beyond tolerance
    "lever_lowers_ceiling",         # both bracketed; candidate ceiling lower beyond tolerance
    "lever_neutral",                # both bracketed; ceilings within tolerance
    "lever_raises_beyond_measured", # baseline bracketed; candidate never hit its SLO wall
    "inconclusive",                 # either sweep did not pin a comparable ceiling
]


def _top_offered(frontier: CapacityFrontier) -> float:
    return max(p.offered_qps for p in frontier.points)


def _derive(
    baseline: CapacityFrontier, candidate: CapacityFrontier, rel_tolerance: float
) -> tuple[ImpactVerdict, Optional[float], Optional[float], Optional[float], list[str]]:
    """Returns (verdict, ceiling_delta_qps, ceiling_delta_pct, cost_delta_pct, reasons)."""
    if baseline.slo != candidate.slo:
        raise ValueError("lever impact requires both sweeps to use the same SLO")

    b, c = baseline.verdict, candidate.verdict

    if b == "ceiling_bracketed" and c == "ceiling_bracketed":
        base_ceil, cand_ceil = baseline.ceiling_qps, candidate.ceiling_qps
        assert base_ceil is not None and cand_ceil is not None  # invariant of the verdict
        delta = round(cand_ceil - base_ceil, 4)
        pct = round(delta / base_ceil, 4) if base_ceil else None
        reasons = [f"ceiling_{base_ceil:g}->{cand_ceil:g}_qps"]
        cost_pct = _cost_delta_pct(baseline, candidate, reasons)
        if pct is not None and pct > rel_tolerance:
            return "lever_raises_ceiling", delta, pct, cost_pct, reasons
        if pct is not None and pct < -rel_tolerance:
            return "lever_lowers_ceiling", delta, pct, cost_pct, reasons
        reasons.append(f"within_{rel_tolerance:g}_tolerance")
        return "lever_neutral", delta, pct, cost_pct, reasons

    if b == "ceiling_bracketed" and c == "no_ceiling_observed":
        top = _top_offered(candidate)
        base_ceil = baseline.ceiling_qps
        assert base_ceil is not None
        return "lever_raises_beyond_measured", None, None, None, [
            f"baseline_ceiling_{base_ceil:g}_qps",
            f"candidate_met_slo_through_{top:g}_qps_no_wall_hit_measure_higher",
        ]

    # Everything else: at least one side gives no comparable ceiling.
    return "inconclusive", None, None, None, [
        f"baseline_verdict_{b}", f"candidate_verdict_{c}",
        "need_a_bracketed_ceiling_on_both_sweeps_or_a_candidate_that_clears_the_baseline_range",
    ]


def _cost_delta_pct(
    baseline: CapacityFrontier, candidate: CapacityFrontier, reasons: list[str]
) -> Optional[float]:
    bc = baseline.cost_per_million_output_tokens_usd
    cc = candidate.cost_per_million_output_tokens_usd
    if bc is None or cc is None or bc == 0:
        return None
    pct = round((cc - bc) / bc, 4)
    reasons.append(f"cost_per_mtok_{bc:g}->{cc:g}_usd")
    return pct


class LeverImpact(SchemaModel):
    """Self-validating measured impact of one config lever on the capacity frontier."""

    lever: str = Field(min_length=1, description="The single config change under test, e.g. 'kv_cache_dtype=fp8'.")
    rel_tolerance: float = Field(default=0.03, ge=0, description="Relative band below which a ceiling shift is 'neutral'.")
    baseline: CapacityFrontier
    candidate: CapacityFrontier
    verdict: ImpactVerdict
    ceiling_delta_qps: Optional[float] = Field(default=None, allow_inf_nan=False)
    ceiling_delta_pct: Optional[float] = Field(default=None, allow_inf_nan=False)
    cost_delta_pct: Optional[float] = Field(default=None, allow_inf_nan=False)
    reasons: list[str]

    @model_validator(mode="after")
    def _check(self) -> "LeverImpact":
        verdict, dq, dp, dc, reasons = _derive(self.baseline, self.candidate, self.rel_tolerance)
        if (self.verdict, self.ceiling_delta_qps, self.ceiling_delta_pct,
                self.cost_delta_pct, self.reasons) != (verdict, dq, dp, dc, reasons):
            raise ValueError("lever impact is inconsistent with its measured frontiers")
        return self


def compare_lever(
    lever: str, baseline: CapacityFrontier, candidate: CapacityFrontier, *,
    rel_tolerance: float = 0.03,
) -> LeverImpact:
    """Measured capacity + cost impact of ``lever``, from a baseline and a candidate sweep.

    ``baseline`` is the incumbent config's rate sweep; ``candidate`` is the same sweep with
    the lever applied. Both must have been built against the same SLO. The result states a
    numeric ceiling shift only when both sweeps pinned a bracketed ceiling.
    """
    verdict, dq, dp, dc, reasons = _derive(baseline, candidate, rel_tolerance)
    return LeverImpact(
        lever=lever, rel_tolerance=rel_tolerance, baseline=baseline, candidate=candidate,
        verdict=verdict, ceiling_delta_qps=dq, ceiling_delta_pct=dp, cost_delta_pct=dc,
        reasons=reasons,
    )
