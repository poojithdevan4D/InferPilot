"""Before you spend a second rate sweep: is this lever even worth testing?

``compare_lever`` tells you what a lever *did*, but only after you have run the
candidate sweep. That sweep costs GPU time, so an engineer wants a cheaper question
answered first: given what already limits my capacity ceiling, is this lever likely
to move it, or am I about to measure a placebo?

This forecast reads the bottleneck at the wall — the operating point just above the
ceiling — and answers honestly:

  * ``worth_testing``        the ceiling is set by a bottleneck this lever relieves
                             (e.g. a KV-preemption wall and ``kv_cache_dtype=fp8``);
                             run the candidate sweep, then confirm with ``compare_lever``.
  * ``not_worth_testing``    the wall is something this lever does not address; spend
                             the sweep on the right lever (or on scaling) instead.
  * ``collect_evidence_first`` the ceiling is pinned but the wall carries no bottleneck
                             diagnosis; attach aligned load evidence and re-diagnose.
  * ``measure_ceiling_first``  the sweep never pinned a ceiling; find it before forecasting.

It is a forecast, not a promise: a ``worth_testing`` still has to be confirmed by a
measured ``compare_lever``. The value is in the negatives — it stops an engineer from
paying for a sweep that cannot help.

Recomputes its verdict from the embedded frontier on load, so it is tamper-evident.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .capacity_frontier import CapacityFrontier, CapacityPoint

ForecastVerdict = Literal[
    "worth_testing",
    "not_worth_testing",
    "collect_evidence_first",
    "measure_ceiling_first",
]


def _wall_point(frontier: CapacityFrontier) -> Optional[CapacityPoint]:
    """The measured point just above the ceiling — the one whose bottleneck sets it."""
    if frontier.ceiling_qps is None:
        return None
    above = [p for p in frontier.points if p.offered_qps >= frontier.ceiling_qps]
    return min(above, key=lambda p: p.offered_qps) if above else None


def _derive(frontier: CapacityFrontier, lever: str) -> tuple[ForecastVerdict, Optional[float], list[str]]:
    """Returns (verdict, suggested_probe_qps, reasons)."""
    if frontier.verdict != "ceiling_bracketed":
        return "measure_ceiling_first", frontier.suggested_next_qps, [
            f"frontier_{frontier.verdict}_find_the_ceiling_before_forecasting_a_lever"
        ]

    wall = _wall_point(frontier)
    if wall is None or wall.regime is None:
        return "collect_evidence_first", frontier.ceiling_qps, [
            "ceiling_pinned_but_wall_point_has_no_bottleneck_diagnosis",
            "attach_aligned_load_evidence_and_re_diagnose_the_limiting_run",
        ]

    if wall.recommended_lever == lever and lever != "none":
        return "worth_testing", frontier.ceiling_qps, [
            f"ceiling_limited_by_{wall.regime}",
            f"{lever}_is_the_mechanistic_lever_for_this_wall",
            "run_the_candidate_sweep_then_confirm_with_compare_lever",
        ]

    return "not_worth_testing", None, [
        f"ceiling_limited_by_{wall.regime}",
        f"{lever}_does_not_address_this_wall_recommended_lever_is_{wall.recommended_lever or 'none'}",
    ]


class LeverForecast(SchemaModel):
    """Self-validating pre-measurement verdict on whether a lever is worth sweeping."""

    lever: str = Field(min_length=1)
    frontier: CapacityFrontier
    verdict: ForecastVerdict
    suggested_probe_qps: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    reasons: list[str]

    @model_validator(mode="after")
    def _check(self) -> "LeverForecast":
        verdict, probe, reasons = _derive(self.frontier, self.lever)
        if (self.verdict, self.suggested_probe_qps, self.reasons) != (verdict, probe, reasons):
            raise ValueError("lever forecast is inconsistent with its frontier")
        return self


def forecast_lever(lever: str, frontier: CapacityFrontier) -> LeverForecast:
    """Forecast whether ``lever`` is worth a candidate sweep, from the frontier's wall point.

    The frontier's points should carry their ``regime`` / ``recommended_lever`` (set from
    ``diagnose()`` on each run) for a substantive forecast; without them the verdict is
    ``collect_evidence_first``.
    """
    verdict, probe, reasons = _derive(frontier, lever)
    return LeverForecast(
        lever=lever, frontier=frontier, verdict=verdict,
        suggested_probe_qps=probe, reasons=reasons,
    )
