"""Honest SLO-capacity frontier: how much load can this deployment take?

The existing ``CapacityAdvisory`` describes the single operating point you measured,
and ``recommend_scale`` extrapolates linearly from one run. Neither answers the
question a platform team actually plans around:

    "How much offered load can this deployment sustain before it breaks my latency
     SLO, and what does it cost per token at that ceiling?"

That question is hard precisely because serving performance is non-linear: there is
a saturation knee where the scheduler starts preempting and p95 latency explodes.
Extrapolating a line through that knee gives a confident, wrong answer — worse than
none.

This module refuses to do that. It estimates the capacity ceiling **only between two
measured rates that bracket the SLO boundary**, by local interpolation on the metric
that actually breaches. If the measured points do not bracket the boundary — every
point passes, every point fails, or the bracket is too wide or non-monotone — it
returns a non-numeric verdict and names the exact next rate to measure. Abstention is
the point: an unbracketed ceiling is a guess, and a guess about capacity is how teams
either overpay for idle GPUs or fall over at peak.

The frontier recomputes its verdict from its embedded points and cost inputs on load,
so a tampered report is rejected.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .config import SLO
from .results import ExperimentResult

Verdict = Literal[
    "ceiling_bracketed",     # feasible and infeasible points bracket the SLO boundary -> numeric ceiling
    "no_ceiling_observed",   # every measured rate met the SLO -> measure higher
    "already_over_slo",      # even the lowest measured rate broke the SLO -> measure lower
    "insufficient_evidence", # <2 points, or the bracket is too wide / non-monotone -> measure the named rate
]


class CapacityPoint(SchemaModel):
    """One measured operating point, reduced to what the frontier needs.

    ``slo_margin`` is the fraction of headroom against the binding SLO metric:
    ``min_over_metrics((threshold - measured) / threshold)``. It is >= 0 when every
    provided SLO threshold is met and < 0 once any is breached; ``None`` when no SLO
    threshold was provided (then feasibility rests on keep-up alone). ``binding_metric``
    names the metric closest to (or past) its threshold.
    """

    offered_qps: float = Field(gt=0, allow_inf_nan=False)
    achieved_qps: float = Field(ge=0, allow_inf_nan=False, description="Requests/s actually served.")
    tokens_per_s: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    overloaded: bool = Field(description="Load assessment / saturation says the server is not keeping up.")
    slo_margin: Optional[float] = Field(default=None, allow_inf_nan=False)
    binding_metric: Optional[str] = None

    def keeps_up(self, keepup_fraction: float) -> bool:
        return not self.overloaded and self.achieved_qps >= keepup_fraction * self.offered_qps

    def feasible(self, keepup_fraction: float) -> bool:
        margin_ok = self.slo_margin is None or self.slo_margin >= 0.0
        return margin_ok and self.keeps_up(keepup_fraction)


def _next_rate(low: Optional[float], high: Optional[float], *, factor: float = 1.5) -> float:
    """The single most informative next offered rate to measure."""
    if low is not None and high is not None:
        return round((low + high) / 2.0, 4)          # bisect the bracket
    if high is not None:                              # everything failed -> probe below the lowest
        return round(high / factor, 4)
    if low is not None:                               # everything passed -> probe above the highest
        return round(low * factor, 4)
    raise ValueError("need at least one measured rate to suggest the next")


def _interpolate_ceiling(
    lo: CapacityPoint, hi: CapacityPoint, keepup_fraction: float
) -> tuple[float, list[str]]:
    """Interpolate the offered rate at which the binding constraint hits zero headroom.

    Strictly inside the measured bracket [lo, hi] — never an extrapolation past a
    measured point. If the SLO-latency margin is what breaches, interpolate on that;
    otherwise the server stopped keeping up, so interpolate on the keep-up gap.
    """
    lo_m, hi_m = lo.slo_margin, hi.slo_margin
    if lo_m is not None and hi_m is not None and lo_m >= 0 > hi_m:
        frac = lo_m / (lo_m - hi_m)  # in (0, 1]; where the latency margin crosses zero
        ceiling = lo.offered_qps + frac * (hi.offered_qps - lo.offered_qps)
        return round(ceiling, 4), [f"latency_slo_breach_between_{lo.offered_qps:g}_and_{hi.offered_qps:g}_qps"]

    lo_gap = lo.achieved_qps - keepup_fraction * lo.offered_qps
    hi_gap = hi.achieved_qps - keepup_fraction * hi.offered_qps
    if lo_gap >= 0 > hi_gap:
        frac = lo_gap / (lo_gap - hi_gap)
        ceiling = lo.offered_qps + frac * (hi.offered_qps - lo.offered_qps)
        return round(ceiling, 4), [f"throughput_keepup_breach_between_{lo.offered_qps:g}_and_{hi.offered_qps:g}_qps"]

    # Both constraints already point to the feasible rate as the last good one.
    return round(lo.offered_qps, 4), ["conservative_ceiling_at_last_feasible_rate"]


def _cost(tokens_per_s: Optional[float], gpu_cost_per_hour_usd: Optional[float],
          gpu_count: int) -> Optional[float]:
    if not tokens_per_s or gpu_cost_per_hour_usd is None:
        return None
    hourly = gpu_cost_per_hour_usd * gpu_count
    return round(hourly / tokens_per_s / 3600.0 * 1e6, 4)


def _derive(
    points: list[CapacityPoint], keepup_fraction: float, max_bracket_rel_width: float,
    gpu_cost_per_hour_usd: Optional[float], gpu_count: int,
) -> tuple[Verdict, Optional[float], Optional[float], list[str], Optional[float]]:
    """Returns (verdict, ceiling_qps, cost_per_million_tokens, reasons, suggested_next_qps)."""
    if len(points) < 2:
        lone = points[0].offered_qps if points else None
        return "insufficient_evidence", None, None, ["need_at_least_two_measured_rates"], (
            _next_rate(lone, None) if lone is not None else None
        )

    pts = sorted(points, key=lambda p: p.offered_qps)
    if len({p.offered_qps for p in pts}) != len(pts):
        raise ValueError("capacity points must have distinct offered rates")

    feasible = [p for p in pts if p.feasible(keepup_fraction)]
    infeasible = [p for p in pts if not p.feasible(keepup_fraction)]

    if not infeasible:
        top = pts[-1]
        return "no_ceiling_observed", None, None, [
            f"all_{len(pts)}_rates_met_slo_up_to_{top.offered_qps:g}_qps"
        ], _next_rate(top.offered_qps, None)

    if not feasible:
        bottom = pts[0]
        return "already_over_slo", None, None, [
            f"lowest_measured_rate_{bottom.offered_qps:g}_qps_already_breaks_slo"
        ], _next_rate(None, bottom.offered_qps)

    # A ceiling exists between the highest feasible rate and the lowest infeasible rate
    # ABOVE it. An infeasible point below a feasible one means a non-monotone / regime-
    # ambiguous curve we will not interpolate through.
    hi_feasible = feasible[-1]
    above = [p for p in infeasible if p.offered_qps > hi_feasible.offered_qps]
    if not above:
        return "insufficient_evidence", None, None, [
            "infeasible_points_lie_below_a_feasible_rate_curve_is_non_monotone"
        ], _next_rate(hi_feasible.offered_qps, None)
    lo_infeasible = min(above, key=lambda p: p.offered_qps)

    rel_width = (lo_infeasible.offered_qps - hi_feasible.offered_qps) / lo_infeasible.offered_qps
    if rel_width > max_bracket_rel_width:
        return "insufficient_evidence", None, None, [
            f"bracket_too_wide_{rel_width:.2f}>{max_bracket_rel_width:.2f}_measure_between"
        ], _next_rate(hi_feasible.offered_qps, lo_infeasible.offered_qps)

    ceiling, reasons = _interpolate_ceiling(hi_feasible, lo_infeasible, keepup_fraction)
    # Conservative $/token: anchored on the highest-feasible point's measured throughput;
    # at the ceiling the deployment serves at least this many tokens/s.
    cost = _cost(hi_feasible.tokens_per_s, gpu_cost_per_hour_usd, gpu_count)
    return "ceiling_bracketed", ceiling, cost, reasons, None


class CapacityFrontier(SchemaModel):
    """Self-validating SLO-capacity estimate over >= 2 measured points."""

    slo: SLO
    keepup_fraction: float = Field(default=0.95, gt=0, le=1)
    max_bracket_rel_width: float = Field(default=0.6, gt=0, le=1)
    gpu_cost_per_hour_usd: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    gpu_count: int = Field(default=1, ge=1)
    points: list[CapacityPoint]
    verdict: Verdict
    ceiling_qps: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    cost_per_million_output_tokens_usd: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    suggested_next_qps: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    reasons: list[str]

    @model_validator(mode="after")
    def _check(self) -> "CapacityFrontier":
        verdict, ceiling, cost, reasons, nxt = _derive(
            self.points, self.keepup_fraction, self.max_bracket_rel_width,
            self.gpu_cost_per_hour_usd, self.gpu_count,
        )
        if (self.verdict, self.ceiling_qps, self.cost_per_million_output_tokens_usd,
                self.reasons, self.suggested_next_qps) != (verdict, ceiling, cost, reasons, nxt):
            raise ValueError("capacity frontier is inconsistent with its measured points")
        return self


def estimate_frontier(
    points: list[CapacityPoint], slo: SLO, *,
    keepup_fraction: float = 0.95, max_bracket_rel_width: float = 0.6,
    gpu_cost_per_hour_usd: Optional[float] = None, gpu_count: int = 1,
) -> CapacityFrontier:
    """Pure estimate over already-reduced points. Abstains unless the SLO boundary is bracketed."""
    verdict, ceiling, cost, reasons, nxt = _derive(
        points, keepup_fraction, max_bracket_rel_width, gpu_cost_per_hour_usd, gpu_count
    )
    return CapacityFrontier(
        slo=slo, keepup_fraction=keepup_fraction, max_bracket_rel_width=max_bracket_rel_width,
        gpu_cost_per_hour_usd=gpu_cost_per_hour_usd, gpu_count=gpu_count,
        points=points, verdict=verdict, ceiling_qps=ceiling,
        cost_per_million_output_tokens_usd=cost, suggested_next_qps=nxt, reasons=reasons,
    )


def _is_overloaded(result: ExperimentResult) -> bool:
    """The TTFT-trend saturation signal. saturated=True is a positive overload signal;
    saturated=False never certifies health, so feasibility also checks keep-up separately."""
    from .saturation import detect_saturation

    return bool(detect_saturation(result.measurements).saturated)


def _point_from_result(result: ExperimentResult, slo: SLO) -> CapacityPoint:
    agg = result.aggregates
    if agg is None:
        raise ValueError("capacity point requires a result with aggregates")
    offered = result.config.workload.request_rate_qps
    if offered is None or offered <= 0:
        raise ValueError("capacity frontier requires an offered request_rate_qps on every result")

    margin: Optional[float] = None
    binding: Optional[str] = None
    for name, threshold in (
        ("ttft_p95_ms", slo.ttft_p95_ms),
        ("tpot_p95_ms", slo.tpot_p95_ms),
        ("e2e_p95_ms", slo.e2e_p95_ms),
    ):
        if threshold is None:
            continue
        measured = getattr(agg, name)
        if measured is None:
            continue
        m = (threshold - measured) / threshold
        if margin is None or m < margin:
            margin, binding = m, name

    return CapacityPoint(
        offered_qps=offered,
        achieved_qps=agg.throughput_requests_per_s or 0.0,
        tokens_per_s=agg.throughput_tokens_per_s,
        overloaded=_is_overloaded(result),
        slo_margin=margin,
        binding_metric=binding,
    )


def _require_same_deployment(results: list[ExperimentResult]) -> None:
    first = results[0]
    fe, fw = first.config.engine, first.config.workload
    fh = first.environment.hardware.gpu_name
    for r in results[1:]:
        e, w = r.config.engine, r.config.workload
        same = (
            e.model == fe.model and e.revision == fe.revision
            and e.kv_cache_dtype == fe.kv_cache_dtype
            and r.environment.hardware.gpu_name == fh
            and w.prompt_tokens == fw.prompt_tokens and w.output_tokens == fw.output_tokens
        )
        if not same:
            raise ValueError(
                "capacity frontier requires the same model/hardware/config/workload-shape; "
                "only request_rate_qps may vary across the sweep"
            )


def frontier_from_results(
    results: list[ExperimentResult], slo: SLO, *,
    gpu_cost_per_hour_usd: Optional[float] = None, gpu_count: int = 1,
    keepup_fraction: float = 0.95, max_bracket_rel_width: float = 0.6,
) -> CapacityFrontier:
    """Build the frontier from a rate sweep of measured runs of the SAME deployment.

    All results must share model, hardware, KV dtype, and per-request workload shape —
    only the offered ``request_rate_qps`` may vary. Pass GPU economics to get a
    conservative $/token at the ceiling.
    """
    if len(results) < 2:
        raise ValueError("capacity frontier needs at least two runs at different offered rates")
    _require_same_deployment(results)
    points = [_point_from_result(r, slo) for r in results]
    return estimate_frontier(
        points, slo, keepup_fraction=keepup_fraction, max_bracket_rel_width=max_bracket_rel_width,
        gpu_cost_per_hour_usd=gpu_cost_per_hour_usd, gpu_count=gpu_count,
    )
