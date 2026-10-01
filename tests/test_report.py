"""The plain-English renderings read correctly and invent no numbers."""

from __future__ import annotations

from inferpilot.capacity_frontier import CapacityPoint, estimate_frontier
from inferpilot.config import SLO
from inferpilot.lever_forecast import forecast_lever
from inferpilot.lever_impact import compare_lever
from inferpilot.report import render_forecast, render_frontier, render_impact

SLO_TTFT = SLO(ttft_p95_ms=250.0)
FP8 = "kv_cache_dtype=fp8"


def _pt(offered, *, margin, tokens_per_s=800.0, regime=None, lever=None):
    return CapacityPoint(offered_qps=offered, achieved_qps=offered, tokens_per_s=tokens_per_s,
                         overloaded=False, slo_margin=margin, regime=regime, recommended_lever=lever)


def test_render_bracketed_frontier_states_ceiling_and_cost() -> None:
    f = estimate_frontier([_pt(4.0, margin=0.1), _pt(6.0, margin=-0.4)], SLO_TTFT,
                          gpu_cost_per_hour_usd=1.10)
    out = render_frontier(f)
    assert "Capacity ceiling" in out and "QPS" in out
    assert "/ 1M output tokens" in out
    assert "TTFT p95 <= 250 ms" in out


def test_render_no_ceiling_points_higher() -> None:
    f = estimate_frontier([_pt(2.0, margin=0.5), _pt(4.0, margin=0.3)], SLO_TTFT)
    out = render_frontier(f)
    assert "No ceiling found" in out and "measure higher" in out


def test_render_forecast_worth_testing() -> None:
    f = estimate_frontier([_pt(4.0, margin=0.2), _pt(6.0, margin=-0.1, regime="kv_pressure", lever=FP8)], SLO_TTFT)
    out = render_forecast(forecast_lever(FP8, f))
    assert "WORTH TESTING" in out and FP8 in out


def test_render_forecast_not_worth() -> None:
    f = estimate_frontier([_pt(4.0, margin=0.2), _pt(6.0, margin=-0.1, regime="no_load_pressure", lever="none")], SLO_TTFT)
    out = render_forecast(forecast_lever(FP8, f))
    assert "NOT WORTH TESTING" in out


def test_render_impact_raises() -> None:
    base = estimate_frontier([_pt(4.0, margin=0.1), _pt(6.0, margin=-0.4)], SLO_TTFT, gpu_cost_per_hour_usd=1.10)
    cand = estimate_frontier([_pt(4.0, margin=0.4, tokens_per_s=1000.0), _pt(6.0, margin=-0.1, tokens_per_s=1000.0)],
                             SLO_TTFT, gpu_cost_per_hour_usd=1.10)
    out = render_impact(compare_lever(FP8, base, cand))
    assert "RAISES capacity" in out and "->" in out
    assert "Cost/token down" in out
