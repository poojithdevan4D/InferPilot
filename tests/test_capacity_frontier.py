"""SLO-capacity frontier: estimate a ceiling ONLY when measured points bracket the
SLO boundary, and abstain (naming the next rate to measure) otherwise."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from inferpilot.capacity_frontier import (
    CapacityFrontier,
    CapacityPoint,
    estimate_frontier,
)
from inferpilot.config import SLO


def _pt(offered, *, achieved=None, tokens_per_s=800.0, overloaded=False, margin=None):
    return CapacityPoint(
        offered_qps=offered,
        achieved_qps=offered if achieved is None else achieved,
        tokens_per_s=tokens_per_s,
        overloaded=overloaded,
        slo_margin=margin,
    )


SLO_TTFT = SLO(ttft_p95_ms=250.0)


def test_bracketed_ceiling_interpolates_on_latency_margin() -> None:
    # feasible at 4 qps (20% headroom), breaches at 6 qps (10% over) -> ceiling in between.
    pts = [_pt(4.0, margin=0.2), _pt(6.0, margin=-0.1)]
    f = estimate_frontier(pts, SLO_TTFT, gpu_cost_per_hour_usd=1.10)
    assert f.verdict == "ceiling_bracketed"
    assert math.isclose(f.ceiling_qps, 5.3333, abs_tol=1e-3)   # 4 + (0.2/0.3)*2
    assert f.suggested_next_qps is None
    # conservative $/1M tokens anchored on the feasible point's 800 tok/s.
    assert math.isclose(f.cost_per_million_output_tokens_usd, 1.10 / 800 / 3600 * 1e6, abs_tol=1e-3)


def test_keepup_breach_interpolates_when_no_latency_slo() -> None:
    # No SLO thresholds: feasibility rests on keep-up. Server stops keeping up between rates.
    pts = [_pt(4.0, achieved=4.0), _pt(6.0, achieved=4.5, overloaded=True)]
    f = estimate_frontier(pts, SLO())
    assert f.verdict == "ceiling_bracketed"
    # gap: lo=4-3.8=0.2, hi=4.5-5.7=-1.2 -> frac=0.2/1.4 -> 4 + (1/7)*2
    assert math.isclose(f.ceiling_qps, 4.2857, abs_tol=1e-3)
    assert f.cost_per_million_output_tokens_usd is None   # no economics given


def test_all_rates_pass_abstains_and_probes_higher() -> None:
    pts = [_pt(2.0, margin=0.5), _pt(4.0, margin=0.3)]
    f = estimate_frontier(pts, SLO_TTFT)
    assert f.verdict == "no_ceiling_observed"
    assert f.ceiling_qps is None
    assert f.suggested_next_qps == 6.0     # 4.0 * 1.5


def test_all_rates_fail_abstains_and_probes_lower() -> None:
    pts = [_pt(4.0, margin=-0.1), _pt(6.0, margin=-0.4)]
    f = estimate_frontier(pts, SLO_TTFT)
    assert f.verdict == "already_over_slo"
    assert f.suggested_next_qps == pytest.approx(4.0 / 1.5, abs=1e-3)


def test_single_point_is_insufficient() -> None:
    f = estimate_frontier([_pt(4.0, margin=0.2)], SLO_TTFT)
    assert f.verdict == "insufficient_evidence"
    assert "need_at_least_two_measured_rates" in f.reasons


def test_wide_bracket_abstains_and_bisects() -> None:
    # feasible at 2, breaks at 10: rel width (10-2)/10 = 0.8 > 0.6 default -> too wide.
    pts = [_pt(2.0, margin=0.4), _pt(10.0, margin=-0.2)]
    f = estimate_frontier(pts, SLO_TTFT)
    assert f.verdict == "insufficient_evidence"
    assert any("bracket_too_wide" in r for r in f.reasons)
    assert f.suggested_next_qps == 6.0     # midpoint of 2 and 10


def test_non_monotone_curve_is_rejected() -> None:
    # an infeasible point BELOW a feasible one -> curve is not monotone, refuse to interpolate.
    pts = [_pt(3.0, margin=-0.1), _pt(5.0, margin=0.2)]
    f = estimate_frontier(pts, SLO_TTFT)
    assert f.verdict == "insufficient_evidence"
    assert any("non_monotone" in r for r in f.reasons)


def test_frontier_roundtrips_and_rejects_tampering() -> None:
    pts = [_pt(4.0, margin=0.2), _pt(6.0, margin=-0.1)]
    f = estimate_frontier(pts, SLO_TTFT, gpu_cost_per_hour_usd=1.10)
    assert CapacityFrontier.model_validate_json(f.model_dump_json()) == f
    raw = f.model_dump(mode="json")
    raw["ceiling_qps"] = 99.0       # a ceiling its own points do not support
    with pytest.raises(ValidationError, match="inconsistent with its measured points"):
        CapacityFrontier.model_validate(raw)


def test_duplicate_offered_rates_rejected() -> None:
    with pytest.raises(ValueError, match="distinct offered rates"):
        estimate_frontier([_pt(4.0, margin=0.2), _pt(4.0, margin=-0.1)], SLO_TTFT)
