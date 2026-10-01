"""Measured impact of a config lever on the SLO-capacity ceiling: a numeric shift
only when both the baseline and candidate sweeps pinned a bracketed ceiling."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from inferpilot.capacity_frontier import CapacityPoint, estimate_frontier
from inferpilot.config import SLO
from inferpilot.lever_impact import LeverImpact, compare_lever

SLO_TTFT = SLO(ttft_p95_ms=250.0)
FP8 = "kv_cache_dtype=fp8"


def _pt(offered, *, margin, tokens_per_s=800.0):
    return CapacityPoint(offered_qps=offered, achieved_qps=offered,
                         tokens_per_s=tokens_per_s, overloaded=False, slo_margin=margin)


def _sweep(lo_margin, hi_margin, *, tokens_per_s=800.0, cost=None):
    pts = [_pt(4.0, margin=lo_margin, tokens_per_s=tokens_per_s),
           _pt(6.0, margin=hi_margin, tokens_per_s=tokens_per_s)]
    return estimate_frontier(pts, SLO_TTFT, gpu_cost_per_hour_usd=cost)


def test_lever_raises_ceiling_and_cuts_cost() -> None:
    base = _sweep(0.1, -0.4, tokens_per_s=800.0, cost=1.10)   # ceiling 4.4
    cand = _sweep(0.4, -0.1, tokens_per_s=1000.0, cost=1.10)  # ceiling 5.6, more tok/s
    imp = compare_lever(FP8, base, cand)
    assert imp.verdict == "lever_raises_ceiling"
    assert math.isclose(imp.ceiling_delta_qps, cand.ceiling_qps - base.ceiling_qps, abs_tol=1e-6)
    assert imp.ceiling_delta_pct > 0.03
    assert imp.cost_delta_pct < 0          # cheaper per token at the ceiling
    assert any("ceiling_" in r for r in imp.reasons)


def test_lever_neutral_within_tolerance() -> None:
    base = _sweep(0.3, -0.2)
    cand = _sweep(0.3, -0.2)               # identical -> same ceiling
    imp = compare_lever(FP8, base, cand)
    assert imp.verdict == "lever_neutral"
    assert imp.ceiling_delta_qps == 0.0


def test_lever_lowers_ceiling() -> None:
    base = _sweep(0.4, -0.1)               # ceiling 5.6
    cand = _sweep(0.1, -0.4)               # ceiling 4.4
    imp = compare_lever(FP8, base, cand)
    assert imp.verdict == "lever_lowers_ceiling"
    assert imp.ceiling_delta_qps < 0


def test_candidate_never_hits_wall_is_beyond_measured() -> None:
    base = _sweep(0.1, -0.4)                           # ceiling 4.4
    cand = estimate_frontier([_pt(4.0, margin=0.5), _pt(6.0, margin=0.3)], SLO_TTFT)  # all pass
    assert cand.verdict == "no_ceiling_observed"
    imp = compare_lever(FP8, base, cand)
    assert imp.verdict == "lever_raises_beyond_measured"
    assert imp.ceiling_delta_qps is None
    assert any("through_6" in r for r in imp.reasons)


def test_inconclusive_when_a_sweep_has_no_ceiling() -> None:
    base = estimate_frontier([_pt(4.0, margin=0.2)], SLO_TTFT)   # single point -> insufficient
    cand = _sweep(0.4, -0.1)
    imp = compare_lever(FP8, base, cand)
    assert imp.verdict == "inconclusive"
    assert any("insufficient_evidence" in r for r in imp.reasons)


def test_mismatched_slo_is_rejected() -> None:
    base = _sweep(0.1, -0.4)
    other = estimate_frontier(
        [_pt(4.0, margin=0.1), _pt(6.0, margin=-0.4)], SLO(ttft_p95_ms=300.0)
    )
    with pytest.raises(ValueError, match="same SLO"):
        compare_lever(FP8, base, other)


def test_impact_roundtrips_and_rejects_tampering() -> None:
    base = _sweep(0.1, -0.4, cost=1.10)
    cand = _sweep(0.4, -0.1, tokens_per_s=1000.0, cost=1.10)
    imp = compare_lever(FP8, base, cand)
    assert LeverImpact.model_validate_json(imp.model_dump_json()) == imp
    raw = imp.model_dump(mode="json")
    raw["verdict"] = "lever_neutral"      # contradicts the measured shift
    with pytest.raises(ValidationError, match="inconsistent with its measured frontiers"):
        LeverImpact.model_validate(raw)
