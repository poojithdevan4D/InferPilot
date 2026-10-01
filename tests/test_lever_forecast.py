"""Pre-measurement forecast: is a lever worth a candidate sweep, given what limits
the ceiling? The value is in the honest negatives."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot.capacity_frontier import CapacityPoint, estimate_frontier
from inferpilot.config import SLO
from inferpilot.lever_forecast import LeverForecast, forecast_lever

SLO_TTFT = SLO(ttft_p95_ms=250.0)
FP8 = "kv_cache_dtype=fp8"


def _pt(offered, *, margin, regime=None, lever=None):
    return CapacityPoint(offered_qps=offered, achieved_qps=offered, tokens_per_s=800.0,
                         overloaded=False, slo_margin=margin, regime=regime,
                         recommended_lever=lever)


def test_worth_testing_when_wall_is_kv_preemption_and_lever_matches() -> None:
    # feasible at 4, wall at 6 diagnosed as a KV-preemption bottleneck fp8 relieves.
    pts = [_pt(4.0, margin=0.2), _pt(6.0, margin=-0.1, regime="kv_pressure", lever=FP8)]
    f = estimate_frontier(pts, SLO_TTFT)
    fc = forecast_lever(FP8, f)
    assert fc.verdict == "worth_testing"
    assert fc.suggested_probe_qps == f.ceiling_qps
    assert any("kv_pressure" in r for r in fc.reasons)


def test_not_worth_testing_when_wall_is_not_this_lever() -> None:
    pts = [_pt(4.0, margin=0.2), _pt(6.0, margin=-0.1, regime="no_load_pressure", lever="none")]
    f = estimate_frontier(pts, SLO_TTFT)
    fc = forecast_lever(FP8, f)
    assert fc.verdict == "not_worth_testing"
    assert fc.suggested_probe_qps is None


def test_collect_evidence_first_when_wall_has_no_diagnosis() -> None:
    pts = [_pt(4.0, margin=0.2), _pt(6.0, margin=-0.1)]   # no regime attached
    f = estimate_frontier(pts, SLO_TTFT)
    fc = forecast_lever(FP8, f)
    assert fc.verdict == "collect_evidence_first"
    assert fc.suggested_probe_qps == f.ceiling_qps


def test_measure_ceiling_first_when_no_ceiling() -> None:
    pts = [_pt(2.0, margin=0.5), _pt(4.0, margin=0.3)]    # all pass -> no ceiling observed
    f = estimate_frontier(pts, SLO_TTFT)
    fc = forecast_lever(FP8, f)
    assert fc.verdict == "measure_ceiling_first"
    assert fc.suggested_probe_qps == f.suggested_next_qps   # probe higher


def test_forecast_roundtrips_and_rejects_tampering() -> None:
    pts = [_pt(4.0, margin=0.2), _pt(6.0, margin=-0.1, regime="kv_pressure", lever=FP8)]
    f = estimate_frontier(pts, SLO_TTFT)
    fc = forecast_lever(FP8, f)
    assert LeverForecast.model_validate_json(fc.model_dump_json()) == fc
    raw = fc.model_dump(mode="json")
    raw["verdict"] = "not_worth_testing"
    with pytest.raises(ValidationError, match="inconsistent with its frontier"):
        LeverForecast.model_validate(raw)
