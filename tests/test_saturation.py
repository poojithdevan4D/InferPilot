"""TTFT-stability saturation signal — drain-robust 'is the server keeping up?'."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import RequestMeasurement, SaturationReport, detect_saturation


def _meas(ttfts, *, spacing_s=0.5):
    """Measurements in arrival order with the given per-request TTFTs (ms)."""
    out = []
    for i, ttft in enumerate(ttfts):
        start = i * spacing_s
        out.append(RequestMeasurement(
            request_id=str(i), prompt_tokens=128, output_tokens=32,
            start_time_s=start, end_time_s=start + ttft / 1000 + 1.0,
            ttft_ms=ttft, tpot_ms=8.0, e2e_latency_ms=ttft + 248, success=True,
        ))
    return out


def test_stable_ttft_is_not_saturated() -> None:
    rep = detect_saturation(_meas([100 + (i % 5) for i in range(40)]))
    assert not rep.saturated and rep.reason == "ttft_stable"


def test_rising_ttft_is_saturated() -> None:
    # early ~100ms, late ~300ms -> queue growing
    rep = detect_saturation(_meas([100] * 20 + [300] * 20))
    assert rep.saturated and rep.growth_ratio == pytest.approx(3.0)
    assert rep.reason == "ttft_growing_across_arrivals"


def test_drain_tail_does_not_trigger_saturation() -> None:
    # The 2026-09-18 decode case: flat TTFT but sparsely spaced over a long window
    # (low throughput/duration). Duration-independent signal must call this NOT saturated.
    rep = detect_saturation(_meas([120 + (i % 8) for i in range(96)], spacing_s=20.0))
    assert not rep.saturated


def test_insufficient_samples_is_inconclusive_not_saturated() -> None:
    rep = detect_saturation(_meas([100, 110, 120]))
    assert not rep.saturated and rep.reason == "insufficient_samples"
    assert rep.growth_ratio is None


def test_report_roundtrips_and_rejects_tampering() -> None:
    rep = detect_saturation(_meas([100] * 20 + [300] * 20))
    assert SaturationReport.model_validate_json(rep.model_dump_json()) == rep
    raw = rep.model_dump(mode="json")
    raw["saturated"] = False  # contradicts growth_ratio 3.0 > threshold 1.5
    with pytest.raises(ValidationError, match="saturated flag inconsistent"):
        SaturationReport.model_validate(raw)
