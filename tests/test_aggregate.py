"""Unit tests: percentile computation, TPOT formula, and aggregation."""

from __future__ import annotations

import math

import pytest

from inferpilot import RequestMeasurement
from inferpilot.runner.aggregate import compute_aggregates, percentile
from inferpilot.runner.client import _compute_metrics


# --- percentile ------------------------------------------------------------- #


def test_percentile_matches_linear_interpolation() -> None:
    data = [1.0, 2.0, 3.0, 4.0]
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 4.0
    assert percentile(data, 50) == pytest.approx(2.5)   # rank 1.5
    assert percentile(data, 95) == pytest.approx(3.85)  # rank 2.85


def test_percentile_single_value() -> None:
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


# --- TPOT formula ----------------------------------------------------------- #


def test_tpot_formula() -> None:
    # start=0, first token at 10ms, end at 80ms, 8 output tokens.
    ttft, tpot, e2e = _compute_metrics(start=0.0, first_token=0.010, end=0.080, output_tokens=8)
    assert e2e == pytest.approx(80.0)
    assert ttft == pytest.approx(10.0)
    # (80 - 10) / (8 - 1) == 10.0 ms/token
    assert tpot == pytest.approx(10.0)


def test_tpot_absent_for_one_token() -> None:
    ttft, tpot, e2e = _compute_metrics(start=0.0, first_token=0.005, end=0.020, output_tokens=1)
    assert ttft == pytest.approx(5.0)
    assert tpot is None


def test_metrics_no_first_token() -> None:
    ttft, tpot, e2e = _compute_metrics(start=0.0, first_token=None, end=0.020, output_tokens=0)
    assert ttft is None and tpot is None
    assert e2e == pytest.approx(20.0)


# --- aggregation ------------------------------------------------------------ #


def _ok(rid: str, ttft: float, tpot: float, e2e: float, out: int = 8) -> RequestMeasurement:
    return RequestMeasurement(
        request_id=rid, prompt_tokens=128, output_tokens=out,
        start_time_s=0.0, end_time_s=e2e / 1000.0,
        ttft_ms=ttft, tpot_ms=tpot, e2e_latency_ms=e2e, success=True,
    )


def _fail(rid: str) -> RequestMeasurement:
    return RequestMeasurement(
        request_id=rid, prompt_tokens=128, output_tokens=0,
        start_time_s=0.0, end_time_s=0.1, success=False, error="boom",
    )


def test_compute_aggregates_counts_and_percentiles() -> None:
    measurements = [
        _ok("a", 10, 5, 100),
        _ok("b", 20, 6, 200),
        _ok("c", 30, 7, 300),
        _fail("d"),
    ]
    agg = compute_aggregates(measurements, duration_s=2.0)

    assert agg.num_requests == 4
    assert agg.num_successful == 3
    assert agg.num_failed == 1
    assert agg.total_output_tokens == 24  # 3 * 8
    assert agg.ttft_p50_ms == pytest.approx(20.0)
    assert agg.e2e_p50_ms == pytest.approx(200.0)
    # throughput uses successful tokens over the measured window
    assert agg.throughput_tokens_per_s == pytest.approx(12.0)      # 24 / 2.0
    assert agg.throughput_requests_per_s == pytest.approx(1.5)     # 3 / 2.0


def test_compute_aggregates_all_failed_has_no_percentiles() -> None:
    agg = compute_aggregates([_fail("a"), _fail("b")], duration_s=1.0)
    assert agg.num_successful == 0
    assert agg.num_failed == 2
    assert agg.ttft_p50_ms is None
    assert agg.total_output_tokens == 0
    assert agg.throughput_tokens_per_s == 0.0
