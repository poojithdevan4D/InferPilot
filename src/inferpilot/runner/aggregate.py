"""Aggregation of per-request measurements into :class:`AggregateMetrics`.

Pure functions over measurements — no I/O, no clocks. Latency percentiles and
throughput are computed from **successful** requests only; failures still count
toward ``num_requests`` / ``num_failed``. ``duration_s`` is the measured window
(monotonic) supplied by the caller — this module never reads a clock.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..measurements import RequestMeasurement
from ..results import AggregateMetrics


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (matches numpy's default 'linear' method).

    ``q`` is in [0, 100]. ``values`` must be non-empty.
    """
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def _p(values: list[float], q: float) -> Optional[float]:
    return percentile(values, q) if values else None


def compute_aggregates(
    measurements: Sequence[RequestMeasurement], duration_s: float
) -> AggregateMetrics:
    """Aggregate measured requests over a measurement window of ``duration_s``."""
    successful = [m for m in measurements if m.success]
    num_requests = len(measurements)
    num_successful = len(successful)
    num_failed = num_requests - num_successful

    ttfts = [m.ttft_ms for m in successful if m.ttft_ms is not None]
    tpots = [m.tpot_ms for m in successful if m.tpot_ms is not None]
    e2es = [m.e2e_latency_ms for m in successful if m.e2e_latency_ms is not None]

    total_output_tokens = sum(m.output_tokens for m in successful)

    throughput_tokens = None
    throughput_requests = None
    if duration_s > 0:
        throughput_tokens = total_output_tokens / duration_s
        throughput_requests = num_successful / duration_s

    return AggregateMetrics(
        num_requests=num_requests,
        num_successful=num_successful,
        num_failed=num_failed,
        duration_s=duration_s,
        ttft_p50_ms=_p(ttfts, 50),
        ttft_p95_ms=_p(ttfts, 95),
        ttft_p99_ms=_p(ttfts, 99),
        tpot_p50_ms=_p(tpots, 50),
        tpot_p95_ms=_p(tpots, 95),
        tpot_p99_ms=_p(tpots, 99),
        e2e_p50_ms=_p(e2es, 50),
        e2e_p95_ms=_p(e2es, 95),
        e2e_p99_ms=_p(e2es, 99),
        throughput_tokens_per_s=throughput_tokens,
        throughput_requests_per_s=throughput_requests,
        total_output_tokens=total_output_tokens,
    )
