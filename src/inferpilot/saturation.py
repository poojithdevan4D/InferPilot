"""Drain-robust "is the server keeping up?" signal, from the 2026-09-18 decode campaign.

The obvious feasibility test — achieved throughput (num_successful / duration) vs the
offered rate — is contaminated by the decode DRAIN TAIL when per-request generation is
long: the measurement window includes the time to finish the last-arriving requests, so
throughput under-reads even when the server never fell behind. (It works only when
per-request latency << the window, e.g. short outputs.)

The queue-theoretic truth for an open-loop arrival process: a server keeps up iff its
backlog does not grow over time, which shows up as TTFT that is STABLE across arrival
order rather than trending upward. This module measures exactly that — comparing the
mean TTFT of the late half of requests to the early half — independent of total window
duration, so it is unaffected by the drain tail.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from pydantic import Field, model_validator

from ._base import SchemaModel
from .measurements import RequestMeasurement


class SaturationReport(SchemaModel):
    """Whether TTFT grew across arrival order (queue building = not keeping up)."""

    saturation_version: Literal["0.1.0"] = "0.1.0"
    num_samples: int = Field(ge=0)
    early_ttft_ms: Optional[float] = None
    late_ttft_ms: Optional[float] = None
    growth_ratio: Optional[float] = Field(default=None, description="late_ttft / early_ttft")
    threshold: float = Field(gt=1.0)
    saturated: bool
    reason: str

    @model_validator(mode="after")
    def _check(self) -> "SaturationReport":
        if self.early_ttft_ms is None or self.late_ttft_ms is None:
            if self.saturated or self.growth_ratio is not None:
                raise ValueError("insufficient-sample report cannot be saturated")
            return self
        ratio = self.late_ttft_ms / self.early_ttft_ms if self.early_ttft_ms > 0 else float("inf")
        if abs((self.growth_ratio or -1) - ratio) > 1e-9:
            raise ValueError("growth_ratio inconsistent with early/late TTFT")
        if self.saturated != (ratio > self.threshold):
            raise ValueError("saturated flag inconsistent with growth_ratio and threshold")
        return self


def detect_saturation(
    measurements: Sequence[RequestMeasurement],
    *,
    threshold: float = 1.5,
    min_samples: int = 8,
) -> SaturationReport:
    """Report whether TTFT trends upward across arrival order (backlog growing).

    ``threshold`` is the late/early mean-TTFT ratio above which the run is judged to be
    saturating (default 1.5 = late half >50% slower than early half). Runs with fewer
    than ``min_samples`` successful TTFTs are reported not-saturated with no ratio.
    """
    ttfts = [
        m.ttft_ms
        for m in sorted(measurements, key=lambda m: m.start_time_s)
        if m.success and m.ttft_ms is not None
    ]
    if len(ttfts) < min_samples:
        return SaturationReport(
            num_samples=len(ttfts), threshold=threshold, saturated=False,
            reason="insufficient_samples",
        )
    half = len(ttfts) // 2
    early = sum(ttfts[:half]) / half
    late = sum(ttfts[half:]) / (len(ttfts) - half)
    ratio = late / early if early > 0 else float("inf")
    saturated = ratio > threshold
    return SaturationReport(
        num_samples=len(ttfts), early_ttft_ms=early, late_ttft_ms=late,
        growth_ratio=ratio, threshold=threshold, saturated=saturated,
        reason="ttft_growing_across_arrivals" if saturated else "ttft_stable",
    )
