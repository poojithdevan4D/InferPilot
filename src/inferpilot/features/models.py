"""Contracts for online-safe arrival evidence and derived workload features."""

from __future__ import annotations

import math
import statistics
from typing import Literal, Optional

from pydantic import Field, model_validator

from .._base import SchemaModel
from ..runner.aggregate import percentile


ARRIVAL_FEATURE_REPORT_VERSION = "0.1.0"
WINDOW_SECONDS = (0.25, 0.5, 1.0, 2.0)
WINDOW_KEYS = tuple(str(value) for value in WINDOW_SECONDS)


def max_arrivals_in_window(offsets: list[float], window_s: float) -> int:
    """Maximum arrivals in any closed interval ``[t, t + window_s]``."""
    best = 0
    left = 0
    for right, value in enumerate(offsets):
        while value - offsets[left] > window_s:
            left += 1
        best = max(best, right - left + 1)
    return best


class ArrivalEvidence(SchemaModel):
    """Validated shape of the runner's immutable ``arrivals.json`` artifact."""

    algorithm: Literal["poisson-v1"]
    request_rate_qps: float = Field(gt=0, allow_inf_nan=False)
    seed: int  # the seed that produced these offsets (= effective arrival seed)
    arrival_seed: Optional[int] = Field(
        default=None,
        description="Explicit arrival seed (0.4.0 artifacts). Absent on 0.3.0 artifacts.",
    )
    scheduled_offsets_s: list[float]
    actual_dispatch_offsets_s: list[float]

    @property
    def effective_arrival_seed(self) -> int:
        """The arrival seed this artifact was produced with.

        Prefers the explicit ``arrival_seed`` (0.4.0); falls back to ``seed`` for
        older artifacts, which never carried a separate arrival seed. This does
        not reinterpret old artifacts — their ``seed`` WAS the arrival seed.
        """
        return self.arrival_seed if self.arrival_seed is not None else self.seed

    @model_validator(mode="after")
    def _check_offsets(self) -> "ArrivalEvidence":
        # New artifacts intentionally duplicate the effective arrival seed in
        # `seed` (the historical field) and `arrival_seed` (the explicit 0.4.0
        # field). Reject contradictory provenance instead of trusting either.
        if self.arrival_seed is not None and self.arrival_seed != self.seed:
            raise ValueError("seed and arrival_seed must agree when both are present")
        if len(self.scheduled_offsets_s) != len(self.actual_dispatch_offsets_s):
            raise ValueError("scheduled and actual arrival offsets must have equal length")
        for name, offsets in (
            ("scheduled", self.scheduled_offsets_s),
            ("actual", self.actual_dispatch_offsets_s),
        ):
            if any(not math.isfinite(value) or value < 0 for value in offsets):
                raise ValueError(f"{name} arrival offsets must be finite and non-negative")
            if any(right < left for left, right in zip(offsets, offsets[1:])):
                raise ValueError(f"{name} arrival offsets must be monotonic")
        return self


class ArrivalFeatureReport(SchemaModel):
    """Self-validating feature prefix safe to expose to a search policy.

    Only dispatch timestamps at or before ``observation_cutoff_s`` are retained.
    No completion, latency, SLO, or unrevealed future-arrival value is present.
    """

    report_version: str = ARRIVAL_FEATURE_REPORT_VERSION
    source_experiment_id: str
    source_exact_fingerprint: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    observation_cutoff_s: float = Field(gt=0, allow_inf_nan=False)
    declared_request_rate_qps: float = Field(gt=0, allow_inf_nan=False)
    declared_prompt_tokens: int = Field(gt=0)
    declared_output_tokens: int = Field(gt=0)
    observed_dispatch_offsets_s: list[float]

    num_arrivals: int = Field(ge=0)
    arrival_count_rate_qps: float = Field(ge=0)
    interarrival_mean_s: Optional[float] = Field(default=None, ge=0)
    interarrival_p50_s: Optional[float] = Field(default=None, ge=0)
    interarrival_p95_s: Optional[float] = Field(default=None, ge=0)
    interarrival_cv: Optional[float] = Field(default=None, ge=0)
    interarrival_rate_qps: Optional[float] = Field(default=None, gt=0)
    max_arrivals_by_window_s: dict[str, int]

    @model_validator(mode="after")
    def _check_report(self) -> "ArrivalFeatureReport":
        if self.report_version != ARRIVAL_FEATURE_REPORT_VERSION:
            raise ValueError(
                f"unsupported arrival feature report_version {self.report_version!r}; "
                f"expected {ARRIVAL_FEATURE_REPORT_VERSION!r}"
            )
        offsets = self.observed_dispatch_offsets_s
        if any(
            not math.isfinite(value)
            or value < 0
            or value > self.observation_cutoff_s
            for value in offsets
        ):
            raise ValueError(
                "observed offsets must be finite and fall within the observation cutoff"
            )
        if any(right < left for left, right in zip(offsets, offsets[1:])):
            raise ValueError("observed dispatch offsets must be monotonic")
        if self.num_arrivals != len(offsets):
            raise ValueError("num_arrivals must match observed dispatch offsets")
        expected_count_rate = len(offsets) / self.observation_cutoff_s
        if self.arrival_count_rate_qps != expected_count_rate:
            raise ValueError("arrival_count_rate_qps is inconsistent with the feature window")

        intervals = [right - left for left, right in zip(offsets, offsets[1:])]
        if intervals:
            mean = statistics.mean(intervals)
            expected = (
                mean,
                percentile(intervals, 50),
                percentile(intervals, 95),
                statistics.pstdev(intervals) / mean if mean > 0 else None,
                1.0 / mean if mean > 0 else None,
            )
        else:
            expected = (None, None, None, None, None)
        actual = (
            self.interarrival_mean_s,
            self.interarrival_p50_s,
            self.interarrival_p95_s,
            self.interarrival_cv,
            self.interarrival_rate_qps,
        )
        if actual != expected:
            raise ValueError("inter-arrival features are inconsistent with observed offsets")

        expected_windows = {
            key: max_arrivals_in_window(offsets, window)
            for key, window in zip(WINDOW_KEYS, WINDOW_SECONDS)
        }
        if self.max_arrivals_by_window_s != expected_windows:
            raise ValueError("window burst features are inconsistent with observed offsets")
        return self
