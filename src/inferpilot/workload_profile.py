"""Passive workload observation + derived profile contracts (M5).

Factual, monotonic-only characterization of observed request traffic. This is
passive **characterization, not classification**: no field here labels traffic
"Poisson", "burst", or any other workload class — only descriptive statistics.

All timing is monotonic (offsets supplied by the caller); no wall-clock is read.
The :class:`WorkloadProfile` embeds the ordered observations and recomputes every
derived field (and the provenance digest) on load, so any tampering is rejected.

Percentiles use explicit **nearest-rank** semantics (the returned value is an
actual observed value), which is appropriate for integer token counts and raw
inter-arrival gaps. Insufficient evidence is represented as ``None`` and is
distinct from a valid ``0`` value.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel

WORKLOAD_PROFILE_VERSION = "0.1.0"


def nearest_rank(sorted_values: list, q: float):
    """Nearest-rank percentile: value at rank ceil(q/100 * n), clamped to [1, n]."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("nearest_rank of empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    index = max(0, min(n - 1, math.ceil(q / 100.0 * n) - 1))
    return sorted_values[index]


class WorkloadObservation(SchemaModel):
    """One passively observed request. Factual data only — no outcome metrics."""

    arrival_offset_s: float = Field(ge=0, description="Monotonic arrival offset from window origin.")
    prompt_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    success: bool
    completion_offset_s: Optional[float] = Field(
        default=None, ge=0, description="Monotonic completion offset; must be >= arrival if present."
    )

    @model_validator(mode="after")
    def _check(self) -> "WorkloadObservation":
        if self.completion_offset_s is not None and self.completion_offset_s < self.arrival_offset_s:
            raise ValueError("completion_offset_s cannot precede arrival_offset_s")
        return self


def _digest(observations: list[WorkloadObservation]) -> str:
    payload = [o.model_dump(mode="json") for o in observations]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _derive(observations: list[WorkloadObservation]) -> dict:
    """Compute all derived fields from ordered observations (single source of truth)."""
    offsets = [o.arrival_offset_s for o in observations]
    if any(b < a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("arrival offsets must be non-decreasing (monotonic)")

    count = len(observations)
    successful = sum(1 for o in observations if o.success)
    window = (offsets[-1] - offsets[0]) if count >= 1 else 0.0

    # Insufficient evidence => None; a valid measured 0 stays 0.
    realized_rate = (count - 1) / window if (count >= 2 and window > 0) else None

    prompts = sorted(o.prompt_tokens for o in observations)
    outputs = sorted(o.output_tokens for o in observations)
    prompt_p50 = nearest_rank(prompts, 50) if count else None
    prompt_p95 = nearest_rank(prompts, 95) if count else None
    output_p50 = nearest_rank(outputs, 50) if count else None
    output_p95 = nearest_rank(outputs, 95) if count else None

    gaps = [b - a for a, b in zip(offsets, offsets[1:])]  # count-1 gaps
    if gaps:
        sorted_gaps = sorted(gaps)
        inter_p50 = nearest_rank(sorted_gaps, 50)
        inter_p95 = nearest_rank(sorted_gaps, 95)
        simultaneous_fraction = sum(1 for g in gaps if g == 0) / len(gaps)
        positive = [g for g in gaps if g > 0]
        # CV of positive inter-arrivals: undefined with no positive gaps.
        cv = statistics.pstdev(positive) / statistics.mean(positive) if positive else None
    else:
        inter_p50 = inter_p95 = simultaneous_fraction = cv = None

    return {
        "observation_count": count,
        "successful_count": successful,
        "window_duration_s": window,
        "realized_request_rate_qps": realized_rate,
        "prompt_tokens_p50": prompt_p50,
        "prompt_tokens_p95": prompt_p95,
        "output_tokens_p50": output_p50,
        "output_tokens_p95": output_p95,
        "interarrival_p50_s": inter_p50,
        "interarrival_p95_s": inter_p95,
        "interarrival_cv": cv,
        "simultaneous_arrival_fraction": simultaneous_fraction,
        "provenance_sha256": _digest(observations),
    }


class WorkloadProfile(SchemaModel):
    """Self-validating passive workload profile derived from ordered observations."""

    profile_version: str = WORKLOAD_PROFILE_VERSION
    observations: list[WorkloadObservation]

    observation_count: int = Field(ge=0)
    successful_count: int = Field(ge=0)
    window_duration_s: float = Field(ge=0)
    realized_request_rate_qps: Optional[float] = Field(default=None, ge=0)
    prompt_tokens_p50: Optional[int] = Field(default=None, ge=0)
    prompt_tokens_p95: Optional[int] = Field(default=None, ge=0)
    output_tokens_p50: Optional[int] = Field(default=None, ge=0)
    output_tokens_p95: Optional[int] = Field(default=None, ge=0)
    interarrival_p50_s: Optional[float] = Field(default=None, ge=0)
    interarrival_p95_s: Optional[float] = Field(default=None, ge=0)
    interarrival_cv: Optional[float] = Field(default=None, ge=0)
    simultaneous_arrival_fraction: Optional[float] = Field(default=None, ge=0, le=1)
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check(self) -> "WorkloadProfile":
        if self.profile_version != WORKLOAD_PROFILE_VERSION:
            raise ValueError(
                f"unsupported workload profile_version {self.profile_version!r}; "
                f"expected {WORKLOAD_PROFILE_VERSION!r}"
            )
        expected = _derive(self.observations)
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"derived field {field!r} is inconsistent with observations")
        return self


def build_workload_profile(observations: list[WorkloadObservation]) -> WorkloadProfile:
    """Derive a self-validating profile from ordered observations."""
    return WorkloadProfile(observations=list(observations), **_derive(observations))
