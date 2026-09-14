"""Aggregate metrics and the top-level experiment result.

``ExperimentResult`` is the object the core loop stores and compares. It bundles
the config that was run, the environment it ran on, the raw per-request
measurements, the derived aggregates (absent on failure), and the status /
failure record.

Milestone 1 only defines these contracts; computing aggregates from
measurements is intentionally left to a later, tested implementation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from ._base import SCHEMA_VERSION, SchemaModel
from .config import ExperimentConfig
from .environment import EnvironmentMetadata
from .measurements import RequestMeasurement
from .status import ExperimentStatus, FailureRecord


class AggregateMetrics(SchemaModel):
    """Summary statistics derived from per-request measurements.

    Only meaningful for a completed run. Latency percentiles are in milliseconds;
    throughput in tokens (or requests) per second.
    """

    num_requests: int = Field(ge=0)
    num_successful: int = Field(ge=0)
    num_failed: int = Field(ge=0)

    duration_s: float = Field(ge=0, description="Wall-clock benchmark duration.")

    ttft_p50_ms: Optional[float] = Field(default=None, ge=0)
    ttft_p95_ms: Optional[float] = Field(default=None, ge=0)
    ttft_p99_ms: Optional[float] = Field(default=None, ge=0)

    tpot_p50_ms: Optional[float] = Field(default=None, ge=0)
    tpot_p95_ms: Optional[float] = Field(default=None, ge=0)
    tpot_p99_ms: Optional[float] = Field(default=None, ge=0)

    e2e_p50_ms: Optional[float] = Field(default=None, ge=0)
    e2e_p95_ms: Optional[float] = Field(default=None, ge=0)
    e2e_p99_ms: Optional[float] = Field(default=None, ge=0)

    throughput_tokens_per_s: Optional[float] = Field(default=None, ge=0)
    throughput_requests_per_s: Optional[float] = Field(default=None, ge=0)

    total_output_tokens: int = Field(default=0, ge=0)
    gpu_memory_peak_mb: Optional[int] = Field(default=None, ge=0)


class ExperimentResult(SchemaModel):
    """Complete, serializable outcome of one experiment run.

    ``aggregates`` is ``None`` when the run did not complete (e.g. OOM); in that
    case ``failure`` explains why. ``measurements`` may still contain whatever was
    collected before the failure.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)

    config: ExperimentConfig
    environment: EnvironmentMetadata

    status: ExperimentStatus
    measurements: list[RequestMeasurement] = Field(default_factory=list)
    aggregates: Optional[AggregateMetrics] = Field(default=None)
    failure: Optional[FailureRecord] = Field(default=None)

    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)
