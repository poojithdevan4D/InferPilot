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

from pydantic import Field, model_validator

from ._base import SchemaModel, VersionedSchemaModel
from .config import ExperimentConfig
from .environment import EnvironmentMetadata
from .measurements import RequestMeasurement
from .status import ExperimentStatus, FailureRecord


class AggregateMetrics(SchemaModel):
    """Summary statistics derived from per-request measurements.

    Only meaningful for a completed run. Latency percentiles are in milliseconds;
    throughput in tokens (or requests) per second.

    Invariants (validated below):

    * ``num_requests == num_successful + num_failed``.
    * ``total_output_tokens`` and ``duration_s`` are non-negative (field-level).
    * **Zero-duration rule:** a zero ``duration_s`` is only valid for a degenerate
      empty aggregate (``num_requests == 0``). Any aggregate covering at least one
      request must have ``duration_s > 0`` — otherwise throughput is undefined and
      the numbers are not real measurements.
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

    @model_validator(mode="after")
    def _check_invariants(self) -> "AggregateMetrics":
        if self.num_requests != self.num_successful + self.num_failed:
            raise ValueError(
                f"num_requests ({self.num_requests}) must equal num_successful "
                f"({self.num_successful}) + num_failed ({self.num_failed})."
            )
        if self.num_requests > 0 and self.duration_s <= 0:
            raise ValueError(
                "duration_s must be > 0 for an aggregate covering >= 1 request; "
                "zero duration is only valid for an empty (num_requests == 0) aggregate."
            )
        return self


class ExperimentResult(VersionedSchemaModel):
    """Complete, serializable outcome of one experiment run.

    ``aggregates`` is ``None`` when the run did not complete (e.g. OOM); in that
    case ``failure`` explains why. ``measurements`` may still contain whatever was
    collected before the failure.

    Status invariants (validated below):

    * ``COMPLETED`` requires ``aggregates`` and must not carry a ``failure``.
    * ``FAILED`` / ``OOM`` / ``TIMEOUT`` require a ``failure`` (aggregates may still
      be present if partial results were computed).
    * When a ``failure`` is present, ``failure.status`` must equal ``status``.
    * ``PENDING`` / ``RUNNING`` are not terminal and must carry neither
      ``aggregates`` nor ``failure``.
    * If both timestamps are set, ``finished_at >= started_at``.

    **``COMPLETED`` means orchestration completed, not that the benchmark is
    valid.** A run whose server started and whose measured workload executed to
    the end is ``COMPLETED`` even if every request failed. Benchmark validity is a
    separate, stricter question answered by :pyattr:`is_baseline_eligible`.

    **Eligibility policy:** only a fully clean ``COMPLETED`` run is eligible to
    serve as a baseline or to drive optimization (see
    :pyattr:`is_baseline_eligible`). A failed run (``FAILED`` / ``OOM`` /
    ``TIMEOUT``) — or a ``COMPLETED`` run with any failed/missing measurements —
    may still *store* partial ``aggregates``, but those numbers must never be
    compared against a baseline or used to accept/reject an optimization.
    """

    config: ExperimentConfig
    environment: EnvironmentMetadata

    status: ExperimentStatus
    measurements: list[RequestMeasurement] = Field(default_factory=list)
    aggregates: Optional[AggregateMetrics] = Field(default=None)
    failure: Optional[FailureRecord] = Field(default=None)

    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)

    @property
    def is_baseline_eligible(self) -> bool:
        """True iff this result is a fully clean run usable as a baseline / signal.

        Requires ALL of:

        * ``status`` is ``COMPLETED``,
        * ``aggregates`` present,
        * every configured request was measured
          (``num_requests == config.workload.num_requests``),
        * every request succeeded (``num_successful == config.workload.num_requests``
          and ``num_failed == 0``),
        * the latency/throughput fields needed for comparison are all populated.

        Partial aggregates from failed or partially-failed runs are excluded.
        """
        if self.status is not ExperimentStatus.COMPLETED:
            return False
        agg = self.aggregates
        if agg is None:
            return False

        expected = self.config.workload.num_requests
        if agg.num_requests != expected:
            return False
        if agg.num_successful != expected or agg.num_failed != 0:
            return False

        required_metrics = (
            agg.ttft_p50_ms,
            agg.ttft_p95_ms,
            agg.ttft_p99_ms,
            agg.tpot_p50_ms,
            agg.tpot_p95_ms,
            agg.tpot_p99_ms,
            agg.e2e_p50_ms,
            agg.e2e_p95_ms,
            agg.e2e_p99_ms,
            agg.throughput_tokens_per_s,
            agg.throughput_requests_per_s,
        )
        return all(v is not None for v in required_metrics)

    @model_validator(mode="after")
    def _check_status_consistency(self) -> "ExperimentResult":
        status = self.status

        if self.failure is not None and self.failure.status != status:
            raise ValueError(
                f"failure.status ({self.failure.status.value}) must equal result "
                f"status ({status.value})."
            )

        if status is ExperimentStatus.COMPLETED:
            if self.aggregates is None:
                raise ValueError("a COMPLETED result must have aggregates.")
            if self.failure is not None:
                raise ValueError("a COMPLETED result must not carry a failure.")

        elif status in (
            ExperimentStatus.FAILED,
            ExperimentStatus.OOM,
            ExperimentStatus.TIMEOUT,
        ):
            if self.failure is None:
                raise ValueError(
                    f"a {status.value.upper()} result must carry a failure record."
                )

        elif status in (ExperimentStatus.PENDING, ExperimentStatus.RUNNING):
            if self.aggregates is not None:
                raise ValueError(
                    f"a {status.value.upper()} result must not carry aggregates."
                )
            if self.failure is not None:
                raise ValueError(
                    f"a {status.value.upper()} result must not carry a failure record."
                )

        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at must be >= started_at.")

        return self
