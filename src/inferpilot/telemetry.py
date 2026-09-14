"""Resource-telemetry data contracts.

Minimal measured-window telemetry: GPU memory used + GPU utilization (NVML) and
KV-cache utilization (scraped from the server's Prometheus ``/metrics``). Raw
samples are stored in a separate immutable artifact; the derived summary
(:class:`ResourceTelemetry`) is embedded in the result.

Telemetry is best-effort: any collection failure is recorded in
:pyattr:`ResourceTelemetry.error` and leaves the affected fields ``None`` — it
never invalidates the request measurements.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ._base import SchemaModel


class ResourceSample(SchemaModel):
    """One telemetry sample taken during the measured window."""

    t_s: float = Field(ge=0, description="Monotonic seconds from the experiment origin (t0).")
    gpu_memory_used_mb: Optional[int] = Field(
        default=None, ge=0, description="GPU memory in use (MiB), from NVML."
    )
    gpu_utilization_pct: Optional[float] = Field(
        default=None, ge=0, le=100, description="GPU utilization percent, from NVML."
    )
    kv_cache_usage_perc: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="vllm:kv_cache_usage_perc gauge (fraction 0..1) from /metrics.",
    )


class ResourceTelemetry(SchemaModel):
    """Derived summary of the measured-window resource samples.

    Raw samples live in a separate ``telemetry.json`` artifact; this is the
    compact summary embedded in the result.
    """

    sample_interval_s: float = Field(gt=0, description="Nominal sampling interval.")
    num_samples: int = Field(ge=0, description="Number of samples actually collected.")

    peak_gpu_memory_mb: Optional[int] = Field(default=None, ge=0)
    gpu_utilization_mean_pct: Optional[float] = Field(default=None, ge=0, le=100)
    gpu_utilization_peak_pct: Optional[float] = Field(default=None, ge=0, le=100)
    kv_cache_usage_mean_perc: Optional[float] = Field(default=None, ge=0, le=1)
    kv_cache_usage_peak_perc: Optional[float] = Field(default=None, ge=0, le=1)

    error: Optional[str] = Field(
        default=None,
        description="Explicit record of any telemetry-collection failure(s); None if clean.",
    )
