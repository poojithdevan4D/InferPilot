"""Environment and hardware metadata captured at run time.

A benchmark number is meaningless without its conditions. This module records
the machine and software stack an experiment actually ran on, so results from
the 4 GB laptop and a rented 24/80 GB GPU never get silently compared.

Milestone 1 only *defines the schema*. Automatic hardware discovery (populating
these fields from the live system) is deliberately out of scope.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from ._base import SchemaModel


class HardwareInfo(SchemaModel):
    """Physical/accelerator description of a machine."""

    gpu_name: Optional[str] = Field(
        default=None, description="Primary GPU name, e.g. 'NVIDIA RTX 3050 Laptop GPU'."
    )
    gpu_count: int = Field(default=0, ge=0, description="Number of visible GPUs.")
    gpu_memory_total_mb: Optional[int] = Field(
        default=None, ge=0, description="Total VRAM of the primary GPU in MiB."
    )
    cuda_version: Optional[str] = Field(default=None, description="CUDA toolkit/runtime version.")
    driver_version: Optional[str] = Field(default=None, description="GPU driver version.")

    cpu_model: Optional[str] = Field(default=None)
    cpu_count: Optional[int] = Field(default=None, ge=0, description="Logical CPU count.")
    system_ram_mb: Optional[int] = Field(default=None, ge=0, description="Total system RAM in MiB.")


class EnvironmentMetadata(SchemaModel):
    """Full run-time environment: host, software stack, and hardware."""

    hostname: Optional[str] = Field(default=None)
    platform: Optional[str] = Field(
        default=None, description="OS/platform string, e.g. 'Linux-6.8-x86_64'."
    )
    python_version: Optional[str] = Field(default=None, description="e.g. '3.11.9'.")
    torch_version: Optional[str] = Field(default=None)
    vllm_version: Optional[str] = Field(
        default=None, description="Serving-engine version (unresolved which to pin — see README)."
    )

    hardware: HardwareInfo = Field(default_factory=HardwareInfo)

    captured_at: Optional[datetime] = Field(
        default=None, description="When this metadata was captured (timezone-aware recommended)."
    )
