"""Environment, hardware, and toolchain metadata captured at run time.

A benchmark number is meaningless without its conditions. This module records
the machine, software stack, and CUDA toolchain an experiment actually ran on,
so results from the 4 GB laptop and a rented 24/80 GB GPU never get silently
compared — and so a toolchain mismatch (e.g. torch built for CUDA 13 but a local
``nvcc`` from CUDA 12.4) is visible in provenance rather than hidden.

There is deliberately **no single ambiguous ``cuda_version``**: the driver's
supported CUDA, PyTorch's compiled CUDA, and the local ``nvcc`` toolkit version
are distinct facts and are recorded separately.
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
    driver_version: Optional[str] = Field(default=None, description="NVIDIA driver version.")
    driver_cuda_version: Optional[str] = Field(
        default=None,
        description=(
            "Max CUDA version the installed driver supports, from the nvidia-smi "
            "banner ('CUDA Version: X.Y'). NOT the toolkit or torch CUDA version."
        ),
    )

    cpu_model: Optional[str] = Field(default=None)
    cpu_count: Optional[int] = Field(default=None, ge=0, description="Logical CPU count.")
    system_ram_mb: Optional[int] = Field(default=None, ge=0, description="Total system RAM in MiB.")


class ToolchainInfo(SchemaModel):
    """CUDA/build toolchain the run resolved — kept distinct from driver CUDA."""

    torch_cuda_version: Optional[str] = Field(
        default=None, description="CUDA version PyTorch was compiled against (torch.version.cuda)."
    )
    nvcc_version: Optional[str] = Field(
        default=None, description="Local nvcc toolkit version (from `nvcc --version`)."
    )
    nvcc_path: Optional[str] = Field(
        default=None, description="Resolved path of the nvcc executable, if any."
    )
    flashinfer_version: Optional[str] = Field(
        default=None, description="Installed FlashInfer version, if any."
    )


class EnvironmentMetadata(SchemaModel):
    """Full run-time environment: host, software stack, hardware, and toolchain."""

    hostname: Optional[str] = Field(default=None)
    platform: Optional[str] = Field(
        default=None, description="OS/platform string, e.g. 'Linux-6.8-x86_64'."
    )
    python_version: Optional[str] = Field(default=None, description="e.g. '3.11.9'.")
    torch_version: Optional[str] = Field(default=None)
    vllm_version: Optional[str] = Field(default=None, description="Serving-engine version.")

    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    toolchain: ToolchainInfo = Field(default_factory=ToolchainInfo)

    # Which sampler backend was requested, and the exact non-secret env overrides
    # applied to the server child process (e.g. {"VLLM_USE_FLASHINFER_SAMPLER": "0"}).
    effective_sampler_backend: Optional[str] = Field(default=None)
    runtime_overrides: dict[str, str] = Field(default_factory=dict)

    captured_at: Optional[datetime] = Field(
        default=None, description="When this metadata was captured (timezone-aware recommended)."
    )
