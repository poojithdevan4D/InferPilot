"""Deployment-fit physics: will this model fit this GPU, and how much KV headroom?

The "will it fit / which GPU" engine behind hardware right-sizing — the repositioned
product's core (see repositioning roadmap). Pure, deterministic arithmetic from model
architecture + GPU VRAM + workload context, so the advisory can say "switch to 1x
A100-40GB with fp8 KV" instead of only "add GPUs".

KV bytes/token = 2 (K,V) x layers x kv_heads x head_dim x dtype_bytes.
weights bytes    = params x weight_dtype_bytes.
KV budget        = vram x gpu_memory_utilization - weights - activation_overhead.
max concurrency  = floor(KV budget / (context_tokens x kv_bytes_per_token)).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel

GB = 1_000_000_000

# Minimal GPU VRAM registry (GB). Extend as needed; validated fields are what matter.
GPU_VRAM_GB: dict[str, float] = {
    "A10G": 24.0, "A10": 24.0, "L4": 24.0, "L40S": 48.0,
    "A100-40GB": 40.0, "A100-80GB": 80.0, "H100": 80.0, "H200": 141.0,
}

_DTYPE_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "fp32": 4.0}


class ModelFootprint(SchemaModel):
    """Architecture-derived memory footprint (from a model's HF config)."""

    model: str
    params_billions: float = Field(gt=0)
    num_layers: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    weight_dtype: Literal["fp16", "bf16", "fp8", "int8", "fp32"] = "bf16"

    @property
    def weights_gb(self) -> float:
        return self.params_billions * _DTYPE_BYTES[self.weight_dtype]  # params(B) x bytes = GB

    def kv_bytes_per_token(self, kv_dtype: str = "bf16") -> float:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * _DTYPE_BYTES[kv_dtype]


class FitAnalysis(SchemaModel):
    """Self-validating fit verdict for (model, GPU, precision, workload context)."""

    footprint: ModelFootprint
    gpu_name: str
    gpu_vram_gb: float = Field(gt=0)
    tensor_parallel: int = Field(default=1, ge=1)
    kv_dtype: Literal["fp16", "bf16", "fp8", "int8", "fp32"] = "bf16"
    gpu_memory_utilization: float = Field(default=0.90, gt=0, le=1)
    activation_overhead_gb: float = Field(default=2.0, ge=0)
    context_tokens: int = Field(gt=0, description="prompt + output tokens per request")

    fits: bool
    weights_gb_per_gpu: float
    kv_budget_gb: float
    max_concurrent_requests: int

    @model_validator(mode="after")
    def _check(self) -> "FitAnalysis":
        weights = self.footprint.weights_gb / self.tensor_parallel
        usable = self.gpu_vram_gb * self.tensor_parallel * self.gpu_memory_utilization
        kv_budget = usable - self.footprint.weights_gb - self.activation_overhead_gb * self.tensor_parallel
        fits = kv_budget > 0
        kv_per_req = self.footprint.kv_bytes_per_token(self.kv_dtype) * self.context_tokens / GB
        max_conc = int(kv_budget / kv_per_req) if fits and kv_per_req > 0 else 0
        if (
            round(self.weights_gb_per_gpu, 6) != round(weights, 6)
            or round(self.kv_budget_gb, 6) != round(kv_budget, 6)
            or self.fits != fits
            or self.max_concurrent_requests != max_conc
        ):
            raise ValueError("fit analysis is inconsistent with model/GPU/context arithmetic")
        return self


def analyze_fit(
    footprint: ModelFootprint, gpu_name: str, *, context_tokens: int,
    tensor_parallel: int = 1, kv_dtype: str = "bf16",
    gpu_memory_utilization: float = 0.90, activation_overhead_gb: float = 2.0,
    gpu_vram_gb: Optional[float] = None,
) -> FitAnalysis:
    """Compute whether the model fits and the max concurrent requests at this context."""
    vram = gpu_vram_gb if gpu_vram_gb is not None else GPU_VRAM_GB.get(gpu_name)
    if vram is None:
        raise ValueError(f"unknown GPU {gpu_name!r}; pass gpu_vram_gb explicitly")
    weights = footprint.weights_gb / tensor_parallel
    usable = vram * tensor_parallel * gpu_memory_utilization
    kv_budget = usable - footprint.weights_gb - activation_overhead_gb * tensor_parallel
    fits = kv_budget > 0
    kv_per_req = footprint.kv_bytes_per_token(kv_dtype) * context_tokens / GB
    max_conc = int(kv_budget / kv_per_req) if fits and kv_per_req > 0 else 0
    return FitAnalysis(
        footprint=footprint, gpu_name=gpu_name, gpu_vram_gb=vram, tensor_parallel=tensor_parallel,
        kv_dtype=kv_dtype, gpu_memory_utilization=gpu_memory_utilization,
        activation_overhead_gb=activation_overhead_gb, context_tokens=context_tokens,
        fits=fits, weights_gb_per_gpu=weights, kv_budget_gb=kv_budget,
        max_concurrent_requests=max_conc,
    )
