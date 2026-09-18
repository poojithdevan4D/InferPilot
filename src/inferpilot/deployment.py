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

# Indicative on-demand hourly USD (order-of-magnitude; operator can override). Cost
# ranking is only as good as these — they are inputs to verify, not ground truth.
GPU_HOURLY_USD: dict[str, float] = {
    "L4": 0.80, "A10G": 1.10, "A10": 1.10, "L40S": 1.90,
    "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.90, "H200": 4.50,
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


def fetch_footprint(model: str, params_billions: float, *, revision: str = "main") -> ModelFootprint:
    """Fetch a model's HF config.json and build its ModelFootprint (network call).

    params_billions is supplied by the caller (config.json rarely states it). Raises on
    network/parse failure so the caller can fall back to an explicit footprint."""
    import json
    import urllib.request

    url = f"https://huggingface.co/{model}/raw/{revision}/config.json"
    with urllib.request.urlopen(url, timeout=10.0) as resp:  # noqa: S310
        config = json.loads(resp.read().decode("utf-8"))
    return footprint_from_hf_config(model, params_billions, config)


def footprint_from_hf_config(model: str, params_billions: float, config: dict) -> ModelFootprint:
    """Build a ModelFootprint from a model's HF config.json dict (+ known param count).

    head_dim falls back to hidden_size / num_attention_heads when absent (Qwen-style)."""
    layers = config["num_hidden_layers"]
    kv_heads = config.get("num_key_value_heads") or config["num_attention_heads"]
    head_dim = config.get("head_dim") or (config["hidden_size"] // config["num_attention_heads"])
    dtype = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
        config.get("torch_dtype", "bfloat16"), "bf16"
    )
    return ModelFootprint(
        model=model, params_billions=params_billions, num_layers=layers,
        num_kv_heads=kv_heads, head_dim=head_dim, weight_dtype=dtype,
    )


class DeploymentOption(SchemaModel):
    """A fitting deployment shape + its hourly cost, for ranking. Embeds a self-validating
    FitAnalysis; throughput-under-SLO still requires canary verification (fit ⇒ can HOLD the
    working set, not that it MEETS latency)."""

    fit: FitAnalysis
    hourly_usd: float = Field(gt=0)


def recommend_deployment(
    footprint: ModelFootprint,
    *,
    context_tokens: int,
    required_concurrency: int,
    candidate_gpus: Optional[list[str]] = None,
    kv_dtypes: tuple[str, ...] = ("bf16", "fp8"),
    tp_options: tuple[int, ...] = (1, 2),
    hourly_usd: Optional[dict[str, float]] = None,
    gpu_memory_utilization: float = 0.90,
) -> list[DeploymentOption]:
    """Rank the deployment shapes that FIT the working set, cheapest first.

    Enumerates (GPU × TP × kv_dtype), keeps those whose KV budget holds
    ``required_concurrency`` requests at ``context_tokens``, and orders by hourly cost
    (gpu $/hr × TP). Answers "what is the cheapest hardware that can hold this workload" —
    the structural recommendation. Decode-throughput must still be canary-verified."""
    gpus = candidate_gpus if candidate_gpus is not None else list(GPU_VRAM_GB)
    prices = hourly_usd or GPU_HOURLY_USD
    options: list[DeploymentOption] = []
    for gpu in gpus:
        if gpu not in GPU_VRAM_GB or gpu not in prices:
            continue
        for tp in tp_options:
            for kv in kv_dtypes:
                fit = analyze_fit(
                    footprint, gpu, context_tokens=context_tokens, tensor_parallel=tp,
                    kv_dtype=kv, gpu_memory_utilization=gpu_memory_utilization,
                )
                if fit.fits and fit.max_concurrent_requests >= required_concurrency:
                    options.append(DeploymentOption(fit=fit, hourly_usd=prices[gpu] * tp))
    return sorted(options, key=lambda o: (o.hourly_usd, o.fit.tensor_parallel))


class ScaleRecommendation(SchemaModel):
    """Self-validating structural recommendation for a scale-out decision."""

    footprint: ModelFootprint
    context_tokens: int = Field(gt=0)
    required_concurrency: int = Field(ge=1)
    candidate_gpus: Optional[list[str]] = None
    options: list[DeploymentOption]
    cheapest: Optional[DeploymentOption] = None

    @model_validator(mode="after")
    def _check(self) -> "ScaleRecommendation":
        expected = recommend_deployment(
            self.footprint, context_tokens=self.context_tokens,
            required_concurrency=self.required_concurrency, candidate_gpus=self.candidate_gpus,
        )
        if self.options != expected or self.cheapest != (expected[0] if expected else None):
            raise ValueError("scale recommendation is inconsistent with the fit physics")
        return self


def recommend_scale(
    result: "ExperimentResult", footprint: ModelFootprint, target_qps: float,
    *, candidate_gpus: Optional[list[str]] = None,
) -> ScaleRecommendation:
    """From a measured baseline + a target QPS, recommend the cheapest fitting deployment.

    Required concurrency uses the queue-robust decode service time (output_tokens x tpot_p50),
    not e2e (which an overloaded baseline inflates with queue wait): Little's law on the
    compute portion. context = prompt + output tokens."""
    import math

    from .results import ExperimentResult  # noqa: F401 (typing only)

    w, a = result.config.workload, result.aggregates
    tpot_s = (a.tpot_p50_ms or a.tpot_p95_ms or 0.0) / 1000.0
    decode_service_s = w.output_tokens * tpot_s
    required = max(1, math.ceil(target_qps * decode_service_s))
    context = w.prompt_tokens + w.output_tokens
    options = recommend_deployment(
        footprint, context_tokens=context, required_concurrency=required,
        candidate_gpus=candidate_gpus,
    )
    return ScaleRecommendation(
        footprint=footprint, context_tokens=context, required_concurrency=required,
        candidate_gpus=candidate_gpus, options=options,
        cheapest=options[0] if options else None,
    )


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
