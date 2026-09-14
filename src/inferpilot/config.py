"""Experiment configuration: the *input* to a single experiment run.

An ``ExperimentConfig`` fully specifies one reproducible experiment: which model,
which engine knobs, which workload, and (optionally) which SLO the result is
judged against. It contains no measurements — those live in ``results``.

The engine knobs mirror the small MVP subset from the project vision. They are
recorded as declarative configuration only; nothing here starts a server.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel, VersionedSchemaModel

SamplerBackend = Literal["auto", "flashinfer", "pytorch"]
GenerationConfigSource = Literal["auto", "vllm"]


class EngineConfig(SchemaModel):
    """vLLM-style engine knobs — the initial optimization search space.

    ``model`` is required and intentionally left as a free-form identifier so the
    operator picks a model that fits the target hardware (see README: model choice
    is unresolved). Nothing here validates that the model fits in VRAM.
    """

    model: str = Field(
        description="Model identifier (e.g. HF repo id or local path). Operator-chosen.",
    )
    revision: Optional[str] = Field(
        default=None,
        description=(
            "Exact model revision (HF commit sha / branch / tag) to pin. Recorded "
            "in the result so a run is reproducible against a specific weights snapshot."
        ),
    )
    dtype: str = Field(default="auto", description="Weight/compute dtype, e.g. 'auto', 'float16'.")
    max_model_len: Optional[int] = Field(
        default=None, gt=0, description="Max sequence length; None => engine default."
    )

    # --- MVP tunables ---------------------------------------------------
    max_num_seqs: Optional[int] = Field(
        default=None, gt=0, description="Max sequences per iteration (batch width)."
    )
    max_num_batched_tokens: Optional[int] = Field(
        default=None, gt=0, description="Max batched tokens per iteration."
    )
    gpu_memory_utilization: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="Fraction of GPU memory the engine may reserve.",
    )
    kv_cache_dtype: str = Field(
        default="auto", description="KV-cache dtype, e.g. 'auto', 'fp8'."
    )
    enable_prefix_caching: Optional[bool] = Field(
        default=None,
        description=(
            "Reuse shared prompt prefixes across requests. True => "
            "--enable-prefix-caching, False => --no-enable-prefix-caching, "
            "None => omit (engine default). None is NOT the same as False."
        ),
    )
    enable_chunked_prefill: Optional[bool] = Field(
        default=None,
        description=(
            "Split large prefills into chunks. True => --enable-chunked-prefill, "
            "False => --no-enable-chunked-prefill, None => omit (engine default)."
        ),
    )

    sampler_backend: SamplerBackend = Field(
        default="auto",
        description=(
            "Which sampler vLLM should use. 'pytorch' sets child-process "
            "VLLM_USE_FLASHINFER_SAMPLER=0 (native sampler, no FlashInfer JIT); "
            "'flashinfer' sets it to 1; 'auto' leaves the variable unset (engine default)."
        ),
    )
    generation_config: GenerationConfigSource = Field(
        default="vllm",
        description=(
            "Generation-default source passed to vLLM. 'vllm' prevents a model's "
            "generation_config.json from silently changing sampling parameters; "
            "'auto' accepts model-provided defaults."
        ),
    )

    extra_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Forward-compatible passthrough for engine args not yet modelled.",
    )


class SLO(SchemaModel):
    """Service-level objective a result is judged against.

    All fields optional: the concrete SLO is an unresolved project decision, and
    an experiment may be run purely to characterize behaviour with no SLO.
    """

    ttft_p95_ms: Optional[float] = Field(
        default=None, gt=0, description="Max acceptable p95 time-to-first-token (ms)."
    )
    tpot_p95_ms: Optional[float] = Field(
        default=None, gt=0, description="Max acceptable p95 time-per-output-token (ms)."
    )
    e2e_p95_ms: Optional[float] = Field(
        default=None, gt=0, description="Max acceptable p95 end-to-end latency (ms)."
    )
    min_throughput_tokens_per_s: Optional[float] = Field(
        default=None, gt=0, description="Minimum acceptable output-token throughput."
    )


class ExperimentConfig(VersionedSchemaModel):
    """Complete, serializable specification of one experiment."""

    experiment_id: str = Field(description="Stable unique id for this experiment.")
    name: str = Field(description="Short human-readable name.")
    description: Optional[str] = Field(default=None)

    engine: EngineConfig
    workload: "WorkloadSpec"  # forward ref, resolved below
    slo: Optional[SLO] = Field(default=None)

    seed: int = Field(default=0, description="Top-level seed for reproducibility.")
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_split_seed_schema(self) -> "ExperimentConfig":
        # Split prompt/arrival seeds are a 0.4.0 feature; a 0.3.0 config must use
        # only the single legacy seed so its meaning is unambiguous.
        if self.schema_version == "0.3.0" and (
            self.workload.prompt_seed is not None or self.workload.arrival_seed is not None
        ):
            raise ValueError(
                "prompt_seed/arrival_seed require schema_version 0.4.0; "
                "schema 0.3.0 uses the single legacy seed"
            )
        return self


# Resolve the forward reference to WorkloadSpec without a circular import cost.
from .workload import WorkloadSpec  # noqa: E402

ExperimentConfig.model_rebuild()
