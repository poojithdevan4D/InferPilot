"""Resolved (effective) engine configuration, as reported by the server.

InferPilot must not claim what ran based only on the *input* config. After the
server starts, the runner parses vLLM's resolved configuration from its startup
logs and records it here. A mismatch between an explicitly-requested value and
the resolved value fails the run before measurement (see the runner).

All fields are ``Optional`` because a value that could not be parsed is recorded
as ``None`` (unverified) rather than guessed.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ._base import SchemaModel


class EffectiveConfig(SchemaModel):
    """Engine configuration as resolved by vLLM at startup."""

    model: Optional[str] = Field(default=None)
    revision: Optional[str] = Field(default=None)
    dtype: Optional[str] = Field(default=None)
    max_model_len: Optional[int] = Field(default=None, ge=1)
    max_num_seqs: Optional[int] = Field(default=None, ge=1)
    max_num_batched_tokens: Optional[int] = Field(default=None, ge=1)
    enable_prefix_caching: Optional[bool] = Field(default=None)
    enable_chunked_prefill: Optional[bool] = Field(default=None)
    sampler_backend: Optional[str] = Field(default=None)
    kv_cache_dtype: Optional[str] = Field(default=None)
    gpu_memory_utilization: Optional[float] = Field(default=None, gt=0, le=1)
    generation_config: Optional[str] = Field(default=None)

    source: str = Field(
        default="startup_log",
        description="How the effective config was obtained (e.g. parsed startup log).",
    )
    unverified_fields: list[str] = Field(
        default_factory=list,
        description="Requested keys that could not be resolved/parsed, so were not checked.",
    )
    verified: bool = Field(
        default=False,
        description="True only after every explicitly requested fidelity field matched.",
    )
