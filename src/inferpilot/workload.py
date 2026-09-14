"""Workload definition.

A workload describes *what requests to send*, independent of the model or engine
configuration. Milestone 1 keeps this deliberately minimal: a fixed number of
requests with a prompt/output length shape and an optional arrival pattern.

The exact workload shape is an unresolved project decision (see README); this
schema is intentionally small and additive-friendly rather than complete.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel


class WorkloadSpec(SchemaModel):
    """A reproducible synthetic workload.

    Lengths are expressed in tokens. ``request_rate_qps`` selects a deterministic
    open-loop schedule; ``None`` selects closed-loop execution with up to
    ``max_concurrency`` requests in flight.
    """

    name: str = Field(description="Human-readable workload label, e.g. 'chat-short'.")
    num_requests: int = Field(gt=0, description="Number of MEASURED requests to issue.")
    warmup_requests: int = Field(
        default=0,
        ge=0,
        description="Requests to issue before measurement; excluded from aggregates.",
    )

    prompt_tokens: int = Field(
        gt=0, description="Target prompt length in tokens (fixed-length workload)."
    )
    output_tokens: int = Field(
        gt=0, description="Number of generated tokens requested per request (max_tokens)."
    )

    request_rate_qps: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Requested open-loop arrival rate in requests/sec. None => closed-loop. "
            "The current runner rejects non-None values rather than silently "
            "executing a different arrival process."
        ),
    )
    arrival_pattern: Literal["poisson-v1", "batched-poisson-v1"] = Field(
        default="poisson-v1",
        description="Open-loop arrival algorithm; legacy schemas use poisson-v1.",
    )
    burst_size: Optional[int] = Field(
        default=None, ge=2,
        description="Requests per simultaneous batch under batched-poisson-v1.",
    )
    max_concurrency: Optional[int] = Field(
        default=None,
        gt=0,
        description="Max in-flight requests for closed-loop mode.",
    )

    # --- generation control (for stable, reproducible timing) -----------
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Sampling temperature; 0.0 => greedy/deterministic generation.",
    )
    ignore_eos: bool = Field(
        default=False,
        description=(
            "Force controlled-length generation: ignore EOS and require exactly "
            "output_tokens tokens (paired with min_tokens on the request) so a "
            "premature EOS does not shorten and invalidate timing."
        ),
    )

    seed: int = Field(
        default=0,
        description=(
            "Legacy RNG seed (schema 0.3.0 semantics): controls BOTH prompt "
            "generation and the arrival schedule. Under 0.4.0 it is the fallback "
            "for prompt_seed / arrival_seed when either is omitted."
        ),
    )
    prompt_seed: Optional[int] = Field(
        default=None,
        description=(
            "Prompt-generation seed (schema 0.4.0). None => fall back to `seed`. "
            "Must remain None under schema 0.3.0."
        ),
    )
    arrival_seed: Optional[int] = Field(
        default=None,
        description=(
            "Arrival-schedule (Poisson) seed (schema 0.4.0). None => fall back to "
            "`seed`. Must remain None under schema 0.3.0."
        ),
    )

    @property
    def effective_prompt_seed(self) -> int:
        """Seed used for prompt generation: prompt_seed if set, else legacy seed."""
        return self.prompt_seed if self.prompt_seed is not None else self.seed

    @property
    def effective_arrival_seed(self) -> int:
        """Seed used for the arrival schedule: arrival_seed if set, else legacy seed."""
        return self.arrival_seed if self.arrival_seed is not None else self.seed

    @model_validator(mode="after")
    def _check_arrival_pattern(self) -> "WorkloadSpec":
        if (self.request_rate_qps is None) == (self.max_concurrency is None):
            raise ValueError(
                "workload must define exactly one arrival pattern: set request_rate_qps "
                "for open-loop or max_concurrency for closed-loop, but not both."
            )
        if self.arrival_pattern == "poisson-v1" and self.burst_size is not None:
            raise ValueError("poisson-v1 must not define burst_size")
        if self.arrival_pattern == "batched-poisson-v1" and self.burst_size is None:
            raise ValueError("batched-poisson-v1 requires burst_size >= 2")
        if self.request_rate_qps is None and self.arrival_pattern != "poisson-v1":
            raise ValueError("batched-poisson-v1 is only valid for open-loop workloads")
        return self
