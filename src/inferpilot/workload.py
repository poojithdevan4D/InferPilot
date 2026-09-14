"""Workload definition.

A workload describes *what requests to send*, independent of the model or engine
configuration. Milestone 1 keeps this deliberately minimal: a fixed number of
requests with a prompt/output length shape and an optional arrival pattern.

The exact workload shape is an unresolved project decision (see README); this
schema is intentionally small and additive-friendly rather than complete.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from ._base import SchemaModel


class WorkloadSpec(SchemaModel):
    """A reproducible synthetic workload.

    Lengths are expressed in tokens. ``request_rate_qps`` selects the arrival
    pattern: ``None`` means closed-loop (send up to ``max_concurrency`` at once),
    a positive value means open-loop Poisson/uniform arrivals.
    """

    name: str = Field(description="Human-readable workload label, e.g. 'chat-short'.")
    num_requests: int = Field(gt=0, description="Total number of requests to issue.")

    prompt_tokens: int = Field(
        gt=0, description="Target prompt length in tokens (fixed-length workload)."
    )
    output_tokens: int = Field(
        gt=0, description="Target number of generated tokens per request."
    )

    request_rate_qps: Optional[float] = Field(
        default=None,
        gt=0,
        description="Open-loop arrival rate in requests/sec. None => closed-loop.",
    )
    max_concurrency: Optional[int] = Field(
        default=None,
        gt=0,
        description="Max in-flight requests for closed-loop mode.",
    )

    seed: int = Field(default=0, description="RNG seed for reproducible generation.")

    @model_validator(mode="after")
    def _check_arrival_pattern(self) -> "WorkloadSpec":
        if self.request_rate_qps is None and self.max_concurrency is None:
            raise ValueError(
                "workload must define an arrival pattern: set request_rate_qps "
                "(open-loop) or max_concurrency (closed-loop)."
            )
        return self
