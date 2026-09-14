"""Per-request measurements: the raw ground-truth signal.

One ``RequestMeasurement`` per issued request. Aggregates (percentiles,
throughput) are derived from these in ``results``; the raw records are always
retained so that any aggregate can be recomputed and audited later.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from ._base import SchemaModel


class RequestMeasurement(SchemaModel):
    """Timing and token counts for a single request.

    Presence rules (validated below):

    * ``end_time_s >= start_time_s`` always.
    * A **successful** request carries no ``error`` and must have produced at
      least one token: ``output_tokens >= 1``, with ``ttft_ms`` and
      ``e2e_latency_ms`` present. It is a real, timed completion.
    * A **failed** request must carry an ``error``. Its timing fields are
      best-effort and may be absent (the request may have died before the first
      token or mid-stream), so they are permitted but not required.
    * ``tpot_ms`` (mean time *between* output tokens, i.e. after the first) is
      only defined when there are at least two output tokens. It must therefore
      be absent whenever ``output_tokens < 2`` (in particular, one-token
      outputs have no TPOT), and must be present for a successful request with
      ``output_tokens >= 2``.

    TPOT is computed with the agreed formula::

        tpot_ms = (e2e_latency_ms - ttft_ms) / (output_tokens - 1)

    i.e. the time spent generating every token *after* the first, divided by the
    number of such inter-token steps. The producer is responsible for computing
    it this way; the schema only enforces the presence/absence rule above.
    """

    request_id: str = Field(description="Unique id within the experiment.")

    prompt_tokens: int = Field(ge=0, description="Prompt length actually sent.")
    output_tokens: int = Field(ge=0, description="Tokens actually generated.")

    # Timestamps are relative seconds from experiment start, to stay stable
    # across machines/clocks and to round-trip exactly as floats.
    start_time_s: float = Field(ge=0, description="Send time, seconds from experiment start.")
    end_time_s: float = Field(ge=0, description="Completion time, seconds from experiment start.")

    ttft_ms: Optional[float] = Field(
        default=None, ge=0, description="Time to first token (ms). None if request failed early."
    )
    tpot_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Mean time per output token (ms) after the first token.",
    )
    e2e_latency_ms: Optional[float] = Field(
        default=None, ge=0, description="End-to-end latency (ms)."
    )

    success: bool = Field(default=True, description="Whether the request completed successfully.")
    error: Optional[str] = Field(
        default=None, description="Error summary when success is False."
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> "RequestMeasurement":
        if self.end_time_s < self.start_time_s:
            raise ValueError(
                f"end_time_s ({self.end_time_s}) must be >= start_time_s "
                f"({self.start_time_s})."
            )

        # success <-> error
        if self.success and self.error is not None:
            raise ValueError("a successful request must not carry an error.")
        if not self.success and self.error is None:
            raise ValueError("a failed request must carry an error.")

        # TPOT is only defined with >= 2 output tokens.
        if self.output_tokens < 2 and self.tpot_ms is not None:
            raise ValueError(
                "tpot_ms must be absent when output_tokens < 2 "
                "(no inter-token interval exists)."
            )

        if self.success:
            if self.output_tokens < 1:
                raise ValueError("a successful request must produce >= 1 output token.")
            if self.ttft_ms is None:
                raise ValueError("a successful request must have ttft_ms.")
            if self.e2e_latency_ms is None:
                raise ValueError("a successful request must have e2e_latency_ms.")
            if self.output_tokens >= 2 and self.tpot_ms is None:
                raise ValueError(
                    "a successful request with >= 2 output tokens must have tpot_ms."
                )

        return self
