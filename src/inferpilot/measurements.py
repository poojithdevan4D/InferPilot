"""Per-request measurements: the raw ground-truth signal.

One ``RequestMeasurement`` per issued request. Aggregates (percentiles,
throughput) are derived from these in ``results``; the raw records are always
retained so that any aggregate can be recomputed and audited later.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ._base import SchemaModel


class RequestMeasurement(SchemaModel):
    """Timing and token counts for a single request."""

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
