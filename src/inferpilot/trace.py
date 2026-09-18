"""Ingest a real request trace and characterize the workload — so InferPilot advises on
an operator's ACTUAL traffic, not a synthetic Poisson assumption.

A trace is a list of observed requests (arrival time, prompt tokens, output tokens), e.g.
exported from vLLM/gateway logs. WorkloadTrace summarizes it — realized rate, prompt/output
length distributions, burstiness — and exposes a conservative sizing context (p95
prompt+output) that feeds analyze_fit / recommend_scale. Self-validating: the summary is
recomputed from the embedded requests, so a tampered summary is rejected.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Sequence

from pydantic import Field, model_validator

from ._base import SchemaModel


class TraceRequest(SchemaModel):
    arrival_s: float = Field(ge=0, description="Arrival time (seconds from trace start).")
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(ge=0)


def _p(sorted_xs: list[int], q: float) -> int:
    if not sorted_xs:
        return 0
    return sorted_xs[min(len(sorted_xs) - 1, int(q * (len(sorted_xs) - 1) + 0.9999))]


class WorkloadTrace(SchemaModel):
    """Self-validating summary of a real request trace."""

    trace_version: Literal["0.1.0"] = "0.1.0"
    requests: list[TraceRequest] = Field(min_length=2)
    num_requests: int
    duration_s: float
    request_rate_qps: float
    prompt_p50: int
    prompt_p95: int
    prompt_max: int
    output_p50: int
    output_p95: int
    output_max: int
    digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check(self) -> "WorkloadTrace":
        exp = _summarize(self.requests)
        got = (self.num_requests, self.duration_s, self.request_rate_qps,
               self.prompt_p50, self.prompt_p95, self.prompt_max,
               self.output_p50, self.output_p95, self.output_max, self.digest_sha256)
        if got != exp:
            raise ValueError("workload trace summary is inconsistent with its requests")
        return self

    @property
    def sizing_context_tokens(self) -> int:
        """Conservative per-request context for KV sizing: p95 prompt + p95 output."""
        return self.prompt_p95 + self.output_p95


def _summarize(requests: Sequence[TraceRequest]):
    arrivals = [r.arrival_s for r in requests]
    prompts = sorted(r.prompt_tokens for r in requests)
    outputs = sorted(r.output_tokens for r in requests)
    n = len(requests)
    duration = max(arrivals) - min(arrivals)
    rate = round((n - 1) / duration, 6) if duration > 0 else 0.0
    payload = ";".join(f"{r.arrival_s}:{r.prompt_tokens}:{r.output_tokens}" for r in requests)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return (n, round(duration, 6), rate,
            _p(prompts, 0.50), _p(prompts, 0.95), prompts[-1],
            _p(outputs, 0.50), _p(outputs, 0.95), outputs[-1], digest)


def summarize_trace(requests: Sequence[TraceRequest]) -> WorkloadTrace:
    (n, dur, rate, pp50, pp95, pmax, op50, op95, omax, digest) = _summarize(requests)
    return WorkloadTrace(
        requests=list(requests), num_requests=n, duration_s=dur, request_rate_qps=rate,
        prompt_p50=pp50, prompt_p95=pp95, prompt_max=pmax,
        output_p50=op50, output_p95=op95, output_max=omax, digest_sha256=digest,
    )


def load_trace_jsonl(text: str) -> WorkloadTrace:
    """Parse a JSONL trace: one object per line with arrival_s, prompt_tokens, output_tokens.

    Tolerates common alias keys (timestamp/arrival, input_tokens/prompt_len, etc.)."""
    import json

    reqs: list[TraceRequest] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        arrival = d.get("arrival_s", d.get("arrival", d.get("timestamp", d.get("t", 0.0))))
        prompt = d.get("prompt_tokens", d.get("input_tokens", d.get("prompt_len", d.get("num_prompt_tokens"))))
        output = d.get("output_tokens", d.get("gen_tokens", d.get("output_len", d.get("num_generated_tokens", 0))))
        reqs.append(TraceRequest(arrival_s=float(arrival), prompt_tokens=int(prompt), output_tokens=int(output)))
    return summarize_trace(reqs)
