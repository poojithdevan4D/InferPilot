"""Read-only diagnosis from a live vLLM's Prometheus /metrics — no benchmark run.

The biggest barrier to using InferPilot is that everything else needs a controlled
rate sweep. But an engineer running vLLM already has a ``/metrics`` endpoint. Two
snapshots of it, 30-60s apart, carry the signals that actually separate the regimes:

  * ``vllm:kv_cache_usage_perc``       — is the KV cache full?
  * ``vllm:num_preemptions_total``     — is the scheduler preempting (the delta over the
                                          two snapshots is the preemption *rate*)?
  * ``vllm:num_requests_waiting``      — is a queue building while KV has headroom?

From those this module gives a calibrated screening result: "KV pressure and
preemption, so fp8 is worth a canary", "a queue is building while KV has headroom,
so prioritize compute or scaling tests", or "capture a fresh interval". It names
the missing metric when it cannot decide. It does not claim a validated bottleneck
or predict the effect of a change.

The reading recomputes its verdict from the embedded snapshots on load.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel

KV_FULL = 0.95

_COUNTERS = ("num_preemptions_total", "recomputed_token_executions_total",
             "prompt_tokens_total", "generation_tokens_total")
_GAUGES = ("kv_cache_usage_perc", "num_requests_running", "num_requests_waiting")


def _scrape(text: str, metric: str, *, aggregate: str) -> Optional[float]:
    """Sum (counters) or max (gauges) a vLLM metric's samples across all label sets."""
    pattern = re.compile(rf"^vllm:{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$", re.MULTILINE)
    vals = []
    for m in pattern.finditer(text):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            continue
    if not vals:
        return None
    return sum(vals) if aggregate == "sum" else max(vals)


class VLLMMetricsSnapshot(SchemaModel):
    """A parsed read of the vLLM metrics this screening needs. Missing fields are None."""

    kv_cache_usage_perc: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    num_requests_running: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    num_requests_waiting: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    num_preemptions_total: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    recomputed_token_executions_total: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    prompt_tokens_total: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    generation_tokens_total: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)

    @classmethod
    def parse(cls, text: str) -> "VLLMMetricsSnapshot":
        data = {name: _scrape(text, name, aggregate="max") for name in _GAUGES}
        data.update({name: _scrape(text, name, aggregate="sum") for name in _COUNTERS})
        return cls(**data)


Verdict = Literal[
    "kv_capacity_bound_preempting",  # KV full AND preempting -> fp8 KV is worth a controlled canary
    "near_capacity",                 # KV full but not preempting in this window
    "not_kv_bound",                  # queueing with KV headroom -> no KV-capacity signal in this snapshot
    "healthy_or_underutilized",      # KV headroom and nothing queuing
    "need_second_snapshot",          # KV full but only one snapshot -> can't measure the preemption rate
    "insufficient_metrics",          # a signal needed to decide is missing
]


def _preemption_rising(before: VLLMMetricsSnapshot, after: Optional[VLLMMetricsSnapshot]) -> Optional[bool]:
    if after is None:
        return None
    for field in ("num_preemptions_total", "recomputed_token_executions_total"):
        b, a = getattr(before, field), getattr(after, field)
        if b is not None and a is not None:
            # A counter decrease means the server/process or metric series reset.
            # Treat the interval as unknown instead of falsely calling it flat.
            if a < b:
                return None
            return (a - b) > 0
    return None


def _derive(
    before: VLLMMetricsSnapshot, after: Optional[VLLMMetricsSnapshot]
) -> tuple[Verdict, str, list[str], str]:
    """Returns (verdict, recommended_lever, reasons, next_step)."""
    snap = after or before
    kv = snap.kv_cache_usage_perc
    waiting = snap.num_requests_waiting

    if kv is None:
        return "insufficient_metrics", "none", ["missing_vllm:kv_cache_usage_perc"], (
            "Scrape vllm:kv_cache_usage_perc (on by default) — it is the core KV-pressure signal.")

    preempting = _preemption_rising(before, after)

    if kv >= KV_FULL:
        if preempting is True:
            return "kv_capacity_bound_preempting", "kv_cache_dtype=fp8", [
                f"kv_usage_{kv:.2f}>=full", "preemptions_rising_between_snapshots",
            ], ("The KV cache is full and the scheduler is preempting. This screening signal makes "
                "fp8 KV worth testing in a controlled canary; it does not predict the size or cause "
                "of any gain.")
        if preempting is False:
            return "near_capacity", "none", [
                f"kv_usage_{kv:.2f}>=full", "no_preemptions_in_this_window",
            ], ("KV is full but not preempting yet. fp8 would buy headroom before the wall, "
                "but it is pre-emptive insurance, not a measured win — watch the preemption counter.")
        return "need_second_snapshot", "none", [
            f"kv_usage_{kv:.2f}>=full", "preemption_rate_unknown_from_one_snapshot",
        ], ("KV is full. Capture a second /metrics 30-60s apart so I can measure whether the "
            "server is actually preempting (a single counter value is cumulative, not a rate).")

    # KV below full.
    if waiting is None:
        return "insufficient_metrics", "none", [
            f"kv_usage_{kv:.2f}_below_full", "missing_vllm:num_requests_waiting",
        ], ("Scrape vllm:num_requests_waiting — with KV headroom, the queue depth is what tells "
            "compute-bound from idle.")
    if waiting > 0:
        return "not_kv_bound", "none", [
            f"kv_usage_{kv:.2f}_below_full", f"requests_waiting_{waiting:g}",
        ], ("Requests are queuing while the KV cache has headroom. This snapshot does not support "
            "KV capacity as the limiting signal, so prioritize a controlled scaling or compute-side "
            "test before spending on an fp8 KV sweep.")
    return "healthy_or_underutilized", "none", [
        f"kv_usage_{kv:.2f}_below_full", "no_requests_waiting",
    ], ("KV has headroom and nothing is queuing. No KV lever is warranted; if latency matters, that "
        "is a separate, low-load question.")


class LiveReading(SchemaModel):
    """Self-validating read-only screening from one or two vLLM /metrics snapshots."""

    before: VLLMMetricsSnapshot
    after: Optional[VLLMMetricsSnapshot] = None
    verdict: Verdict
    recommended_lever: str
    reasons: list[str]
    next_step: str

    @model_validator(mode="after")
    def _check(self) -> "LiveReading":
        verdict, lever, reasons, nxt = _derive(self.before, self.after)
        if (self.verdict, self.recommended_lever, self.reasons, self.next_step) != (verdict, lever, reasons, nxt):
            raise ValueError("live reading is inconsistent with its metrics snapshots")
        return self


def inspect_metrics(before_text: str, after_text: Optional[str] = None) -> LiveReading:
    """Screen a live vLLM from one or two raw /metrics snapshots. No run required."""
    before = VLLMMetricsSnapshot.parse(before_text)
    after = VLLMMetricsSnapshot.parse(after_text) if after_text is not None else None
    verdict, lever, reasons, nxt = _derive(before, after)
    return LiveReading(before=before, after=after, verdict=verdict,
                       recommended_lever=lever, reasons=reasons, next_step=nxt)
