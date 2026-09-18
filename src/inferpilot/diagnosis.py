"""Bottleneck diagnosis: WHY a config performs as it does, and whether a knob can help.

The lesson from the 7B/decode/14B campaigns: config tuning is not a lottery. A serving
config can only beat the vLLM default when the default leaves a *specific* bottleneck
addressable by a *specific* lever. Blindly sweeping knobs and hoping is search-by-prayer;
diagnosing the bottleneck first tells you whether any lever exists at all — and which.

This classifies a measured run into a regime from signals we already collect
(GPU utilization, KV-cache usage) plus the TTFT-stability saturation signal, and maps
the regime to a recommended lever with a mechanistic predicted effect. It ABSTAINS
(lever "none") whenever no config lever can help — which is the honest and common case.

Roofline reasoning (the discriminator is GPU-utilization headroom):

  * GPU pinned (~100%) + saturated  -> COMPUTE/BANDWIDTH-bound. The wall is raw decode of
    the weights; more concurrency (fp8, higher max_num_seqs) cannot add throughput. No
    config lever — needs quantization / smaller model / more GPUs / speculative decoding.
  * GPU has headroom (<~85%) + KV full (~100%) + saturated -> KV-CAPACITY-bound. Concurrency
    is throttled by KV, not compute. Lever: kv_cache_dtype=fp8 (or lower KV footprint) to
    fit more concurrent sequences; predicted throughput gain up to the KV compression ratio,
    bounded by the remaining GPU headroom. THIS is where config wins live.
  * Not saturated + GPU low -> UNDERUTILIZED / latency-bound. The default already keeps up;
    config barely matters.
  * Saturated + GPU low + KV not full -> OTHER (scheduler/client/CPU) — investigate, no
    weights-level lever implied.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .results import ExperimentResult
from .saturation import detect_saturation

GPU_PINNED_PCT = 90.0     # >= this = compute/bandwidth is the wall
KV_FULL = 0.95            # >= this = KV cache is the concurrency limiter
GPU_LOW_PCT = 60.0        # <  this (and not saturated) = latency-lever headroom

# Operator-grade regime map (grounded in vLLM mechanics; see DeepSeek review 2026-09-18).
Regime = Literal[
    "compute_bound",              # GPU pinned -> no config lever
    "kv_capacity_bound_decode",   # GPU headroom + KV full + decode-heavy -> fp8 KV
    "kv_capacity_bound_prefill",  # GPU headroom + KV full + prefill/ITL-sensitive -> lower batched-tokens
    "low_utilization_latency",    # spare GPU, keeps up -> spec-decode IF latency-SLO
    "underutilized",              # keeps up -> default adequate
    "other_bottleneck",           # saturated, spare GPU+KV -> scheduler/client/CPU
]


def _classify(
    gpu_mean: float, kv_peak: float, saturated: bool, decode_heavy: bool
) -> tuple[Regime, str, list[str], str]:
    """Pure regime -> (regime, recommended_lever, preconditions, predicted_effect).

    Saturation gates a *limiting* bottleneck: a run that keeps up (TTFT stable) is not
    throughput-bottlenecked. GPU *mean* (sustained) decides compute-pinned. Levers carry
    preconditions to VERIFY before applying — a diagnostic condition, not a blanket flag."""
    if not saturated:
        if gpu_mean < GPU_LOW_PCT and kv_peak < 0.5:
            return (
                "low_utilization_latency",
                "speculative_decoding",
                ["latency_slo_present", "itl_sensitive", "qps_low"],
                "low QPS with spare GPU/KV: speculative decoding (EAGLE/MTP) can cut ITL "
                "1.5-3x at low concurrency; gains vanish at high QPS, so apply ONLY under a "
                "latency SLO, never as a throughput lever",
            )
        return (
            "underutilized",
            "none",
            [],
            "server keeps up (TTFT stable); the vLLM default is already adequate here",
        )
    if gpu_mean >= GPU_PINNED_PCT:
        return (
            "compute_bound",
            "none",
            [],
            "GPU is compute/bandwidth-pinned; concurrency levers (fp8 KV, max_num_seqs) "
            "cannot raise throughput. Needs quantization / smaller model / more GPUs",
        )
    if kv_peak >= KV_FULL:
        if decode_heavy:
            return (
                "kv_capacity_bound_decode",
                "kv_cache_dtype=fp8",
                ["model_attention_not_sliding_window", "model_head_dim_ne_256",
                 "workload_decode_dominated"],
                "KV limits concurrency with GPU headroom; fp8 KV (~54% of BF16 bytes) should "
                "raise concurrent sequences ~30-50% at similar ITL. CAVEAT: hybrid/sliding-"
                "window models gain little decode speedup and head_dim=256 can regress prefill",
            )
        return (
            "kv_capacity_bound_prefill",
            "max_num_batched_tokens_lower",
            ["chunked_prefill_enabled", "prefill_bursts_disrupting_decode"],
            "KV-bound but prefill bursts disrupt decode ITL; lowering max_num_batched_tokens "
            "(e.g. 512-1024) reduces prefill interference and improves ITL, at slightly worse "
            "TTFT — a genuine tradeoff, valid only when prefill is the disruptor",
        )
    return (
        "other_bottleneck",
        "none",
        [],
        "saturated with spare GPU and KV — likely scheduler/client/CPU bound; no "
        "weights-level config lever implied",
    )


class BottleneckDiagnosis(SchemaModel):
    """Self-validating regime diagnosis + mechanistic lever recommendation."""

    diagnosis_version: Literal["0.2.0"] = "0.2.0"
    gpu_utilization_mean_pct: float = Field(ge=0)
    kv_cache_usage_peak_perc: float = Field(ge=0)
    saturated: bool
    decode_heavy: bool
    saturation_growth_ratio: Optional[float] = None
    achieved_throughput_rps: Optional[float] = None
    offered_rate_qps: Optional[float] = None
    regime: Regime
    recommended_lever: str
    lever_preconditions: list[str]
    predicted_effect: str

    @model_validator(mode="after")
    def _check(self) -> "BottleneckDiagnosis":
        regime, lever, preconds, effect = _classify(
            self.gpu_utilization_mean_pct, self.kv_cache_usage_peak_perc,
            self.saturated, self.decode_heavy,
        )
        if (self.regime, self.recommended_lever, self.lever_preconditions, self.predicted_effect) != (
            regime, lever, preconds, effect
        ):
            raise ValueError("diagnosis is inconsistent with its signals")
        return self


def diagnose(result: ExperimentResult) -> BottleneckDiagnosis:
    """Diagnose the bottleneck of a measured run from telemetry + saturation."""
    if result.telemetry is None or result.aggregates is None:
        raise ValueError("diagnosis requires telemetry and aggregates")
    t = result.telemetry
    w = result.config.workload
    sat = detect_saturation(result.measurements)
    decode_heavy = w.output_tokens >= w.prompt_tokens  # long outputs, modest inputs
    regime, lever, preconds, effect = _classify(
        t.gpu_utilization_mean_pct, t.kv_cache_usage_peak_perc, sat.saturated, decode_heavy
    )
    return BottleneckDiagnosis(
        gpu_utilization_mean_pct=t.gpu_utilization_mean_pct,
        kv_cache_usage_peak_perc=t.kv_cache_usage_peak_perc,
        saturated=sat.saturated,
        decode_heavy=decode_heavy,
        saturation_growth_ratio=sat.growth_ratio,
        achieved_throughput_rps=result.aggregates.throughput_requests_per_s,
        offered_rate_qps=w.request_rate_qps,
        regime=regime,
        recommended_lever=lever,
        lever_preconditions=preconds,
        predicted_effect=effect,
    )
