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
GPU_HEADROOM_PCT = 85.0   # <  this = spare compute a concurrency lever could exploit
KV_FULL = 0.95            # >= this = KV cache is the concurrency limiter

Regime = Literal[
    "compute_bound",
    "kv_capacity_bound",
    "underutilized",
    "other_bottleneck",
]


def _classify(gpu_mean: float, kv_peak: float, saturated: bool) -> tuple[Regime, str, str]:
    """Pure regime -> (regime, recommended_lever, predicted_effect).

    Saturation is the precondition for a *limiting* bottleneck: a run that keeps up
    (TTFT stable) is adequate and needs no lever, regardless of utilization. GPU *mean*
    (sustained), not peak, decides compute-pinned."""
    if not saturated:
        return (
            "underutilized",
            "none",
            "server keeps up (TTFT stable); the vLLM default is already adequate here",
        )
    if gpu_mean >= GPU_PINNED_PCT:
        return (
            "compute_bound",
            "none",
            "GPU is compute/bandwidth-pinned; concurrency levers (fp8 KV, max_num_seqs) "
            "cannot raise throughput. Needs quantization / smaller model / more GPUs / "
            "speculative decoding",
        )
    if kv_peak >= KV_FULL:
        return (
            "kv_capacity_bound",
            "kv_cache_dtype=fp8",
            "KV cache limits concurrency while GPU has compute headroom; halving KV bytes "
            "should raise sustainable concurrency and throughput, bounded by the remaining "
            "GPU headroom",
        )
    return (
        "other_bottleneck",
        "none",
        "saturated with spare GPU and KV — likely scheduler/client/CPU bound; no "
        "weights-level config lever implied",
    )


class BottleneckDiagnosis(SchemaModel):
    """Self-validating regime diagnosis + mechanistic lever recommendation."""

    diagnosis_version: Literal["0.1.0"] = "0.1.0"
    gpu_utilization_mean_pct: float = Field(ge=0)
    kv_cache_usage_peak_perc: float = Field(ge=0)
    saturated: bool
    saturation_growth_ratio: Optional[float] = None
    achieved_throughput_rps: Optional[float] = None
    offered_rate_qps: Optional[float] = None
    regime: Regime
    recommended_lever: str
    predicted_effect: str

    @model_validator(mode="after")
    def _check(self) -> "BottleneckDiagnosis":
        regime, lever, effect = _classify(
            self.gpu_utilization_mean_pct, self.kv_cache_usage_peak_perc, self.saturated
        )
        if (self.regime, self.recommended_lever, self.predicted_effect) != (regime, lever, effect):
            raise ValueError("diagnosis is inconsistent with its signals")
        return self


def diagnose(result: ExperimentResult) -> BottleneckDiagnosis:
    """Diagnose the bottleneck of a measured run from telemetry + saturation."""
    if result.telemetry is None or result.aggregates is None:
        raise ValueError("diagnosis requires telemetry and aggregates")
    t = result.telemetry
    sat = detect_saturation(result.measurements)
    regime, lever, effect = _classify(t.gpu_utilization_mean_pct, t.kv_cache_usage_peak_perc, sat.saturated)
    return BottleneckDiagnosis(
        gpu_utilization_mean_pct=t.gpu_utilization_mean_pct,
        kv_cache_usage_peak_perc=t.kv_cache_usage_peak_perc,
        saturated=sat.saturated,
        saturation_growth_ratio=sat.growth_ratio,
        achieved_throughput_rps=result.aggregates.throughput_requests_per_s,
        offered_rate_qps=result.config.workload.request_rate_qps,
        regime=regime,
        recommended_lever=lever,
        predicted_effect=effect,
    )
