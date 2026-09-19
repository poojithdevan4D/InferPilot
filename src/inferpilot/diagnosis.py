"""Separate observed load state from hypotheses about its cause.

GPU utilization cannot identify compute versus memory bandwidth versus recompute.
KV occupancy and preemption events nominate a canary; they do not predict its gain.
Version 0.2 diagnoses must be recomputed, not replayed as version 0.3 decisions.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .results import ExperimentResult
from .saturation import LoadAssessment, LoadEvidence, LoadState, assess_load_state, detect_saturation

GPU_PINNED_PCT = 90.0
KV_FULL = 0.95
GPU_LOW_PCT = 60.0

Regime = Literal[
    "compute_bound", "kv_capacity_bound_decode", "kv_capacity_bound_prefill",
    "low_utilization_latency", "underutilized", "other_bottleneck",
    "unknown", "no_load_pressure", "kv_pressure",
]


def _classify(
    gpu_mean: Optional[float], kv_peak: Optional[float], saturated: bool,
    decode_heavy: bool, preemptions: Optional[int] = None, *,
    load_state: LoadState = "indeterminate",
) -> tuple[Regime, str, list[str], str]:
    # Legacy positional arguments remain accepted, but no boolean can certify health.
    if load_state == "healthy":
        return "no_load_pressure", "none", [], "No load pressure observed in this window; headroom and SLO compliance are separate questions."
    if load_state == "overloaded" and kv_peak is not None and kv_peak >= KV_FULL and preemptions is not None and preemptions > 0:
        return (
            "kv_pressure", "kv_cache_dtype=fp8",
            ["engine_model_hardware_support_verified", "representative_quality_gate_passed",
             "long_context_accuracy_verified", "controlled_canary_and_rollback_required"],
            "Aligned KV pressure and preemption nominate an fp8 KV canary. Event counts do not measure recompute waste or predict a throughput gain.",
        )
    return "unknown", "none", [], f"Load state is {load_state}; evidence does not identify an actionable bottleneck. Collect aligned signals; do not infer compute-bound from GPU utilization."


class BottleneckDiagnosis(SchemaModel):
    diagnosis_version: Literal["0.3.0"] = "0.3.0"
    gpu_utilization_mean_pct: Optional[float] = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    kv_cache_usage_peak_perc: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    saturated: bool
    decode_heavy: bool
    preemptions: Optional[int] = Field(default=None, ge=0)
    saturation_growth_ratio: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    achieved_throughput_rps: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    offered_rate_qps: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    load_assessment: Optional[LoadAssessment] = None
    regime: Regime
    recommended_lever: str
    lever_preconditions: list[str]
    predicted_effect: str

    @property
    def load_state(self) -> LoadState:
        return self.load_assessment.state if self.load_assessment is not None else "indeterminate"

    @model_validator(mode="after")
    def _check(self) -> "BottleneckDiagnosis":
        if self.saturated != (self.load_state == "overloaded"):
            raise ValueError("diagnosis is inconsistent with its signals: load state")
        e = self.load_assessment.evidence if self.load_assessment is not None else None
        signals = (self.gpu_utilization_mean_pct, self.kv_cache_usage_peak_perc, self.preemptions)
        expected_signals = (None, None, None) if e is None else (e.gpu_utilization_mean_pct, e.kv_cache_usage_peak_perc, e.preemptions)
        if signals != expected_signals:
            raise ValueError("diagnosis is inconsistent with its signals: unaligned telemetry")
        expected = _classify(signals[0], signals[1], self.saturated, self.decode_heavy,
                             self.preemptions, load_state=self.load_state)
        if (self.regime, self.recommended_lever, self.lever_preconditions, self.predicted_effect) != expected:
            raise ValueError("diagnosis is inconsistent with its signals")
        return self


def diagnose(result: ExperimentResult, *, load_evidence: Optional[LoadEvidence] = None) -> BottleneckDiagnosis:
    if load_evidence is None:
        load_evidence = result.load_evidence
    if load_evidence is not None and load_evidence.experiment_id != result.config.experiment_id:
        raise ValueError("load evidence experiment ID mismatch")
    assessment = assess_load_state(result.measurements, evidence=load_evidence)
    e = assessment.evidence
    gpu, kv, preemptions = ((None, None, None) if e is None else
                           (e.gpu_utilization_mean_pct, e.kv_cache_usage_peak_perc, e.preemptions))
    w = result.config.workload
    saturated = assessment.state == "overloaded"
    decode_heavy = w.output_tokens >= w.prompt_tokens  # descriptive only, not causal
    regime, lever, preconditions, effect = _classify(
        gpu, kv, saturated, decode_heavy, preemptions, load_state=assessment.state,
    )
    return BottleneckDiagnosis(
        gpu_utilization_mean_pct=gpu, kv_cache_usage_peak_perc=kv, preemptions=preemptions,
        saturated=saturated, decode_heavy=decode_heavy, load_assessment=assessment,
        saturation_growth_ratio=detect_saturation(result.measurements).growth_ratio,
        achieved_throughput_rps=(result.aggregates.throughput_requests_per_s if result.aggregates else None),
        offered_rate_qps=w.request_rate_qps, regime=regime, recommended_lever=lever,
        lever_preconditions=preconditions, predicted_effect=effect,
    )
