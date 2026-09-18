"""Capacity advisory: tune / scale / adequate with cost framing — the repositioned core."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import (
    SLO,
    EffectiveConfig,
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentStatus,
    HardwareInfo,
    RequestMeasurement,
    ResourceTelemetry,
    WorkloadSpec,
)
from inferpilot.advisor import CapacityAdvisory, OperatorEconomics, advise_capacity
from inferpilot.runner.aggregate import compute_aggregates


def _result(*, gpu_mean, kv_peak, saturating, ttft=150.0, tpot=40.0, rate=2.0,
            count=40, prompt=2048, out=512, throughput=1.5) -> "ExperimentResult":
    from inferpilot import ExperimentResult
    duration = count / throughput
    ttfts = ([ttft * (0.3 + 0.7 * i / (count - 1)) for i in range(count)] if saturating
             else [ttft] * count)
    meas = [
        RequestMeasurement(
            request_id=str(i), prompt_tokens=prompt, output_tokens=out,
            start_time_s=i * 0.5, end_time_s=i * 0.5 + (t + (out - 1) * tpot) / 1000,
            ttft_ms=t, tpot_ms=tpot, e2e_latency_ms=t + (out - 1) * tpot, success=True,
        )
        for i, t in enumerate(ttfts)
    ]
    agg = compute_aggregates(meas, duration).model_copy(update={"gpu_memory_peak_mb": 20000})
    return ExperimentResult(
        config=ExperimentConfig(
            experiment_id="c", name="c",
            engine=EngineConfig(model="m", revision="r"),
            workload=WorkloadSpec(name="c", num_requests=count, prompt_tokens=prompt,
                                  output_tokens=out, request_rate_qps=rate),
        ),
        environment=EnvironmentMetadata(hardware=HardwareInfo(gpu_name="A100")),
        status=ExperimentStatus.COMPLETED, measurements=meas, aggregates=agg,
        effective_config=EffectiveConfig(model="m", revision="r", verified=True),
        telemetry=ResourceTelemetry(
            sample_interval_s=0.25, num_samples=8, peak_gpu_memory_mb=20000,
            gpu_utilization_mean_pct=gpu_mean, gpu_utilization_peak_pct=min(100.0, gpu_mean + 1),
            kv_cache_usage_mean_perc=kv_peak * 0.9, kv_cache_usage_peak_perc=kv_peak,
        ),
    )


SLO_ = SLO(ttft_p95_ms=5000, tpot_p95_ms=60)
ECON = OperatorEconomics(gpu_cost_per_hour_usd=2.10, gpu_count=1, target_qps=4.0)


def test_compute_bound_recommends_scale_with_gpu_delta() -> None:
    # GPU pinned + saturated (TTFT ramps) -> no lever -> scale
    r = _result(gpu_mean=99.0, kv_peak=0.5, saturating=True, throughput=1.5)
    adv = advise_capacity(r, SLO_, ECON)
    assert adv.action == "scale" and adv.diagnosis.regime == "compute_bound"
    assert adv.gpus_needed_for_target is not None and adv.gpus_needed_for_target >= 2
    assert adv.cost_per_million_output_tokens_usd and adv.cost_per_million_output_tokens_usd > 0


def test_kv_capacity_bound_decode_recommends_fp8() -> None:
    # decode-heavy (out >= prompt) -> fp8 lever
    r = _result(gpu_mean=60.0, kv_peak=0.99, saturating=True, prompt=128, out=512)
    adv = advise_capacity(r, SLO_, ECON)
    assert adv.action == "tune" and "kv_cache_dtype=fp8" in adv.recommendation


def test_kv_capacity_bound_prefill_recommends_batched_tokens() -> None:
    # prefill-heavy (prompt > out) -> lower max_num_batched_tokens
    r = _result(gpu_mean=60.0, kv_peak=0.99, saturating=True, prompt=2048, out=256)
    adv = advise_capacity(r, SLO_, ECON)
    assert adv.action == "tune" and "max_num_batched_tokens_lower" in adv.recommendation


def test_adequate_when_meets_slo_and_target() -> None:
    # keeps up (flat TTFT), high util but not saturated, offered rate >= target
    r = _result(gpu_mean=100.0, kv_peak=0.5, saturating=False, rate=4.0, throughput=4.0)
    adv = advise_capacity(r, SLO_, ECON)
    assert adv.action == "adequate" and adv.met_slo


def test_roundtrip_and_tamper_rejected() -> None:
    r = _result(gpu_mean=99.0, kv_peak=0.5, saturating=True)
    adv = advise_capacity(r, SLO_, ECON)
    assert CapacityAdvisory.model_validate_json(adv.model_dump_json()) == adv
    raw = adv.model_dump(mode="json")
    raw["action"] = "adequate"
    with pytest.raises(ValidationError, match="inconsistent with its diagnosis"):
        CapacityAdvisory.model_validate(raw)
