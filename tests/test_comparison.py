"""Cohort construction and controlled run-level comparison."""

from __future__ import annotations

import pytest

from inferpilot import (
    AggregateMetrics,
    EffectiveConfig,
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    HardwareInfo,
    ResourceTelemetry,
    ToolchainInfo,
    WorkloadSpec,
)
from inferpilot.comparison import build_cohort, compare_cohorts


def _result(
    experiment_id: str,
    run_number: int,
    *,
    max_num_seqs: int = 1,
    throughput: float = 100.0,
    ttft_p50: float = 20.0,
    gpu_name: str = "GPU-A",
) -> ExperimentResult:
    config = ExperimentConfig(
        experiment_id=experiment_id,
        name=experiment_id,
        engine=EngineConfig(
            model="org/model",
            revision="abc",
            max_model_len=2048,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=False,
            sampler_backend="pytorch",
        ),
        workload=WorkloadSpec(
            name="c4",
            num_requests=32,
            warmup_requests=4,
            prompt_tokens=128,
            output_tokens=32,
            max_concurrency=4,
            temperature=0,
            ignore_eos=True,
        ),
    )
    environment = EnvironmentMetadata(
        platform="Linux-test",
        python_version="3.12.0",
        torch_version="2.13.0",
        vllm_version="0.29.0",
        hardware=HardwareInfo(gpu_name=gpu_name, gpu_count=1, gpu_memory_total_mb=4096),
        toolchain=ToolchainInfo(torch_cuda_version="13.0"),
        effective_sampler_backend="pytorch",
        runtime_overrides={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
    )
    effective = EffectiveConfig(
        model="org/model",
        revision="abc",
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        sampler_backend="pytorch",
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.85,
        generation_config="vllm",
        verified=True,
    )
    aggregates = AggregateMetrics(
        num_requests=32,
        num_successful=32,
        num_failed=0,
        duration_s=10,
        ttft_p50_ms=ttft_p50,
        ttft_p95_ms=ttft_p50 + 5,
        ttft_p99_ms=ttft_p50 + 8,
        tpot_p50_ms=6,
        tpot_p95_ms=7,
        tpot_p99_ms=8,
        e2e_p50_ms=210,
        e2e_p95_ms=230,
        e2e_p99_ms=240,
        throughput_tokens_per_s=throughput,
        throughput_requests_per_s=throughput / 32,
        total_output_tokens=1024,
        gpu_memory_peak_mb=3000,
    )
    telemetry = ResourceTelemetry(
        sample_interval_s=0.25,
        num_samples=20,
        peak_gpu_memory_mb=3000,
        gpu_utilization_mean_pct=90 + run_number,
        gpu_utilization_peak_pct=100,
        kv_cache_usage_mean_perc=0.1,
        kv_cache_usage_peak_perc=0.2,
    )
    return ExperimentResult(
        config=config,
        environment=environment,
        status=ExperimentStatus.COMPLETED,
        aggregates=aggregates,
        effective_config=effective,
        telemetry=telemetry,
    )


def _cohort(experiment_id: str, *, max_num_seqs: int, throughput: float, ttft: float):
    return [
        (
            f"{experiment_id}-{i}",
            _result(
                experiment_id,
                i,
                max_num_seqs=max_num_seqs,
                throughput=throughput + i,
                ttft_p50=ttft + i * 0.1,
            ),
        )
        for i in range(3)
    ]


def test_build_cohort_uses_run_level_statistics() -> None:
    cohort = build_cohort(_cohort("base", max_num_seqs=1, throughput=100, ttft=20))
    stats = cohort.metrics["throughput_tokens_per_s"]
    assert stats.count == 3
    assert stats.mean == 101
    assert stats.sample_stddev == 1


def test_build_cohort_requires_three_exactly_compatible_runs() -> None:
    runs = _cohort("base", max_num_seqs=1, throughput=100, ttft=20)
    with pytest.raises(ValueError, match="at least 3"):
        build_cohort(runs[:2])

    runs[2] = ("base-2", _result("base", 2, gpu_name="GPU-B"))
    with pytest.raises(ValueError, match="non-equivalent"):
        build_cohort(runs)


def test_compare_allows_only_declared_engine_change_and_applies_direction() -> None:
    baseline = _cohort("base", max_num_seqs=1, throughput=100, ttft=20)
    candidate = _cohort("candidate", max_num_seqs=4, throughput=110, ttft=18)
    report = compare_cohorts(
        baseline, candidate, varied_engine_fields=["max_num_seqs"]
    )

    assert report.varied_engine_fields == ["max_num_seqs"]
    assert report.metrics["throughput_tokens_per_s"].improvement_pct > 0
    assert report.metrics["ttft_p50_ms"].improvement_pct > 0
    assert report.metrics["peak_kv_cache_usage"].direction == "context_only"
    assert report.metrics["peak_kv_cache_usage"].improvement_pct is None


def test_compare_refuses_hidden_workload_or_environment_change() -> None:
    baseline = _cohort("base", max_num_seqs=1, throughput=100, ttft=20)
    candidate = _cohort("candidate", max_num_seqs=4, throughput=110, ttft=18)
    changed = _result("candidate", 2, max_num_seqs=4, gpu_name="GPU-B")
    candidate[2] = ("candidate-2", changed)
    with pytest.raises(ValueError, match="non-equivalent"):
        compare_cohorts(baseline, candidate, varied_engine_fields=["max_num_seqs"])


def test_compare_refuses_no_actual_change_and_unknown_field() -> None:
    baseline = _cohort("base", max_num_seqs=1, throughput=100, ttft=20)
    same = _cohort("candidate", max_num_seqs=1, throughput=110, ttft=18)
    with pytest.raises(ValueError, match="did not actually change"):
        compare_cohorts(baseline, same, varied_engine_fields=["max_num_seqs"])
    with pytest.raises(ValueError, match="unknown EngineConfig"):
        compare_cohorts(baseline, same, varied_engine_fields=["not_a_knob"])
