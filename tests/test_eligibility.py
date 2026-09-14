"""Tests for ExperimentResult.is_baseline_eligible (benchmark-validity gate)."""

from __future__ import annotations

from inferpilot import (
    AggregateMetrics,
    EngineConfig,
    EnvironmentMetadata,
    EffectiveConfig,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    RequestMeasurement,
    ResourceTelemetry,
    WorkloadSpec,
)


def _config(num_requests: int = 4) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp-elig",
        name="elig",
        engine=EngineConfig(model="org/model"),
        workload=WorkloadSpec(
            name="w", num_requests=num_requests, prompt_tokens=8, output_tokens=8,
            max_concurrency=1,
        ),
    )


def _full_metrics(num_requests: int, num_successful: int) -> AggregateMetrics:
    """A metrics object with every comparison field populated."""
    return AggregateMetrics(
        num_requests=num_requests,
        num_successful=num_successful,
        num_failed=num_requests - num_successful,
        duration_s=2.0,
        ttft_p50_ms=10.0, ttft_p95_ms=12.0, ttft_p99_ms=13.0,
        tpot_p50_ms=5.0, tpot_p95_ms=6.0, tpot_p99_ms=7.0,
        e2e_p50_ms=100.0, e2e_p95_ms=120.0, e2e_p99_ms=130.0,
        throughput_tokens_per_s=50.0, throughput_requests_per_s=2.0,
        total_output_tokens=32, gpu_memory_peak_mb=1024,
    )


def _result(status, aggregates, config=None) -> ExperimentResult:
    return ExperimentResult(
        config=config or _config(),
        environment=EnvironmentMetadata(),
        status=status,
        aggregates=aggregates,
        effective_config=EffectiveConfig(verified=True),
        telemetry=ResourceTelemetry(
            sample_interval_s=0.25,
            num_samples=4,
            peak_gpu_memory_mb=1024,
            gpu_utilization_mean_pct=50,
            gpu_utilization_peak_pct=75,
            kv_cache_usage_mean_perc=0.1,
            kv_cache_usage_peak_perc=0.2,
        ),
    )


def test_fully_clean_completed_run_is_eligible() -> None:
    res = _result(ExperimentStatus.COMPLETED, _full_metrics(4, 4))
    assert res.is_baseline_eligible is True


def test_all_failed_completed_run_is_ineligible() -> None:
    agg = _full_metrics(4, 0)  # 0 successful, 4 failed
    res = _result(ExperimentStatus.COMPLETED, agg)
    assert res.status is ExperimentStatus.COMPLETED  # orchestration completed
    assert res.is_baseline_eligible is False  # but not a valid benchmark


def test_partially_failed_run_is_ineligible() -> None:
    res = _result(ExperimentStatus.COMPLETED, _full_metrics(4, 3))
    assert res.is_baseline_eligible is False


def test_wrong_request_count_is_ineligible() -> None:
    # Only 3 measured though the workload asked for 4.
    res = _result(ExperimentStatus.COMPLETED, _full_metrics(3, 3), config=_config(4))
    assert res.is_baseline_eligible is False


def test_missing_comparison_metric_is_ineligible() -> None:
    agg = _full_metrics(4, 4)
    agg = agg.model_copy(update={"tpot_p95_ms": None})
    res = _result(ExperimentStatus.COMPLETED, agg)
    assert res.is_baseline_eligible is False


def test_missing_throughput_is_ineligible() -> None:
    agg = _full_metrics(4, 4).model_copy(update={"throughput_tokens_per_s": None})
    res = _result(ExperimentStatus.COMPLETED, agg)
    assert res.is_baseline_eligible is False


def test_unverified_effective_config_is_ineligible() -> None:
    res = _result(ExperimentStatus.COMPLETED, _full_metrics(4, 4))
    res.effective_config = EffectiveConfig(
        verified=False, unverified_fields=["max_num_seqs"]
    )
    assert res.is_baseline_eligible is False


def test_missing_or_incomplete_telemetry_is_ineligible() -> None:
    res = _result(ExperimentStatus.COMPLETED, _full_metrics(4, 4))
    res.telemetry = None
    assert res.is_baseline_eligible is False

    res = _result(ExperimentStatus.COMPLETED, _full_metrics(4, 4))
    assert res.telemetry is not None
    res.telemetry = res.telemetry.model_copy(update={"kv_cache_usage_peak_perc": None})
    assert res.is_baseline_eligible is False


def test_telemetry_error_is_ineligible() -> None:
    res = _result(ExperimentStatus.COMPLETED, _full_metrics(4, 4))
    assert res.telemetry is not None
    res.telemetry = res.telemetry.model_copy(update={"error": "metrics_fetch_failed"})
    assert res.is_baseline_eligible is False


def test_non_completed_is_ineligible() -> None:
    res = ExperimentResult(
        config=_config(),
        environment=EnvironmentMetadata(),
        status=ExperimentStatus.OOM,
        aggregates=None,
        failure={"status": "oom", "error_type": "CudaOOM", "message": "boom"},
    )
    assert res.is_baseline_eligible is False
