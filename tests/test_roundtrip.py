"""JSON round-trip tests: no information loss through serialize/deserialize."""

from __future__ import annotations

from datetime import datetime, timezone

from inferpilot import (
    AggregateMetrics,
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    FailureRecord,
    HardwareInfo,
    RequestMeasurement,
    WorkloadSpec,
)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp-rt-001",
        name="roundtrip",
        engine=EngineConfig(model="some-org/some-model", max_num_seqs=8),
        workload=WorkloadSpec(
            name="chat-short",
            num_requests=2,
            prompt_tokens=64,
            output_tokens=64,
            max_concurrency=2,
        ),
    )


def _environment() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        hostname="laptop",
        platform="Linux-6.8-x86_64",
        python_version="3.11.9",
        torch_version="2.3.0",
        vllm_version="0.5.0",
        hardware=HardwareInfo(
            gpu_name="NVIDIA RTX 3050 Laptop GPU",
            gpu_count=1,
            gpu_memory_total_mb=4096,
            cuda_version="12.4",
        ),
        captured_at=datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc),
    )


def _measurements() -> list[RequestMeasurement]:
    return [
        RequestMeasurement(
            request_id="r0",
            prompt_tokens=64,
            output_tokens=64,
            start_time_s=0.0,
            end_time_s=1.5,
            ttft_ms=40.0,
            tpot_ms=12.5,
            e2e_latency_ms=1500.0,
        ),
        RequestMeasurement(
            request_id="r1",
            prompt_tokens=64,
            output_tokens=0,
            start_time_s=0.1,
            end_time_s=0.2,
            success=False,
            error="CUDA out of memory",
        ),
    ]


def _assert_roundtrips(result: ExperimentResult) -> None:
    restored = ExperimentResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_completed_result_roundtrips() -> None:
    result = ExperimentResult(
        config=_config(),
        environment=_environment(),
        status=ExperimentStatus.COMPLETED,
        measurements=_measurements(),
        aggregates=AggregateMetrics(
            num_requests=2,
            num_successful=1,
            num_failed=1,
            duration_s=1.6,
            ttft_p50_ms=40.0,
            e2e_p95_ms=1500.0,
            throughput_tokens_per_s=40.0,
            total_output_tokens=64,
            gpu_memory_peak_mb=3900,
        ),
        started_at=datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 14, 12, 0, 2, tzinfo=timezone.utc),
    )
    _assert_roundtrips(result)


def test_oom_result_roundtrips() -> None:
    """OOM is first-class: no aggregates, populated failure record."""
    result = ExperimentResult(
        config=_config(),
        environment=_environment(),
        status=ExperimentStatus.OOM,
        measurements=[],
        aggregates=None,
        failure=FailureRecord(
            status=ExperimentStatus.OOM,
            error_type="CudaOOM",
            message="Tried to allocate KV cache exceeding 4 GB VRAM.",
            traceback="torch.cuda.OutOfMemoryError: CUDA out of memory ...",
            occurred_at=datetime(2026, 9, 14, 12, 0, 1, tzinfo=timezone.utc),
        ),
    )
    _assert_roundtrips(result)
    assert result.status is ExperimentStatus.OOM
    assert result.aggregates is None


def test_failed_result_with_all_sections_roundtrips() -> None:
    """A single result carrying environment metadata, raw measurements,
    aggregates, AND a failed status must round-trip with no information loss."""
    result = ExperimentResult(
        config=_config(),
        environment=_environment(),
        status=ExperimentStatus.FAILED,
        measurements=_measurements(),
        aggregates=AggregateMetrics(
            num_requests=2,
            num_successful=1,
            num_failed=1,
            duration_s=1.6,
            ttft_p50_ms=40.0,
            total_output_tokens=64,
        ),
        failure=FailureRecord(
            status=ExperimentStatus.FAILED,
            error_type="EngineCrash",
            message="Engine died mid-benchmark after partial results.",
        ),
        started_at=datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 14, 12, 0, 2, tzinfo=timezone.utc),
    )

    restored = ExperimentResult.model_validate_json(result.model_dump_json())
    assert restored == result
    # every section survived
    assert restored.environment.hardware.gpu_memory_total_mb == 4096
    assert len(restored.measurements) == 2
    assert restored.measurements[1].error == "CUDA out of memory"
    assert restored.aggregates is not None and restored.aggregates.num_failed == 1
    assert restored.failure is not None and restored.failure.error_type == "EngineCrash"
    assert restored.status is ExperimentStatus.FAILED
