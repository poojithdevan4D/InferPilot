"""Focused negative tests: every invalid contract state must be rejected."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from inferpilot import (
    AggregateMetrics,
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    FailureRecord,
    RequestMeasurement,
    WorkloadSpec,
)

# --------------------------------------------------------------------------- #
# Shared valid building blocks (each negative test perturbs exactly one thing).
# --------------------------------------------------------------------------- #


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp-neg",
        name="neg",
        engine=EngineConfig(model="org/model"),
        workload=WorkloadSpec(
            name="w", num_requests=1, prompt_tokens=8, output_tokens=8, max_concurrency=1
        ),
    )


def test_workload_refuses_ambiguous_arrival_patterns() -> None:
    with pytest.raises(ValidationError, match="exactly one arrival pattern"):
        WorkloadSpec(
            name="w",
            num_requests=1,
            prompt_tokens=8,
            output_tokens=8,
            request_rate_qps=1.0,
            max_concurrency=1,
        )


def _env() -> EnvironmentMetadata:
    return EnvironmentMetadata(hostname="h")


def _ok_aggregates() -> AggregateMetrics:
    return AggregateMetrics(
        num_requests=1, num_successful=1, num_failed=0, duration_s=1.0, total_output_tokens=8
    )


def _failure(status: ExperimentStatus) -> FailureRecord:
    return FailureRecord(status=status, error_type="X", message="boom")


# --------------------------------------------------------------------------- #
# RequestMeasurement
# --------------------------------------------------------------------------- #


def test_measurement_end_before_start_rejected() -> None:
    with pytest.raises(ValidationError, match="end_time_s"):
        RequestMeasurement(
            request_id="r", prompt_tokens=1, output_tokens=1,
            start_time_s=2.0, end_time_s=1.0, ttft_ms=1.0, e2e_latency_ms=1.0,
        )


def test_measurement_success_with_error_rejected() -> None:
    with pytest.raises(ValidationError, match="must not carry an error"):
        RequestMeasurement(
            request_id="r", prompt_tokens=1, output_tokens=1,
            start_time_s=0.0, end_time_s=1.0, ttft_ms=1.0, e2e_latency_ms=1.0,
            success=True, error="oops",
        )


def test_measurement_failed_without_error_rejected() -> None:
    with pytest.raises(ValidationError, match="must carry an error"):
        RequestMeasurement(
            request_id="r", prompt_tokens=1, output_tokens=0,
            start_time_s=0.0, end_time_s=1.0, success=False, error=None,
        )


def test_measurement_success_without_ttft_rejected() -> None:
    with pytest.raises(ValidationError, match="ttft_ms"):
        RequestMeasurement(
            request_id="r", prompt_tokens=1, output_tokens=1,
            start_time_s=0.0, end_time_s=1.0, ttft_ms=None, e2e_latency_ms=1.0,
        )


def test_measurement_success_without_e2e_rejected() -> None:
    with pytest.raises(ValidationError, match="e2e_latency_ms"):
        RequestMeasurement(
            request_id="r", prompt_tokens=1, output_tokens=1,
            start_time_s=0.0, end_time_s=1.0, ttft_ms=1.0, e2e_latency_ms=None,
        )


def test_measurement_success_zero_output_tokens_rejected() -> None:
    with pytest.raises(ValidationError, match=">= 1 output token"):
        RequestMeasurement(
            request_id="r", prompt_tokens=1, output_tokens=0,
            start_time_s=0.0, end_time_s=1.0, ttft_ms=1.0, e2e_latency_ms=1.0,
        )


def test_measurement_one_token_output_with_tpot_rejected() -> None:
    """One-token outputs have no inter-token interval, so TPOT must be absent."""
    with pytest.raises(ValidationError, match="tpot_ms must be absent"):
        RequestMeasurement(
            request_id="r", prompt_tokens=1, output_tokens=1,
            start_time_s=0.0, end_time_s=1.0, ttft_ms=1.0, tpot_ms=5.0, e2e_latency_ms=1.0,
        )


def test_measurement_multi_token_success_without_tpot_rejected() -> None:
    with pytest.raises(ValidationError, match="must have tpot_ms"):
        RequestMeasurement(
            request_id="r", prompt_tokens=1, output_tokens=4,
            start_time_s=0.0, end_time_s=1.0, ttft_ms=1.0, tpot_ms=None, e2e_latency_ms=1.0,
        )


def test_measurement_one_token_success_is_valid() -> None:
    m = RequestMeasurement(
        request_id="r", prompt_tokens=1, output_tokens=1,
        start_time_s=0.0, end_time_s=1.0, ttft_ms=1.0, e2e_latency_ms=1.0,
    )
    assert m.tpot_ms is None


# --------------------------------------------------------------------------- #
# AggregateMetrics
# --------------------------------------------------------------------------- #


def test_aggregate_count_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="must equal num_successful"):
        AggregateMetrics(num_requests=5, num_successful=2, num_failed=2, duration_s=1.0)


def test_aggregate_zero_duration_with_requests_rejected() -> None:
    with pytest.raises(ValidationError, match="duration_s must be > 0"):
        AggregateMetrics(num_requests=1, num_successful=1, num_failed=0, duration_s=0.0)


def test_aggregate_empty_zero_duration_is_valid() -> None:
    agg = AggregateMetrics(num_requests=0, num_successful=0, num_failed=0, duration_s=0.0)
    assert agg.num_requests == 0


def test_aggregate_negative_output_tokens_rejected() -> None:
    with pytest.raises(ValidationError):
        AggregateMetrics(
            num_requests=1, num_successful=1, num_failed=0, duration_s=1.0,
            total_output_tokens=-1,
        )


def test_aggregate_negative_duration_rejected() -> None:
    with pytest.raises(ValidationError):
        AggregateMetrics(num_requests=0, num_successful=0, num_failed=0, duration_s=-1.0)


# --------------------------------------------------------------------------- #
# ExperimentResult status <-> aggregates/failure/timestamps
# --------------------------------------------------------------------------- #


def test_completed_without_aggregates_rejected() -> None:
    with pytest.raises(ValidationError, match="COMPLETED result must have aggregates"):
        ExperimentResult(
            config=_config(), environment=_env(),
            status=ExperimentStatus.COMPLETED, aggregates=None,
        )


def test_completed_with_failure_rejected() -> None:
    with pytest.raises(ValidationError, match="must not carry a failure"):
        ExperimentResult(
            config=_config(), environment=_env(),
            status=ExperimentStatus.COMPLETED, aggregates=_ok_aggregates(),
            failure=_failure(ExperimentStatus.COMPLETED),
        )


@pytest.mark.parametrize("status", [ExperimentStatus.FAILED, ExperimentStatus.OOM, ExperimentStatus.TIMEOUT])
def test_failing_status_without_failure_rejected(status: ExperimentStatus) -> None:
    with pytest.raises(ValidationError, match="must carry a failure record"):
        ExperimentResult(
            config=_config(), environment=_env(), status=status, failure=None,
        )


def test_failure_status_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="must equal result"):
        ExperimentResult(
            config=_config(), environment=_env(),
            status=ExperimentStatus.OOM, failure=_failure(ExperimentStatus.FAILED),
        )


@pytest.mark.parametrize("status", [ExperimentStatus.PENDING, ExperimentStatus.RUNNING])
def test_non_terminal_with_aggregates_rejected(status: ExperimentStatus) -> None:
    with pytest.raises(ValidationError, match="must not carry aggregates"):
        ExperimentResult(
            config=_config(), environment=_env(), status=status,
            aggregates=_ok_aggregates(),
        )


@pytest.mark.parametrize("status", [ExperimentStatus.PENDING, ExperimentStatus.RUNNING])
def test_non_terminal_with_failure_rejected(status: ExperimentStatus) -> None:
    with pytest.raises(ValidationError, match="must not carry a failure record"):
        ExperimentResult(
            config=_config(), environment=_env(), status=status,
            failure=_failure(status),
        )


def test_finished_before_started_rejected() -> None:
    with pytest.raises(ValidationError, match="finished_at must be >= started_at"):
        ExperimentResult(
            config=_config(), environment=_env(),
            status=ExperimentStatus.COMPLETED, aggregates=_ok_aggregates(),
            started_at=datetime(2026, 9, 14, 12, 0, 5, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc),
        )


# --------------------------------------------------------------------------- #
# Schema version enforcement
# --------------------------------------------------------------------------- #


def test_config_unknown_schema_version_rejected() -> None:
    raw = _config().model_dump()
    raw["schema_version"] = "9.9.9"
    with pytest.raises(ValidationError, match="unsupported schema_version"):
        ExperimentConfig.model_validate(raw)


def test_result_unknown_schema_version_rejected() -> None:
    result = ExperimentResult(
        config=_config(), environment=_env(),
        status=ExperimentStatus.COMPLETED, aggregates=_ok_aggregates(),
    )
    raw = result.model_dump()
    raw["schema_version"] = "0.0.1"
    with pytest.raises(ValidationError, match="unsupported schema_version"):
        ExperimentResult.model_validate(raw)


def test_known_schema_version_accepted() -> None:
    cfg = _config()
    assert ExperimentConfig.model_validate(cfg.model_dump()).schema_version == cfg.schema_version
