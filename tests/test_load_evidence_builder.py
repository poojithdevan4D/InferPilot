"""Synthetic tests for runner-native aligned load evidence."""

from __future__ import annotations

from inferpilot import (
    AggregateMetrics, EffectiveConfig, EngineConfig, EnvironmentMetadata,
    ExperimentConfig, ExperimentResult, ExperimentStatus, HardwareInfo,
    RequestMeasurement, ResourceSample, ResourceTelemetry, WorkloadSpec, diagnose,
)
from inferpilot.runner.aggregate import compute_aggregates
from inferpilot.runner.load_evidence import build_load_evidence, intended_replay_digest
from inferpilot.saturation import LoadEvidence, assess_load_state, measurement_digest


def _measurements(*, overloaded: bool) -> list[RequestMeasurement]:
    rows = []
    for index in range(200):
        start = index / 2
        latency = 0.5 + (index * 0.25 if overloaded else 0)
        rows.append(
            RequestMeasurement(
                request_id=str(index),
                prompt_tokens=128,
                output_tokens=32,
                start_time_s=start,
                end_time_s=start + latency,
                ttft_ms=100,
                tpot_ms=(latency * 1000 - 100) / 31,
                e2e_latency_ms=latency * 1000,
                success=True,
            )
        )
    return rows


def _telemetry(waiting: int = 0) -> list[ResourceSample]:
    return [
        ResourceSample(
            t_s=float(second),
            gpu_utilization_pct=80,
            kv_cache_usage_perc=0.5,
            num_requests_waiting=waiting,
            num_requests_running=0,
            num_preemptions_total=float(second // 20),
        )
        for second in range(101)
    ]


def _build(rows, *, waiting: int = 0) -> LoadEvidence:
    return build_load_evidence(
        experiment_id="synthetic",
        measured_window_t0_s=1000,
        measured_window_end_s=1100,
        measurements=rows,
        telemetry_samples=_telemetry(waiting),
        requested_output_tokens=32,
        window_width_s=5,
        coverage_complete=True,
        steady_state=True,
    )


def test_builder_produces_valid_aligned_balanced_evidence() -> None:
    rows = _measurements(overloaded=False)
    evidence = _build(rows)
    assert LoadEvidence.model_validate_json(evidence.model_dump_json()) == evidence
    assert evidence.measurement_sha256 == measurement_digest(rows)
    assert len(evidence.boundaries_s) == 21
    assert evidence.waiting_requests == [0] * 21
    assert evidence.offered_output_tokens == [320] * 20
    assert evidence.preemptions == 5
    assessment = assess_load_state(rows, evidence=evidence)
    assert assessment.state == "healthy"
    for index in range(len(assessment.arrivals)):
        assert (
            assessment.inflight[index + 1] - assessment.inflight[index]
            == assessment.arrivals[index]
            - assessment.successful_exits[index]
            - assessment.failed_exits[index]
        )


def test_builder_exposes_persistent_backlog_as_overloaded() -> None:
    rows = _measurements(overloaded=True)
    assessment = assess_load_state(rows, evidence=_build(rows))
    assert assessment.state == "overloaded"
    assert "persistent_request_backlog_growth" in assessment.reasons


def test_missing_queue_metrics_abstains_from_health() -> None:
    rows = _measurements(overloaded=False)
    samples = [ResourceSample(t_s=float(second)) for second in range(101)]
    evidence = build_load_evidence(
        experiment_id="synthetic",
        measured_window_t0_s=0,
        measured_window_end_s=100,
        measurements=rows,
        telemetry_samples=samples,
        requested_output_tokens=32,
        coverage_complete=True,
        steady_state=True,
    )
    assert evidence.waiting_requests is None and evidence.preemptions is None
    assessment = assess_load_state(rows, evidence=evidence)
    assert assessment.state == "indeterminate"
    assert "missing_waiting_queue_samples" in assessment.reasons


def test_builder_caps_at_contract_maximum_and_rejects_short_window() -> None:
    rows = _measurements(overloaded=False)
    long = build_load_evidence(
        experiment_id="synthetic", measured_window_t0_s=0, measured_window_end_s=1000,
        measurements=rows, telemetry_samples=[], requested_output_tokens=32,
        coverage_complete=True, steady_state=True,
    )
    assert len(long.boundaries_s) == 65

    import pytest
    with pytest.raises(ValueError, match="at least four"):
        build_load_evidence(
            experiment_id="synthetic", measured_window_t0_s=0, measured_window_end_s=19,
            measurements=rows, telemetry_samples=[], requested_output_tokens=32,
            coverage_complete=True, steady_state=True,
        )


def test_queue_cross_check_fails_closed_on_contradictory_gauges() -> None:
    rows = _measurements(overloaded=False)
    evidence = _build(rows, waiting=999)
    assert evidence.waiting_requests is None
    assessment = assess_load_state(rows, evidence=evidence)
    assert assessment.state == "indeterminate"


def test_replay_digest_binds_inputs_but_not_engine_configuration() -> None:
    digest = intended_replay_digest(["a", "b"], [0.0, 0.5], 32)
    assert digest == intended_replay_digest(["a", "b"], [0.0, 0.5], 32)
    assert digest != intended_replay_digest(["a", "c"], [0.0, 0.5], 32)
    assert digest != intended_replay_digest(["a", "b"], [0.0, 0.6], 32)


def test_embedded_evidence_flows_through_result_and_diagnosis() -> None:
    rows = _measurements(overloaded=True)
    evidence = _build(rows)
    aggregate = compute_aggregates(rows, max(row.end_time_s for row in rows)).model_copy(
        update={"gpu_memory_peak_mb": 1024}
    )
    result = ExperimentResult(
        config=ExperimentConfig(
            experiment_id="synthetic", name="synthetic",
            engine=EngineConfig(model="q", revision="abc", max_num_seqs=4,
                                max_num_batched_tokens=512),
            workload=WorkloadSpec(name="synthetic", num_requests=200,
                                  prompt_tokens=128, output_tokens=32,
                                  request_rate_qps=2),
        ),
        environment=EnvironmentMetadata(hardware=HardwareInfo(gpu_name="g")),
        status=ExperimentStatus.COMPLETED,
        measurements=rows,
        aggregates=aggregate,
        effective_config=EffectiveConfig(verified=True),
        telemetry=ResourceTelemetry(
            sample_interval_s=.25, num_samples=400, peak_gpu_memory_mb=1024,
            gpu_utilization_mean_pct=80, gpu_utilization_peak_pct=90,
            kv_cache_usage_mean_perc=.5, kv_cache_usage_peak_perc=.5,
        ),
        load_evidence=evidence,
    )
    reloaded = ExperimentResult.model_validate_json(result.model_dump_json())
    assert reloaded.load_evidence == evidence
    assert diagnose(reloaded).load_assessment.state == "overloaded"
