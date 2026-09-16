from copy import deepcopy

import pytest
from pydantic import ValidationError

from inferpilot import (
    AggregateMetrics,
    EffectiveConfig,
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    HardwareInfo,
    RequestMeasurement,
    ResourceTelemetry,
    WorkloadObservation,
    WorkloadSpec,
    build_workload_profile,
)
from inferpilot.advisor import (
    AdvisorPolicy,
    CanaryEvaluation,
    CanarySpec,
    ControllerEvent,
    ControllerSpec,
    ControllerState,
    ProfileContext,
    advance_controller,
    advise_from_profile,
    evaluate_canary,
)


def _policy() -> AdvisorPolicy:
    return AdvisorPolicy(
        policy_version="0.2.0", policy_id="m3", model="q", revision="abc",
        gpu_name="g", prompt_tokens=128, output_tokens=32,
        arrival_pattern="poisson-v1",
        fixed_engine_fields={"max_num_batched_tokens": 512},
        regimes=[
            {"request_rate_qps": 2, "min_rate_qps": 1.5, "max_rate_qps": 2.5,
             "engine_overrides": {"max_num_seqs": 2}},
            {"request_rate_qps": 6, "min_rate_qps": 5.5, "max_rate_qps": 6.5,
             "engine_overrides": {"max_num_seqs": 4}},
        ],
        evidence_report_sha256="a" * 64,
        applicability_report_sha256="b" * 64,
    )


def _decision():
    profile = build_workload_profile([
        WorkloadObservation(
            arrival_offset_s=i / 6, prompt_tokens=128, output_tokens=32, success=True
        )
        for i in range(5)
    ])
    return advise_from_profile(
        _policy(), profile,
        ProfileContext(model="q", revision="abc", gpu_name="g",
                       arrival_pattern="poisson-v1", declared_fixed_length=True),
    )


def _result(ttft: float = 100, tpot: float = 8) -> ExperimentResult:
    count = 4
    e2e = ttft + 31 * tpot
    return ExperimentResult(
        config=ExperimentConfig(
            experiment_id="canary", name="canary",
            engine=EngineConfig(
                model="q", revision="abc", max_num_seqs=4,
                max_num_batched_tokens=512,
            ),
            workload=WorkloadSpec(
                name="canary", num_requests=count, prompt_tokens=128, output_tokens=32,
                request_rate_qps=6,
            ),
        ),
        environment=EnvironmentMetadata(hardware=HardwareInfo(gpu_name="g")),
        status=ExperimentStatus.COMPLETED,
        measurements=[
            RequestMeasurement(
                request_id=str(i), prompt_tokens=128, output_tokens=32,
                start_time_s=i * .25, end_time_s=i * .25 + e2e / 1000,
                ttft_ms=ttft, tpot_ms=tpot, e2e_latency_ms=e2e,
            )
            for i in range(count)
        ],
        aggregates=AggregateMetrics(
            num_requests=count, num_successful=count, num_failed=0, duration_s=2,
            ttft_p50_ms=ttft, ttft_p95_ms=ttft, ttft_p99_ms=ttft,
            tpot_p50_ms=tpot, tpot_p95_ms=tpot, tpot_p99_ms=tpot,
            e2e_p50_ms=e2e, e2e_p95_ms=e2e, e2e_p99_ms=e2e,
            throughput_tokens_per_s=64, throughput_requests_per_s=2,
            total_output_tokens=128, gpu_memory_peak_mb=1024,
        ),
        effective_config=EffectiveConfig(
            model="q", revision="abc", max_num_seqs=4,
            max_num_batched_tokens=512, verified=True,
        ),
        telemetry=ResourceTelemetry(
            sample_interval_s=.25, num_samples=4, peak_gpu_memory_mb=1024,
            gpu_utilization_mean_pct=50, gpu_utilization_peak_pct=75,
            kv_cache_usage_mean_perc=.1, kv_cache_usage_peak_perc=.2,
        ),
    )


def _spec() -> CanarySpec:
    return CanarySpec(ttft_p95_limit_ms=250, tpot_p95_limit_ms=8.5)


def test_measured_canary_passes_and_roundtrips() -> None:
    evaluation = evaluate_canary(_spec(), _decision(), _result())
    assert evaluation.passed and evaluation.reasons == []
    assert CanaryEvaluation.model_validate_json(evaluation.model_dump_json()) == evaluation


@pytest.mark.parametrize(
    "result,reason",
    [
        (_result(ttft=251), "ttft_p95_slo_failed"),
        (_result(tpot=8.6), "tpot_p95_slo_failed"),
    ],
)
def test_slo_failure_is_derived(result, reason) -> None:
    evaluation = evaluate_canary(_spec(), _decision(), result)
    assert not evaluation.passed and reason in evaluation.reasons


def test_candidate_and_context_mismatches_fail() -> None:
    result = _result()
    result.config.engine.max_num_seqs = 3
    result.environment.hardware.gpu_name = "other"
    evaluation = evaluate_canary(_spec(), _decision(), result)
    assert not evaluation.passed
    assert "requested_max_num_seqs_mismatch" in evaluation.reasons
    assert "gpu_mismatch" in evaluation.reasons


def test_tampered_evaluation_is_rejected() -> None:
    raw = evaluate_canary(_spec(), _decision(), _result()).model_dump(mode="json")
    raw["passed"] = False
    raw["reasons"] = ["invented"]
    with pytest.raises(ValidationError, match="inconsistent"):
        CanaryEvaluation.model_validate(raw)


def test_aggregate_tampering_is_detected_from_raw_measurements() -> None:
    result = _result()
    result.aggregates.ttft_p95_ms = 1
    evaluation = evaluate_canary(_spec(), _decision(), result)
    assert not evaluation.passed
    assert "aggregates_inconsistent_with_measurements" in evaluation.reasons


def test_controller_v2_requires_measured_canary_evidence() -> None:
    spec = ControllerSpec(
        controller_version="0.2.0", advisor_policy=_policy(),
        min_consecutive_windows=1, canary_spec=_spec(),
    )
    initial = ControllerState(
        current_engine_overrides={"max_num_batched_tokens": 512, "max_num_seqs": 2}
    )
    test = advance_controller(
        spec, initial,
        ControllerEvent(event_version="0.2.0", event_type="observation", decision=_decision()),
    )
    assert test.action == "test_candidate"
    evaluation = evaluate_canary(_spec(), _decision(), _result())
    apply = advance_controller(
        spec, test.after,
        ControllerEvent(event_version="0.2.0", event_type="canary_result",
                        canary_evaluation=evaluation),
    )
    assert apply.action == "apply_candidate"
    with pytest.raises(ValidationError, match="measured canary evidence"):
        ControllerEvent(event_version="0.2.0", event_type="canary_result", canary_passed=True)


def test_controller_v2_rolls_back_from_measured_slo_failure() -> None:
    spec = ControllerSpec(
        controller_version="0.2.0", advisor_policy=_policy(),
        min_consecutive_windows=1, canary_spec=_spec(),
    )
    initial = ControllerState(
        current_engine_overrides={"max_num_batched_tokens": 512, "max_num_seqs": 2}
    )
    test = advance_controller(
        spec, initial,
        ControllerEvent(event_version="0.2.0", event_type="observation", decision=_decision()),
    )
    evaluation = evaluate_canary(_spec(), _decision(), _result(tpot=9))
    rolled_back = advance_controller(
        spec, test.after,
        ControllerEvent(event_version="0.2.0", event_type="canary_result",
                        canary_evaluation=evaluation),
    )
    assert rolled_back.action == "rollback"
    assert "tpot_p95_slo_failed" in rolled_back.reasons
    assert rolled_back.after.current_engine_overrides == initial.current_engine_overrides


def test_v1_boolean_canary_remains_read_compatible() -> None:
    spec = ControllerSpec(controller_version="0.1.0", advisor_policy=_policy(), min_consecutive_windows=1)
    initial = ControllerState(
        current_engine_overrides={"max_num_batched_tokens": 512, "max_num_seqs": 2}
    )
    test = advance_controller(spec, initial, ControllerEvent(event_version="0.1.0", event_type="observation", decision=_decision()))
    applied = advance_controller(
        spec, test.after, ControllerEvent(event_version="0.1.0", event_type="canary_result", canary_passed=True)
    )
    assert applied.action == "apply_candidate"
