"""Synthetic, explicitly scoped windows. These are not production telemetry adapters."""

import hashlib
import math

import pytest
from pydantic import ValidationError

from inferpilot import (
    EffectiveConfig, EngineConfig, EnvironmentMetadata, ExperimentConfig,
    ExperimentResult, ExperimentStatus, HardwareInfo, RequestMeasurement,
    ResourceTelemetry, WorkloadSpec,
)
from inferpilot.runner.aggregate import compute_aggregates
from inferpilot.saturation import (
    LoadAssessment, LoadEvidence, assess_load_state, detect_saturation, measurement_digest,
)


def make_result(*, ttft=100.0, tpot=20.0, rate=2.0, count=200, seqs=4,
                saturating=False, gpu_mean=99.0, kv_peak=0.5, preemptions=None,
                prompt=128, out=32, failed=()):
    rows = []
    for i in range(count):
        start = i / rate
        # Deliberately FLAT TTFT while decode work backs up.
        latency = (ttft + (out - 1) * tpot) / 1000 + (i * 0.25 if saturating else 0)
        success = i not in failed
        rows.append(RequestMeasurement(
            request_id=str(i), prompt_tokens=prompt, output_tokens=out if success else 0,
            start_time_s=start, end_time_s=start + latency, success=success,
            ttft_ms=ttft if success else None,
            tpot_ms=(latency * 1000 - ttft) / (out - 1) if success else None,
            e2e_latency_ms=latency * 1000, error=None if success else "timeout",
        ))
    duration = max(m.end_time_s for m in rows)
    agg = compute_aggregates(rows, duration).model_copy(update={"gpu_memory_peak_mb": 1024})
    return ExperimentResult(
        config=ExperimentConfig(
            experiment_id="synthetic", name="synthetic",
            engine=EngineConfig(model="q", revision="abc", max_num_seqs=seqs, max_num_batched_tokens=512),
            workload=WorkloadSpec(name="synthetic", num_requests=count, prompt_tokens=prompt,
                                  output_tokens=out, request_rate_qps=rate),
        ),
        environment=EnvironmentMetadata(hardware=HardwareInfo(gpu_name="g")),
        status=ExperimentStatus.COMPLETED, measurements=rows, aggregates=agg,
        effective_config=EffectiveConfig(model="q", revision="abc", max_num_seqs=seqs,
                                         max_num_batched_tokens=512, verified=True),
        telemetry=ResourceTelemetry(sample_interval_s=0.25, num_samples=400, peak_gpu_memory_mb=1024,
            gpu_utilization_mean_pct=gpu_mean, gpu_utilization_peak_pct=gpu_mean,
            kv_cache_usage_mean_perc=kv_peak, kv_cache_usage_peak_perc=kv_peak,
            preemptions_total=preemptions),
    )


def make_evidence(result, **overrides):
    """Known fixed-length synthetic work; accepted outputs counted at completion.

    Do NOT copy this adapter to production: production needs independently measured
    demand/delivery and queue observations plus a certified complete census.
    """
    ts = overrides.pop("boundaries_s", [10.0, 30.0, 50.0, 70.0, 90.0])
    rows, t = result.measurements, result.telemetry
    values = dict(
        experiment_id=result.config.experiment_id, measurement_sha256=measurement_digest(rows),
        replay_sha256=hashlib.sha256(repr((result.config.workload.request_rate_qps,
            result.config.workload.num_requests, result.config.workload.prompt_tokens,
            result.config.workload.output_tokens)).encode()).hexdigest(),
        source="synthetic-test-fixture", boundaries_s=ts, coverage_complete=True, steady_state=True,
        waiting_requests=[0] * len(ts),
        offered_output_tokens=[sum(result.config.workload.output_tokens for m in rows if a <= m.start_time_s < b)
                               for a, b in zip(ts, ts[1:])],
        delivered_output_tokens=[sum(m.output_tokens for m in rows if m.success and a <= m.end_time_s < b)
                                 for a, b in zip(ts, ts[1:])],
        gpu_utilization_mean_pct=t.gpu_utilization_mean_pct if t else None,
        kv_cache_usage_peak_perc=t.kv_cache_usage_peak_perc if t else None,
        preemptions=t.preemptions_total if t else None,
    )
    values.update(overrides)
    return LoadEvidence(**values)


def test_old_results_are_indeterminate_even_with_flat_ttft():
    r = make_result()
    assert not detect_saturation(r.measurements).saturated
    a = assess_load_state(r.measurements)
    assert a.state == "indeterminate"
    assert a.duration_s is None and a.failure_fraction is None


def test_complete_balanced_window_is_healthy_not_a_capacity_claim():
    r = make_result()
    a = assess_load_state(r.measurements, evidence=make_evidence(r))
    assert a.state == "healthy"
    assert a.arrivals == [40] * 4 and a.successful_exits == [40] * 4
    assert a.backlog_slope_rps == 0
    assert "not_a_headroom" in a.reasons[0]


def test_decode_backlog_detected_despite_flat_ttft_and_drain_tail():
    r = make_result(saturating=True)
    assert not detect_saturation(r.measurements).saturated
    a = assess_load_state(r.measurements, evidence=make_evidence(r))
    assert a.state == "overloaded" and a.backlog_slope_rps > 0
    assert "persistent_request_backlog_growth" in a.reasons


def test_preexisting_stable_queue_is_not_called_healthy():
    r = make_result()
    a = assess_load_state(r.measurements, evidence=make_evidence(r, waiting_requests=[1] * 5))
    assert a.state == "near_capacity" and a.backlog_slope_rps == 0


@pytest.mark.parametrize("patch", [
    {"coverage_complete": False}, {"steady_state": False},
    {"waiting_requests": None}, {"offered_output_tokens": None},
    {"delivered_output_tokens": None},
    {"boundaries_s": [10, 15, 20, 25, 30]},
])
def test_missing_or_short_evidence_cannot_certify_health(patch):
    r = make_result()
    assert assess_load_state(r.measurements, evidence=make_evidence(r, **patch)).state == "indeterminate"


def test_request_overload_does_not_require_token_telemetry():
    r = make_result(saturating=True)
    e = make_evidence(r, delivered_output_tokens=None, waiting_requests=None)
    assert assess_load_state(r.measurements, evidence=e).state == "overloaded"


def test_token_work_deficit_can_veto_balanced_request_counts():
    r = make_result()
    e = make_evidence(r, delivered_output_tokens=[640] * 4)
    a = assess_load_state(r.measurements, evidence=e)
    assert a.arrivals == a.successful_exits
    assert a.state == "overloaded" and a.reasons == ["persistent_useful_output_work_deficit"]


def test_failures_and_timeouts_are_not_dropped_or_assumed_capacity_failures():
    r = make_result(failed=range(20, 180, 4))
    ids = [m.request_id for m in r.measurements if not m.success]
    e = make_evidence(r, timeout_request_ids=ids, delivered_output_tokens=None)
    a = assess_load_state(r.measurements, evidence=e)
    assert a.state == "indeterminate" and a.failure_fraction == pytest.approx(0.25)
    assert a.timeout_exits == sum(a.failed_exits) == 40
    assert sum(a.arrivals) == sum(a.successful_exits) + sum(a.failed_exits)


def test_failure_after_window_is_still_in_arrival_cohort():
    r = make_result(failed=[174, 175, 176])
    rows = [m.model_copy(update={"end_time_s": 100.0, "e2e_latency_ms": (100 - m.start_time_s) * 1000})
            if not m.success else m for m in r.measurements]
    r = r.model_copy(update={"measurements": rows})
    a = assess_load_state(rows, evidence=make_evidence(r, delivered_output_tokens=None))
    assert a.state == "indeterminate" and a.arrival_failures == 3


def test_failure_induced_token_deficit_alone_is_not_capacity_evidence():
    r = make_result(failed=range(200))
    a = assess_load_state(r.measurements, evidence=make_evidence(r))
    assert a.state == "indeterminate" and a.failure_fraction == 1


def test_draining_prewindow_backlog_does_not_certify_steady_capacity():
    r = make_result()
    extra = [RequestMeasurement(request_id=f"carry-{i}", prompt_tokens=128, output_tokens=32,
        start_time_s=0, end_time_s=12, ttft_ms=100, tpot_ms=11900 / 31,
        e2e_latency_ms=12000, success=True) for i in range(10)]
    r = r.model_copy(update={"measurements": r.measurements + extra})
    a = assess_load_state(r.measurements, evidence=make_evidence(r))
    assert a.state == "indeterminate" and "prior_backlog_is_draining" in a.reasons[0]


def test_all_completed_requests_can_fail_without_crashing_missing_evidence_path():
    assert assess_load_state(make_result(failed=range(200)).measurements).state == "indeterminate"


def test_reordering_rows_is_invariant_and_report_roundtrips():
    r = make_result()
    e = make_evidence(r)
    a = assess_load_state(r.measurements, evidence=e)
    assert assess_load_state(list(reversed(r.measurements)), evidence=e) == a
    assert LoadAssessment.model_validate_json(a.model_dump_json()) == a
    raw = a.model_dump(mode="json")
    raw["state"] = "overloaded"
    with pytest.raises(ValidationError, match="inconsistent"):
        LoadAssessment.model_validate(raw)


def test_boundary_conservation_and_fail_closed_validation():
    r = make_result(ttft=1000, tpot=0)
    a = assess_load_state(r.measurements, evidence=make_evidence(r))
    assert a.state == "healthy"
    for i in range(4):
        assert a.inflight[i + 1] - a.inflight[i] == a.arrivals[i] - a.successful_exits[i] - a.failed_exits[i]
    raw = a.model_dump(mode="json")
    raw["inflight"][1] += 1
    with pytest.raises(ValidationError, match="conservation"):
        LoadAssessment.model_validate(raw)


@pytest.mark.parametrize("patch", [
    {"boundaries_s": [10, 30, 20, 70, 90]},
    {"boundaries_s": [10, 30, 50, 70, math.inf]},
    {"waiting_requests": [-1] * 5}, {"delivered_output_tokens": [1]},
])
def test_invalid_evidence_rejected(patch):
    with pytest.raises(ValidationError):
        make_evidence(make_result(), **patch)


def test_wrong_census_duplicate_ids_and_wrong_timeout_ids_rejected():
    r = make_result()
    e = make_evidence(r)
    with pytest.raises(ValueError, match="does not match"):
        assess_load_state(r.measurements[:-1], evidence=e)
    with pytest.raises(ValueError, match="duplicate"):
        assess_load_state(r.measurements + [r.measurements[0]], evidence=e)
    with pytest.raises(ValueError, match="timeout IDs"):
        assess_load_state(r.measurements, evidence=make_evidence(r, timeout_request_ids=["0"]))
    with pytest.raises(ValueError, match="waiting requests exceed"):
        assess_load_state(r.measurements, evidence=make_evidence(r, waiting_requests=[999] * 5))
    with pytest.raises(ValueError, match="delivery exceeds"):
        assess_load_state(r.measurements, evidence=make_evidence(r, delivered_output_tokens=[1000000] * 4))


def test_zero_ttft_legacy_report_is_json_safe_and_not_a_health_signal():
    r = make_result(ttft=0)
    report = detect_saturation(r.measurements)
    assert report.reason == "nonpositive_baseline_ttft" and report.growth_ratio is None
    assert "Infinity" not in report.model_dump_json()
    with pytest.raises(ValueError, match="min_samples"):
        detect_saturation(r.measurements, min_samples=1)
