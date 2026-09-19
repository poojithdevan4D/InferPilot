import pytest
from pydantic import ValidationError

from inferpilot import SLO
from inferpilot.advisor import CapacityAdvisory, OperatorEconomics, advise_capacity
from inferpilot.runner.aggregate import compute_aggregates
from test_load_state import make_evidence, make_result

_result = make_result
SLO_ = SLO(ttft_p95_ms=500, tpot_p95_ms=60)
ECON = OperatorEconomics(gpu_cost_per_hour_usd=2.10, gpu_count=1, target_qps=2)


def test_unknown_is_not_adequate_and_not_a_zero_capacity_estimate():
    a = advise_capacity(make_result(), SLO_, ECON)
    assert a.action == "abstain" and a.met_slo is None and a.goodput_qps is None
    assert a.cost_per_million_output_tokens_usd is None and a.gpus_needed_for_target is None


def test_healthy_cohort_has_measured_goodput_and_cost():
    r = make_result()
    a = advise_capacity(r, SLO_, ECON, load_evidence=make_evidence(r))
    assert a.action == "adequate" and a.met_slo is True and a.goodput_qps == 2
    assert a.cost_per_million_output_tokens_usd == pytest.approx(2.10 / (64 * 3600) * 1e6)
    assert a.gpus_needed_for_target is None


def test_nominal_qps_cannot_be_used_as_capacity():
    r = make_result()
    config = r.config.model_copy(update={"workload": r.config.workload.model_copy(update={"request_rate_qps": 4})})
    r = r.model_copy(update={"config": config})
    econ = OperatorEconomics(gpu_cost_per_hour_usd=2.10, target_qps=4)
    a = advise_capacity(r, SLO_, econ, load_evidence=make_evidence(r))
    assert a.offered_qps == 4 and a.goodput_qps == 2 and a.action == "abstain"
    assert a.gpus_needed_for_target is None


def test_overload_is_not_automatically_a_scaling_prescription():
    r = make_result(saturating=True)
    a = advise_capacity(r, SLO_, ECON, load_evidence=make_evidence(r))
    assert a.diagnosis.saturated and a.action == "abstain"
    assert a.goodput_qps is None and a.gpus_needed_for_target is None


def test_aligned_preemption_pressure_nominates_canary_without_false_cost():
    r = make_result(saturating=True, kv_peak=1, preemptions=2)
    a = advise_capacity(r, SLO_, ECON, load_evidence=make_evidence(r))
    assert a.action == "tune" and "kv_cache_dtype=fp8" in a.recommendation
    assert a.met_slo is None and a.cost_per_million_output_tokens_usd is None


def test_near_capacity_cannot_be_adequate_even_with_passing_latencies():
    r = make_result()
    a = advise_capacity(r, SLO_, ECON, load_evidence=make_evidence(r, waiting_requests=[1] * 5))
    assert a.action == "abstain" and a.met_slo is None


def test_slo_failure_is_separate_from_load_state():
    r = make_result(ttft=6000)
    a = advise_capacity(r, SLO_, ECON, load_evidence=make_evidence(r))
    assert a.diagnosis.load_state == "healthy" and a.met_slo is False
    assert a.goodput_qps == 0 and a.cost_per_million_output_tokens_usd is None
    assert a.action == "abstain"


@pytest.mark.parametrize("slo", [SLO(e2e_p95_ms=500), SLO(min_throughput_tokens_per_s=100)])
def test_all_existing_slo_fields_are_honored(slo):
    r = make_result()
    a = advise_capacity(r, slo, ECON, load_evidence=make_evidence(r))
    assert a.met_slo is False and a.action == "abstain"


def test_no_slo_is_unknown_not_an_automatic_pass():
    r = make_result()
    a = advise_capacity(r, SLO(), ECON, load_evidence=make_evidence(r))
    assert a.met_slo is None and a.action == "abstain"


def test_goodput_counts_passing_requests_not_whole_cohort_when_p95_passes():
    r = make_result(ttft=100)
    rows = [m.model_copy(update={"ttft_ms": 1000.0, "end_time_s": m.end_time_s + 0.9,
                                "e2e_latency_ms": m.e2e_latency_ms + 900})
            if int(m.request_id) % 20 == 0 else m for m in r.measurements]
    agg = compute_aggregates(rows, max(m.end_time_s for m in rows)).model_copy(update={"gpu_memory_peak_mb": 1024})
    r = r.model_copy(update={"measurements": rows, "aggregates": agg})
    a = advise_capacity(r, SLO_, ECON, load_evidence=make_evidence(r))
    assert a.met_slo is True and a.goodput_qps == pytest.approx(1.9)
    assert a.action == "abstain"  # 1.9 observed goodput is below the target of 2


def test_roundtrip_and_derived_field_tampering_rejected():
    r = make_result()
    a = advise_capacity(r, SLO_, ECON, load_evidence=make_evidence(r))
    assert CapacityAdvisory.model_validate_json(a.model_dump_json()) == a
    for key, value in (("goodput_qps", 100), ("met_slo", False),
                       ("cost_per_million_output_tokens_usd", 0), ("gpus_needed_for_target", 10)):
        raw = a.model_dump(mode="json")
        raw[key] = value
        with pytest.raises(ValidationError, match="inconsistent"):
            CapacityAdvisory.model_validate(raw)
