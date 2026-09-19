import pytest
from pydantic import ValidationError

from inferpilot.advisor import OptimizationPlan, plan_from_diagnosis, plan_optimization
from inferpilot.diagnosis import diagnose
from test_load_state import make_evidence, make_result


def test_old_result_abstains_without_inventing_a_lever():
    p = plan_optimization(make_result(gpu_mean=100, kv_peak=1, preemptions=10))
    assert not p.should_explore and p.lever == "none"
    assert p.baseline_diagnosis.regime == "unknown"


def test_healthy_or_near_capacity_does_not_trigger_exploration():
    r = make_result()
    for queue in ([0] * 5, [1] * 5):
        d = diagnose(r, load_evidence=make_evidence(r, waiting_requests=queue))
        assert not plan_from_diagnosis(d).should_explore


def test_aligned_kv_pressure_can_nominate_quality_gated_canary():
    r = make_result(saturating=True, kv_peak=1, preemptions=2)
    p = plan_from_diagnosis(diagnose(r, load_evidence=make_evidence(r)))
    assert p.should_explore and p.lever == "kv_cache_dtype=fp8"
    assert OptimizationPlan.model_validate_json(p.model_dump_json()) == p
    raw = p.model_dump(mode="json")
    raw["should_explore"] = False
    with pytest.raises(ValidationError, match="inconsistent"):
        OptimizationPlan.model_validate(raw)
