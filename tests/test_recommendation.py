import pytest
from pydantic import ValidationError

from inferpilot import SLO, ModelFootprint
from inferpilot.advisor import InferPilotRecommendation, OperatorEconomics, recommend_plan
from test_load_state import make_evidence, make_result

SLO_ = SLO(ttft_p95_ms=2000, tpot_p95_ms=60)
ECON = OperatorEconomics(gpu_cost_per_hour_usd=2.10, target_qps=2)
Q14B = ModelFootprint(model="Qwen2.5-14B", params_billions=14.77, num_layers=48,
                     num_kv_heads=8, head_dim=128, weight_dtype="bf16")


def test_missing_evidence_does_not_scale_and_renders_unknown_values():
    r = make_result().model_copy(update={"telemetry": None})
    rec = recommend_plan(r, SLO_, ECON, footprint=Q14B)
    assert rec.advisory.action == "abstain" and rec.scale is None
    assert "GOODPUT: unknown" in rec.action_log and "gpu_mean=unknown" in rec.action_log
    assert "STRUCTURAL:" not in rec.action_log


def test_tune_recommendation_lists_real_quality_preconditions():
    r = make_result(saturating=True, kv_peak=1, preemptions=2)
    rec = recommend_plan(r, SLO_, ECON, load_evidence=make_evidence(r))
    assert rec.advisory.action == "tune"
    assert "VERIFY FIRST:" in rec.action_log and "kv_cache_dtype=fp8" in rec.action_log
    assert "token agreement alone does not establish quality" in rec.action_log


def test_healthy_observed_recommendation_and_roundtrip_tamper():
    r = make_result()
    rec = recommend_plan(r, SLO_, ECON, load_evidence=make_evidence(r))
    assert rec.advisory.action == "adequate" and "not capacity" in rec.action_log
    assert InferPilotRecommendation.model_validate_json(rec.model_dump_json()) == rec
    raw = rec.model_dump(mode="json")
    raw["action_log"] = "tampered"
    with pytest.raises(ValidationError, match="action log is inconsistent"):
        InferPilotRecommendation.model_validate(raw)
