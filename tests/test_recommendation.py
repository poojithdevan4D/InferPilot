"""Unified InferPilot recommendation entry point + mechanistic action log."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import SLO, ModelFootprint
from inferpilot.advisor import InferPilotRecommendation, OperatorEconomics, recommend_plan

from test_capacity_advisory import _result

SLO_ = SLO(ttft_p95_ms=2000, tpot_p95_ms=60)
Q14B = ModelFootprint(model="Qwen2.5-14B", params_billions=14.77, num_layers=48,
                      num_kv_heads=8, head_dim=128, weight_dtype="bf16")


def test_scale_recommendation_includes_structural_option() -> None:
    r = _result(gpu_mean=99, kv_peak=0.5, saturating=True, prompt=2048, out=512, throughput=0.5)
    econ = OperatorEconomics(gpu_cost_per_hour_usd=2.10, gpu_count=1, target_qps=1.0)
    rec = recommend_plan(r, SLO_, econ, footprint=Q14B)
    assert rec.advisory.action == "scale"
    assert "DIAGNOSIS:" in rec.action_log and "STRUCTURAL:" in rec.action_log
    assert rec.scale is not None


def test_tune_recommendation_lists_preconditions() -> None:
    r = _result(gpu_mean=60, kv_peak=0.99, saturating=True, prompt=128, out=512)
    econ = OperatorEconomics(gpu_cost_per_hour_usd=2.10, gpu_count=1)
    rec = recommend_plan(r, SLO_, econ)
    assert rec.advisory.action == "tune"
    assert "VERIFY FIRST:" in rec.action_log and "kv_cache_dtype=fp8" in rec.action_log


def test_roundtrip_and_tamper() -> None:
    r = _result(gpu_mean=99, kv_peak=0.5, saturating=True, throughput=0.5)
    econ = OperatorEconomics(gpu_cost_per_hour_usd=2.10, gpu_count=1, target_qps=1.0)
    rec = recommend_plan(r, SLO_, econ, footprint=Q14B)
    assert InferPilotRecommendation.model_validate_json(rec.model_dump_json()) == rec
    raw = rec.model_dump(mode="json")
    raw["action_log"] = "tampered"
    with pytest.raises(ValidationError, match="action log is inconsistent"):
        InferPilotRecommendation.model_validate(raw)
