"""Optimization gate: explore a lever only in a winnable regime, else keep the default."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot.advisor import OptimizationPlan, plan_from_diagnosis
from inferpilot.diagnosis import BottleneckDiagnosis


def _diag(*, gpu, kv, sat, decode=True):
    from inferpilot.diagnosis import _classify
    regime, lever, pre, effect = _classify(gpu, kv, sat, decode)
    return BottleneckDiagnosis(
        gpu_utilization_mean_pct=gpu, kv_cache_usage_peak_perc=kv, saturated=sat,
        decode_heavy=decode, regime=regime, recommended_lever=lever,
        lever_preconditions=pre, predicted_effect=effect,
    )


def test_compute_bound_abstains() -> None:
    plan = plan_from_diagnosis(_diag(gpu=99, kv=1.0, sat=True))
    assert not plan.should_explore and plan.lever == "none"
    assert "keep default" in plan.rationale


def test_kv_capacity_bound_explores_fp8() -> None:
    plan = plan_from_diagnosis(_diag(gpu=60, kv=0.99, sat=True))
    assert plan.should_explore and plan.lever == "kv_cache_dtype=fp8"
    assert "explore" in plan.rationale


def test_underutilized_abstains() -> None:
    plan = plan_from_diagnosis(_diag(gpu=100, kv=0.5, sat=False))
    assert not plan.should_explore


def test_plan_roundtrips_and_rejects_tampering() -> None:
    plan = plan_from_diagnosis(_diag(gpu=99, kv=1.0, sat=True))
    assert OptimizationPlan.model_validate_json(plan.model_dump_json()) == plan
    raw = plan.model_dump(mode="json")
    raw["should_explore"] = True  # contradicts a compute_bound/none diagnosis
    with pytest.raises(ValidationError, match="inconsistent with its diagnosis"):
        OptimizationPlan.model_validate(raw)
