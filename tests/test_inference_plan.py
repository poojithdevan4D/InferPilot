"""The planner automates the engineer's next move from a capacity frontier + target,
and stays honest: it labels levers as 'confirm', scale as an estimate, and refuses to
plan without a ceiling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot.capacity_frontier import CapacityPoint, estimate_frontier
from inferpilot.config import SLO
from inferpilot.inference_plan import DeploymentPlan, plan_to_target
from inferpilot.report import render_plan

SLO_TTFT = SLO(ttft_p95_ms=250.0)
FP8 = "kv_cache_dtype=fp8"


def _pt(offered, *, margin, regime=None, lever=None, tokens_per_s=800.0):
    return CapacityPoint(offered_qps=offered, achieved_qps=offered, tokens_per_s=tokens_per_s,
                         overloaded=False, slo_margin=margin, regime=regime, recommended_lever=lever)


def _kv_frontier(**kw):  # wall diagnosed as KV preemption that fp8 relieves; ceiling ~4.4
    return estimate_frontier([_pt(4.0, margin=0.1), _pt(6.0, margin=-0.4, regime="kv_pressure", lever=FP8)],
                             SLO_TTFT, **kw)


def test_meets_target_now() -> None:
    plan = plan_to_target(_kv_frontier(), target_qps=3.0)
    assert plan.verdict == "meets_target_now"
    assert plan.steps[0].action == "accept"


def test_below_target_with_relievable_wall_tries_lever() -> None:
    plan = plan_to_target(_kv_frontier(), target_qps=8.0)
    assert plan.verdict == "lever_then_confirm"
    assert plan.steps[0].action == "apply_lever"
    assert any("if_still_short_scale" in r for r in plan.reasons)


def test_below_target_no_lever_scales_out_with_estimate() -> None:
    f = estimate_frontier([_pt(4.0, margin=0.1), _pt(6.0, margin=-0.4, regime="compute_bound", lever="none")],
                          SLO_TTFT)
    plan = plan_to_target(f, target_qps=8.0)
    assert plan.verdict == "scale_out"
    assert plan.steps[0].action == "scale_out"
    assert "Linear estimate" in (plan.steps[0].confirm or "")


def test_over_budget_but_meets_qps_recommends_cost_lever() -> None:
    f = _kv_frontier(gpu_cost_per_hour_usd=1.10)   # cost ~0.38/1M at 800 tok/s
    plan = plan_to_target(f, target_qps=3.0, budget_per_million_output_tokens_usd=0.10)
    assert plan.verdict == "reduce_cost_via_lever"
    assert plan.steps[0].action == "reduce_cost"


def test_no_ceiling_means_measure_first() -> None:
    f = estimate_frontier([_pt(2.0, margin=0.5), _pt(4.0, margin=0.3)], SLO_TTFT)  # all pass
    plan = plan_to_target(f, target_qps=8.0)
    assert plan.verdict == "measure_first"
    assert plan.steps[0].action == "measure_ceiling"


def test_undiagnosed_wall_collects_evidence() -> None:
    f = estimate_frontier([_pt(4.0, margin=0.1), _pt(6.0, margin=-0.4)], SLO_TTFT)  # no regime on wall
    plan = plan_to_target(f, target_qps=8.0)
    assert plan.verdict == "collect_evidence"


def test_plan_roundtrips_and_rejects_tampering() -> None:
    plan = plan_to_target(_kv_frontier(), target_qps=8.0)
    assert DeploymentPlan.model_validate_json(plan.model_dump_json()) == plan
    raw = plan.model_dump(mode="json")
    raw["verdict"] = "meets_target_now"
    with pytest.raises(ValidationError, match="inconsistent with its frontier and target"):
        DeploymentPlan.model_validate(raw)


def test_render_plan_reads_as_actions() -> None:
    out = render_plan(plan_to_target(_kv_frontier(), target_qps=8.0))
    assert "Plan to 8 QPS" in out
    assert "TRY A LEVER" in out
    assert "[apply_lever]" in out
