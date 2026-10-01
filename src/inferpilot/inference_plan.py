"""The engineer's decision, automated: given where you are and where you need to be,
what should you actually do?

An inference engineer with a capacity frontier in hand still has to reason:
"my ceiling is 5 QPS, I need 8, the wall is KV preemption, so try fp8 first, and if
that is not enough, scale out — and keep an eye on cost." This module encodes that
reasoning into an ordered, honest plan.

It is deliberately conservative, and it only ever recommends a next action it can
justify from measured evidence:

  * It will not plan at all without a bracketed ceiling — the first step is to find it.
  * It recommends a lever only when the forecast says the wall is something that lever
    relieves, and always labels the lever as "measure to confirm", never as a done deal.
  * When no lever addresses the wall, it recommends scaling, with a GPU count that is an
    explicit linear *estimate* to be confirmed by re-measurement — not a promise.
  * It separates "meets your QPS" from "meets your budget", because a config can do one
    and not the other.

The plan recomputes its steps from the embedded frontier and target on load, so a
tampered plan is rejected.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .capacity_frontier import CapacityFrontier
from .lever_forecast import _wall_point, forecast_lever

PlanVerdict = Literal[
    "meets_target_now",       # ceiling already clears the target QPS (and budget, if given)
    "lever_then_confirm",     # a lever likely raises the ceiling to target; measure to confirm
    "scale_out",              # no lever addresses the wall; add GPUs (estimate, confirm)
    "reduce_cost_via_lever",  # meets QPS but over budget; a throughput lever should cut $/token
    "over_budget_no_lever",   # meets QPS, over budget, and no lever helps; renegotiate or change GPU
    "collect_evidence",       # ceiling pinned but the wall has no diagnosis to plan from
    "measure_first",          # no ceiling yet — find it before planning
]

Action = Literal["measure_ceiling", "apply_lever", "scale_out", "reduce_cost", "accept", "collect_evidence"]


class PlanStep(SchemaModel):
    order: int = Field(ge=1)
    action: Action
    detail: str
    confirm: Optional[str] = None


def _meets(ceiling: float, target: float) -> bool:
    return ceiling >= target


def _applicable_lever(frontier: CapacityFrontier, levers: list[str]) -> Optional[tuple[str, Optional[float]]]:
    for lever in levers:
        fc = forecast_lever(lever, frontier)
        if fc.verdict == "worth_testing":
            return lever, fc.suggested_probe_qps
    return None


def _plan(
    frontier: CapacityFrontier, target_qps: float,
    budget: Optional[float], levers: list[str],
) -> tuple[PlanVerdict, list[PlanStep], list[str]]:
    if frontier.verdict != "ceiling_bracketed":
        return "measure_first", [PlanStep(
            order=1, action="measure_ceiling",
            detail=f"No capacity ceiling pinned yet ({frontier.verdict}).",
            confirm=(f"Measure at ~{frontier.suggested_next_qps:g} QPS and re-run."
                     if frontier.suggested_next_qps else "Add a rate that brackets the SLO boundary."),
        )], [f"frontier_{frontier.verdict}"]

    ceiling = frontier.ceiling_qps
    assert ceiling is not None
    cost = frontier.cost_per_million_output_tokens_usd

    if _meets(ceiling, target_qps):
        if budget is not None and cost is not None and cost > budget:
            found = _applicable_lever(frontier, levers)
            if found is not None:
                lever, probe = found
                return "reduce_cost_via_lever", [PlanStep(
                    order=1, action="reduce_cost",
                    detail=(f"Meets {target_qps:g} QPS, but ${cost:g}/1M tokens is over the "
                            f"${budget:g} budget. The wall is lever-relievable, and relieving it "
                            f"raises throughput, which lowers $/token."),
                    confirm=f"Sweep with {lever} (bracket ~{probe:g} QPS) and compare cost/token.",
                )], [f"ceiling_{ceiling:g}_meets_target", f"cost_{cost:g}_over_budget_{budget:g}"]
            return "over_budget_no_lever", [PlanStep(
                order=1, action="accept",
                detail=(f"Meets {target_qps:g} QPS but ${cost:g}/1M tokens is over the ${budget:g} "
                        f"budget, and no modeled lever addresses the wall."),
                confirm="Consider a cheaper GPU/precision, or renegotiate the budget.",
            )], [f"ceiling_{ceiling:g}_meets_target", "no_lever_for_cost"]
        return "meets_target_now", [PlanStep(
            order=1, action="accept",
            detail=(f"Current config sustains {target_qps:g} QPS under the SLO "
                    f"(ceiling ~{ceiling:g} QPS)"
                    + (f" at ~${cost:g}/1M output tokens." if cost is not None else ".")),
            confirm="Keep the incumbent; re-check if the workload shape changes.",
        )], [f"ceiling_{ceiling:g}_meets_target_{target_qps:g}"]

    # Ceiling is below target: need more capacity.
    found = _applicable_lever(frontier, levers)
    if found is not None:
        lever, probe = found
        gpus = math.ceil((target_qps / ceiling) * frontier.gpu_count)
        return "lever_then_confirm", [PlanStep(
            order=1, action="apply_lever",
            detail=(f"Ceiling ~{ceiling:g} QPS is below the {target_qps:g} QPS target, and the wall "
                    f"is relievable by {lever}."),
            confirm=f"Sweep with {lever} (bracket ~{probe:g} QPS), then compare_lever against the target.",
        )], [f"ceiling_{ceiling:g}_below_target_{target_qps:g}", f"lever_{lever}_worth_testing",
             f"if_still_short_scale_toward_~{gpus}_gpus"]

    wall = _wall_point(frontier)
    if wall is None or wall.regime is None or wall.regime == "unknown":
        return "collect_evidence", [PlanStep(
            order=1, action="collect_evidence",
            detail=(f"Ceiling ~{ceiling:g} QPS is below target, but the limiting run carries no "
                    f"bottleneck diagnosis to plan from."),
            confirm="Attach aligned load evidence at the wall and re-diagnose.",
        )], [f"ceiling_{ceiling:g}_below_target_{target_qps:g}", "wall_undiagnosed"]

    gpus = math.ceil((target_qps / ceiling) * frontier.gpu_count)
    return "scale_out", [PlanStep(
        order=1, action="scale_out",
        detail=(f"Ceiling ~{ceiling:g} QPS is below the {target_qps:g} QPS target, the wall is "
                f"'{wall.regime}', and no modeled lever addresses it. Scaling is the lever."),
        confirm=(f"Linear estimate: ~{gpus} GPU(s) (from {frontier.gpu_count}). This assumes near-linear "
                 f"scaling — confirm by measuring a {gpus}-GPU deployment."),
    )], [f"ceiling_{ceiling:g}_below_target_{target_qps:g}", f"wall_{wall.regime}_no_lever",
         f"linear_estimate_~{gpus}_gpus"]


class DeploymentPlan(SchemaModel):
    """Self-validating ordered plan from a capacity frontier to a target."""

    frontier: CapacityFrontier
    target_qps: float = Field(gt=0, allow_inf_nan=False)
    budget_per_million_output_tokens_usd: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    available_levers: list[str] = Field(default_factory=lambda: ["kv_cache_dtype=fp8"])
    verdict: PlanVerdict
    steps: list[PlanStep]
    reasons: list[str]

    @model_validator(mode="after")
    def _check(self) -> "DeploymentPlan":
        verdict, steps, reasons = _plan(
            self.frontier, self.target_qps,
            self.budget_per_million_output_tokens_usd, self.available_levers,
        )
        if (self.verdict, self.steps, self.reasons) != (verdict, steps, reasons):
            raise ValueError("deployment plan is inconsistent with its frontier and target")
        return self


def plan_to_target(
    frontier: CapacityFrontier, target_qps: float, *,
    budget_per_million_output_tokens_usd: Optional[float] = None,
    available_levers: Optional[list[str]] = None,
) -> DeploymentPlan:
    """Automate the engineer's next move: given a capacity frontier and a target QPS
    (and optional $/token budget), emit the ordered, honest action plan."""
    levers = available_levers if available_levers is not None else ["kv_cache_dtype=fp8"]
    verdict, steps, reasons = _plan(frontier, target_qps, budget_per_million_output_tokens_usd, levers)
    return DeploymentPlan(
        frontier=frontier, target_qps=target_qps,
        budget_per_million_output_tokens_usd=budget_per_million_output_tokens_usd,
        available_levers=levers, verdict=verdict, steps=steps, reasons=reasons,
    )
