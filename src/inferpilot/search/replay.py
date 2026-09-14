"""Run fixed-budget, outcome-blind search baselines over blocked evidence."""

from __future__ import annotations

from typing import Optional

from ..comparison.models import BlockedStudyReport
from .models import (
    ReplayCostReport,
    ReplayCostTrial,
    ReplaySearchReport,
    ReplaySearchSpec,
    ReplayTrial,
)
from .policy import candidate_order


def _build_cost_overlay(
    order: list[int], candidate_costs_s: list[float], budget: int, cost_budget_s: Optional[float]
) -> ReplayCostReport:
    """Overlay realized per-trial cost onto the fixed policy order (no reordering)."""
    trials: list[ReplayCostTrial] = []
    cumulative = 0.0
    for number, index in enumerate(order, 1):
        step = float(candidate_costs_s[index])
        if cost_budget_s is not None and cumulative + step > cost_budget_s:
            break  # cannot afford this trial; stop (candidate budget still preserved)
        cumulative += step
        trials.append(
            ReplayCostTrial(
                trial_number=number,
                candidate_index=index,
                server_process_seconds=step,
                cumulative_server_process_seconds=cumulative,
            )
        )
    capped = cost_budget_s is not None and len(trials) < budget
    return ReplayCostReport(
        candidate_budget=budget,
        cost_budget_s=cost_budget_s,
        trials=trials,
        affordable_trials=len(trials),
        total_server_process_seconds=cumulative,
        stopped_on="cost_budget" if capped else "candidate_budget",
    )


def evaluate_replay_search(
    source: BlockedStudyReport,
    spec: ReplaySearchSpec,
    *,
    candidate_costs_s: Optional[list[float]] = None,
    cost_budget_s: Optional[float] = None,
) -> ReplaySearchReport:
    """Reveal candidate outcomes in policy order, then score against the oracle.

    Cost-aware mode is enabled only when ``candidate_costs_s`` (realized
    server-process-seconds per candidate, from timing evidence) is supplied; it
    adds a SEPARATE cost budget/metric and never reorders the outcome-blind policy.
    """
    count = len(source.candidates)
    if spec.budget > count:
        raise ValueError("search budget cannot exceed candidate count")
    if candidate_costs_s is None and cost_budget_s is not None:
        raise ValueError("cost_budget_s requires candidate_costs_s (timing evidence)")
    full_order = candidate_order(spec.policy, count, spec.seed)
    order = full_order[: spec.budget]
    trials = [
        ReplayTrial(
            trial_number=number,
            candidate_index=index,
            engine_values=source.candidates[index].engine_values,
            robust_feasible=source.candidates[index].robust_feasible,
            mean_objective_mean=source.candidates[index].mean_objective_mean,
        )
        for number, index in enumerate(order, 1)
    ]

    feasible = [trial for trial in trials if trial.robust_feasible]
    direction = source.candidates[0].objective_direction
    if feasible:
        values = [trial.mean_objective_mean for trial in feasible]
        best_value = min(values) if direction == "lower_is_better" else max(values)
        best = [
            trial.candidate_index
            for trial in feasible
            if trial.mean_objective_mean == best_value
        ]
        status = "selected"
    else:
        best = []
        status = (
            "search_space_exhausted_no_feasible"
            if spec.budget == count
            else "budget_exhausted_no_feasible"
        )
    recommendation = source.candidates[best[0]].engine_values if len(best) == 1 else None
    oracle_hit = bool(set(best) & set(source.best_candidate_indices))
    regret = None
    if best and source.best_candidate_indices:
        observed = source.candidates[best[0]].mean_objective_mean
        oracle = source.candidates[source.best_candidate_indices[0]].mean_objective_mean
        regret = observed - oracle if direction == "lower_is_better" else oracle - observed
        regret = max(0.0, regret)

    cost = None
    if candidate_costs_s is not None:
        if len(candidate_costs_s) != count:
            raise ValueError("candidate_costs_s must provide one cost per candidate")
        cost = _build_cost_overlay(order, candidate_costs_s, spec.budget, cost_budget_s)

    return ReplaySearchReport(
        spec=spec,
        source=source,
        candidate_count=count,
        trials=trials,
        search_space_exhausted=spec.budget == count,
        status=status,
        trials_to_first_feasible=next(
            (trial.trial_number for trial in trials if trial.robust_feasible), None
        ),
        best_candidate_indices=best,
        recommended_engine_values=recommendation,
        oracle_status=source.status,
        oracle_best_candidate_indices=source.best_candidate_indices,
        oracle_hit=oracle_hit,
        simple_regret=regret,
        cost=cost,
    )
