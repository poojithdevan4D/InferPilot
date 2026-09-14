"""Run fixed-budget, outcome-blind search baselines over blocked evidence."""

from __future__ import annotations

from ..comparison.models import BlockedStudyReport
from .models import ReplaySearchReport, ReplaySearchSpec, ReplayTrial
from .policy import candidate_order


def evaluate_replay_search(
    source: BlockedStudyReport, spec: ReplaySearchSpec
) -> ReplaySearchReport:
    """Reveal candidate outcomes in policy order, then score against the oracle."""
    count = len(source.candidates)
    if spec.budget > count:
        raise ValueError("search budget cannot exceed candidate count")
    order = candidate_order(spec.policy, count, spec.seed)[: spec.budget]
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
    )
