"""Explicit, conservative SLO evaluation over compatible run cohorts."""

from __future__ import annotations

from collections.abc import Sequence

from ..results import ExperimentResult
from .cohorts import prepare_compatible_cohorts
from .models import (
    OBJECTIVE_DIRECTIONS,
    SLO_RULES,
    DecisionCandidate,
    DecisionReport,
    SLOCheck,
    StudySpec,
)

Run = tuple[str, ExperimentResult]
RunCohort = Sequence[Run]

def evaluate_study(
    cohorts: Sequence[RunCohort],
    study: StudySpec,
) -> DecisionReport:
    """Evaluate observed SLO feasibility, then rank feasible cohort means."""
    compatible = prepare_compatible_cohorts(
        cohorts,
        varied_engine_fields=study.varied_engine_fields,
        min_runs=study.min_runs,
    )
    actual_ids = [summary.experiment_id for summary in compatible.summaries]
    if actual_ids != study.experiment_ids:
        raise ValueError("cohort order/ids must exactly match study.experiment_ids")

    constraints = [
        (field, *SLO_RULES[field], threshold)
        for field, threshold in study.slo.model_dump().items()
        if threshold is not None
    ]
    direction = OBJECTIVE_DIRECTIONS[study.objective]
    candidates: list[DecisionCandidate] = []
    for index, summary in enumerate(compatible.summaries):
        checks: list[SLOCheck] = []
        for slo_field, metric, operator, threshold in constraints:
            stats = summary.metrics[metric]
            observed_worst = stats.maximum if operator == "<=" else stats.minimum
            passed = observed_worst <= threshold if operator == "<=" else observed_worst >= threshold
            checks.append(
                SLOCheck(
                    slo_field=slo_field,
                    metric=metric,
                    operator=operator,
                    threshold=threshold,
                    observed_worst=observed_worst,
                    passed=passed,
                )
            )
        feasible = all(check.passed for check in checks)
        candidates.append(
            DecisionCandidate(
                cohort=summary,
                engine_values=compatible.engine_values[index],
                slo_checks=checks,
                feasible=feasible,
                objective_metric=study.objective,
                objective_direction=direction,
                objective_mean=summary.metrics[study.objective].mean,
                rank=1 if feasible else None,
            )
        )

    feasible = [candidate for candidate in candidates if candidate.feasible]
    if feasible:
        reverse = direction == "higher_is_better"
        ordered_values = sorted(
            {candidate.objective_mean for candidate in feasible}, reverse=reverse
        )
        ranks = {value: rank for rank, value in enumerate(ordered_values, start=1)}
        for candidate in feasible:
            candidate.rank = ranks[candidate.objective_mean]

    best = [candidate.cohort.experiment_id for candidate in feasible if candidate.rank == 1]
    status = "no_feasible_candidate" if not best else "selected" if len(best) == 1 else "tie"
    return DecisionReport(
        study=study,
        context_fingerprint=compatible.context_fingerprint,
        candidates=candidates,
        status=status,
        recommended_experiment_id=best[0] if len(best) == 1 else None,
        best_experiment_ids=best,
    )
