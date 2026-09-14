"""Compatibility-guarded Pareto analysis over repeated-run cohorts."""

from __future__ import annotations

from collections.abc import Sequence

from ..results import ExperimentResult
from .cohorts import prepare_compatible_cohorts
from .compare import METRICS
from .models import FrontierEntry, ObjectiveSpec, ParetoReport

Run = tuple[str, ExperimentResult]
RunCohort = Sequence[Run]


def _validate_objectives(metrics: Sequence[str]) -> list[ObjectiveSpec]:
    normalized = list(dict.fromkeys(metrics))
    if len(normalized) < 2:
        raise ValueError("Pareto analysis requires at least two distinct objectives")

    unknown = [metric for metric in normalized if metric not in METRICS]
    if unknown:
        raise ValueError(f"unknown comparison metric(s): {unknown}")

    contextual = [metric for metric in normalized if METRICS[metric][0] == "context_only"]
    if contextual:
        raise ValueError(
            "context-only metrics cannot be optimization objectives without a study-specific "
            f"direction: {contextual}"
        )

    return [
        ObjectiveSpec(metric=metric, direction=METRICS[metric][0])
        for metric in normalized
    ]


def _dominates(
    challenger: dict[str, float],
    subject: dict[str, float],
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    """Return whether challenger is no worse everywhere and better somewhere."""
    no_worse = True
    strictly_better = False
    for objective in objectives:
        left = challenger[objective.metric]
        right = subject[objective.metric]
        if objective.direction == "lower_is_better":
            no_worse &= left <= right
            strictly_better |= left < right
        else:
            no_worse &= left >= right
            strictly_better |= left > right
    return no_worse and strictly_better


def build_pareto_frontier(
    cohorts: Sequence[RunCohort],
    *,
    varied_engine_fields: Sequence[str],
    objective_metrics: Sequence[str],
    min_runs: int = 3,
) -> ParetoReport:
    """Build a descriptive Pareto frontier across compatible exact cohorts.

    Every cohort must independently satisfy the repeated-run evidence policy and
    all cohorts must differ only in the explicitly allowlisted engine fields.
    Objective metrics are mandatory so the analysis never invents a hidden utility
    function or treats contextual resource telemetry as intrinsically good/bad.
    """
    objectives = _validate_objectives(objective_metrics)
    compatible = prepare_compatible_cohorts(
        cohorts,
        varied_engine_fields=varied_engine_fields,
        min_runs=min_runs,
    )
    summaries = compatible.summaries
    experiment_ids = [summary.experiment_id for summary in summaries]

    objective_means = [
        {objective.metric: summary.metrics[objective.metric].mean for objective in objectives}
        for summary in summaries
    ]
    entries: list[FrontierEntry] = []
    for index, summary in enumerate(summaries):
        dominators = [
            experiment_ids[other]
            for other in range(len(summaries))
            if other != index
            and _dominates(objective_means[other], objective_means[index], objectives)
        ]
        entries.append(
            FrontierEntry(
                cohort=summary,
                engine_values=compatible.engine_values[index],
                objective_means=objective_means[index],
                dominated_by=dominators,
                is_nondominated=not dominators,
            )
        )

    return ParetoReport(
        context_fingerprint=compatible.context_fingerprint,
        varied_engine_fields=compatible.varied_engine_fields,
        objectives=objectives,
        entries=entries,
        nondominated_experiment_ids=[
            entry.cohort.experiment_id for entry in entries if entry.is_nondominated
        ],
    )
