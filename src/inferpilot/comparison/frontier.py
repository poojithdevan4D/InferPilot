"""Compatibility-guarded Pareto analysis over repeated-run cohorts."""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..results import ExperimentResult
from .compare import METRICS, _validate_varied_fields, build_cohort
from .fingerprint import comparison_fingerprint
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
    if len(cohorts) < 2:
        raise ValueError("Pareto analysis requires at least two cohorts")

    fields = _validate_varied_fields(varied_engine_fields)
    objectives = _validate_objectives(objective_metrics)
    summaries = [build_cohort(runs, min_runs=min_runs) for runs in cohorts]

    experiment_ids = [summary.experiment_id for summary in summaries]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("Pareto analysis requires unique experiment ids")

    cohort_contexts: list[str] = []
    for runs in cohorts:
        contexts = {comparison_fingerprint(result, fields) for _, result in runs}
        if len(contexts) != 1:
            raise ValueError("a cohort differs outside the explicitly varied engine fields")
        cohort_contexts.append(next(iter(contexts)))
    if len(set(cohort_contexts)) != 1:
        raise ValueError(
            "cohorts differ outside the explicitly varied engine field(s); frontier refused"
        )

    engine_values: list[dict[str, object]] = []
    for runs in cohorts:
        engine = runs[0][1].config.engine
        engine_values.append({field: getattr(engine, field) for field in fields})
    unchanged = [
        field
        for field in fields
        if len({json.dumps(values[field], sort_keys=True) for values in engine_values}) == 1
    ]
    if unchanged:
        raise ValueError(f"allowlisted engine field(s) did not actually change: {unchanged}")

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
                engine_values=engine_values[index],
                objective_means=objective_means[index],
                dominated_by=dominators,
                is_nondominated=not dominators,
            )
        )

    return ParetoReport(
        context_fingerprint=cohort_contexts[0],
        varied_engine_fields=fields,
        objectives=objectives,
        entries=entries,
        nondominated_experiment_ids=[
            entry.cohort.experiment_id for entry in entries if entry.is_nondominated
        ],
    )
