"""Shared compatibility checks for analyses spanning multiple cohorts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..results import ExperimentResult
from .compare import _validate_varied_fields, build_cohort
from .fingerprint import comparison_fingerprint
from .models import CohortSummary

Run = tuple[str, ExperimentResult]
RunCohort = Sequence[Run]


@dataclass(frozen=True)
class CompatibleCohorts:
    varied_engine_fields: list[str]
    summaries: list[CohortSummary]
    context_fingerprint: str
    engine_values: list[dict[str, Any]]


def prepare_compatible_cohorts(
    cohorts: Sequence[RunCohort],
    *,
    varied_engine_fields: Sequence[str],
    min_runs: int,
) -> CompatibleCohorts:
    """Validate exact cohorts and prove all non-varied conditions agree."""
    if len(cohorts) < 2:
        raise ValueError("analysis requires at least two cohorts")

    fields = _validate_varied_fields(varied_engine_fields)
    summaries = [build_cohort(runs, min_runs=min_runs) for runs in cohorts]
    experiment_ids = [summary.experiment_id for summary in summaries]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("analysis requires unique experiment ids")

    cohort_contexts: list[str] = []
    for runs in cohorts:
        contexts = {comparison_fingerprint(result, fields) for _, result in runs}
        if len(contexts) != 1:
            raise ValueError("a cohort differs outside the explicitly varied engine fields")
        cohort_contexts.append(next(iter(contexts)))
    if len(set(cohort_contexts)) != 1:
        raise ValueError(
            "cohorts differ outside the explicitly varied engine field(s); analysis refused"
        )

    engine_values: list[dict[str, Any]] = []
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

    return CompatibleCohorts(
        varied_engine_fields=fields,
        summaries=summaries,
        context_fingerprint=cohort_contexts[0],
        engine_values=engine_values,
    )
