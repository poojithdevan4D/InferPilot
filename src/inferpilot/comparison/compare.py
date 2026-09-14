"""Pure run-level cohort summarization and controlled comparison."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence

from ..config import EngineConfig
from ..results import ExperimentResult
from .fingerprint import comparison_fingerprint, exact_fingerprint
from .models import CohortSummary, ComparisonReport, MetricComparison, StatisticSummary

MetricGetter = Callable[[ExperimentResult], float]

METRICS: dict[str, tuple[str, MetricGetter]] = {
    "ttft_p50_ms": ("lower_is_better", lambda r: float(r.aggregates.ttft_p50_ms)),
    "ttft_p95_ms": ("lower_is_better", lambda r: float(r.aggregates.ttft_p95_ms)),
    "tpot_p50_ms": ("lower_is_better", lambda r: float(r.aggregates.tpot_p50_ms)),
    "tpot_p95_ms": ("lower_is_better", lambda r: float(r.aggregates.tpot_p95_ms)),
    "e2e_p50_ms": ("lower_is_better", lambda r: float(r.aggregates.e2e_p50_ms)),
    "e2e_p95_ms": ("lower_is_better", lambda r: float(r.aggregates.e2e_p95_ms)),
    "throughput_tokens_per_s": (
        "higher_is_better",
        lambda r: float(r.aggregates.throughput_tokens_per_s),
    ),
    "peak_gpu_memory_mb": (
        "lower_is_better",
        lambda r: float(r.telemetry.peak_gpu_memory_mb),
    ),
    "peak_kv_cache_usage": (
        "context_only",
        lambda r: float(r.telemetry.kv_cache_usage_peak_perc),
    ),
}


def _stats(values: Sequence[float]) -> StatisticSummary:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    mean = statistics.mean(values)
    stddev = statistics.stdev(values) if len(values) >= 2 else 0.0
    cv = abs(stddev / mean) * 100 if mean != 0 else None
    return StatisticSummary(
        count=len(values),
        mean=mean,
        sample_stddev=stddev,
        coefficient_of_variation_pct=cv,
        minimum=min(values),
        maximum=max(values),
    )


def build_cohort(
    runs: Sequence[tuple[str, ExperimentResult]], *, min_runs: int = 3
) -> CohortSummary:
    """Build an exact-repeat cohort or fail loudly on weak/mixed evidence."""
    if len(runs) < min_runs:
        raise ValueError(f"cohort requires at least {min_runs} eligible runs; got {len(runs)}")
    if any(not result.is_baseline_eligible for _, result in runs):
        raise ValueError("cohort contains a baseline-ineligible result")

    fingerprints = {exact_fingerprint(result) for _, result in runs}
    if len(fingerprints) != 1:
        raise ValueError("cohort mixes non-equivalent configurations or environments")
    experiment_ids = {result.config.experiment_id for _, result in runs}
    if len(experiment_ids) != 1:
        raise ValueError("cohort mixes experiment ids")

    summaries = {
        name: _stats([getter(result) for _, result in runs])
        for name, (_, getter) in METRICS.items()
    }
    return CohortSummary(
        experiment_id=next(iter(experiment_ids)),
        exact_fingerprint=next(iter(fingerprints)),
        run_ids=[run_id for run_id, _ in runs],
        metrics=summaries,
    )


def _validate_varied_fields(fields: Sequence[str]) -> list[str]:
    normalized = sorted(set(fields))
    if not normalized:
        raise ValueError("at least one varied engine field is required")
    known = set(EngineConfig.model_fields)
    unknown = [field for field in normalized if field not in known]
    if unknown:
        raise ValueError(f"unknown EngineConfig field(s): {unknown}")
    return normalized


def compare_cohorts(
    baseline_runs: Sequence[tuple[str, ExperimentResult]],
    candidate_runs: Sequence[tuple[str, ExperimentResult]],
    *,
    varied_engine_fields: Sequence[str],
    min_runs: int = 3,
) -> ComparisonReport:
    """Compare repeated cohorts differing only in explicitly allowlisted knobs."""
    fields = _validate_varied_fields(varied_engine_fields)
    baseline = build_cohort(baseline_runs, min_runs=min_runs)
    candidate = build_cohort(candidate_runs, min_runs=min_runs)

    baseline_contexts = {
        comparison_fingerprint(result, fields) for _, result in baseline_runs
    }
    candidate_contexts = {
        comparison_fingerprint(result, fields) for _, result in candidate_runs
    }
    if len(baseline_contexts) != 1 or baseline_contexts != candidate_contexts:
        raise ValueError(
            "cohorts differ outside the explicitly varied engine field(s); comparison refused"
        )

    baseline_engine = baseline_runs[0][1].config.engine
    candidate_engine = candidate_runs[0][1].config.engine
    if all(
        getattr(baseline_engine, field) == getattr(candidate_engine, field)
        for field in fields
    ):
        raise ValueError("allowlisted engine field values did not actually change")

    comparisons: dict[str, MetricComparison] = {}
    for name, (direction, _) in METRICS.items():
        base_stats = baseline.metrics[name]
        candidate_stats = candidate.metrics[name]
        if base_stats.mean == 0:
            relative = improvement = None
        else:
            relative = (candidate_stats.mean - base_stats.mean) / base_stats.mean * 100
            if direction == "lower_is_better":
                improvement = -relative
            elif direction == "higher_is_better":
                improvement = relative
            else:
                improvement = None
        comparisons[name] = MetricComparison(
            direction=direction,
            baseline=base_stats,
            candidate=candidate_stats,
            relative_change_pct=relative,
            improvement_pct=improvement,
        )

    return ComparisonReport(
        context_fingerprint=next(iter(baseline_contexts)),
        varied_engine_fields=fields,
        baseline=baseline,
        candidate=candidate,
        metrics=comparisons,
    )
