"""Serializable contracts for run-level cohort summaries and comparisons."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from .._base import SchemaModel
from ..config import SLO

REPORT_VERSION = "0.2.1"
FRONTIER_REPORT_VERSION = "0.1.1"
DECISION_REPORT_VERSION = "0.1.1"

SLO_RULES = {
    "ttft_p95_ms": ("ttft_p95_ms", "<="),
    "tpot_p95_ms": ("tpot_p95_ms", "<="),
    "e2e_p95_ms": ("e2e_p95_ms", "<="),
    "min_throughput_tokens_per_s": ("throughput_tokens_per_s", ">="),
}

OBJECTIVE_DIRECTIONS = {
    "ttft_p95_ms": "lower_is_better",
    "tpot_p95_ms": "lower_is_better",
    "e2e_p95_ms": "lower_is_better",
    "throughput_tokens_per_s": "higher_is_better",
}


class StatisticSummary(SchemaModel):
    """Descriptive statistics over one run-level metric."""

    count: int = Field(ge=1)
    mean: float
    sample_stddev: float = Field(ge=0)
    coefficient_of_variation_pct: Optional[float] = Field(default=None, ge=0)
    minimum: float
    maximum: float


class CohortSummary(SchemaModel):
    """Summary of repeated, exactly equivalent benchmark runs."""

    experiment_id: str
    exact_fingerprint: str
    run_ids: list[str] = Field(min_length=1)
    metrics: dict[str, StatisticSummary]


class MetricComparison(SchemaModel):
    """Observed candidate change for one metric; not a significance claim."""

    direction: Literal["lower_is_better", "higher_is_better", "context_only"]
    baseline: StatisticSummary
    candidate: StatisticSummary
    relative_change_pct: Optional[float]
    improvement_pct: Optional[float]


class ComparisonReport(SchemaModel):
    """Comparison of compatible baseline and candidate cohorts."""

    report_version: str = REPORT_VERSION
    context_fingerprint: str
    varied_engine_fields: list[str] = Field(min_length=1)
    baseline: CohortSummary
    candidate: CohortSummary
    metrics: dict[str, MetricComparison]
    interpretation: str = Field(
        default=(
            "Observed run-level differences only. Do not treat this report as a "
            "statistical significance claim or automatic accept/reject decision."
        )
    )


class ObjectiveSpec(SchemaModel):
    """One explicitly selected optimization objective."""

    metric: str
    direction: Literal["lower_is_better", "higher_is_better"]


class FrontierEntry(SchemaModel):
    """One cohort's position in an observed-mean Pareto comparison."""

    cohort: CohortSummary
    engine_values: dict[str, Any]
    objective_means: dict[str, float]
    dominated_by: list[str] = Field(default_factory=list)
    is_nondominated: bool

    @model_validator(mode="after")
    def _check_dominance_flag(self) -> "FrontierEntry":
        if len(set(self.dominated_by)) != len(self.dominated_by):
            raise ValueError("dominated_by must not contain duplicate experiment ids")
        if self.cohort.experiment_id in self.dominated_by:
            raise ValueError("a frontier entry cannot dominate itself")
        if self.is_nondominated != (not self.dominated_by):
            raise ValueError("is_nondominated must agree with dominated_by")
        return self


class ParetoReport(SchemaModel):
    """Multi-objective frontier across compatible repeated-run cohorts."""

    report_version: str = FRONTIER_REPORT_VERSION
    context_fingerprint: str
    varied_engine_fields: list[str] = Field(min_length=1)
    objectives: list[ObjectiveSpec] = Field(min_length=2)
    entries: list[FrontierEntry] = Field(min_length=2)
    nondominated_experiment_ids: list[str] = Field(min_length=1)
    interpretation: str = Field(
        default=(
            "Pareto dominance is computed from run-level cohort means. It identifies "
            "observed trade-offs only; it is not a statistical significance claim, "
            "an SLO evaluation, or an automatic deployment decision."
        )
    )

    @model_validator(mode="after")
    def _check_report_consistency(self) -> "ParetoReport":
        objective_names = [objective.metric for objective in self.objectives]
        if len(set(objective_names)) != len(objective_names):
            raise ValueError("objectives must contain distinct metrics")

        experiment_ids = [entry.cohort.experiment_id for entry in self.entries]
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("entries must contain unique experiment ids")
        known_ids = set(experiment_ids)

        expected_objectives = set(objective_names)
        expected_fields = set(self.varied_engine_fields)
        for entry in self.entries:
            if set(entry.objective_means) != expected_objectives:
                raise ValueError("every entry must contain exactly the declared objectives")
            if set(entry.engine_values) != expected_fields:
                raise ValueError("every entry must contain exactly the varied engine fields")
            if not set(entry.dominated_by) <= known_ids:
                raise ValueError("dominated_by contains an unknown experiment id")

        expected_frontier = [
            entry.cohort.experiment_id for entry in self.entries if entry.is_nondominated
        ]
        if self.nondominated_experiment_ids != expected_frontier:
            raise ValueError(
                "nondominated_experiment_ids must match nondominated entries in entry order"
            )
        return self


DecisionMetric = Literal[
    "ttft_p95_ms",
    "tpot_p95_ms",
    "e2e_p95_ms",
    "throughput_tokens_per_s",
]


class StudySpec(SchemaModel):
    """Explicit product/research policy used to choose among measured cohorts."""

    spec_version: Literal["0.1.0"] = "0.1.0"
    study_id: str
    experiment_ids: list[str] = Field(min_length=2)
    varied_engine_fields: list[str] = Field(min_length=1)
    objective: DecisionMetric
    slo: SLO
    min_runs: int = Field(default=3, ge=3)

    @model_validator(mode="after")
    def _check_study(self) -> "StudySpec":
        if len(set(self.experiment_ids)) != len(self.experiment_ids):
            raise ValueError("experiment_ids must be distinct")
        if len(set(self.varied_engine_fields)) != len(self.varied_engine_fields):
            raise ValueError("varied_engine_fields must be distinct")
        if not any(value is not None for value in self.slo.model_dump().values()):
            raise ValueError("study SLO must declare at least one constraint")
        return self


class SLOCheck(SchemaModel):
    """Conservative check of one SLO against repeated observed runs."""

    slo_field: str
    metric: str
    operator: Literal["<=", ">="]
    threshold: float
    observed_worst: float
    passed: bool


class DecisionCandidate(SchemaModel):
    """Eligibility and ranking outcome for one measured cohort."""

    cohort: CohortSummary
    engine_values: dict[str, Any]
    slo_checks: list[SLOCheck] = Field(min_length=1)
    feasible: bool
    objective_metric: DecisionMetric
    objective_direction: Literal["lower_is_better", "higher_is_better"]
    objective_mean: float
    rank: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_feasibility(self) -> "DecisionCandidate":
        if self.feasible != all(check.passed for check in self.slo_checks):
            raise ValueError("feasible must agree with all SLO checks")
        if self.feasible != (self.rank is not None):
            raise ValueError("only feasible candidates may have a rank")
        return self


class DecisionReport(SchemaModel):
    """Reproducible SLO-constrained selection from compatible evidence."""

    report_version: str = DECISION_REPORT_VERSION
    study: StudySpec
    context_fingerprint: str
    candidates: list[DecisionCandidate] = Field(min_length=2)
    status: Literal["selected", "tie", "no_feasible_candidate"]
    recommended_experiment_id: Optional[str] = None
    best_experiment_ids: list[str] = Field(default_factory=list)
    interpretation: str = Field(
        default=(
            "SLO feasibility requires every observed run-level metric to meet each "
            "constraint. Ranking uses the feasible cohorts' objective means. This is "
            "an evidence-backed recommendation under the declared study policy, not "
            "a statistical guarantee or an automatic deployment action."
        )
    )

    @model_validator(mode="after")
    def _check_decision(self) -> "DecisionReport":
        experiment_ids = [candidate.cohort.experiment_id for candidate in self.candidates]
        if experiment_ids != self.study.experiment_ids:
            raise ValueError("candidate order must match study.experiment_ids")
        feasible = [candidate for candidate in self.candidates if candidate.feasible]
        expected_slo = [
            (field, *SLO_RULES[field], threshold)
            for field, threshold in self.study.slo.model_dump().items()
            if threshold is not None
        ]
        for candidate in self.candidates:
            if set(candidate.engine_values) != set(self.study.varied_engine_fields):
                raise ValueError("candidate engine values must match varied_engine_fields")
            if candidate.objective_metric != self.study.objective:
                raise ValueError("candidate objective must match the study objective")
            if candidate.objective_direction != OBJECTIVE_DIRECTIONS[self.study.objective]:
                raise ValueError("candidate objective direction is incorrect")
            actual_slo = [
                (check.slo_field, check.metric, check.operator, check.threshold)
                for check in candidate.slo_checks
            ]
            if actual_slo != expected_slo:
                raise ValueError("candidate SLO checks must exactly match the study SLO")
            for check in candidate.slo_checks:
                expected_pass = (
                    check.observed_worst <= check.threshold
                    if check.operator == "<="
                    else check.observed_worst >= check.threshold
                )
                if check.passed != expected_pass:
                    raise ValueError("SLO check result is inconsistent with its observed value")

        if feasible:
            reverse = OBJECTIVE_DIRECTIONS[self.study.objective] == "higher_is_better"
            ordered_values = sorted(
                {candidate.objective_mean for candidate in feasible}, reverse=reverse
            )
            expected_ranks = {
                value: rank for rank, value in enumerate(ordered_values, start=1)
            }
            if any(
                candidate.rank != expected_ranks[candidate.objective_mean]
                for candidate in feasible
            ):
                raise ValueError("candidate ranks are inconsistent with the study objective")
        best = [candidate.cohort.experiment_id for candidate in feasible if candidate.rank == 1]
        if self.best_experiment_ids != best:
            raise ValueError("best_experiment_ids must match feasible rank-1 candidates")
        expected_status = (
            "no_feasible_candidate" if not best else "selected" if len(best) == 1 else "tie"
        )
        if self.status != expected_status:
            raise ValueError("decision status is inconsistent with candidate ranks")
        expected_recommendation = best[0] if len(best) == 1 else None
        if self.recommended_experiment_id != expected_recommendation:
            raise ValueError("recommended_experiment_id is inconsistent with rank-1 candidates")
        return self
