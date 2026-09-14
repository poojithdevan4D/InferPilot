"""Serializable contracts for run-level cohort summaries and comparisons."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from .._base import SchemaModel

REPORT_VERSION = "0.2.0"
FRONTIER_REPORT_VERSION = "0.1.0"


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
