"""Serializable contracts for run-level cohort summaries and comparisons."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from .._base import SchemaModel

REPORT_VERSION = "0.1.0"


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
