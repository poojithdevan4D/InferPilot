"""Result catalog and compatible multi-run cohort comparison."""

from .compare import build_cohort, compare_cohorts
from .fingerprint import comparison_fingerprint, exact_fingerprint
from .models import CohortSummary, ComparisonReport, MetricComparison, StatisticSummary
from .store import IngestedRun, ResultStore

__all__ = [
    "ResultStore",
    "IngestedRun",
    "StatisticSummary",
    "CohortSummary",
    "MetricComparison",
    "ComparisonReport",
    "exact_fingerprint",
    "comparison_fingerprint",
    "build_cohort",
    "compare_cohorts",
]
