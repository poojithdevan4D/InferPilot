"""Result catalog and compatible multi-run cohort comparison."""

from .compare import build_cohort, compare_cohorts
from .decision import evaluate_study
from .fingerprint import comparison_fingerprint, exact_fingerprint
from .frontier import build_pareto_frontier
from .models import (
    CohortSummary,
    ComparisonReport,
    DecisionCandidate,
    DecisionReport,
    FrontierEntry,
    MetricComparison,
    ObjectiveSpec,
    ParetoReport,
    SLOCheck,
    StatisticSummary,
    StudySpec,
)
from .store import IngestedRun, ResultStore

__all__ = [
    "ResultStore",
    "IngestedRun",
    "StatisticSummary",
    "CohortSummary",
    "MetricComparison",
    "ComparisonReport",
    "StudySpec",
    "SLOCheck",
    "DecisionCandidate",
    "DecisionReport",
    "ObjectiveSpec",
    "FrontierEntry",
    "ParetoReport",
    "exact_fingerprint",
    "comparison_fingerprint",
    "build_cohort",
    "compare_cohorts",
    "build_pareto_frontier",
    "evaluate_study",
]
