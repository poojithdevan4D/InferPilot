"""Result catalog and compatible multi-run cohort comparison."""

from .blocked import evaluate_blocked_study
from .compare import build_cohort, compare_cohorts
from .decision import evaluate_study
from .fingerprint import comparison_fingerprint, cross_block_fingerprint, exact_fingerprint
from .frontier import build_pareto_frontier
from .models import (
    BlockCandidateResult,
    BlockResult,
    BlockedStudyReport,
    BlockedStudySpec,
    CohortSummary,
    ComparisonReport,
    DecisionCandidate,
    DecisionReport,
    FrontierEntry,
    MetricComparison,
    ObjectiveSpec,
    ParetoReport,
    RobustCandidate,
    SLOCheck,
    StatisticSummary,
    StudyBlock,
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
    "evaluate_blocked_study",
    "cross_block_fingerprint",
    "StudyBlock",
    "BlockedStudySpec",
    "BlockCandidateResult",
    "BlockResult",
    "RobustCandidate",
    "BlockedStudyReport",
]
