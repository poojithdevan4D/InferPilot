"""Serializable contracts for fixed-budget search replay reports."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from .._base import SchemaModel
from ..comparison.models import BlockedStudyReport


SEARCH_REPLAY_REPORT_VERSION = "0.1.0"
SearchPolicy = Literal["declared_order-v1", "seeded_random-v1"]
ReplayStatus = Literal[
    "selected",
    "budget_exhausted_no_feasible",
    "search_space_exhausted_no_feasible",
]


class ReplaySearchSpec(SchemaModel):
    """An outcome-blind baseline policy and candidate-evaluation budget."""

    spec_version: Literal["0.1.0"] = "0.1.0"
    search_id: str
    policy: SearchPolicy
    budget: int = Field(ge=1)
    seed: int = 0


class ReplayTrial(SchemaModel):
    """One revealed candidate outcome in policy-selected order."""

    trial_number: int = Field(ge=1)
    candidate_index: int = Field(ge=0)
    engine_values: dict[str, Any]
    robust_feasible: bool
    mean_objective_mean: float


class ReplaySearchReport(SchemaModel):
    """Self-validating fixed-budget replay against a complete blocked study.

    The complete source report is deliberately embedded so every replay field
    can be rebound on load. The source's unobserved candidates are used only for
    post-hoc oracle evaluation; baseline ordering cannot inspect their outcomes.
    """

    report_version: str = SEARCH_REPLAY_REPORT_VERSION
    spec: ReplaySearchSpec
    source: BlockedStudyReport
    candidate_count: int = Field(ge=1)
    trials: list[ReplayTrial] = Field(min_length=1)
    search_space_exhausted: bool
    status: ReplayStatus
    trials_to_first_feasible: Optional[int] = Field(default=None, ge=1)
    best_candidate_indices: list[int] = Field(default_factory=list)
    recommended_engine_values: Optional[dict[str, Any]] = None
    oracle_status: Literal["selected", "tie", "no_feasible_candidate"]
    oracle_best_candidate_indices: list[int] = Field(default_factory=list)
    oracle_hit: bool
    simple_regret: Optional[float] = Field(default=None, ge=0)
    interpretation: str = (
        "Candidate order is fixed without access to outcomes. Each trial reveals one "
        "complete blocked-candidate result, whose cost is the source study's number "
        "of workload blocks. Oracle fields are computed only after replay to score "
        "the baseline. This is an offline benchmark, not a live optimizer or a "
        "statistical/deployment guarantee."
    )

    @model_validator(mode="after")
    def _validate_report(self) -> "ReplaySearchReport":
        if self.report_version != SEARCH_REPLAY_REPORT_VERSION:
            raise ValueError(
                f"unsupported replay report_version {self.report_version!r}; "
                f"expected {SEARCH_REPLAY_REPORT_VERSION!r}"
            )
        candidates = self.source.candidates
        count = len(candidates)
        if self.candidate_count != count:
            raise ValueError("candidate_count must match the source report")
        if self.spec.budget > count:
            raise ValueError("search budget cannot exceed candidate count")
        if len(self.trials) != self.spec.budget:
            raise ValueError("trial count must equal the search budget")

        # Local import avoids a models -> policy -> models import cycle.
        from .policy import candidate_order

        expected_indices = candidate_order(
            self.spec.policy, count, self.spec.seed
        )[: self.spec.budget]
        actual_indices = [trial.candidate_index for trial in self.trials]
        if actual_indices != expected_indices:
            raise ValueError("trial order is inconsistent with the outcome-blind policy")
        if len(set(actual_indices)) != len(actual_indices):
            raise ValueError("a candidate may be evaluated at most once")

        for ordinal, trial in enumerate(self.trials, 1):
            if trial.trial_number != ordinal:
                raise ValueError("trial_number must be contiguous and one-based")
            source_candidate = candidates[trial.candidate_index]
            expected = (
                source_candidate.engine_values,
                source_candidate.robust_feasible,
                source_candidate.mean_objective_mean,
            )
            actual = (
                trial.engine_values,
                trial.robust_feasible,
                trial.mean_objective_mean,
            )
            if actual != expected:
                raise ValueError("trial outcome must exactly match its source candidate")

        exhausted = self.spec.budget == count
        if self.search_space_exhausted != exhausted:
            raise ValueError("search_space_exhausted is inconsistent with budget")

        feasible_trials = [trial for trial in self.trials if trial.robust_feasible]
        expected_first = (
            feasible_trials[0].trial_number if feasible_trials else None
        )
        if self.trials_to_first_feasible != expected_first:
            raise ValueError("trials_to_first_feasible is inconsistent with trials")

        direction = self.source.candidates[0].objective_direction
        if feasible_trials:
            values = [trial.mean_objective_mean for trial in feasible_trials]
            best_value = min(values) if direction == "lower_is_better" else max(values)
            best = [
                trial.candidate_index
                for trial in feasible_trials
                if trial.mean_objective_mean == best_value
            ]
            expected_status: ReplayStatus = "selected"
        else:
            best = []
            expected_status = (
                "search_space_exhausted_no_feasible"
                if exhausted
                else "budget_exhausted_no_feasible"
            )
        if self.best_candidate_indices != best:
            raise ValueError("best_candidate_indices are inconsistent with revealed trials")
        if self.status != expected_status:
            raise ValueError("replay status is inconsistent with revealed trials")
        expected_recommendation = candidates[best[0]].engine_values if len(best) == 1 else None
        if self.recommended_engine_values != expected_recommendation:
            raise ValueError("recommended_engine_values is inconsistent with replay result")

        if self.oracle_status != self.source.status:
            raise ValueError("oracle_status must match the complete source study")
        if self.oracle_best_candidate_indices != self.source.best_candidate_indices:
            raise ValueError("oracle candidates must match the complete source study")
        expected_hit = bool(set(best) & set(self.source.best_candidate_indices))
        if self.oracle_hit != expected_hit:
            raise ValueError("oracle_hit is inconsistent with replay and source results")

        expected_regret: float | None = None
        if best and self.source.best_candidate_indices:
            observed_value = candidates[best[0]].mean_objective_mean
            oracle_value = candidates[self.source.best_candidate_indices[0]].mean_objective_mean
            expected_regret = (
                observed_value - oracle_value
                if direction == "lower_is_better"
                else oracle_value - observed_value
            )
            # Equal values and floating arithmetic can produce negative zero.
            expected_regret = max(0.0, expected_regret)
        if self.simple_regret != expected_regret:
            raise ValueError("simple_regret is inconsistent with replay and oracle")
        return self
