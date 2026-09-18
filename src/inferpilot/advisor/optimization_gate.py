"""Should InferPilot even try to tune this config? Diagnose first, then decide.

The loop must not fish. Before proposing/testing any candidate, InferPilot diagnoses
the current (baseline) config's bottleneck; it explores a lever ONLY when the diagnosis
names one (i.e. a winnable regime such as kv_capacity_bound). In compute_bound or
underutilized regimes the honest answer is "keep the default — no config lever can help",
and the gate says so instead of burning a search.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from .._base import SchemaModel
from ..diagnosis import BottleneckDiagnosis, diagnose
from ..results import ExperimentResult


class OptimizationPlan(SchemaModel):
    """Self-validating decision: explore a named lever, or abstain and keep the default."""

    plan_version: Literal["0.1.0"] = "0.1.0"
    baseline_diagnosis: BottleneckDiagnosis
    should_explore: bool
    lever: str
    rationale: str

    @model_validator(mode="after")
    def _check(self) -> "OptimizationPlan":
        lever = self.baseline_diagnosis.recommended_lever
        should = lever != "none"
        rationale = (
            f"{self.baseline_diagnosis.regime}: explore {lever} — {self.baseline_diagnosis.predicted_effect}"
            if should
            else f"{self.baseline_diagnosis.regime}: keep default — {self.baseline_diagnosis.predicted_effect}"
        )
        if (self.should_explore, self.lever, self.rationale) != (should, lever, rationale):
            raise ValueError("optimization plan is inconsistent with its diagnosis")
        return self


def plan_from_diagnosis(diagnosis: BottleneckDiagnosis) -> OptimizationPlan:
    lever = diagnosis.recommended_lever
    should = lever != "none"
    rationale = (
        f"{diagnosis.regime}: explore {lever} — {diagnosis.predicted_effect}"
        if should
        else f"{diagnosis.regime}: keep default — {diagnosis.predicted_effect}"
    )
    return OptimizationPlan(
        baseline_diagnosis=diagnosis, should_explore=should, lever=lever, rationale=rationale,
    )


def plan_optimization(baseline: ExperimentResult) -> OptimizationPlan:
    """Diagnose the baseline and decide whether any config lever is worth exploring."""
    return plan_from_diagnosis(diagnose(baseline))
