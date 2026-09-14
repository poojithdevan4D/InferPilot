"""Preregistered multi-task corpus contracts for held-out search evaluation.

The corpus spec contains no measurements or outcomes.  It binds a fixed engine
candidate grid to development and held-out blocked studies before either set is
evaluated, so later aggregate scoring cannot silently change the search space,
SLOs, budgets, or evaluation partition.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .comparison.models import BlockedStudySpec


CORPUS_SPEC_VERSION = "0.1.0"
CorpusPartition = Literal["development", "heldout"]


class CorpusCandidate(SchemaModel):
    """One position in the engine grid, shared by every corpus task."""

    candidate_id: str = Field(min_length=1)
    engine_values: dict[str, Any] = Field(min_length=1)


class CorpusTask(SchemaModel):
    """One blocked study assigned to a development or held-out partition.

    Candidate position is semantic: position ``i`` in every study block must be
    the global corpus candidate at position ``i``.  Engine values are verified
    against run evidence later by ``evaluate_blocked_study``.
    """

    task_id: str = Field(min_length=1)
    shape_id: str = Field(min_length=1)
    partition: CorpusPartition
    study: BlockedStudySpec


class HeldOutCorpusSpec(SchemaModel):
    """Exact-versioned preregistration for a rectangular held-out corpus.

    This contract intentionally does not claim that a held-out task is secret;
    sealing is an operational procedure.  It makes accidental protocol drift
    visible by fixing task membership, candidate order, SLOs, and budgets in one
    immutable object before outcome reports are produced.
    """

    spec_version: str = CORPUS_SPEC_VERSION
    corpus_id: str = Field(min_length=1)
    varied_engine_fields: list[str] = Field(min_length=2)
    candidates: list[CorpusCandidate] = Field(min_length=4)
    tasks: list[CorpusTask] = Field(min_length=2)
    candidate_budgets: list[int] = Field(min_length=1)
    cost_budget_s: Optional[float] = Field(default=None, gt=0)
    hypotheses: list[str] = Field(min_length=1)
    failure_policy: Literal["fail_closed"] = "fail_closed"
    budget_exhaustion_policy: Literal["unknown_not_infeasible"] = "unknown_not_infeasible"
    heldout_policy: Literal[
        "freeze_policy_before_opening_heldout_outcomes"
    ] = "freeze_policy_before_opening_heldout_outcomes"
    interpretation: str = (
        "Development outcomes may inform a frozen policy. Held-out outcomes may only "
        "score that frozen policy. This corpus measures performance on its declared "
        "tasks; it is not a statistical, deployment-safety, or hardware-generalization "
        "guarantee."
    )

    @model_validator(mode="after")
    def _check(self) -> "HeldOutCorpusSpec":
        if self.spec_version != CORPUS_SPEC_VERSION:
            raise ValueError(
                f"unsupported corpus spec_version {self.spec_version!r}; "
                f"expected {CORPUS_SPEC_VERSION!r}"
            )
        if len(set(self.varied_engine_fields)) != len(self.varied_engine_fields):
            raise ValueError("varied_engine_fields must be distinct")

        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate ids must be distinct")
        expected_keys = set(self.varied_engine_fields)
        encoded_values: list[str] = []
        for candidate in self.candidates:
            if set(candidate.engine_values) != expected_keys:
                raise ValueError(
                    "each candidate must define exactly the varied engine fields"
                )
            encoded_values.append(
                json.dumps(candidate.engine_values, sort_keys=True, separators=(",", ":"))
            )
        if len(set(encoded_values)) != len(encoded_values):
            raise ValueError("candidate engine-value points must be distinct")

        task_ids = [task.task_id for task in self.tasks]
        study_ids = [task.study.study_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task ids must be distinct")
        if len(set(study_ids)) != len(study_ids):
            raise ValueError("blocked study ids must be distinct across the corpus")

        partitions = {task.partition for task in self.tasks}
        if partitions != {"development", "heldout"}:
            raise ValueError("corpus must contain development and heldout tasks")
        dev_shapes = {task.shape_id for task in self.tasks if task.partition == "development"}
        heldout_shapes = {task.shape_id for task in self.tasks if task.partition == "heldout"}
        if dev_shapes != heldout_shapes:
            raise ValueError(
                "development and heldout partitions must contain the same shape ids"
            )
        shape_partition_pairs = [(task.shape_id, task.partition) for task in self.tasks]
        if len(set(shape_partition_pairs)) != len(shape_partition_pairs):
            raise ValueError("each shape may appear only once per partition")

        width = len(self.candidates)
        all_experiment_ids: list[str] = []
        dev_seeds: set[int] = set()
        heldout_seeds: set[int] = set()
        for task in self.tasks:
            if task.study.varied_engine_fields != self.varied_engine_fields:
                raise ValueError(
                    "every task must use the corpus varied_engine_fields in order"
                )
            if any(len(block.experiment_ids) != width for block in task.study.blocks):
                raise ValueError(
                    "every task block width must equal the corpus candidate count"
                )
            seeds = {block.seed for block in task.study.blocks}
            (dev_seeds if task.partition == "development" else heldout_seeds).update(seeds)
            for block in task.study.blocks:
                all_experiment_ids.extend(block.experiment_ids)
        if dev_seeds & heldout_seeds:
            raise ValueError("development and heldout workload seeds must be disjoint")
        if len(set(all_experiment_ids)) != len(all_experiment_ids):
            raise ValueError("experiment ids must be globally unique across the corpus")

        if self.candidate_budgets != sorted(set(self.candidate_budgets)):
            raise ValueError("candidate_budgets must be distinct and increasing")
        if any(budget < 1 or budget > width for budget in self.candidate_budgets):
            raise ValueError("candidate budgets must fall within the candidate grid")
        return self

