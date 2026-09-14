"""Preregistered multi-task corpus contracts for held-out search evaluation.

The corpus spec contains no measurements or outcomes.  It binds a fixed engine
candidate grid to development and held-out blocked studies before either set is
evaluated, so later aggregate scoring cannot silently change the search space,
SLOs, budgets, or evaluation partition.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .comparison.models import BlockedStudySpec
from .config import ExperimentConfig


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


def _task_context(
    config: ExperimentConfig, varied_engine_fields: list[str]
) -> dict[str, Any]:
    """Identity expected to stay fixed for one shape across both partitions."""
    payload = config.model_dump(mode="json")
    for key in ("experiment_id", "name", "description", "tags"):
        payload.pop(key, None)
    for field in varied_engine_fields:
        payload["engine"].pop(field, None)
    # Prompt and arrival samples may differ between partitions/blocks; the shape
    # and every other generation/arrival parameter must remain identical.
    for field in ("seed", "prompt_seed", "arrival_seed"):
        payload["workload"].pop(field, None)
    return payload


def validate_corpus_configs(
    spec: HeldOutCorpusSpec,
    configs: Mapping[str, ExperimentConfig],
) -> None:
    """Bind generated 0.5.0 configs to a preregistered corpus, fail closed.

    The mapping must contain exactly the experiment ids declared by ``spec``.
    New corpus configs use explicit independent prompt and arrival seeds. Within
    a task, prompt content is fixed while blocks vary arrival seed. Development
    and held-out instances of a shape may use different prompt samples, but all
    non-seed workload and non-varied engine fields must match.
    """
    expected_ids = {
        experiment_id
        for task in spec.tasks
        for block in task.study.blocks
        for experiment_id in block.experiment_ids
    }
    actual_ids = set(configs)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"corpus config ids mismatch: missing={missing}, extra={extra}")

    shape_contexts: dict[str, str] = {}
    for task in spec.tasks:
        task_prompt_seed: Optional[int] = None
        for block in task.study.blocks:
            for position, experiment_id in enumerate(block.experiment_ids):
                config = configs[experiment_id]
                if config.experiment_id != experiment_id:
                    raise ValueError("config mapping key must equal config.experiment_id")
                if config.schema_version != "0.5.0":
                    raise ValueError("M2D corpus configs must use schema_version 0.5.0")
                workload = config.workload
                if workload.prompt_seed is None or workload.arrival_seed is None:
                    raise ValueError(
                        "M2D corpus configs must set prompt_seed and arrival_seed explicitly"
                    )
                if workload.arrival_seed != block.seed:
                    raise ValueError("config arrival_seed must equal its study block seed")
                if task_prompt_seed is None:
                    task_prompt_seed = workload.prompt_seed
                elif workload.prompt_seed != task_prompt_seed:
                    raise ValueError("prompt_seed must stay fixed within a blocked task")

                candidate = spec.candidates[position]
                actual_engine = {
                    field: getattr(config.engine, field)
                    for field in spec.varied_engine_fields
                }
                if actual_engine != candidate.engine_values:
                    raise ValueError(
                        "config engine values must match its corpus candidate position"
                    )

                context = json.dumps(
                    _task_context(config, spec.varied_engine_fields),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                prior = shape_contexts.setdefault(task.shape_id, context)
                if context != prior:
                    raise ValueError(
                        "non-seed workload and non-varied engine context must match "
                        "for a shape across development and heldout partitions"
                    )
