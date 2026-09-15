"""Multi-seed blocked confirmatory study evaluation.

Each block fixes one workload seed and evaluates the same rectangular set of
engine candidates against an explicit SLO. A candidate is *robust-feasible* only
if it passes every block; robust-feasible candidates are ranked by the mean of
their per-block objective means. All per-block results are preserved.

Guardrails reused from the existing comparison layer: within a block, candidates
must be compatible under the strict comparison fingerprint (only the declared
engine fields differ, workload seed included); across blocks, only the workload
seed and human identifiers may differ (checked via the cross-block fingerprint,
which does not weaken the within-block fingerprint).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from ..results import ExperimentResult
from .cohorts import prepare_compatible_cohorts
from .fingerprint import cross_block_fingerprint
from .models import (
    OBJECTIVE_DIRECTIONS,
    SLO_RULES,
    BlockCandidateResult,
    BlockedStudyReport,
    BlockedStudySpec,
    BlockResult,
    RobustCandidate,
    SLOCheck,
)

Run = tuple[str, ExperimentResult]
RunCohort = Sequence[Run]


def _slo_checks(summary, constraints) -> list[SLOCheck]:
    checks: list[SLOCheck] = []
    for slo_field, metric, operator, threshold in constraints:
        stats = summary.metrics[metric]
        worst = stats.maximum if operator == "<=" else stats.minimum
        passed = worst <= threshold if operator == "<=" else worst >= threshold
        checks.append(
            SLOCheck(
                slo_field=slo_field, metric=metric, operator=operator,
                threshold=threshold, observed_worst=worst, passed=passed,
            )
        )
    return checks


def evaluate_blocked_study(
    block_cohorts: Sequence[Sequence[RunCohort]],
    study: BlockedStudySpec,
) -> BlockedStudyReport:
    """Evaluate ``study`` given one cohort list per block (aligned to study.blocks)."""
    if len(block_cohorts) != len(study.blocks):
        raise ValueError("one cohort list per block is required")

    fields = list(study.varied_engine_fields)
    constraints = [
        (field, *SLO_RULES[field], threshold)
        for field, threshold in study.slo.model_dump().items()
        if threshold is not None
    ]
    direction = OBJECTIVE_DIRECTIONS[study.objective]
    num_blocks = len(study.blocks)
    width = len(study.blocks[0].experiment_ids)

    block_results: list[BlockResult] = []
    engine_values_by_block: list[list[dict]] = []
    all_cross_fps: set[str] = set()

    for block_spec, cohorts in zip(study.blocks, block_cohorts):
        compatible = prepare_compatible_cohorts(
            cohorts, varied_engine_fields=fields, min_runs=study.min_runs
        )
        actual_ids = [s.experiment_id for s in compatible.summaries]
        if actual_ids != block_spec.experiment_ids:
            raise ValueError(
                f"block seed={block_spec.seed}: cohort ids/order must match the block spec"
            )
        # Every candidate in the block must share the effective arrival seed exactly.
        # For legacy configs this property falls back to workload.seed, preserving
        # the pre-split-seed behavior.
        for runs in cohorts:
            seed = runs[0][1].config.workload.effective_arrival_seed
            if seed != block_spec.seed:
                raise ValueError(
                    f"block seed={block_spec.seed}: a candidate ran with "
                    f"effective arrival seed {seed}"
                )
            for _, result in runs:
                all_cross_fps.add(cross_block_fingerprint(result, fields))

        candidates: list[BlockCandidateResult] = []
        for index, summary in enumerate(compatible.summaries):
            checks = _slo_checks(summary, constraints)
            candidates.append(
                BlockCandidateResult(
                    experiment_id=summary.experiment_id,
                    engine_values=compatible.engine_values[index],
                    slo_checks=checks,
                    feasible=all(c.passed for c in checks),
                    objective_mean=summary.metrics[study.objective].mean,
                )
            )
        block_results.append(BlockResult(seed=block_spec.seed, candidates=candidates))
        engine_values_by_block.append(compatible.engine_values)

    # Rectangular design: candidate engine values + ordering identical across blocks.
    reference = engine_values_by_block[0]
    for values in engine_values_by_block[1:]:
        if values != reference:
            raise ValueError(
                "blocks do not form the same rectangular design "
                "(candidate engine values/order differ across blocks)"
            )
    # Cross-block context: only workload seed + human ids may differ.
    if len(all_cross_fps) != 1:
        raise ValueError(
            "blocks differ outside workload seed and the declared engine fields"
        )
    context_fingerprint = next(iter(all_cross_fps))

    # Aggregate each rectangular candidate across blocks (compute ranks first so
    # each RobustCandidate is constructed already consistent).
    agg = []
    for index in range(width):
        block_means = [block_results[j].candidates[index].objective_mean for j in range(num_blocks)]
        blocks_feasible = sum(
            1 for j in range(num_blocks) if block_results[j].candidates[index].feasible
        )
        agg.append({
            "candidate_index": index,
            "engine_values": reference[index],
            "experiment_ids": [block_results[j].candidates[index].experiment_id for j in range(num_blocks)],
            "block_objective_means": block_means,
            "mean_objective_mean": statistics.mean(block_means),
            "blocks_feasible": blocks_feasible,
            "robust_feasible": blocks_feasible == num_blocks,
        })

    robust_means = sorted(
        {a["mean_objective_mean"] for a in agg if a["robust_feasible"]},
        reverse=direction == "higher_is_better",
    )
    ranks = {value: rank for rank, value in enumerate(robust_means, start=1)}

    candidates: list[RobustCandidate] = [
        RobustCandidate(
            **a,
            num_blocks=num_blocks,
            objective_direction=direction,
            rank=ranks[a["mean_objective_mean"]] if a["robust_feasible"] else None,
        )
        for a in agg
    ]

    best = [c.candidate_index for c in candidates if c.rank == 1]
    status = "no_feasible_candidate" if not best else "selected" if len(best) == 1 else "tie"
    return BlockedStudyReport(
        study=study,
        context_fingerprint=context_fingerprint,
        blocks=block_results,
        candidates=candidates,
        status=status,
        recommended_engine_values=candidates[best[0]].engine_values if len(best) == 1 else None,
        best_candidate_indices=best,
    )
