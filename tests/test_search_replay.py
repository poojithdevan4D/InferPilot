"""Outcome-blind, fixed-budget search replay baselines."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from inferpilot.comparison import (
    BlockCandidateResult,
    BlockResult,
    BlockedStudyReport,
    BlockedStudySpec,
    RobustCandidate,
    SLOCheck,
)
from inferpilot.search import (
    ReplaySearchReport,
    ReplaySearchSpec,
    candidate_order,
    evaluate_replay_search,
)


def _source(
    feasible_indices: set[int] = {2, 3},
    objective_values: tuple[float, ...] = (6.0, 7.0, 8.0, 8.2),
) -> BlockedStudyReport:
    seeds = (10, 11, 12)
    width = len(objective_values)
    study = BlockedStudySpec(
        study_id="source",
        objective="tpot_p95_ms",
        slo={"ttft_p95_ms": 250},
        varied_engine_fields=["max_num_seqs"],
        blocks=[
            {
                "seed": seed,
                "experiment_ids": [f"source-s{seed}-seq{i + 1}" for i in range(width)],
            }
            for seed in seeds
        ],
        min_runs=1,
    )
    blocks = []
    for seed in seeds:
        cells = []
        for index, objective in enumerate(objective_values):
            passed = index in feasible_indices
            cells.append(
                BlockCandidateResult(
                    experiment_id=f"source-s{seed}-seq{index + 1}",
                    engine_values={"max_num_seqs": index + 1},
                    slo_checks=[
                        SLOCheck(
                            slo_field="ttft_p95_ms",
                            metric="ttft_p95_ms",
                            operator="<=",
                            threshold=250,
                            observed_worst=100 if passed else 500,
                            passed=passed,
                        )
                    ],
                    feasible=passed,
                    objective_mean=objective,
                )
            )
        blocks.append(BlockResult(seed=seed, candidates=cells))

    feasible_values = sorted(
        {objective_values[index] for index in feasible_indices}
    )
    ranks = {value: rank for rank, value in enumerate(feasible_values, 1)}
    candidates = []
    for index, objective in enumerate(objective_values):
        feasible = index in feasible_indices
        candidates.append(
            RobustCandidate(
                candidate_index=index,
                engine_values={"max_num_seqs": index + 1},
                experiment_ids=[f"source-s{seed}-seq{index + 1}" for seed in seeds],
                block_objective_means=[objective] * len(seeds),
                mean_objective_mean=objective,
                blocks_feasible=len(seeds) if feasible else 0,
                num_blocks=len(seeds),
                robust_feasible=feasible,
                objective_direction="lower_is_better",
                rank=ranks[objective] if feasible else None,
            )
        )
    best = [
        index
        for index in feasible_indices
        if ranks[objective_values[index]] == 1
    ]
    return BlockedStudyReport(
        study=study,
        context_fingerprint="test-context",
        blocks=blocks,
        candidates=candidates,
        status="no_feasible_candidate" if not best else "selected" if len(best) == 1 else "tie",
        recommended_engine_values=candidates[best[0]].engine_values if len(best) == 1 else None,
        best_candidate_indices=best,
    )


def _spec(**updates) -> ReplaySearchSpec:
    values = {
        "search_id": "baseline",
        "policy": "declared_order-v1",
        "budget": 3,
        "seed": 0,
    }
    values.update(updates)
    return ReplaySearchSpec(**values)


def test_declared_order_budget_can_stop_without_feasible_candidate() -> None:
    report = evaluate_replay_search(_source(), _spec(budget=2))
    assert [trial.candidate_index for trial in report.trials] == [0, 1]
    assert report.status == "budget_exhausted_no_feasible"
    assert report.search_space_exhausted is False
    assert report.trials_to_first_feasible is None
    assert report.oracle_hit is False
    assert report.simple_regret is None


def test_declared_order_finds_oracle_at_trial_three() -> None:
    report = evaluate_replay_search(_source(), _spec())
    assert report.status == "selected"
    assert report.trials_to_first_feasible == 3
    assert report.best_candidate_indices == [2]
    assert report.recommended_engine_values == {"max_num_seqs": 3}
    assert report.oracle_hit is True
    assert report.simple_regret == 0
    assert ReplaySearchReport.model_validate_json(report.model_dump_json()) == report


def test_replay_reports_positive_simple_regret_for_suboptimal_observation() -> None:
    source = _source(feasible_indices={1, 2}, objective_values=(6.0, 8.0, 7.0, 9.0))
    report = evaluate_replay_search(source, _spec(budget=2))
    assert report.best_candidate_indices == [1]
    assert report.oracle_best_candidate_indices == [2]
    assert report.oracle_hit is False
    assert report.simple_regret == 1.0


def test_seeded_random_order_is_stable_and_outcome_blind() -> None:
    order = candidate_order("seeded_random-v1", 8, seed=17)
    assert order == candidate_order("seeded_random-v1", 8, seed=17)
    assert sorted(order) == list(range(8))
    assert any(candidate_order("seeded_random-v1", 8, seed=s) != order for s in range(18, 25))

    spec = _spec(policy="seeded_random-v1", budget=4, seed=17)
    a = evaluate_replay_search(_source(), spec)
    b = evaluate_replay_search(
        _source(feasible_indices={0}, objective_values=(9.0, 8.0, 7.0, 6.0)), spec
    )
    assert [t.candidate_index for t in a.trials] == [t.candidate_index for t in b.trials]


def test_full_search_distinguishes_proven_exhaustion() -> None:
    report = evaluate_replay_search(_source(feasible_indices=set()), _spec(budget=4))
    assert report.status == "search_space_exhausted_no_feasible"
    assert report.search_space_exhausted is True
    assert report.oracle_status == "no_feasible_candidate"
    assert report.best_candidate_indices == []


def test_budget_cannot_exceed_candidate_count() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        evaluate_replay_search(_source(), _spec(budget=5))


@pytest.mark.parametrize(
    "mutator,message",
    [
        (lambda raw: raw.__setitem__("report_version", "9.9.9"), "unsupported replay"),
        (lambda raw: raw["trials"][0].__setitem__("candidate_index", 3), "trial order"),
        (lambda raw: raw["trials"][0].__setitem__("mean_objective_mean", 999), "source candidate"),
        (lambda raw: raw.__setitem__("trials_to_first_feasible", 1), "trials_to_first"),
        (lambda raw: raw.__setitem__("oracle_hit", False), "oracle_hit"),
        (lambda raw: raw.__setitem__("simple_regret", 1), "simple_regret"),
    ],
)
def test_report_rejects_independent_tampering(mutator, message) -> None:
    raw = evaluate_replay_search(_source(), _spec()).model_dump(mode="json")
    mutated = deepcopy(raw)
    mutator(mutated)
    with pytest.raises(ValidationError, match=message):
        ReplaySearchReport.model_validate(mutated)


# --- optional cost-aware overlay (M2C) -------------------------------------- #

_COSTS = [1.0, 2.0, 3.0, 4.0]  # server-process-seconds per candidate


def test_cost_overlay_metric_without_cap() -> None:
    report = evaluate_replay_search(_source(), _spec(budget=3), candidate_costs_s=_COSTS)
    assert report.cost is not None
    assert report.cost.stopped_on == "candidate_budget"
    assert report.cost.affordable_trials == 3
    assert [t.candidate_index for t in report.cost.trials] == [0, 1, 2]
    assert report.cost.total_server_process_seconds == 6.0  # 1+2+3


def test_cost_budget_stops_before_candidate_budget() -> None:
    report = evaluate_replay_search(
        _source(), _spec(budget=3), candidate_costs_s=_COSTS, cost_budget_s=3.5
    )
    assert report.cost.stopped_on == "cost_budget"
    assert [t.candidate_index for t in report.cost.trials] == [0, 1]  # 1+2=3<=3.5; +3 exceeds
    assert report.cost.total_server_process_seconds == 3.0
    # candidate-count replay is preserved regardless of the cost cap
    assert [t.candidate_index for t in report.trials] == [0, 1, 2]


def test_no_cost_args_leaves_cost_none() -> None:
    report = evaluate_replay_search(_source(), _spec(budget=3))
    assert report.cost is None


def test_cost_budget_requires_costs() -> None:
    with pytest.raises(ValueError, match="requires candidate_costs_s"):
        evaluate_replay_search(_source(), _spec(budget=3), cost_budget_s=5.0)


def test_wrong_length_costs_rejected() -> None:
    with pytest.raises(ValueError, match="one cost per candidate"):
        evaluate_replay_search(_source(), _spec(budget=3), candidate_costs_s=[1.0, 2.0])


def test_tampered_cost_order_rejected() -> None:
    report = evaluate_replay_search(_source(), _spec(budget=3), candidate_costs_s=_COSTS)
    raw = report.model_dump(mode="json")
    raw["cost"]["trials"][0]["candidate_index"] = 99  # break policy-order agreement
    with pytest.raises(ValidationError, match="cost overlay order must match"):
        ReplaySearchReport.model_validate(raw)
