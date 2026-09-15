"""Blocked multi-seed confirmatory study: robustness gate + negative cases."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import (
    AggregateMetrics,
    EffectiveConfig,
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    HardwareInfo,
    ResourceTelemetry,
    ToolchainInfo,
    WorkloadSpec,
)
from inferpilot.comparison import (
    BlockedStudyReport,
    BlockedStudySpec,
    evaluate_blocked_study,
)


def _run(
    experiment_id: str, seed: int, seqs: int, *,
    ttft95: float, throughput: float, gpu_name: str = "GPU-A",
    kv_dtype: str = "auto", arrival_seed: int | None = None,
) -> ExperimentResult:
    config = ExperimentConfig(
        experiment_id=experiment_id, name=experiment_id,
        engine=EngineConfig(
            model="org/model", revision="abc", max_model_len=2048, max_num_seqs=seqs,
            max_num_batched_tokens=2048, gpu_memory_utilization=0.85,
            enable_prefix_caching=False, sampler_backend="pytorch", kv_cache_dtype=kv_dtype,
        ),
        workload=WorkloadSpec(
            name="c6", num_requests=32, warmup_requests=4, prompt_tokens=128,
            output_tokens=32, max_concurrency=4, temperature=0, ignore_eos=True, seed=seed,
            arrival_seed=arrival_seed,
        ),
    )
    environment = EnvironmentMetadata(
        platform="Linux-test", python_version="3.12.0", torch_version="2.13.0",
        vllm_version="0.29.0",
        hardware=HardwareInfo(gpu_name=gpu_name, gpu_count=1, gpu_memory_total_mb=4096),
        toolchain=ToolchainInfo(torch_cuda_version="13.0"),
        effective_sampler_backend="pytorch",
        runtime_overrides={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
    )
    effective = EffectiveConfig(
        model="org/model", revision="abc", dtype="bfloat16", max_model_len=2048,
        max_num_seqs=seqs, max_num_batched_tokens=2048, enable_prefix_caching=False,
        enable_chunked_prefill=True, sampler_backend="pytorch", kv_cache_dtype="auto",
        gpu_memory_utilization=0.85, generation_config="vllm", verified=True,
    )
    aggregates = AggregateMetrics(
        num_requests=32, num_successful=32, num_failed=0, duration_s=10,
        ttft_p50_ms=ttft95 * 0.5, ttft_p95_ms=ttft95, ttft_p99_ms=ttft95 * 1.1,
        tpot_p50_ms=6.0, tpot_p95_ms=6.5, tpot_p99_ms=7.0,
        e2e_p50_ms=210, e2e_p95_ms=230, e2e_p99_ms=240,
        throughput_tokens_per_s=throughput, throughput_requests_per_s=throughput / 32,
        total_output_tokens=1024, gpu_memory_peak_mb=3000,
    )
    telemetry = ResourceTelemetry(
        sample_interval_s=0.25, num_samples=20, peak_gpu_memory_mb=3000,
        gpu_utilization_mean_pct=90, gpu_utilization_peak_pct=100,
        kv_cache_usage_mean_perc=0.1, kv_cache_usage_peak_perc=0.2,
    )
    return ExperimentResult(
        config=config, environment=environment, status=ExperimentStatus.COMPLETED,
        aggregates=aggregates, effective_config=effective, telemetry=telemetry,
    )


def _cell(seq_label: str, seed: int, seqs: int, *, ttft95, throughput, **kw):
    """One candidate cell in one block: a single-run cohort (min_runs=1)."""
    eid = f"c6-{seq_label}-s{seed}"
    return [(f"{eid}-0", _run(eid, seed, seqs, ttft95=ttft95, throughput=throughput, **kw))]


# ttft95 per (seq, seed): seq2 always <=40 (passes), seq1 spikes at seed 2 (fails).
_TTFT = {("seq1", 0): 30, ("seq1", 1): 32, ("seq1", 2): 90,
         ("seq2", 0): 20, ("seq2", 1): 22, ("seq2", 2): 24}
_THR = {"seq1": 150, "seq2": 240}


def _spec(**kw) -> BlockedStudySpec:
    base = dict(
        study_id="c6", objective="throughput_tokens_per_s",
        slo={"ttft_p95_ms": 40}, varied_engine_fields=["max_num_seqs"],
        blocks=[
            {"seed": 0, "experiment_ids": ["c6-seq1-s0", "c6-seq2-s0"]},
            {"seed": 1, "experiment_ids": ["c6-seq1-s1", "c6-seq2-s1"]},
            {"seed": 2, "experiment_ids": ["c6-seq1-s2", "c6-seq2-s2"]},
        ],
    )
    base.update(kw)
    return BlockedStudySpec(**base)


def _block_cohorts(**overrides):
    """Rectangular cohorts for 3 seeds x {seq1,seq2}. `overrides` mutate one cell."""
    cohorts = []
    for seed in (0, 1, 2):
        row = [
            _cell("seq1", seed, 1, ttft95=_TTFT[("seq1", seed)], throughput=_THR["seq1"]),
            _cell("seq2", seed, 2, ttft95=_TTFT[("seq2", seed)], throughput=_THR["seq2"]),
        ]
        cohorts.append(row)
    ov = overrides.get("mutate")
    if ov:
        ov(cohorts)
    return cohorts


# --- positive --------------------------------------------------------------- #


def test_robust_feasible_selection_across_blocks() -> None:
    report = evaluate_blocked_study(_block_cohorts(), _spec())
    # seq1 fails the SLO at seed 2 (ttft95=90>40); seq2 passes every block.
    seq1, seq2 = report.candidates
    assert seq1.robust_feasible is False and seq1.blocks_feasible == 2
    assert seq2.robust_feasible is True and seq2.blocks_feasible == 3
    assert report.status == "selected"
    assert report.best_candidate_indices == [1]
    assert report.recommended_engine_values == {"max_num_seqs": 2}
    assert len(report.blocks) == 3  # per-block results preserved


def test_report_roundtrips() -> None:
    report = evaluate_blocked_study(_block_cohorts(), _spec())
    assert BlockedStudyReport.model_validate_json(report.model_dump_json()) == report
    assert report.report_version == "0.1.1"


def test_split_arrival_seed_defines_blocks_while_legacy_seed_stays_fixed() -> None:
    cohorts = []
    for block_seed in (10, 11, 12):
        cohorts.append([
            [("a", _run(
                f"split-seq1-s{block_seed}", 0, 1, ttft95=30,
                throughput=150, arrival_seed=block_seed,
            ))],
            [("b", _run(
                f"split-seq2-s{block_seed}", 0, 2, ttft95=20,
                throughput=240, arrival_seed=block_seed,
            ))],
        ])
    spec = _spec(blocks=[
        {
            "seed": block_seed,
            "experiment_ids": [
                f"split-seq1-s{block_seed}", f"split-seq2-s{block_seed}",
            ],
        }
        for block_seed in (10, 11, 12)
    ])
    report = evaluate_blocked_study(cohorts, spec)
    assert report.status == "selected"
    assert report.best_candidate_indices == [1]


# --- negative --------------------------------------------------------------- #


def test_duplicate_seeds_rejected() -> None:
    with pytest.raises(ValidationError, match="distinct workload seeds"):
        _spec(blocks=[
            {"seed": 0, "experiment_ids": ["c6-seq1-s0", "c6-seq2-s0"]},
            {"seed": 0, "experiment_ids": ["c6-seq1-s1", "c6-seq2-s1"]},
            {"seed": 2, "experiment_ids": ["c6-seq1-s2", "c6-seq2-s2"]},
        ])


def test_missing_cell_rejected() -> None:
    def drop(cohorts):
        cohorts[2] = [cohorts[2][0]]  # block seed=2 missing the seq2 candidate
    with pytest.raises(ValueError, match="at least two cohorts"):
        evaluate_blocked_study(_block_cohorts(mutate=drop), _spec())


def test_candidate_id_mismatch_rejected() -> None:
    def rename(cohorts):
        run = cohorts[0][0][0][1]
        cohorts[0][0] = [("x", run)]  # cohort still c6-seq1-s0 by config id; spec unchanged
    # Change the spec's expected id instead -> actual != expected.
    spec = _spec(blocks=[
        {"seed": 0, "experiment_ids": ["c6-seqX-s0", "c6-seq2-s0"]},
        {"seed": 1, "experiment_ids": ["c6-seq1-s1", "c6-seq2-s1"]},
        {"seed": 2, "experiment_ids": ["c6-seq1-s2", "c6-seq2-s2"]},
    ])
    with pytest.raises(ValueError, match="ids/order must match"):
        evaluate_blocked_study(_block_cohorts(), spec)


def test_non_seed_context_drift_rejected() -> None:
    def drift(cohorts):
        # Whole block seed=2 ran on different hardware (compatible within block,
        # but differs across blocks on something other than seed).
        cohorts[2] = [
            _cell("seq1", 2, 1, ttft95=30, throughput=150, gpu_name="GPU-B"),
            _cell("seq2", 2, 2, ttft95=24, throughput=240, gpu_name="GPU-B"),
        ]
    with pytest.raises(ValueError, match="outside workload seed"):
        evaluate_blocked_study(_block_cohorts(mutate=drift), _spec())


def test_incompatible_engine_change_within_block_rejected() -> None:
    def tweak(cohorts):
        # seq2 at seed 0 also changed a non-declared engine field (kv_cache_dtype).
        cohorts[0][1] = _cell("seq2", 0, 2, ttft95=20, throughput=240, kv_dtype="fp8")
    with pytest.raises(ValueError, match="outside the explicitly varied"):
        evaluate_blocked_study(_block_cohorts(mutate=tweak), _spec())


def test_rectangular_design_mismatch_rejected() -> None:
    def bend(cohorts):
        # seed=2 uses max_num_seqs=3 for the second candidate -> not the same design.
        cohorts[2][1] = _cell("seq2", 2, 3, ttft95=24, throughput=240)
    with pytest.raises(ValueError, match="rectangular design"):
        evaluate_blocked_study(_block_cohorts(mutate=bend), _spec())


def test_ineligible_run_rejected() -> None:
    def spoil(cohorts):
        bad = _run("c6-seq2-s0", 0, 2, ttft95=20, throughput=240)
        bad.telemetry = None  # -> not baseline eligible
        cohorts[0][1] = [("c6-seq2-s0-0", bad)]
    with pytest.raises(ValueError, match="baseline-ineligible"):
        evaluate_blocked_study(_block_cohorts(mutate=spoil), _spec())


def test_tampered_report_rejected() -> None:
    report = evaluate_blocked_study(_block_cohorts(), _spec())
    raw = report.model_dump(mode="json")
    # Flip the losing candidate to robust-feasible without fixing per-block data.
    raw["candidates"][0]["robust_feasible"] = True
    raw["candidates"][0]["blocks_feasible"] = 3
    raw["candidates"][0]["rank"] = 1  # keep candidate-level validator satisfied
    with pytest.raises(ValidationError, match="blocks_feasible is inconsistent"):
        BlockedStudyReport.model_validate(raw)


# --- independent per-field tampering (each mutates exactly one bound field) --- #


def _valid_raw() -> dict:
    return evaluate_blocked_study(_block_cohorts(), _spec()).model_dump(mode="json")


def _m(fn):
    raw = _valid_raw()
    fn(raw)
    return raw


TAMPERS = [
    # (label, mutator, expected message fragment)
    ("block_candidate_id_order",
     lambda r: r["blocks"][0]["candidates"].__setitem__(0, {**r["blocks"][0]["candidates"][0], "experiment_id": "zzz"}),
     "match StudyBlock.experiment_ids"),
    ("engine_value_keys",
     lambda r: r["blocks"][0]["candidates"][0]["engine_values"].__setitem__("extra", 9),
     "engine keys must equal varied_engine_fields"),
    ("position_engine_values",
     lambda r: r["blocks"][0]["candidates"][0]["engine_values"].__setitem__("max_num_seqs", 99),
     "block engine values must equal the robust candidate values"),
    ("slo_check_threshold",
     lambda r: r["blocks"][0]["candidates"][0]["slo_checks"][0].__setitem__("threshold", 999.0),
     "SLO checks must exactly match"),
    ("slo_check_passed",
     lambda r: (r["blocks"][0]["candidates"][1]["slo_checks"][0].__setitem__("passed", False),
                r["blocks"][0]["candidates"][1].__setitem__("feasible", False)),
     "passed is inconsistent"),
    ("block_feasibility",
     lambda r: r["blocks"][0]["candidates"][0].__setitem__("feasible", False),
     "feasible must agree with all SLO checks"),
    ("robust_experiment_ids",
     lambda r: r["candidates"][1]["experiment_ids"].__setitem__(0, "wrong"),
     "experiment_ids must match per-block ids"),
    ("block_objective_means",
     lambda r: r["candidates"][1]["block_objective_means"].__setitem__(0, 0.0),
     "block_objective_means inconsistent"),
    ("mean_objective_mean",
     lambda r: r["candidates"][1].__setitem__("mean_objective_mean", 0.0),
     "mean_objective_mean inconsistent"),
    ("blocks_feasible",
     lambda r: (r["candidates"][1].__setitem__("blocks_feasible", 2),
                r["candidates"][1].__setitem__("robust_feasible", False),
                r["candidates"][1].__setitem__("rank", None),
                r.__setitem__("best_candidate_indices", []),
                r.__setitem__("status", "no_feasible_candidate"),
                r.__setitem__("recommended_engine_values", None)),
     "blocks_feasible is inconsistent"),
    ("rank",
     lambda r: r["candidates"][1].__setitem__("rank", 2),
     "ranks are inconsistent"),
    ("best_candidate_indices",
     lambda r: r.__setitem__("best_candidate_indices", [0]),
     "best_candidate_indices must match"),
    ("status",
     lambda r: r.__setitem__("status", "tie"),
     "status is inconsistent"),
    ("recommended_engine_values",
     lambda r: r.__setitem__("recommended_engine_values", {"max_num_seqs": 1}),
     "recommended_engine_values inconsistent"),
    ("report_version",
     lambda r: r.__setitem__("report_version", "0.1.0"),
     "unsupported blocked report_version"),
]


@pytest.mark.parametrize("label,mutator,message", TAMPERS, ids=[t[0] for t in TAMPERS])
def test_tampered_bound_field_rejected(label, mutator, message) -> None:
    raw = _m(mutator)
    with pytest.raises(ValidationError, match=message):
        BlockedStudyReport.model_validate(raw)
