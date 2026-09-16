"""Phase 1: independent single-field tamper tests for PolicyBenchmarkReport.

The report recomputes every score from its embedded spec + source studies on
load; these tests confirm each derived/persisted field is actually rebound.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import (
    AggregateMetrics, EffectiveConfig, EngineConfig, EnvironmentMetadata,
    ExperimentConfig, ExperimentResult, ExperimentStatus, HardwareInfo,
    ResourceTelemetry, ToolchainInfo, WorkloadSpec,
)
from inferpilot.comparison import BlockedStudySpec, evaluate_blocked_study
from inferpilot.search.policy import candidate_order
from inferpilot.search.policy_benchmark import (
    PolicyBenchmarkReport, PolicyBenchmarkSpec, PolicyDefinition,
    evaluate_policy_benchmark,
)

_SEQS = (1, 2, 4, 8)  # four distinct max_num_seqs candidates
_SEEDS = (10, 11, 12)


def _run(eid: str, seed: int, seqs: int, *, ttft95: float, throughput: float) -> ExperimentResult:
    config = ExperimentConfig(
        experiment_id=eid, name=eid,
        engine=EngineConfig(model="org/model", revision="abc", max_model_len=2048,
                            max_num_seqs=seqs, max_num_batched_tokens=2048,
                            gpu_memory_utilization=0.85, enable_prefix_caching=False),
        workload=WorkloadSpec(name="w", num_requests=32, warmup_requests=4, prompt_tokens=128,
                              output_tokens=32, max_concurrency=4, ignore_eos=True, seed=seed),
    )
    env = EnvironmentMetadata(
        platform="Linux-test", python_version="3.12.0", torch_version="2.13.0", vllm_version="0.29.0",
        hardware=HardwareInfo(gpu_name="GPU-A", gpu_count=1, gpu_memory_total_mb=4096),
        toolchain=ToolchainInfo(torch_cuda_version="13.0"),
        effective_sampler_backend="pytorch", runtime_overrides={"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    eff = EffectiveConfig(model="org/model", revision="abc", dtype="bfloat16", max_model_len=2048,
                          max_num_seqs=seqs, max_num_batched_tokens=2048, enable_prefix_caching=False,
                          enable_chunked_prefill=True, kv_cache_dtype="auto", gpu_memory_utilization=0.85,
                          generation_config="vllm", verified=True)
    agg = AggregateMetrics(num_requests=32, num_successful=32, num_failed=0, duration_s=10,
                           ttft_p50_ms=ttft95 * 0.5, ttft_p95_ms=ttft95, ttft_p99_ms=ttft95 * 1.1,
                           tpot_p50_ms=6.0, tpot_p95_ms=6.5, tpot_p99_ms=7.0,
                           e2e_p50_ms=210, e2e_p95_ms=230, e2e_p99_ms=240,
                           throughput_tokens_per_s=throughput, throughput_requests_per_s=throughput / 32,
                           total_output_tokens=1024, gpu_memory_peak_mb=3000)
    tel = ResourceTelemetry(sample_interval_s=0.25, num_samples=20, peak_gpu_memory_mb=3000,
                            gpu_utilization_mean_pct=90, gpu_utilization_peak_pct=100,
                            kv_cache_usage_mean_perc=0.1, kv_cache_usage_peak_perc=0.2)
    return ExperimentResult(config=config, environment=env, status=ExperimentStatus.COMPLETED,
                            aggregates=agg, effective_config=eff, telemetry=tel)


def _blocked(study_id: str, throughputs=(150, 200, 240, 260), ttfts=(200, 30, 25, 20)):
    """4-candidate blocked study; SLO ttft<=100 makes candidate 0 infeasible."""
    def eid(seed, seqs):
        return f"{study_id}-s{seed}-c{seqs}"
    blocks = [{"seed": s, "experiment_ids": [eid(s, q) for q in _SEQS]} for s in _SEEDS]
    spec = BlockedStudySpec(study_id=study_id, objective="throughput_tokens_per_s",
                            slo={"ttft_p95_ms": 100}, varied_engine_fields=["max_num_seqs"], blocks=blocks)
    block_cohorts = [
        [[(f"{eid(s, q)}-0", _run(eid(s, q), s, q, ttft95=t, throughput=thr))]
         for q, thr, t in zip(_SEQS, throughputs, ttfts)]
        for s in _SEEDS
    ]
    return evaluate_blocked_study(block_cohorts, spec)


def _spec() -> PolicyBenchmarkSpec:
    return PolicyBenchmarkSpec(
        benchmark_id="bench", task_ids=["t0", "t1"], candidate_count=4,
        policies=[
            PolicyDefinition(name="aware", kind="task_map", task_actions={"t0": 3, "t1": 3}),
            PolicyDefinition(name="static", kind="static", candidate_index=3),
            PolicyDefinition(name="random", kind="seeded_random", random_seeds=[0, 1, 2, 3, 4]),
        ],
    )


def _report() -> PolicyBenchmarkReport:
    return evaluate_policy_benchmark(_spec(), {"t0": _blocked("t0"), "t1": _blocked("t1")})


def test_report_roundtrips_and_recomputes() -> None:
    r = _report()
    assert PolicyBenchmarkReport.model_validate_json(r.model_dump_json()) == r


def test_random_ordering_is_outcome_blind() -> None:
    # candidate_order depends only on (policy, count, seed) — never on outcomes.
    a = candidate_order("seeded_random-v1", 4, 7)
    b = candidate_order("seeded_random-v1", 4, 7)
    assert a == b and sorted(a) == [0, 1, 2, 3]
    assert candidate_order("seeded_random-v1", 4, 7) != candidate_order("seeded_random-v1", 4, 8)


TAMPERS = [
    ("report_version", lambda d: d.__setitem__("report_version", "0.1.2"), "Input should be"),
    ("slo_success_rate", lambda d: d["scores"][0].__setitem__("slo_success_rate", 0.123), "inconsistent"),
    ("oracle_hit_rate", lambda d: d["scores"][1].__setitem__("oracle_hit_rate", 0.999), "inconsistent"),
    ("mean_regret", lambda d: d["scores"][0].__setitem__("mean_simple_regret_ms", 42.0), "inconsistent"),
    ("evaluations", lambda d: d["scores"][2].__setitem__("evaluations", 99), "inconsistent"),
    ("policy_name", lambda d: d["scores"][0].__setitem__("policy_name", "renamed"), "inconsistent"),
    ("candidate_count", lambda d: d["spec"].__setitem__("candidate_count", 3), None),
    ("drop_task", lambda d: d["sources"].pop("t1"), None),
]


@pytest.mark.parametrize("label,mutate,match", TAMPERS, ids=[t[0] for t in TAMPERS])
def test_single_field_tamper_rejected(label, mutate, match) -> None:
    raw = _report().model_dump(mode="json")
    mutate(raw)
    with pytest.raises(ValidationError if match == "Input should be" else (ValidationError, ValueError),
                       match=match):
        PolicyBenchmarkReport.model_validate(raw)
