"""phases.json participates in bundle integrity; old (no-phases) bundles stay valid."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from inferpilot import (
    AggregateMetrics,
    EffectiveConfig,
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ResourceTelemetry,
    RunnerPhaseTiming,
    WorkloadSpec,
)
from inferpilot.comparison.store import ResultStore
from inferpilot.runner.phases import PhaseTimer


def _eligible_result() -> ExperimentResult:
    config = ExperimentConfig(
        experiment_id="store-phz", name="store-phz",
        engine=EngineConfig(model="org/m", revision="abc", max_model_len=2048,
                            max_num_seqs=1, gpu_memory_utilization=0.85,
                            enable_prefix_caching=False),
        workload=WorkloadSpec(name="w", num_requests=2, warmup_requests=0, prompt_tokens=8,
                              output_tokens=8, max_concurrency=1, ignore_eos=True),
    )
    agg = AggregateMetrics(
        num_requests=2, num_successful=2, num_failed=0, duration_s=5.0,
        ttft_p50_ms=10, ttft_p95_ms=12, ttft_p99_ms=13, tpot_p50_ms=6, tpot_p95_ms=7,
        tpot_p99_ms=8, e2e_p50_ms=100, e2e_p95_ms=120, e2e_p99_ms=130,
        throughput_tokens_per_s=50, throughput_requests_per_s=2, total_output_tokens=16,
        gpu_memory_peak_mb=3000,
    )
    tel = ResourceTelemetry(
        sample_interval_s=0.25, num_samples=10, peak_gpu_memory_mb=3000,
        gpu_utilization_mean_pct=80, gpu_utilization_peak_pct=95,
        kv_cache_usage_mean_perc=0.1, kv_cache_usage_peak_perc=0.2,
    )
    eff = EffectiveConfig(
        model="org/m", revision="abc", dtype="bfloat16", max_model_len=2048, max_num_seqs=1,
        enable_prefix_caching=False, enable_chunked_prefill=True, kv_cache_dtype="auto",
        gpu_memory_utilization=0.85, generation_config="vllm", verified=True,
    )
    return ExperimentResult(config=config, environment=EnvironmentMetadata(),
                            status=ExperimentStatus.COMPLETED, aggregates=agg,
                            effective_config=eff, telemetry=tel)


def _timing() -> RunnerPhaseTiming:
    t = PhaseTimer()
    t.launch()
    origin = t._origin  # noqa: SLF001
    for name, off in {"server_ready": 0.5, "config_verify_start": 0.6, "config_verify_end": 0.7,
                      "measured_start": 1.0, "measured_end": 6.0, "finalize_start": 6.0,
                      "finalize_end": 6.1, "teardown_start": 6.1, "teardown_end": 6.3}.items():
        t.mark_at(name, origin + off)
    return t.build("completed", aggregate_duration_s=5.0)


def _write_bundle(run_dir: Path, *, with_phases: bool) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(_eligible_result().model_dump_json(indent=2))
    if with_phases:
        (run_dir / "phases.json").write_text(json.dumps(_timing().model_dump(mode="json")))


def test_phases_included_in_bundle_integrity(tmp_path) -> None:
    store = ResultStore(tmp_path / "store")
    src = tmp_path / "run-a"
    _write_bundle(src, with_phases=True)
    ingested = store.ingest(src)

    # manifest covers phases.json
    manifest = json.loads((ingested.path / "manifest.json").read_text())
    assert "phases.json" in manifest
    assert store.load(ingested.run_id).status is ExperimentStatus.COMPLETED

    # tampering the stored phases.json breaks integrity
    stored = ingested.path / "phases.json"
    stored.chmod(stat.S_IWUSR | stat.S_IRUSR)
    stored.write_text(stored.read_text() + " ")
    with pytest.raises(ValueError, match="bundle integrity"):
        store.load(ingested.run_id)


def test_old_bundle_without_phases_still_valid(tmp_path) -> None:
    store = ResultStore(tmp_path / "store")
    src = tmp_path / "run-old"
    _write_bundle(src, with_phases=False)  # schema-0.3.0 run predating M2C
    ingested = store.ingest(src)
    manifest = json.loads((ingested.path / "manifest.json").read_text())
    assert "phases.json" not in manifest
    assert store.load(ingested.run_id).status is ExperimentStatus.COMPLETED
