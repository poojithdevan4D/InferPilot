"""End-to-end orchestrator tests against the fake server subprocess.

Covers COMPLETED, request-failure, TIMEOUT, FAILED, and OOM outcomes, plus the
immutable artifact. No GPU, model, or vLLM required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from inferpilot import (
    EngineConfig,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    WorkloadSpec,
)
from inferpilot.runner.artifacts import RESULT_FILENAME
from inferpilot.runner.orchestrator import run_experiment

FAKE = Path(__file__).parent / "fake_vllm_server.py"


def _builder(mode: str, output_tokens: int = 8, response_delay: float = 0.0):
    def build(port: int) -> list[str]:
        return [
            sys.executable, str(FAKE),
            "--port", str(port),
            "--mode", mode,
            "--output-tokens", str(output_tokens),
            "--response-delay", str(response_delay),
            "--emit-effective", "False",
        ]
    return build


def _config(num_requests: int = 3, warmup: int = 1, output_tokens: int = 8) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp-orch",
        name="orch-test",
        engine=EngineConfig(
            model="fake/model",
            revision="deadbeef",
            max_model_len=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=False,
        ),
        workload=WorkloadSpec(
            name="w", num_requests=num_requests, warmup_requests=warmup,
            prompt_tokens=128, output_tokens=output_tokens,
            max_concurrency=1, temperature=0.0, ignore_eos=True,
        ),
    )


def test_completed_run_produces_immutable_result(tmp_path) -> None:
    cfg = _config(num_requests=3, warmup=1, output_tokens=8)
    result = run_experiment(
        cfg, str(tmp_path), command_builder=_builder("normal", 8), ready_timeout_s=15.0
    )

    assert result.status is ExperimentStatus.COMPLETED
    # Core test environment intentionally lacks NVML, so request execution is
    # valid but the result cannot be promoted to a resource-complete baseline.
    assert result.is_baseline_eligible is False
    assert result.aggregates is not None
    # warm-up (1) excluded; only the 3 measured requests are aggregated.
    assert result.aggregates.num_requests == 3
    assert result.aggregates.num_successful == 3
    assert result.aggregates.throughput_tokens_per_s and result.aggregates.throughput_tokens_per_s > 0
    assert len(result.measurements) == 3
    # exact revision recorded in the stored config
    assert result.config.engine.revision == "deadbeef"

    # artifact exists, round-trips, and is immutable (read-only).
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    result_file = run_dirs[0] / RESULT_FILENAME
    assert result_file.exists()
    reloaded = ExperimentResult.model_validate_json(result_file.read_text())
    assert reloaded == result
    assert (run_dirs[0] / "server.stdout.log").exists()
    with pytest.raises(PermissionError):
        result_file.open("w")

    # telemetry present (KV scraped from the fake /metrics; NVML absent in test env)
    assert result.telemetry is not None
    assert result.telemetry.num_samples >= 1
    assert result.telemetry.kv_cache_usage_peak_perc == 0.42
    assert (run_dirs[0] / "telemetry.json").exists()
    # lifecycle classification written; fake server emits no ERROR lines
    lifecycle = json.loads((run_dirs[0] / "lifecycle.json").read_text())
    assert lifecycle["pre_teardown"] == []


def test_config_fidelity_mismatch_fails_before_measurement(tmp_path) -> None:
    # Server resolves enable_prefix_caching=True; we requested False -> mismatch.
    cfg = ExperimentConfig(
        experiment_id="exp-fid",
        name="fid",
        engine=EngineConfig(
            model="fake/model", revision="deadbeef", max_model_len=2048, max_num_seqs=1,
            gpu_memory_utilization=0.85, kv_cache_dtype="auto", enable_prefix_caching=False,
        ),
        workload=WorkloadSpec(
            name="w", num_requests=4, warmup_requests=2, prompt_tokens=128,
            output_tokens=8, max_concurrency=1, ignore_eos=True,
        ),
    )

    def builder(port: int) -> list[str]:
        return [
            sys.executable, str(FAKE), "--port", str(port),
            "--mode", "normal", "--emit-effective", "True",
        ]

    result = run_experiment(cfg, str(tmp_path), command_builder=builder, ready_timeout_s=15.0)
    assert result.status is ExperimentStatus.FAILED
    assert result.failure is not None
    assert result.failure.error_type == "ConfigFidelityMismatch"
    assert "enable_prefix_caching" in result.failure.message
    # measurement never ran
    assert result.measurements == []
    assert result.aggregates is None
    assert result.is_baseline_eligible is False
    # resolved value captured for provenance
    assert result.effective_config is not None
    assert result.effective_config.enable_prefix_caching is True


def test_request_failures_still_complete_with_failed_measurements(tmp_path) -> None:
    cfg = _config(num_requests=2, warmup=0)
    result = run_experiment(
        cfg, str(tmp_path), command_builder=_builder("error500"), ready_timeout_s=15.0
    )
    assert result.status is ExperimentStatus.COMPLETED  # orchestration completed
    assert result.aggregates is not None
    assert result.aggregates.num_failed == 2
    assert result.aggregates.num_successful == 0
    assert result.is_baseline_eligible is False  # but not a valid benchmark


def test_warmup_failure_prevents_measured_execution(tmp_path) -> None:
    cfg = _config(num_requests=5, warmup=2)
    result = run_experiment(
        cfg, str(tmp_path), command_builder=_builder("error500"), ready_timeout_s=15.0
    )
    assert result.status is ExperimentStatus.FAILED
    assert result.failure is not None
    assert result.failure.error_type == "WarmupFailure"
    assert result.aggregates is None
    assert result.measurements == []  # measured requests never ran
    assert result.is_baseline_eligible is False

    # warm-up diagnostics preserved separately, not mixed into measured aggregates.
    run_dir = list(tmp_path.iterdir())[0]
    warmup_file = run_dir / "warmup.json"
    assert warmup_file.exists()
    entries = json.loads(warmup_file.read_text())
    assert len(entries) == 2
    assert all(e["success"] is False for e in entries)


def _open_loop_config(num_requests=3, warmup=2, rate=20.0, seed=7) -> ExperimentConfig:
    base = _config(num_requests=num_requests, warmup=warmup)
    workload = WorkloadSpec(
        name="open-loop", num_requests=num_requests, warmup_requests=warmup,
        prompt_tokens=128, output_tokens=8, request_rate_qps=rate, seed=seed,
        ignore_eos=True,
    )
    return base.model_copy(update={"workload": workload})


def test_open_loop_run_produces_arrivals_and_excludes_warmups(tmp_path) -> None:
    cfg = _open_loop_config(num_requests=3, warmup=2, rate=20.0, seed=7)
    result = run_experiment(
        cfg, str(tmp_path), command_builder=_builder("normal", 8), ready_timeout_s=15.0
    )
    assert result.status is ExperimentStatus.COMPLETED
    assert len(result.measurements) == 3

    run_dir = list(tmp_path.iterdir())[0]
    arrivals = json.loads((run_dir / "arrivals.json").read_text())
    assert arrivals["algorithm"] == "poisson-v1"
    assert arrivals["seed"] == 7
    assert arrivals["scheduled_offsets_s"][0] == 0.0
    assert len(arrivals["scheduled_offsets_s"]) == 3
    assert len(arrivals["actual_dispatch_offsets_s"]) == 3

    # Warm-ups excluded from the measured arrival timeline: measured clock starts
    # at the arrival origin, so the first measured request starts near 0.
    assert min(m.start_time_s for m in result.measurements) < 0.2
    warm = json.loads((run_dir / "warmup.json").read_text())
    assert len(warm) == 2


def test_batched_open_loop_dispatches_burst_without_waiting(tmp_path) -> None:
    cfg = _open_loop_config(num_requests=6, warmup=0, rate=100.0, seed=7)
    workload_payload = cfg.workload.model_dump()
    workload_payload.update(arrival_pattern="batched-poisson-v1", burst_size=4)
    cfg.workload = WorkloadSpec(**workload_payload)
    result = run_experiment(
        cfg, str(tmp_path),
        command_builder=_builder("normal", 8, response_delay=0.3),
        ready_timeout_s=15.0,
    )
    assert result.status is ExperimentStatus.COMPLETED
    run_dir = list(tmp_path.iterdir())[0]
    arrivals = json.loads((run_dir / "arrivals.json").read_text())
    assert arrivals["algorithm"] == "batched-poisson-v1"
    assert arrivals["burst_size"] == 4
    assert arrivals["scheduled_offsets_s"][:4] == [0.0] * 4
    # All first-batch requests dispatch long before any 0.3-second response ends.
    assert max(arrivals["actual_dispatch_offsets_s"][:4]) < 0.2


def test_open_loop_request_failures_still_structured(tmp_path) -> None:
    cfg = _open_loop_config(num_requests=3, warmup=0, rate=20.0, seed=1)
    result = run_experiment(
        cfg, str(tmp_path), command_builder=_builder("error500"), ready_timeout_s=15.0
    )
    assert result.status is ExperimentStatus.COMPLETED
    assert result.aggregates is not None
    assert result.aggregates.num_failed == 3
    assert len(result.measurements) == 3
    assert all(not m.success for m in result.measurements)


def test_readiness_timeout_produces_timeout_result(tmp_path) -> None:
    cfg = _config(num_requests=1, warmup=0)
    result = run_experiment(
        cfg, str(tmp_path), command_builder=_builder("never_ready"), ready_timeout_s=2.0
    )
    assert result.status is ExperimentStatus.TIMEOUT
    assert result.is_baseline_eligible is False
    assert result.aggregates is None
    assert result.failure is not None
    assert result.failure.error_type == "ServerReadinessTimeout"
    # result still written
    assert (list(tmp_path.iterdir())[0] / RESULT_FILENAME).exists()


def test_startup_crash_produces_failed_result(tmp_path) -> None:
    cfg = _config(num_requests=1, warmup=0)
    result = run_experiment(
        cfg, str(tmp_path), command_builder=_builder("startup_crash"), ready_timeout_s=5.0
    )
    assert result.status is ExperimentStatus.FAILED
    assert result.failure is not None
    assert result.failure.error_type == "ServerStartupError"


def test_oom_crash_produces_oom_result(tmp_path) -> None:
    cfg = _config(num_requests=1, warmup=0)
    result = run_experiment(
        cfg, str(tmp_path), command_builder=_builder("oom_crash"), ready_timeout_s=5.0
    )
    assert result.status is ExperimentStatus.OOM
    assert result.failure is not None
    assert result.failure.error_type == "CudaOOM"
