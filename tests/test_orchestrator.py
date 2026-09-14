"""End-to-end orchestrator tests against the fake server subprocess.

Covers COMPLETED, request-failure, TIMEOUT, FAILED, and OOM outcomes, plus the
immutable artifact. No GPU, model, or vLLM required.
"""

from __future__ import annotations

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


def _builder(mode: str, output_tokens: int = 8):
    def build(port: int) -> list[str]:
        return [
            sys.executable, str(FAKE),
            "--port", str(port),
            "--mode", mode,
            "--output-tokens", str(output_tokens),
        ]
    return build


def _config(num_requests: int = 3, warmup: int = 1, output_tokens: int = 8) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp-orch",
        name="orch-test",
        engine=EngineConfig(model="fake/model", revision="deadbeef", max_num_seqs=1),
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
    assert result.is_baseline_eligible is True
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


def test_request_failures_still_complete_with_failed_measurements(tmp_path) -> None:
    cfg = _config(num_requests=2, warmup=0)
    result = run_experiment(
        cfg, str(tmp_path), command_builder=_builder("error500"), ready_timeout_s=15.0
    )
    assert result.status is ExperimentStatus.COMPLETED
    assert result.aggregates is not None
    assert result.aggregates.num_failed == 2
    assert result.aggregates.num_successful == 0


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
