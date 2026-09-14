"""Runner integration: phases.json is written on every terminal path."""

from __future__ import annotations

import sys
from pathlib import Path

from inferpilot import EngineConfig, ExperimentConfig, RunnerPhaseTiming, WorkloadSpec
from inferpilot.runner.orchestrator import run_experiment

FAKE = Path(__file__).parent / "fake_vllm_server.py"


def _builder(mode: str, output_tokens: int = 8, emit: str = "False"):
    def build(port: int) -> list[str]:
        return [
            sys.executable, str(FAKE), "--port", str(port), "--mode", mode,
            "--output-tokens", str(output_tokens), "--emit-effective", emit,
        ]
    return build


def _config(num_requests: int = 2, warmup: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="phz", name="phz",
        engine=EngineConfig(
            model="fake/model", revision="deadbeef", max_model_len=2048, max_num_seqs=1,
            gpu_memory_utilization=0.85, enable_prefix_caching=False,
        ),
        workload=WorkloadSpec(
            name="w", num_requests=num_requests, warmup_requests=warmup, prompt_tokens=128,
            output_tokens=8, max_concurrency=1, ignore_eos=True,
        ),
    )


def _timing(run_dir: Path) -> RunnerPhaseTiming:
    return RunnerPhaseTiming.model_validate_json((run_dir / "phases.json").read_text())


def _run(tmp_path, cfg, mode, **kw):
    result = run_experiment(cfg, str(tmp_path), command_builder=_builder(mode), **kw)
    run_dir = list(tmp_path.iterdir())[0]
    return result, _timing(run_dir)


def _spans(timing):
    return {p.name: p for p in timing.phases}


def test_success_records_all_phases(tmp_path) -> None:
    result, timing = _run(tmp_path, _config(2, 1), "normal", ready_timeout_s=15.0)
    assert result.status.value == "completed"
    assert timing.terminal_status == "completed"
    s = _spans(timing)
    for name in ("server_startup", "config_verification", "warmup",
                 "measured_window", "finalization", "teardown", "total_occupancy"):
        assert s[name].completed, name
    assert timing.aggregate_duration_s == result.aggregates.duration_s


def test_startup_failure_phase_never_completes(tmp_path) -> None:
    _result, timing = _run(tmp_path, _config(2, 0), "startup_crash", ready_timeout_s=5.0)
    assert timing.terminal_status == "failed"
    s = _spans(timing)
    assert s["server_startup"].began and not s["server_startup"].completed
    assert not s["measured_window"].began
    assert s["teardown"].completed and s["total_occupancy"].completed


def test_readiness_timeout_phase_never_completes(tmp_path) -> None:
    _result, timing = _run(tmp_path, _config(1, 0), "never_ready", ready_timeout_s=2.0)
    assert timing.terminal_status == "timeout"
    s = _spans(timing)
    assert s["server_startup"].began and not s["server_startup"].completed
    assert not s["config_verification"].began
    assert s["teardown"].completed


def test_warmup_failure_stops_before_measured(tmp_path) -> None:
    result, timing = _run(tmp_path, _config(3, 2), "error500", ready_timeout_s=15.0)
    assert result.failure is not None and result.failure.error_type == "WarmupFailure"
    s = _spans(timing)
    assert s["config_verification"].completed
    assert s["warmup"].completed  # warm-up ran (then a request failed)
    assert not s["measured_window"].began
    assert s["teardown"].completed


def test_finalization_failure_still_persists_phases(tmp_path, monkeypatch) -> None:
    # Inject a finalization failure AFTER the measured window completes: the run
    # ends FAILED, cleanup still runs, and phases.json is present and valid with
    # a completed measured window and a finalization that began but never finished.
    import inferpilot.runner.orchestrator as orch

    def _boom(*_a, **_k):
        raise RuntimeError("injected finalization failure")

    monkeypatch.setattr(orch, "write_telemetry", _boom)

    cfg = _config(2, 1)
    result = run_experiment(cfg, str(tmp_path), command_builder=_builder("normal"),
                            ready_timeout_s=15.0)
    assert result.status.value == "failed"
    run_dir = list(tmp_path.iterdir())[0]
    timing = _timing(run_dir)  # phases.json exists and self-validates
    assert timing.terminal_status == "failed"
    s = _spans(timing)
    assert s["measured_window"].completed
    assert s["finalization"].began and not s["finalization"].completed
    assert s["teardown"].completed  # cleanup completed
    assert timing.aggregate_duration_s is not None


def test_measurement_failure_still_completes_window(tmp_path) -> None:
    # error500 with no warm-up: measured window runs to completion though every
    # request fails -> COMPLETED (orchestration finished).
    result, timing = _run(tmp_path, _config(2, 0), "error500", ready_timeout_s=15.0)
    assert result.status.value == "completed"
    s = _spans(timing)
    assert s["measured_window"].completed and s["finalization"].completed
    assert s["teardown"].completed
