"""Mechanism evidence is aligned, exact, and self-validating."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferpilot import (
    MechanismCanaryReport,
    MechanismEvidence,
    evaluate_mechanism_canary,
)
from inferpilot.runner.mechanism_evidence import (
    build_mechanism_evidence,
    parse_scheduler_iterations,
)
from inferpilot.telemetry import ResourceSample

ROOT = Path(__file__).resolve().parents[1]

ROW = (
    "INFO Engine 000: Iteration(7): 2 context requests, 128 context tokens, "
    "3 generation requests, 24 generation tokens, iteration elapsed time: "
    "1.25 ms, GPU KV cache usage: 95.0%\n"
)


def _evidence() -> MechanismEvidence:
    samples = [
        ResourceSample(
            t_s=0.0,
            num_preemptions_total=3,
            recomputed_token_executions_total=10,
        ),
        ResourceSample(
            t_s=1.0,
            num_preemptions_total=5,
            recomputed_token_executions_total=18,
        ),
    ]
    raw = ROW.encode()
    return build_mechanism_evidence(
        experiment_id="mechanism-test",
        measured_duration_s=1.0,
        sample_interval_s=0.25,
        max_num_seqs=5,
        telemetry_samples=samples,
        counter_metric_name="vllm:recomputed_token_executions_total",
        stdout_slice=raw,
        stderr_slice=b"",
        stdout_bounds=(100, 100 + len(raw)),
        stderr_bounds=(20, 20),
    )


def test_iteration_parser_and_counter_delta_are_exact() -> None:
    rows = parse_scheduler_iterations(ROW)
    assert len(rows) == 1
    assert rows[0].effective_batch_size == 5

    evidence = _evidence()
    assert evidence.coverage_complete
    assert evidence.recomputed_token_executions == 8
    assert evidence.preemptions == 2
    assert evidence.scheduled_context_tokens == 128
    assert evidence.scheduled_generation_tokens == 24
    assert evidence.effective_batch_sizes == (5,)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("recomputed_token_executions", 7, "delta contradicts"),
        ("scheduled_context_tokens", 127, "context-token total"),
        ("effective_batch_sizes", [4], "batch sizes"),
        ("max_num_seqs", 4, "exceeds max_num_seqs"),
        ("counter_metric_name", "invented:counter", "Input should be"),
        ("coverage_complete", False, "coverage flag"),
    ],
)
def test_persisted_derivations_reject_tampering(field: str, value, match: str) -> None:
    payload = _evidence().model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError, match=match):
        MechanismEvidence.model_validate(payload)


def test_canary_fails_closed_when_pressure_counter_does_not_move() -> None:
    control = _evidence().model_copy(
        update={
            "recompute_start": 10,
            "recompute_end": 10,
            "recomputed_token_executions": 0,
            "preemption_start": 3,
            "preemption_end": 3,
            "preemptions": 0,
        }
    )
    control = MechanismEvidence.model_validate(control.model_dump(mode="json"))
    report = evaluate_mechanism_canary(control, control)
    assert report.passed is False
    assert report.reasons == (
        "pressure_preemptions_not_positive",
        "pressure_recomputation_not_positive",
    )

    payload = report.model_dump(mode="json")
    payload["passed"] = True
    with pytest.raises(ValidationError, match="verdict contradicts"):
        MechanismCanaryReport.model_validate(payload)


def test_zero_gpu_canary_runs_full_pipeline(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "mechanism_canary_dry_run.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["canary"] == "PASS"
    assert summary["control_recomputed_tokens"] == 0
    assert summary["pressure_recomputed_tokens"] == 24
    assert summary["pressure_preemptions"] == 6
    assert summary["control_iterations"] == 12
    assert summary["pressure_iterations"] == 12

    report_path = tmp_path / "mechanism-canary-report.json"
    report = MechanismCanaryReport.model_validate_json(report_path.read_text())
    assert report.passed is True
    assert report_path.stat().st_mode & 0o222 == 0
    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 2
    for run_dir in run_dirs:
        evidence = MechanismEvidence.model_validate_json(
            (run_dir / "mechanism-evidence.json").read_text()
        )
        assert evidence.coverage_complete
