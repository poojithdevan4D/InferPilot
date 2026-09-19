"""Zero-GPU end-to-end canary for exact recomputation mechanism evidence."""

from __future__ import annotations

import argparse
import json
import stat
import sys
import tempfile
from pathlib import Path

from inferpilot import (
    EngineConfig,
    ExperimentConfig,
    MechanismEvidence,
    WorkloadSpec,
    evaluate_mechanism_canary,
)
from inferpilot.runner.artifacts import MECHANISM_EVIDENCE_FILENAME
from inferpilot.runner.metrics_capabilities import FP8_MECHANISM_REQUIREMENTS
from inferpilot.runner.orchestrator import run_experiment

FAKE_SERVER = Path(__file__).resolve().parents[1] / "tests" / "fake_vllm_server.py"


def _config(experiment_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        name=experiment_id,
        description="Synthetic mechanism canary; plumbing proof only.",
        engine=EngineConfig(
            model="fake/model",
            revision="deadbeef",
            max_model_len=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=False,
            extra_args={"async-scheduling": False},
        ),
        workload=WorkloadSpec(
            name="mechanism-canary",
            num_requests=6,
            warmup_requests=1,
            prompt_tokens=128,
            output_tokens=8,
            max_concurrency=1,
            temperature=0.0,
            ignore_eos=True,
            seed=19,
        ),
        tags=["dry-run", "no-gpu", "mechanism-canary"],
    )


def _command(pressure: bool):
    def build(port: int) -> list[str]:
        return [
            sys.executable,
            str(FAKE_SERVER),
            "--port",
            str(port),
            "--mode",
            "normal",
            "--output-tokens",
            "8",
            "--emit-effective",
            "False",
            "--emit-iteration-details",
            "--expose-recomputed-metric",
            "--preemptions-per-request",
            "1" if pressure else "0",
            "--recomputed-tokens-per-request",
            "4" if pressure else "0",
        ]

    return build


def _run_cell(output_dir: Path, name: str, pressure: bool) -> MechanismEvidence:
    before = set(output_dir.iterdir())
    result = run_experiment(
        _config(name),
        str(output_dir),
        command_builder=_command(pressure),
        ready_timeout_s=15.0,
        metric_requirements=FP8_MECHANISM_REQUIREMENTS,
        require_metric_capabilities=True,
    )
    if result.status.value != "completed":
        raise RuntimeError(f"{name} failed: {result.failure}")
    created = set(output_dir.iterdir()) - before
    if len(created) != 1:
        raise RuntimeError(f"{name}: expected one run directory, found {len(created)}")
    path = created.pop() / MECHANISM_EVIDENCE_FILENAME
    return MechanismEvidence.model_validate_json(path.read_text())


def run_canary(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    control = _run_cell(output_dir, "mechanism-canary-control", False)
    pressure = _run_cell(output_dir, "mechanism-canary-pressure", True)
    report = evaluate_mechanism_canary(control, pressure)
    report_path = output_dir / "mechanism-canary-report.json"
    if report_path.exists():
        raise FileExistsError(f"canary report already exists: {report_path}")
    report_path.write_text(report.model_dump_json(indent=2))
    report_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    summary = {
        "report_version": report.report_version,
        "canary": "PASS" if report.passed else "FAIL",
        "control_recomputed_tokens": control.recomputed_token_executions,
        "pressure_recomputed_tokens": pressure.recomputed_token_executions,
        "pressure_preemptions": pressure.preemptions,
        "control_iterations": len(control.iterations),
        "pressure_iterations": len(pressure.iterations),
        "reasons": list(report.reasons),
        "report_path": str(report_path),
    }
    if not report.passed:
        raise RuntimeError(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        print(json.dumps(run_canary(args.output_dir), indent=2))
        return 0
    with tempfile.TemporaryDirectory(
        prefix="inferpilot-mechanism-canary-"
    ) as directory:
        summary = run_canary(Path(directory))
        summary["report_path"] = "temporary (removed after successful verification)"
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
