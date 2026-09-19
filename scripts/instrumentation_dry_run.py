"""Zero-GPU end-to-end proof of the instrumentation and diagnosis chain.

The fake server deliberately exposes the ordinary queue/KV/preemption metrics
but not the exact scheduler/recompute counters required by the fp8 mechanism
study.  A successful run therefore has two simultaneous outcomes:

* pipeline: PASS (valid aligned evidence reaches ``diagnose``), and
* real-pilot gate: NO_GO (required mechanism metrics are still absent).

Run from the repository root:
    uv run python scripts/instrumentation_dry_run.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from inferpilot import EngineConfig, ExperimentConfig, ExperimentResult, WorkloadSpec, diagnose
from inferpilot.runner.artifacts import METRICS_CAPABILITIES_FILENAME, RESULT_FILENAME
from inferpilot.runner.metrics_capabilities import (
    FP8_MECHANISM_REQUIREMENTS,
    MetricsCapabilityReport,
)
from inferpilot.runner.orchestrator import run_experiment

FAKE_SERVER = Path(__file__).resolve().parents[1] / "tests" / "fake_vllm_server.py"


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="instrumentation-dry-run",
        name="zero-gpu-instrumentation-dry-run",
        description=(
            "Synthetic open-loop backlog with aligned queue/KV/preemption evidence; "
            "proves plumbing only, never empirical diagnostic accuracy."
        ),
        engine=EngineConfig(
            model="fake/model",
            revision="deadbeef",
            max_model_len=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.85,
            kv_cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        workload=WorkloadSpec(
            name="synthetic-overload",
            num_requests=100,
            warmup_requests=1,
            prompt_tokens=128,
            output_tokens=2,
            request_rate_qps=4.0,
            temperature=0.0,
            ignore_eos=True,
            seed=4,
            prompt_seed=4,
            arrival_seed=4,
        ),
        tags=["dry-run", "no-gpu", "instrumentation-only"],
    )


def _command(port: int) -> list[str]:
    return [
        sys.executable,
        str(FAKE_SERVER),
        "--port", str(port),
        "--mode", "normal",
        "--output-tokens", "2",
        "--kv-usage", "0.99",
        "--response-delay", "1.5",
        "--response-delay-slope", "0.1",
        "--preemption-increment", "1",
        "--emit-effective", "False",
    ]


def run_dry_run(output_dir: Path) -> dict:
    """Execute the real orchestrator against the fake subprocess and verify artifacts."""

    before = set(output_dir.iterdir()) if output_dir.exists() else set()
    result = run_experiment(
        _config(),
        str(output_dir),
        command_builder=_command,
        ready_timeout_s=15.0,
        request_timeout_s=45.0,
        metric_requirements=FP8_MECHANISM_REQUIREMENTS,
    )
    created = set(output_dir.iterdir()) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new run directory, found {len(created)}")
    run_dir = created.pop()

    stored = ExperimentResult.model_validate_json((run_dir / RESULT_FILENAME).read_text())
    if stored != result:
        raise RuntimeError("stored result does not round-trip to the returned result")
    capability = MetricsCapabilityReport.model_validate_json(
        (run_dir / METRICS_CAPABILITIES_FILENAME).read_text()
    )
    diagnosis = diagnose(stored)

    evidence = stored.load_evidence
    pipeline_valid = all(
        (
            stored.status.value == "completed",
            evidence is not None,
            diagnosis.load_state == "overloaded",
            diagnosis.regime == "kv_pressure",
            diagnosis.recommended_lever == "kv_cache_dtype=fp8",
        )
    )
    # The fake server intentionally lacks exact mechanism counters.  This must
    # remain NO_GO even though the benchmark/diagnosis plumbing is valid.
    expected_missing = {"recomputed_token_executions"}
    fail_closed_valid = (
        not capability.ready and set(capability.missing_required) == expected_missing
    )
    report = {
        "report_version": "0.1.0",
        "pipeline": "PASS" if pipeline_valid else "FAIL",
        "real_pilot_gate": "NO_GO_EXPECTED" if fail_closed_valid else "UNEXPECTED",
        "experiment_id": stored.config.experiment_id,
        "result_status": stored.status.value,
        "load_evidence_present": evidence is not None,
        "load_state": diagnosis.load_state,
        "diagnosis_regime": diagnosis.regime,
        "recommended_lever": diagnosis.recommended_lever,
        "metrics_ready": capability.ready,
        "missing_required_metrics": list(capability.missing_required),
        "run_dir": str(run_dir),
    }
    if not pipeline_valid or not fail_closed_valid:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep artifacts here; default uses and removes a temporary directory.",
    )
    args = parser.parse_args(argv)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = run_dry_run(args.output_dir)
        print(json.dumps(report, indent=2))
        return 0

    with tempfile.TemporaryDirectory(prefix="inferpilot-dry-run-") as directory:
        report = run_dry_run(Path(directory))
        report["run_dir"] = "temporary (removed after successful verification)"
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
