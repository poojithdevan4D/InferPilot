"""Reusable fail-closed validation for instrumented mechanism-study cells."""

from __future__ import annotations

import json
from pathlib import Path

from ..mechanism import MechanismEvidence
from ..phases import RunnerPhaseTiming
from ..results import ExperimentResult
from ..saturation import assess_load_state
from .metrics_capabilities import MetricsCapabilityReport


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(quantile * (len(ordered) - 1)))]


def _read_json(path: Path):
    return json.loads(path.read_text())


def validate_mechanism_cell(
    result: ExperimentResult,
    run_dir: Path,
    *,
    max_dispatch_drift_p95_ms: float = 10.0,
    minimum_window_seconds: float = 30.0,
) -> dict:
    """Validate one accepted-study candidate without inspecting outcomes.

    Request/rate/seed expectations come from the embedded config.  The function
    checks completeness and provenance only; it deliberately does not gate on
    throughput, latency, load-state label, recomputation magnitude, or arm.
    """
    problems: list[str] = []
    workload = result.config.workload
    expected_n = workload.num_requests
    expected_warmups = workload.warmup_requests
    if result.status.value != "completed":
        problems.append(f"status={result.status.value}")
    if not result.is_baseline_eligible:
        problems.append("not_baseline_eligible")
    aggregate = result.aggregates
    counts = None if aggregate is None else (
        aggregate.num_requests,
        aggregate.num_successful,
        aggregate.num_failed,
    )
    if counts != (expected_n, expected_n, 0):
        problems.append(f"counts={counts}")
    effective = result.effective_config
    if effective is None or not effective.verified or effective.unverified_fields:
        problems.append("effective_config_unverified")
    telemetry = result.telemetry
    if telemetry is None or telemetry.error is not None or telemetry.num_samples < 1:
        problems.append("telemetry_incomplete")

    load = result.load_evidence
    load_state = None
    if load is None:
        problems.append("load_evidence_missing")
    else:
        if not load.coverage_complete:
            problems.append("load_coverage_incomplete")
        if not load.steady_state:
            problems.append("load_not_steady_state")
        if (
            len(load.boundaries_s) < 2
            or load.boundaries_s[-1] - load.boundaries_s[0] < minimum_window_seconds
        ):
            problems.append("load_window_too_short")
        try:
            assessment = assess_load_state(result.measurements, evidence=load)
            load_state = assessment.state
            if sum(assessment.arrivals) < expected_n:
                problems.append(f"load_arrivals_below_{expected_n}")
        except ValueError:
            problems.append("load_evidence_invalid")

    warmup_path = run_dir / "warmup.json"
    if not warmup_path.is_file():
        problems.append("warmup_missing")
    else:
        warmups = _read_json(warmup_path)
        if len(warmups) != expected_warmups or any(
            not row.get("success") for row in warmups
        ):
            problems.append("warmup_invalid")

    lifecycle_path = run_dir / "lifecycle.json"
    if not lifecycle_path.is_file():
        problems.append("lifecycle_missing")
    elif _read_json(lifecycle_path).get("pre_teardown"):
        problems.append("lifecycle_pre_teardown_errors")

    phases_path = run_dir / "phases.json"
    if not phases_path.is_file():
        problems.append("phases_missing")
    else:
        try:
            phases = RunnerPhaseTiming.model_validate_json(phases_path.read_text())
            by_name = {phase.name: phase for phase in phases.phases}
            required = ("server_startup", "measured_window", "finalization", "teardown")
            if phases.terminal_status != "completed" or not all(
                by_name[name].completed for name in required
            ):
                problems.append("phases_incomplete")
        except (KeyError, ValueError):
            problems.append("phases_invalid")

    capability_path = run_dir / "metrics-capabilities.json"
    if not capability_path.is_file():
        problems.append("metrics_capabilities_missing")
    else:
        try:
            capabilities = MetricsCapabilityReport.model_validate_json(
                capability_path.read_text()
            )
            if not capabilities.ready or capabilities.missing_required:
                problems.append("metrics_capabilities_not_ready")
        except ValueError:
            problems.append("metrics_capabilities_invalid")

    mechanism_path = run_dir / "mechanism-evidence.json"
    mechanism = None
    if not mechanism_path.is_file():
        problems.append("mechanism_evidence_missing")
    else:
        try:
            mechanism = MechanismEvidence.model_validate_json(mechanism_path.read_text())
            if not mechanism.coverage_complete:
                problems.append("mechanism_coverage_incomplete")
            if mechanism.experiment_id != result.config.experiment_id:
                problems.append("mechanism_experiment_mismatch")
        except ValueError:
            problems.append("mechanism_evidence_invalid")

    drift: dict = {}
    arrivals_path = run_dir / "arrivals.json"
    if not arrivals_path.is_file():
        problems.append("arrivals_missing")
    else:
        arrivals = _read_json(arrivals_path)
        scheduled = arrivals.get("scheduled_offsets_s", [])
        actual = arrivals.get("actual_dispatch_offsets_s", [])
        if len(scheduled) != expected_n or len(actual) != expected_n:
            problems.append("arrival_counts_invalid")
        else:
            drifts = [
                (observed - planned) * 1000
                for planned, observed in zip(scheduled, actual)
            ]
            p95 = _percentile(drifts, 0.95)
            drift = {
                "mean_ms": sum(drifts) / len(drifts),
                "p95_ms": p95,
                "max_ms": max(drifts),
            }
            if p95 is not None and p95 > max_dispatch_drift_p95_ms:
                problems.append("dispatch_drift_exceeded")
        if (
            arrivals.get("algorithm") != workload.arrival_pattern
            or arrivals.get("arrival_seed") != workload.effective_arrival_seed
            or arrivals.get("request_rate_qps") != workload.request_rate_qps
            or arrivals.get("burst_size") != workload.burst_size
        ):
            problems.append("arrival_provenance_mismatch")

    return {
        "problems": sorted(set(problems)),
        "drift": drift,
        "load_state": load_state,
        "mechanism_coverage_complete": bool(
            mechanism and mechanism.coverage_complete
        ),
    }


def mechanism_cell_outcome(
    result: ExperimentResult, evidence: MechanismEvidence
) -> dict:
    """Derive registered outcome values after a cell has passed acceptance."""
    if result.aggregates is None or result.load_evidence is None:
        raise ValueError("accepted mechanism outcome requires aggregates and load evidence")
    actual_tokens = sum(
        row.prompt_tokens + row.output_tokens
        for row in result.measurements
        if row.success
    )
    recomputed = evidence.recomputed_token_executions
    burden = (
        None
        if recomputed is None or actual_tokens == 0
        else recomputed / actual_tokens
    )
    assessment = assess_load_state(result.measurements, evidence=result.load_evidence)
    return {
        "experiment_id": result.config.experiment_id,
        "arm": "fp8_kv"
        if result.config.engine.kv_cache_dtype == "fp8"
        else "bf16_kv",
        "throughput_tokens_per_s": result.aggregates.throughput_tokens_per_s,
        "ttft_p95_ms": result.aggregates.ttft_p95_ms,
        "tpot_p95_ms": result.aggregates.tpot_p95_ms,
        "e2e_p95_ms": result.aggregates.e2e_p95_ms,
        "successes": result.aggregates.num_successful,
        "load_state": assessment.state,
        "kv_cache_usage_peak_perc": result.load_evidence.kv_cache_usage_peak_perc,
        "preemptions": evidence.preemptions,
        "recomputed_token_executions": recomputed,
        "actual_prompt_plus_output_tokens": actual_tokens,
        "recompute_burden": burden,
        "measured_window_seconds": result.aggregates.duration_s,
    }
