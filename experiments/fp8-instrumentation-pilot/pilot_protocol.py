"""Frozen execution, acceptance, and decision logic for the fp8 pilot.

This module is intentionally free of Modal imports.  The paid launcher calls it,
while unit tests exercise the protocol without allocating a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from inferpilot import (
    EngineConfig,
    ExperimentConfig,
    ExperimentResult,
    MechanismEvidence,
    RunnerPhaseTiming,
    WorkloadSpec,
)
from inferpilot.runner.metrics_capabilities import MetricsCapabilityReport
from inferpilot.saturation import assess_load_state

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
STUDY_ID = "inferpilot-fp8-instrumentation-pilot-v2"
EXPERIMENT_PREFIX = "fp8-instrumentation-pilot-v2"
MAX_DRIFT_P95_MS = 10.0
NUM_REQUESTS = 100
WARMUP_REQUESTS = 4
BLOCKS = (
    (1, 9201, 121, ("bf16_kv", "fp8_kv")),
    (2, 9202, 122, ("fp8_kv", "bf16_kv")),
    (3, 9203, 123, ("bf16_kv", "fp8_kv")),
)
KV_DTYPE = {"bf16_kv": "auto", "fp8_kv": "fp8"}


def execution_order() -> tuple[tuple[int, int, int, str], ...]:
    """Return the exact six-cell order frozen before measurement."""
    return tuple(
        (block, prompt_seed, arrival_seed, arm)
        for block, prompt_seed, arrival_seed, arms in BLOCKS
        for arm in arms
    )


def experiment_id(block: int, arm: str) -> str:
    return f"{EXPERIMENT_PREFIX}-b{block}-{arm.replace('_', '-')}"


def config_for(block: int, prompt_seed: int, arrival_seed: int, arm: str) -> ExperimentConfig:
    """Materialize one registered cell with every scheduler control explicit."""
    if (block, prompt_seed, arrival_seed, arm) not in execution_order():
        raise ValueError("cell is not in the frozen execution order")
    eid = experiment_id(block, arm)
    return ExperimentConfig(
        experiment_id=eid,
        name=eid,
        description="Preregistered fp8 instrumentation pilot cell.",
        engine=EngineConfig(
            model=MODEL,
            revision=MODEL_REVISION,
            dtype="bfloat16",
            max_model_len=9216,
            max_num_seqs=128,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.90,
            kv_cache_dtype=KV_DTYPE[arm],
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            sampler_backend="pytorch",
            generation_config="vllm",
            extra_args={
                "async-scheduling": False,
                "enable-logging-iteration-details": True,
            },
        ),
        workload=WorkloadSpec(
            name="long-context-poisson-pressure",
            num_requests=NUM_REQUESTS,
            warmup_requests=WARMUP_REQUESTS,
            prompt_tokens=7680,
            output_tokens=512,
            request_rate_qps=0.70,
            temperature=0.0,
            ignore_eos=True,
            seed=0,
            prompt_seed=prompt_seed,
            arrival_seed=arrival_seed,
        ),
        tags=["fp8-instrumentation", "performance-pilot", f"block-{block}", arm],
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(quantile * (len(ordered) - 1)))]


def _read_json(path: Path):
    return json.loads(path.read_text())


def validate_cell(result: ExperimentResult, run_dir: Path) -> dict:
    """Apply every preregistered acceptance gate without reading outcome values."""
    problems: list[str] = []
    expected = result.config.workload
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
    if counts != (NUM_REQUESTS, NUM_REQUESTS, 0):
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
        if len(load.boundaries_s) < 2 or load.boundaries_s[-1] - load.boundaries_s[0] < 30:
            problems.append("load_window_too_short")
        try:
            assessment = assess_load_state(result.measurements, evidence=load)
            load_state = assessment.state
            if sum(assessment.arrivals) < NUM_REQUESTS:
                problems.append("load_arrivals_below_100")
        except ValueError:
            problems.append("load_evidence_invalid")

    warmup_path = run_dir / "warmup.json"
    if not warmup_path.is_file():
        problems.append("warmup_missing")
    else:
        warmups = _read_json(warmup_path)
        if len(warmups) != WARMUP_REQUESTS or any(not row.get("success") for row in warmups):
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
            if phases.terminal_status != "completed" or not all(
                by_name[name].completed
                for name in ("server_startup", "measured_window", "finalization", "teardown")
            ):
                problems.append("phases_incomplete")
        except ValueError:
            problems.append("phases_invalid")

    capability_path = run_dir / "metrics-capabilities.json"
    if not capability_path.is_file():
        problems.append("metrics_capabilities_missing")
    else:
        try:
            capabilities = MetricsCapabilityReport.model_validate_json(capability_path.read_text())
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
        if len(scheduled) != NUM_REQUESTS or len(actual) != NUM_REQUESTS:
            problems.append("arrival_counts_invalid")
        else:
            drifts = [(observed - planned) * 1000 for planned, observed in zip(scheduled, actual)]
            p95 = _percentile(drifts, 0.95)
            drift = {
                "mean_ms": sum(drifts) / len(drifts),
                "p95_ms": p95,
                "max_ms": max(drifts),
            }
            if p95 is not None and p95 > MAX_DRIFT_P95_MS:
                problems.append("dispatch_drift_exceeded")
        if (
            arrivals.get("algorithm") != "poisson-v1"
            or arrivals.get("arrival_seed") != expected.effective_arrival_seed
            or arrivals.get("request_rate_qps") != 0.70
        ):
            problems.append("arrival_provenance_mismatch")

    return {
        "problems": sorted(set(problems)),
        "drift": drift,
        "load_state": load_state,
        "mechanism_coverage_complete": bool(mechanism and mechanism.coverage_complete),
    }


def retry_allowed(problems: list[str], attempt: int) -> bool:
    return problems == ["dispatch_drift_exceeded"] and attempt < 2


def _cell_outcome(result: ExperimentResult, evidence: MechanismEvidence) -> dict:
    assert result.aggregates is not None
    actual_tokens = sum(
        row.prompt_tokens + row.output_tokens for row in result.measurements if row.success
    )
    recomputed = evidence.recomputed_token_executions
    burden = None if recomputed is None or actual_tokens == 0 else recomputed / actual_tokens
    assessment = assess_load_state(result.measurements, evidence=result.load_evidence)
    return {
        "experiment_id": result.config.experiment_id,
        "arm": "fp8_kv" if result.config.engine.kv_cache_dtype == "fp8" else "bf16_kv",
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


def evaluate_pilot(cells: list[tuple[int, ExperimentResult, MechanismEvidence]]) -> dict:
    """Evaluate the frozen decision rule only after all six cells are accepted."""
    if len(cells) != 6:
        raise ValueError("the pilot decision requires exactly six accepted cells")
    by_block: dict[int, dict[str, dict]] = {}
    for block, result, evidence in cells:
        outcome = _cell_outcome(result, evidence)
        if outcome["arm"] in by_block.setdefault(block, {}):
            raise ValueError("duplicate arm in block")
        by_block[block][outcome["arm"]] = outcome
    if set(by_block) != {1, 2, 3} or any(set(arms) != set(KV_DTYPE) for arms in by_block.values()):
        raise ValueError("pilot cells do not form three complete paired blocks")

    pairs = []
    for block in (1, 2, 3):
        base, fp8 = by_block[block]["bf16_kv"], by_block[block]["fp8_kv"]
        throughput_ratio = fp8["throughput_tokens_per_s"] / base["throughput_tokens_per_s"]
        base_burden = base["recompute_burden"]
        burden_reduction = (
            None if base_burden in (None, 0) or fp8["recompute_burden"] is None
            else (base_burden - fp8["recompute_burden"]) / base_burden
        )
        latency_ratios = {
            name: fp8[name] / base[name]
            for name in ("ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms")
        }
        pairs.append({
            "block": block,
            "bf16": base,
            "fp8": fp8,
            "throughput_ratio_fp8_over_bf16": throughput_ratio,
            "recompute_burden_reduction_fraction": burden_reduction,
            "latency_ratios_fp8_over_bf16": latency_ratios,
        })

    ratios = [row["throughput_ratio_fp8_over_bf16"] for row in pairs]
    reductions = [row["recompute_burden_reduction_fraction"] for row in pairs]
    geometric_ratio = math.prod(ratios) ** (1 / len(ratios))
    defined_reductions = sorted(value for value in reductions if value is not None)
    median_reduction = (
        defined_reductions[1] if len(defined_reductions) == 3 else None
    )
    target_regime = all(
        row["bf16"]["load_state"] == "overloaded"
        and (row["bf16"]["kv_cache_usage_peak_perc"] or 0) >= 0.95
        for row in pairs
    )
    counter_tracks_preemption = all(
        not outcome["preemptions"] or (outcome["recomputed_token_executions"] or 0) > 0
        for row in pairs
        for outcome in (row["bf16"], row["fp8"])
    )
    no_latency_or_success_regression = all(
        row["fp8"]["successes"] >= row["bf16"]["successes"]
        and all(value <= 1.10 for value in row["latency_ratios_fp8_over_bf16"].values())
        for row in pairs
    )
    go = (
        target_regime
        and all((row["bf16"]["recompute_burden"] or 0) >= 0.10 for row in pairs)
        and all(value >= 1.15 for value in ratios)
        and geometric_ratio >= 1.25
        and median_reduction is not None
        and median_reduction >= 0.50
        and no_latency_or_success_regression
    )
    no_go = (
        any(value < 1.0 for value in ratios)
        or geometric_ratio < 1.15
        or median_reduction is None
        or median_reduction < 0.25
        or not counter_tracks_preemption
    )
    decision = "NOT_TARGET_REGIME" if not target_regime else (
        "GO" if go else ("NO_GO" if no_go else "INCONCLUSIVE")
    )
    payload = {
        "report_version": "0.1.0",
        "study_id": STUDY_ID,
        "decision": decision,
        "paired_blocks": pairs,
        "geometric_mean_throughput_ratio": geometric_ratio,
        "median_recompute_burden_reduction_fraction": median_reduction,
        "target_regime": target_regime,
        "counter_tracks_preemption": counter_tracks_preemption,
        "no_latency_or_success_regression": no_latency_or_success_regression,
        "interpretation": (
            "Pilot decision only; no cross-model, deployment, quality, confidence, or "
            "maximum-capacity claim."
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload
