"""Frozen execution, acceptance, and decision logic for the fp8 pilot.

This module is intentionally free of Modal imports.  The paid launcher calls it,
while unit tests exercise the protocol without allocating a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from inferpilot import EngineConfig, ExperimentConfig, ExperimentResult, MechanismEvidence, WorkloadSpec
from inferpilot.runner.mechanism_study import (
    mechanism_cell_outcome,
    validate_mechanism_cell,
)

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


def validate_cell(result: ExperimentResult, run_dir: Path) -> dict:
    """Apply every preregistered acceptance gate without reading outcome values."""
    return validate_mechanism_cell(result, run_dir)

def retry_allowed(problems: list[str], attempt: int) -> bool:
    return problems == ["dispatch_drift_exceeded"] and attempt < 2


def _cell_outcome(result: ExperimentResult, evidence: MechanismEvidence) -> dict:
    return mechanism_cell_outcome(result, evidence)


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
