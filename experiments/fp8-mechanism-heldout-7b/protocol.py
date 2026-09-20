"""Frozen configuration and decision logic for the focused 7B mechanism study."""

from __future__ import annotations

import hashlib
import json
import math

from inferpilot import EngineConfig, ExperimentConfig, ExperimentResult, MechanismEvidence, WorkloadSpec
from inferpilot.runner.mechanism_study import mechanism_cell_outcome

STUDY_ID = "inferpilot-fp8-mechanism-7b-heldout-v1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ARMS = {"bf16_kv": "auto", "fp8_kv": "fp8"}
BLOCKS = {
    1: (9401, 141),
    2: (9402, 142),
    3: (9403, 143),
}
WORKLOADS = {
    "long_context_positive": {
        "max_model_len": 5120,
        "prompt_tokens": 4096,
        "output_tokens": 512,
    },
    "short_context_negative": {
        "max_model_len": 1024,
        "prompt_tokens": 512,
        "output_tokens": 128,
    },
}
_ORDER = (
    (1, "long_context_positive", "bf16_kv"),
    (1, "short_context_negative", "fp8_kv"),
    (1, "long_context_positive", "fp8_kv"),
    (1, "short_context_negative", "bf16_kv"),
    (2, "short_context_negative", "bf16_kv"),
    (2, "long_context_positive", "fp8_kv"),
    (2, "short_context_negative", "fp8_kv"),
    (2, "long_context_positive", "bf16_kv"),
    (3, "long_context_positive", "bf16_kv"),
    (3, "short_context_negative", "fp8_kv"),
    (3, "long_context_positive", "fp8_kv"),
    (3, "short_context_negative", "bf16_kv"),
)


def execution_order() -> tuple[tuple[int, int, int, str, str], ...]:
    return tuple(
        (block, *BLOCKS[block], workload, arm)
        for block, workload, arm in _ORDER
    )


def experiment_id(block: int, workload: str, arm: str) -> str:
    shape = "long" if workload == "long_context_positive" else "short"
    return f"fp8-mechanism-7b-heldout-b{block}-{shape}-{arm.replace('_', '-')}"


def config_for(
    block: int,
    prompt_seed: int,
    arrival_seed: int,
    workload: str,
    arm: str,
) -> ExperimentConfig:
    cell = (block, prompt_seed, arrival_seed, workload, arm)
    if cell not in execution_order():
        raise ValueError("cell is not in the frozen execution order")
    shape = WORKLOADS[workload]
    eid = experiment_id(block, workload, arm)
    return ExperimentConfig(
        experiment_id=eid,
        name=eid,
        description="Preregistered held-out 7B fp8 recomputation mechanism cell.",
        engine=EngineConfig(
            model=MODEL,
            revision=MODEL_REVISION,
            dtype="bfloat16",
            max_model_len=shape["max_model_len"],
            max_num_seqs=128,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.90,
            kv_cache_dtype=ARMS[arm],
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
            name=workload,
            num_requests=100,
            warmup_requests=4,
            prompt_tokens=shape["prompt_tokens"],
            output_tokens=shape["output_tokens"],
            request_rate_qps=4.0,
            temperature=0.0,
            ignore_eos=True,
            seed=0,
            prompt_seed=prompt_seed,
            arrival_seed=arrival_seed,
        ),
        tags=["fp8-mechanism", "heldout", "qwen2.5-7b", workload, f"block-{block}", arm],
    )


def retry_allowed(problems: list[str], attempt: int) -> bool:
    return problems == ["dispatch_drift_exceeded"] and attempt < 2


def _paired_outcomes(
    cells: list[tuple[int, str, ExperimentResult, MechanismEvidence]],
) -> list[dict]:
    if len(cells) != 12:
        raise ValueError("the held-out decision requires exactly twelve accepted cells")
    grouped: dict[tuple[int, str], dict[str, dict]] = {}
    for block, workload, result, evidence in cells:
        outcome = mechanism_cell_outcome(result, evidence)
        key = (block, workload)
        if outcome["arm"] in grouped.setdefault(key, {}):
            raise ValueError("duplicate arm in paired block")
        grouped[key][outcome["arm"]] = outcome
    expected = {(block, workload) for block in BLOCKS for workload in WORKLOADS}
    if set(grouped) != expected or any(set(arms) != set(ARMS) for arms in grouped.values()):
        raise ValueError("cells do not form the frozen rectangular paired design")

    pairs = []
    for block in BLOCKS:
        for workload in WORKLOADS:
            base = grouped[(block, workload)]["bf16_kv"]
            fp8 = grouped[(block, workload)]["fp8_kv"]
            ratio = fp8["throughput_tokens_per_s"] / base["throughput_tokens_per_s"]
            base_burden = base["recompute_burden"]
            reduction = (
                None
                if base_burden in (None, 0) or fp8["recompute_burden"] is None
                else (base_burden - fp8["recompute_burden"]) / base_burden
            )
            latency_ratios = {
                metric: fp8[metric] / base[metric]
                for metric in ("ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms")
            }
            pairs.append({
                "block": block,
                "workload": workload,
                "bf16": base,
                "fp8": fp8,
                "throughput_ratio_fp8_over_bf16": ratio,
                "recompute_burden_reduction_fraction": reduction,
                "latency_ratios_fp8_over_bf16": latency_ratios,
            })
    return pairs


def _geometric_mean(values: list[float]) -> float:
    return math.prod(values) ** (1 / len(values))


def _median_three(values: list[float | None]) -> float | None:
    defined = sorted(value for value in values if value is not None)
    return defined[1] if len(defined) == 3 else None


def evaluate_study(
    cells: list[tuple[int, str, ExperimentResult, MechanismEvidence]],
) -> dict:
    pairs = _paired_outcomes(cells)
    positive = [row for row in pairs if row["workload"] == "long_context_positive"]
    negative = [row for row in pairs if row["workload"] == "short_context_negative"]
    positive_ratios = [row["throughput_ratio_fp8_over_bf16"] for row in positive]
    negative_ratios = [row["throughput_ratio_fp8_over_bf16"] for row in negative]
    positive_gmean = _geometric_mean(positive_ratios)
    negative_gmean = _geometric_mean(negative_ratios)
    positive_reduction = _median_three(
        [row["recompute_burden_reduction_fraction"] for row in positive]
    )

    positive_target = all(
        row["bf16"]["load_state"] == "overloaded"
        and (row["bf16"]["preemptions"] or 0) > 0
        and (row["bf16"]["recompute_burden"] or 0) >= 0.01
        for row in positive
    )
    negative_target = all(
        (row["bf16"]["preemptions"] or 0) == 0
        and row["bf16"]["recompute_burden"] is not None
        and row["bf16"]["recompute_burden"] <= 0.01
        for row in negative
    )
    no_positive_regression = all(
        row["fp8"]["successes"] >= row["bf16"]["successes"]
        and all(value <= 1.10 for value in row["latency_ratios_fp8_over_bf16"].values())
        for row in positive
    )
    no_negative_regression = all(
        row["fp8"]["successes"] >= row["bf16"]["successes"]
        and all(value <= 1.10 for value in row["latency_ratios_fp8_over_bf16"].values())
        for row in negative
    )
    positive_effect = (
        positive_gmean >= 1.20
        and all(value >= 1.10 for value in positive_ratios)
        and positive_reduction is not None
        and positive_reduction >= 0.50
        and no_positive_regression
    )
    negative_effect = (
        0.90 <= negative_gmean <= 1.10
        and all(0.85 <= value <= 1.15 for value in negative_ratios)
        and no_negative_regression
    )
    positive_burdens = [row["bf16"]["recompute_burden"] for row in positive]
    negative_burdens = [row["bf16"]["recompute_burden"] for row in negative]
    separation = (
        all(value is not None for value in positive_burdens + negative_burdens)
        and min(positive_burdens) > max(negative_burdens)
        and min(positive_ratios) > max(negative_ratios)
    )
    counter_tracks_preemption = all(
        not outcome["preemptions"] or (outcome["recomputed_token_executions"] or 0) > 0
        for row in pairs
        for outcome in (row["bf16"], row["fp8"])
    )
    if not positive_target or not negative_target:
        decision = "NOT_TARGET_REGIME"
    elif all((positive_effect, negative_effect, separation, counter_tracks_preemption)):
        decision = "SUPPORTS_MECHANISM_WITHIN_ENVELOPE"
    else:
        decision = "INCONCLUSIVE"

    payload = {
        "report_version": "0.1.0",
        "study_id": STUDY_ID,
        "decision": decision,
        "paired_blocks": pairs,
        "positive_geometric_mean_throughput_ratio": positive_gmean,
        "negative_geometric_mean_throughput_ratio": negative_gmean,
        "positive_median_recompute_burden_reduction_fraction": positive_reduction,
        "positive_target": positive_target,
        "negative_target": negative_target,
        "positive_effect": positive_effect,
        "negative_effect": negative_effect,
        "separation": separation,
        "counter_tracks_preemption": counter_tracks_preemption,
        "interpretation": (
            "Held-out support is limited to this model, GPU, engine, and two synthetic fixed-length "
            "workloads; it is not a causal, quality, deployment, or cross-model certificate."
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload
