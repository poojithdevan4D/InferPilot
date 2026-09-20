"""Frozen repair protocol for the invalid 7B mechanism study.

The first held-out attempt was invalidated before comparison because one
short-context schedule ended before the registered 30-second evidence window.
This protocol changes only design mechanics: new seeds and 128 short requests
whose frozen schedules all exceed 30 seconds.  The original hypotheses and
decision thresholds are imported unchanged.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

from inferpilot import EngineConfig, ExperimentConfig, WorkloadSpec
from inferpilot.runner.schedule import generate_poisson_offsets

_base_path = Path(__file__).resolve().parents[1] / "fp8-mechanism-heldout-7b" / "protocol.py"
_spec = importlib.util.spec_from_file_location("fp8_mechanism_7b_frozen_base", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load frozen base protocol")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

MODEL = _base.MODEL
MODEL_REVISION = _base.MODEL_REVISION
ARMS = _base.ARMS
WORKLOADS = _base.WORKLOADS

STUDY_ID = "inferpilot-fp8-mechanism-7b-repair-v1"
BLOCKS = {1: (9501, 151), 2: (9502, 152), 3: (9503, 153)}
_ORDER = _base._ORDER


def execution_order() -> tuple[tuple[int, int, int, str, str], ...]:
    return tuple((block, *BLOCKS[block], workload, arm) for block, workload, arm in _ORDER)


def experiment_id(block: int, workload: str, arm: str) -> str:
    shape = "long" if workload == "long_context_positive" else "short"
    return f"fp8-mechanism-7b-repair-b{block}-{shape}-{arm.replace('_', '-')}"


def config_for(
    block: int,
    prompt_seed: int,
    arrival_seed: int,
    workload: str,
    arm: str,
) -> ExperimentConfig:
    if (block, prompt_seed, arrival_seed, workload, arm) not in execution_order():
        raise ValueError("cell is not in the frozen repair execution order")
    shape = WORKLOADS[workload]
    requests = 128 if workload == "short_context_negative" else 100
    # This is a preregistration invariant, not an observed outcome check.
    if (
        workload == "short_context_negative"
        and generate_poisson_offsets(requests, 4.0, arrival_seed)[-1] < 30.0
    ):
        raise ValueError("frozen schedule cannot cover the required evidence window")
    eid = experiment_id(block, workload, arm)
    return ExperimentConfig(
        experiment_id=eid,
        name=eid,
        description="Preregistered repair of the invalid 7B fp8 mechanism study.",
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
            num_requests=requests,
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
        tags=["fp8-mechanism", "repair", "qwen2.5-7b", workload, f"block-{block}", arm],
    )


def retry_allowed(problems: list[str], attempt: int) -> bool:
    return problems == ["dispatch_drift_exceeded"] and attempt < 2


def evaluate_study(cells):
    """Reuse the frozen decision rule while binding the repair study identity."""
    report = _base.evaluate_study(cells)
    report.pop("content_sha256")
    report["study_id"] = STUDY_ID
    canonical = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    report["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report
