"""Decode-heavy GPU campaign (Qwen2.5-7B @ A10 24 GB) config matrix.

Hypothesis: at a decode-heavy operating point (long outputs) under load, the vLLM
default over-admits requests, causing KV-cache pressure/preemption; a moderate
max_num_seqs cap can beat the default. This is the regime the 128/32 run did not
stress (there defaults trivially won). Same harness, corrected single-variable design
(search cells differ from the default in ONLY max_num_seqs) and the fail-closed Pareto
evaluator. See 2026-09-18-gpu-campaign-7b-decode-preregistration.md.

`python generate_configs.py [--check]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferpilot import ExperimentConfig

HERE = Path(__file__).resolve().parent

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
RATES = (2,)                     # feasible near-capacity rate: the rate-4 probe showed the
                                 # A10 saturates at ~2.2 req/s on this decode workload
                                 # (compute-bound), so rate 4 was beyond capacity for all configs.
DEV_SEEDS = (60, 61, 62)
HELD_SEEDS = (73, 74, 75)
WIDTHS = (16, 32, 64, 128)       # caps BELOW the vLLM default (~256), where preemption bites
PROMPT_SEED = 7007
NUM_REQUESTS = 96                # decode-heavy runs are slow; bound duration/cost
WARMUP = 4
PROMPT_TOKENS = 128
OUTPUT_TOKENS = 512              # decode-heavy: ~16x the KV growth of the 32-token run

# Engine context shared by default and search cells; search adds ONLY max_num_seqs.
BASE_ENGINE = dict(
    dtype="auto",
    max_model_len=1024,          # 128 + 512 + margin
    gpu_memory_utilization=0.90,
    sampler_backend="pytorch",
    generation_config="vllm",
)


def _workload(rate: int, arrival_seed: int, tag: str) -> dict:
    return dict(
        name=f"g7d-{tag}-qps{rate}",
        num_requests=NUM_REQUESTS, warmup_requests=WARMUP,
        prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
        request_rate_qps=float(rate), max_concurrency=None, temperature=0.0,
        ignore_eos=True, seed=0, prompt_seed=PROMPT_SEED, arrival_seed=arrival_seed,
        arrival_pattern="poisson-v1", burst_size=None,
    )


def _config(eid: str, engine: dict, workload: dict, tags: list[str], desc: str) -> dict:
    payload = dict(
        schema_version="0.5.0", experiment_id=eid, name=eid, description=desc,
        engine=dict(model=MODEL, revision=REVISION, **engine),
        workload=workload, slo=None, seed=0, tags=tags,
    )
    return ExperimentConfig.model_validate(payload).model_dump(mode="json")


def default_id(rate: int, seed: int) -> str:
    return f"g7d-default-qps{rate}-a{seed}"


def dev_id(rate: int, seed: int, width: int) -> str:
    return f"g7d-dev-qps{rate}-a{seed}-seq{width}"


def held_id(rate: int, seed: int, width: int) -> str:
    return f"g7d-held-qps{rate}-a{seed}-seq{width}"


def expected() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for rate in RATES:
        for seed in DEV_SEEDS:
            eid = default_id(rate, seed)
            files[f"{eid}.json"] = (json.dumps(_config(
                eid, dict(BASE_ENGINE), _workload(rate, seed, "default"),
                ["gpu-campaign", "qwen2.5-7b", "decode-heavy", "default-baseline"],
                "Decode-heavy Phase-1 vLLM-default baseline.",
            ), indent=2) + "\n").encode()
    for rate in RATES:
        for seed in DEV_SEEDS:
            for width in WIDTHS:
                eid = dev_id(rate, seed, width)
                files[f"{eid}.json"] = (json.dumps(_config(
                    eid, dict(max_num_seqs=width, **BASE_ENGINE), _workload(rate, seed, "dev"),
                    ["gpu-campaign", "qwen2.5-7b", "decode-heavy", "development"],
                    "Decode-heavy Phase-2 crossover development cell.",
                ), indent=2) + "\n").encode()
    for rate in RATES:
        for seed in HELD_SEEDS:
            for width in WIDTHS:
                eid = held_id(rate, seed, width)
                files[f"{eid}.json"] = (json.dumps(_config(
                    eid, dict(max_num_seqs=width, **BASE_ENGINE), _workload(rate, seed, "held"),
                    ["gpu-campaign", "qwen2.5-7b", "decode-heavy", "held-out"],
                    "Decode-heavy Phase-3 held-out confirmation cell.",
                ), indent=2) + "\n").encode()
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    want = expected()
    if args.check:
        have = {p.name: p.read_bytes() for p in HERE.glob("*.json")}
        if have != want:
            missing = sorted(set(want) - set(have))
            extra = sorted(set(have) - set(want))
            changed = sorted(n for n in set(have) & set(want) if have[n] != want[n])
            raise SystemExit(f"config drift: missing={missing} extra={extra} changed={changed}")
        return 0
    for name, payload in want.items():
        (HERE / name).write_bytes(payload)
    print(f"wrote {len(want)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
