"""KV-bound GPU campaign (Qwen2.5-14B @ A100-40GB) config matrix.

Hypothesis: with a big model on tight VRAM and long context, KV cache is scarce. The
vLLM default (max_num_seqs ~256) over-admits -> KV exhaustion -> preemption/recompute
thrash -> TTFT grows (saturates). A max_num_seqs cap that fits the KV budget runs clean
and keeps up, dominating the default. This is the regime where the 7B/A10 runs (defaults
robust) did not reach. Single-variable design + saturation-aware Pareto evaluator.

14B fp16 ~28 GB on A100-40GB -> ~8 GB KV -> ~16 concurrent 2560-token requests. The rate
is calibrated empirically from the Phase-1 baseline (does the default saturate?).

`python generate_configs.py [--check]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferpilot import ExperimentConfig

HERE = Path(__file__).resolve().parent

MODEL = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
RATES = (2,)                     # calibrated from Phase-1 baseline (KV pressure check)
DEV_SEEDS = (60, 61, 62)
HELD_SEEDS = (73, 74, 75)
WIDTHS = (4, 8, 16, 32)          # caps around the ~16 concurrent KV budget; default is ~256
PROMPT_SEED = 7007
NUM_REQUESTS = 40                # long cells are slow + KV-heavy; bound duration/cost
WARMUP = 3
PROMPT_TOKENS = 2048             # long context -> large KV per request
OUTPUT_TOKENS = 512

BASE_ENGINE = dict(
    dtype="auto",
    max_model_len=3072,          # 2048 + 512 + margin
    gpu_memory_utilization=0.90,
    sampler_backend="pytorch",
    generation_config="vllm",
)


def _workload(rate: int, arrival_seed: int, tag: str) -> dict:
    return dict(
        name=f"g7k-{tag}-qps{rate}",
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
    return f"g7k-default-qps{rate}-a{seed}"


def dev_id(rate: int, seed: int, width: int) -> str:
    return f"g7k-dev-qps{rate}-a{seed}-seq{width}"


def held_id(rate: int, seed: int, width: int) -> str:
    return f"g7k-held-qps{rate}-a{seed}-seq{width}"


def expected() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for rate in RATES:
        for seed in (*DEV_SEEDS, *HELD_SEEDS):
            eid = default_id(rate, seed)
            files[f"{eid}.json"] = (json.dumps(_config(
                eid, dict(BASE_ENGINE), _workload(rate, seed, "default"),
                ["gpu-campaign", "qwen2.5-14b", "kv-bound", "default-baseline"],
                "KV-bound vLLM-default baseline.",
            ), indent=2) + "\n").encode()
    for rate in RATES:
        for seed in DEV_SEEDS:
            for width in WIDTHS:
                eid = dev_id(rate, seed, width)
                files[f"{eid}.json"] = (json.dumps(_config(
                    eid, dict(max_num_seqs=width, **BASE_ENGINE), _workload(rate, seed, "dev"),
                    ["gpu-campaign", "qwen2.5-14b", "kv-bound", "development"],
                    "KV-bound Phase-2 crossover development cell.",
                ), indent=2) + "\n").encode()
    for rate in RATES:
        for seed in HELD_SEEDS:
            for width in WIDTHS:
                eid = held_id(rate, seed, width)
                files[f"{eid}.json"] = (json.dumps(_config(
                    eid, dict(max_num_seqs=width, **BASE_ENGINE), _workload(rate, seed, "held"),
                    ["gpu-campaign", "qwen2.5-14b", "kv-bound", "held-out"],
                    "KV-bound Phase-3 held-out confirmation cell.",
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
