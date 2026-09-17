"""Generate the preregistered GPU-campaign config matrix (Qwen2.5-7B @ 24 GB).

Deterministic + byte-stable. See docs/experiments/2026-09-17-gpu-campaign-preregistration.md.

Three config families:
  default : pure vLLM defaults (no engine overrides) — the Phase-1 "beats-defaults" baseline.
  dev     : crossover search, varies max_num_seqs over dev arrival seeds.
  held    : held-out confirmation grid (disjoint arrival seeds); only winners are run.

`python generate_configs.py`          writes the configs
`python generate_configs.py --check`  asserts on-disk == regenerated (freeze guard)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferpilot import ExperimentConfig

HERE = Path(__file__).resolve().parent

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
RATES = (2, 6)
DEV_SEEDS = (60, 61, 62)
HELD_SEEDS = (73, 74, 75)
WIDTHS = (1, 2, 4, 8)
PROMPT_SEED = 7007
NUM_REQUESTS = 256
WARMUP = 4

# Engine context for the SEARCH cells. Deliberately IDENTICAL to the default baseline
# (see _config default family below) except for max_num_seqs, so the crossover is a
# clean single-variable test — the 2026-09-18 run confounded max_num_seqs with
# max_num_batched_tokens + chunked_prefill; this corrects that.
FIXED_ENGINE = dict(
    dtype="auto",
    max_model_len=2048,
    gpu_memory_utilization=0.90,
    sampler_backend="pytorch",  # matches all prior evidence; avoids FlashInfer JIT risk
    generation_config="vllm",
)


def _workload(rate: int, arrival_seed: int, tag: str) -> dict:
    return dict(
        name=f"g7-{tag}-qps{rate}",
        num_requests=NUM_REQUESTS,
        warmup_requests=WARMUP,
        prompt_tokens=128,
        output_tokens=32,
        request_rate_qps=float(rate),
        max_concurrency=None,
        temperature=0.0,
        ignore_eos=True,
        seed=0,
        prompt_seed=PROMPT_SEED,
        arrival_seed=arrival_seed,
        arrival_pattern="poisson-v1",
        burst_size=None,
    )


def _config(eid: str, engine: dict, workload: dict, tags: list[str], desc: str) -> dict:
    payload = dict(
        schema_version="0.5.0",
        experiment_id=eid,
        name=eid,
        description=desc,
        engine=dict(model=MODEL, revision=REVISION, **engine),
        workload=workload,
        slo=None,
        seed=0,
        tags=tags,
    )
    return ExperimentConfig.model_validate(payload).model_dump(mode="json")


def default_id(rate: int, seed: int) -> str:
    return f"g7-default-qps{rate}-a{seed}"


def dev_id(rate: int, seed: int, width: int) -> str:
    return f"g7-dev-qps{rate}-a{seed}-seq{width}"


def held_id(rate: int, seed: int, width: int) -> str:
    return f"g7-held-qps{rate}-a{seed}-seq{width}"


def expected() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    # Phase 1 — pure vLLM defaults (only rate/seed vary; no tunable overrides).
    for rate in RATES:
        for seed in DEV_SEEDS:
            eid = default_id(rate, seed)
            cfg = _config(
                eid,
                dict(dtype="auto", max_model_len=2048, gpu_memory_utilization=0.90,
                     sampler_backend="pytorch", generation_config="vllm"),
                _workload(rate, seed, "default"),
                ["gpu-campaign", "qwen2.5-7b", "default-baseline", "poisson-v1"],
                "Phase 1 vLLM-default baseline; no engine tunables set.",
            )
            files[f"{eid}.json"] = (json.dumps(cfg, indent=2) + "\n").encode()
    # Phase 2 — crossover search over dev seeds.
    for rate in RATES:
        for seed in DEV_SEEDS:
            for width in WIDTHS:
                eid = dev_id(rate, seed, width)
                cfg = _config(
                    eid, dict(max_num_seqs=width, **FIXED_ENGINE),
                    _workload(rate, seed, "dev"),
                    ["gpu-campaign", "qwen2.5-7b", "crossover", "development", "poisson-v1"],
                    "Phase 2 crossover-search development cell; not held-out evidence.",
                )
                files[f"{eid}.json"] = (json.dumps(cfg, indent=2) + "\n").encode()
    # Phase 3 — held-out grid (disjoint seeds); only winners are run.
    for rate in RATES:
        for seed in HELD_SEEDS:
            for width in WIDTHS:
                eid = held_id(rate, seed, width)
                cfg = _config(
                    eid, dict(max_num_seqs=width, **FIXED_ENGINE),
                    _workload(rate, seed, "held"),
                    ["gpu-campaign", "qwen2.5-7b", "crossover", "held-out", "poisson-v1"],
                    "Phase 3 held-out confirmation cell.",
                )
                files[f"{eid}.json"] = (json.dumps(cfg, indent=2) + "\n").encode()
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
