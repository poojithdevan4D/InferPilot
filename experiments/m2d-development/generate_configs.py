"""Generate and verify the preregistered M2D development-pilot matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferpilot import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "experiments/c5-openloop/seq1.json"
ARRIVAL_SEEDS = (10, 11, 12)
CANDIDATES = ((1, 2048), (1, 4096), (4, 2048), (4, 4096))
SHAPES = {
    "prefill": {"prompt_tokens": 1024, "output_tokens": 16, "request_rate_qps": 4.0,
                "arrival_pattern": "poisson-v1", "burst_size": None, "prompt_seed": 1001},
    "decode": {"prompt_tokens": 64, "output_tokens": 256, "request_rate_qps": 1.0,
               "arrival_pattern": "poisson-v1", "burst_size": None, "prompt_seed": 1002},
    "burst": {"prompt_tokens": 128, "output_tokens": 32, "request_rate_qps": 6.0,
              "arrival_pattern": "batched-poisson-v1", "burst_size": 4, "prompt_seed": 1003},
}


def experiment_id(shape: str, arrival_seed: int, seqs: int, tokens: int) -> str:
    return f"m2d-dev-{shape}-a{arrival_seed}-seq{seqs}-tok{tokens}"


def expected_payload(shape: str, arrival_seed: int, seqs: int, tokens: int) -> dict:
    payload = json.loads(BASE_CONFIG.read_text())
    eid = experiment_id(shape, arrival_seed, seqs, tokens)
    settings = SHAPES[shape]
    payload.update(
        schema_version="0.5.0", experiment_id=eid, name=eid,
        description=(
            "M2D preregistered development pilot; characterization only. "
            f"shape={shape}, arrival_seed={arrival_seed}, max_num_seqs={seqs}, "
            f"max_num_batched_tokens={tokens}. No SLO or winner declared."
        ),
        slo=None, seed=0,
        tags=["m2d", "development", "pilot", shape, settings["arrival_pattern"]],
    )
    payload["engine"]["max_num_seqs"] = seqs
    payload["engine"]["max_num_batched_tokens"] = tokens
    payload["workload"].update(
        name=f"m2d-dev-{shape}", num_requests=128, warmup_requests=4,
        prompt_tokens=settings["prompt_tokens"], output_tokens=settings["output_tokens"],
        request_rate_qps=settings["request_rate_qps"], max_concurrency=None,
        temperature=0.0, ignore_eos=True, seed=0,
        prompt_seed=settings["prompt_seed"], arrival_seed=arrival_seed,
        arrival_pattern=settings["arrival_pattern"], burst_size=settings["burst_size"],
    )
    return ExperimentConfig.model_validate(payload).model_dump(mode="json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mismatches: list[str] = []
    for shape in SHAPES:
        for arrival_seed in ARRIVAL_SEEDS:
            for seqs, tokens in CANDIDATES:
                path = HERE / f"{experiment_id(shape, arrival_seed, seqs, tokens)}.json"
                expected = expected_payload(shape, arrival_seed, seqs, tokens)
                if args.check:
                    if not path.is_file() or json.loads(path.read_text()) != expected:
                        mismatches.append(path.name)
                else:
                    path.write_text(json.dumps(expected, indent=2) + "\n")
    if mismatches:
        raise SystemExit(f"M2D development config drift: {', '.join(mismatches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
