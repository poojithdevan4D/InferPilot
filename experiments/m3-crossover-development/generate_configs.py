"""Generate and check the preregistered M3 crossover-development matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferpilot import ExperimentConfig
from inferpilot.comparison.models import BlockedStudySpec


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "experiments/c7-rate-regimes/qps4-s6-seq1.json"
RATES = (2, 6)
ARRIVAL_SEEDS = (60, 61, 62)
WIDTHS = (1, 2, 3, 4)
PROMPT_SEED = 3001


def experiment_id(rate: int, arrival_seed: int, width: int) -> str:
    return f"m3-dev-qps{rate}-a{arrival_seed}-seq{width}"


def config_payload(rate: int, arrival_seed: int, width: int) -> dict:
    payload = json.loads(BASE.read_text())
    eid = experiment_id(rate, arrival_seed, width)
    payload.update(
        schema_version="0.5.0",
        experiment_id=eid,
        name=eid,
        description=(
            "Preregistered M3 development cell for locating a workload-rate scheduler-width "
            "crossover; not held-out evidence."
        ),
        tags=["m3", "development", "crossover", "poisson-v1"],
    )
    payload["engine"].update(
        max_num_seqs=width,
        max_num_batched_tokens=512,
        enable_chunked_prefill=True,
    )
    payload["workload"].update(
        name=f"m3-crossover-qps{rate}",
        request_rate_qps=float(rate),
        seed=0,
        prompt_seed=PROMPT_SEED,
        arrival_seed=arrival_seed,
        arrival_pattern="poisson-v1",
        burst_size=None,
    )
    return ExperimentConfig.model_validate(payload).model_dump(mode="json")


def study(rate: int) -> BlockedStudySpec:
    return BlockedStudySpec(
        study_id=f"m3-dev-qps{rate}",
        objective="tpot_p95_ms",
        slo={"ttft_p95_ms": 250, "tpot_p95_ms": 8.5},
        varied_engine_fields=["max_num_seqs"],
        blocks=[
            {
                "seed": seed,
                "experiment_ids": [experiment_id(rate, seed, width) for width in WIDTHS],
            }
            for seed in ARRIVAL_SEEDS
        ],
        min_runs=1,
    )


def expected_files() -> dict[str, bytes]:
    files = {
        f"{experiment_id(rate, seed, width)}.json": (
            json.dumps(config_payload(rate, seed, width), indent=2) + "\n"
        ).encode()
        for seed in ARRIVAL_SEEDS
        for width in WIDTHS
        for rate in RATES
    }
    files.update({
        f"study-qps{rate}.json": (study(rate).model_dump_json(indent=2) + "\n").encode()
        for rate in RATES
    })
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = expected_files()
    if args.check:
        actual = {path.name: path.read_bytes() for path in HERE.glob("*.json")}
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            changed = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
            raise SystemExit(
                f"M3 config drift: missing={missing}, extra={extra}, changed={changed}"
            )
        return 0
    for name, payload in expected.items():
        (HERE / name).write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
