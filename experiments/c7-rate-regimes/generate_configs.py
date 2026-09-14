"""Generate and verify the preregistered C7 rate-regime matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferpilot import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "experiments/c5-openloop/seq1.json"
RATES = (4, 6, 8)
SEEDS = (6, 7, 8)
MAX_NUM_SEQS = (1, 2, 3, 4)


def expected_payload(rate: int, seed: int, max_num_seqs: int) -> dict:
    payload = json.loads(BASE_CONFIG.read_text())
    experiment_id = f"c7-qps{rate}-s{seed}-seq{max_num_seqs}"
    payload.update(
        experiment_id=experiment_id,
        name=experiment_id,
        description=(
            "C7 preregistered rate-regime study: "
            f"nominal {rate} QPS, workload seed {seed}, "
            f"max_num_seqs={max_num_seqs}. Decision policy is in study-qps{rate}.json."
        ),
        slo=None,
        seed=0,
        tags=["c7", "open-loop", "poisson-v1", "rate-regime", "confirmatory"],
    )
    payload["engine"]["max_num_seqs"] = max_num_seqs
    payload["workload"]["name"] = f"c7-openloop-poisson-qps{rate}"
    payload["workload"]["request_rate_qps"] = float(rate)
    payload["workload"]["seed"] = seed
    return ExperimentConfig.model_validate(payload).model_dump(mode="json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    mismatches: list[str] = []
    for rate in RATES:
        for seed in SEEDS:
            for seqs in MAX_NUM_SEQS:
                path = HERE / f"qps{rate}-s{seed}-seq{seqs}.json"
                expected = expected_payload(rate, seed, seqs)
                if args.check:
                    if not path.is_file() or json.loads(path.read_text()) != expected:
                        mismatches.append(path.name)
                else:
                    path.write_text(json.dumps(expected, indent=2) + "\n")
    if mismatches:
        raise SystemExit(f"C7 config drift: {', '.join(mismatches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
