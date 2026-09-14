"""Generate and verify the preregistered C6 experiment matrix.

The checked-in JSON files are the executable protocol.  This small generator
keeps the 5-seed x 4-candidate matrix mechanical and provides a ``--check``
mode so accidental drift is caught before a GPU run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferpilot import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "experiments/c5-openloop/seq1.json"
SEEDS = (1, 2, 3, 4, 5)
MAX_NUM_SEQS = (1, 2, 3, 4)


def expected_payload(seed: int, max_num_seqs: int) -> dict:
    payload = json.loads(BASE_CONFIG.read_text())
    experiment_id = f"c6-s{seed}-seq{max_num_seqs}"
    payload.update(
        experiment_id=experiment_id,
        name=f"c6-openloop-s{seed}-seq{max_num_seqs}",
        description=(
            "C6 preregistered blocked confirmatory study: nominal 8 QPS, "
            f"workload seed {seed}, max_num_seqs={max_num_seqs}. The explicit "
            "study SLO and objective live in study.json, not in this run config."
        ),
        slo=None,
        seed=0,
        tags=["c6", "open-loop", "poisson-v1", "blocked", "confirmatory"],
    )
    payload["engine"]["max_num_seqs"] = max_num_seqs
    payload["workload"]["name"] = "c6-openloop-poisson8"
    payload["workload"]["seed"] = seed
    # Validate with the actual contract before returning anything writable.
    return ExperimentConfig.model_validate(payload).model_dump(mode="json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked-in configs differ from the deterministic matrix",
    )
    args = parser.parse_args()

    mismatches: list[str] = []
    for seed in SEEDS:
        for seqs in MAX_NUM_SEQS:
            path = HERE / f"s{seed}-seq{seqs}.json"
            expected = expected_payload(seed, seqs)
            if args.check:
                if not path.is_file() or json.loads(path.read_text()) != expected:
                    mismatches.append(path.name)
            else:
                path.write_text(json.dumps(expected, indent=2) + "\n")

    if mismatches:
        raise SystemExit(f"C6 config drift: {', '.join(mismatches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
