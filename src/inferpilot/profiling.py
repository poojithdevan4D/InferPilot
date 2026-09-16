"""CLI: derive an immutable WorkloadProfile from observations.

Fully offline — it reads local JSON/JSONL observation evidence and writes a
self-validating profile. It never contacts a model server, Hugging Face, or the
network. Run with ``python -m inferpilot.profiling <observations> <output>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .workload_profile import WorkloadObservation, build_workload_profile


def _load_observations(path: Path) -> list[WorkloadObservation]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON observations must be a list")
        rows = data
    return [WorkloadObservation.model_validate(row) for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m inferpilot.profiling",
        description="Derive an immutable WorkloadProfile from observations "
                    "(offline; never contacts a server or the network).",
    )
    parser.add_argument("observations", type=Path, help="JSON array or .jsonl of observations")
    parser.add_argument("output", type=Path, help="profile output path (refuses to overwrite)")
    parser.add_argument("--min-observations", type=int, default=2,
                        help="minimum observations required (default 2)")
    args = parser.parse_args(argv)

    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2

    try:
        observations = _load_observations(args.observations)
    except (json.JSONDecodeError, ValidationError, ValueError, OSError) as exc:
        print(f"malformed observations: {exc}", file=sys.stderr)
        return 2

    if not observations:
        print("empty evidence: no observations", file=sys.stderr)
        return 2
    if len(observations) < args.min_observations:
        print(
            f"insufficient evidence: {len(observations)} observation(s) < "
            f"{args.min_observations} required",
            file=sys.stderr,
        )
        return 2

    try:
        profile = build_workload_profile(observations)
    except (ValidationError, ValueError) as exc:
        print(f"invalid observations: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(profile.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
