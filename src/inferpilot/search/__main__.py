"""CLI for fixed-budget search replay baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..comparison.models import BlockedStudyReport
from .models import ReplaySearchSpec
from .replay import evaluate_replay_search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m inferpilot.search")
    parser.add_argument("source", type=Path, help="complete BlockedStudyReport JSON")
    parser.add_argument("spec", type=Path, help="ReplaySearchSpec JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    source = BlockedStudyReport.model_validate_json(args.source.read_text())
    spec = ReplaySearchSpec.model_validate_json(args.spec.read_text())
    report = evaluate_replay_search(source, spec)
    payload = report.model_dump_json(indent=2)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite immutable report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
