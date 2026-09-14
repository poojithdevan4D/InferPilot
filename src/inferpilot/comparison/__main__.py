"""CLI for ingesting, summarizing, and comparing stored run cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .blocked import evaluate_blocked_study
from .compare import build_cohort, compare_cohorts
from .decision import evaluate_study
from .frontier import build_pareto_frontier
from .models import BlockedStudySpec, StudySpec
from .store import ResultStore


def _write_report(path: Path, payload: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inferpilot.comparison")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("store", type=Path)
    ingest.add_argument("run_dirs", nargs="+", type=Path)

    summary = subparsers.add_parser("summary")
    summary.add_argument("store", type=Path)
    summary.add_argument("experiment_id")
    summary.add_argument("--min-runs", type=int, default=3)

    compare = subparsers.add_parser("compare")
    compare.add_argument("store", type=Path)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--vary", action="append", required=True)
    compare.add_argument("--min-runs", type=int, default=3)
    compare.add_argument("--output", type=Path)

    frontier = subparsers.add_parser("frontier")
    frontier.add_argument("store", type=Path)
    frontier.add_argument("--experiment", action="append", required=True)
    frontier.add_argument("--vary", action="append", required=True)
    frontier.add_argument("--objective", action="append", required=True)
    frontier.add_argument("--min-runs", type=int, default=3)
    frontier.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("store", type=Path)
    evaluate.add_argument("study", type=Path)
    evaluate.add_argument("--output", type=Path)

    blocked = subparsers.add_parser("blocked-evaluate")
    blocked.add_argument("store", type=Path)
    blocked.add_argument("study", type=Path)
    blocked.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    store = ResultStore(args.store)

    if args.command == "ingest":
        output = []
        for run_dir in args.run_dirs:
            ingested = store.ingest(run_dir)
            output.append(
                {"run_id": ingested.run_id, "path": str(ingested.path), "created": ingested.created}
            )
        print(json.dumps(output, indent=2))
        return 0

    if args.command == "summary":
        cohort = build_cohort(
            store.list_runs(args.experiment_id), min_runs=args.min_runs
        )
        print(cohort.model_dump_json(indent=2))
        return 0

    if args.command == "compare":
        report = compare_cohorts(
            store.list_runs(args.baseline),
            store.list_runs(args.candidate),
            varied_engine_fields=args.vary,
            min_runs=args.min_runs,
        )
    elif args.command == "frontier":
        report = build_pareto_frontier(
            [store.list_runs(experiment_id) for experiment_id in args.experiment],
            varied_engine_fields=args.vary,
            objective_metrics=args.objective,
            min_runs=args.min_runs,
        )
    elif args.command == "evaluate":
        study = StudySpec.model_validate_json(args.study.read_text())
        report = evaluate_study(
            [store.list_runs(experiment_id) for experiment_id in study.experiment_ids],
            study,
        )
    else:
        study = BlockedStudySpec.model_validate_json(args.study.read_text())
        report = evaluate_blocked_study(
            [
                [store.list_runs(eid) for eid in block.experiment_ids]
                for block in study.blocks
            ],
            study,
        )
    payload = report.model_dump_json(indent=2)
    if args.output is not None:
        _write_report(args.output, payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
