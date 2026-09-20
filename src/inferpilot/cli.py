"""Unified operator CLI for InferPilot's first product vertical slice."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .advisor.capacity_advisory import OperatorEconomics
from .config import SLO
from .evidence_card import OptimizationEvidenceCard, build_evidence_card
from .mechanism import MechanismEvidence
from .phases import RunnerPhaseTiming
from .results import ExperimentResult


def _bundle_files(path: Path) -> tuple[Path, Path | None, Path | None]:
    if path.is_dir():
        return path / "result.json", path / "phases.json", path / "mechanism-evidence.json"
    return path, None, None


def _load_card(args: argparse.Namespace) -> OptimizationEvidenceCard:
    result_path, phases_path, mechanism_path = _bundle_files(args.bundle)
    result = ExperimentResult.model_validate_json(result_path.read_text())
    occupancy = None
    if phases_path is not None and phases_path.is_file():
        timing = RunnerPhaseTiming.model_validate_json(phases_path.read_text())
        occupancy = next(
            phase.duration_s for phase in timing.phases if phase.name == "total_occupancy"
        )
    mechanism = None
    if mechanism_path is not None and mechanism_path.is_file():
        mechanism = MechanismEvidence.model_validate_json(mechanism_path.read_text())
    slo = SLO(
        ttft_p95_ms=args.ttft_p95_ms,
        tpot_p95_ms=args.tpot_p95_ms,
        e2e_p95_ms=args.e2e_p95_ms,
        min_throughput_tokens_per_s=args.min_throughput_tokens_per_s,
    )
    economics = OperatorEconomics(
        gpu_cost_per_hour_usd=args.gpu_cost_per_hour,
        gpu_count=args.gpu_count,
        target_qps=args.target_qps,
    )
    return build_evidence_card(
        result,
        slo,
        economics,
        total_occupancy_s=occupancy,
        mechanism_evidence=mechanism,
    )


def _print_summary(card: OptimizationEvidenceCard) -> None:
    diagnosis = card.recommendation.advisory.diagnosis
    print(f"Status: {card.status}")
    print(f"Diagnosis: {diagnosis.regime} (load={diagnosis.load_state})")
    print(f"Evidence: {card.evidence_strength}; transferability={card.transferability}")
    if card.candidate_engine_overrides:
        print(f"Candidate: {card.candidate_engine_overrides}")
        print(
            "Estimated paired experiment: "
            f"{card.estimated_paired_experiment_gpu_seconds:.1f} GPU-s / "
            f"${card.estimated_paired_experiment_cost_usd:.4f}"
        )
    else:
        print("Candidate: none")
    cost = card.observed_gpu_cost_per_million_output_tokens_usd
    print(
        "Observed baseline GPU cost: "
        + ("unavailable" if cost is None else f"${cost:.2f}/1M successful output tokens")
    )
    print("Next: " + (
        "run the registered baseline/candidate experiment and quality gates"
        if card.status == "experiment_recommended"
        else "keep the current configuration for this observed window"
        if card.status == "keep_current"
        else "collect stronger aligned evidence; InferPilot abstains"
    ))


def _analyze(args: argparse.Namespace) -> int:
    if not any(
        value is not None
        for value in (
            args.ttft_p95_ms,
            args.tpot_p95_ms,
            args.e2e_p95_ms,
            args.min_throughput_tokens_per_s,
        )
    ):
        print("at least one SLO constraint is required", file=sys.stderr)
        return 2
    if args.output is not None and args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    try:
        card = _load_card(args)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"cannot analyze evidence bundle: {exc}", file=sys.stderr)
        return 2
    _print_summary(card)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(card.model_dump_json(indent=2) + "\n")
        print(f"Evidence card: {args.output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inferpilot")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser(
        "analyze",
        help="diagnose one metadata-only run bundle and recommend a bounded experiment",
    )
    analyze.add_argument("bundle", type=Path, help="run directory or result.json")
    analyze.add_argument("--ttft-p95-ms", type=float)
    analyze.add_argument("--tpot-p95-ms", type=float)
    analyze.add_argument("--e2e-p95-ms", type=float)
    analyze.add_argument("--min-throughput-tokens-per-s", type=float)
    analyze.add_argument("--gpu-cost-per-hour", type=float, required=True)
    analyze.add_argument("--gpu-count", type=int, default=1)
    analyze.add_argument("--target-qps", type=float)
    analyze.add_argument("--output", type=Path, help="write immutable-style evidence-card JSON")
    analyze.set_defaults(handler=_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
