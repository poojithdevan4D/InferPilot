"""Unified operator CLI for InferPilot's first product vertical slice."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from pydantic import ValidationError

from .advisor.capacity_advisory import OperatorEconomics
from .config import ExperimentConfig, SLO
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
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"cannot analyze evidence bundle: {exc}", file=sys.stderr)
        return 2
    _print_summary(card)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(card.model_dump_json(indent=2) + "\n")
        print(f"Evidence card: {args.output}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    from .demo import build_demo_card

    if args.output is not None and args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    card = build_demo_card()
    print("InferPilot quickstart (synthetic metadata; no GPU)\n")
    _print_summary(card)
    print("\nWhy: aligned arrivals/completions show backlog growth while KV is 99% full")
    print("      and preemptions are observed. The candidate is a test, not a deployment.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(card.model_dump_json(indent=2) + "\n")
        print(f"\nEvidence card: {args.output}")
    return 0


def _print_preflight(plan) -> None:
    print("InferPilot assessment preflight\n")
    for check in plan.checks:
        marker = {"pass": "PASS", "fail": "FAIL", "unknown": "UNKNOWN"}[check.status]
        print(f"[{marker:7}] {check.name}: {check.detail}")
    print(f"\nCommand: {shlex.join(plan.planned_command)}")
    print(
        "Hard budget: "
        f"{plan.budget.max_wall_time_s:g}s wall / "
        f"{plan.budget.max_gpu_seconds:g} GPU-s / "
        f"${plan.budget.maximum_cost_usd:.4f} maximum"
    )
    print("Execution allowed: " + ("yes" if plan.execution_allowed else "no"))


def _assess(args: argparse.Namespace) -> int:
    from .assessment import build_budget, create_assessment

    try:
        config = ExperimentConfig.model_validate_json(args.config.read_text())
        budget = build_budget(
            max_wall_time_s=args.max_wall_time_s,
            gpu_cost_per_hour_usd=args.gpu_cost_per_hour,
            gpu_count=args.gpu_count,
            max_cost_usd=args.max_cost_usd,
        )
        outcome = create_assessment(
            config,
            budget,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            ready_timeout_s=args.ready_timeout,
            request_timeout_s=args.request_timeout,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"cannot create assessment: {exc}", file=sys.stderr)
        return 2

    _print_preflight(outcome.plan)
    print(f"Assessment artifacts: {outcome.assessment_dir}")
    if args.dry_run or not outcome.plan.execution_allowed:
        if args.dry_run and outcome.plan.execution_allowed:
            print("Dry run passed. Run without --dry-run to execute.")
            return 0
        print("Assessment stopped before GPU execution.")
        return 2
    assert outcome.result is not None
    print(f"\nExecution: {outcome.result.status.value}")
    print(f"Run bundle: {outcome.run_dir}")
    if outcome.result.status.value != "completed":
        assert outcome.result.failure is not None
        print(
            f"Failure: {outcome.result.failure.error_type}: "
            f"{outcome.result.failure.message}",
            file=sys.stderr,
        )
        return 1
    if outcome.card is None:
        print("No Evidence Card was produced.", file=sys.stderr)
        return 1
    print()
    _print_summary(outcome.card)
    if outcome.candidate_plan is not None:
        print("Unverified candidate preconditions:")
        for criterion in outcome.candidate_plan.acceptance_criteria:
            if "quality" in criterion or "accuracy" in criterion or "pareto" in criterion:
                print(f"  - {criterion}")
        print("Candidate execution: NOT AUTHORIZED; review experiment-plan.json first")
    return 0


def _gate(args: argparse.Namespace) -> int:
    from .configuration_gate import (
        BoundQualityEvidence,
        ConfigurationGateSpec,
        evaluate_configuration_gate,
    )

    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    try:
        spec = ConfigurationGateSpec.model_validate_json(args.spec.read_text())
        baseline_path, _, _ = _bundle_files(args.baseline)
        candidate_path, _, _ = _bundle_files(args.candidate)
        baseline = ExperimentResult.model_validate_json(baseline_path.read_text())
        candidate = ExperimentResult.model_validate_json(candidate_path.read_text())
        quality = tuple(
            BoundQualityEvidence.model_validate_json(path.read_text())
            for path in args.quality_evidence
        )
        report = evaluate_configuration_gate(spec, baseline, candidate, quality)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"cannot evaluate configuration gate: {exc}", file=sys.stderr)
        return 2

    print(f"Configuration gate: {report.verdict.upper()}")
    if report.comparison is not None:
        print(f"Performance: {report.comparison.verdict}")
    for check in report.candidate_slo_checks:
        print(
            f"SLO: {check.metric}={check.observed_worst:g} "
            f"{check.operator} {check.threshold:g} "
            f"({'PASS' if check.passed else 'FAIL'})"
        )
    for reason in report.reasons:
        print(f"Reason: {reason}")
    print("Deployment: NOT AUTHORIZED; operator review and canary remain required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n")
    print(f"Gate report: {args.output}")
    return {"pass": 0, "fail": 1, "abstain": 2}[report.verdict]


def _bind_quality(args: argparse.Namespace) -> int:
    from .configuration_gate import bind_quality_evidence
    from .quality import KLQualityGate, NeedleQualityGate

    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    try:
        baseline_path, _, _ = _bundle_files(args.baseline)
        candidate_path, _, _ = _bundle_files(args.candidate)
        baseline = ExperimentResult.model_validate_json(baseline_path.read_text())
        candidate = ExperimentResult.model_validate_json(candidate_path.read_text())
        raw_gate = args.gate.read_text()
        gate_type = {
            "teacher-forced-kl": KLQualityGate,
            "needle-retrieval": NeedleQualityGate,
        }[args.kind]
        gate = gate_type.model_validate_json(raw_gate)
        evidence = bind_quality_evidence(
            baseline,
            candidate,
            corpus_id=args.corpus_id,
            corpus_sha256=args.corpus_sha256,
            gate=gate,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"cannot bind quality evidence: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(evidence.model_dump_json(indent=2) + "\n")
    print(f"Bound quality evidence: {args.output}")
    print(f"Gate: {evidence.kind} ({'PASS' if evidence.gate.passed else 'FAIL'})")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferpilot",
        description="Evidence-gated optimization for LLM inference serving.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser(
        "demo",
        help="run the complete decision pipeline on synthetic metadata (no GPU)",
    )
    demo.add_argument("--output", type=Path, help="optionally write the Evidence Card JSON")
    demo.set_defaults(handler=_demo)
    assess = sub.add_parser(
        "assess",
        help="preflight and optionally run one bounded inference assessment",
    )
    assess.add_argument("config", type=Path, help="ExperimentConfig JSON")
    assess.add_argument("--dry-run", action="store_true", help="plan only; never start a server")
    assess.add_argument("--output-dir", type=Path, default=Path("assessments"))
    assess.add_argument("--max-wall-time-s", type=float, default=600.0)
    assess.add_argument("--gpu-cost-per-hour", type=float, required=True)
    assess.add_argument("--gpu-count", type=int, default=1)
    assess.add_argument("--max-cost-usd", type=float)
    assess.add_argument("--ready-timeout", type=float, default=300.0)
    assess.add_argument("--request-timeout", type=float, default=120.0)
    assess.set_defaults(handler=_assess)
    gate = sub.add_parser(
        "gate",
        help="evaluate one controlled candidate as PASS, FAIL, or ABSTAIN",
    )
    gate.add_argument("spec", type=Path, help="ConfigurationGateSpec JSON")
    gate.add_argument("baseline", type=Path, help="baseline run directory or result.json")
    gate.add_argument("candidate", type=Path, help="candidate run directory or result.json")
    gate.add_argument(
        "--quality-evidence",
        type=Path,
        action="append",
        default=[],
        help="bound quality-evidence JSON; repeat for every required gate",
    )
    gate.add_argument("--output", type=Path, required=True)
    gate.set_defaults(handler=_gate)
    bind_quality = sub.add_parser(
        "bind-quality",
        help="bind a measured quality gate to exact baseline/candidate runs",
    )
    bind_quality.add_argument(
        "kind", choices=("teacher-forced-kl", "needle-retrieval")
    )
    bind_quality.add_argument("gate", type=Path, help="measured quality-gate JSON")
    bind_quality.add_argument("baseline", type=Path)
    bind_quality.add_argument("candidate", type=Path)
    bind_quality.add_argument("--corpus-id", required=True)
    bind_quality.add_argument("--corpus-sha256", required=True)
    bind_quality.add_argument("--output", type=Path, required=True)
    bind_quality.set_defaults(handler=_bind_quality)
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
