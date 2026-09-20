"""Guided, bounded assessment over the existing benchmark runner.

This module owns product orchestration only.  Measurement remains exclusively in
``runner.orchestrator`` and diagnosis remains exclusively in the advisor stack.
"""

from __future__ import annotations

import math
import stat
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel
from .advisor.capacity_advisory import OperatorEconomics
from .config import ExperimentConfig
from .environment import EnvironmentMetadata
from .evidence_card import OptimizationEvidenceCard, build_evidence_card
from .mechanism import MechanismEvidence
from .phases import RunnerPhaseTiming
from .results import ExperimentResult
from .runner.artifacts import create_run_dir
from .runner.defaults import BENCH_PYTHON_VERSION, PINNED_VLLM_VERSION
from .runner.orchestrator import CommandBuilder, capture_environment, run_experiment
from .runner.server import build_server_env, build_vllm_command, default_vllm_executable


CheckStatus = Literal["pass", "fail", "unknown"]
ASSESSMENT_TERMINATE_RESERVE_S = 15.0


class AssessmentBudget(SchemaModel):
    """Operator-owned hard execution ceilings; estimates never replace them."""

    max_wall_time_s: float = Field(gt=0, allow_inf_nan=False)
    gpu_count: int = Field(default=1, ge=1)
    gpu_cost_per_hour_usd: float = Field(gt=0, allow_inf_nan=False)
    max_cost_usd: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    max_gpu_seconds: float = Field(gt=0, allow_inf_nan=False)
    maximum_cost_usd: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check(self) -> "AssessmentBudget":
        expected_gpu_s = self.max_wall_time_s * self.gpu_count
        expected_cost = expected_gpu_s * self.gpu_cost_per_hour_usd / 3600
        if self.max_gpu_seconds != expected_gpu_s:
            raise ValueError("max_gpu_seconds contradicts wall-time and GPU-count ceilings")
        if not math.isclose(self.maximum_cost_usd, expected_cost):
            raise ValueError("maximum_cost_usd contradicts the configured ceilings")
        return self


def build_budget(
    *,
    max_wall_time_s: float,
    gpu_cost_per_hour_usd: float,
    gpu_count: int = 1,
    max_cost_usd: Optional[float] = None,
) -> AssessmentBudget:
    gpu_seconds = max_wall_time_s * gpu_count
    return AssessmentBudget(
        max_wall_time_s=max_wall_time_s,
        gpu_count=gpu_count,
        gpu_cost_per_hour_usd=gpu_cost_per_hour_usd,
        max_cost_usd=max_cost_usd,
        max_gpu_seconds=gpu_seconds,
        maximum_cost_usd=gpu_seconds * gpu_cost_per_hour_usd / 3600,
    )


class AssessmentRuntime(SchemaModel):
    """Read-only local facts collected without loading a model or allocating VRAM."""

    environment: EnvironmentMetadata
    vllm_executable: str
    vllm_version: Optional[str] = None
    executable_exists: bool


def capture_assessment_runtime(config: ExperimentConfig) -> AssessmentRuntime:
    executable = default_vllm_executable()
    try:
        version = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        version = None
    return AssessmentRuntime(
        environment=capture_environment(
            config.engine, build_server_env(config.engine)
        ),
        vllm_executable=executable,
        vllm_version=version,
        executable_exists=Path(executable).is_file(),
    )


class PreflightCheck(SchemaModel):
    name: str = Field(min_length=1)
    status: CheckStatus
    detail: str = Field(min_length=1)
    blocks_execution: bool


def _has_slo(config: ExperimentConfig) -> bool:
    slo = config.slo
    return bool(
        slo
        and any(
            value is not None
            for value in (
                slo.ttft_p95_ms,
                slo.tpot_p95_ms,
                slo.e2e_p95_ms,
                slo.min_throughput_tokens_per_s,
            )
        )
    )


def _checks(
    config: ExperimentConfig,
    budget: AssessmentBudget,
    runtime: AssessmentRuntime,
) -> tuple[PreflightCheck, ...]:
    workload = config.workload
    checks: list[PreflightCheck] = []

    def add(name: str, passed: bool, success: str, failure: str) -> None:
        checks.append(
            PreflightCheck(
                name=name,
                status="pass" if passed else "fail",
                detail=success if passed else failure,
                blocks_execution=not passed,
            )
        )

    add(
        "model_revision_pinned",
        config.engine.revision is not None,
        f"revision={config.engine.revision}",
        "an exact model revision is required for a reproducible assessment",
    )
    add(
        "operator_slo_present",
        _has_slo(config),
        "at least one explicit SLO constraint is present",
        "assessment requires an operator-supplied SLO",
    )
    add(
        "aligned_evidence_workload",
        workload.request_rate_qps is not None
        and workload.num_requests >= 100
        and workload.warmup_requests >= 1,
        "open-loop workload can produce aligned evidence if runtime coverage reaches 30s",
        "use open-loop arrivals, at least 100 measured requests, and at least one warm-up",
    )
    add(
        "vllm_installed",
        runtime.vllm_version is not None and runtime.executable_exists,
        f"vllm {runtime.vllm_version} executable={runtime.vllm_executable}",
        "vLLM is not installed beside the current Python interpreter",
    )
    python_version = runtime.environment.python_version
    add(
        "benchmark_python_version",
        bool(python_version and python_version.startswith(BENCH_PYTHON_VERSION + ".")),
        f"Python {python_version} matches pinned {BENCH_PYTHON_VERSION}",
        f"expected Python {BENCH_PYTHON_VERSION}, found {python_version or 'unknown'}",
    )
    add(
        "vllm_version_pinned",
        runtime.vllm_version == PINNED_VLLM_VERSION,
        f"vllm version matches pinned {PINNED_VLLM_VERSION}",
        f"expected vllm {PINNED_VLLM_VERSION}, found {runtime.vllm_version or 'none'}",
    )
    visible = runtime.environment.hardware.gpu_count
    add(
        "gpu_count_available",
        visible >= budget.gpu_count,
        f"{visible} visible GPU(s); {budget.gpu_count} requested",
        f"{visible} visible GPU(s); {budget.gpu_count} required",
    )
    within_cost = (
        budget.max_cost_usd is None
        or budget.maximum_cost_usd <= budget.max_cost_usd
    )
    add(
        "cost_ceiling",
        within_cost,
        f"maximum ${budget.maximum_cost_usd:.4f} is within the operator ceiling",
        (
            f"maximum ${budget.maximum_cost_usd:.4f} exceeds "
            f"${budget.max_cost_usd:.4f}"
            if budget.max_cost_usd is not None
            else "cost ceiling check failed"
        ),
    )
    add(
        "cleanup_budget_reserved",
        budget.max_wall_time_s > ASSESSMENT_TERMINATE_RESERVE_S,
        (
            f"{ASSESSMENT_TERMINATE_RESERVE_S:g}s of the wall-time ceiling "
            "is reserved for forced cleanup"
        ),
        (
            f"max wall time must exceed the "
            f"{ASSESSMENT_TERMINATE_RESERVE_S:g}s cleanup reserve"
        ),
    )
    checks.append(
        PreflightCheck(
            name="model_fit",
            status="unknown",
            detail=(
                "static metadata cannot prove model+KV+runtime memory fit; "
                "startup remains a measured, structured failure boundary"
            ),
            blocks_execution=False,
        )
    )
    checks.append(
        PreflightCheck(
            name="candidate_quality",
            status="unknown",
            detail="candidate quality and long-context gates have not run",
            blocks_execution=False,
        )
    )
    return tuple(checks)


def _planned_command(config: ExperimentConfig, runtime: AssessmentRuntime) -> tuple[str, ...]:
    command = build_vllm_command(
        config.engine,
        0,
        vllm_executable=runtime.vllm_executable,
    )
    port_index = command.index("--port") + 1
    command[port_index] = "<free-port>"
    return tuple(command)


class AssessmentPlan(SchemaModel):
    plan_version: Literal["0.1.0"] = "0.1.0"
    config: ExperimentConfig
    budget: AssessmentBudget
    runtime: AssessmentRuntime
    planned_command: tuple[str, ...]
    checks: tuple[PreflightCheck, ...]
    execution_allowed: bool
    dry_run_claim: Literal[
        "no_server_launch_no_model_load_no_gpu_allocation"
    ] = "no_server_launch_no_model_load_no_gpu_allocation"

    @model_validator(mode="after")
    def _check(self) -> "AssessmentPlan":
        expected_checks = _checks(self.config, self.budget, self.runtime)
        expected_allowed = not any(
            check.blocks_execution and check.status != "pass"
            for check in expected_checks
        )
        if self.checks != expected_checks:
            raise ValueError("preflight checks contradict assessment inputs")
        if self.planned_command != _planned_command(self.config, self.runtime):
            raise ValueError("planned command contradicts assessment config/runtime")
        if self.execution_allowed != expected_allowed:
            raise ValueError("execution_allowed contradicts preflight checks")
        return self


def build_assessment_plan(
    config: ExperimentConfig,
    budget: AssessmentBudget,
    *,
    runtime: Optional[AssessmentRuntime] = None,
) -> AssessmentPlan:
    facts = runtime or capture_assessment_runtime(config)
    checks = _checks(config, budget, facts)
    return AssessmentPlan(
        config=config,
        budget=budget,
        runtime=facts,
        planned_command=_planned_command(config, facts),
        checks=checks,
        execution_allowed=not any(
            check.blocks_execution and check.status != "pass" for check in checks
        ),
    )


class CandidateExperimentPlan(SchemaModel):
    plan_version: Literal["0.1.0"] = "0.1.0"
    source_card: OptimizationEvidenceCard
    candidate_config: ExperimentConfig
    varied_engine_fields: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    execution_authorized: Literal[False] = False
    next_action: Literal["review_then_run_controlled_candidate"] = (
        "review_then_run_controlled_candidate"
    )

    @model_validator(mode="after")
    def _check(self) -> "CandidateExperimentPlan":
        card = self.source_card
        overrides = card.candidate_engine_overrides
        if card.status != "experiment_recommended" or not overrides:
            raise ValueError("candidate plan requires an experiment recommendation")
        expected_engine = card.baseline_result.config.engine.model_copy(update=overrides)
        expected_config = card.baseline_result.config.model_copy(
            update={
                "experiment_id": card.baseline_result.config.experiment_id + "-candidate",
                "name": card.baseline_result.config.name + " candidate",
                "engine": expected_engine,
                "slo": card.slo,
                "tags": card.baseline_result.config.tags
                + ["inferpilot-generated-candidate"],
            }
        )
        if self.candidate_config != expected_config:
            raise ValueError("candidate config contradicts the Evidence Card")
        if self.varied_engine_fields != tuple(sorted(overrides)):
            raise ValueError("varied engine fields contradict candidate overrides")
        if self.acceptance_criteria != card.acceptance_criteria:
            raise ValueError("acceptance criteria contradict the Evidence Card")
        return self


def build_candidate_plan(card: OptimizationEvidenceCard) -> CandidateExperimentPlan:
    overrides = card.candidate_engine_overrides
    if card.status != "experiment_recommended" or not overrides:
        raise ValueError("Evidence Card does not support a candidate experiment")
    base = card.baseline_result.config
    candidate = base.model_copy(
        update={
            "experiment_id": base.experiment_id + "-candidate",
            "name": base.name + " candidate",
            "engine": base.engine.model_copy(update=overrides),
            "slo": card.slo,
            "tags": base.tags + ["inferpilot-generated-candidate"],
        }
    )
    return CandidateExperimentPlan(
        source_card=card,
        candidate_config=candidate,
        varied_engine_fields=tuple(sorted(overrides)),
        acceptance_criteria=card.acceptance_criteria,
    )


def _write_immutable(path: Path, model: SchemaModel) -> Path:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.write_text(model.model_dump_json(indent=2) + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path


@dataclass(frozen=True)
class AssessmentOutcome:
    assessment_dir: Path
    plan: AssessmentPlan
    run_dir: Optional[Path] = None
    result: Optional[ExperimentResult] = None
    card: Optional[OptimizationEvidenceCard] = None
    candidate_plan: Optional[CandidateExperimentPlan] = None


def create_assessment(
    config: ExperimentConfig,
    budget: AssessmentBudget,
    *,
    output_dir: str | Path = "assessments",
    dry_run: bool = False,
    runtime: Optional[AssessmentRuntime] = None,
    command_builder: Optional[CommandBuilder] = None,
    ready_timeout_s: float = 300.0,
    request_timeout_s: float = 120.0,
) -> AssessmentOutcome:
    """Plan and optionally execute one bounded assessment.

    Test-only callers may inject runtime facts and a fake-server command builder;
    the CLI never does so.
    """

    plan = build_assessment_plan(config, budget, runtime=runtime)
    assessment_dir = create_run_dir(output_dir, config.experiment_id + "-assessment")
    _write_immutable(assessment_dir / "assessment-plan.json", plan)
    if dry_run or not plan.execution_allowed:
        return AssessmentOutcome(assessment_dir=assessment_dir, plan=plan)

    observed_run_dirs: list[Path] = []
    result = run_experiment(
        config,
        str(assessment_dir / "run"),
        command_builder=command_builder,
        ready_timeout_s=min(ready_timeout_s, budget.max_wall_time_s),
        request_timeout_s=min(request_timeout_s, budget.max_wall_time_s),
        terminate_timeout_s=ASSESSMENT_TERMINATE_RESERVE_S,
        max_wall_time_s=(
            budget.max_wall_time_s - ASSESSMENT_TERMINATE_RESERVE_S
        ),
        on_run_dir=observed_run_dirs.append,
    )
    if len(observed_run_dirs) != 1:
        raise RuntimeError("runner did not report exactly one artifact directory")
    run_dir = observed_run_dirs[0]
    card = None
    candidate_plan = None
    if result.status.value == "completed" and config.slo is not None:
        phases_path = run_dir / "phases.json"
        occupancy = None
        if phases_path.is_file():
            phases = RunnerPhaseTiming.model_validate_json(phases_path.read_text())
            occupancy = next(
                phase.duration_s
                for phase in phases.phases
                if phase.name == "total_occupancy"
            )
        mechanism_path = run_dir / "mechanism-evidence.json"
        mechanism = (
            MechanismEvidence.model_validate_json(mechanism_path.read_text())
            if mechanism_path.is_file()
            else None
        )
        card = build_evidence_card(
            result,
            config.slo,
            OperatorEconomics(
                gpu_cost_per_hour_usd=budget.gpu_cost_per_hour_usd,
                gpu_count=budget.gpu_count,
                target_qps=None,
            ),
            total_occupancy_s=occupancy,
            mechanism_evidence=mechanism,
        )
        _write_immutable(assessment_dir / "evidence-card.json", card)
        if card.status == "experiment_recommended":
            candidate_plan = build_candidate_plan(card)
            _write_immutable(assessment_dir / "experiment-plan.json", candidate_plan)
    return AssessmentOutcome(
        assessment_dir=assessment_dir,
        plan=plan,
        run_dir=run_dir,
        result=result,
        card=card,
        candidate_plan=candidate_plan,
    )
