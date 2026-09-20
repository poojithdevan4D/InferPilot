from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferpilot import (
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    HardwareInfo,
    SLO,
    WorkloadSpec,
    build_evidence_card,
)
from inferpilot.advisor import OperatorEconomics
from inferpilot.assessment import (
    AssessmentPlan,
    AssessmentRuntime,
    build_assessment_plan,
    build_budget,
    build_candidate_plan,
    create_assessment,
)
from inferpilot.cli import main
from inferpilot.runner.defaults import PINNED_VLLM_VERSION
from test_load_state import make_evidence, make_result


FAKE = Path(__file__).parent / "fake_vllm_server.py"


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="assessment-test",
        name="assessment test",
        engine=EngineConfig(
            model="fake/model",
            revision="deadbeef",
            max_model_len=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=False,
        ),
        workload=WorkloadSpec(
            name="open-loop",
            num_requests=100,
            warmup_requests=1,
            prompt_tokens=128,
            output_tokens=8,
            request_rate_qps=100.0,
            ignore_eos=True,
        ),
        slo=SLO(ttft_p95_ms=500, tpot_p95_ms=60),
    )


def _runtime(*, gpu_count=1, version=PINNED_VLLM_VERSION) -> AssessmentRuntime:
    return AssessmentRuntime(
        environment=EnvironmentMetadata(
            python_version="3.12.13",
            hardware=HardwareInfo(
                gpu_name="Synthetic GPU", gpu_count=gpu_count, gpu_memory_total_mb=24_576
            )
        ),
        vllm_executable=sys.executable,
        vllm_version=version,
        executable_exists=True,
    )


def _budget(**changes):
    values = dict(max_wall_time_s=30, gpu_cost_per_hour_usd=1.20, gpu_count=1)
    values.update(changes)
    return build_budget(**values)


def _builder(port: int) -> list[str]:
    return [
        sys.executable,
        str(FAKE),
        "--port",
        str(port),
        "--mode",
        "normal",
        "--output-tokens",
        "8",
        "--emit-effective",
        "False",
    ]


def test_dry_plan_is_gpu_free_explicit_and_self_validating():
    plan = build_assessment_plan(_config(), _budget(), runtime=_runtime())
    assert plan.execution_allowed
    assert plan.dry_run_claim == "no_server_launch_no_model_load_no_gpu_allocation"
    assert "<free-port>" in plan.planned_command
    assert plan.budget.max_gpu_seconds == 30
    assert plan.budget.maximum_cost_usd == pytest.approx(0.01)
    assert AssessmentPlan.model_validate_json(plan.model_dump_json()) == plan

    raw = plan.model_dump(mode="json")
    raw["execution_allowed"] = False
    with pytest.raises(ValidationError, match="execution_allowed contradicts"):
        AssessmentPlan.model_validate(raw)


def test_preflight_fails_closed_on_runtime_and_cost_constraints():
    plan = build_assessment_plan(
        _config(),
        _budget(max_cost_usd=0.001),
        runtime=_runtime(gpu_count=0, version="different"),
    )
    assert not plan.execution_allowed
    failed = {check.name for check in plan.checks if check.status == "fail"}
    assert {"vllm_version_pinned", "gpu_count_available", "cost_ceiling"} <= failed


def test_dry_run_writes_only_the_plan_and_never_calls_runner(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("runner must not execute during dry-run")

    monkeypatch.setattr("inferpilot.assessment.run_experiment", forbidden)
    outcome = create_assessment(
        _config(), _budget(), output_dir=tmp_path, dry_run=True, runtime=_runtime()
    )
    assert outcome.result is None and outcome.run_dir is None
    assert (outcome.assessment_dir / "assessment-plan.json").exists()
    assert list(outcome.assessment_dir.iterdir()) == [
        outcome.assessment_dir / "assessment-plan.json"
    ]


def test_assessment_reuses_runner_persists_result_then_card(tmp_path):
    outcome = create_assessment(
        _config(),
        _budget(),
        output_dir=tmp_path,
        runtime=_runtime(),
        command_builder=_builder,
        ready_timeout_s=15,
    )
    assert outcome.result is not None and outcome.result.status.value == "completed"
    assert outcome.run_dir is not None
    assert (outcome.run_dir / "result.json").exists()
    assert outcome.card is not None and outcome.card.status == "abstain"
    assert (outcome.assessment_dir / "evidence-card.json").exists()
    assert outcome.candidate_plan is None


def test_candidate_plan_is_executable_but_never_authorized_automatically():
    result = make_result(saturating=True, kv_peak=1.0, preemptions=2)
    result = result.model_copy(update={"load_evidence": make_evidence(result)})
    card = build_evidence_card(
        result,
        SLO(ttft_p95_ms=500, tpot_p95_ms=60),
        OperatorEconomics(gpu_cost_per_hour_usd=2.10),
    )
    plan = build_candidate_plan(card)
    assert plan.candidate_config.engine.kv_cache_dtype == "fp8"
    assert plan.varied_engine_fields == ("kv_cache_dtype",)
    assert plan.execution_authorized is False
    assert plan.candidate_config.slo == card.slo


def test_assess_cli_dry_run_is_an_understandable_front_door(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "config.json"
    config_path.write_text(_config().model_dump_json(indent=2))
    monkeypatch.setattr(
        "inferpilot.assessment.capture_assessment_runtime", lambda _config: _runtime()
    )
    code = main(
        [
            "assess",
            str(config_path),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "assessments"),
            "--max-wall-time-s",
            "30",
            "--gpu-cost-per-hour",
            "1.20",
            "--max-cost-usd",
            "0.02",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "Execution allowed: yes" in output
    assert "no_server_launch" not in output
    assert "Dry run passed" in output
    assert len(list((tmp_path / "assessments").glob("*/assessment-plan.json"))) == 1
