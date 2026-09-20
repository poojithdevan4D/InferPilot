import json

import pytest
from pydantic import ValidationError

from inferpilot import OptimizationEvidenceCard, SLO, build_evidence_card
from inferpilot.advisor import OperatorEconomics
from inferpilot.cli import main
from test_load_state import make_evidence, make_result


SLO_ = SLO(ttft_p95_ms=500, tpot_p95_ms=60)
ECON = OperatorEconomics(gpu_cost_per_hour_usd=2.10, gpu_count=1, target_qps=2)


def _with_evidence(result, **overrides):
    return result.model_copy(update={"load_evidence": make_evidence(result, **overrides)})


def test_kv_pressure_produces_one_bounded_experiment_not_a_deployment_change():
    result = _with_evidence(
        make_result(saturating=True, kv_peak=1.0, preemptions=2)
    )
    card = build_evidence_card(result, SLO_, ECON, total_occupancy_s=120)

    assert card.status == "experiment_recommended"
    assert card.candidate_engine_overrides == {"kv_cache_dtype": "fp8"}
    assert card.transferability == "test_before_use"
    assert card.evidence_strength == "aligned"
    assert card.estimated_paired_experiment_gpu_seconds == 240
    assert card.estimated_paired_experiment_cost_usd == pytest.approx(0.14)
    assert "representative_task_quality_gate_passes" in card.acceptance_criteria
    assert "No deployment is changed" in card.recommendation.action_log


def test_healthy_window_keeps_current_config_without_claiming_headroom():
    result = _with_evidence(make_result())
    card = build_evidence_card(result, SLO_, ECON)

    assert card.status == "keep_current"
    assert card.candidate_engine_overrides is None
    assert card.transferability == "observed_window_only"
    assert card.estimated_paired_experiment_cost_usd is None
    assert "headroom and SLO compliance are separate" in card.recommendation.action_log


def test_missing_aligned_evidence_abstains():
    card = build_evidence_card(make_result(), SLO_, ECON)

    assert card.status == "abstain"
    assert card.evidence_strength == "insufficient"
    assert card.transferability == "unsupported"
    assert card.candidate_engine_overrides is None


def test_card_roundtrip_and_derived_tampering_are_rejected():
    card = build_evidence_card(_with_evidence(make_result()), SLO_, ECON)
    assert OptimizationEvidenceCard.model_validate_json(card.model_dump_json()) == card

    raw = card.model_dump(mode="json")
    raw["status"] = "experiment_recommended"
    with pytest.raises(ValidationError, match="field 'status' is inconsistent"):
        OptimizationEvidenceCard.model_validate(raw)


def test_cli_writes_machine_readable_card_and_refuses_overwrite(tmp_path, capsys):
    bundle = tmp_path / "run"
    bundle.mkdir()
    result = _with_evidence(make_result())
    (bundle / "result.json").write_text(result.model_dump_json(indent=2))
    output = tmp_path / "card.json"

    args = [
        "analyze", str(bundle), "--ttft-p95-ms", "500",
        "--tpot-p95-ms", "60", "--gpu-cost-per-hour", "2.10",
        "--target-qps", "2", "--output", str(output),
    ]
    assert main(args) == 0
    saved = OptimizationEvidenceCard.model_validate_json(output.read_text())
    assert saved.status == "keep_current" and saved.metadata_only is True
    assert "Status: keep_current" in capsys.readouterr().out

    assert main(args) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cli_requires_an_explicit_operator_slo(tmp_path, capsys):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(make_result().model_dump(mode="json")))
    assert main(["analyze", str(path), "--gpu-cost-per-hour", "2.10"]) == 2
    assert "at least one SLO" in capsys.readouterr().err
