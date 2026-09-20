from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import SLO
from inferpilot.configuration_gate import (
    BoundQualityEvidence,
    ConfigurationGateReport,
    ConfigurationGateSpec,
    evaluate_configuration_gate,
)
from inferpilot.comparison.fingerprint import exact_fingerprint
from inferpilot.quality import (
    KLQualitySpec,
    NeedleQualitySpec,
    evaluate_kl_quality,
    evaluate_needle_quality,
)
from inferpilot.cli import main
from test_load_state import make_evidence, make_result

SLO_ = SLO(ttft_p95_ms=500, tpot_p95_ms=60)
SHA = "a" * 64


def _with_identity(result, experiment_id):
    config = result.config.model_copy(
        update={"experiment_id": experiment_id, "slo": SLO_}
    )
    result = result.model_copy(update={"config": config})
    return result.model_copy(update={"load_evidence": make_evidence(result)})


def _pair(*, candidate_ttft=150, candidate_tpot=38):
    baseline = _with_identity(
        make_result(ttft=178, tpot=42, seqs=1), "gate-baseline"
    )
    candidate = _with_identity(
        make_result(ttft=candidate_ttft, tpot=candidate_tpot, seqs=4),
        "gate-candidate",
    )
    return baseline, candidate


def _quality(baseline=None, candidate=None, *, mean_kl=0.006, p99_kl=0.05):
    if baseline is None or candidate is None:
        baseline, candidate = _pair()
    gate = evaluate_kl_quality(
        KLQualitySpec(),
        tokens_compared=8000,
        mean_kl=mean_kl,
        p99_kl=p99_kl,
        noise_floor_kl=0.002,
    )
    return BoundQualityEvidence(
        baseline_experiment_id="gate-baseline",
        candidate_experiment_id="gate-candidate",
        baseline_fingerprint=exact_fingerprint(baseline),
        candidate_fingerprint=exact_fingerprint(candidate),
        corpus_id="operator-eval-v1",
        corpus_sha256=SHA,
        kind="teacher_forced_kl",
        gate=gate,
    )


def _spec(**updates):
    values = {
        "varied_engine_fields": ("max_num_seqs",),
        "slo": SLO_,
        "required_quality_gates": ("teacher_forced_kl",),
    }
    values.update(updates)
    return ConfigurationGateSpec(**values)


def _needle_quality(baseline=None, candidate=None):
    if baseline is None or candidate is None:
        baseline, candidate = _pair()
    return BoundQualityEvidence(
        baseline_experiment_id="gate-baseline",
        candidate_experiment_id="gate-candidate",
        baseline_fingerprint=exact_fingerprint(baseline),
        candidate_fingerprint=exact_fingerprint(candidate),
        corpus_id="long-context-retrieval-v1",
        corpus_sha256="b" * 64,
        kind="needle_retrieval",
        gate=evaluate_needle_quality(
            NeedleQualitySpec(),
            num_probes=20,
            candidate_accuracy=0.92,
            baseline_accuracy=0.93,
        ),
    )


def test_candidate_pass_requires_performance_slo_and_bound_quality():
    baseline, candidate = _pair()
    report = evaluate_configuration_gate(
        _spec(), baseline, candidate, (_quality(),)
    )
    assert report.verdict == "pass"
    assert report.comparison is not None
    assert report.comparison.verdict == "candidate_dominates"
    assert all(check.passed for check in report.candidate_slo_checks)
    assert report.deployment_authorized is False
    assert ConfigurationGateReport.model_validate_json(report.model_dump_json()) == report


def test_missing_quality_and_load_evidence_abstain():
    baseline, candidate = _pair()
    missing_quality = evaluate_configuration_gate(_spec(), baseline, candidate)
    assert missing_quality.verdict == "abstain"
    assert missing_quality.reasons == (
        "quality_evidence_missing:teacher_forced_kl",
    )

    candidate = candidate.model_copy(update={"load_evidence": None})
    missing_load = evaluate_configuration_gate(
        _spec(), baseline, candidate, (_quality(),)
    )
    assert missing_load.verdict == "abstain"
    assert "performance:inconclusive_evidence" in missing_load.reasons


def test_measured_quality_or_performance_regression_fails():
    baseline, candidate = _pair()
    quality_failure = evaluate_configuration_gate(
        _spec(), baseline, candidate, (_quality(mean_kl=0.2),)
    )
    assert quality_failure.verdict == "fail"
    assert quality_failure.reasons == (
        "quality_gate_failed:teacher_forced_kl",
    )

    _, slow_candidate = _pair(candidate_ttft=450, candidate_tpot=80)
    performance_failure = evaluate_configuration_gate(
        _spec(), baseline, slow_candidate, (_quality(),)
    )
    assert performance_failure.verdict == "fail"
    assert "candidate_slo_failed" in performance_failure.reasons
    assert "performance:inconclusive_tradeoff" in performance_failure.reasons


def test_insufficient_quality_evidence_abstains_instead_of_claiming_regression():
    baseline, candidate = _pair()
    gate = evaluate_kl_quality(
        KLQualitySpec(),
        tokens_compared=20,
        mean_kl=0.2,
        p99_kl=1.0,
        noise_floor_kl=0.002,
    )
    evidence = _quality().model_copy(update={"gate": gate})
    report = evaluate_configuration_gate(_spec(), baseline, candidate, (evidence,))
    assert report.verdict == "abstain"
    assert report.reasons == (
        "quality_evidence_insufficient:teacher_forced_kl",
    )


def test_hidden_config_change_and_quality_binding_mismatch_are_rejected():
    baseline, candidate = _pair()
    engine = candidate.config.engine.model_copy(update={"max_num_batched_tokens": 1024})
    config = candidate.config.model_copy(update={"engine": engine})
    hidden_change = candidate.model_copy(update={"config": config})
    with pytest.raises(ValueError, match="outside the declared engine fields"):
        evaluate_configuration_gate(
            _spec(), baseline, hidden_change, (_quality(),)
        )

    wrong_binding = _quality().model_copy(
        update={"candidate_experiment_id": "some-other-candidate"}
    )
    with pytest.raises(ValueError, match="candidate experiment ID mismatch"):
        evaluate_configuration_gate(
            _spec(), baseline, candidate, (wrong_binding,)
        )

    wrong_fingerprint = _quality().model_copy(
        update={"candidate_fingerprint": "0" * 64}
    )
    with pytest.raises(ValueError, match="candidate fingerprint mismatch"):
        evaluate_configuration_gate(
            _spec(), baseline, candidate, (wrong_fingerprint,)
        )


def test_precision_change_requires_kl_and_long_context_retrieval():
    with pytest.raises(ValidationError, match="require both"):
        _spec(
            varied_engine_fields=("kv_cache_dtype",),
            required_quality_gates=("teacher_forced_kl",),
        )

    baseline, candidate = _pair()
    baseline_engine = baseline.config.engine.model_copy(
        update={"max_num_seqs": 1, "kv_cache_dtype": "auto"}
    )
    candidate_engine = candidate.config.engine.model_copy(
        update={"max_num_seqs": 1, "kv_cache_dtype": "fp8"}
    )
    baseline = baseline.model_copy(
        update={
            "config": baseline.config.model_copy(update={"engine": baseline_engine}),
            "effective_config": baseline.effective_config.model_copy(
                update={"max_num_seqs": 1, "kv_cache_dtype": "auto"}
            ),
        }
    )
    candidate = candidate.model_copy(
        update={
            "config": candidate.config.model_copy(update={"engine": candidate_engine}),
            "effective_config": candidate.effective_config.model_copy(
                update={"max_num_seqs": 1, "kv_cache_dtype": "fp8"}
            ),
        }
    )
    spec = _spec(
        varied_engine_fields=("kv_cache_dtype",),
        required_quality_gates=("teacher_forced_kl", "needle_retrieval"),
    )
    report = evaluate_configuration_gate(
        spec,
        baseline,
        candidate,
        (_quality(baseline, candidate), _needle_quality(baseline, candidate)),
    )
    assert report.verdict == "pass"
    assert (
        BoundQualityEvidence.model_validate_json(
            _needle_quality(baseline, candidate).model_dump_json()
        )
        == _needle_quality(baseline, candidate)
    )


def test_derived_report_tampering_is_rejected():
    baseline, candidate = _pair()
    report = evaluate_configuration_gate(
        _spec(), baseline, candidate, (_quality(),)
    )
    raw = report.model_dump(mode="json")
    raw["verdict"] = "fail"
    with pytest.raises(ValidationError, match="contradicts its evidence"):
        ConfigurationGateReport.model_validate(raw)


def test_gate_cli_writes_report_and_uses_verdict_as_exit_status(tmp_path, capsys):
    baseline, candidate = _pair()
    paths = {
        "spec": tmp_path / "spec.json",
        "baseline": tmp_path / "baseline.json",
        "candidate": tmp_path / "candidate.json",
        "quality": tmp_path / "quality.json",
        "report": tmp_path / "report.json",
    }
    paths["spec"].write_text(_spec().model_dump_json())
    paths["baseline"].write_text(baseline.model_dump_json())
    paths["candidate"].write_text(candidate.model_dump_json())
    paths["quality"].write_text(_quality().model_dump_json())

    code = main(
        [
            "gate",
            str(paths["spec"]),
            str(paths["baseline"]),
            str(paths["candidate"]),
            "--quality-evidence",
            str(paths["quality"]),
            "--output",
            str(paths["report"]),
        ]
    )
    assert code == 0
    assert "Configuration gate: PASS" in capsys.readouterr().out
    report = ConfigurationGateReport.model_validate_json(paths["report"].read_text())
    assert report.verdict == "pass"
