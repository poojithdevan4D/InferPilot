import pytest
from pydantic import ValidationError

from inferpilot.diagnosis import BottleneckDiagnosis, _classify, diagnose
from test_load_state import make_evidence, make_result


def test_missing_window_is_unknown_not_compute_bound():
    d = diagnose(make_result(gpu_mean=100, kv_peak=1, preemptions=2))
    assert d.load_state == "indeterminate" and d.regime == "unknown"
    assert not d.saturated and d.recommended_lever == "none"
    assert d.preemptions is None  # whole-run counter is not an aligned window delta


def test_missing_resource_telemetry_does_not_crash():
    r = make_result().model_copy(update={"telemetry": None})
    d = diagnose(r)
    assert d.regime == "unknown" and d.gpu_utilization_mean_pct is None


def test_gpu_pinned_zero_or_missing_preemptions_does_not_prove_compute_bound():
    for preemptions in (0, None):
        r = make_result(saturating=True, gpu_mean=100, kv_peak=1, preemptions=preemptions)
        d = diagnose(r, load_evidence=make_evidence(r))
        assert d.saturated and d.regime == "unknown" and d.recommended_lever == "none"


def test_preempting_kv_pressure_nominates_canary_not_a_predicted_gain():
    r = make_result(saturating=True, gpu_mean=100, kv_peak=1, preemptions=2)
    d = diagnose(r, load_evidence=make_evidence(r))
    assert d.regime == "kv_pressure" and d.recommended_lever == "kv_cache_dtype=fp8"
    assert "representative_quality_gate_passed" in d.lever_preconditions
    assert "do not measure recompute waste" in d.predicted_effect


def test_healthy_does_not_imply_underutilized_or_spec_decode():
    r = make_result(gpu_mean=40, kv_peak=0.2)
    d = diagnose(r, load_evidence=make_evidence(r))
    assert d.regime == "no_load_pressure" and d.recommended_lever == "none"


def test_legacy_boolean_arguments_do_not_certify_load_state():
    assert _classify(99, 1, True, True)[0] == "unknown"
    assert _classify(40, 0.2, False, True)[0] == "unknown"


def test_diagnosis_roundtrip_tamper_and_binding():
    r = make_result()
    d = diagnose(r, load_evidence=make_evidence(r))
    assert BottleneckDiagnosis.model_validate_json(d.model_dump_json()) == d
    for key, value in (("regime", "compute_bound"), ("saturated", True), ("preemptions", 99)):
        raw = d.model_dump(mode="json")
        raw[key] = value
        with pytest.raises(ValidationError, match="inconsistent with its signals"):
            BottleneckDiagnosis.model_validate(raw)
    with pytest.raises(ValueError, match="experiment ID"):
        diagnose(r, load_evidence=make_evidence(r, experiment_id="another-run"))
