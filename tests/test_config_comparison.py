import pytest
from pydantic import ValidationError

from inferpilot.advisor import ComparisonSpec, ConfigComparison, compare_configs
from inferpilot.runner.aggregate import compute_aggregates
from test_load_state import make_evidence, make_result


def _result(*, ttft, tpot, rate=6.0, count=600, seqs=4, saturating=False):
    return make_result(ttft=ttft, tpot=tpot, rate=rate, count=count, seqs=seqs, saturating=saturating)


def _compare(incumbent, candidate, **overrides):
    kwargs = dict(incumbent_load_evidence=make_evidence(incumbent),
                  candidate_load_evidence=make_evidence(candidate))
    kwargs.update(overrides)
    return compare_configs(ComparisonSpec(), incumbent, candidate, **kwargs)


def test_no_window_evidence_cannot_promote_candidate():
    i, c = _result(ttft=178, tpot=42), _result(ttft=150, tpot=38)
    comparison = compare_configs(ComparisonSpec(), i, c)
    assert comparison.verdict == "inconclusive_evidence" and not comparison.should_switch


def test_true_pareto_win_requires_healthy_measured_windows():
    i, c = _result(ttft=178, tpot=42), _result(ttft=150, tpot=38)
    comparison = _compare(i, c)
    assert comparison.verdict == "candidate_dominates" and comparison.should_switch


def test_illusory_tpot_win_is_rejected():
    comparison = _compare(_result(ttft=178, tpot=42), _result(ttft=1976, tpot=38.2))
    assert comparison.verdict == "inconclusive_tradeoff" and not comparison.should_switch
    assert any("ttft_p95_ms_worse" in reason for reason in comparison.reasons)
    assert any("tpot_p95_ms_better" in reason for reason in comparison.reasons)


def test_overloaded_candidate_with_flat_ttft_never_wins():
    comparison = _compare(_result(ttft=178, tpot=42), _result(ttft=30, tpot=32, saturating=True))
    assert comparison.verdict == "candidate_infeasible" and not comparison.should_switch


def test_near_capacity_and_unknown_incumbent_do_not_allow_promotion():
    i, c = _result(ttft=178, tpot=42), _result(ttft=150, tpot=38)
    assert _compare(i, c, incumbent_load_evidence=None).verdict == "inconclusive_evidence"
    e = make_evidence(c, waiting_requests=[1] * 5)
    assert _compare(i, c, candidate_load_evidence=e).verdict == "inconclusive_evidence"


def test_tie_keeps_incumbent():
    comparison = _compare(_result(ttft=178, tpot=42), _result(ttft=179, tpot=42.3))
    assert comparison.verdict == "incumbent_kept" and not comparison.should_switch


def test_healthy_candidate_can_dominate_overloaded_incumbent_but_not_by_feasibility_alone():
    i, c = _result(ttft=214000, tpot=32, saturating=True), _result(ttft=180, tpot=40)
    comparison = _compare(i, c)
    assert comparison.verdict == "candidate_dominates" and "incumbent_overloaded" in comparison.reasons
    # A candidate with worse TTFT cannot bypass the Pareto guard just because it is healthy.
    comparison = _compare(_result(ttft=10, tpot=32, saturating=True), c)
    assert comparison.verdict == "inconclusive_tradeoff" and not comparison.should_switch


def test_roundtrip_tampering_context_and_evidence_binding():
    i, c = _result(ttft=178, tpot=42), _result(ttft=1976, tpot=38.2)
    comparison = _compare(i, c)
    assert ConfigComparison.model_validate_json(comparison.model_dump_json()) == comparison
    raw = comparison.model_dump(mode="json")
    raw.update(verdict="candidate_dominates", should_switch=True)
    with pytest.raises(ValidationError, match="inconsistent"):
        ConfigComparison.model_validate(raw)
    with pytest.raises(ValueError, match="same model/hardware/workload"):
        _compare(i, _result(ttft=150, tpot=38, rate=2))
    with pytest.raises(ValueError, match="does not match measurements"):
        _compare(i, c, candidate_load_evidence=make_evidence(i))


def test_old_spec_knobs_cannot_bypass_missing_load_evidence():
    i, c = _result(ttft=178, tpot=42), _result(ttft=150, tpot=38)
    spec = ComparisonSpec(keepup_fraction=0.01, saturation_threshold=100, min_saturation_samples=2)
    assert not compare_configs(spec, i, c).should_switch


def test_missing_or_different_intended_trace_is_not_comparable():
    i, c = _result(ttft=178, tpot=42), _result(ttft=150, tpot=38)
    assert _compare(i, c, candidate_load_evidence=make_evidence(c, replay_sha256=None)).verdict == "inconclusive_evidence"
    with pytest.raises(ValueError, match="same intended replay"):
        _compare(i, c, candidate_load_evidence=make_evidence(c, replay_sha256="0" * 64))


def test_actual_admission_jitter_does_not_change_intended_trace_identity():
    i, c = _result(ttft=178, tpot=42), _result(ttft=150, tpot=38)
    rows = [m.model_copy(update={"start_time_s": m.start_time_s + 0.001,
                                "end_time_s": m.end_time_s + 0.001}) for m in c.measurements]
    agg = compute_aggregates(rows, max(m.end_time_s for m in rows)).model_copy(update={"gpu_memory_peak_mb": 1024})
    c = c.model_copy(update={"measurements": rows, "aggregates": agg})
    assert _compare(i, c).should_switch
