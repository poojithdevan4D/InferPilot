import pytest
from pydantic import ValidationError

from inferpilot.advisor import (
    CanarySpec, ComparisonSpec, ControllerEvent, ControllerSpec, ControllerState,
    advance_controller, compare_configs,
)
from test_canary import _decision, _policy
from test_config_comparison import _result
from test_load_state import make_evidence

CURRENT = {"max_num_batched_tokens": 512, "max_num_seqs": 2}
CANDIDATE = {"max_num_batched_tokens": 512, "max_num_seqs": 4}


def _spec():
    return ControllerSpec(controller_version="0.3.0", advisor_policy=_policy(),
                          min_consecutive_windows=1, comparison_spec=ComparisonSpec())


def _to_test_candidate(spec):
    initial = ControllerState(current_engine_overrides=dict(CURRENT))
    t = advance_controller(spec, initial,
        ControllerEvent(event_version="0.3.0", event_type="observation", decision=_decision()))
    assert t.action == "test_candidate" and t.after.canary_candidate == CANDIDATE
    return t.after


def _comparison(candidate, *, with_evidence=True):
    incumbent = _result(ttft=178, tpot=42, seqs=2)
    return compare_configs(ComparisonSpec(), incumbent, candidate,
        incumbent_load_evidence=make_evidence(incumbent) if with_evidence else None,
        candidate_load_evidence=make_evidence(candidate) if with_evidence else None)


def test_dominant_candidate_is_applied():
    spec = _spec()
    state = _to_test_candidate(spec)
    comp = _comparison(_result(ttft=150, tpot=38, seqs=4))
    applied = advance_controller(spec, state,
        ControllerEvent(event_version="0.3.0", event_type="canary_result", config_comparison=comp))
    assert comp.verdict == "candidate_dominates" and applied.action == "apply_candidate"
    assert applied.after.current_engine_overrides == CANDIDATE


@pytest.mark.parametrize("with_evidence", [True, False])
def test_non_dominant_or_indeterminate_candidate_rolls_back(with_evidence):
    spec = _spec()
    state = _to_test_candidate(spec)
    comp = _comparison(_result(ttft=1976, tpot=38.2, seqs=4), with_evidence=with_evidence)
    expected = "inconclusive_tradeoff" if with_evidence else "inconclusive_evidence"
    assert comp.verdict == expected
    rolled = advance_controller(spec, state,
        ControllerEvent(event_version="0.3.0", event_type="canary_result", config_comparison=comp))
    assert rolled.action == "rollback" and rolled.after.current_engine_overrides == CURRENT


def test_comparison_candidate_must_match_outstanding_candidate():
    spec = _spec()
    state = _to_test_candidate(spec)
    comp = _comparison(_result(ttft=150, tpot=38, seqs=8))
    with pytest.raises(ValueError, match="does not match outstanding candidate"):
        advance_controller(spec, state,
            ControllerEvent(event_version="0.3.0", event_type="canary_result", config_comparison=comp))


def test_spec_requires_comparison_spec():
    with pytest.raises(ValidationError, match="0.3.0 requires comparison_spec"):
        ControllerSpec(controller_version="0.3.0", advisor_policy=_policy(), min_consecutive_windows=1)
    with pytest.raises(ValidationError, match="other versions forbid"):
        ControllerSpec(controller_version="0.2.0", advisor_policy=_policy(),
                       canary_spec=CanarySpec(ttft_p95_limit_ms=250, tpot_p95_limit_ms=8.5),
                       comparison_spec=ComparisonSpec())
