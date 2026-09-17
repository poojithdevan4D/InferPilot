"""Controller 0.3.0: apply/rollback gated by the Pareto config comparison.

The online loop now switches off the current config only when a candidate is MEASURED
to dominate it (compare_configs), not merely to pass an absolute SLO. This is the
2026-09-18 learning wired into the live decision.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot.advisor import (
    CanarySpec,
    ComparisonSpec,
    ControllerEvent,
    ControllerSpec,
    ControllerState,
    advance_controller,
    compare_configs,
)

from test_canary import _decision, _policy
from test_config_comparison import _result

CURRENT = {"max_num_batched_tokens": 512, "max_num_seqs": 2}
CANDIDATE = {"max_num_batched_tokens": 512, "max_num_seqs": 4}


def _spec() -> ControllerSpec:
    return ControllerSpec(
        controller_version="0.3.0", advisor_policy=_policy(),
        min_consecutive_windows=1, comparison_spec=ComparisonSpec(),
    )


def _to_test_candidate(spec):
    initial = ControllerState(current_engine_overrides=dict(CURRENT))
    t = advance_controller(
        spec, initial,
        ControllerEvent(event_version="0.3.0", event_type="observation", decision=_decision()),
    )
    assert t.action == "test_candidate" and t.after.canary_candidate == CANDIDATE
    return t.after


def _comparison(candidate_result):
    incumbent = _result(ttft=178, tpot=42, throughput=5.9, seqs=2)
    return compare_configs(ComparisonSpec(), incumbent, candidate_result)


def test_dominant_candidate_is_applied() -> None:
    spec = _spec()
    after_test = _to_test_candidate(spec)
    comp = _comparison(_result(ttft=150, tpot=38, throughput=5.9, seqs=4))  # dominates
    assert comp.verdict == "candidate_dominates"
    applied = advance_controller(
        spec, after_test,
        ControllerEvent(event_version="0.3.0", event_type="canary_result", config_comparison=comp),
    )
    assert applied.action == "apply_candidate"
    assert applied.after.current_engine_overrides == CANDIDATE


def test_non_dominant_candidate_rolls_back() -> None:
    spec = _spec()
    after_test = _to_test_candidate(spec)
    comp = _comparison(_result(ttft=1976, tpot=38.2, throughput=5.9, seqs=4))  # ttft catastrophe
    assert comp.verdict == "inconclusive_tradeoff"
    rolled = advance_controller(
        spec, after_test,
        ControllerEvent(event_version="0.3.0", event_type="canary_result", config_comparison=comp),
    )
    assert rolled.action == "rollback"
    assert rolled.after.current_engine_overrides == CURRENT


def test_comparison_candidate_must_match_outstanding_candidate() -> None:
    spec = _spec()
    after_test = _to_test_candidate(spec)
    # candidate result built with seqs=8, not the advised seqs=4
    comp = _comparison(_result(ttft=150, tpot=38, throughput=5.9, seqs=8))
    with pytest.raises(ValueError, match="does not match outstanding candidate"):
        advance_controller(
            spec, after_test,
            ControllerEvent(event_version="0.3.0", event_type="canary_result", config_comparison=comp),
        )


def test_spec_requires_comparison_spec() -> None:
    with pytest.raises(ValidationError, match="0.3.0 requires comparison_spec"):
        ControllerSpec(controller_version="0.3.0", advisor_policy=_policy(), min_consecutive_windows=1)
    with pytest.raises(ValidationError, match="other versions forbid"):
        ControllerSpec(controller_version="0.2.0", advisor_policy=_policy(),
                       canary_spec=CanarySpec(ttft_p95_limit_ms=250, tpot_p95_limit_ms=8.5),
                       comparison_spec=ComparisonSpec())
