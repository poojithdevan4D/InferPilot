"""M10 AdvisorSession: end-to-end self-validating closed-loop replay."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import WorkloadObservation, build_workload_profile
from inferpilot.advisor import (
    AdvisorSession,
    ControllerSpec,
    ControllerState,
    ProfileContext,
    SessionStep,
    evaluate_canary,
    run_session,
)

from test_canary import _policy, _result, _spec


def _ctx() -> ProfileContext:
    return ProfileContext(
        model="q", revision="abc", gpu_name="g",
        arrival_pattern="poisson-v1", declared_fixed_length=True,
    )


def _profile(rate: float):
    return build_workload_profile([
        WorkloadObservation(
            arrival_offset_s=i / rate, prompt_tokens=128, output_tokens=32, success=True
        )
        for i in range(5)
    ])


def _obs(rate: float) -> SessionStep:
    return SessionStep(kind="observation", profile=_profile(rate), context=_ctx())


def _canary_step(result) -> SessionStep:
    # decision the controller will hold as candidate == rate-6 recommendation (seqs=4)
    from inferpilot.advisor import advise_from_profile
    decision = advise_from_profile(_policy(), _profile(6), _ctx())
    evaluation = evaluate_canary(_spec(), decision, result)
    return SessionStep(kind="canary_result", canary_evaluation=evaluation)


def _spec_v2() -> ControllerSpec:
    return ControllerSpec(
        controller_version="0.2.0", advisor_policy=_policy(),
        min_consecutive_windows=3, canary_spec=_spec(),
    )


def _initial() -> ControllerState:
    return ControllerState(
        current_engine_overrides={"max_num_batched_tokens": 512, "max_num_seqs": 2}
    )


def test_full_loop_applies_candidate_after_passing_canary() -> None:
    # 3 stable rate-6 observations -> test_candidate, then a passing measured canary -> apply.
    steps = [_obs(6), _obs(6), _obs(6), _canary_step(_result())]
    session = run_session(_spec_v2(), _initial(), steps)
    actions = [t.action for t in session.replay.transitions]
    assert actions == [
        "keep_current", "keep_current", "test_candidate", "apply_candidate",
    ]
    assert session.final_state.current_engine_overrides == {
        "max_num_batched_tokens": 512, "max_num_seqs": 4
    }


def test_failing_canary_rolls_back() -> None:
    steps = [_obs(6), _obs(6), _obs(6), _canary_step(_result(tpot=9))]
    session = run_session(_spec_v2(), _initial(), steps)
    assert session.replay.transitions[-1].action == "rollback"
    assert session.final_state.current_engine_overrides == _initial().current_engine_overrides


def test_noisy_sequence_never_tests_candidate() -> None:
    # Alternating rate-6 (candidate seqs=4) and rate-2 (matches current) resets the
    # stability counter, so the threshold is never met and nothing is applied.
    steps = [_obs(6), _obs(2), _obs(6), _obs(2), _obs(6)]
    session = run_session(_spec_v2(), _initial(), steps)
    actions = {t.action for t in session.replay.transitions}
    assert actions == {"keep_current"}
    assert session.final_state.current_engine_overrides == _initial().current_engine_overrides


def test_session_roundtrips() -> None:
    steps = [_obs(6), _obs(6), _obs(6), _canary_step(_result())]
    session = run_session(_spec_v2(), _initial(), steps)
    assert AdvisorSession.model_validate_json(session.model_dump_json()) == session


def test_tampered_final_state_is_rejected() -> None:
    session = run_session(_spec_v2(), _initial(), [_obs(6), _obs(6), _obs(6), _canary_step(_result())])
    raw = session.model_dump(mode="json")
    raw["final_state"]["current_engine_overrides"]["max_num_seqs"] = 2
    with pytest.raises(ValidationError, match="inconsistent"):
        AdvisorSession.model_validate(raw)


def test_tampered_step_desyncs_replay() -> None:
    # Swap a passing canary for a failing measurement without touching the replay -> rejected.
    session = run_session(_spec_v2(), _initial(), [_obs(6), _obs(6), _obs(6), _canary_step(_result())])
    raw = session.model_dump(mode="json")
    bad = _canary_step(_result(tpot=9))
    raw["steps"][-1] = bad.model_dump(mode="json")
    with pytest.raises(ValidationError, match="inconsistent"):
        AdvisorSession.model_validate(raw)


def test_step_shape_is_enforced() -> None:
    with pytest.raises(ValidationError, match="observation step requires"):
        SessionStep(kind="observation", profile=_profile(6))  # missing context
    with pytest.raises(ValidationError, match="exactly one"):
        SessionStep(kind="canary_result")  # neither evaluation nor passed
