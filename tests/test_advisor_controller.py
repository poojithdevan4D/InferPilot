from copy import deepcopy

import pytest
from pydantic import ValidationError

from inferpilot import WorkloadObservation, build_workload_profile
from inferpilot.advisor import (
    AdvisorPolicy,
    ControllerEvent,
    ControllerReplay,
    ControllerSpec,
    ControllerState,
    ProfileContext,
    advance_controller,
    advise_from_profile,
    replay_controller,
)


def _policy(policy_id: str = "m3") -> AdvisorPolicy:
    return AdvisorPolicy(
        policy_version="0.2.0",
        policy_id=policy_id,
        model="q",
        revision="abc",
        gpu_name="g",
        prompt_tokens=128,
        output_tokens=32,
        arrival_pattern="poisson-v1",
        fixed_engine_fields={"max_num_batched_tokens": 512},
        regimes=[
            {"request_rate_qps": 2, "min_rate_qps": 1.5, "max_rate_qps": 2.5,
             "engine_overrides": {"max_num_seqs": 2}},
            {"request_rate_qps": 6, "min_rate_qps": 5.5, "max_rate_qps": 6.5,
             "engine_overrides": {"max_num_seqs": 4}},
        ],
        evidence_report_sha256="a" * 64,
        applicability_report_sha256="b" * 64,
    )


def _decision(rate: float, policy_id: str = "m3"):
    observations = [
        WorkloadObservation(
            arrival_offset_s=index / rate,
            prompt_tokens=128,
            output_tokens=32,
            success=True,
        )
        for index in range(5)
    ]
    return advise_from_profile(
        _policy(policy_id),
        build_workload_profile(observations),
        ProfileContext(
            model="q", revision="abc", gpu_name="g",
            arrival_pattern="poisson-v1", declared_fixed_length=True,
        ),
    )


def _event(rate: float, policy_id: str = "m3") -> ControllerEvent:
    return ControllerEvent(event_version="0.1.0", event_type="observation", decision=_decision(rate, policy_id))


def _initial() -> ControllerState:
    return ControllerState(
        current_engine_overrides={"max_num_batched_tokens": 512, "max_num_seqs": 2}
    )


def test_three_stable_windows_require_canary_before_apply() -> None:
    spec = ControllerSpec(controller_version="0.1.0", advisor_policy=_policy())
    replay = replay_controller(spec, _initial(), [_event(6), _event(6), _event(6)])
    assert [step.action for step in replay.transitions] == [
        "keep_current", "keep_current", "test_candidate"
    ]
    assert replay.final_state.current_engine_overrides["max_num_seqs"] == 2
    assert replay.final_state.canary_candidate["max_num_seqs"] == 4
    applied = advance_controller(
        spec, replay.final_state,
        ControllerEvent(event_version="0.1.0", event_type="canary_result", canary_passed=True),
    )
    assert applied.action == "apply_candidate"
    assert applied.after.current_engine_overrides["max_num_seqs"] == 4
    assert applied.after.cooldown_windows_remaining == 2


def test_alternating_recommendations_never_trigger_test() -> None:
    spec = ControllerSpec(controller_version="0.1.0", advisor_policy=_policy(), min_consecutive_windows=3)
    replay = replay_controller(
        spec, _initial(), [_event(6), _event(2), _event(6), _event(2), _event(6)]
    )
    assert all(step.action == "keep_current" for step in replay.transitions)
    assert replay.final_state.canary_candidate is None


def test_abstention_resets_pending_stability() -> None:
    spec = ControllerSpec(controller_version="0.1.0", advisor_policy=_policy(), min_consecutive_windows=2)
    replay = replay_controller(spec, _initial(), [_event(6), _event(3), _event(6)])
    assert [step.action for step in replay.transitions] == [
        "keep_current", "keep_current", "keep_current"
    ]
    assert replay.transitions[1].reasons[0] == "advisor_abstained"
    assert replay.final_state.consecutive_candidate_windows == 1


def test_failed_canary_rolls_back_and_enters_cooldown() -> None:
    spec = ControllerSpec(controller_version="0.1.0", advisor_policy=_policy(), min_consecutive_windows=1)
    test = advance_controller(spec, _initial(), _event(6))
    rolled_back = advance_controller(
        spec, test.after,
        ControllerEvent(event_version="0.1.0", event_type="canary_result", canary_passed=False),
    )
    assert rolled_back.action == "rollback"
    assert rolled_back.after.current_engine_overrides == _initial().current_engine_overrides
    assert rolled_back.after.cooldown_windows_remaining == 2
    first = advance_controller(spec, rolled_back.after, _event(6))
    second = advance_controller(spec, first.after, _event(6))
    assert first.reasons == ["cooldown_active"]
    assert second.reasons == ["cooldown_active"]
    assert second.after.cooldown_windows_remaining == 0


def test_observations_do_not_advance_an_outstanding_canary() -> None:
    spec = ControllerSpec(controller_version="0.1.0", advisor_policy=_policy(), min_consecutive_windows=1)
    test = advance_controller(spec, _initial(), _event(6))
    waiting = advance_controller(spec, test.after, _event(2))
    assert waiting.action == "keep_current"
    assert waiting.reasons == ["awaiting_canary_result"]
    assert waiting.after == test.after


def test_wrong_policy_and_unsolicited_canary_fail_loudly() -> None:
    spec = ControllerSpec(controller_version="0.1.0", advisor_policy=_policy())
    with pytest.raises(ValueError, match="policy"):
        advance_controller(spec, _initial(), _event(6, policy_id="other"))
    with pytest.raises(ValueError, match="without an outstanding"):
        advance_controller(
            spec, _initial(),
            ControllerEvent(event_version="0.1.0", event_type="canary_result", canary_passed=True),
        )


def test_replay_roundtrip_and_tampering_rejected() -> None:
    replay = replay_controller(
        ControllerSpec(controller_version="0.1.0", advisor_policy=_policy()),
        _initial(),
        [_event(6), _event(6), _event(6)],
    )
    assert ControllerReplay.model_validate_json(replay.model_dump_json()) == replay
    raw = replay.model_dump(mode="json")
    tampered = deepcopy(raw)
    tampered["transitions"][0]["reasons"] = ["invented"]
    with pytest.raises(ValidationError, match="inconsistent"):
        ControllerReplay.model_validate(tampered)


@pytest.mark.parametrize(
    "event",
    [
        {"event_version": "0.1.0", "event_type": "observation", "decision": None, "canary_passed": None},
        {"event_version": "0.1.0", "event_type": "canary_result", "decision": None, "canary_passed": None},
    ],
)
def test_incomplete_events_are_rejected(event) -> None:
    with pytest.raises(ValidationError):
        ControllerEvent.model_validate(event)
