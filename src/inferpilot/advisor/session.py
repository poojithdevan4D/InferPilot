"""End-to-end autonomous advisor session (M10).

Ties the existing pieces into one self-validating artifact: a sequence of passive
workload windows (profiles) and canary results is turned into advisor decisions,
controller events, and a full controller replay. The session recomputes the
entire chain on load — profile → advise_from_profile → controller event →
transition → final configuration — so tampering anywhere is rejected.

This is an OFFLINE, replayable demonstrator of the closed loop
(observe → decide → test → canary → apply/rollback). It performs no live server
mutation; applying a candidate still requires a passing measured canary.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import model_validator

from .._base import SchemaModel
from ..workload_profile import WorkloadProfile
from .canary import CanaryEvaluation
from .config_comparison import ConfigComparison
from .controller import (
    ControllerEvent,
    ControllerReplay,
    ControllerSpec,
    ControllerState,
    replay_controller,
)
from .profile_adapter import ProfileContext, advise_from_profile

SESSION_VERSION = "0.1.0"


class SessionStep(SchemaModel):
    """One session input: a passive observation window, or a canary result."""

    kind: Literal["observation", "canary_result"]
    # observation
    profile: Optional[WorkloadProfile] = None
    context: Optional[ProfileContext] = None
    # canary_result — exactly one of these, matching the controller version
    canary_evaluation: Optional[CanaryEvaluation] = None    # controller 0.2.0
    canary_passed: Optional[bool] = None                    # controller 0.1.0
    config_comparison: Optional[ConfigComparison] = None    # controller 0.3.0

    @model_validator(mode="after")
    def _check(self) -> "SessionStep":
        evidence = (self.canary_evaluation, self.canary_passed, self.config_comparison)
        if self.kind == "observation":
            if self.profile is None or self.context is None or any(x is not None for x in evidence):
                raise ValueError("an observation step requires only profile + context")
        else:
            if self.profile is not None or self.context is not None:
                raise ValueError("a canary_result step must not carry profile/context")
            if sum(x is not None for x in evidence) != 1:
                raise ValueError("a canary_result step needs exactly one of evaluation/passed/comparison")
        return self


def _event(spec: ControllerSpec, step: SessionStep) -> ControllerEvent:
    if step.kind == "observation":
        decision = advise_from_profile(spec.advisor_policy, step.profile, step.context)
        return ControllerEvent(
            event_version=spec.controller_version, event_type="observation", decision=decision
        )
    if spec.controller_version == "0.3.0":
        return ControllerEvent(
            event_version="0.3.0", event_type="canary_result",
            config_comparison=step.config_comparison,
        )
    if spec.controller_version == "0.2.0":
        return ControllerEvent(
            event_version="0.2.0", event_type="canary_result",
            canary_evaluation=step.canary_evaluation,
        )
    return ControllerEvent(
        event_version="0.1.0", event_type="canary_result", canary_passed=step.canary_passed,
    )


class AdvisorSession(SchemaModel):
    """Self-validating end-to-end advisor session over ordered steps."""

    session_version: Literal["0.1.0"] = "0.1.0"
    spec: ControllerSpec
    initial_state: ControllerState
    steps: list[SessionStep]
    replay: ControllerReplay
    final_state: ControllerState

    @model_validator(mode="after")
    def _check(self) -> "AdvisorSession":
        events = [_event(self.spec, step) for step in self.steps]
        expected = replay_controller(self.spec, self.initial_state, events)
        if self.replay != expected or self.final_state != expected.final_state:
            raise ValueError("advisor session is inconsistent with its steps")
        return self


def run_session(
    spec: ControllerSpec, initial_state: ControllerState, steps: list[SessionStep]
) -> AdvisorSession:
    """Derive advisor decisions + controller replay from ordered session steps."""
    events = [_event(spec, step) for step in steps]
    replay = replay_controller(spec, initial_state, events)
    return AdvisorSession(
        spec=spec, initial_state=initial_state, steps=steps,
        replay=replay, final_state=replay.final_state,
    )
