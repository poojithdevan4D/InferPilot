"""Pure, replayable controller boundary for evidence-backed advisor decisions.

The controller never mutates a serving engine.  It converts provenance-bound
profile decisions and explicit canary outcomes into auditable actions.  An
external executor remains responsible for isolated testing, restart, health
checks, and rollback.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .._base import SchemaModel
from .canary import CanaryEvaluation, CanarySpec
from .config_comparison import ComparisonSpec, ConfigComparison
from .models import AdvisorPolicy
from .profile_adapter import ProfileAdvisorDecision


CONTROLLER_VERSION = "0.1.0"


class ControllerSpec(SchemaModel):
    controller_version: Literal["0.1.0", "0.2.0", "0.3.0"]
    advisor_policy: AdvisorPolicy
    min_consecutive_windows: int = Field(default=3, ge=1)
    cooldown_windows: int = Field(default=2, ge=0)
    canary_spec: CanarySpec | None = None
    comparison_spec: ComparisonSpec | None = None

    @model_validator(mode="after")
    def check(self):
        # 0.1.0: boolean canary, no specs. 0.2.0: absolute-SLO canary (canary_spec).
        # 0.3.0: candidate must Pareto-dominate the incumbent (comparison_spec).
        if (self.controller_version == "0.2.0") != (self.canary_spec is not None):
            raise ValueError("controller 0.2.0 requires canary_spec; other versions forbid it")
        if (self.controller_version == "0.3.0") != (self.comparison_spec is not None):
            raise ValueError("controller 0.3.0 requires comparison_spec; other versions forbid it")
        return self


class ControllerState(SchemaModel):
    current_engine_overrides: dict[str, Any]
    pending_candidate: dict[str, Any] | None = None
    consecutive_candidate_windows: int = Field(default=0, ge=0)
    canary_candidate: dict[str, Any] | None = None
    cooldown_windows_remaining: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check(self):
        if (self.pending_candidate is None) != (self.consecutive_candidate_windows == 0):
            raise ValueError("pending candidate and consecutive count must be present together")
        if self.pending_candidate is not None and self.canary_candidate is not None:
            raise ValueError("pending and canary candidates are mutually exclusive")
        if self.pending_candidate == self.current_engine_overrides:
            raise ValueError("current configuration cannot be pending")
        if self.canary_candidate == self.current_engine_overrides:
            raise ValueError("current configuration cannot be its own canary")
        return self


class ControllerEvent(SchemaModel):
    event_version: Literal["0.1.0", "0.2.0", "0.3.0"]
    event_type: Literal["observation", "canary_result"]
    decision: ProfileAdvisorDecision | None = None
    canary_passed: bool | None = None
    canary_evaluation: CanaryEvaluation | None = None
    config_comparison: ConfigComparison | None = None

    @model_validator(mode="after")
    def check(self):
        extras = (self.canary_passed, self.canary_evaluation, self.config_comparison)
        if self.event_type == "observation":
            if self.decision is None or any(x is not None for x in extras):
                raise ValueError("observation requires only an advisor decision")
        elif self.event_version == "0.1.0":
            if self.decision is not None or self.canary_passed is None or self.canary_evaluation is not None or self.config_comparison is not None:
                raise ValueError("canary_result 0.1.0 requires only a pass/fail outcome")
        elif self.event_version == "0.2.0":
            if self.decision is not None or self.canary_passed is not None or self.canary_evaluation is None or self.config_comparison is not None:
                raise ValueError("canary_result 0.2.0 requires only measured canary evidence")
        elif self.decision is not None or self.canary_passed is not None or self.canary_evaluation is not None or self.config_comparison is None:
            raise ValueError("canary_result 0.3.0 requires only a config comparison")
        return self


class ControllerTransition(SchemaModel):
    transition_version: Literal["0.1.0"] = CONTROLLER_VERSION
    spec: ControllerSpec
    before: ControllerState
    event: ControllerEvent
    action: Literal["keep_current", "test_candidate", "apply_candidate", "rollback"]
    candidate: dict[str, Any] | None = None
    reasons: list[str]
    after: ControllerState

    @model_validator(mode="after")
    def check(self):
        expected = _derive_transition(self.spec, self.before, self.event)
        actual = (self.action, self.candidate, self.reasons, self.after)
        if actual != expected:
            raise ValueError("controller transition is inconsistent with inputs")
        return self


class ControllerReplay(SchemaModel):
    replay_version: Literal["0.1.0"] = CONTROLLER_VERSION
    spec: ControllerSpec
    initial_state: ControllerState
    events: list[ControllerEvent]
    transitions: list[ControllerTransition]
    final_state: ControllerState

    @model_validator(mode="after")
    def check(self):
        state = self.initial_state
        expected = []
        for event in self.events:
            transition = advance_controller(self.spec, state, event)
            expected.append(transition)
            state = transition.after
        if self.transitions != expected or self.final_state != state:
            raise ValueError("controller replay is inconsistent with events")
        return self


def _state(
    current: dict[str, Any],
    *,
    pending: dict[str, Any] | None = None,
    consecutive: int = 0,
    canary: dict[str, Any] | None = None,
    cooldown: int = 0,
) -> ControllerState:
    return ControllerState(
        current_engine_overrides=current,
        pending_candidate=pending,
        consecutive_candidate_windows=consecutive,
        canary_candidate=canary,
        cooldown_windows_remaining=cooldown,
    )


def _engine_matches(result, overrides: dict[str, Any]) -> bool:
    """True iff the result's engine config sets exactly the given overrides."""
    engine = result.config.engine
    return all(
        hasattr(engine, field) and getattr(engine, field) == value
        for field, value in overrides.items()
    )


def _derive_transition(
    spec: ControllerSpec, before: ControllerState, event: ControllerEvent
) -> tuple[str, dict[str, Any] | None, list[str], ControllerState]:
    if event.event_version != spec.controller_version:
        raise ValueError("controller event version does not match controller spec")
    if event.event_type == "canary_result":
        if before.canary_candidate is None:
            raise ValueError("canary result received without an outstanding canary")
        candidate = before.canary_candidate
        if spec.controller_version == "0.2.0":
            evaluation = event.canary_evaluation
            if evaluation.spec != spec.canary_spec:
                raise ValueError("canary evaluation spec does not match controller spec")
            if evaluation.decision.policy != spec.advisor_policy:
                raise ValueError("canary advisor policy does not match controller spec")
            if evaluation.decision.engine_overrides != candidate:
                raise ValueError("canary evidence does not match outstanding candidate")
            passed = evaluation.passed
            canary_reasons = evaluation.reasons
        elif spec.controller_version == "0.3.0":
            comparison = event.config_comparison
            if comparison.spec != spec.comparison_spec:
                raise ValueError("config comparison spec does not match controller spec")
            if not _engine_matches(comparison.candidate, candidate):
                raise ValueError("config comparison candidate does not match outstanding candidate")
            if not _engine_matches(comparison.incumbent, before.current_engine_overrides):
                raise ValueError("config comparison incumbent does not match current configuration")
            passed = comparison.should_switch
            canary_reasons = comparison.reasons
        else:
            passed = event.canary_passed
            canary_reasons = []
        if passed:
            return (
                "apply_candidate",
                candidate,
                ["canary_passed"],
                _state(candidate, cooldown=spec.cooldown_windows),
            )
        return (
            "rollback",
            candidate,
            ["canary_failed", *canary_reasons],
            _state(
                before.current_engine_overrides,
                cooldown=spec.cooldown_windows,
            ),
        )

    decision = event.decision
    if decision.policy != spec.advisor_policy:
        raise ValueError("advisor decision policy does not match controller spec")
    if before.canary_candidate is not None:
        return "keep_current", None, ["awaiting_canary_result"], before

    cooldown = before.cooldown_windows_remaining
    if decision.status == "abstain":
        return (
            "keep_current",
            None,
            ["advisor_abstained", *decision.reasons],
            _state(before.current_engine_overrides, cooldown=max(0, cooldown - 1)),
        )
    candidate = decision.engine_overrides
    if candidate == before.current_engine_overrides:
        return (
            "keep_current",
            None,
            ["recommendation_matches_current"],
            _state(before.current_engine_overrides, cooldown=max(0, cooldown - 1)),
        )
    if cooldown > 0:
        return (
            "keep_current",
            None,
            ["cooldown_active"],
            _state(before.current_engine_overrides, cooldown=cooldown - 1),
        )

    consecutive = (
        before.consecutive_candidate_windows + 1
        if before.pending_candidate == candidate
        else 1
    )
    if consecutive < spec.min_consecutive_windows:
        return (
            "keep_current",
            candidate,
            ["awaiting_stable_recommendation"],
            _state(before.current_engine_overrides, pending=candidate, consecutive=consecutive),
        )
    return (
        "test_candidate",
        candidate,
        ["stable_recommendation_threshold_met"],
        _state(before.current_engine_overrides, canary=candidate),
    )


def advance_controller(
    spec: ControllerSpec, state: ControllerState, event: ControllerEvent
) -> ControllerTransition:
    action, candidate, reasons, after = _derive_transition(spec, state, event)
    return ControllerTransition(
        spec=spec,
        before=state,
        event=event,
        action=action,
        candidate=candidate,
        reasons=reasons,
        after=after,
    )


def replay_controller(
    spec: ControllerSpec,
    initial_state: ControllerState,
    events: list[ControllerEvent],
) -> ControllerReplay:
    state = initial_state
    transitions = []
    for event in events:
        transition = advance_controller(spec, state, event)
        transitions.append(transition)
        state = transition.after
    return ControllerReplay(
        spec=spec,
        initial_state=initial_state,
        events=events,
        transitions=transitions,
        final_state=state,
    )
