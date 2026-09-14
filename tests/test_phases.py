"""RunnerPhaseTiming contract: self-validation of the monotonic timing artifact."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import RunnerPhaseTiming
from inferpilot.runner.phases import PhaseTimer


def _completed_timer() -> PhaseTimer:
    """A timer whose marks describe a clean COMPLETED run (synthetic offsets)."""
    t = PhaseTimer()
    t.launch()
    origin = t._origin  # noqa: SLF001 (test introspection)
    offsets = {
        "server_ready": 0.5,
        "config_verify_start": 0.6, "config_verify_end": 0.7,
        "warmup_start": 0.8, "warmup_end": 1.0,
        "measured_start": 1.0, "measured_end": 3.5,  # duration 2.5
        "finalize_start": 3.5, "finalize_end": 3.6,
        "teardown_start": 3.6, "teardown_end": 3.8,
    }
    for name, off in offsets.items():
        t.mark_at(name, origin + off)
    return t


def test_completed_timing_builds_and_roundtrips() -> None:
    timing = _completed_timer().build("completed", aggregate_duration_s=2.5)
    assert RunnerPhaseTiming.model_validate_json(timing.model_dump_json()) == timing
    spans = {p.name: p for p in timing.phases}
    assert spans["measured_window"].duration_s == pytest.approx(2.5)
    assert all(spans[n].completed for n in
               ("server_startup", "config_verification", "warmup",
                "measured_window", "finalization", "teardown", "total_occupancy"))
    assert timing.report_version == "0.1.0"


def test_startup_failure_phases_never_begin() -> None:
    t = PhaseTimer()
    t.launch()
    t.mark("teardown_start")
    t.mark("teardown_end")
    timing = t.build("failed")
    spans = {p.name: p for p in timing.phases}
    assert spans["server_startup"].began and not spans["server_startup"].completed
    for absent in ("config_verification", "warmup", "measured_window", "finalization"):
        assert not spans[absent].began
    assert spans["teardown"].completed and spans["total_occupancy"].completed


def test_aggregate_duration_disagreement_rejected() -> None:
    timing = _completed_timer().build("completed", aggregate_duration_s=2.5)
    raw = timing.model_dump(mode="json")
    raw["aggregate_duration_s"] = 9.9  # disagrees with measured_window (2.5)
    with pytest.raises(ValidationError, match="agree with AggregateMetrics"):
        RunnerPhaseTiming.model_validate(raw)


def test_out_of_order_events_rejected() -> None:
    timing = _completed_timer().build("completed", aggregate_duration_s=2.5)
    raw = timing.model_dump(mode="json")
    raw["events"][-1]["offset_s"] = 0.1  # last event jumps backwards (still >= 0)
    with pytest.raises(ValidationError, match="monotonically nondecreasing"):
        RunnerPhaseTiming.model_validate(raw)


def test_measured_completed_requires_completed_status() -> None:
    raw = _completed_timer().build("completed", aggregate_duration_s=2.5).model_dump(mode="json")
    raw["terminal_status"] = "failed"
    raw["aggregate_duration_s"] = None
    with pytest.raises(ValidationError, match="measured_window completes iff"):
        RunnerPhaseTiming.model_validate(raw)


def test_completed_requires_aggregate_duration() -> None:
    raw = _completed_timer().build("completed", aggregate_duration_s=2.5).model_dump(mode="json")
    raw["aggregate_duration_s"] = None
    with pytest.raises(ValidationError, match="aggregate_duration_s is present iff"):
        RunnerPhaseTiming.model_validate(raw)


def test_version_gate() -> None:
    raw = _completed_timer().build("completed", aggregate_duration_s=2.5).model_dump(mode="json")
    raw["report_version"] = "0.0.9"
    with pytest.raises(ValidationError, match="unsupported phase-timing report_version"):
        RunnerPhaseTiming.model_validate(raw)
