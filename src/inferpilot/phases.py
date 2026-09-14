"""Runner phase-timing contract (M2C cost evidence).

An immutable, exact-versioned artifact describing where wall time went during a
single ``run_experiment`` invocation. It is derived **exclusively from a monotonic
clock**; wall-clock timestamps are never used here.

Design: the source of truth is an ordered log of monotonic-relative ``events``
(offsets in seconds from *server process launch*, offset ``0.0``). Named
:class:`PhaseSpan`s are *derived* from paired events, so gaps and failure paths
stay auditable. A phase whose begin event is absent never began; a phase with a
begin but no end event began but did not finish.

Phase semantics on non-COMPLETED terminal paths:

* readiness ``TIMEOUT`` / startup ``FAILED`` / ``OOM``: ``server_startup`` began
  (launch) but did not complete (no ``server_ready``); ``config_verification``,
  ``warmup``, ``measured_window`` and ``finalization`` never began.
* ``ConfigFidelityMismatch`` (FAILED): ``server_startup`` and
  ``config_verification`` completed; ``warmup`` / ``measured_window`` /
  ``finalization`` never began.
* ``WarmupFailure`` (FAILED): ``warmup`` completed (it ran) but a request failed,
  so ``measured_window`` and ``finalization`` never began.
* ``teardown`` and ``total_occupancy`` are recorded on **every** terminal path,
  because cleanup always runs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel

PHASE_TIMING_VERSION = "0.1.0"

# Derived phase name -> (begin event, end event).
PHASE_DEFS: dict[str, tuple[str, str]] = {
    "server_startup": ("server_launch", "server_ready"),
    "config_verification": ("config_verify_start", "config_verify_end"),
    "warmup": ("warmup_start", "warmup_end"),
    "measured_window": ("measured_start", "measured_end"),
    "finalization": ("finalize_start", "finalize_end"),
    "teardown": ("teardown_start", "teardown_end"),
    "total_occupancy": ("server_launch", "teardown_end"),
}

_AGGREGATE_TOLERANCE_S = 1e-6


class PhaseEvent(SchemaModel):
    """One monotonic-relative event offset (seconds from server-process launch)."""

    name: str
    offset_s: float = Field(ge=0)


class PhaseSpan(SchemaModel):
    """A derived phase span. ``began``/``completed`` make partial phases explicit."""

    name: str
    began: bool
    completed: bool
    begin_offset_s: Optional[float] = Field(default=None, ge=0)
    end_offset_s: Optional[float] = Field(default=None, ge=0)
    duration_s: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "PhaseSpan":
        if self.began != (self.begin_offset_s is not None):
            raise ValueError("began must agree with begin_offset_s")
        if self.completed != (self.end_offset_s is not None):
            raise ValueError("completed must agree with end_offset_s")
        if self.completed and not self.began:
            raise ValueError("a completed phase must have begun")
        if self.completed:
            if self.end_offset_s < self.begin_offset_s:
                raise ValueError("phase end precedes its begin")
            if self.duration_s != self.end_offset_s - self.begin_offset_s:
                raise ValueError("duration_s must equal end - begin")
        elif self.duration_s is not None:
            raise ValueError("an unfinished phase must not carry a duration")
        return self


class RunnerPhaseTiming(SchemaModel):
    """Self-validating, monotonic phase-timing artifact for one run."""

    report_version: str = PHASE_TIMING_VERSION
    origin: Literal["server_process_launch"] = "server_process_launch"
    terminal_status: Literal["completed", "failed", "oom", "timeout"]
    events: list[PhaseEvent] = Field(min_length=1)
    phases: list[PhaseSpan]
    # Copied from the run's AggregateMetrics.duration_s when COMPLETED (else None).
    aggregate_duration_s: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "RunnerPhaseTiming":
        if self.report_version != PHASE_TIMING_VERSION:
            raise ValueError(
                f"unsupported phase-timing report_version {self.report_version!r}; "
                f"expected {PHASE_TIMING_VERSION!r}"
            )

        # Events: first is the launch origin at 0.0; offsets nonnegative + ordered.
        if self.events[0].name != "server_launch" or self.events[0].offset_s != 0.0:
            raise ValueError("first event must be server_launch at offset 0.0")
        names = [e.name for e in self.events]
        if len(set(names)) != len(names):
            raise ValueError("event names must be unique")
        offsets = [e.offset_s for e in self.events]
        if any(b < a for a, b in zip(offsets, offsets[1:])):
            raise ValueError("event offsets must be monotonically nondecreasing")
        at = {e.name: e.offset_s for e in self.events}

        # Phases must be exactly the derived set, recomputed from the events.
        if [p.name for p in self.phases] != list(PHASE_DEFS):
            raise ValueError("phases must be the full derived set in canonical order")
        for span in self.phases:
            begin_name, end_name = PHASE_DEFS[span.name]
            begin = at.get(begin_name)
            end = at.get(end_name)
            if span.begin_offset_s != begin or span.end_offset_s != end:
                raise ValueError(f"phase {span.name} offsets are inconsistent with events")

        spans = {p.name: p for p in self.phases}
        # Cleanup always runs.
        if not spans["teardown"].completed or not spans["total_occupancy"].completed:
            raise ValueError("teardown and total_occupancy must complete on every path")

        completed = self.terminal_status == "completed"
        measured_done = spans["measured_window"].completed
        # A COMPLETED run requires a finished measured window. The converse does
        # NOT hold: the measured window may complete and the run still end FAILED
        # if aggregation or artifact finalization fails afterward.
        if completed and not measured_done:
            raise ValueError("a COMPLETED run must have a completed measured_window")
        # aggregate_duration_s IS the measured-window duration, so it is available
        # exactly when the measured window completed — regardless of terminal status.
        if (self.aggregate_duration_s is not None) != measured_done:
            raise ValueError("aggregate_duration_s is present iff the measured window completed")
        if measured_done:
            measured = spans["measured_window"].duration_s
            if abs(measured - self.aggregate_duration_s) > _AGGREGATE_TOLERANCE_S:
                raise ValueError(
                    "measured_window duration must agree with AggregateMetrics.duration_s"
                )
        if completed:
            for required in ("server_startup", "config_verification", "finalization"):
                if not spans[required].completed:
                    raise ValueError(f"{required} must complete on a COMPLETED run")
        return self
