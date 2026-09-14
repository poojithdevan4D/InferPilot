"""Monotonic phase timer + builder for the RunnerPhaseTiming artifact."""

from __future__ import annotations

from time import monotonic
from typing import Optional

from ..phases import PHASE_DEFS, PhaseEvent, PhaseSpan, RunnerPhaseTiming


class PhaseTimer:
    """Records ordered monotonic-relative event offsets for one run.

    ``launch()`` sets the origin (server-process launch, offset 0.0). ``mark()``
    records later events; unreached events are simply never marked, which is how
    failure paths stay auditable.
    """

    def __init__(self) -> None:
        self._origin: Optional[float] = None
        self._events: list[tuple[str, float]] = []

    def launch(self) -> None:
        self._origin = monotonic()
        self._events.append(("server_launch", 0.0))

    def mark(self, name: str) -> None:
        if self._origin is None:
            return
        # Never let a tiny clock hiccup produce a negative offset.
        self._events.append((name, max(0.0, monotonic() - self._origin)))

    def mark_at(self, name: str, monotonic_value: float) -> None:
        """Record an event at an already-sampled monotonic instant.

        Used for the measured-window boundaries so the derived duration matches
        the value handed to ``compute_aggregates`` (no re-sampling drift).
        """
        if self._origin is None:
            return
        self._events.append((name, max(0.0, monotonic_value - self._origin)))

    @property
    def started(self) -> bool:
        return self._origin is not None

    def build(
        self, terminal_status: str, aggregate_duration_s: Optional[float] = None
    ) -> RunnerPhaseTiming:
        at = dict(self._events)
        phases: list[PhaseSpan] = []
        for name, (begin_name, end_name) in PHASE_DEFS.items():
            begin = at.get(begin_name)
            end = at.get(end_name)
            phases.append(
                PhaseSpan(
                    name=name,
                    began=begin is not None,
                    completed=end is not None,
                    begin_offset_s=begin,
                    end_offset_s=end,
                    duration_s=(end - begin) if (begin is not None and end is not None) else None,
                )
            )
        return RunnerPhaseTiming(
            terminal_status=terminal_status,
            events=[PhaseEvent(name=n, offset_s=o) for n, o in self._events],
            phases=phases,
            aggregate_duration_s=aggregate_duration_s,
        )
