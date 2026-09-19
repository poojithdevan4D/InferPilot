"""Build aligned load evidence from one completed measured window."""

from __future__ import annotations

import math
import hashlib
import json
from typing import Sequence

from ..measurements import RequestMeasurement
from ..saturation import LoadEvidence, measurement_digest
from ..telemetry import ResourceSample


def intended_replay_digest(
    prompts: Sequence[str],
    scheduled_offsets_s: Sequence[float],
    requested_output_tokens: int,
) -> str:
    """Digest intended input work and arrivals, excluding observed outcomes/engine knobs."""
    payload = {
        "prompts": list(prompts),
        "scheduled_offsets_s": list(scheduled_offsets_s),
        "requested_output_tokens": requested_output_tokens,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _nearest(samples: Sequence[ResourceSample], boundary: float) -> ResourceSample:
    return min(samples, key=lambda sample: (abs(sample.t_s - boundary), sample.t_s))


def _covers(samples: Sequence[ResourceSample], start: float, end: float, tolerance: float) -> bool:
    return bool(samples) and min(sample.t_s for sample in samples) <= start + tolerance and max(
        sample.t_s for sample in samples
    ) >= end - tolerance


def build_load_evidence(
    *,
    experiment_id: str,
    measured_window_t0_s: float,
    measured_window_end_s: float,
    measurements: Sequence[RequestMeasurement],
    telemetry_samples: Sequence[ResourceSample],
    requested_output_tokens: int,
    window_width_s: float = 5.0,
    coverage_complete: bool,
    steady_state: bool,
    source: str = "runner-measured-window-v1",
    replay_sha256: str | None = None,
) -> LoadEvidence:
    """Construct evidence without duplicating request-count derivations.

    Request timestamps and telemetry sample timestamps are offsets from the
    measured-window origin.  The absolute monotonic start/end values are used
    only to establish duration.  A trailing partial bin is excluded.
    """
    duration = measured_window_end_s - measured_window_t0_s
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("measured window must have a finite positive duration")
    if not math.isfinite(window_width_s) or window_width_s <= 0:
        raise ValueError("window_width_s must be finite and positive")
    if requested_output_tokens <= 0:
        raise ValueError("requested_output_tokens must be positive")
    num_windows = min(64, math.floor(duration / window_width_s))
    if num_windows < 4:
        raise ValueError("load evidence requires at least four complete windows")
    boundaries = [index * window_width_s for index in range(num_windows + 1)]

    offered = []
    delivered = []
    for left, right in zip(boundaries, boundaries[1:]):
        offered.append(
            requested_output_tokens
            * sum(left <= measurement.start_time_s < right for measurement in measurements)
        )
        delivered.append(
            sum(
                measurement.output_tokens
                for measurement in measurements
                if measurement.success and left <= measurement.end_time_s < right
            )
        )

    waiting_samples = [
        sample for sample in telemetry_samples if sample.num_requests_waiting is not None
    ]
    waiting = (
        [_nearest(waiting_samples, boundary).num_requests_waiting for boundary in boundaries]
        if _covers(waiting_samples, boundaries[0], boundaries[-1], window_width_s)
        else None
    )
    inflight = [
        sum(
            measurement.start_time_s < boundary <= measurement.end_time_s
            for measurement in measurements
        )
        for boundary in boundaries
    ]
    if waiting is not None and any(wait > census for wait, census in zip(waiting, inflight)):
        waiting = None
    running_samples = [
        sample for sample in telemetry_samples if sample.num_requests_running is not None
    ]
    if waiting is not None and _covers(
        running_samples, boundaries[0], boundaries[-1], window_width_s
    ):
        running = [
            _nearest(running_samples, boundary).num_requests_running
            for boundary in boundaries
        ]
        # Misaligned or contradictory gauges cannot certify an empty queue.
        if any(wait + run > census for wait, run, census in zip(waiting, running, inflight)):
            waiting = None
    preemption_samples = [
        sample for sample in telemetry_samples if sample.num_preemptions_total is not None
    ]
    preemptions = None
    if len(preemption_samples) >= 2 and _covers(
        preemption_samples, boundaries[0], boundaries[-1], window_width_s
    ):
        first = _nearest(preemption_samples, boundaries[0]).num_preemptions_total
        last = _nearest(preemption_samples, boundaries[-1]).num_preemptions_total
        if last >= first:
            preemptions = int(round(last - first))

    gpu_values = [
        sample.gpu_utilization_pct
        for sample in telemetry_samples
        if sample.gpu_utilization_pct is not None
    ]
    kv_values = [
        sample.kv_cache_usage_perc
        for sample in telemetry_samples
        if sample.kv_cache_usage_perc is not None
    ]
    return LoadEvidence(
        experiment_id=experiment_id,
        measurement_sha256=measurement_digest(measurements),
        replay_sha256=replay_sha256,
        source=source,
        boundaries_s=boundaries,
        coverage_complete=coverage_complete,
        steady_state=steady_state,
        waiting_requests=waiting,
        offered_output_tokens=offered,
        delivered_output_tokens=delivered,
        gpu_utilization_mean_pct=(sum(gpu_values) / len(gpu_values) if gpu_values else None),
        kv_cache_usage_peak_perc=max(kv_values) if kv_values else None,
        preemptions=preemptions,
    )
