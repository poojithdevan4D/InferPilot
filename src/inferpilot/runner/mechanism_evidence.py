"""Build byte-bound, measured-window mechanism evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence

from ..mechanism import MechanismEvidence, MechanismLogSlice, SchedulerIteration
from ..telemetry import ResourceSample

_ITERATION_RE = re.compile(
    r"Iteration\((?P<index>\d+)\):\s*"
    r"(?P<context_requests>\d+) context requests,\s*"
    r"(?P<context_tokens>\d+) context tokens,\s*"
    r"(?P<generation_requests>\d+) generation requests,\s*"
    r"(?P<generation_tokens>\d+) generation tokens,\s*"
    r"iteration elapsed time:\s*(?P<elapsed>[0-9.eE+-]+) ms"
    r"(?P<dummy> \(dummy\))?,\s*GPU KV cache usage:\s*"
    r"(?P<kv>[0-9.eE+-]+)%"
)


def telemetry_digest(samples: Sequence[ResourceSample]) -> str:
    payload = [sample.model_dump(mode="json") for sample in samples]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_scheduler_iterations(text: str) -> tuple[SchedulerIteration, ...]:
    rows = []
    for match in _ITERATION_RE.finditer(text):
        rows.append(
            SchedulerIteration(
                iteration_index=int(match.group("index")),
                context_requests=int(match.group("context_requests")),
                context_tokens=int(match.group("context_tokens")),
                generation_requests=int(match.group("generation_requests")),
                generation_tokens=int(match.group("generation_tokens")),
                elapsed_ms=float(match.group("elapsed")),
                kv_cache_usage_pct=float(match.group("kv")),
                is_dummy=match.group("dummy") is not None,
            )
        )
    return tuple(rows)


def _integer_counter_bounds(
    samples: Sequence[ResourceSample], field: str, errors: list[str]
) -> tuple[float | None, float | None, int | None, int | None]:
    observed = [
        (sample.t_s, getattr(sample, field))
        for sample in samples
        if getattr(sample, field) is not None
    ]
    label = field.removesuffix("_total")
    if not observed:
        errors.append(f"{label}_counter_missing")
        return None, None, None, None
    values = [value for _, value in observed]
    if any(
        not math.isfinite(value) or value < 0 or not value.is_integer()
        for value in values
    ):
        errors.append(f"{label}_counter_non_integer")
        return None, None, None, None
    integers = [int(value) for value in values]
    if any(right < left for left, right in zip(integers, integers[1:])):
        errors.append(f"{label}_counter_decreased")
        return None, None, None, None
    return observed[0][0], observed[-1][0], integers[0], integers[-1]


def build_mechanism_evidence(
    *,
    experiment_id: str,
    measured_duration_s: float,
    sample_interval_s: float,
    max_num_seqs: int,
    telemetry_samples: Sequence[ResourceSample],
    counter_metric_name: str,
    stdout_slice: bytes,
    stderr_slice: bytes,
    stdout_bounds: tuple[int, int],
    stderr_bounds: tuple[int, int],
) -> MechanismEvidence:
    """Build evidence from exact log byte slices and boundary telemetry samples."""

    errors: list[str] = []
    recompute_t0, recompute_t1, recompute_start, recompute_end = (
        _integer_counter_bounds(
            telemetry_samples, "recomputed_token_executions_total", errors
        )
    )
    _, _, preemption_start, preemption_end = _integer_counter_bounds(
        telemetry_samples, "num_preemptions_total", errors
    )

    stdout_rows = parse_scheduler_iterations(stdout_slice.decode(errors="replace"))
    stderr_rows = parse_scheduler_iterations(stderr_slice.decode(errors="replace"))
    iterations = tuple(
        sorted(stdout_rows + stderr_rows, key=lambda row: row.iteration_index)
    )
    if not iterations:
        errors.append("iteration_details_missing")
    indexes = [row.iteration_index for row in iterations]
    if len(indexes) != len(set(indexes)):
        errors.append("iteration_index_duplicate")

    recomputed = (
        recompute_end - recompute_start
        if recompute_start is not None and recompute_end is not None
        else None
    )
    preemptions = (
        preemption_end - preemption_start
        if preemption_start is not None and preemption_end is not None
        else None
    )
    context_tokens = sum(row.context_tokens for row in iterations)
    generation_tokens = sum(row.generation_tokens for row in iterations)
    if recomputed is not None and recomputed > context_tokens + generation_tokens:
        errors.append("recomputation_exceeds_scheduled_work")

    tolerance = sample_interval_s * 2
    if recompute_t0 is not None and recompute_t0 > tolerance:
        errors.append("recompute_start_boundary_missing")
    if recompute_t1 is not None and recompute_t1 < measured_duration_s - tolerance:
        errors.append("recompute_end_boundary_missing")

    frozen_errors = tuple(sorted(set(errors)))
    counter_covers = (
        recompute_t0 is not None
        and recompute_t1 is not None
        and recompute_t0 <= tolerance
        and recompute_t1 >= measured_duration_s - tolerance
    )
    return MechanismEvidence(
        experiment_id=experiment_id,
        measured_duration_s=measured_duration_s,
        sample_interval_s=sample_interval_s,
        max_num_seqs=max_num_seqs,
        telemetry_sha256=telemetry_digest(telemetry_samples),
        log_slices=(
            MechanismLogSlice(
                stream="stdout",
                start_byte=stdout_bounds[0],
                end_byte=stdout_bounds[1],
                sha256=hashlib.sha256(stdout_slice).hexdigest(),
            ),
            MechanismLogSlice(
                stream="stderr",
                start_byte=stderr_bounds[0],
                end_byte=stderr_bounds[1],
                sha256=hashlib.sha256(stderr_slice).hexdigest(),
            ),
        ),
        counter_metric_name=counter_metric_name,
        recompute_start_t_s=recompute_t0,
        recompute_end_t_s=recompute_t1,
        recompute_start=recompute_start,
        recompute_end=recompute_end,
        recomputed_token_executions=recomputed,
        preemption_start=preemption_start,
        preemption_end=preemption_end,
        preemptions=preemptions,
        iterations=iterations,
        scheduled_context_tokens=context_tokens,
        scheduled_generation_tokens=generation_tokens,
        effective_batch_sizes=tuple(row.effective_batch_size for row in iterations),
        coverage_complete=bool(not frozen_errors and iterations and counter_covers),
        errors=frozen_errors,
    )
