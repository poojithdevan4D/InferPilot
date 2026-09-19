"""Immutable mechanism evidence for the fp8/preemption pilot."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import SchemaModel


class MechanismLogSlice(SchemaModel):
    """Byte-exact measured-window slice of one server log."""

    stream: Literal["stdout", "stderr"]
    start_byte: int = Field(ge=0)
    end_byte: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check(self) -> "MechanismLogSlice":
        if self.end_byte < self.start_byte:
            raise ValueError("log slice ends before it starts")
        return self


class SchedulerIteration(SchemaModel):
    """One stock vLLM iteration-detail log row."""

    iteration_index: int = Field(ge=0)
    context_requests: int = Field(ge=0)
    context_tokens: int = Field(ge=0)
    generation_requests: int = Field(ge=0)
    generation_tokens: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
    kv_cache_usage_pct: float = Field(ge=0, le=100, allow_inf_nan=False)
    is_dummy: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.context_requests + self.generation_requests


class MechanismEvidence(SchemaModel):
    """Measured-window scheduler and exact recomputation evidence.

    Derived fields are rebound on every load. Raw telemetry and logs remain
    separate immutable bundle files; their semantic and byte digests bind this
    report to those sources.
    """

    report_version: Literal["0.1.0"] = "0.1.0"
    experiment_id: str = Field(min_length=1)
    measured_duration_s: float = Field(gt=0, allow_inf_nan=False)
    sample_interval_s: float = Field(gt=0, allow_inf_nan=False)
    max_num_seqs: int = Field(ge=1)
    telemetry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    log_slices: tuple[MechanismLogSlice, MechanismLogSlice]
    counter_metric_name: Literal[
        "vllm:recomputed_token_executions_total",
        "inferpilot:recomputed_token_executions_total",
        "unobserved-recomputation-counter",
    ]

    recompute_start_t_s: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    recompute_end_t_s: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    recompute_start: Optional[int] = Field(default=None, ge=0)
    recompute_end: Optional[int] = Field(default=None, ge=0)
    recomputed_token_executions: Optional[int] = Field(default=None, ge=0)

    preemption_start: Optional[int] = Field(default=None, ge=0)
    preemption_end: Optional[int] = Field(default=None, ge=0)
    preemptions: Optional[int] = Field(default=None, ge=0)

    iterations: tuple[SchedulerIteration, ...] = ()
    scheduled_context_tokens: int = Field(ge=0)
    scheduled_generation_tokens: int = Field(ge=0)
    effective_batch_sizes: tuple[int, ...] = ()
    coverage_complete: bool
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "MechanismEvidence":
        if {item.stream for item in self.log_slices} != {"stdout", "stderr"}:
            raise ValueError("log slices must contain stdout and stderr exactly once")
        if tuple(sorted(set(self.errors))) != self.errors:
            raise ValueError("errors must be sorted and distinct")

        recompute_bounds = (
            self.recompute_start_t_s,
            self.recompute_end_t_s,
            self.recompute_start,
            self.recompute_end,
        )
        if any(value is None for value in recompute_bounds):
            if any(value is not None for value in recompute_bounds):
                raise ValueError(
                    "recompute counter boundaries must be all present or all absent"
                )
            expected_recomputed = None
        else:
            assert self.recompute_start_t_s is not None
            assert self.recompute_end_t_s is not None
            assert self.recompute_start is not None
            assert self.recompute_end is not None
            if self.recompute_end_t_s < self.recompute_start_t_s:
                raise ValueError("recompute counter timestamps are reversed")
            if self.recompute_end < self.recompute_start:
                raise ValueError("recompute counter decreased")
            expected_recomputed = self.recompute_end - self.recompute_start
        if self.recomputed_token_executions != expected_recomputed:
            raise ValueError("recomputed-token delta contradicts counter boundaries")
        if (
            self.counter_metric_name == "unobserved-recomputation-counter"
            and expected_recomputed is not None
        ):
            raise ValueError("unobserved metric name contradicts counter boundaries")

        if (self.preemption_start is None) != (self.preemption_end is None):
            raise ValueError("preemption boundaries must be both present or absent")
        expected_preemptions = None
        if self.preemption_start is not None and self.preemption_end is not None:
            if self.preemption_end < self.preemption_start:
                raise ValueError("preemption counter decreased")
            expected_preemptions = self.preemption_end - self.preemption_start
        if self.preemptions != expected_preemptions:
            raise ValueError("preemption delta contradicts counter boundaries")

        if self.scheduled_context_tokens != sum(
            row.context_tokens for row in self.iterations
        ):
            raise ValueError("scheduled context-token total contradicts iteration rows")
        if self.scheduled_generation_tokens != sum(
            row.generation_tokens for row in self.iterations
        ):
            raise ValueError(
                "scheduled generation-token total contradicts iteration rows"
            )
        expected_batches = tuple(row.effective_batch_size for row in self.iterations)
        if self.effective_batch_sizes != expected_batches:
            raise ValueError("effective batch sizes contradict iteration rows")
        if any(size > self.max_num_seqs for size in expected_batches):
            raise ValueError("effective batch size exceeds max_num_seqs")

        tolerance = self.sample_interval_s * 2
        counter_covers = (
            self.recompute_start_t_s is not None
            and self.recompute_end_t_s is not None
            and self.recompute_start_t_s <= tolerance
            and self.recompute_end_t_s >= self.measured_duration_s - tolerance
        )
        expected_coverage = bool(not self.errors and self.iterations and counter_covers)
        if self.coverage_complete != expected_coverage:
            raise ValueError("coverage flag contradicts mechanism evidence")
        return self


class MechanismCanaryReport(SchemaModel):
    """Self-validating low-pressure versus forced-preemption canary verdict."""

    report_version: Literal["0.1.0"] = "0.1.0"
    control: MechanismEvidence
    pressure: MechanismEvidence
    passed: bool
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> "MechanismCanaryReport":
        expected_reasons = _canary_reasons(self.control, self.pressure)
        if self.reasons != expected_reasons or self.passed != (not expected_reasons):
            raise ValueError("canary verdict contradicts embedded evidence")
        return self


def _canary_reasons(
    control: MechanismEvidence, pressure: MechanismEvidence
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not control.coverage_complete:
        reasons.append("control_incomplete")
    if not pressure.coverage_complete:
        reasons.append("pressure_incomplete")
    if control.recomputed_token_executions != 0:
        reasons.append("control_recomputation_nonzero")
    if not pressure.recomputed_token_executions:
        reasons.append("pressure_recomputation_not_positive")
    if not pressure.preemptions:
        reasons.append("pressure_preemptions_not_positive")
    scheduled = pressure.scheduled_context_tokens + pressure.scheduled_generation_tokens
    if (
        pressure.recomputed_token_executions is not None
        and pressure.recomputed_token_executions > scheduled
    ):
        reasons.append("recomputation_exceeds_scheduled_work")
    return tuple(sorted(reasons))


def evaluate_mechanism_canary(
    control: MechanismEvidence, pressure: MechanismEvidence
) -> MechanismCanaryReport:
    frozen = _canary_reasons(control, pressure)
    return MechanismCanaryReport(
        control=control,
        pressure=pressure,
        passed=not frozen,
        reasons=frozen,
    )
