"""Conservative finite-window load assessment; not a capacity estimator.

The caller must provide a post-warmup, open-loop window and a complete request
census, including requests admitted before the window and finished after it.
Current ExperimentResult does NOT establish those conditions automatically.

Token demand must be known replay work, not max_tokens or an output-length guess.
Delivered tokens are accepted useful output, excluding recompute/padding/failed
outputs. All counters and queue samples must cover the same deployment/workload.
Thresholds below are engineering guardrails, not calibrated confidence bounds.
steady_state attests to stationary offered work and exclusion of startup/inflight
fill transients, NOT stationarity of the queue itself. Do not set it merely because
a fixed number of warmup seconds elapsed. Healthy means observed balance only.
"""

from __future__ import annotations

import hashlib
import json
import math
from statistics import median
from typing import Literal, Optional, Sequence

from pydantic import Field, model_validator

from ._base import SchemaModel
from .measurements import RequestMeasurement

LoadState = Literal["healthy", "near_capacity", "overloaded", "indeterminate"]


def measurement_digest(measurements: Sequence[RequestMeasurement]) -> str:
    """Identity binding, not proof that the observations are truthful/complete."""
    rows = [m.model_dump(mode="json") for m in sorted(measurements, key=lambda m: m.request_id)]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


class LoadEvidence(SchemaModel):
    experiment_id: str = Field(min_length=1)
    measurement_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$",
        description="Digest of the intended input/arrival trace, excluding observed outputs/config knobs.")
    source: str = Field(min_length=1, description="Raw evidence artifact/collection provenance.")
    boundaries_s: list[float] = Field(min_length=5, max_length=65)
    coverage_complete: bool = False
    steady_state: bool = False
    waiting_requests: Optional[list[int]] = None
    offered_output_tokens: Optional[list[int]] = None
    delivered_output_tokens: Optional[list[int]] = None
    timeout_request_ids: Optional[list[str]] = None
    gpu_utilization_mean_pct: Optional[float] = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    kv_cache_usage_peak_perc: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    preemptions: Optional[int] = Field(default=None, ge=0, description="Counter delta over exactly these boundaries.")

    @model_validator(mode="after")
    def _check(self) -> "LoadEvidence":
        ts = self.boundaries_s
        if any(not math.isfinite(t) or t < 0 for t in ts):
            raise ValueError("boundaries must be finite and nonnegative")
        widths = [b - a for a, b in zip(ts, ts[1:])]
        if widths[0] <= 0 or any(not math.isclose(w, widths[0]) for w in widths):
            raise ValueError("use increasing, equally spaced boundaries")
        for name, size in (("waiting_requests", len(ts)),
                           ("offered_output_tokens", len(ts) - 1),
                           ("delivered_output_tokens", len(ts) - 1)):
            values = getattr(self, name)
            if values is not None and (len(values) != size or any(x < 0 for x in values)):
                raise ValueError(f"{name}: invalid length or negative counter")
        if self.timeout_request_ids is not None and len(set(self.timeout_request_ids)) != len(self.timeout_request_ids):
            raise ValueError("duplicate timeout request IDs")
        return self


class LoadAssessment(SchemaModel):
    assessment_version: Literal["0.1.0"] = "0.1.0"
    evidence: Optional[LoadEvidence] = None
    arrivals: list[int] = Field(default_factory=list)
    successful_exits: list[int] = Field(default_factory=list)
    failed_exits: list[int] = Field(default_factory=list)
    inflight: list[int] = Field(default_factory=list)
    arrival_failures: int = Field(default=0, ge=0)
    timeout_exits: Optional[int] = Field(default=None, ge=0)
    state: LoadState
    reasons: list[str]

    @property
    def duration_s(self) -> Optional[float]:
        e = self.evidence
        return None if e is None else e.boundaries_s[-1] - e.boundaries_s[0]

    @property
    def backlog_slope_rps(self) -> Optional[float]:
        if self.evidence is None:
            return None
        ts = self.evidence.boundaries_s
        return median((self.inflight[j] - self.inflight[i]) / (ts[j] - ts[i])
                      for i in range(len(ts)) for j in range(i + 1, len(ts)))

    @property
    def failure_fraction(self) -> Optional[float]:
        exits = sum(self.successful_exits) + sum(self.failed_exits)
        if not exits or not sum(self.arrivals):
            return None
        return max(sum(self.failed_exits) / exits, self.arrival_failures / sum(self.arrivals))

    @model_validator(mode="after")
    def _check(self) -> "LoadAssessment":
        e = self.evidence
        if e is None:
            if any((self.arrivals, self.successful_exits, self.failed_exits, self.inflight, self.arrival_failures)) or self.timeout_exits is not None:
                raise ValueError("counts require window evidence")
        else:
            n = len(e.boundaries_s) - 1
            for xs, size in ((self.arrivals, n), (self.successful_exits, n),
                             (self.failed_exits, n), (self.inflight, n + 1)):
                if len(xs) != size or any(x < 0 for x in xs):
                    raise ValueError("invalid load counts")
            for i in range(n):
                if self.inflight[i + 1] - self.inflight[i] != self.arrivals[i] - self.successful_exits[i] - self.failed_exits[i]:
                    raise ValueError("arrival/exit/inflight conservation violated")
            if e.waiting_requests is not None and any(q > b for q, b in zip(e.waiting_requests, self.inflight)):
                raise ValueError("waiting requests exceed complete request census")
            if self.timeout_exits is not None and self.timeout_exits > sum(self.failed_exits):
                raise ValueError("timeouts must be a subset of failed exits")
            if self.arrival_failures > sum(self.arrivals):
                raise ValueError("failed arrivals exceed arrival cohort")
        if (self.state, self.reasons) != _derive_load(self):
            raise ValueError("load assessment is inconsistent with its evidence")
        return self


def _derive_load(r: LoadAssessment) -> tuple[LoadState, list[str]]:
    e = r.evidence
    if e is None:
        return "indeterminate", ["missing_aligned_window_and_coverage_evidence"]
    missing = []
    if not e.coverage_complete:
        missing.append("request_census_not_certified_complete")
    if not e.steady_state:
        missing.append("post_warmup_open_loop_window_not_certified")
    if r.duration_s < 30 or sum(r.arrivals) < 100:
        missing.append("need_at_least_30_seconds_and_100_arrivals")
    if missing:
        return "indeterminate", missing
    n = len(r.arrivals)
    exits = [s + f for s, f in zip(r.successful_exits, r.failed_exits)]
    delta = sum(r.arrivals) - sum(exits)
    # Require both net accumulation AND a robust positive slope AND persistence.
    rate = sum(r.arrivals) / r.duration_s
    request_pressure = (delta > 0.05 * sum(r.arrivals)
                        and r.backlog_slope_rps > 0.05 * rate
                        and sum(a > x for a, x in zip(r.arrivals, exits)) >= 0.75 * n)
    demand, delivered = e.offered_output_tokens, e.delivered_output_tokens
    tokens_known = demand is not None and delivered is not None and sum(demand) > 0
    clean_outputs = not sum(r.failed_exits) and not r.arrival_failures
    token_pressure = tokens_known and clean_outputs and (sum(demand) - sum(delivered) > 0.05 * sum(demand)
                                      and sum(d > u for d, u in zip(demand, delivered)) >= 0.75 * n)
    pressure = []
    if request_pressure:
        pressure.append("persistent_request_backlog_growth")
    if token_pressure:
        pressure.append("persistent_useful_output_work_deficit")
    if pressure:
        return "overloaded", pressure
    if -delta > 0.05 * sum(r.arrivals):
        return "indeterminate", ["prior_backlog_is_draining_extend_observation"]
    # Failures may be capacity-related OR authentication/network/application errors.
    # They veto health but alone do not establish a capacity bottleneck.
    if r.failure_fraction is None or r.failure_fraction > 0.01:
        return "indeterminate", ["insufficient_successful_service_or_failure_budget_exceeded"]
    if e.waiting_requests is None:
        missing.append("missing_waiting_queue_samples")
    if not tokens_known:
        missing.append("missing_known_token_demand_or_useful_delivery")
    if missing:
        return "indeterminate", missing
    if (any(e.waiting_requests) or delta > 0.01 * sum(r.arrivals)
            or sum(delivered) < 0.99 * sum(demand)):
        return "near_capacity", ["residual_queue_or_service_deficit_extend_window_and_sweep_rate"]
    return "healthy", ["no_observed_load_pressure_not_a_headroom_or_slo_certificate"]


def assess_load_state(
    measurements: Sequence[RequestMeasurement], *, evidence: Optional[LoadEvidence] = None,
) -> LoadAssessment:
    """Bins use [start, end); census at boundary t uses start < t <= end.

    Failed exits (including timeouts) close requests but do not count as useful
    completions. No timeout type is guessed from free-form error strings.
    Missing evidence abstains; malformed or mismatched evidence raises ValueError.
    """
    data: dict = {"evidence": evidence}
    if evidence is not None:
        evidence = LoadEvidence.model_validate(evidence.model_dump())
        data["evidence"] = evidence
        if len({m.request_id for m in measurements}) != len(measurements):
            raise ValueError("duplicate request IDs")
        for m in measurements:
            if any(v is not None and not math.isfinite(v) for v in
                   (m.start_time_s, m.end_time_s, m.ttft_ms, m.tpot_ms, m.e2e_latency_ms)):
                raise ValueError("nonfinite measurement")
            if m.start_time_s < 0 or m.end_time_s < m.start_time_s:
                raise ValueError("invalid request timestamp ordering")
            elapsed_ms = (m.end_time_s - m.start_time_s) * 1000
            if (m.e2e_latency_ms is not None and not math.isclose(m.e2e_latency_ms, elapsed_ms, rel_tol=1e-6, abs_tol=1e-3)) or (m.ttft_ms is not None and m.ttft_ms > elapsed_ms + 1e-3):
                raise ValueError("measurement latency contradicts request timestamps")
        if evidence.measurement_sha256 != measurement_digest(measurements):
            raise ValueError("load evidence does not match measurements")
        ts = evidence.boundaries_s
        if evidence.delivered_output_tokens is not None:
            available = sum(m.output_tokens for m in measurements
                            if m.success and m.start_time_s < ts[-1] and m.end_time_s >= ts[0])
            if sum(evidence.delivered_output_tokens) > available:
                raise ValueError("useful delivery exceeds overlapping successful output census")
        data.update(arrivals=[], successful_exits=[], failed_exits=[],
                    arrival_failures=sum(not m.success and ts[0] <= m.start_time_s < ts[-1] for m in measurements),
                    inflight=[sum(m.start_time_s < t <= m.end_time_s for m in measurements) for t in ts])
        for a, b in zip(ts, ts[1:]):
            data["arrivals"].append(sum(a <= m.start_time_s < b for m in measurements))
            data["successful_exits"].append(sum(m.success and a <= m.end_time_s < b for m in measurements))
            data["failed_exits"].append(sum(not m.success and a <= m.end_time_s < b for m in measurements))
        if evidence.timeout_request_ids is not None:
            failed = {m.request_id for m in measurements if not m.success}
            if not set(evidence.timeout_request_ids) <= failed:
                raise ValueError("timeout IDs must identify failed requests in the census")
            data["timeout_exits"] = sum(m.request_id in evidence.timeout_request_ids and ts[0] <= m.end_time_s < ts[-1]
                                        for m in measurements)
    # Construct only an internal provisional object, then validate the final report.
    provisional = LoadAssessment.model_construct(**data)
    state, reasons = _derive_load(provisional)
    return LoadAssessment(**data, state=state, reasons=reasons)


class SaturationReport(SchemaModel):
    """Legacy TTFT trend only. saturated=False NEVER certifies healthy load."""

    saturation_version: Literal["0.1.0"] = "0.1.0"
    num_samples: int = Field(ge=0)
    early_ttft_ms: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    late_ttft_ms: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    growth_ratio: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    threshold: float = Field(gt=1, allow_inf_nan=False)
    saturated: bool
    reason: str

    @model_validator(mode="after")
    def _check(self) -> "SaturationReport":
        if self.early_ttft_ms is None or self.late_ttft_ms is None:
            if self.saturated or self.growth_ratio is not None:
                raise ValueError("insufficient-sample report cannot be saturated")
        else:
            if self.early_ttft_ms <= 0 or self.growth_ratio is None or not math.isclose(self.growth_ratio, self.late_ttft_ms / self.early_ttft_ms):
                raise ValueError("growth_ratio inconsistent with early/late TTFT")
            if self.saturated != (self.growth_ratio > self.threshold):
                raise ValueError("saturated flag inconsistent with growth_ratio and threshold")
        return self


def _legacy_ttft_report(measurements, threshold, min_samples) -> SaturationReport:
    if min_samples < 2:
        raise ValueError("min_samples must be >= 2")
    xs = [m.ttft_ms for m in sorted(measurements, key=lambda m: m.start_time_s)
          if m.success and m.ttft_ms is not None]
    if any(not math.isfinite(x) for x in xs):
        raise ValueError("nonfinite TTFT")
    if len(xs) < min_samples:
        return SaturationReport(num_samples=len(xs), threshold=threshold, saturated=False, reason="insufficient_samples")
    h = len(xs) // 2
    early, late = sum(xs[:h]) / h, sum(xs[h:]) / (len(xs) - h)
    if early <= 0:
        return SaturationReport(num_samples=len(xs), threshold=threshold, saturated=False, reason="nonpositive_baseline_ttft")
    ratio = late / early
    return SaturationReport(num_samples=len(xs), threshold=threshold, early_ttft_ms=early,
                            late_ttft_ms=late, growth_ratio=ratio, saturated=ratio > threshold,
                            reason="ttft_growing_across_arrivals" if ratio > threshold else "ttft_stable")


def detect_saturation(measurements: Sequence[RequestMeasurement], *, threshold: float = 1.5,
                      min_samples: int = 8) -> SaturationReport:
    """Compatibility shim for the old TTFT statistic, NOT a load-state decision."""
    return _legacy_ttft_report(measurements, threshold, min_samples)
