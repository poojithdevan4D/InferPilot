"""Pure extraction of online-safe arrival features from stored run evidence."""

from __future__ import annotations

import statistics
from typing import Any

from ..comparison.fingerprint import exact_fingerprint
from ..results import ExperimentResult
from ..runner.aggregate import percentile
from .models import (
    WINDOW_KEYS,
    WINDOW_SECONDS,
    ArrivalEvidence,
    ArrivalFeatureReport,
    max_arrivals_in_window,
)


def extract_arrival_features(
    result: ExperimentResult,
    arrivals: ArrivalEvidence | dict[str, Any],
    *,
    observation_cutoff_s: float,
) -> ArrivalFeatureReport:
    """Return features using only dispatches observed by ``observation_cutoff_s``.

    The complete artifact is accepted for offline replay, but future offsets are
    filtered before any feature is computed or retained. Request completions and
    latency outcomes are intentionally not inputs to the feature calculation.
    """
    evidence = (
        arrivals
        if isinstance(arrivals, ArrivalEvidence)
        else ArrivalEvidence.model_validate(arrivals)
    )
    workload = result.config.workload
    if workload.request_rate_qps is None:
        raise ValueError("arrival features require an open-loop workload")
    if evidence.request_rate_qps != workload.request_rate_qps:
        raise ValueError("arrival artifact rate does not match the experiment config")
    if evidence.algorithm != workload.arrival_pattern:
        raise ValueError("arrival artifact algorithm does not match the experiment config")
    if evidence.burst_size != workload.burst_size:
        raise ValueError("arrival artifact burst_size does not match the experiment config")
    # Bind to the EFFECTIVE arrival seed. 0.3.0 artifacts carry only `seed` (which
    # was the arrival seed); 0.4.0 artifacts carry an explicit arrival_seed. Either
    # way it must equal the config's effective arrival seed — no reinterpretation.
    if evidence.effective_arrival_seed != workload.effective_arrival_seed:
        raise ValueError("arrival artifact arrival-seed does not match the experiment config")
    if len(evidence.actual_dispatch_offsets_s) != workload.num_requests:
        raise ValueError("arrival artifact count does not match the experiment config")
    if observation_cutoff_s <= 0:
        raise ValueError("observation_cutoff_s must be positive")

    observed = [
        value
        for value in evidence.actual_dispatch_offsets_s
        if value <= observation_cutoff_s
    ]
    intervals = [right - left for left, right in zip(observed, observed[1:])]
    if intervals:
        mean = statistics.mean(intervals)
        p50 = percentile(intervals, 50)
        p95 = percentile(intervals, 95)
        cv = statistics.pstdev(intervals) / mean if mean > 0 else None
        rate = 1.0 / mean if mean > 0 else None
    else:
        mean = p50 = p95 = cv = rate = None

    return ArrivalFeatureReport(
        source_experiment_id=result.config.experiment_id,
        source_exact_fingerprint=exact_fingerprint(result),
        observation_cutoff_s=observation_cutoff_s,
        declared_request_rate_qps=workload.request_rate_qps,
        declared_prompt_tokens=workload.prompt_tokens,
        declared_output_tokens=workload.output_tokens,
        observed_dispatch_offsets_s=observed,
        num_arrivals=len(observed),
        arrival_count_rate_qps=len(observed) / observation_cutoff_s,
        interarrival_mean_s=mean,
        interarrival_p50_s=p50,
        interarrival_p95_s=p95,
        interarrival_cv=cv,
        interarrival_rate_qps=rate,
        max_arrivals_by_window_s={
            key: max_arrivals_in_window(observed, window)
            for key, window in zip(WINDOW_KEYS, WINDOW_SECONDS)
        },
    )
