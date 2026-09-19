"""Metric capability preflight is deterministic, bound, and fail-closed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot.runner.metrics_capabilities import (
    FP8_MECHANISM_REQUIREMENTS,
    MetricsCapabilityReport,
    build_metrics_capability_report,
    parse_prometheus_metric_names,
)


BASE = """\
# HELP vllm:kv_cache_usage_perc Cache occupancy.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="fake"} 0.5
vllm:num_requests_waiting 2
vllm:num_requests_running 4
vllm:num_preemptions_total 3
"""


def test_parser_reads_help_type_and_labelled_samples() -> None:
    assert parse_prometheus_metric_names(BASE) == (
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    )


def test_current_public_surface_fails_closed_for_mechanism_study() -> None:
    report = build_metrics_capability_report(BASE, source="fixture")
    assert report.ready is False
    assert report.missing_required == (
        "recomputed_token_executions",
        "scheduled_prefill_tokens",
        "scheduled_decode_tokens",
        "effective_batch_size",
    )
    assert MetricsCapabilityReport.model_validate_json(report.model_dump_json()) == report


def test_instrumented_surface_is_ready() -> None:
    additions = "\n".join(
        f"{requirement.acceptable_names[0]} 1"
        for requirement in FP8_MECHANISM_REQUIREMENTS
        if requirement.semantic in {
            "recomputed_token_executions",
            "scheduled_prefill_tokens",
            "scheduled_decode_tokens",
            "effective_batch_size",
        }
    )
    report = build_metrics_capability_report(BASE + additions, source="instrumented-fixture")
    assert report.ready is True
    assert report.missing_required == ()


@pytest.mark.parametrize("field", ["ready", "matched_names"])
def test_persisted_derivations_reject_tampering(field: str) -> None:
    report = build_metrics_capability_report(BASE, source="fixture")
    payload = report.model_dump(mode="json")
    if field == "ready":
        payload["ready"] = True
        match = "ready flag"
    else:
        payload["capabilities"][0]["matched_names"] = []
        match = "contradict observed names"
    with pytest.raises(ValidationError, match=match):
        MetricsCapabilityReport.model_validate(payload)
