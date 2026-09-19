"""Fail-closed capability check for a server's Prometheus metric surface.

This module proves only that named metric families are exposed.  It does not
prove their semantic correctness; pinned-version semantic tests remain required
before a study can claim that a counter measures recomputation or useful work.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import urllib.request
from typing import Literal, Sequence

from pydantic import Field, model_validator

from .._base import SchemaModel

_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


class MetricRequirement(SchemaModel):
    """One registered semantic and the exact metric names allowed to satisfy it."""

    semantic: str = Field(min_length=1)
    acceptable_names: tuple[str, ...] = Field(min_length=1)
    required: bool = True

    @model_validator(mode="after")
    def _check(self) -> "MetricRequirement":
        if len(set(self.acceptable_names)) != len(self.acceptable_names):
            raise ValueError("acceptable metric names must be distinct")
        if any(not _METRIC_NAME.fullmatch(name) for name in self.acceptable_names):
            raise ValueError("invalid Prometheus metric name")
        return self


class MetricCapability(SchemaModel):
    """Observed matches for a registered requirement."""

    requirement: MetricRequirement
    matched_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "MetricCapability":
        if tuple(sorted(set(self.matched_names))) != self.matched_names:
            raise ValueError("matched metric names must be sorted and distinct")
        if not set(self.matched_names) <= set(self.requirement.acceptable_names):
            raise ValueError("matched metric name is not allowed by the requirement")
        return self

    @property
    def satisfied(self) -> bool:
        return bool(self.matched_names)


class MetricsCapabilityReport(SchemaModel):
    """Self-validating snapshot of names exposed by one ``/metrics`` response."""

    report_version: Literal["0.1.0"] = "0.1.0"
    source: str = Field(min_length=1)
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_names: tuple[str, ...]
    capabilities: tuple[MetricCapability, ...]
    ready: bool

    @model_validator(mode="after")
    def _check(self) -> "MetricsCapabilityReport":
        if tuple(sorted(set(self.observed_names))) != self.observed_names:
            raise ValueError("observed metric names must be sorted and distinct")
        semantics = [cap.requirement.semantic for cap in self.capabilities]
        if len(set(semantics)) != len(semantics):
            raise ValueError("metric requirement semantics must be distinct")
        observed = set(self.observed_names)
        for capability in self.capabilities:
            expected = tuple(sorted(observed & set(capability.requirement.acceptable_names)))
            if capability.matched_names != expected:
                raise ValueError(
                    f"matches for {capability.requirement.semantic!r} contradict observed names"
                )
        expected_ready = all(
            capability.satisfied
            for capability in self.capabilities
            if capability.requirement.required
        )
        if self.ready != expected_ready:
            raise ValueError("ready flag contradicts required metric capabilities")
        return self

    @property
    def missing_required(self) -> tuple[str, ...]:
        return tuple(
            capability.requirement.semantic
            for capability in self.capabilities
            if capability.requirement.required and not capability.satisfied
        )


# The first four names exist on the current vLLM Prometheus surface. Exact
# recomputation comes from the pinned InferPilot fork. Context/generation token
# counts and effective batch size are supplied by vLLM's separately gated
# ``--enable-logging-iteration-details`` stream, not by /metrics, so they do not
# belong in this endpoint-only capability contract.
FP8_MECHANISM_REQUIREMENTS: tuple[MetricRequirement, ...] = (
    MetricRequirement(
        semantic="waiting_requests",
        acceptable_names=("vllm:num_requests_waiting",),
    ),
    MetricRequirement(
        semantic="running_requests",
        acceptable_names=("vllm:num_requests_running",),
    ),
    MetricRequirement(
        semantic="kv_cache_occupancy",
        acceptable_names=("vllm:kv_cache_usage_perc",),
    ),
    MetricRequirement(
        semantic="preemption_events",
        acceptable_names=("vllm:num_preemptions", "vllm:num_preemptions_total"),
    ),
    MetricRequirement(
        semantic="recomputed_token_executions",
        acceptable_names=(
            "inferpilot:recomputed_token_executions_total",
            "vllm:recomputed_token_executions_total",
        ),
    ),
)


def parse_prometheus_metric_names(text: str) -> tuple[str, ...]:
    """Return sorted metric-family/sample names from Prometheus exposition text."""

    names: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# HELP ") or line.startswith("# TYPE "):
            parts = line.split(maxsplit=3)
            if len(parts) >= 3 and _METRIC_NAME.fullmatch(parts[2]):
                names.add(parts[2])
            continue
        if line.startswith("#"):
            continue
        token = line.split(maxsplit=1)[0]
        name = token.split("{", maxsplit=1)[0]
        if _METRIC_NAME.fullmatch(name):
            names.add(name)
    return tuple(sorted(names))


def build_metrics_capability_report(
    text: str,
    *,
    source: str,
    requirements: Sequence[MetricRequirement] = FP8_MECHANISM_REQUIREMENTS,
) -> MetricsCapabilityReport:
    """Bind one raw response digest to registered metric-name requirements."""

    observed = parse_prometheus_metric_names(text)
    observed_set = set(observed)
    capabilities = tuple(
        MetricCapability(
            requirement=requirement,
            matched_names=tuple(sorted(observed_set & set(requirement.acceptable_names))),
        )
        for requirement in requirements
    )
    return MetricsCapabilityReport(
        source=source,
        raw_sha256=hashlib.sha256(text.encode()).hexdigest(),
        observed_names=observed,
        capabilities=capabilities,
        ready=all(cap.satisfied for cap in capabilities if cap.requirement.required),
    )


def fetch_metrics_capability_report(
    base_url: str,
    *,
    timeout_s: float = 5.0,
    requirements: Sequence[MetricRequirement] = FP8_MECHANISM_REQUIREMENTS,
) -> MetricsCapabilityReport:
    """Fetch ``/metrics`` from a running local benchmark server and inspect it."""

    url = base_url.rstrip("/") + "/metrics"
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 (caller target)
        text = response.read().decode("utf-8", errors="replace")
    return build_metrics_capability_report(text, source=url, requirements=requirements)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed fp8-study metric-capability preflight."
    )
    parser.add_argument("base_url", help="Running benchmark server, e.g. http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    report = fetch_metrics_capability_report(args.base_url)
    print(report.model_dump_json(indent=2))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
