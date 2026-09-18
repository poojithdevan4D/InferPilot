"""SLO-aware Pareto config comparison — the fix for the 2026-09-18 illusory-win bug.

The 7B campaign's tpot-only objective declared width-8 a "winner" at 6 qps despite it
being 11x worse on ttft_p95. These tests lock in that such a candidate is NOT a win.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import (
    EffectiveConfig,
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    HardwareInfo,
    RequestMeasurement,
    ResourceTelemetry,
    WorkloadSpec,
)
from inferpilot.advisor import ComparisonSpec, ConfigComparison, compare_configs
from inferpilot.runner.aggregate import compute_aggregates


def _result(*, ttft: float, tpot: float, throughput: float, rate: float = 6.0,
            count: int = 12, seqs: int = 4, saturating: bool = False) -> ExperimentResult:
    """A measured result. ``saturating`` ramps TTFT across arrival order (queue growing)
    so feasibility (detect_saturation) sees an overloaded server; otherwise TTFT is flat
    at ``ttft`` (keeps up). ``ttft`` is the p95 either way."""
    out = 32
    duration = count / throughput
    if saturating:
        # TTFT rises from ~ttft/3 up to ttft across arrivals -> late/early >> threshold.
        ttfts = [ttft * (0.3 + 0.7 * i / (count - 1)) for i in range(count)]
    else:
        ttfts = [ttft] * count
    meas = [
        RequestMeasurement(
            request_id=str(i), prompt_tokens=128, output_tokens=out,
            start_time_s=i * 0.1, end_time_s=i * 0.1 + (t + (out - 1) * tpot) / 1000,
            ttft_ms=t, tpot_ms=tpot, e2e_latency_ms=t + (out - 1) * tpot, success=True,
        )
        for i, t in enumerate(ttfts)
    ]
    agg = compute_aggregates(meas, duration).model_copy(update={"gpu_memory_peak_mb": 1024})
    return ExperimentResult(
        config=ExperimentConfig(
            experiment_id="c", name="c",
            engine=EngineConfig(model="q", revision="abc", max_num_seqs=seqs,
                                max_num_batched_tokens=512),
            workload=WorkloadSpec(name="c", num_requests=count, prompt_tokens=128,
                                  output_tokens=out, request_rate_qps=rate),
        ),
        environment=EnvironmentMetadata(hardware=HardwareInfo(gpu_name="g")),
        status=ExperimentStatus.COMPLETED,
        measurements=meas,
        aggregates=agg,
        effective_config=EffectiveConfig(model="q", revision="abc", max_num_seqs=seqs,
                                         max_num_batched_tokens=512, verified=True),
        telemetry=ResourceTelemetry(
            sample_interval_s=0.25, num_samples=4, peak_gpu_memory_mb=1024,
            gpu_utilization_mean_pct=50, gpu_utilization_peak_pct=75,
            kv_cache_usage_mean_perc=0.1, kv_cache_usage_peak_perc=0.2,
        ),
    )


def test_true_pareto_win_switches() -> None:
    incumbent = _result(ttft=178, tpot=42, throughput=5.9)
    candidate = _result(ttft=150, tpot=38, throughput=5.9)  # better on both, keeps up
    cmp = compare_configs(ComparisonSpec(), incumbent, candidate)
    assert cmp.verdict == "candidate_dominates" and cmp.should_switch


def test_illusory_tpot_win_is_rejected_real_campaign_numbers() -> None:
    # vLLM default vs width-8 at 6 qps, as measured 2026-09-18: better tpot, far worse ttft.
    incumbent = _result(ttft=178.0, tpot=42.0, throughput=5.9)
    candidate = _result(ttft=1976.0, tpot=38.2, throughput=5.9)
    cmp = compare_configs(ComparisonSpec(), incumbent, candidate)
    assert cmp.verdict == "inconclusive_tradeoff"
    assert not cmp.should_switch
    assert any("ttft_p95_ms_worse" in r for r in cmp.reasons)
    assert any("tpot_p95_ms_better" in r for r in cmp.reasons)


def test_overloaded_candidate_never_wins() -> None:
    incumbent = _result(ttft=178, tpot=42, throughput=5.9)
    # low per-token latency but TTFT ramps upward => queue growing => not keeping up
    candidate = _result(ttft=30, tpot=32, throughput=0.95, saturating=True)
    cmp = compare_configs(ComparisonSpec(), incumbent, candidate)
    assert cmp.verdict == "candidate_infeasible" and not cmp.should_switch


def test_tie_within_tolerance_keeps_incumbent() -> None:
    incumbent = _result(ttft=178, tpot=42, throughput=5.9)
    candidate = _result(ttft=179, tpot=42.3, throughput=5.9)  # within 2% both ways
    cmp = compare_configs(ComparisonSpec(), incumbent, candidate)
    assert cmp.verdict == "incumbent_kept" and not cmp.should_switch


def test_candidate_wins_when_incumbent_overloaded() -> None:
    incumbent = _result(ttft=214000, tpot=32, throughput=0.95, saturating=True)  # default thrashing
    candidate = _result(ttft=180, tpot=40, throughput=5.9)      # keeps up, stable TTFT
    cmp = compare_configs(ComparisonSpec(), incumbent, candidate)
    assert cmp.verdict == "candidate_dominates" and cmp.should_switch
    assert "incumbent_overloaded" in cmp.reasons


def test_roundtrip_and_tamper_rejected() -> None:
    incumbent = _result(ttft=178, tpot=42, throughput=5.9)
    candidate = _result(ttft=1976, tpot=38.2, throughput=5.9)
    cmp = compare_configs(ComparisonSpec(), incumbent, candidate)
    assert ConfigComparison.model_validate_json(cmp.model_dump_json()) == cmp
    raw = cmp.model_dump(mode="json")
    raw["verdict"] = "candidate_dominates"
    raw["should_switch"] = True
    with pytest.raises(ValidationError, match="inconsistent"):
        ConfigComparison.model_validate(raw)


def test_context_mismatch_is_rejected() -> None:
    incumbent = _result(ttft=178, tpot=42, throughput=5.9, rate=6.0)
    candidate = _result(ttft=150, tpot=38, throughput=2.0, rate=2.0)  # different rate
    with pytest.raises(ValueError, match="same model/hardware/workload"):
        compare_configs(ComparisonSpec(), incumbent, candidate)
