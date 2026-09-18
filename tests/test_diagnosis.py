"""Bottleneck diagnosis: operator-grade regime map + mechanistic lever, self-validating."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot.diagnosis import BottleneckDiagnosis, _classify


def test_compute_bound_gets_no_lever() -> None:
    regime, lever, pre, _ = _classify(gpu_mean=99.0, kv_peak=1.0, saturated=True, decode_heavy=True)
    assert regime == "compute_bound" and lever == "none" and pre == []


def test_kv_bound_decode_recommends_fp8_with_preconditions() -> None:
    regime, lever, pre, _ = _classify(gpu_mean=60.0, kv_peak=0.99, saturated=True, decode_heavy=True)
    assert regime == "kv_capacity_bound_decode" and lever == "kv_cache_dtype=fp8"
    assert "model_attention_not_sliding_window" in pre and "workload_decode_dominated" in pre


def test_kv_bound_prefill_recommends_lower_batched_tokens() -> None:
    regime, lever, pre, _ = _classify(gpu_mean=60.0, kv_peak=0.99, saturated=True, decode_heavy=False)
    assert regime == "kv_capacity_bound_prefill" and lever == "max_num_batched_tokens_lower"


def test_low_utilization_offers_spec_decode_only_under_slo() -> None:
    regime, lever, pre, _ = _classify(gpu_mean=40.0, kv_peak=0.2, saturated=False, decode_heavy=True)
    assert regime == "low_utilization_latency" and lever == "speculative_decoding"
    assert "latency_slo_present" in pre


def test_keeping_up_moderate_util_is_underutilized() -> None:
    regime, lever, _, _ = _classify(gpu_mean=100.0, kv_peak=0.5, saturated=False, decode_heavy=True)
    assert regime == "underutilized" and lever == "none"


def test_saturated_spare_everything_is_other() -> None:
    regime, lever, _, _ = _classify(gpu_mean=40.0, kv_peak=0.30, saturated=True, decode_heavy=True)
    assert regime == "other_bottleneck" and lever == "none"


def test_report_is_self_validating() -> None:
    regime, lever, pre, effect = _classify(gpu_mean=60.0, kv_peak=0.99, saturated=True, decode_heavy=True)
    good = BottleneckDiagnosis(
        gpu_utilization_mean_pct=60.0, kv_cache_usage_peak_perc=0.99, saturated=True,
        decode_heavy=True, regime=regime, recommended_lever=lever,
        lever_preconditions=pre, predicted_effect=effect,
    )
    assert BottleneckDiagnosis.model_validate_json(good.model_dump_json()) == good
    raw = good.model_dump(mode="json")
    raw["regime"] = "compute_bound"
    with pytest.raises(ValidationError, match="inconsistent with its signals"):
        BottleneckDiagnosis.model_validate(raw)
