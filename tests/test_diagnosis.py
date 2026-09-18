"""Bottleneck diagnosis: regime classification + mechanistic lever, self-validating."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot.diagnosis import BottleneckDiagnosis, _classify


def test_compute_bound_gets_no_lever() -> None:
    regime, lever, _ = _classify(gpu_mean=99.0, kv_peak=1.0, saturated=True)
    assert regime == "compute_bound" and lever == "none"


def test_kv_capacity_bound_recommends_fp8() -> None:
    # saturated, GPU headroom, KV full -> the one regime where a config lever wins
    regime, lever, _ = _classify(gpu_mean=60.0, kv_peak=0.99, saturated=True)
    assert regime == "kv_capacity_bound" and lever == "kv_cache_dtype=fp8"


def test_keeping_up_is_underutilized_regardless_of_gpu() -> None:
    regime, lever, _ = _classify(gpu_mean=100.0, kv_peak=0.02, saturated=False)
    assert regime == "underutilized" and lever == "none"


def test_saturated_with_spare_gpu_and_kv_is_other() -> None:
    regime, lever, _ = _classify(gpu_mean=40.0, kv_peak=0.30, saturated=True)
    assert regime == "other_bottleneck" and lever == "none"


def test_report_is_self_validating() -> None:
    good = BottleneckDiagnosis(
        gpu_utilization_mean_pct=60.0, kv_cache_usage_peak_perc=0.99, saturated=True,
        regime="kv_capacity_bound", recommended_lever="kv_cache_dtype=fp8",
        predicted_effect="KV cache limits concurrency while GPU has compute headroom; "
        "halving KV bytes should raise sustainable concurrency and throughput, bounded "
        "by the remaining GPU headroom",
    )
    assert BottleneckDiagnosis.model_validate_json(good.model_dump_json()) == good
    raw = good.model_dump(mode="json")
    raw["regime"] = "compute_bound"  # contradicts the signals
    with pytest.raises(ValidationError, match="inconsistent with its signals"):
        BottleneckDiagnosis.model_validate(raw)
