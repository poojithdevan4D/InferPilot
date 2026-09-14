"""Telemetry sampler: KV scrape from /metrics, derived stats, graceful failure."""

from __future__ import annotations

import time
from time import monotonic

from fake_vllm_server import serve_in_thread

from inferpilot.runner.telemetry import TelemetrySampler


def test_kv_scrape_and_derivation() -> None:
    with serve_in_thread(mode="normal", kv_usage=0.37) as url:
        sampler = TelemetrySampler(url, t0=monotonic(), interval_s=0.05)
        sampler.start()
        time.sleep(0.2)
        telem = sampler.stop()

    assert telem.num_samples >= 1
    assert telem.sample_interval_s == 0.05
    # KV usage scraped from vllm:kv_cache_usage_perc
    assert telem.kv_cache_usage_peak_perc == 0.37
    assert telem.kv_cache_usage_mean_perc == 0.37
    # samples carry monotonic-relative timestamps
    assert all(s.t_s >= 0 for s in sampler.samples)


def test_nvml_unavailable_is_recorded_not_fatal() -> None:
    # pynvml is not installed in the test env: GPU fields must be None with an
    # explicit error, and sampling must still proceed (KV still scraped).
    with serve_in_thread(mode="normal", kv_usage=0.5) as url:
        sampler = TelemetrySampler(url, t0=monotonic(), interval_s=0.05)
        sampler.start()
        time.sleep(0.15)
        telem = sampler.stop()

    assert telem.peak_gpu_memory_mb is None
    assert telem.gpu_utilization_peak_pct is None
    assert telem.error is not None and "nvml" in telem.error
    assert telem.num_samples >= 1  # not fatal


def test_metrics_endpoint_unreachable_is_recorded_not_fatal() -> None:
    # Point at a dead port: fetch fails, recorded, no crash.
    sampler = TelemetrySampler("http://127.0.0.1:1", t0=monotonic(), interval_s=0.05)
    sampler.start()
    time.sleep(0.15)
    telem = sampler.stop()

    assert telem.num_samples >= 1
    assert telem.kv_cache_usage_peak_perc is None
    assert telem.error is not None
