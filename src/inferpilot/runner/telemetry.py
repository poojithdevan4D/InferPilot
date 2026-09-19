"""Measured-window resource telemetry sampler.

A lightweight background thread that samples, on a monotonic clock:
* GPU memory used and GPU utilization via NVML (pynvml), and
* KV usage, waiting/running request gauges, and the cumulative preemption
  counter scraped from the server's ``/metrics`` endpoint.

Design constraints:
* sampling is best-effort — any failure is recorded in the returned
  :class:`ResourceTelemetry.error` and never propagates to request measurement;
* the default interval (250 ms) keeps overhead negligible for a batch-1 workload;
* the sampler only runs during the measured window (started/stopped by the runner).
"""

from __future__ import annotations

import re
import threading
import urllib.error
import urllib.request
from time import monotonic
from typing import Optional

from ..telemetry import ResourceSample, ResourceTelemetry

DEFAULT_INTERVAL_S = 0.25  # 250 ms — lightweight; documented here and in the schema.

_KV_METRIC = "vllm:kv_cache_usage_perc"
_KV_RE = re.compile(r"^vllm:kv_cache_usage_perc(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE)
# Cumulative counter; window preemptions = last - first observed.
_PREEMPT_RE = re.compile(
    r"^vllm:num_preemptions_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE
)
_WAITING_RE = re.compile(
    r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE
)
_RUNNING_RE = re.compile(
    r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE
)


class TelemetrySampler:
    """Background sampler; construct, ``start()``, then ``stop()`` for the summary."""

    def __init__(
        self,
        base_url: str,
        t0: float,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        gpu_index: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.t0 = t0
        self.interval_s = interval_s
        self.gpu_index = gpu_index

        self.samples: list[ResourceSample] = []
        self._preempt_first: Optional[float] = None  # cumulative counter bounds over window
        self._preempt_last: Optional[float] = None
        self._errors: list[str] = []  # deduped, order-preserving
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._pynvml = None
        self._handle = None

    # -- error helper ----------------------------------------------------
    def _note_error(self, msg: str) -> None:
        if msg not in self._errors:
            self._errors.append(msg)

    # -- NVML ------------------------------------------------------------
    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._pynvml = pynvml
        except Exception as exc:  # missing lib, no driver, bad index...
            self._note_error(f"nvml_init_failed: {exc!r}")

    def _sample_gpu(self) -> tuple[Optional[int], Optional[float]]:
        if self._pynvml is None or self._handle is None:
            return None, None
        mem = util = None
        try:
            mem = int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used) // (1024 * 1024)
        except Exception as exc:
            self._note_error(f"nvml_memory_failed: {exc!r}")
        try:
            util = float(self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        except Exception as exc:
            self._note_error(f"nvml_util_failed: {exc!r}")
        return mem, util

    # -- /metrics --------------------------------------------------------
    def _sample_metrics(
        self,
    ) -> tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
        url = f"{self.base_url}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:  # noqa: S310 (localhost)
                text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self._note_error(f"metrics_fetch_failed: {exc!r}")
            return None, None, None, None
        preempt_value = None
        preempt = _PREEMPT_RE.search(text)
        if preempt:
            try:
                value = float(preempt.group(1))
                preempt_value = value
                if self._preempt_first is None:
                    self._preempt_first = value
                self._preempt_last = value
            except ValueError:
                pass
        else:
            self._note_error("metric_not_found: vllm:num_preemptions_total")
        queues = []
        for name, regex in (
            ("vllm:num_requests_waiting", _WAITING_RE),
            ("vllm:num_requests_running", _RUNNING_RE),
        ):
            queue_match = regex.search(text)
            if queue_match is None:
                self._note_error(f"metric_not_found: {name}")
                queues.append(None)
                continue
            try:
                value = float(queue_match.group(1))
                queues.append(int(value) if value >= 0 and value.is_integer() else None)
                if queues[-1] is None:
                    self._note_error(f"metric_parse_failed: {name}")
            except ValueError:
                self._note_error(f"metric_parse_failed: {name}")
                queues.append(None)
        match = _KV_RE.search(text)
        if not match:
            self._note_error(f"metric_not_found: {_KV_METRIC}")
            return None, preempt_value, queues[0], queues[1]
        try:
            kv = float(match.group(1))
        except ValueError:
            self._note_error("metric_parse_failed")
            kv = None
        return kv, preempt_value, queues[0], queues[1]

    # -- loop ------------------------------------------------------------
    def _sample_once(self) -> None:
        t = monotonic() - self.t0
        mem, util = self._sample_gpu()
        kv, preemptions, waiting, running = self._sample_metrics()
        try:
            self.samples.append(
                ResourceSample(
                    t_s=max(0.0, t),
                    gpu_memory_used_mb=mem,
                    gpu_utilization_pct=util,
                    kv_cache_usage_perc=kv,
                    num_requests_waiting=waiting,
                    num_requests_running=running,
                    num_preemptions_total=preemptions,
                )
            )
        except Exception as exc:  # e.g. an out-of-range value; never fatal
            self._note_error(f"sample_build_failed: {exc!r}")

    def _run(self) -> None:
        # Sample once before checking the stop flag so a very short measured
        # window still yields at least one sample.
        while True:
            self._sample_once()
            if self._stop.wait(self.interval_s):
                break

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        try:
            self._init_nvml()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception as exc:  # sampler must never break the benchmark
            self._note_error(f"sampler_start_failed: {exc!r}")

    def stop(self) -> ResourceTelemetry:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        return self._derive()

    # -- derive ----------------------------------------------------------
    def _derive(self) -> ResourceTelemetry:
        mems = [s.gpu_memory_used_mb for s in self.samples if s.gpu_memory_used_mb is not None]
        utils = [s.gpu_utilization_pct for s in self.samples if s.gpu_utilization_pct is not None]
        kvs = [s.kv_cache_usage_perc for s in self.samples if s.kv_cache_usage_perc is not None]

        def _mean(xs):
            return sum(xs) / len(xs) if xs else None

        preemptions = None
        if self._preempt_first is not None and self._preempt_last is not None:
            preemptions = max(0, int(round(self._preempt_last - self._preempt_first)))

        return ResourceTelemetry(
            sample_interval_s=self.interval_s,
            num_samples=len(self.samples),
            peak_gpu_memory_mb=max(mems) if mems else None,
            gpu_utilization_mean_pct=_mean(utils),
            gpu_utilization_peak_pct=max(utils) if utils else None,
            kv_cache_usage_mean_perc=_mean(kvs),
            kv_cache_usage_peak_perc=max(kvs) if kvs else None,
            preemptions_total=preemptions,
            error="; ".join(self._errors) if self._errors else None,
        )
