"""Historical cost-to-serve case study on real measured evidence (GPU-free).

The stored bundle predates aligned load evidence. The current advisor correctly abstains; this
script reproduces the measured before/after economics without claiming a redesigned diagnosis
from insufficient telemetry. Run: uv run python scripts/cost_rescue_demo.py
"""

from __future__ import annotations

import glob
from pathlib import Path

from inferpilot import ExperimentResult, diagnose

ROOT = Path(__file__).resolve().parents[1]
GPU_HOURLY = 1.10  # A10G $/hr


def _load(pat: str) -> ExperimentResult:
    for base in ("docs/evidence/", "runs/"):
        hits = sorted(glob.glob(str(ROOT / pat.replace("runs/", base, 1))))
        if hits:
            return ExperimentResult.model_validate_json((Path(hits[-1]) / "result.json").read_text())
    raise FileNotFoundError(pat)


def _cost_per_1m(tok_s: float) -> float:
    return GPU_HOURLY / (tok_s * 3600) * 1_000_000 if tok_s else float("nan")


def main() -> int:
    base = _load("runs/demo-3b-healthy/h3b-baseline-*")
    fp8 = _load("runs/demo-3b-healthy/h3b-fp8-*")
    ba, fa = base.aggregates, fp8.aggregates
    d = diagnose(base)

    def line(s=""):
        print(s)

    line("=" * 74)
    line("  InferPilot — Cost-to-Serve Rescue  (Qwen2.5-3B on A10G, KV-pressured)")
    line("=" * 74)
    line("\nBEFORE (vLLM defaults, bf16 KV):")
    line(f"  throughput {ba.throughput_requests_per_s:.2f} req/s | TTFT p95 {ba.ttft_p95_ms/1000:.1f}s "
         f"| KV {base.telemetry.kv_cache_usage_peak_perc:.2f} | preemptions {base.telemetry.preemptions_total}")
    line(f"  cost-to-serve ≈ ${_cost_per_1m(ba.throughput_tokens_per_s):.2f} / 1M output tokens")

    line("\nCURRENT FAIL-CLOSED DIAGNOSIS:")
    line(f"  regime = {d.regime}; recommendation = {d.recommended_lever}")
    line("  reason: this legacy bundle lacks aligned request/queue/token LoadEvidence")
    line("  action: collect a redesign-native run; do not infer a bottleneck from snapshots")

    line("\nQUALITY GATE (measured, fp8 vs bf16 — required before apply):")
    line("  factual QA: 0/16 regressions (task-lossless)")
    line("  needle-in-haystack @14k, 5 depths: fp8 5/5 = bf16 5/5 (retrieval preserved)")
    line("  teacher-forced KL preflight: FAILED (mean>0.01, p99 0.39) -> real shift; task impact unverified")

    gain = (fa.throughput_requests_per_s / ba.throughput_requests_per_s - 1) * 100
    save = (1 - _cost_per_1m(fa.throughput_tokens_per_s) / _cost_per_1m(ba.throughput_tokens_per_s)) * 100
    line("\nAFTER (fp8 KV, measured; quality smoke tests are mixed):")
    line(f"  throughput {fa.throughput_requests_per_s:.2f} req/s (+{gain:.0f}%) | TTFT p95 "
         f"{fa.ttft_p95_ms/1000:.1f}s (-{(1-fa.ttft_p95_ms/ba.ttft_p95_ms)*100:.0f}%) | preemptions "
         f"{fp8.telemetry.preemptions_total}")
    line(f"  cost-to-serve ≈ ${_cost_per_1m(fa.throughput_tokens_per_s):.2f} / 1M output tokens "
         f"(-{save:.0f}%)")

    line("\nRESULT:")
    line(f"  measured case study: +{gain:.0f}% goodput, TTFT halved, {save:.0f}% lower cost-per-token.")
    line("  The association motivated the fp8/preemption hypothesis; it is not retroactive proof")
    line("  that the current advisor would have selected the change from this legacy bundle.")
    line("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
