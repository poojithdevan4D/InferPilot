"""Inspect the measured fp8/preemption pattern in legacy stored evidence (GPU-free).

These bundles predate aligned ``LoadEvidence``. The current fail-closed diagnosis therefore
returns ``unknown``; this script reports that honestly alongside the historical measured
association. It does not retroactively treat scalar telemetry as causal evidence.

    uv run python scripts/fp8_law_demo.py
"""

from __future__ import annotations

import glob
from pathlib import Path

from inferpilot import ExperimentResult, diagnose

ROOT = Path(__file__).resolve().parents[1]

# (label, baseline glob, fp8 glob)
CASES = [
    ("3B / A10 / 8k ctx",   "runs/demo-3b-kv/demo-3b-baseline-*",      "runs/demo-3b-kv/demo-3b-fp8-*"),
    ("7B / A10 / 4k ctx",   "runs/demo-7b-kv/demo-7b-baseline-*",      "runs/demo-7b-kv/demo-7b-fp8-*"),
    ("14B / A100 / 2k ctx", "runs/g7k-confirm/g7k-default-qps2-a60-*", "runs/g7k-confirm/g7k-fp8-qps2-a60-*"),
    ("7B / A10 / short (decode-bound)", "runs/demo-7b-compute/demo-7bc-baseline-*", "runs/demo-7b-compute/demo-7bc-fp8-*"),
]


def _load(pattern: str) -> ExperimentResult | None:
    # committed evidence bundle first (reproducible on a fresh clone), then local runs/
    for base in ("docs/evidence/", "runs/"):
        hits = sorted(glob.glob(str(ROOT / pattern.replace("runs/", base, 1))))
        if hits:
            return ExperimentResult.model_validate_json((Path(hits[-1]) / "result.json").read_text())
    return None


def main() -> int:
    print("InferPilot fp8/preemption finding — legacy measured evidence\n")
    print(f"{'case':34} {'current diagnosis':18} {'KV/preempt':11} {'fp8 Δthru':9} evidence status")
    print("-" * 100)
    for label, base_g, fp8_g in CASES:
        base, fp8 = _load(base_g), _load(fp8_g)
        if base is None or fp8 is None or fp8.status.value != "completed":
            print(f"{label:34} (evidence missing)")
            continue
        d = diagnose(base)
        gain = (fp8.aggregates.throughput_requests_per_s / base.aggregates.throughput_requests_per_s - 1) * 100
        kvpre = f"{base.telemetry.kv_cache_usage_peak_perc:.2f}/{base.telemetry.preemptions_total}"
        status = "legacy bundle: no aligned load evidence"
        print(f"{label:34} {d.regime:18} {kvpre:11} {gain:+7.0f}%  {status}")
    print("\nObserved pattern: preempting cases gained +40–53%; the zero-preemption case gained +2%.")
    print("This is a post-hoc empirical heuristic, not a validated diagnosis on these legacy bundles.")
    print("New runs embed aligned request/queue/token evidence before the advisor names a load state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
