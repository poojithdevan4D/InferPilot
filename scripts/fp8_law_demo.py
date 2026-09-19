"""Reproduce InferPilot's fp8-preemption law on the stored real evidence (GPU-free).

For each (model, GPU, workload) it loads the measured baseline (bf16 KV) and the fp8 KV
run, prints InferPilot's DIAGNOSIS of the baseline, and the MEASURED fp8 outcome — showing
that the diagnosis correctly predicts BOTH where fp8 wins and where it does nothing. The
discriminator is preemptions, not KV-fullness.

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
    print("InferPilot fp8-preemption law — diagnosis vs measured outcome (real evidence)\n")
    print(f"{'case':34} {'diagnosis(baseline)':26} {'lever':14} {'KV/preempt':11} {'fp8 Δthru':9} verdict")
    print("-" * 108)
    for label, base_g, fp8_g in CASES:
        base, fp8 = _load(base_g), _load(fp8_g)
        if base is None or fp8 is None or fp8.status.value != "completed":
            print(f"{label:34} (evidence missing)")
            continue
        d = diagnose(base)
        gain = (fp8.aggregates.throughput_requests_per_s / base.aggregates.throughput_requests_per_s - 1) * 100
        recommends_fp8 = d.recommended_lever == "kv_cache_dtype=fp8"
        # a "win" is a material throughput gain; predicted iff diagnosis recommends fp8
        material = gain >= 15
        correct = recommends_fp8 == material
        kvpre = f"{base.telemetry.kv_cache_usage_peak_perc:.2f}/{base.telemetry.preemptions_total}"
        print(f"{label:34} {d.regime:26} {d.recommended_lever:14} {kvpre:11} {gain:+7.0f}%  "
              f"{'✓ correct' if correct else '✗ MISMATCH'}")
    print("\nLaw: preemptions>0 (not KV-fullness) => KV-bound => fp8 wins (+40-52%).")
    print("     preemptions=0 => compute/decode-bound => fp8 does nothing (~0%); advisor abstains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
