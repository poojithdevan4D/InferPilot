"""Decode-campaign crossover analysis with drain-robust feasibility.

Uses detect_saturation (TTFT-stability) for 'keeps up', NOT throughput/duration, which
is drain-contaminated for long generations. A width beats the default at a rate iff, on
a majority of shared seeds, both keep up (not saturated) and the width Pareto-dominates
the default on ttft_p95 + tpot_p95 (no worse on either within tolerance, strictly better
on >=1). Reads the local manifest/store; no GPU.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from inferpilot import ExperimentResult, detect_saturation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/gpu-campaign-14b-kv"
TOL = 0.02


def _load():
    out = {}
    for l in (OUT / "manifest.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["accepted"]:
            out[r["experiment_id"]] = ExperimentResult.model_validate_json(
                (OUT / "store/objects" / r["store_run_id"] / "result.json").read_text()
            )
    return out


def _pareto(default: ExperimentResult, cand: ExperimentResult) -> str:
    dsat = detect_saturation(default.measurements).saturated
    csat = detect_saturation(cand.measurements).saturated
    if csat:
        return "candidate_saturated"
    if dsat:
        return "candidate_dominates"  # keeps up where default doesn't
    da, ca = default.aggregates, cand.aggregates
    nw, sb = True, False
    for m in ("ttft_p95_ms", "tpot_p95_ms"):
        dv, cv = getattr(da, m), getattr(ca, m)
        if cv <= dv * (1 - TOL):
            sb = True
        elif cv > dv * (1 + TOL):
            nw = False
    if nw and sb:
        return "candidate_dominates"
    return "incumbent_kept" if nw else "inconclusive_tradeoff"


def analyze():
    cells = _load()
    rates = sorted({int(r.config.workload.request_rate_qps) for e, r in cells.items() if "-dev-" in e})
    seeds = sorted({r.config.workload.arrival_seed for e, r in cells.items() if "-dev-" in e})
    widths = sorted({r.config.engine.max_num_seqs for e, r in cells.items() if "-dev-" in e})
    winners = []
    for rate in rates:
        print(f"\n=== rate {rate} qps — width vs vLLM default (saturation-aware, per seed) ===")
        for width in widths:
            verdicts = defaultdict(int)
            for seed in seeds:
                d = cells.get(f"g7k-default-qps{rate}-a{seed}")
                c = cells.get(f"g7k-dev-qps{rate}-a{seed}-seq{width}")
                if d and c:
                    verdicts[_pareto(d, c)] += 1
            summ = ", ".join(f"{k}×{n}" for k, n in sorted(verdicts.items()))
            maj = verdicts["candidate_dominates"] > sum(verdicts.values()) / 2 if verdicts else False
            print(f"  width {width}: {summ or 'no data'}{'   <- BEATS DEFAULT' if maj else ''}")
            if maj:
                winners.append(width)
    print("\nwidths beating default:", ",".join(map(str, sorted(set(winners)))) or "NONE — advisor keeps the default")
    return winners


def confirm_heldout(widths):
    """Confirm winning widths against the default on the held-out seed block."""
    cells = _load()
    held_seeds = sorted({r.config.workload.arrival_seed for e, r in cells.items()
                         if e.startswith("g7k-held-")})
    rates = sorted({int(r.config.workload.request_rate_qps) for e, r in cells.items()
                    if e.startswith("g7k-held-")})
    if not held_seeds:
        print("\n(no held-out cells yet)")
        return {}
    confirmed = {}
    for rate in rates:
        print(f"\n=== HELD-OUT rate {rate} qps (seeds {held_seeds}) ===")
        for width in widths:
            verdicts = defaultdict(int)
            for seed in held_seeds:
                d = cells.get(f"g7k-default-qps{rate}-a{seed}")
                c = cells.get(f"g7k-held-qps{rate}-a{seed}-seq{width}")
                if d and c:
                    verdicts[_pareto(d, c)] += 1
            summ = ", ".join(f"{k}×{n}" for k, n in sorted(verdicts.items()))
            maj = verdicts["candidate_dominates"] > sum(verdicts.values()) / 2 if verdicts else False
            print(f"  width {width}: {summ or 'no data'}{'   <- CONFIRMED' if maj else '   (not confirmed)'}")
            confirmed[width] = maj
    return confirmed


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "heldout":
        confirm_heldout([int(w) for w in sys.argv[2].split(",")] if len(sys.argv) > 2 else [64, 128])
    else:
        analyze()
