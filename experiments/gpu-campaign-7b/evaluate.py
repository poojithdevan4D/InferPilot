"""Crossover analysis using the fail-closed Pareto comparison (inferpilot.advisor).

A candidate width "beats the default" at a rate ONLY if compare_configs judges it to
dominate — feasible (keeps up) AND no worse on any guarded latency metric (ttft_p95,
tpot_p95) AND strictly better on at least one. This is the fix for the 2026-09-18
tpot-only objective that declared TTFT-catastrophic "winners". A single seed can't
carry a verdict: a width wins a rate only if it dominates the default on a MAJORITY
of shared seeds. Reads the local manifest/store; no GPU, no network.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from inferpilot import ExperimentResult
from inferpilot.advisor import ComparisonSpec, compare_configs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/gpu-campaign-7b"
SPEC = ComparisonSpec()


def _load() -> dict[str, ExperimentResult]:
    recs = [json.loads(l) for l in (OUT / "manifest.jsonl").read_text().splitlines()]
    out = {}
    for r in recs:
        if r["accepted"]:
            out[r["experiment_id"]] = ExperimentResult.model_validate_json(
                (OUT / "store/objects" / r["store_run_id"] / "result.json").read_text()
            )
    return out


def analyze():
    cells = _load()

    def get(eid):
        return cells.get(eid)

    def rates_seeds():
        rates, seeds = set(), set()
        for eid, res in cells.items():
            if eid.startswith("g7-dev-"):
                rates.add(int(res.config.workload.request_rate_qps))
                seeds.add(res.config.workload.arrival_seed)
        return sorted(rates), sorted(seeds)

    rates, seeds = rates_seeds()
    widths = (1, 2, 4, 8)
    winners = {}
    for rate in rates:
        print(f"\n=== rate {rate} qps — candidate width vs vLLM default (per seed) ===")
        for width in widths:
            verdicts = defaultdict(int)
            for seed in seeds:
                default = get(f"g7-default-qps{rate}-a{seed}")
                cand = get(f"g7-dev-qps{rate}-a{seed}-seq{width}")
                if default is None or cand is None:
                    continue
                v = compare_configs(SPEC, default, cand).verdict
                verdicts[v] += 1
            summary = ", ".join(f"{k}×{n}" for k, n in sorted(verdicts.items()))
            dominates = verdicts["candidate_dominates"]
            majority = dominates > sum(verdicts.values()) / 2 if verdicts else False
            print(f"  width {width}: {summary or 'no data'}"
                  f"{'   <- BEATS DEFAULT (majority)' if majority else ''}")
            if majority:
                winners.setdefault(rate, []).append(width)
    won = sorted({w for ws in winners.values() for w in ws})
    if won:
        print("\nwidths that beat the default (Phase 3 --widths):", ",".join(map(str, won)))
    else:
        print("\nNo width beats the vLLM default at any rate — advisor keeps the default.")
    return winners


if __name__ == "__main__":
    analyze()
