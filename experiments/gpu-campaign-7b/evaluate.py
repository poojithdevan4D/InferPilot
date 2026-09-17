"""Crossover analysis for the GPU campaign: pick per-rate best width, vs vLLM default.

Objective (preregistration-consistent, adapted to the 7B envelope where the 0.5B SLO
is irrelevant): a config is VIABLE at a rate if it keeps up with the offered load
(achieved throughput >= KEEPUP * offered_rate, i.e. not overloaded). Among viable
configs at a rate, the WINNER minimizes tpot_p95_ms (best per-token latency), with
ttft_p95_ms as the tie-breaker. We then report the winner vs the vLLM default at the
same rate — the honest "beats defaults" delta.

Reads accepted dev + default cells from the local manifest/store. Prints a table and
the winning widths for Phase 3 (held-out confirmation). No GPU, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from inferpilot import ExperimentResult

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/gpu-campaign-7b"
KEEPUP = 0.95  # achieved throughput must be >= 95% of offered rate to count as "keeping up"


def _load():
    recs = [json.loads(l) for l in (OUT / "manifest.jsonl").read_text().splitlines()]
    cells = {}
    for r in recs:
        if not r["accepted"]:
            continue
        res = ExperimentResult.model_validate_json(
            (OUT / "store/objects" / r["store_run_id"] / "result.json").read_text()
        )
        a = res.aggregates
        cells[r["experiment_id"]] = {
            "rate": float(r["request_rate_qps"]), "seed": r["arrival_seed"],
            "width": r["max_num_seqs"], "id": r["experiment_id"],
            "ttft_p95": a.ttft_p95_ms, "tpot_p95": a.tpot_p95_ms,
            "thru": a.throughput_requests_per_s,
        }
    return cells


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def analyze():
    cells = _load()
    rates = sorted({c["rate"] for c in cells.values()})
    winners = {}
    print(f"{'rate':>5} {'width':>5} {'viable':>6} {'tpot_p95':>9} {'ttft_p95':>9} {'thru':>6}  (mean over seeds)")
    for rate in rates:
        # aggregate dev cells by width (mean over seeds)
        by_width = {}
        for c in cells.values():
            if c["rate"] != rate or c["id"].startswith("g7-default") or "-dev-" not in c["id"]:
                continue
            by_width.setdefault(c["width"], []).append(c)
        rows = []
        for width, cs in sorted(by_width.items()):
            tpot = _mean([c["tpot_p95"] for c in cs])
            ttft = _mean([c["ttft_p95"] for c in cs])
            thru = _mean([c["thru"] for c in cs])
            viable = thru >= KEEPUP * rate
            rows.append({"width": width, "tpot": tpot, "ttft": ttft, "thru": thru, "viable": viable})
        for r in rows:
            print(f"{rate:5.0f} {r['width']:5d} {str(r['viable']):>6} {r['tpot']:9.2f} {r['ttft']:9.1f} {r['thru']:6.2f}")
        viable_rows = [r for r in rows if r["viable"]]
        if viable_rows:
            win = min(viable_rows, key=lambda r: (r["tpot"], r["ttft"]))
            winners[rate] = win["width"]
            # default at this rate
            defs = [c for c in cells.values() if c["rate"] == rate and c["id"].startswith("g7-default")]
            dtpot, dttft = _mean([c["tpot_p95"] for c in defs]), _mean([c["ttft_p95"] for c in defs])
            print(f"   -> WINNER width={win['width']}  tpot {win['tpot']:.2f} vs default {dtpot:.2f}ms "
                  f"({(dtpot-win['tpot'])/dtpot*100:+.1f}%)  |  ttft {win['ttft']:.1f} vs {dttft:.1f}ms")
        else:
            print(f"   -> no viable width at rate {rate:.0f} (all overloaded)")
    print("\nwinning widths (for Phase 3 --widths):", ",".join(str(w) for w in sorted(set(winners.values()))))
    return winners


if __name__ == "__main__":
    analyze()
