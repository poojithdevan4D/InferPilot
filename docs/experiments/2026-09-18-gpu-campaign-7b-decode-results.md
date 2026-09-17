# Decode-heavy GPU campaign (Qwen2.5-7B @ A10 24 GB) — results

Per `2026-09-18-gpu-campaign-7b-decode-preregistration.md`, on Modal A10 24 GB. Raw evidence in
`runs/gpu-campaign-7b-decode/` (gitignored). Hypothesis: in a decode-heavy regime the vLLM default
over-admits → KV/scheduling pressure → a moderate `max_num_seqs` cap beats it.

## Two measurement findings that shaped the run

1. **The A10 is decode-compute-bound.** A rate-4 probe measured the default at only ~2.2 req/s
   achieved with *low* TTFT — the GPU saturates on decode FLOPS, not admission. Rate 4 was beyond
   capacity for all configs, so the campaign was retargeted to **rate 2** (near-capacity, feasible).
2. **Throughput/duration is not a valid feasibility signal for long-decode.** With ~23 s generations,
   the measurement window includes a long drain tail, so `num_successful/duration` under-reads even
   when the server keeps up (real rate-2 default: 1.51 req/s but TTFT flat, growth ratio 1.13). Fixed
   by `inferpilot.detect_saturation` — a drain-robust TTFT-stability signal (commit d47f24d), used for
   feasibility in this analysis.

## Result — no robust win; the held-out gate caught a false one

Rate-2 crossover (dev seeds 60/61/62), saturation-aware Pareto vs the default:

| width | dev verdict | held-out verdict (seeds 73/74/75) |
|---|---|---|
| 16 | saturated ×3 | — (overloaded) |
| 32 | saturated ×3 | — (overloaded) |
| 64 | **dominates ×3** | dominates ×1, inconclusive ×2 → **not confirmed** |
| 128 | **dominates ×3** | dominates ×1, inconclusive ×1, tied ×1 → **not confirmed** |

The dev set showed widths 64/128 dominating the default on all 3 seeds — a candidate win. **Held-out
refuted it.** The margins were tiny and did not generalize:

| (rate 2, mean) | ttft_p95 | tpot_p95 |
|---|---|---|
| dev default | 172.0 | 38.99 |
| dev width-64 | 166.4 | 38.07 |
| **held-out default** | **164.2** | **38.82** |
| **held-out width-64** | **165.4** | **38.83** |
| **held-out width-128** | **163.7** | **38.88** |

On unseen seeds the default and widths 64/128 are **statistically indistinguishable (<1%)**. The dev
"win" was an artifact of the dev default happening to read ~5% high on those particular seeds. Widths
16/32 genuinely saturate (too few slots for the ~46 concurrency at rate 2); 64/128 keep up but do not
beat the default.

## Conclusion

**No `max_num_seqs` setting robustly beats the vLLM default in the decode-heavy regime either.** This is
the second consistent negative (after the 128/32 run): on 7B/A10 the default is well-provisioned and the
bottleneck is decode compute, which a concurrency cap cannot improve. Combined meta-finding: across
short- and long-output workloads on this hardware, `max_num_seqs` is not a useful lever.

The genuinely valuable outcomes: (1) a drain-robust feasibility signal now exists; (2) **this is a clean
demonstration of the held-out gate doing its job** — the fail-closed discipline rejected a plausible
dev-set win rather than shipping it. A real win likely requires a KV-bound regime (much longer context,
or a model that nearly fills VRAM so KV is scarce) — a different setup, flagged for a future campaign.
