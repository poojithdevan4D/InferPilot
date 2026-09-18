# Demo: first real config win (fp8 KV) + a diagnostic-model refinement

Qwen2.5-3B on Modal A10G, 8k context (7680 prompt / 512 output), rate 5, 60 requests.
Single-variable: baseline `kv_cache_dtype=auto` vs `kv_cache_dtype=fp8`. Raw evidence in
`runs/demo-3b-kv/` (gitignored). ~₹few of Modal spend, staged.

## Result — fp8 KV is a real win here

| | throughput | ttft_p95 | tpot_p95 | kv_peak | preemptions |
|---|---|---|---|---|---|
| baseline (bf16 KV) | 0.56 req/s | 79.9 s | 146.4 ms | **1.00** | **2** |
| fp8 KV | **0.85 req/s (+52%)** | 41.1 s | 119.9 ms | 0.71 | 0 |

fp8 Pareto-dominates the baseline: higher throughput AND lower TTFT AND lower TPOT. This is
InferPilot's first measured beats-defaults config win — in exactly the regime the physics
pointed to (small, cheap-to-decode model where KV, not weights, is the pressure).

## Mechanism — and why our diagnosis was initially WRONG

Both runs showed GPU mean ~100%, so the diagnosis first called both `compute_bound` and
predicted fp8 useless. **That prediction was wrong.** The tell is the preemption counter:
the baseline had KV **100% full and 2 preemptions**; fp8 halved KV bytes → KV **71%, 0
preemptions**. The baseline's "100% GPU" included compute **wasted on preemption/recompute**
(evict a sequence's KV, recompute it later). fp8 freed KV blocks, eliminated the recompute
waste, and converted it into real throughput.

So GPU utilization alone does NOT distinguish compute-bound from KV-pressure-bound. The
discriminator is **KV-full + preemptions > 0**.

## The fix (diagnose → predict → verify → REFINE)

`BottleneckDiagnosis` now checks KV-pressure BEFORE compute-pinned: if KV is full and the
server is preempting, it classifies `kv_capacity_bound_decode` and recommends fp8 — even at
~100% GPU — because the recompute waste is reclaimable. This is exactly why the preemption
counter was added as first-class telemetry. Re-run on the real baseline now yields:
`kv_capacity_bound_decode → kv_cache_dtype=fp8`, matching the measured +52%. Regression tests
pin both the preempting case (→ fp8) and the no-preemption case (→ still compute_bound).

## Why this matters

1. **First validated config win**, mechanistically explained, not stumbled into.
2. **The predictive model self-corrected** from a measured contradiction — the scientific loop
   working as intended, with the fix grounded in a signal (preemptions) both external reviews
   independently demanded.
3. Caveat still enforced: fp8 KV can silently wreck long-context (>~100k) accuracy — the fp8
   recommendation carries `long_context_accuracy_verified` as a hard precondition.

## Honest scope

One model/GPU/workload; throughput/latency only (no accuracy eval on this run — the workload
is 8k context, below the ~100k fp8 danger zone, but a production apply MUST run the KL/accuracy
gate). The win is real and mechanistic; generality across models/hardware is future work.
