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

## Confirmation on 14B (preemption capture re-run, 2026-09-18)

Re-ran 14B/A100-40GB default vs fp8 with the preemption counter now captured:

| 14B | throughput | ttft_p95 | tpot_p95 | kv_peak | preemptions |
|---|---|---|---|---|---|
| default | 0.51 | 39.4 s | 36.0 ms | 1.00 | 4 |
| fp8 | **0.73 (+43%)** | 24.3 s | 39.6 ms | 1.00 | 4 |

The default now diagnoses `kv_capacity_bound_decode → kv_cache_dtype=fp8` (KV full + preempting),
which the earlier pre-preemption diagnosis wrongly called `compute_bound`. fp8 gives **+43%**
throughput and much better TTFT. Mechanism nuance vs 3B: here fp8 doubled effective KV capacity so
the batch grew (KV refilled to 100%), a **goodput** win (+43% throughput, −38% TTFT) trading a small
TPOT increase (36→40 ms) — still under a typical 50 ms SLO. Net: fp8 KV is now a confirmed,
repeatable capacity win in KV-pressured regimes across **two models** (3B +52%, 14B +43%), and the
preemption-aware diagnosis correctly identifies it in both.

## Generality: the fp8/preemption law across 3 models × 2 GPUs (2026-09-18)

Same single-variable test (baseline vs fp8 KV) in a KV-pressured regime for three model sizes:

| Model / GPU / context | default KV / preemptions | diagnosis | fp8 throughput gain |
|---|---|---|---|
| Qwen2.5-3B / A10 / 8k | 100% / 2 | kv_capacity_bound → fp8 | **+52%** |
| Qwen2.5-7B / A10 / 4k | 100% / 7 | kv_capacity_bound → fp8 | **+40%** |
| Qwen2.5-14B / A100-40GB / 2k | 100% / 4 | kv_capacity_bound → fp8 | **+43%** |

In all three, the default runs at GPU ~100% (which the pre-preemption diagnosis wrongly called
compute-bound) but with KV full and the server preempting — and fp8 KV yields a consistent
**+40–52%** throughput win with much lower TTFT. The **preemption-aware diagnosis correctly identifies
the winnable regime in every case.** This is a repeatable, mechanistically-grounded law across model
sizes (3B→14B) and GPUs (A10, A100), not a single-point result. Total Modal spend for the full
3-model study: a few dollars.

## The negative half: fp8 correctly does NOTHING when there's no preemption (2026-09-18)

To prove the model is right about where fp8 CANNOT help (not just where it can), a compute/decode-
bound test: 7B / A10 / 512-prompt 128-output / rate 8.

| 7B compute-bound | throughput | ttft_p95 | tpot_p95 | kv_peak | preemptions |
|---|---|---|---|---|---|
| default | 5.10 | 1129 ms | 100.8 ms | 0.91 | 0 |
| fp8 | 5.18 (**+1.7%**) | 1108 ms | 98.0 ms | 0.52 | 0 |

fp8 halved KV usage (0.91→0.52) but throughput barely moved (+1.7%, noise) — because there were **no
preemptions to reclaim**; the bottleneck is decode compute, not KV capacity. InferPilot's diagnosis
correctly recommended **no lever** here, and fp8 correctly delivered nothing.

**Key validation:** KV being *near-full* (0.91) is NOT the winnable signal — **preemptions > 0** is.
The model is now shown correct in BOTH directions: it recommends fp8 exactly when preemption-driven
recompute waste exists (+40–52%), and abstains when the wall is compute (+1.7%). Two-sided correctness
is what makes the diagnosis trustworthy rather than a lucky pattern-match.

## Reproduce + a known limitation (honest)

`scripts/fp8_law_demo.py` reproduces the full result from stored evidence (GPU-free): all four
cases show the diagnosis's fp8 recommendation matching the measured outcome (4/4 correct).

Known limitation (labeling, not action): the 7B decode-bound case (KV 0.91, GPU 100%, 0 preemptions,
achieved 5.1 vs 8 offered qps) is labeled `underutilized` because the saturation signal is
TTFT-stability-based and this server admits fast but is decode-throughput-limited (TTFT stays low
while throughput caps). The recommended ACTION is correct (no config lever), but the regime LABEL is
imprecise — it is compute/decode-bound, not idle. A future refinement should add a throughput-
saturation signal (achieved << offered with stable TTFT) to relabel this as compute_bound. Flagged
rather than silently rushed.

## Critical caveat: the fp8 win is a GOODPUT-CEILING lever, not a free speedup (2026-09-18)

Sharpest critique of the wins above: they all occurred at TTFT of 24–94 s — overloaded servers.
Is fp8 a real win, or just "helps a drowning server"? Tested 3B/A10/8k at a lower rate (2 qps):

| 3B/8k @ rate 2 | throughput | ttft_p95 | kv_peak | preemptions | saturated |
|---|---|---|---|---|---|
| baseline | 0.55 | 46.5 s | 1.00 | 2 | yes |
| fp8 | 0.84 (**+52%**) | 20.6 s | 0.57 | 0 | **yes (still)** |

fp8 halves TTFT and adds +52% throughput and clears preemption — but it is **still saturated**: even
fp8 cannot make rate 2 healthy, because 8k-context 3B saturates below ~1 qps. **There is no healthy
operating point (for this large-KV workload) where fp8 is a free speedup.** The honest framing:

> **fp8 KV is a goodput-ceiling / overload-resilience lever: it lets the server sustain ~+50% more
> load before TTFT explodes, and roughly halves TTFT under pressure — it is NOT a low-load speedup.**

This matters for how InferPilot reports value: as **goodput-under-SLO and capacity-per-dollar**, never
as "faster at current load." (The advisory already frames it this way — `CapacityAdvisory` reports
goodput + $/token, not raw latency deltas.) A truly *healthy* KV-bound sweet spot (KV full while TTFT
stays low) would need a many-small-requests workload and is untested; for large-per-request-KV
workloads the KV-full regime coincides with saturation. Stated plainly rather than over-claimed.

## Quality: fp8's +52% is phrasing-divergent but correctness-preserving (2026-09-18)

Closed the biggest blind spot — does the fp8 throughput win cost output quality? Two measurements on
Qwen2.5-3B (bf16 KV vs fp8 KV, greedy, identical prompts):

1. **Open-ended generation, position-wise token agreement: 12%** (248/2048). Looks alarming, but is
   misleading: greedy generation cascades — one early perturbed logit → different (but often equally
   valid) continuation. Exact-token-match conflates "different" with "worse"; it is a **drift
   detector**, not a quality verdict.
2. **Canonical factual QA (16 questions with known answers): bf16 12/16, fp8 12/16, ZERO regressions.**
   Despite the phrasing divergence, fp8 preserved actual correctness.

**Conclusion:** on short-context factual tasks, fp8 KV's +40–52% throughput win is **quality-preserving**
(0 regressions), and the 12% token divergence is rewording, not degradation. This makes the fp8
recommendation *quality-verified*, not just quality-gated.

**Caveats (kept honest):** (a) exact-token-agreement over-flags open-ended generation — the definitive
gate should use canonical-task accuracy and/or teacher-forced KL of next-token distributions (avoids
the cascade problem); (b) this tested short-context QA — the documented fp8 failure mode is
long-context (>~100k) retrieval-accuracy collapse, which was NOT tested here, so the
`long_context_accuracy_verified` precondition on the fp8 lever remains mandatory.
